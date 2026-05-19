import time
import os
import gc
import torch
import itertools
from data.dataloader_phased import PhasedReplayDataloaders

def auto_tune_pipeline(
    model, 
    sm_files, 
    window_spec, 
    device="cuda", 
    max_workers=8, 
    test_batches=30
):
    print("Initiating Auto-Tuner Sequence...\n")
    
    # ==========================================
    # PHASE 1: Find Maximum Safe Batch Size
    # ==========================================
    print("--- Phase 1: VRAM Batch Size Sweep ---")
    test_batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    optimal_batch_size = test_batch_sizes[0]
    
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for bs in test_batch_sizes:
        try:
            print(f"Testing Batch Size: {bs}...", end=" ")
            # Create dummy tensors mimicking your dataloader output
            # Adjust the shapes based on your window_spec!
            dummy_x = torch.randn(bs, 3, window_spec.input_len, device=device)
            dummy_agg = torch.randn(bs, 1, window_spec.input_len, device=device) 
            
            # Simulate a forward and backward pass to allocate gradient memory
            optimizer.zero_grad()
            output = model(dummy_x) # Adjust this if your model takes multiple inputs
            loss = output.sum()
            loss.backward()
            optimizer.step()
            
            print("Pass")
            optimal_batch_size = bs
            
            # Clean up VRAM immediately
            del dummy_x, dummy_agg, output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[WARNING] OOM Error")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    # Back off by 15% to leave room for val loaders and gradient spikes
    safe_batch_size = int(optimal_batch_size * 0.85)
    # Snap to the nearest multiple of 16 for Tensor Core efficiency
    safe_batch_size = safe_batch_size - (safe_batch_size % 16) 
    print(f"\n Optimal Safe Batch Size Locked: {safe_batch_size}\n")

    # ==========================================
    # PHASE 2: Throughput Grid Search (CPU/IO)
    # ==========================================
    print("--- Phase 2: Dataloader Throughput Sweep ---")
    cpu_cores = os.cpu_count() or 4
    worker_grid = [w for w in [0, 2, 4, 6, 8] if w <= min(max_workers, cpu_cores)]
    prefetch_grid = [2, 3, 4]
    
    best_throughput = 0.0
    best_config = {"workers": 0, "prefetch": 2}
    results = []

    for workers, prefetch in itertools.product(worker_grid, prefetch_grid):
        # Prefetch factor cannot be set if workers == 0
        if workers == 0 and prefetch != 2:
            continue
            
        print(f"Testing [Workers: {workers} | Prefetch: {prefetch}]...", end=" ", flush=True)
        
        # Instantiate your optimized pipeline
        data = PhasedReplayDataloaders(
            sm_files=sm_files,
            window_spec=window_spec,
            batch_size=safe_batch_size,
            num_workers=workers,
            prefetch_factor=prefetch if workers > 0 else None,
            replay_buffer_capacity=10000,
            min_buffer_size=1000, # Keep low for fast testing
        )
        
        loader = data.get_train_loader()
        iterator = iter(loader)
        
        # Warmup (wait for background threads to hit min_buffer_size)
        next(iterator) 
        
        # Benchmark
        start_time = time.perf_counter()
        batches_pulled = 0
        
        try:
            for _ in range(test_batches):
                # Pull the batch and simulate non-blocking GPU transfer
                x, m_id, ts, agg, tags = next(iterator)
                x = x.to(device, non_blocking=True)
                batches_pulled += 1
        except StopIteration:
            pass
            
        elapsed = time.perf_counter() - start_time
        # Metric: Total Individual Samples per Second
        throughput = (batches_pulled * safe_batch_size) / elapsed 
        results.append((workers, prefetch, throughput))
        
        print(f"{throughput:,.0f} samples/sec")
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = {"workers": workers, "prefetch": prefetch}

        # Nuclear garbage collection to kill threading leaks between tests
        del iterator, loader, data, x
        gc.collect()

    print("\n--- Auto-Tune Complete ---")
    print(f"   Best Configuration:")
    print(f"   Batch Size:      {safe_batch_size}")
    print(f"   Num Workers:     {best_config['workers']}")
    print(f"   Prefetch Factor: {best_config['prefetch']}")
    print(f"   Peak Throughput: {best_throughput:,.0f} samples/sec")
    
    return safe_batch_size, best_config['workers'], best_config['prefetch']

if __name__ == "__main__":
    # Example usage (replace with your actual model and data paths)
    
    @dataclass(frozen=True)
    class WindowSpec:
        input_len: int = 24 # length of input window - how many past hours go into the model as x
        stride: int = 24 # how many hours to move the window each step
    
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1)
            self.fc = torch.nn.Linear(16 * 60, 10)  # Adjust based on window_spec.input_len
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = DummyModel()
    sm_files = ["data/smart_meter_1.parquet", "data/smart_meter_2.parquet"]  # Replace with actual paths
    window_spec = WindowSpec(input_len=60, target_len=10)  # Adjust as needed
    
    auto_tune_pipeline(model, sm_files, window_spec)