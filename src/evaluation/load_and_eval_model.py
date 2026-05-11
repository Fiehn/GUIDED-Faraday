"""
Load a trained model from a folder and rerun the full evaluation framework 
using the same data and config as was used during training.

Usage:
    python src/evaluation/load_and_eval_model.py <model_folder_path>

Example:
    python src/evaluation/load_and_eval_model.py models/hpc_runs/my_model_20260510_123456
"""

from pathlib import Path
import sys
import os
from types import SimpleNamespace
import torch
import copy
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model.factory import _load_train_config, create_model_from_config
from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec
from src.data.normalizer import Normalizer
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


NON_PHASED_CONFIG_MAP = (
    ("AutoRegressiveTransformerVAE_", "art.yaml"),
    ("AverageBaselineModel_", "baseline_mean.yaml"),
    ("CVAE_", "cvae.yaml"),
    ("DiffusionModel_", "diffusion.yaml"),
    ("guide_cnn_", "guide_cnn.yaml"),
    ("hierarchy_", "guide_hierarchy.yaml"),
    ("vamp_", "guide_vamp.yaml"),
    ("guide_", "guide.yaml"),
)

PHASED_CONFIG_MAP = (
    ("AutoRegressiveTransformerVAE_", "art.yaml"),
    ("AverageBaselineModel_", "baseline_average.yaml"),
    ("QuantileBaselineModel_", "baseline_quantile.yaml"),
    ("DiffusionModel_", "diffusion.yaml"),
    ("HierarchicalSplitter_", "phase_splitter.yaml"),
    ("SetGuideVAE_", "set_guide.yaml"),
)


def _resolve_config_from_model_name(model_path: str) -> str | None:
    """
    Map a model folder name to its training config file.

    The mapping is based on the naming conventions used in the hpc_runs
    directories. For unknown names, return None so we can fall back to local
    config discovery.
    """
    model_name = os.path.basename(os.path.normpath(model_path))
    path_parts = Path(model_path).parts
    is_phased_dir = "3-phase" in path_parts

    mapping = PHASED_CONFIG_MAP if is_phased_dir else NON_PHASED_CONFIG_MAP
    config_root = ROOT_DIR / "config" / "tests" / ("phased" if is_phased_dir else "")

    for prefix, config_name in mapping:
        if model_name.startswith(prefix):
            candidate = config_root / config_name
            if candidate.exists():
                return str(candidate)

    return None


class SimpleWindowDataset(Dataset):
    """Simple PyTorch dataset for windowed time series data."""
    
    def __init__(self, windows, meter_ids, timestamps, aggregates=None, normalizer=None):
        """
        Args:
            windows: (n_windows, window_size, n_features) array
            meter_ids: (n_windows,) meter ID for each window
            timestamps: (n_windows,) start timestamp for each window
            aggregates: (n_windows, n_features) optional aggregate values
            normalizer: Optional normalizer to denormalize data
        """
        self.windows = torch.from_numpy(windows).float()
        self.meter_ids = torch.from_numpy(meter_ids).long()
        self.timestamps = torch.from_numpy(timestamps).long()
        self.aggregates = torch.from_numpy(aggregates).float() if aggregates is not None else None
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = self.windows[idx]
        m = self.meter_ids[idx]
        t = self.timestamps[idx]
        a = self.aggregates[idx] if self.aggregates is not None else torch.zeros(x.shape[-1])
        return x, m, t, a


def load_parquet_data_for_eval(parquet_files, window_size=24, stride=1, normalizer=None, 
                               n_limit=None, meter_col="meter_id", value_cols=None):
    """
    Load parquet files and create windowed dataset for evaluation.
    
    Args:
        parquet_files: List of parquet file paths
        window_size: Size of sliding window
        stride: Stride for sliding window
        normalizer: Optional Normalizer to apply
        n_limit: Limit number of windows (for smaller test sets)
        meter_col: Name of meter ID column
        value_cols: List of value columns (if None, uses all numeric except meter_col)
    
    Returns:
        SimpleWindowDataset with all windows
    """
    import pandas as pd
    
    all_windows = []
    all_meter_ids = []
    all_timestamps = []
    all_aggregates = []
    
    for pfile in parquet_files:
        print(f"  Loading: {pfile}")
        df = pd.read_parquet(pfile)
        
        if value_cols is None:
            value_cols = [col for col in df.columns if col != meter_col and df[col].dtype in [np.float32, np.float64]]
        
        # Group by meter
        for meter_id, group in df.groupby(meter_col, sort=False):
            group = group.reset_index(drop=True)
            values = group[value_cols].values.astype(np.float32)  # (n_samples, n_features)
            
            # Apply normalizer if provided
            if normalizer is not None:
                normalized_values = normalizer.transform(values)
                # Convert back to numpy if it's a torch tensor
                if torch.is_tensor(normalized_values):
                    values = normalized_values.numpy()
                else:
                    values = normalized_values
            
            # Create sliding windows
            for start_idx in range(0, len(values) - window_size + 1, stride):
                end_idx = start_idx + window_size
                window = values[start_idx:end_idx]  # (window_size, n_features)
                
                all_windows.append(window)
                all_meter_ids.append(meter_id)
                
                # Use start timestamp
                if "timestamp" in group.columns:
                    ts = group.iloc[start_idx]["timestamp"]
                else:
                    ts = start_idx
                all_timestamps.append(ts)
                
                # Aggregate: hourly sum (if window_size=24, this gives daily total)
                hourly_totals = np.sum(window, axis=1)  # (window_size,)
                all_aggregates.append(hourly_totals)
                
                if n_limit and len(all_windows) >= n_limit:
                    break
            
            if n_limit and len(all_windows) >= n_limit:
                break
        
        if n_limit and len(all_windows) >= n_limit:
            break
    
    if not all_windows:
        raise ValueError(f"No windows created from parquet files: {parquet_files}")
    
    windows_array = np.array(all_windows, dtype=np.float32)  # (n_windows, window_size, n_features)
    meter_ids_array = np.array(all_meter_ids, dtype=np.int64)
    timestamps_array = np.array(all_timestamps, dtype=np.int64)
    aggregates_array = np.array(all_aggregates, dtype=np.float32)
    
    print(f"✓ Loaded {len(windows_array)} windows from {len(parquet_files)} files")
    
    return SimpleWindowDataset(windows_array, meter_ids_array, timestamps_array, aggregates_array, normalizer)


def load_model(model_path: str, device: str = "cpu"):
    """
    Load a trained model from a folder.
    
    The model_path should point to a folder containing:
    - A config file (*.yaml)
    - Best model checkpoint (best_model_*.pt)
    - Optionally: normalizer.pt, user_profiler.pt
    
    Args:
        model_path: Path to model folder
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        tuple: (model, normalizer, conditioner, config)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(model_path)

    # 1. Find and load config file
    # Prefer the exact config snapshot saved alongside the model run.
    preferred = os.path.join(model_path, "training_config_used.yaml")
    if os.path.exists(preferred):
        config_path = preferred
    else:
        config_path = _resolve_config_from_model_name(model_path)
        if config_path is None:
            for file in os.listdir(model_path):
                if file.endswith(".yaml"):
                    config_path = os.path.join(model_path, file)
                    break
    
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {model_path}")
    
    print(f"Loading config from: {config_path}")
    config = _load_train_config(config_path)

    # 2. Create model from config
    model = create_model_from_config(config, device)
    print(f"Created model: {model.__class__.__name__}")

    # 3. Find and load best checkpoint
    checkpoint_path = None
    max_epoch = -1
    for file in os.listdir(model_path):
        if file.startswith("best_model_") and file.endswith(".pt"):
            try:
                epoch = int(file[:-3].split("_")[-1])
                if epoch > max_epoch:
                    max_epoch = epoch
                    checkpoint_path = os.path.join(model_path, file)
            except ValueError:
                if checkpoint_path is None:
                    checkpoint_path = os.path.join(model_path, file)
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint file (best_model_*.pt) found in {model_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    normalizer, conditioner = model.load_model(checkpoint_path, device)
    model.eval()
    
    return model, normalizer, conditioner, config


def setup_data_loader(config, normalizer, device="cpu"):
    """
    Set up a simple test data loader for evaluation.
    For evaluation, we load data directly as a static dataset rather than
    using the streaming mechanism from training.
    
    Args:
        config: Training config loaded from YAML
        normalizer: Normalizer object (to reuse stats from training)
        device: Device to use
    
    Returns:
        DataLoader instance with test data
    """
    data_cfg = config.data

    # Find data files
    data_dir = getattr(config, "data_dir", None) or getattr(data_cfg, "data_dir", None) or getattr(data_cfg, "path", "data/_parquet_cache")
    configured_files = list(getattr(config, "data_files", []) or getattr(data_cfg, "data_files", []))

    available_parquet_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                available_parquet_files.append(os.path.join(root, file))

    parquet_files = []
    if configured_files:
        for file_path in configured_files:
            if os.path.isabs(file_path):
                parquet_files.append(file_path)
            else:
                parquet_files.append(os.path.join(data_dir, file_path))
        parquet_files = [path for path in parquet_files if os.path.exists(path)]
        if not parquet_files and available_parquet_files:
            print(
                f"Warning: none of the configured data_files were found in {data_dir}; "
                "falling back to all parquet files in that directory."
            )
            parquet_files = sorted(available_parquet_files)
    else:
        parquet_files = [path for path in available_parquet_files if os.path.exists(path)]

    parquet_files = sorted(parquet_files)
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    # Load data as static dataset
    print(f"Loading data for evaluation...")
    window_size = int(data_cfg.window.input_len)
    stride = int(getattr(data_cfg.window, "stride", 1))
    batch_size = int(data_cfg.batch_size)
    
    dataset = load_parquet_data_for_eval(
        parquet_files,
        window_size=window_size,
        stride=stride,
        normalizer=normalizer,
        n_limit=None,  # Use all data
        meter_col=data_cfg.columns.meter_id,
        value_cols=data_cfg.columns.value_cols,
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers for simplicity in evaluation
        pin_memory=(str(device) != "cpu"),
    )
    
    return test_loader


def run_evaluation(model, test_loader, normalizer, conditioner, config, output_path, device="cpu"):
    """
    Run the evaluation framework on the test loader.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        normalizer: Normalizer object
        conditioner: Conditioner object (if CVAE)
        config: Training config
        output_path: Path to save evaluation results
        device: Device to use for evaluation
    """
    os.makedirs(output_path, exist_ok=True)
    
    eval_cfg = getattr(config, "evaluation", SimpleNamespace())
    eval_device = str(getattr(eval_cfg, "device", device))
    default_synth_samples = 300 if model.__class__.__name__ == "AutoRegressiveTransformerVAE" else 1000
    n_synth_samples = int(getattr(eval_cfg, "n_synthetic_samples", default_synth_samples))

    eval_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
    if eval_normalizer is not None and hasattr(eval_normalizer, "to"):
        eval_normalizer = eval_normalizer.to(eval_device)

    phased_mode = bool(getattr(config.data, "phased", True))
    
    print(f"\nRunning evaluation (phased_mode={phased_mode})...")
    print(f"  Output path: {output_path}")
    print(f"  Device: {eval_device}")
    print(f"  Synthetic samples: {n_synth_samples}")

    if phased_mode:
        evaluator = PhasedEvaluationSuite(
            model,
            test_loader,
            normalizer=eval_normalizer,
            conditioner=conditioner,
            device=eval_device,
            window_size=int(config.data.window.input_len),
            n_synthetic_samples=n_synth_samples,
            output_path=output_path,
        )
        try:
            evaluator.evaluate()
            print("✓ Evaluation completed successfully.")
        except RuntimeError as e:
            if eval_device != "cpu" and "out of memory" in str(e).lower():
                print("Warning: GPU OOM during evaluation. Retrying on CPU.")
                cpu_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
                evaluator = PhasedEvaluationSuite(
                    model,
                    test_loader,
                    normalizer=cpu_normalizer,
                    conditioner=conditioner,
                    device="cpu",
                    window_size=int(config.data.window.input_len),
                    n_synthetic_samples=n_synth_samples,
                    output_path=output_path,
                )
                evaluator.evaluate()
                print("✓ Evaluation completed successfully on CPU.")
            else:
                raise
    else:
        evaluator = EvalFramework(
            model,
            test_loader,
            eval_normalizer,
            conditioner=conditioner,
            device=eval_device,
            n_synthetic_samples=n_synth_samples,
            output_path=output_path,
        )
        evaluator.register_all()
        try:
            evaluator.evaluate()
            print("✓ Evaluation completed successfully.")
        except RuntimeError as e:
            if eval_device != "cpu" and "out of memory" in str(e).lower():
                print("Warning: GPU OOM during evaluation. Retrying on CPU.")
                cpu_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
                evaluator = EvalFramework(
                    model,
                    test_loader,
                    cpu_normalizer,
                    conditioner=conditioner,
                    device="cpu",
                    n_synthetic_samples=n_synth_samples,
                    output_path=output_path,
                )
                evaluator.register_all()
                evaluator.evaluate()
                print("✓ Evaluation completed successfully on CPU.")
            else:
                raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load a trained model and rerun evaluation with training config."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model folder (e.g., models/hpc_runs/my_model_20260510_123456)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    model_path = str(args.model_path).rstrip("/")
    device = str(args.device)
    
    print(f"\n{'='*70}")
    print("RERUN EVALUATION FOR TRAINED MODEL")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}\n")
    
    # Check that model folder exists
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")
    
    # 1. Load model
    print("Step 1: Loading model...")
    model, normalizer, conditioner, config = load_model(model_path, device=device)
    
    # 2. Setup data loader
    print("\nStep 2: Setting up data loader...")
    test_loader = setup_data_loader(config, normalizer, device=device)
    
    # 3. Verify test loader is ready
    print("\nStep 3: Verifying test data...")
    if test_loader is None or len(test_loader) == 0:
        raise ValueError("Could not prepare test loader. No data available.")
    
    print(f"✓ Test loader ready with {len(test_loader.dataset)} windows")
    
    # 4. Create output folder with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    eval_output_dir = os.path.join(model_path, f"rerun_evaluation_{timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    print(f"\nStep 4: Running evaluation...")
    print(f"Output directory: {eval_output_dir}\n")
    
    # 5. Run evaluation
    run_evaluation(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        conditioner=conditioner,
        config=config,
        output_path=eval_output_dir,
        device=device,
    )
    
    print(f"\n{'='*70}")
    print(f"✓ EVALUATION COMPLETE")
    print(f"Results saved to: {eval_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


