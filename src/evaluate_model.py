import argparse
import os
import sys
import random
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.model.set_guide_vae import SetGuideVAE
from src.train import _load_train_config, _build_phased_eval_loader, _run_evaluation_and_visualization
from src.data.dataloader_phased import StreamingParquetBuffer, WindowSpec
from src.visualization.writer import Writer
from src.data.conditioner import Conditioner, UserProfiler, DayOfWeekConditioner, MonthOfYearConditioner, HolidayConditioner

def create_conditioner_from_config(config):
    conditioner = Conditioner()
    tags = getattr(config.conditioning, "tags", [])
    
    for tag in tags:
        if tag == "day_of_week":
            conditioner.add_condition_item(tag, DayOfWeekConditioner())
        elif tag == "user_profile":
            user_profile_config = config.conditioning.user_profile
            if getattr(user_profile_config, "online", False):
                online_cfg = getattr(config.conditioning, "online_user_profile", None)
                decay = getattr(online_cfg, "decay", 0.9) if online_cfg else 0.9
                update_freq = getattr(online_cfg, "update_frequency", 10) if online_cfg else 10
            else:
                decay = 1.0
                update_freq = -1
                
            up = UserProfiler(
                n_topics=getattr(user_profile_config, "n_topics", 10),
                num_clusters=getattr(user_profile_config, "num_clusters", 100),
                path=getattr(user_profile_config, "cache_path", 'models/UserProfiles/user_profiles.pkl'),
                verbose=getattr(user_profile_config, "verbose", False),
                decay=decay,
                update_frequency=update_freq
            )
            conditioner.add_condition_item(tag, up)
        elif tag == "holiday":
            conditioner.add_condition_item(tag, HolidayConditioner())
        elif tag == "month_of_year":
            conditioner.add_condition_item(tag, MonthOfYearConditioner())
    
    return conditioner

def evaluate_model(model_path, num_test_files=2, config_path="config/config.yaml"):
    print(f"Loading configuration from {config_path}")
    config = _load_train_config(config_path)
    
    device = torch.device(config.model.device if hasattr(config, 'model') and hasattr(config.model, 'device') else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {model_path}")
    model_dir = os.path.dirname(model_path)
    
    from src.model.factory import create_model_from_config
    model = create_model_from_config(config, device)
    normalizer, _ = model.load_model(model_path, device=device)
    model.eval()
    
    conditioner = getattr(model, 'conditioner', None)
    if conditioner is not None:
        for tag, item in conditioner.condition_items.items():
            if isinstance(item, UserProfiler):
                try:
                    item.load(item.path)
                    print(f"Loaded UserProfiler state from {item.path}")
                except Exception as e:
                    print(f"Warning: Failed to load UserProfiler state from {item.path}: {e}")

    data_dir = getattr(config, "data_dir", None)
    if not data_dir or not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}. Please check your configuration.")
    
    print(f"Sampling {num_test_files} files from {data_dir}")
    all_files = []
    
    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phase", "parquet", "phased_parquet"}
    
    if phased_mode:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and not f.startswith('test')]
    else:
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('test')]
        
    if not all_files:
        raise ValueError(f"No suitable data files found in {data_dir}")
        
    sampled_files = random.sample(all_files, min(num_test_files, len(all_files)))
    sampled_file_paths = [os.path.join(data_dir, f) for f in sampled_files]
    print(f"Sampled files: {sampled_file_paths}")

    writer = Writer(model_dir)

    print("Building evaluation loader...")
    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)
    
    if phased_mode:
        phased_cfg = getattr(config.data, "phased", None)
        aggregation_energy_unit = str(getattr(phased_cfg, "aggregation_energy_unit", "wh")).lower()
        if aggregation_energy_unit not in {"wh", "kwh"}:
            print(f"Warning: unknown data.phased.aggregation_energy_unit='{aggregation_energy_unit}'. Falling back to 'wh'.")
            aggregation_energy_unit = "wh"
        hourly_aggregation = bool(getattr(config.window, "aggregated", False))
        aggregation_divisor = 1000.0 if aggregation_energy_unit == "kwh" else 1.0

        eval_ds = StreamingParquetBuffer(
            sm_files=sampled_file_paths,
            window_spec=window,
            normalizer=normalizer,
            replay_buffer_capacity=max(1000, config.data.phased.replay_buffer_capacity),
            num_workers=getattr(config.advanced, "num_workers", 0) if hasattr(config, "advanced") else 0,
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
        )
        
        batch_size = getattr(config.dataloader, "batch_size", 2048)
        eval_batches = getattr(config.data.phased, "eval_batches", 5)
        num_workers = getattr(config.advanced, "num_workers", 0) if hasattr(config, "advanced") else 0
        
        from torch.utils.data import DataLoader
        eval_stream = DataLoader(eval_ds, batch_size=batch_size, num_workers=num_workers)
        
        test_loader = _build_phased_eval_loader(
            eval_stream=eval_stream,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_batches=eval_batches,
            normalizer=normalizer
        )
    else:
        raise NotImplementedError("Evaluation for non-phased data mode is not currently implemented in this script.")

    print(f"Starting evaluation and visualization (Outputs will be saved in {model_dir})")
    _run_evaluation_and_visualization(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        conditioner=conditioner,
        writer=writer,
        device=device,
        log_path=model_dir,
        config=config,
        phased_mode=phased_mode
    )
    print(f"Evaluation complete. Results and visualizations saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize a pre-trained SetGuideVAE model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--num_test_files", type=int, default=2, help="Number of random data files to sample for evaluation.")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to the configuration file.")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.num_test_files, args.config_path)
