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
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model.factory import _load_train_config
from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec
from src.evaluation.load_model import load_model
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
 
def _load_config_for_model(model_path: str):
    """Load config snapshot stored with a trained model folder."""
    preferred = os.path.join(model_path, "training_config_used.yaml")
    if os.path.exists(preferred):
        print(f"Loading config from: {preferred}")
        return _load_train_config(preferred)

    for file in os.listdir(model_path):
        if file.endswith(".yaml") or file.endswith(".yml"):
            config_path = os.path.join(model_path, file)
            print(f"Loading config from: {config_path}")
            return _load_train_config(config_path)

    raise FileNotFoundError(f"No config file found in {model_path}")


def setup_data_loader(config, normalizer, device="cpu"):
    """Build test loader using the phased replay dataloader implementation."""
    data_cfg = config.data
    data_mode = str(getattr(data_cfg, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}

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

    window = WindowSpec(
        input_len=int(data_cfg.window.input_len),
        stride=int(getattr(data_cfg.window, "stride", 1)),
    )

    replay_capacity = int(getattr(data_cfg, "replay_buffer_capacity", 10000))
    min_buffer_size = int(getattr(data_cfg, "min_buffer_size", 2000))
    workers = int(getattr(data_cfg, "num_workers", 0))
    eval_batches = int(getattr(data_cfg, "eval_batches", 5))
    skip_corrupt_files = bool(getattr(data_cfg, "skip_corrupt_files", True))
    max_skip_warnings = int(getattr(data_cfg, "max_skip_warnings", 20))
    output_aggregate_stream = bool(getattr(data_cfg, "output_aggregate_stream", False))
    hourly_aggregation = bool(getattr(data_cfg.window, "aggregated", False))
    aggregation_energy_unit = str(getattr(data_cfg, "aggregation_energy_unit", "wh")).lower()
    agg_divisor = 1000.0 if hourly_aggregation and aggregation_energy_unit == "kwh" else float(getattr(data_cfg, "aggregation_divisor", 1.0))

    # Ensure test split exists for evaluation if training used test_ratio=0.
    train_ratio = float(getattr(data_cfg.splits, "train_ratio", 1.0))
    val_ratio = float(getattr(data_cfg.splits, "val_ratio", 0.0))
    test_ratio = float(getattr(data_cfg.splits, "test_ratio", 0.0))
    if test_ratio <= 0.0:
        print("Warning: test_ratio<=0 in config; using 10% test split for evaluation.")
        test_ratio = 0.1
        total = train_ratio + val_ratio + test_ratio
        train_ratio = train_ratio / total
        val_ratio = val_ratio / total
        test_ratio = test_ratio / total

    loader_obj = PhasedReplayDataloaders(
        sm_files=parquet_files,
        replay_buffer_capacity=replay_capacity,
        min_buffer_size=min_buffer_size,
        window_spec=window,
        normalizer=normalizer,
        num_workers=workers,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=agg_divisor,
        batch_size=int(data_cfg.batch_size),
        eval_cache_device=None,
        eval_cache_batches=eval_batches,
        cache_scan_batches=None,
        pin_memory=(str(device) != "cpu"),
        output_aggregate_stream=output_aggregate_stream,
        skip_corrupt_files=skip_corrupt_files,
        max_skip_warnings=max_skip_warnings,
        phased_mode=phased_mode,
        feature_cols=data_cfg.columns.value_cols,
        meter_col=data_cfg.columns.meter_id,
        split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
        split_ratios={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        max_value=float(getattr(data_cfg, "max_value", None)) or None,
        max_hour_value=float(getattr(data_cfg, "max_hour_value", 20.0)) or None,
    )

    n_batches = max(1, eval_batches)
    loader_obj.refresh_test_cache(n_batches)
    test_loader = loader_obj.get_test_loader()

    # Fallback when streamed cache is empty but test dataset exists.
    if test_loader is None and hasattr(loader_obj, "test_dataset"):
        print("[WARN] get_test_loader() returned None; building DataLoader from test_dataset.")
        test_loader = DataLoader(
            loader_obj.test_dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(str(device) != "cpu"),
        )

    # If split-based test stream is empty, force all data into test split for rerun evaluation.
    if test_loader is not None:
        try:
            _ = next(iter(test_loader))
        except StopIteration:
            print("[WARN] test split produced zero batches; retrying with forced test_ratio=1.0.")
            loader_obj = PhasedReplayDataloaders(
                sm_files=parquet_files,
                replay_buffer_capacity=replay_capacity,
                min_buffer_size=min_buffer_size,
                window_spec=window,
                normalizer=normalizer,
                num_workers=workers,
                hourly_aggregation=hourly_aggregation,
                aggregation_divisor=agg_divisor,
                batch_size=int(data_cfg.batch_size),
                eval_cache_device=None,
                eval_cache_batches=eval_batches,
                cache_scan_batches=None,
                pin_memory=(str(device) != "cpu"),
                output_aggregate_stream=output_aggregate_stream,
                skip_corrupt_files=skip_corrupt_files,
                max_skip_warnings=max_skip_warnings,
                phased_mode=phased_mode,
                feature_cols=data_cfg.columns.value_cols,
                meter_col=data_cfg.columns.meter_id,
                split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
                split_ratios={"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0},
                max_value=float(getattr(data_cfg, "max_value", None)) or None,
                max_hour_value=float(getattr(data_cfg, "max_hour_value", 20.0)) or None,
            )
            loader_obj.refresh_test_cache(n_batches)
            test_loader = loader_obj.get_test_loader()
            if test_loader is None and hasattr(loader_obj, "test_dataset"):
                test_loader = DataLoader(
                    loader_obj.test_dataset,
                    batch_size=int(data_cfg.batch_size),
                    shuffle=False,
                    num_workers=0,
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
    model, normalizer, conditioner = load_model(model_path, device=device)
    config = _load_config_for_model(model_path)
    
    # 2. Setup data loader
    print("\nStep 2: Setting up data loader...")
    test_loader = setup_data_loader(config, normalizer, device=device)
    
    # 3. Verify test loader is ready
    print("\nStep 3: Verifying test data...")
    if test_loader is None:
        raise ValueError("Could not prepare test loader. No data available.")

    try:
        batch_probe = next(iter(test_loader))
    except StopIteration:
        raise ValueError("Could not prepare test loader. Test split produced zero batches.")

    if not isinstance(batch_probe, (list, tuple)) or len(batch_probe) != 4:
        raise ValueError("Unexpected batch format from test loader; expected 4-tuple.")

    print("✓ Test loader ready")
    
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


