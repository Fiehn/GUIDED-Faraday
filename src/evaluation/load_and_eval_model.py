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


def setup_data_loader(config, normalizer, device="cpu", model_path=None):
    """Build test loader using the phased replay dataloader implementation."""
    data_cfg = config.data
    data_mode = str(getattr(data_cfg, "mode", "csv")).lower()
    # data_mode = "phased" # hardcode for testing phase splitter
    phased_mode = data_mode in {"phased", "phased_parquet"}

    # data_dir = getattr(config, "data_dir", None) or getattr(data_cfg, "data_dir", None) or getattr(data_cfg, "path", "data/_parquet_cache")
    data_dir = getattr(config, "data_dir", None) or getattr(data_cfg, "data_dir", None) or getattr(data_cfg, "path", "/Volumes/USB/parq_hourly")
    # data_dir = getattr(config, "data_dir", None) or getattr(data_cfg, "data_dir", None) or getattr(data_cfg, "path", "/Volumes/USB/raw_profiles")
    # data_dir = "/Volumes/USB/raw_profiles" # hardcode for testing phase splitter
    configured_files = list(getattr(config, "data_files", []) or getattr(data_cfg, "data_files", []))

    # Match training file selection: walk data_dir and include all .parquet
    # except files that start with 'test' (training excluded those).
    data_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".parquet") and not f.startswith("test"):
                data_paths.append(os.path.join(root, f))
    available_parquet_files = sorted(data_paths)

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
    workers = int(getattr(data_cfg, "num_workers", 4))
    eval_batches = int(getattr(data_cfg, "eval_batches", 5))
    skip_corrupt_files = bool(getattr(data_cfg, "skip_corrupt_files", True))
    max_skip_warnings = int(getattr(data_cfg, "max_skip_warnings", 20))
    output_aggregate_stream = bool(getattr(data_cfg, "output_aggregate_stream", False))
    hourly_aggregation = bool(getattr(data_cfg.window, "aggregated", False))
    aggregation_energy_unit = str(getattr(data_cfg, "aggregation_energy_unit", "wh")).lower()
    agg_divisor = 1000.0 if hourly_aggregation and aggregation_energy_unit == "kwh" else 1.0
    # agg_divisor = float(getattr(data_cfg, "aggregation_divisor", 1.0))                        # use this when doing 1-phase models
    cache_scan_batches = int(getattr(data_cfg, "cache_scan_batches", 0)) or None
    max_value_cfg = getattr(data_cfg, "max_value", None)
    max_value = float(max_value_cfg) if max_value_cfg is not None else None

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
        cache_scan_batches=cache_scan_batches,
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
        max_value=max_value,
        max_hour_value=float(getattr(data_cfg, "max_hour_value", 20.0)) or None,
    )

    if phased_mode:
        print(f"Phased aggregation unit: {aggregation_energy_unit} (divisor={agg_divisor:g})")

    n_batches = max(1, eval_batches)
    loader_obj.refresh_test_cache(n_batches)
    test_loader = loader_obj.get_test_loader()

    # train_loader = None
    # try:
    #     train_loader = loader_obj.get_train_epoch_loader(steps_per_epoch=eval_batches)
    # except Exception as e:
    #     print(f"[WARN] Could not build train loader for MIA: {e}")

    # Verify data config matches training
    if model_path is not None:
        data_summary_path = os.path.join(model_path, "data_config_summary.json")
        if os.path.exists(data_summary_path):
            try:
                import json as json_module
                with open(data_summary_path, "r") as f:
                    training_data_config = json_module.load(f)
                rerun_config = {
                    "data_files": parquet_files,
                    "num_files": len(parquet_files),
                    "split_seed": int(getattr(data_cfg.splits, "random_seed", 0)),
                    "split_ratios": {
                        "train_ratio": train_ratio,
                        "val_ratio": val_ratio,
                        "test_ratio": test_ratio,
                    },
                    "batch_size": int(data_cfg.batch_size),
                    "window_input_len": int(data_cfg.window.input_len),
                    "window_stride": int(getattr(data_cfg.window, "stride", 1)),
                    "replay_buffer_capacity": replay_capacity,
                    "min_buffer_size": min_buffer_size,
                    "num_workers": workers,
                    "hourly_aggregation": hourly_aggregation,
                    "aggregation_divisor": agg_divisor,
                    "max_value": max_value,
                    "max_hour_value": float(getattr(data_cfg, "max_hour_value", 20.0)) or None,
                }
                key_diffs = []
                for key in ["split_seed", "batch_size", "window_input_len", "window_stride", "replay_buffer_capacity", "min_buffer_size", "num_workers", "hourly_aggregation", "aggregation_divisor", "max_value", "max_hour_value"]:
                    if training_data_config.get(key) != rerun_config.get(key):
                        key_diffs.append(f"{key}: training={training_data_config.get(key)} vs rerun={rerun_config.get(key)}")
                if training_data_config.get("split_ratios") != rerun_config.get("split_ratios"):
                    key_diffs.append(f"split_ratios: training={training_data_config.get('split_ratios')} vs rerun={rerun_config.get('split_ratios')}")
                if key_diffs:
                    print("[WARN] Data config differences detected between training and rerun:")
                    for diff in key_diffs:
                        print(f"  - {diff}")
                if training_data_config.get("num_files") != rerun_config.get("num_files"):
                    print(f"[WARN] File count mismatch: training={training_data_config.get('num_files')} vs rerun={rerun_config.get('num_files')}")
                else:
                    print(f"✓ Data config matches training ({training_data_config.get('num_files')} files, seed={training_data_config.get('split_seed')})")
            except Exception as e:
                print(f"[WARN] Could not verify data config: {e}")

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
                cache_scan_batches=cache_scan_batches,
                pin_memory=(str(device) != "cpu"),
                output_aggregate_stream=output_aggregate_stream,
                skip_corrupt_files=skip_corrupt_files,
                max_skip_warnings=max_skip_warnings,
                phased_mode=phased_mode,
                feature_cols=data_cfg.columns.value_cols,
                meter_col=data_cfg.columns.meter_id,
                split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
                split_ratios={"train_ratio": 0.0, "val_ratio": 0.0, "test_ratio": 1.0},
                max_value=max_value,
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

    return test_loader #, train_loader


def run_evaluation(model, test_loader, normalizer, conditioner, config, output_path, device="cpu", secondary_model=None, train_loader=None):
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
    default_synth_samples = 1000
    n_synth_samples = int(getattr(eval_cfg, "n_synthetic_samples", default_synth_samples))

    eval_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
    if eval_normalizer is not None and hasattr(eval_normalizer, "to"):
        eval_normalizer = eval_normalizer.to(eval_device)

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}
    is_diffusion = hasattr(model, 'noise_scheduler') and hasattr(model, 'reconstruct')
    mia_k = 5 if is_diffusion else 1

    print(f"\nRunning evaluation (phased_mode={phased_mode})...")
    print(f"  Output path: {output_path}")
    print(f"  Device: {eval_device}")
    print(f"  Synthetic samples: {n_synth_samples}")
    if train_loader is not None:
        print(f"  MIA enabled (K={mia_k} reconstruction samples)")

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
            secondary_model=secondary_model,
            train_dataloader=train_loader,
            mia_reconstruction_samples=mia_k,
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
                    secondary_model=secondary_model,
                    train_dataloader=train_loader,
                    mia_reconstruction_samples=mia_k,
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
            secondary_model=secondary_model,
            train_dataloader=train_loader,
            mia_reconstruction_samples=mia_k,
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
                    secondary_model=secondary_model,
                    train_dataloader=train_loader,
                    mia_reconstruction_samples=mia_k,
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

    parser.add_argument(
        "--phase-splitter-path",
        type=str,
        default=None,
        help="Optional path to a trained HierarchicalSplitter folder used to split generated aggregate profiles into phases.",
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
    secondary_model = None
    if args.phase_splitter_path is not None:
        print(f"Loading phase splitter from: {args.phase_splitter_path}")
        sec_model, sec_normalizer, sec_conditioner = load_model(args.phase_splitter_path, device=device)
        # Attach the secondary model's normalizer to the model instance so
        # downstream code can access it during evaluation (e.g. phase-split inverse transform).
        try:
            setattr(sec_model, "normalizer", sec_normalizer)
        except Exception:
            pass
        secondary_model = sec_model
    
    # 2. Setup data loader
    print("\nStep 2: Setting up data loader...")
    # If a phase splitter was provided, use the splitter's config and normalizer
    # to build the test loader (real 3-phase data). The generator will still
    # use the primary model (which may produce 1-phase aggregates) and the
    # secondary model will be used internally to split generated aggregates.
    if secondary_model is not None and args.phase_splitter_path is not None:
        try:
            sec_cfg = _load_config_for_model(args.phase_splitter_path)
            print(f"Using phase-splitter data config from: {args.phase_splitter_path}")
            test_loader = setup_data_loader(sec_cfg, sec_normalizer, device=device, model_path=args.phase_splitter_path)
        except Exception as e:
            print(f"[WARN] Could not build test loader from phase-splitter config: {e}")
            print("Falling back to primary model data loader.")
            test_loader = setup_data_loader(config, normalizer, device=device, model_path=model_path)
    else:
        # Default: use primary model's data config (existing behavior)
        test_loader = setup_data_loader(config, normalizer, device=device, model_path=model_path)

    # 3. Verify test loader is ready
    print("\nStep 3: Verifying test data...")
    if test_loader is None:
        raise ValueError("Could not prepare test loader. No data available.")

    try:
        batch_probe = next(iter(test_loader))
    except StopIteration:
        raise ValueError(
            "Could not prepare test loader. The test stream is empty even after forcing test_ratio=1.0. "
            "This usually means the replay buffer threshold is too high for the available test windows, "
            "or the cached parquet data contains no valid windows after filtering."
        )

    if not isinstance(batch_probe, (list, tuple)) or len(batch_probe) != 4:
        raise ValueError("Unexpected batch format from test loader; expected 4-tuple.")

    print("✓ Test loader ready")
    
    # 4. Create output folder with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    print(f"[DEBUG] main model_path = {model_path}")
    print(f"[DEBUG] phase_splitter_path = {args.phase_splitter_path}")
    eval_output_dir = os.path.join(model_path, f"rerun_evaluation_{timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)
    print(f"[DEBUG] eval_output_dir = {eval_output_dir}")
    
    print(f"\nStep 4: Running evaluation...")
    print(f"Output directory: {eval_output_dir}\n")
    
    # 5. Run evaluation
    run_evaluation(
        model=model,
        test_loader=test_loader,
        # train_loader=train_loader,
        normalizer=normalizer,
        conditioner=conditioner,
        config=config,
        output_path=eval_output_dir,
        device=device,
        secondary_model=secondary_model,
    )
    
    print(f"\n{'='*70}")
    print(f"✓ EVALUATION COMPLETE")
    print(f"Results saved to: {eval_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


