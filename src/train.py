import os
import sys
import argparse
import shutil
import yaml
import copy
import json
from types import SimpleNamespace
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model.encoder_decoder_model import *
from data.conditioner import *
from src.model.cvae import *
from src.model.set import *
from src.model.vae import *
from data.dataloader_phased import *
from data.normalizer import *
from src.model.hierarchical.aggregation import *
from src.model.hierarchical.config import *
from src.visualization.visualizer import Visualizer
from src.visualization.writer import Writer
from evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
from src.visualization.plot_styling import PlotStyle
from src.model.factory import *

def _run_evaluation_and_visualization(model, test_loader, normalizer, conditioner, writer, device, log_path, config, phased_mode: bool, secondary_model = None):
    """Route evaluation by mode and run shared visualizations."""
    if test_loader is None:
        print("Warning: test_loader is None; skipping evaluation and visualization.")
        return
    try:
        if len(test_loader) == 0:
            print("Warning: test_loader has no batches; skipping evaluation and visualization.")
            return
    except TypeError:
        # Some iterable loaders may not implement __len__; continue and rely on iteration.
        pass

    eval_cfg = getattr(config, "evaluation", SimpleNamespace())
    eval_device = str(getattr(eval_cfg, "device", device))
    default_synth_samples = 300 if model.__class__.__name__ == "AutoRegressiveTransformerVAE" else 1000
    n_synth_samples = int(getattr(eval_cfg, "n_synthetic_samples", default_synth_samples))

    eval_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
    if eval_normalizer is not None and hasattr(eval_normalizer, "to"):
        eval_normalizer = eval_normalizer.to(eval_device)

    if phased_mode:
        phased_evaluator = PhasedEvaluationSuite(
            model,
            test_loader,
            normalizer=eval_normalizer,
            conditioner=conditioner,
            device=eval_device,
            window_size=config.window.input_len,
            n_synthetic_samples=n_synth_samples,
            output_path=log_path,
            secondary_model=secondary_model,
        )
        try:
            phased_evaluator.evaluate()
        except RuntimeError as e:
            # Fall back to CPU for evaluation if GPU memory is insufficient.
            if eval_device != "cpu" and "out of memory" in str(e).lower():
                print("Warning: GPU OOM during evaluation. Retrying evaluation on CPU.")
                cpu_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
                if cpu_normalizer is not None and hasattr(cpu_normalizer, "to"):
                    cpu_normalizer = cpu_normalizer.to("cpu")
                phased_evaluator = PhasedEvaluationSuite(
                    model,
                    test_loader,
                    normalizer=cpu_normalizer,
                    conditioner=conditioner,
                    device="cpu",
                    window_size=config.window.input_len,
                    n_synthetic_samples=n_synth_samples,
                    output_path=log_path,
                )
                phased_evaluator.evaluate()
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
            output_path=log_path,
        )
        evaluator.register_all()
        try:
            evaluator.evaluate()
        except RuntimeError as e:
            if eval_device != "cpu" and "out of memory" in str(e).lower():
                print("Warning: GPU OOM during evaluation. Retrying evaluation on CPU.")
                cpu_normalizer = copy.deepcopy(normalizer) if normalizer is not None else None
                if cpu_normalizer is not None and hasattr(cpu_normalizer, "to"):
                    cpu_normalizer = cpu_normalizer.to("cpu")
                evaluator = EvalFramework(
                    model,
                    test_loader,
                    cpu_normalizer,
                    conditioner=conditioner,
                    device="cpu",
                    n_synthetic_samples=n_synth_samples,
                    output_path=log_path,
                    secondary_model=secondary_model,
                )
                evaluator.register_all()
                evaluator.evaluate()
            else:
                raise
        evaluator.save_results(os.path.join(log_path, 'eval_results.json'))

    visualizer = Visualizer(model=model, writer=writer,
                            plot_style=PlotStyle(),
                            log_dir=log_path,
                            device=device,
                            normalizer=normalizer,
                            conditioner=conditioner,
                            secondary_model=secondary_model)
    visualizer.visualize(test_loader=test_loader,
                         num_samples=config.visualize.num_samples,
                         num_recon_per=config.visualize.num_recon_per,
                         num_synthetic=config.visualize.num_synthetic,
                         num_originals=config.visualize.num_originals,
                         num_synthetic_per=config.visualize.num_synthetic_per)

def train(config, train_only_user_profile: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to be 999

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}
    
    from src.model.factory import create_model_from_config
    model = create_model_from_config(config, device)
    conditioner = getattr(model, "conditioner", None)
    
    model_type = str(getattr(config.model, "type", "vae")).lower()

    # === Save config snapshot and generate Tag ===
    model_tag = model.get_tag()
    log_path = os.path.join(config.logging.log_dir, model_tag)
    os.makedirs(log_path, exist_ok=True)
    _save_training_config_snapshot(config=config, log_path=log_path)

    # === Data Configuration === 
    data_dir = config.data_dir
    nan_as_zero = bool(getattr(config.data, "nan_as_zero", False))
    parquet_only = bool(getattr(config.data, "parquet_only", False))

    if config.window.aggregated and config.window.input_len != 24:
        print(f"Warning: window.aggregated is True but window.input_len is {config.window.input_len}. For hourly aggregation, consider setting input_len to 24.")
        config.window.input_len = 24
        config.window.stride = 24

    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)

    writer = Writer(log_dir=os.path.join(log_path, 'logs'))

    # === Data loader ===
    data_paths = []
    configured_files = list(config.data_files) if len(config.data_files) > 0 else []
    if len(configured_files) > 0:
        for f in configured_files:
            if os.path.isabs(f):
                data_paths.append(f)
            else:
                data_paths.append(os.path.join(data_dir, f))
    else:
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".parquet") and not f.startswith("test"):
                    data_paths.append(os.path.join(root, f))
    
    parquet_paths = sorted([p for p in data_paths if os.path.exists(p)])
    if len(parquet_paths) == 0:
        raise ValueError(f"No data files (.parquet) found in: {data_dir}")
    
    data_cfg = getattr(config, "data", SimpleNamespace())
    
    if phased_mode:
        print(f"Data is Phased.")
        aggregation_energy_unit = str(getattr(data_cfg, "aggregation_energy_unit", "wh")).lower()
        if aggregation_energy_unit not in {"wh", "kwh"}:
            print(f"Warning: unknown data.aggregation_energy_unit='{aggregation_energy_unit}'. Falling back to 'wh'.")
            aggregation_energy_unit = "wh"
        aggregation_divisor = 1000.0 if bool(config.window.aggregated) and aggregation_energy_unit == "kwh" else 1.0
        if bool(config.window.aggregated):
            print(f"Hourly aggregation unit: {aggregation_energy_unit} (divisor={aggregation_divisor:g})")
        
        if model_type == "phase_splitter":
            print("For phase_splitter, forcing data.phased.output_aggregate_stream=true.")
            output_aggregate_stream = True
        else:
            output_aggregate_stream = bool(getattr(config.data, "output_aggregate_stream", False))
    else:
        print(f"Data is CSV/Parquet without phased processing. Aggregation energy unit and divisor are not applicable in this mode.")
        aggregation_energy_unit = "kwh"
        aggregation_divisor = 1.0
        output_aggregate_stream = False
    

    replay_capacity = int(getattr(data_cfg, "replay_buffer_capacity", 10000))
    min_buffer_size = int(getattr(data_cfg, "min_buffer_size", 2000))
    workers = int(getattr(data_cfg, "num_workers", 4))
    eval_batches = int(getattr(data_cfg, "eval_batches", 5))
    eval_cache_on_gpu = bool(getattr(data_cfg, "eval_cache_on_gpu", False))
    cache_scan_batches = int(getattr(data_cfg, "cache_scan_batches", 0)) or None
    skip_corrupt_files = bool(getattr(data_cfg, "skip_corrupt_files", True))
    max_skip_warnings = int(getattr(data_cfg, "max_skip_warnings", 20))
    
    normalizer_cfg = getattr(config, "normalizer", SimpleNamespace())
    norm_method = str(getattr(normalizer_cfg, "method", "log1p")).lower() if not phased_mode else "log1p_phase"
    norm_cache_dir = getattr(normalizer_cfg, "cache_dir", None)
    if isinstance(norm_cache_dir, str) and norm_cache_dir.strip() == "":
        norm_cache_dir = None
    norm_force_recompute = bool(getattr(normalizer_cfg, "force_recompute", False))
    normalizer = Normalizer(method=norm_method)
    normalizer.fit_normalizer_on_files(
        filenames=parquet_paths,
        window_spec=window,
        feature_cols=data_cfg.columns.value_cols,
        meter_col=data_cfg.columns.meter_id,
        hourly_aggregation=bool(data_cfg.window.aggregated),
        aggregation_divisor=aggregation_divisor,
        nan_as_zero=nan_as_zero,
        phased=phased_mode,
        force_recompute=norm_force_recompute,
        save_path=norm_cache_dir,
        device=device,
    )
    normalizer.save(os.path.join(log_path, "normalizer.pt"))

    if normalizer is not None:
        # Attach the normalizer to the model so it can be automatically saved/loaded
        model.normalizer = copy.deepcopy(normalizer)

    if conditioner.condition_items.get("user_profile") is not None:
        conditioner.condition_items.get("user_profile", None).fit_on_files(parquet_paths)
        conditioner.condition_items.get("user_profile", None).save(os.path.join(log_path, "user_profiler.pt"))

    eval_cache_device = device if eval_cache_on_gpu and str(device) != "cpu" else None

    data_loader = PhasedReplayDataloaders(
            sm_files=parquet_paths,
            replay_buffer_capacity=replay_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=window,
            normalizer=normalizer,
            num_workers=workers,
            hourly_aggregation=data_cfg.window.aggregated,
            aggregation_divisor=aggregation_divisor,
            batch_size=data_cfg.batch_size,
            eval_cache_device=eval_cache_device,
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
                "train_ratio": float(getattr(data_cfg.splits, "train_ratio", 1.0)),
                "val_ratio": float(getattr(data_cfg.splits, "val_ratio", 0.0)),
                "test_ratio": float(getattr(data_cfg.splits, "test_ratio", 0.0)),
            },
            max_value=float(getattr(data_cfg, "max_value", None)) or None,
        )

    model.fit(data_loader=data_loader,
              device=device,
                writer=writer,
                logging_dir = log_path,
                data_cfg=data_cfg,
                checkpoint_cfg=getattr(config.training, "checkpointing", SimpleNamespace()),
                train_cfg = getattr(config, "training", SimpleNamespace()),
                prior_cfg = getattr(config.model, "prior", SimpleNamespace())
              )
    
    # True held-out test set: never used during training.
    combined_test_loader = data_loader.get_test_loader()
    if combined_test_loader is None and eval_batches > 0:
        data_loader.refresh_test_cache(eval_batches)
        combined_test_loader = data_loader.get_test_loader()

    if combined_test_loader is not None and model_type != "phase_splitter":
        _run_evaluation_and_visualization(
            model=model,
            test_loader=combined_test_loader,
            normalizer=normalizer,
            writer=writer,
            device=device,
            log_path=log_path,
            config=config,
            phased_mode=True,
            conditioner=conditioner,
            secondary_model=None,
        )
    elif model_type == "phase_splitter":
        if combined_test_loader is None:
            print("Warning: no phased eval loader available; skipping phase_splitter quality evaluation.")
        else:
            try:
                split_metrics = model.evaluate_split_quality(
                    combined_test_loader,
                    device=device,
                    normalizer=normalizer,
                    plot_dir=log_path,
                    num_plot_samples=6,
                )
                print("Phase Splitter quality metrics:")
                print(
                    f"  PI-MSE={split_metrics['pi_mse']:.6f} | "
                    f"PI-MAE={split_metrics['pi_mae']:.6f} | "
                    f"Phase-Energy-MAE={split_metrics['phase_energy_mae']:.6f} | "
                    f"Agg-Conservation-MAE={split_metrics['aggregate_conservation_mae']:.6f} | "
                    f"PI-R2={split_metrics['pi_r2']:.6f} | "
                    f"N={int(split_metrics['num_samples'])}"
                )
                if "examples_plot_path" in split_metrics:
                    print(f"  Examples plot: {split_metrics['examples_plot_path']}")
                if "phase_mae_boxplot_path" in split_metrics:
                    print(f"  Phase MAE boxplot: {split_metrics['phase_mae_boxplot_path']}")
                split_metrics_path = os.path.join(log_path, "phase_splitter_eval.json")
                with open(split_metrics_path, "w") as f:
                    json.dump(split_metrics, f, indent=2)
                print(f"Saved phase splitter evaluation to {split_metrics_path}")
            except Exception as e:
                print(f"Warning: phase_splitter quality evaluation failed: {e}")

    if writer is not None:
        writer.close()
        print(f"Writer logs saved to: {log_path}")

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE/CVAE/SetGuideVAE/ART from YAML config")
    parser.add_argument("--config", type=str, default=os.path.join(parent_dir, "config", "config.yaml"))
    parser.add_argument("--skip-validation", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--load_path_model", type=str, default=None, help="Path to a saved model to load and visualize without training")
    parser.add_argument("--fit_user_profiler_only", action="store_true", default=False, help="Only fit the user profiler on the data without training a model (implies --skip-validation)")
    args = parser.parse_args()

    config = _load_train_config(args.config)
    config.skip_validation = bool(args.skip_validation)
    config.verbose = not bool(args.quiet)

    train(config, args.fit_user_profiler_only)


