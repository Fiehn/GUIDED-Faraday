import os
import sys
import argparse
import shutil
import yaml
import copy
import json
import faulthandler
from types import SimpleNamespace

faulthandler.enable()  # dumps C-level traceback on SIGSEGV / fatal errors
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model.base.encoder_decoder_model import *
from data.conditioner import *
from model.balanced_phase.cvae import *
from model.three_single_phased.set import *
from model.balanced_phase.vae import *
from data.dataloader_phased import *
from data.normalizer import *
from model.balanced_phase.hierarchical.aggregation import *
from model.balanced_phase.hierarchical.config import *
from src.visualization.visualizer import Visualizer
from src.visualization.writer import Writer
from src.visualization.plot_styling import PlotStyle
from model.base.factory import _load_train_config, _save_training_config_snapshot, create_model_from_config

def _run_visualization(model, test_loader, normalizer, conditioner, writer, device, log_path, config, secondary_model=None):
    """Run post-training visualizations. Metric evaluation is exclusively in compare_models (ADR-0011)."""
    if test_loader is None:
        print("Warning: test_loader is None; skipping visualization.")
        return
    try:
        if len(test_loader) == 0:
            print("Warning: test_loader has no batches; skipping visualization.")
            return
    except TypeError:
        pass

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

    _holdout_feeder_id = getattr(config.data, "holdout_feeder_id", None)
    if _holdout_feeder_id is not None and not getattr(config.data, "topology_path", None):
        raise ValueError(
            f"data.holdout_feeder_id={_holdout_feeder_id!r} is set but "
            "data.topology_path is missing or null. "
            "Provide the topology JSON path in data.topology_path, "
            "or set data.holdout_feeder_id to null to disable feeder holdout."
        )

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

    if config.window.aggregated and config.window.input_len != 24:
        print(f"Warning: window.aggregated is True but window.input_len is {config.window.input_len}. For hourly aggregation, consider setting input_len to 24.")
        config.window.input_len = 24
        config.window.stride = 24

    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)

    writer = Writer(log_dir=os.path.join(log_path, 'logs'))

    # === Data loader ===
    configured_files = list(config.data_files) if len(config.data_files) > 0 else []
    if len(configured_files) > 0:
        data_paths = []
        for f in configured_files:
            p = f if os.path.isabs(f) else os.path.join(data_dir, f)
            data_paths.append(p)
        # configured paths may not exist yet — filter with os.path.exists
        parquet_paths = sorted([p for p in data_paths if os.path.exists(p)])
    else:
        # os.scandir yields DirEntry objects whose .is_file() costs no extra syscall
        # on Windows (the file type is included in the directory listing result).
        # This avoids both the overhead of os.walk's recursive bookkeeping and the
        # redundant os.path.exists() calls that would follow — critical when data_dir
        # contains tens of thousands of files on a slow drive.
        data_paths = []
        def _scan_parquet(directory):
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        _scan_parquet(entry.path)
                    elif entry.is_file() and entry.name.endswith(".parquet") and not entry.name.startswith("test"):
                        data_paths.append(entry.path)
        _scan_parquet(data_dir)
        # Files discovered by scandir already exist — no os.path.exists needed.
        parquet_paths = sorted(data_paths)
    if len(parquet_paths) == 0:
        raise ValueError(f"No data files (.parquet) found in: {data_dir}")
    
    data_cfg = getattr(config, "data", SimpleNamespace())
    holdout_feeder_id = getattr(data_cfg, "holdout_feeder_id", None)
    topology_path = getattr(data_cfg, "topology_path", None)
    _hm = getattr(data_cfg, "holdout_meter_ids", None)
    holdout_meter_ids: list[int] | None = [int(m) for m in _hm] if _hm else None

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
    
    normalizer_cfg = getattr(config, "normalization", SimpleNamespace())
    if normalizer_cfg.method is None or (isinstance(normalizer_cfg.method, str) and normalizer_cfg.method.strip() == "") or str(normalizer_cfg.method).lower() == "none":
        normalizer = None
    else:
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
            max_value=float(getattr(data_cfg, "max_value", 200)),
            max_hour_value=float(getattr(data_cfg, "max_hour_value", 20))
        )
        normalizer.save(os.path.join(log_path, "normalizer.pt"))
    
    if normalizer is not None:
        # Attach the normalizer to the model so it can be automatically saved/loaded
        model.normalizer = copy.deepcopy(normalizer)

    eval_cache_device = device if eval_cache_on_gpu and str(device) != "cpu" else None

    in_memory = bool(getattr(data_cfg, "in_memory", False))

    if in_memory and phased_mode:
        _phased_feature_cols = ["active_power_p14_l1", "active_power_p14_l2", "active_power_p14_l3"]
        print(
            f"Using InMemoryPhasedDataloaders — loading {len(parquet_paths):,} files into RAM. "
            f"If the process is killed silently here, the system ran out of memory.",
            flush=True,
        )
        try:
            data_loader = InMemoryPhasedDataloaders(
                sm_files=parquet_paths,
                feature_cols=_phased_feature_cols,
                window_len=int(config.window.input_len),
                stride=int(config.window.stride),
                batch_size=int(data_cfg.batch_size),
                aggregation=bool(data_cfg.window.aggregated),
                base_frequency="1h" if data_cfg.window.aggregated else "15min",
                aggregation_divisor=aggregation_divisor,
                split_ratios={
                    "train_ratio": float(getattr(data_cfg.splits, "train_ratio", 1.0)),
                    "val_ratio":   float(getattr(data_cfg.splits, "val_ratio",   0.0)),
                    "test_ratio":  float(getattr(data_cfg.splits, "test_ratio",  0.0)),
                },
                split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
                max_hour_value=float(v) if (v := getattr(data_cfg, "max_hour_value", None)) is not None else None,
                max_value=float(v) if (v := getattr(data_cfg, "max_value", None)) is not None else None,
                nan_as_zero=nan_as_zero,
                normalizer=normalizer,
                output_aggregate_stream=output_aggregate_stream,
                pin_memory=(str(device) != "cpu"),
                eval_cache_device=eval_cache_device,
                holdout_feeder_id=holdout_feeder_id,
                topology_path=topology_path,
                holdout_meter_ids=holdout_meter_ids,
                num_workers=workers,
                preprocessed_cache_dir=getattr(data_cfg, "preprocessed_cache_dir", None) or None,
            )
        except MemoryError as _e:
            print(
                f"\n[train] FATAL: ran out of memory while building InMemoryPhasedDataloaders.\n"
                f"  Reduce num_workers, use in_memory: false, or add more RAM.\n"
                f"  Details: {_e}",
                flush=True,
            )
            raise SystemExit(1) from _e
        except Exception as _e:
            import traceback
            print(
                f"\n[train] FATAL: InMemoryPhasedDataloaders construction failed:\n"
                f"  {type(_e).__name__}: {_e}\n",
                flush=True,
            )
            traceback.print_exc()
            raise SystemExit(1) from _e
    elif in_memory:
        print("Using InMemoryCSVDataloaders. Note: this may lead to high memory usage.")
        data_loader = InMemoryCSVDataloaders(
            sm_files=parquet_paths,
            window_len=int(config.window.input_len),
            stride=int(config.window.stride),
            batch_size=int(data_cfg.batch_size),
            split_ratios={
                "train_ratio": float(getattr(data_cfg.splits, "train_ratio", 1.0)),
                "val_ratio":   float(getattr(data_cfg.splits, "val_ratio",   0.0)),
                "test_ratio":  float(getattr(data_cfg.splits, "test_ratio",  0.0)),
            },
            split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
            max_hour_value=float(v) if (v := getattr(data_cfg, "max_hour_value", None)) is not None else None,
            max_value=float(v) if (v := getattr(data_cfg, "max_value", None)) is not None else None,
            nan_as_zero=nan_as_zero,
            normalizer=normalizer,
            output_aggregate_stream=output_aggregate_stream,
            meter_col=data_cfg.columns.meter_id,
            value_cols=list(data_cfg.columns.value_cols),
            pin_memory=(str(device) != "cpu"),
            eval_cache_device=eval_cache_device,
            max_passes=int(v) if (v := getattr(data_cfg, "max_passes", None)) is not None else None,
            holdout_feeder_id=holdout_feeder_id,
            topology_path=topology_path,
        )
    else:
        print(f"Using PhasedReplayDataloaders with replay_capacity={replay_capacity}, min_buffer_size={min_buffer_size}, workers={workers}, eval_batches={eval_batches}, eval_cache_device={eval_cache_device}, cache_scan_batches={cache_scan_batches}, skip_corrupt_files={skip_corrupt_files}, max_skip_warnings={max_skip_warnings}, phased_mode={phased_mode}")
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
            max_value=float(v) if (v := getattr(data_cfg, "max_value", None)) is not None else None,
            max_hour_value=float(v) if (v := getattr(data_cfg, "max_hour_value", None)) is not None else None,
            holdout_feeder_id=holdout_feeder_id,
            topology_path=topology_path,
            holdout_meter_ids=holdout_meter_ids,
        )

    if conditioner is not None:
        if conditioner.condition_items.get("user_profile") is not None:
            conditioner.condition_items.get("user_profile", None).fit_on_files(parquet_paths, 
                                                                               phased=phased_mode, 
                                                                               feature_cols=data_cfg.columns.value_cols, 
                                                                               hourly_aggregation=data_cfg.window.aggregated, 
                                                                               aggregation_divisor=aggregation_divisor,
                                                                               window_spec=window
                                                                               )
            conditioner.condition_items.get("user_profile", None).save(config.conditioning.user_profile.cache_path)
            if train_only_user_profile:
                return

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
        _run_visualization(
            model=model,
            test_loader=combined_test_loader,
            normalizer=normalizer,
            writer=writer,
            device=device,
            log_path=log_path,
            config=config,
            conditioner=conditioner,
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

    # Holdout eval pass removed — evaluation is exclusively in compare_models (ADR-0011).

    if writer is not None:
        writer.close()
        print(f"Writer logs saved to: {log_path}")

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE/CVAE/SetGuideVAE/ART from YAML config")
    parser.add_argument("--config", type=str, default=os.path.join(parent_dir, "../config", "config.yaml"))
    parser.add_argument("--skip-validation", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--load_path_model", type=str, default=None, help="Path to a saved model to load and visualize without training")
    parser.add_argument("--fit_user_profiler_only", action="store_true", default=False, help="Only fit the user profiler on the data without training a model (implies --skip-validation)")
    args = parser.parse_args()

    config = _load_train_config(args.config)
    config.skip_validation = bool(args.skip_validation)
    config.verbose = not bool(args.quiet)

    train(config, args.fit_user_profiler_only)


