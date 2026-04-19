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
from src.model.art import AutoRegressiveTransformerVAE
from src.model.diffusion import DiffusionModel, NoiseScheduler, ConditionalUNet1D
from src.model.phase_splitter import HierarchicalSplitter


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_train_config(config_path: str = None):
    """Load YAML config and add aliases expected by train()."""
    if config_path is None:
        config_path = os.path.join(parent_dir, "config", "config.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _to_namespace(raw)
    cfg._config_path = os.path.abspath(config_path)

    # Aliases expected by existing train() implementation
    if not hasattr(cfg, "data"):
        raise ValueError("Config must contain a top-level 'data' section")
    cfg.window = cfg.data.window
    cfg.columns = cfg.data.columns
    cfg.splits = cfg.data.splits
    cfg.data_dir = cfg.data.data_dir
    cfg.data_files = getattr(cfg.data, "data_files", [])
    cfg.model_type = str(cfg.model.type).lower() if hasattr(cfg, "model") and hasattr(cfg.model, "type") else "vae"

    # Backward-compatible alias: train() expects config.normalizer.method
    if hasattr(cfg, "normalization"):
        cfg.normalizer = cfg.normalization
    elif not hasattr(cfg, "normalizer"):
        cfg.normalizer = SimpleNamespace(method="log1p")

    # Runtime toggles used in train()
    if not hasattr(cfg, "skip_validation"):
        cfg.skip_validation = False
    if not hasattr(cfg, "verbose"):
        cfg.verbose = True

    return cfg

def _save_training_config_snapshot(config, log_path: str):
    """Save the YAML config used for training into the run report directory."""
    os.makedirs(log_path, exist_ok=True)
    src_path = getattr(config, "_config_path", None)
    dst_path = os.path.join(log_path, "training_config_used.yaml")

    if src_path and os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print("Warning: could not resolve original config file path; skipping config snapshot save.")


def _run_evaluation_and_visualization(model, test_loader, normalizer, conditioner, writer, device, log_path, config, phased_mode: bool):
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
                            conditioner=conditioner)
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

    vamp_initialized = False

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}
    
    # === Hierarchical latent space setup ===
    if config.hierarchy.enabled:
        hierarchy_cfg = getattr(config, "hierarchy", SimpleNamespace())
        levels = []
        for level_type in config.hierarchy.levels:
            input_dim = config.model.latent_dim if len(levels) == 0 else levels[-1].latent_dim
            levels.append(LevelConfig(
                name=f"Level {len(levels)+1}",
                strategy=level_type,
                input_dim=input_dim,
                latent_dim=getattr(hierarchy_cfg, "latent_dims", SimpleNamespace())[len(levels)],
                top_down=getattr(hierarchy_cfg,"top_down",SimpleNamespace)[len(levels)],
                bottom_up=getattr(hierarchy_cfg,"bottom_up",SimpleNamespace)[len(levels)],
                loss_weight=getattr(hierarchy_cfg,"loss_weights", SimpleNamespace())[len(levels)],
            ))
        hierarchy = HierarchicalConfig(
            levels=levels,
            aggregation=getattr(hierarchy_cfg, "aggregation", SimpleNamespace()),
            bidirectional=getattr(hierarchy_cfg, "bidirectional", False),
            residual_connections=getattr(hierarchy_cfg, "residual_connections", False),
            kl_annealing=getattr(hierarchy_cfg, "kl_annealing", True),
            kl_warmup_steps=getattr(hierarchy_cfg, "kl_warmup_steps", 1000),
            free_bits=getattr(hierarchy_cfg, "free_bits", 0.1),
        )
    else:
        hierarchy = None

    # === Model Setup ===
    model_type = str(config.model.type).lower()
    conditioner = None
    encoder_cfg = getattr(config.model, "encoder", SimpleNamespace())
    decoder_cfg = getattr(config.model, "decoder", SimpleNamespace())
    encoder_hidden_dims = getattr(encoder_cfg, "hidden_dims", None)
    decoder_hidden_dims = getattr(decoder_cfg, "hidden_dims", None)
    encoder_dropout = getattr(encoder_cfg, "dropout", None)
    decoder_dropout = getattr(decoder_cfg, "dropout", None)

    if (model_type == "phase_splitter" or model_type == "set_guide_vae") and not phased_mode:
        print("You have indicated a phased only model type is currently supported only with data.mode='phased_parquet'. Forcing phased mode for this run.")
        data_mode = "phased_parquet"
        phased_mode = True

    if model_type == "diffusion" and hierarchy is not None:
        print("DiffusionModel currently does not support hierarchy.enabled=true. Ignoring hierarchical config.")

    # === Conditioners ===
    if model_type == "vae" or len(config.conditioning.tags) == 0:
        print("No conditioning will be used for this run.")
    else:
        conditioner = Conditioner()
        for condition_tag in config.conditioning.tags:
            if condition_tag == "day_of_week":
                conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            elif condition_tag == "user_profile":
                if config.conditioning.user_profile.online:
                    decay = config.conditioning.online_user_profile.decay
                else:
                    decay = 1.0

                profiler = UserProfiler(n_topics=config.conditioning.user_profile.n_topics,
                    num_clusters=config.conditioning.user_profile.num_clusters, 
                    path=config.conditioning.user_profile.cache_path, 
                    verbose=config.conditioning.user_profile.verbose, 
                    decay=decay, 
                    )
                
                conditioner.add_condition_item("user_profile", profiler)
                    
            elif condition_tag == "holiday":
                conditioner.add_condition_item("holiday", HolidayConditioner(holiday_path='data/holidays/denmark_holidays.json'))
            elif condition_tag == "month_of_year":
                conditioner.add_condition_item("month_of_year", MonthOfYearConditioner())
            elif condition_tag == "nr_phases_active":
                conditioner.add_condition_item("nr_phases_active", ActivePhaseCountConditioner())
            elif condition_tag == "heatpump_phase":
                conditioner.add_condition_item("heatpump_phase", HeatpumpPhaseConditioner())
            else:
                print(f"Warning: Unknown condition tag '{condition_tag}' in config. Skipping this conditioner.")

    # === Get number of features for model initialization ===
    set_cfg = getattr(config.model, "set", SimpleNamespace(num_features=None))
    csv_num_features = len(config.columns.value_cols)
    if phased_mode:
        phased_num_features = getattr(set_cfg, "num_features", None)
        set_num_features = int(phased_num_features) if phased_num_features is not None else 3
    else:
        set_num_features = int(csv_num_features)
    input_dim = config.window.input_len * int(set_num_features)

    # === Model Initialization ===
    model_cfg = getattr(config, "model", SimpleNamespace())
    if model_type == "cvae":
        model = CVAE(input_dim=input_dim, latent_dim=model_cfg.latent_dim, 
                    conditioner=conditioner, dictionary_size=model_cfg.prior.dictionary_size, 
                    vamp_num_components=model_cfg.prior.vamp_num_components,
                    encoder_type=model_cfg.encoder.type,
                    decoder_type=model_cfg.decoder.type,
                    prior_type=model_cfg.prior.type,
                    hierarchy_config=hierarchy,
                    encoder_hidden_dims=encoder_hidden_dims,
                    decoder_hidden_dims=decoder_hidden_dims,
                    encoder_dropout=encoder_dropout,
                    decoder_dropout=decoder_dropout,
                    phased_mode=phased_mode,
                    sequence_length=config.window.input_len,
                    )
        print(f"Initialized CVAE with input_dim={input_dim}, latent_dim={model_cfg.latent_dim}, encoder_type={model_cfg.encoder.type}, decoder_type={model_cfg.decoder.type}, prior_type={model_cfg.prior.type}, conditioner_tags={config.conditioning.tags}")
    elif model_type == "set_guide_vae":
        set_cfg = config.model.set
        if phased_mode:
            set_num_features = set_cfg.num_features if set_cfg.num_features is not None else int(set_num_features)
        else:
            set_num_features = len(config.columns.value_cols)
        model = SetGuideVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            dictionary_size=model_cfg.prior.dictionary_size,
            vamp_num_components=model_cfg.prior.vamp_num_components,
            encoder_type=model_cfg.encoder.type,
            decoder_type=model_cfg.decoder.type,
            prior_type=model_cfg.prior.type,
            hierarchy_config=hierarchy,
            sequence_length=config.window.input_len,
            num_features=set_num_features,
            set_hidden_dim=set_cfg.hidden_dim,
            set_num_heads=set_cfg.num_heads,
            set_num_inds=set_cfg.num_inds,
            set_num_blocks=set_cfg.num_blocks,
            is_isab=set_cfg.is_isab,
            use_abl_pool=bool(getattr(set_cfg, "use_abl_pool", False)),
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
        print(f"Initialized SetGuideVAE with input_dim={input_dim}, latent_dim={model_cfg.latent_dim}, encoder_type={model_cfg.encoder.type}, decoder_type={model_cfg.decoder.type}, prior_type={model_cfg.prior.type}, conditioner_tags={config.conditioning.tags}")
    elif model_type == "art":
        art_cfg = getattr(model_cfg, "art", SimpleNamespace())
        model = AutoRegressiveTransformerVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            dictionary_size=model_cfg.prior.dictionary_size,
            vamp_num_components=model_cfg.prior.vamp_num_components,
            encoder_type=model_cfg.encoder.type,
            decoder_type=model_cfg.decoder.type,
            prior_type=model_cfg.prior.type,
            hierarchy_config=hierarchy,
            bin_size=art_cfg.num_bins,
            nhead=art_cfg.transformer_num_heads,
            num_layers=art_cfg.transformer_num_layers,
            d_model=art_cfg.d_model,
            sequence_length=config.window.input_len,
            num_features=set_num_features,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
    elif model_type == "diffusion":
        diff_cfg = getattr(config.model, "diffusion", SimpleNamespace())
        scheduler = NoiseScheduler(
            num_timesteps=int(getattr(diff_cfg, "num_timesteps", 1000)),
            beta_start=float(getattr(diff_cfg, "beta_start", 1e-4)),
            beta_end=float(getattr(diff_cfg, "beta_end", 0.02)),
            schedule_type=str(getattr(diff_cfg, "schedule_type", "linear")),
            cosine_s=float(getattr(diff_cfg, "cosine_s", 0.008)),
            max_beta=float(getattr(diff_cfg, "max_beta", 0.999)),
            device=device,
        )
        denoiser = ConditionalUNet1D(
            channels=int(set_num_features),
            condition_dim=int(conditioner.get_condition_dim()),
            time_dim=int(getattr(diff_cfg, "time_dim", 128)),
            base_channels=int(getattr(diff_cfg, "base_channels", 128)),
            num_blocks=int(getattr(diff_cfg, "num_blocks", 4)),
            self_attention_heads=int(getattr(diff_cfg, "self_attention_heads", 4)),
            use_condition_cross_attention=bool(getattr(diff_cfg, "use_condition_cross_attention", False)),
            condition_attention_heads=int(getattr(diff_cfg, "condition_cross_attention_heads", 4)),
            condition_tokens=int(getattr(diff_cfg, "condition_tokens", 4)),
        )
        model = DiffusionModel(
            noise_scheduler=scheduler,
            model=denoiser,
            conditioner=conditioner,
            sequence_length=config.window.input_len,
            num_features=int(set_num_features),
            condition_dropout_prob=float(getattr(diff_cfg, "condition_dropout_prob", 0.0)),
            reconstruction_noise_fraction=float(getattr(diff_cfg, "reconstruction_noise_fraction", 0.3)),
            deterministic_reconstruction=bool(getattr(diff_cfg, "deterministic_reconstruction", True)),
            phased_mode=phased_mode,
        )
        print(
            f"Initialized DiffusionModel with sequence_length={config.window.input_len}, "
            f"num_features={int(set_num_features)}, condition_dim={conditioner.get_condition_dim()}, "
            f"schedule={scheduler.schedule_type}, cross_attention={denoiser.use_condition_cross_attention}"
        )
    elif model_type == "phase_splitter":
        phase_splitter_cfg = getattr(config.model, "phase_splitter", SimpleNamespace())
        model = HierarchicalSplitter(
            conditioner=conditioner,
            sequence_length=config.window.input_len,
            condition_dim=int(conditioner.get_condition_dim()),
            base_channels=int(getattr(phase_splitter_cfg, "base_channels", 64)),
            phased_mode=phased_mode,
        )
        print(
            f"Initialized HierarchicalSplitter with sequence_length={config.window.input_len}, "
            f"condition_dim={int(conditioner.get_condition_dim())}, conditioner_tags={config.conditioning.tags}"
        )
    elif model_type == "vae":
        model = VAE(input_dim=input_dim, latent_dim=model_cfg.latent_dim,
                    encoder_type=model_cfg.encoder.type,
                    decoder_type=model_cfg.decoder.type,
                    prior_type=model_cfg.prior.type,
                    dictionary_size=model_cfg.prior.dictionary_size,
                    vamp_num_components=model_cfg.prior.vamp_num_components,
                    hierarchy_config=hierarchy,
                    encoder_hidden_dims=encoder_hidden_dims,
                    decoder_hidden_dims=decoder_hidden_dims,
                    encoder_dropout=encoder_dropout,
                    decoder_dropout=decoder_dropout,
                    phased_mode=phased_mode,
                    sequence_length=config.window.input_len,
                    )
        print(f"Initialized VAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}")
    else:
        raise ValueError(f"Unknown model type '{config.model.type}' specified in config.")

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
    
    norm_method = str(getattr(config.normalizer, "method", "log1p")).lower() if not phased_mode else "log1p_phase"
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
    )
    normalizer.save(os.path.join(log_path, "normalizer.pt"))

    if model_type == "art" and normalizer is not None:
        # We need to denormalize for tokenization in ART, so we must ensure the model has access to the normalizer.
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
    
    combined_test_loader = data_loader.get_eval_loader()
    if combined_test_loader is None and eval_batches > 0:
        data_loader.refresh_eval_cache(eval_batches)
        combined_test_loader = data_loader.get_eval_loader()

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

    if phased_mode:

        if norm_method == "log1p_phase":
            normalizer = Normalizer(method="log1p_phase")
            agg_tag = "hourly" if bool(config.window.aggregated) else "raw"
            stats_name = (
                f"normalization_stats_{os.path.basename(os.path.normpath(data_dir))}_"
                f"{agg_tag}_{aggregation_energy_unit}.json"
            )
            stats_path = os.path.join(config.logging.log_dir, stats_name)
            normalizer.fit(
                X=None,
                sm_files=parquet_paths,
                stats_path=stats_path,
                hourly_aggregation=bool(config.window.aggregated),
                aggregation_divisor=aggregation_divisor,
            )
            print("Using phased normalizer: log1p_phase")
        else:
            print(f"Warning: phased_parquet mode currently expects normalization.method='log1p_phase'. Got '{norm_method}'. Continuing without normalization.")

   
        # Use cached eval loader (kept separate from training cache)
    
    
    # === Fit Normalizer and User Profiler (if applicable) on all data files ===
    normalizer = Normalizer(method=config.normalization.method)
    fit_normalizer_on_files(
        normalizer,
        data_dir=data_dir,
        filenames=data_files,
        feature_cols=config.data.columns.value_cols,
        meter_col=config.data.columns.meter_id,
        timestamp_dst_col=config.data.columns.timestamp_dst,
        train_frac=config.data.splits.train_ratio,
        seed=config.data.splits.random_seed,
        nan_as_zero=nan_as_zero,
        parquet_only=parquet_only,
    )
    
    if hasattr(model, "conditioner") and model.conditioner is not None:
        user_profiler = model.conditioner.condition_items.get("user_profile")
        if user_profiler is not None:
            fitted = fit_userprofiler_on_files(
                user_profiler=user_profiler,
                data_dir=data_dir,
                filenames=data_files,
                feature_cols=config.data.columns.value_cols,
                window=window,
                meter_col=config.data.columns.meter_id,
                timestamp_dst_col=config.data.columns.timestamp_dst,
                timestamp_utc_col=config.data.columns.timestamp_utc,
                batch_size=config.data.dataloader.batch_size,
                num_workers=config.data.dataloader.num_workers,
                seed=config.data.splits.random_seed,
                train_frac=config.data.splits.train_ratio,
                normalizer=normalizer,
                nan_as_zero=nan_as_zero,
                parquet_only=parquet_only,
                device=device,
            )
            if train_only_user_profile:
                # This is done because the user profiler is trained on the cpu..
                # This means it needs a different node than the rest of the model, which is on the gpu.
                # A different slurm job will be used to train just the user profiler, and then the rest of the model will be trained with the user profiler frozen.
                print("train_only_user_profile is True, so exiting after fitting user profiler.")
                return model
            if not fitted:
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


