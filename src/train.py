import os
import sys
import argparse
import shutil
import yaml
import time
import math
import copy
import json
from types import SimpleNamespace
from xml.parsers.expat import model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model.encoder_decoder_model import *
from data.conditioner import *
from src.model.cvae import *
from src.model.set import *
from src.model.vae import *
from data.dataloader import *
from data.dataloader_phased import WindowSpec as PhaseWindowSpec, PhasedReplayDataloaders
from data.normalizer import *
from src.test.run_validation import run_validation
from src.model.hierarchical.aggregation import *
from src.model.hierarchical.config import *
from src.visualization.visualizer import Visualizer
from src.visualization.writer import Writer
from evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
from src.visualization.plot_styling import PlotStyle
from src.model.model_utils import EarlyStopping
from src.model.art import AutoRegressiveTransformerVAE
from src.model.diffusion import DiffusionModel, NoiseScheduler, ConditionalUNet1D
from src.model.phase_splitter import HierarchicalSplitter
import gc


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
    cfg.dataloader = cfg.data.dataloader
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

def _init_vamp_prior_from_batch(model, batch, device, noise_scale: float = 0.01) -> bool:
    if not isinstance(getattr(model, "prior", None), VampPrior):
        return False

    if batch is None:
        return False

    model.eval()
    with torch.no_grad():
        if isinstance(model, SetGuideVAE):
            x, meter_id, timestamps = model._unpack_batch(batch)
            x_phase_first = model._to_phase_first(x).float()
            x_for_conditions = model._build_condition_input(x_phase_first)
            c = model.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
            c = model.move_to_device(c, device=device)
            x_phase_first = model.move_to_device(x_phase_first, device=device)
            set_repr = model._set_encode(x_phase_first)
            enc_in = torch.cat([set_repr, c], dim=-1)
        elif isinstance(model, CVAE):
            if len(batch) == 4:
                x, second, third, fourth = batch
                if torch.is_tensor(second) and second.dim() >= 2:
                    c = second
                    meter_id = third
                    timestamps = fourth
                else:
                    c = None
                    meter_id = second
                    timestamps = third
                if c is None or c.numel() == 0 or c.shape[-1] != model.condition_dim:
                    c = model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            elif len(batch) == 3:
                x, second, third = batch
                if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
                    meter_id = second[:, 0, 0].long()
                    timestamps = second[:, 1, 0].long()
                else:
                    meter_id = second
                    timestamps = third
                c = model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            elif len(batch) == 2:
                x, meter_id = batch
                c = model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=None)
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                c = model.conditioner.get_conditions(x)

            c = model.move_to_device(c, device=device)
            x_flat = model._prep_inputs(x).float()
            x_flat = model.move_to_device(x_flat, device=device)
            enc_in = torch.cat([x_flat, c], dim=-1)
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_flat = x.reshape(x.shape[0], -1) if x.dim() == 3 else x
            enc_in = model.move_to_device(x_flat.float(), device=device)

        if enc_in.numel() == 0:
            return False

        K = int(model.prior.num_components)
        n = int(enc_in.shape[0])
        if n >= K:
            idx = torch.randperm(n, device=enc_in.device)[:K]
        else:
            idx = torch.randint(0, n, (K,), device=enc_in.device)
        init = enc_in[idx]
        if noise_scale and noise_scale > 0:
            init = init + (float(noise_scale) * torch.randn_like(init))

        model.prior.pseudo_inputs.data.copy_(init)

    return True


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

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to be 999

    vamp_initialized = False

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phase", "parquet", "phased_parquet"}
    validation_model_type = str(getattr(config.model, "type", "vae")).lower()

    if phased_mode and not config.skip_validation:
        print("Phased parquet mode detected. Skipping CSV-based pre-validation (run_validation).")
    if (not config.skip_validation) and (not phased_mode):
        if validation_model_type in {"vae", "cvae"}:
            validation_passed = run_validation(
                model_type=validation_model_type,
                use_test_data=True,  # use small test data for fast validation
                verbose=config.verbose
            )
            if not validation_passed:
                print("\nValidation failed! Please fix issues before training.")
                print("To skip validation, set skip_validation=True in the config (not recommended)")
                return
            print("\nValidation passed! Proceeding to training\n")
        else:
            print(
                "CSV mode pre-validation currently supports only VAE/CVAE. "
                f"Skipping pre-validation for model_type='{validation_model_type}'."
            )
    
    # === Hierarchical latent space setup ===
    if config.hierarchy.enabled:
        levels = []
        for level_type in config.hierarchy.levels:
            input_dim = config.model.latent_dim if len(levels) == 0 else levels[-1].latent_dim
            levels.append(LevelConfig(
                name=f"Level {len(levels)+1}",
                strategy=level_type,
                input_dim=input_dim,
                latent_dim=config.hierarchy.latent_dims[len(levels)],
                top_down=config.hierarchy.top_down[len(levels)],
                bottom_up=config.hierarchy.bottom_up[len(levels)],
                loss_weight=config.hierarchy.loss_weights[len(levels)],
            ))
        hierarchy = HierarchicalConfig(
            levels=levels,
            aggregation=config.hierarchy.aggregation,
            bidirectional=config.hierarchy.bidirectional,
            residual_connections=config.hierarchy.residual_connections,
            kl_annealing=config.hierarchy.kl_annealing,
            kl_warmup_steps=config.hierarchy.kl_warmup_steps,
            free_bits=config.hierarchy.free_bits,
        )
    else:
        hierarchy = None

    # === Model Setup ===
    model_type = str(config.model.type).lower()
    conditioner = None

    if model_type == "phase_splitter" and not phased_mode:
        raise ValueError("phase_splitter is currently supported only with data.mode='phased_parquet'.")

    if model_type == "diffusion" and hierarchy is not None:
        raise ValueError("DiffusionModel currently does not support hierarchy.enabled=true.")

    if model_type in {"cvae", "set_guide_vae", "art", "diffusion", "phase_splitter"}:
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
        
        set_cfg = getattr(config.model, "set", SimpleNamespace(num_features=None))
        csv_num_features = len(config.columns.value_cols)
        if phased_mode and model_type in {"set_guide_vae", "art", "diffusion", "phase_splitter"}:
            phased_num_features = getattr(set_cfg, "num_features", None)
            set_num_features = int(phased_num_features) if phased_num_features is not None else 3
        else:
            set_num_features = int(csv_num_features)
        input_dim = config.window.input_len * int(set_num_features)
        if model_type == "cvae":
            model = CVAE(input_dim=input_dim, latent_dim=config.model.latent_dim, 
                        conditioner=conditioner, dictionary_size=config.model.prior.dictionary_size, 
                        vamp_num_components=config.model.prior.vamp_num_components,
                        encoder_type=config.model.encoder.type,
                        decoder_type=config.model.decoder.type,
                        prior_type=config.model.prior.type,
                        hierarchy_config=hierarchy,
                        )
            print(f"Initialized CVAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}, conditioner_tags={config.conditioning.tags}")
        elif model_type == "set_guide_vae":
            set_cfg = config.model.set
            if phased_mode:
                set_num_features = set_cfg.num_features if set_cfg.num_features is not None else int(set_num_features)
            else:
                set_num_features = len(config.columns.value_cols)
            model = SetGuideVAE(
                input_dim=input_dim,
                latent_dim=config.model.latent_dim,
                conditioner=conditioner,
                dictionary_size=config.model.prior.dictionary_size,
                vamp_num_components=config.model.prior.vamp_num_components,
                encoder_type=config.model.encoder.type,
                decoder_type=config.model.decoder.type,
                prior_type=config.model.prior.type,
                hierarchy_config=hierarchy,
                sequence_length=config.window.input_len,
                num_features=set_num_features,
                set_hidden_dim=set_cfg.hidden_dim,
                set_num_heads=set_cfg.num_heads,
                set_num_inds=set_cfg.num_inds,
                set_num_blocks=set_cfg.num_blocks,
                is_isab=set_cfg.is_isab,
                use_abl_pool=bool(getattr(set_cfg, "use_abl_pool", False)),
            )
            print(f"Initialized SetGuideVAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}, conditioner_tags={config.conditioning.tags}")
        elif model_type == "art":
            model = AutoRegressiveTransformerVAE(
                input_dim=input_dim,
                latent_dim=config.model.latent_dim,
                conditioner=conditioner,
                dictionary_size=config.model.prior.dictionary_size,
                vamp_num_components=config.model.prior.vamp_num_components,
                encoder_type=config.model.encoder.type,
                decoder_type=config.model.decoder.type,
                prior_type=config.model.prior.type,
                hierarchy_config=hierarchy,
                bin_size=config.model.art.num_bins,
                nhead=config.model.art.transformer_num_heads,
                num_layers=config.model.art.transformer_num_layers,
                sequence_length=config.window.input_len,
                num_features=set_num_features,
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
                condition_attention_heads=int(
                    getattr(
                        diff_cfg,
                        "condition_cross_attention_heads",
                        getattr(diff_cfg, "condition_attention_heads", 4),
                    )
                ),
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
            )
            print(
                f"Initialized DiffusionModel with sequence_length={config.window.input_len}, "
                f"num_features={int(set_num_features)}, condition_dim={conditioner.get_condition_dim()}, "
                f"schedule={scheduler.schedule_type}, cross_attention={denoiser.use_condition_cross_attention}"
            )
        elif model_type == "phase_splitter":
            model = HierarchicalSplitter(
                conditioner=conditioner,
                sequence_length=config.window.input_len,
                condition_dim=int(conditioner.get_condition_dim()),
            )
            print(
                f"Initialized HierarchicalSplitter with sequence_length={config.window.input_len}, "
                f"condition_dim={int(conditioner.get_condition_dim())}, conditioner_tags={config.conditioning.tags}"
            )
    elif model_type == "vae":
        input_dim = config.window.input_len * len(config.columns.value_cols)
        model = VAE(input_dim=input_dim, latent_dim=config.model.latent_dim,
                    encoder_type=config.model.encoder.type,
                    decoder_type=config.model.decoder.type,
                    prior_type=config.model.prior.type,
                    dictionary_size=config.model.prior.dictionary_size,
                    vamp_num_components=config.model.prior.vamp_num_components,
                    hierarchy_config=hierarchy,
                    )
        print(f"Initialized VAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}")
    else:
        raise ValueError(f"Unknown model type '{config.model.type}' specified in config.")

    model_tag = model.get_tag()
    _save_training_config_snapshot(config=config, log_path=os.path.join(config.logging.log_dir, model_tag))

    # === Load Data === 
    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)
    data_dir = config.data_dir

    # === Phased parquet path ===
    if phased_mode:
        if model_type not in {"set_guide_vae", "art", "diffusion", "phase_splitter"}:
            raise ValueError(
                "Phased parquet mode currently supports set_guide_vae, art, diffusion, and phase_splitter model types."
            )

        parquet_paths = []
        configured_files = list(config.data_files) if len(config.data_files) > 0 else []

        if len(configured_files) > 0:
            for f in configured_files:
                if os.path.isabs(f):
                    parquet_paths.append(f)
                else:
                    parquet_paths.append(os.path.join(data_dir, f))
        else:
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(".parquet"):
                        parquet_paths.append(os.path.join(root, f))

        parquet_paths = sorted([p for p in parquet_paths if os.path.exists(p)])
        if len(parquet_paths) == 0:
            raise ValueError(f"No parquet files found for phased mode in: {data_dir}")
        
        # TODO: REMOVE THIS:
        #parquet_paths = parquet_paths[:2000]  # temporary limit for testing

        print(f"Phased mode enabled. Found {len(parquet_paths)} parquet file(s).")

        phased_cfg = getattr(config.data, "phased", SimpleNamespace())
        aggregation_energy_unit = str(getattr(phased_cfg, "aggregation_energy_unit", "wh")).lower()
        if aggregation_energy_unit not in {"wh", "kwh"}:
            print(f"Warning: unknown data.phased.aggregation_energy_unit='{aggregation_energy_unit}'. Falling back to 'wh'.")
            aggregation_energy_unit = "wh"
        aggregation_divisor = 1000.0 if bool(config.window.aggregated) and aggregation_energy_unit == "kwh" else 1.0
        if bool(config.window.aggregated):
            print(f"Hourly aggregation unit: {aggregation_energy_unit} (divisor={aggregation_divisor:g})")

        replay_capacity = int(getattr(phased_cfg, "replay_buffer_capacity", 10000))
        min_buffer_size = int(getattr(phased_cfg, "min_buffer_size", 2000))
        phased_workers = int(getattr(phased_cfg, "num_workers", config.dataloader.num_workers))
        steps_per_epoch = int(getattr(phased_cfg, "steps_per_epoch", 100))
        eval_batches = int(getattr(phased_cfg, "eval_batches", 5))
        eval_every_n_epochs = int(getattr(phased_cfg, "eval_every_n_epochs", 1))
        eval_cache_refresh = bool(getattr(phased_cfg, "eval_cache_refresh", False))
        # Keep eval cache on CPU by default to avoid GPU OOM from preloading full eval tensors.
        eval_cache_on_gpu = bool(getattr(phased_cfg, "eval_cache_on_gpu", False))
        cache_scan_batches = int(getattr(phased_cfg, "cache_scan_batches", 0)) or None
        skip_corrupt_files = bool(getattr(phased_cfg, "skip_corrupt_files", True))
        max_skip_warnings = int(getattr(phased_cfg, "max_skip_warnings", 20))
        output_aggregate_stream = bool(getattr(phased_cfg, "output_aggregate_stream", False))
        if model_type == "phase_splitter" and not output_aggregate_stream:
            output_aggregate_stream = True
            print("For phase_splitter, forcing data.phased.output_aggregate_stream=true.")

        if output_aggregate_stream:
            print(
                "Phased aggregate stream enabled: train/eval loaders emit aggregate tensors "
                "alongside phased tensors. Current ART/Diffusion/SET-GUIDE paths will ignore "
                "the aggregate unless a splitter-specific model consumes it."
            )

        print("Phased mode will allocate train/eval from one shared replay cache each epoch.")
        if skip_corrupt_files:
            print("Phased mode will skip unreadable/corrupt parquet files.")
        else:
            print("Phased mode will fail fast on unreadable/corrupt parquet files.")

        normalizer = None
        norm_method = str(getattr(config.normalizer, "method", "log1p")).lower()
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
        if model_type == "art" and normalizer is not None:
            # We need to denormalize for tokenization in ART, so we must ensure the model has access to the normalizer.
            model.normalizer = copy.deepcopy(normalizer)
        
        if conditioner.condition_items.get("user_profile") is not None:
            conditioner.condition_items.get("user_profile", None).fit_on_files(parquet_paths)
        if config.window.aggregated and config.window.input_len != 24:
            print(f"Warning: window.aggregated is True but window.input_len is {config.window.input_len}. For hourly aggregation, consider setting input_len to 24.")
            config.window.input_len = 24
            config.window.stride = 24
        phase_window = PhaseWindowSpec(input_len=config.window.input_len, stride=config.window.stride)
        eval_cache_device = device if eval_cache_on_gpu and str(device) != "cpu" else None
        if eval_cache_device is None:
            print("Phased eval cache device: CPU (safer memory profile).")
        else:
            print(f"Phased eval cache device: {eval_cache_device}.")
        phased_replay = PhasedReplayDataloaders(
            sm_files=parquet_paths,
            replay_buffer_capacity=replay_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=phase_window,
            normalizer=normalizer,
            num_workers=phased_workers,
            hourly_aggregation=config.window.aggregated,
            aggregation_divisor=aggregation_divisor,
            batch_size=config.dataloader.batch_size,
            eval_cache_device=eval_cache_device,
            eval_cache_batches=eval_batches,
            cache_scan_batches=cache_scan_batches,
            pin_memory=(str(device) != "cpu"),
            output_aggregate_stream=output_aggregate_stream,
            skip_corrupt_files=skip_corrupt_files,
            max_skip_warnings=max_skip_warnings,
        )

        writer = Writer(log_dir=os.path.join(config.logging.log_dir, model_tag))
        optim = model.get_optimizer(lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

        checkpoint_cfg = getattr(config.training, "checkpointing", SimpleNamespace())
        save_best_model = bool(getattr(checkpoint_cfg, "save_best_model", True))
        save_best_after_fraction = float(getattr(checkpoint_cfg, "save_best_after_fraction", 0.25))
        save_best_after_fraction = max(0.0, min(1.0, save_best_after_fraction))
        save_best_after_epoch = max(1, int(math.ceil(config.training.num_epochs * save_best_after_fraction)))
        if save_best_model:
            print(
                f"Best-model checkpointing starts at epoch {save_best_after_epoch} "
                f"({save_best_after_fraction:.0%} of training)."
            )
        else:
            print("Best-model checkpointing is disabled by config.training.checkpointing.save_best_model.")

        use_early_stopping = bool(getattr(config.training.early_stopping, "enabled", False))
        stop_on_all_files_used = bool(getattr(config.training.early_stopping, "stop_on_all_files_used", True))
        min_epochs_before_full_data_stop = int(getattr(config.training.early_stopping, "min_epochs_before_full_data_stop", 1))
        early_stopper = None
        best_eval_loss = float("inf")
        best_eval_loss_for_saving = float("inf")
        if use_early_stopping:
            early_stopper = EarlyStopping(
                patience=int(getattr(config.training.early_stopping, "patience", 10)),
                min_delta=float(getattr(config.training.early_stopping, "min_delta", 0.001)),
                mode='min',
            )

        model = model.to(device)
        if hasattr(model, "prior") and getattr(model, "prior", None) is not None and hasattr(model.prior, "to"):
            model.prior.to(device)
        if model_type == "art" and hasattr(model, "normalizer") and model.normalizer is not None:
            model.normalizer = model.normalizer.to(device)

        if not vamp_initialized and hasattr(model, "prior") and isinstance(model.prior, VampPrior):
            init_from_data = bool(getattr(config.model.prior, "init_from_data", True))
            if init_from_data:
                init_loader, _ = phased_replay.get_epoch_loaders(
                    steps_per_epoch=1,
                    eval_batches=0,
                    refresh_eval=False,
                )
                init_batch = next(iter(init_loader)) if init_loader is not None else None
                noise_scale = float(getattr(config.model.prior, "init_noise_scale", 0.01))
                if _init_vamp_prior_from_batch(model, init_batch, device=device, noise_scale=noise_scale):
                    vamp_initialized = True
                    print("Initialized VampPrior pseudo-inputs from data batch.")

        print(f"Starting phased training for {config.training.num_epochs} epoch(s), {steps_per_epoch} step(s)/epoch")
        for epoch in range(config.training.num_epochs):
            epoch_start_t = time.perf_counter()
            should_eval_epoch = eval_batches > 0 and ((epoch + 1) % max(1, eval_every_n_epochs) == 0)

            cache_start_t = time.perf_counter()
            train_loader, epoch_eval_loader = phased_replay.get_epoch_loaders(
                steps_per_epoch=steps_per_epoch,
                eval_batches=eval_batches if should_eval_epoch else 0,
                refresh_eval=eval_cache_refresh and should_eval_epoch,
            )
            cache_time_s = time.perf_counter() - cache_start_t
            if train_loader is None:
                raise RuntimeError("No phased training batches could be collected from replay stream.")

            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n = 0
            eval_time_s = 0.0
            save_time_s = 0.0

            train_start_t = time.perf_counter()
            for step, batch in enumerate(train_loader, start=1):
                batch = model.move_to_device(batch, device)
                loss_dict = model.train_core(
                    batch,
                    optim,
                    beta=config.training.beta,
                    weighted_loss=config.training.weighted_loss,
                    gradient_clip=config.training.gradient_clip,
                    gradient_clip_kwargs={"max_norm": config.training.gradient_clip_max_norm},
                )
                epoch_loss += float(loss_dict["loss"].detach())
                epoch_recon += float(loss_dict["recon_loglikelihood"].detach())
                epoch_kl += float(loss_dict["kl_divergence"].detach())
                n += 1
            train_time_s = time.perf_counter() - train_start_t

            if n == 0:
                raise RuntimeError("No batches were produced in phased training. Check parquet files and buffer settings.")

            avg_loss = epoch_loss / n
            avg_recon = epoch_recon / n
            avg_kl = epoch_kl / n
            file_progress = phased_replay.get_file_progress()
            print(
                f"Epoch [{epoch+1:4d}/{config.training.num_epochs}] "
                f"Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
                f"Files seen/completed/skipped: {file_progress['seen']}/{file_progress['completed']}/{file_progress.get('skipped', 0)} "
                f"of {file_progress['total']}"
            )

            writer.add_scalar('train_loss_epoch', avg_loss, epoch)
            writer.add_scalar('train_recon_loglikelihood_epoch', avg_recon, epoch)
            writer.add_scalar('train_kl_divergence_epoch', avg_kl, epoch)
            writer.add_scalar('stream_files_seen', file_progress['seen'], epoch)
            writer.add_scalar('stream_files_completed', file_progress['completed'], epoch)
            writer.add_scalar('stream_files_skipped', file_progress.get('skipped', 0), epoch)

            # Optional epoch-level evaluation for phased SetGuideVAE runs.
            if should_eval_epoch and hasattr(model, "validate") and epoch_eval_loader is not None:
                eval_start_t = time.perf_counter()
                try:
                    eval_metrics = model.validate(
                        valloader=epoch_eval_loader,
                        num_samples=1,
                        beta=config.training.beta,
                        device=device,
                    )
                    print(
                        f"Epoch [{epoch+1:4d}/{config.training.num_epochs}] "
                        f"Eval Loss: {eval_metrics['loss']:.4f} | "
                        f"Recon: {eval_metrics['recon_loglikelihood']:.4f} | "
                        f"KL: {eval_metrics['kl_divergence']:.4f}"
                    )
                    writer.add_scalar('eval_loss_epoch', eval_metrics['loss'], epoch)
                    writer.add_scalar('eval_recon_loglikelihood_epoch', eval_metrics['recon_loglikelihood'], epoch)
                    writer.add_scalar('eval_kl_divergence_epoch', eval_metrics['kl_divergence'], epoch)
                    # Alias eval metrics to val_* tags for shared visualization.
                    writer.add_scalar('val_loss_epoch', eval_metrics['loss'], epoch)
                    writer.add_scalar('val_recon_loglikelihood_epoch', eval_metrics['recon_loglikelihood'], epoch)
                    writer.add_scalar('val_kl_divergence_epoch', eval_metrics['kl_divergence'], epoch)

                    if eval_metrics['loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['loss']

                    if (
                        save_best_model
                        and (epoch + 1) >= save_best_after_epoch
                        and eval_metrics['loss'] < best_eval_loss_for_saving
                    ):
                        best_eval_loss_for_saving = eval_metrics['loss']
                        save_dir = os.path.join(config.logging.log_dir, model_tag)
                        os.makedirs(save_dir, exist_ok=True)
                        save_start_t = time.perf_counter()
                        model.save_model(
                            save_path=save_dir,
                            model_name=f"{model_tag}_best",
                            device=device,
                            normalizer=normalizer,
                        )
                        save_time_s += time.perf_counter() - save_start_t

                    if use_early_stopping and early_stopper is not None:
                        early_stopper(train_metric=avg_loss, val_metric=eval_metrics['loss'], epoch=epoch)
                        if early_stopper.early_stop:
                            print(f"Early stopping triggered at epoch {epoch+1} (best eval loss: {best_eval_loss:.4f})")
                            break
                except Exception as e:
                    print(f"Warning: skipped epoch eval loss computation due to: {e}")
                finally:
                    eval_time_s = time.perf_counter() - eval_start_t

            if use_early_stopping and early_stopper is not None and early_stopper.early_stop:
                break

            if (epoch + 1) % config.training.checkpointing.save_freq == 0:
                save_dir = os.path.join(config.logging.log_dir, model_tag)
                os.makedirs(save_dir, exist_ok=True)
                save_start_t = time.perf_counter()
                model.save_model(save_path=save_dir, model_name=f"{model_tag}_epoch_{epoch+1}", device=device, normalizer=normalizer)
                save_time_s += time.perf_counter() - save_start_t

            total_epoch_time_s = time.perf_counter() - epoch_start_t
            print(
                f"Epoch [{epoch+1:4d}/{config.training.num_epochs}] Timing (s): "
                f"cache={cache_time_s:.2f} | train={train_time_s:.2f} | eval={eval_time_s:.2f} | "
                f"save={save_time_s:.2f} | total={total_epoch_time_s:.2f}"
            )

            if (
                stop_on_all_files_used
                and file_progress['total'] > 0
                and file_progress['completed'] >= file_progress['total']
                and (epoch + 1) >= max(1, min_epochs_before_full_data_stop)
            ):
                print(
                    f"Early stop (all data used): completed {file_progress['completed']} "
                    f"of {file_progress['total']} files at epoch {epoch+1}."
                )
                break

        log_path = os.path.join(config.logging.log_dir, model_tag)
        os.makedirs(log_path, exist_ok=True)
        model.save_model(save_path=log_path, model_name=f"{model_tag}_final", device=device, normalizer=normalizer)

        # Use cached eval loader (kept separate from training cache)
        combined_test_loader = phased_replay.get_eval_loader()
        if combined_test_loader is None and eval_batches > 0:
            phased_replay.refresh_eval_cache(eval_batches)
            combined_test_loader = phased_replay.get_eval_loader()

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

    # Load data sequentially based on what is in the folder and train the model in an online fashion
    nan_as_zero = bool(getattr(config.data, "nan_as_zero", True))
    parquet_only = bool(getattr(config.data, "parquet_only", False))
    if parquet_only:
        cache_dirs = [
            os.path.join(data_dir, "_parquet_cache"),
            os.path.join(data_dir, "parquet"),
        ]
        all_data_files = []
        for cache_dir in cache_dirs:
            if not os.path.isdir(cache_dir):
                continue
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".parquet") and not f.startswith("test")]
            all_data_files.extend(cache_files)
        all_data_files = sorted(set(all_data_files))
        print(
            f"CSV mode with parquet-only loading enabled. "
            f"Found {len(all_data_files)} parquet cache file(s). "
            f"NaN handling is applied at load time (nan_as_zero={nan_as_zero})."
        )
    else:
        all_data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("test")])
    
    data_files = config.data.data_files if len(config.data.data_files) > 0 else all_data_files
    if len(data_files) == 0:
        raise FileNotFoundError(
            f"No training data files found (parquet_only={parquet_only}, nan_as_zero={nan_as_zero})."
        )
    print(f"Data files to be used for training: {data_files}")
    
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
    if model_type == "art" and hasattr(model, "normalizer"):
        model.normalizer = copy.deepcopy(normalizer).to(device)

    if hasattr(model, "conditioner") and model.conditioner is not None:
        user_profiler = model.conditioner.condition_items.get("user_profile")
        if user_profiler is not None:
            fit_userprofiler_on_files(
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
            )

    all_test_tensors = []
    log_path = os.path.join(config.logging.log_dir, model_tag)
    writer = Writer(log_dir=os.path.join(log_path, 'logs'))

    i = 0
    for data_file in data_files:
        i += 1
        print(f"\n=== Processing file {i}/{len(data_files)}: {data_file} ===")
        # === Clean Up Previous Data ===
        if i > 1:
            # Explicitly clear DataLoaders first (they may hold worker refs)
            if 'train_loader' in dir():
                del train_loader
            if 'val_loader' in dir():
                del val_loader
            # Clear the DataModule's internal datasets explicitly
            if dm is not None:
                dm._train_ds = None
                dm._val_ds   = None
                dm._test_ds  = None
                del dm
                dm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Memory cleared before loading {data_file}")


        dm = SmartMeterDataModule(
            data_dir=data_dir,
            filename=data_file,
            meter_col=config.data.columns.meter_id,
            timestamp_dst_col=config.data.columns.timestamp_dst,
            timestamp_utc_col=config.data.columns.timestamp_utc,
            feature_cols=config.data.columns.value_cols,
            window=window,
            batch_size=config.data.dataloader.batch_size,
            num_workers=config.data.dataloader.num_workers,
            shuffle_train=config.data.dataloader.shuffle,
            seed=config.data.splits.random_seed,
            train_frac=config.data.splits.train_ratio,
            val_frac=config.data.splits.val_ratio,
            max_meters=None,
            nan_as_zero=nan_as_zero,
            parquet_only=parquet_only,
            normalizer=normalizer,
            conditioner=conditioner,
        )
        dm.setup()

        all_test_tensors.append((
            dm._test_ds.tensors[0].clone(),  # (N, L, F)
            dm._test_ds.tensors[1].clone(),  # (N, Cond)
            dm._test_ds.tensors[2].clone(),  # (N,) meter ids
            dm._test_ds.tensors[3].clone(),  # (N,) timestamps
        ))

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        if not vamp_initialized and hasattr(model, "prior") and isinstance(model.prior, VampPrior):
            init_from_data = bool(getattr(config.model.prior, "init_from_data", True))
            if init_from_data:
                init_batch = next(iter(train_loader), None)
                noise_scale = float(getattr(config.model.prior, "init_noise_scale", 0.01))
                if _init_vamp_prior_from_batch(model, init_batch, device=device, noise_scale=noise_scale):
                    vamp_initialized = True
                    print("Initialized VampPrior pseudo-inputs from data batch.")

        model_tag, writer = model.fit(
            trainloader = train_loader,
            valloader = val_loader,
            lr = config.training.learning_rate,
            weight_decay = config.training.weight_decay,
            gradient_clip = config.training.gradient_clip,
            gradient_clip_kwargs= {"max_norm": config.training.gradient_clip_max_norm},
            lr_scheduler = config.training.lr_scheduler.enabled,
            lr_scheduler_kwargs = {"step_size": config.training.lr_scheduler.step_size, "gamma": config.training.lr_scheduler.gamma},
            beta = config.training.beta,
            beta_annealing = config.training.beta_annealing.enabled,
            beta_annealing_kwargs = {"n_cycles": config.training.beta_annealing.n_cycles, "ratio": config.training.beta_annealing.ratio},
            epochs = config.training.num_epochs,
            device = device,
            early_stopping = config.training.early_stopping.enabled,
            early_stopping_kwargs= {"patience": config.training.early_stopping.patience, "min_delta": config.training.early_stopping.min_delta},
            save_epoch_freq = config.training.checkpointing.save_freq,
            writer = writer,
            log_dir = config.logging.log_dir,
            weighted_loss = config.training.weighted_loss,
            heirarchical_config = hierarchy,
            normalizer = normalizer,
            compile_model = config.training.compile_model,
            model_name = model_tag,
        )

    combined_test_ds = TensorDataset(
        torch.cat([t[0] for t in all_test_tensors], dim=0),
        torch.cat([t[1] for t in all_test_tensors], dim=0),
        torch.cat([t[2] for t in all_test_tensors], dim=0),
        torch.cat([t[3] for t in all_test_tensors], dim=0),
    )
    combined_test_loader = DataLoader(
        combined_test_ds,
        batch_size=config.data.dataloader.batch_size,
        shuffle=False,
        num_workers=config.data.dataloader.num_workers,
    )
    log_path = os.path.join(config.logging.log_dir, model_tag)
    _save_training_config_snapshot(config=config, log_path=log_path)
    _run_evaluation_and_visualization(
        model=model,
	    conditioner=conditioner,
        test_loader=combined_test_loader,
        normalizer=normalizer,
        writer=writer,
        device=device,
        log_path=log_path,
        config=config,
        phased_mode=False,
    )


# run either VAE or CVAE test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE/CVAE/SetGuideVAE/ART from YAML config")
    parser.add_argument("--config", type=str, default=os.path.join(parent_dir, "config", "config.yaml"))
    parser.add_argument("--skip-validation", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--load_path_model", type=str, default=None, help="Path to a saved model to load and visualize without training")
    args = parser.parse_args()

    config = _load_train_config(args.config)
    config.skip_validation = bool(args.skip_validation)
    config.verbose = not bool(args.quiet)

    train(config)


