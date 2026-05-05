import os
import yaml
import shutil
from types import SimpleNamespace
from src.model.cvae import CVAE
from src.model.set import SetGuideVAE
from src.model.sequential import SequentialPhaseVAE
from src.model.vae import VAE
from src.model.art import AutoRegressiveTransformerVAE
from src.model.diffusion import DiffusionModel, NoiseScheduler, ConditionalUNet1D
from src.model.phase_splitter import HierarchicalSplitter
from src.model.hierarchical.config import HierarchicalConfig, LevelConfig
from src.data.conditioner import (
    Conditioner, DayOfWeekConditioner, UserProfiler, 
    HolidayConditioner, MonthOfYearConditioner, 
    ActivePhaseCountConditioner, HeatpumpPhaseConditioner, ElectricVehicleConditioner, SolarPanelConditioner
)
from src.model.baseline import AverageBaselineModel, QuantileBaselineModel, FlatMeanBaselineModel
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

def create_model_from_config(config, device="cpu"):
    """
    Creates and returns a BaseModel subclass and its Conditioner based on the Hydra config.
    Returns:
        model: initialized BaseModel subclass (VAE, CVAE, DiffusionModel, etc.)
    """
    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}
    
    # === Hierarchical latent space setup ===
    hierarchy = None
    hierarchy_cfg = getattr(config, "hierarchy", SimpleNamespace())
    if hierarchy_cfg and getattr(hierarchy_cfg, "enabled", False):
        levels = []
        for level_type in getattr(hierarchy_cfg, "levels", []):
            input_dim_k = config.model.latent_dim if len(levels) == 0 else levels[-1].latent_dim
            levels.append(LevelConfig(
                name=f"Level {len(levels)+1}",
                strategy=level_type,
                input_dim=input_dim_k,
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

    # === Model Setup ===
    model_type = str(config.model.type).lower() if hasattr(config, "model") and hasattr(config.model, "type") else "vae"
    conditioner = None
    encoder_cfg = getattr(config.model, "encoder", SimpleNamespace())
    decoder_cfg = getattr(config.model, "decoder", SimpleNamespace())
    encoder_hidden_dims = getattr(encoder_cfg, "hidden_dims", None)
    decoder_hidden_dims = getattr(decoder_cfg, "hidden_dims", None)
    encoder_dropout = getattr(encoder_cfg, "dropout", None)
    decoder_dropout = getattr(decoder_cfg, "dropout", None)

    if (model_type in ("phase_splitter", "set_guide_vae", "sequential_phase_vae")) and not phased_mode:
        print("You have indicated a phased only model type is currently supported only with data.mode='phased_parquet'. Forcing phased mode for this run.")
        phased_mode = True

    if model_type == "diffusion" and hierarchy is not None:
        print("DiffusionModel currently does not support hierarchy.enabled=true. Ignoring hierarchical config.")
        hierarchy = None

    # === Conditioners ===
    conditioning_cfg = getattr(config, "conditioning", SimpleNamespace())
    tags = getattr(conditioning_cfg, "tags", [])
    if model_type == "vae":
        print("No conditioning will be used for this run.")
    else:
        conditioner = Conditioner()
        for condition_tag in tags:
            if condition_tag == "day_of_week":
                conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            elif condition_tag == "user_profile":
                user_prof_cfg = getattr(conditioning_cfg, "user_profile", SimpleNamespace())
                online_cfg = getattr(conditioning_cfg, "online_user_profile", SimpleNamespace())
                is_online = getattr(user_prof_cfg, "online", False)
                decay = getattr(online_cfg, "decay", 1.0) if is_online else 1.0
                
                profiler = UserProfiler(
                    n_topics=getattr(user_prof_cfg, "n_topics", 10),
                    num_clusters=getattr(user_prof_cfg, "num_clusters", 100), 
                    path=getattr(user_prof_cfg, "cache_path", "models/UserProfiles/user_profiles.pkl"), 
                    verbose=getattr(user_prof_cfg, "verbose", False), 
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
            elif condition_tag in {"ev_phase", "electric_vehicle_phase"}:
                ev_cfg = getattr(conditioning_cfg, "ev_phase", SimpleNamespace())
                ev_path = getattr(ev_cfg, "path", "data/EV_detection_6000_SM.csv")
                conditioner.add_condition_item("ev_phase", ElectricVehicleConditioner(topology_path=ev_path))
            elif condition_tag == "pv" or condition_tag == "solar_panel":
                pv_cfg = getattr(conditioning_cfg, "solar_panel", SimpleNamespace())
                pv_path = getattr(pv_cfg, "path", "data/topology/cleaned/meter_cabinet_connection.csv")
                conditioner.add_condition_item("solar_panel", SolarPanelConditioner(topology_path=pv_path))
            else:
                print(f"Warning: Unknown condition tag '{condition_tag}' in config. Skipping this conditioner.")

    # === Get number of features for model initialization ===
    set_cfg = getattr(config.model, "set", SimpleNamespace(num_features=None))
    csv_num_features = len(getattr(getattr(config, "columns", SimpleNamespace()), "value_cols", []))
    if phased_mode:
        phased_num_features = getattr(set_cfg, "num_features", None)
        set_num_features = int(phased_num_features) if phased_num_features is not None else 3
    else:
        set_num_features = int(csv_num_features)
    
    seq_len = int(getattr(config.window, "input_len", 96))
    input_dim = seq_len * int(set_num_features)
    
    model_cfg = getattr(config, "model", SimpleNamespace())
    prior_cfg = getattr(model_cfg, "prior", SimpleNamespace())

    # === Model Initialization ===
    if model_type == "cvae":
        model = CVAE(input_dim=input_dim, latent_dim=model_cfg.latent_dim, 
                    conditioner=conditioner, dictionary_size=getattr(prior_cfg, "dictionary_size", 512), 
                    vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
                    encoder_type=encoder_cfg.type,
                    decoder_type=decoder_cfg.type,
                    prior_type=prior_cfg.type,
                    hierarchy_config=hierarchy,
                    encoder_hidden_dims=encoder_hidden_dims,
                    decoder_hidden_dims=decoder_hidden_dims,
                    encoder_dropout=encoder_dropout,
                    decoder_dropout=decoder_dropout,
                    phased_mode=phased_mode,
                    sequence_length=seq_len,
                    )
    elif model_type == "set_guide_vae":
        model = SetGuideVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            dictionary_size=getattr(prior_cfg, "dictionary_size", 512),
            vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
            encoder_type=encoder_cfg.type,
            decoder_type=decoder_cfg.type,
            prior_type=prior_cfg.type,
            hierarchy_config=hierarchy,
            sequence_length=seq_len,
            num_features=set_num_features,
            set_hidden_dim=getattr(set_cfg, "hidden_dim", 128),
            set_num_heads=getattr(set_cfg, "num_heads", 4),
            set_num_inds=getattr(set_cfg, "num_inds", 16),
            set_num_blocks=getattr(set_cfg, "num_blocks", 2),
            is_isab=getattr(set_cfg, "is_isab", True),
            use_abl_pool=bool(getattr(set_cfg, "use_abl_pool", False)),
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
    elif model_type == "sequential_phase_vae":
        model = SequentialPhaseVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            distribution_dict=getattr(model_cfg, "distribution_dict", {}),
            prior_type=prior_cfg.type,
            hierarchy_config=hierarchy,
            sequence_length=seq_len,
            encoder_type=encoder_cfg.type,
            decoder_type=decoder_cfg.type,
            dictionary_size=getattr(prior_cfg, "dictionary_size", 512),
            vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
    elif model_type == "art":
        art_cfg = getattr(model_cfg, "art", SimpleNamespace())
        model = AutoRegressiveTransformerVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            dictionary_size=getattr(prior_cfg, "dictionary_size", 512),
            vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
            encoder_type=encoder_cfg.type,
            decoder_type=decoder_cfg.type,
            prior_type=prior_cfg.type,
            hierarchy_config=hierarchy,
            bin_size=getattr(art_cfg, "num_bins", 1024),
            nhead=getattr(art_cfg, "transformer_num_heads", 4),
            num_layers=getattr(art_cfg, "transformer_num_layers", 2),
            d_model=getattr(art_cfg, "d_model", 128),
            sequence_length=seq_len,
            num_features=set_num_features,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
    elif model_type == "diffusion":
        diff_cfg = getattr(model_cfg, "diffusion", SimpleNamespace())
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
            condition_dim=int(conditioner.get_condition_dim() if conditioner else 0),
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
            sequence_length=seq_len,
            num_features=int(set_num_features),
            condition_dropout_prob=float(getattr(diff_cfg, "condition_dropout_prob", 0.0)),
            reconstruction_noise_fraction=float(getattr(diff_cfg, "reconstruction_noise_fraction", 0.3)),
            deterministic_reconstruction=bool(getattr(diff_cfg, "deterministic_reconstruction", True)),
            phased_mode=phased_mode,
        )
    elif model_type == "phase_splitter":
        phase_splitter_cfg = getattr(model_cfg, "phase_splitter", SimpleNamespace())
        model = HierarchicalSplitter(
            conditioner=conditioner,
            sequence_length=seq_len,
            condition_dim=int(conditioner.get_condition_dim() if conditioner else 0),
            base_channels=int(getattr(phase_splitter_cfg, "base_channels", 64)),
            temperature=float(getattr(phase_splitter_cfg, "temperature", 2.0)),
            phased_mode=phased_mode,
        )
    elif model_type == "vae":
        model = VAE(input_dim=input_dim, latent_dim=model_cfg.latent_dim,
                    encoder_type=encoder_cfg.type,
                    decoder_type=decoder_cfg.type,
                    prior_type=prior_cfg.type,
                    dictionary_size=getattr(prior_cfg, "dictionary_size", 512),
                    vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
                    hierarchy_config=hierarchy,
                    encoder_hidden_dims=encoder_hidden_dims,
                    decoder_hidden_dims=decoder_hidden_dims,
                    encoder_dropout=encoder_dropout,
                    decoder_dropout=decoder_dropout,
                    phased_mode=phased_mode,
                    sequence_length=seq_len,
                    )
    elif model_type == "baseline_average":
        model = AverageBaselineModel(phased_mode=phased_mode, 
                                     sequence_length=seq_len, 
                                     device=device, 
                                     )
    elif model_type == "baseline_quantile":
        model = QuantileBaselineModel(phased_mode=phased_mode, 
                                     sequence_length=seq_len, 
                                     device=device, 
                                     reservoir_size=getattr(model_cfg, "reservoir_size", 5000)
                                     )
    elif model_type == "baseline_mean":
        model = FlatMeanBaselineModel(phased_mode=phased_mode,
                                        sequence_length=seq_len,
                                        device=device,
                                     )
    elif model_type == "sequential_vae":
        model = SequentialPhaseVAE(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            conditioner=conditioner,
            distribution_dict=getattr(model_cfg, "distribution_dict", {}),
            prior_type=prior_cfg.type,
            hierarchy_config=hierarchy,
            sequence_length=seq_len,
            encoder_type=encoder_cfg.type,
            decoder_type=decoder_cfg.type,
            dictionary_size=getattr(prior_cfg, "dictionary_size", 512),
            vamp_num_components=getattr(prior_cfg, "vamp_num_components", 20),
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            phased_mode=phased_mode,
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}' specified in config.")

    return model

