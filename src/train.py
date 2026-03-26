import os
import sys
import argparse
import yaml
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
from data.dataloader_phased import StreamingParquetBuffer, WindowSpec as PhaseWindowSpec
from data.normalizer import *
from src.test.run_validation import run_validation
from src.model.hierarchical.aggregation import *
from src.model.hierarchical.config import *
from src.visualization.visualizer import Visualizer
from src.visualization.writer import Writer
from evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
from src.visualization.plot_styling import PlotStyle
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


def _cfg_get(config, path, default=None):
    """Safe getter for dict/namespace-like configs using dot paths."""
    if config is None:
        return default

    cur = config
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _build_phased_eval_loader(eval_stream, batch_size: int, num_workers: int, eval_batches: int, normalizer=None):
    """Build finite eval loader from phased stream, preferring non-zero windows."""
    xs, ms, ts = [], [], []
    max_scan_batches = max(20, max(1, eval_batches) * 20)
    collected_samples = 0
    target_samples = max(1, eval_batches) * int(batch_size)

    for i, (x, cond) in enumerate(eval_stream):
        meter_id = cond[:, 0, 0].long()
        timestamps = cond[:, 1, 0].long()

        # Detect non-zero in RAW space (important when x is normalized).
        x_for_mask = x.float()
        if normalizer is not None and hasattr(normalizer, "inverse_transform") and x_for_mask.dim() == 3:
            try:
                # phased loader yields [B, F, L]; inverse expects feature-last [B, L, F]
                x_tf = x_for_mask.transpose(1, 2).detach().cpu()
                x_raw_tf = normalizer.inverse_transform(x_tf)
                x_for_mask = x_raw_tf.transpose(1, 2)
            except Exception as e:
                print(f"Warning: raw-space non-zero check failed ({e}); falling back to normalized-space check.")

        nonzero_mask = x_for_mask.reshape(x_for_mask.shape[0], -1).abs().sum(dim=1) > 1e-8
        if nonzero_mask.any():
            xs.append(x[nonzero_mask].float())
            ms.append(meter_id[nonzero_mask])
            ts.append(timestamps[nonzero_mask])
            collected_samples += int(nonzero_mask.sum().item())

        if collected_samples >= target_samples:
            break

        if i >= max_scan_batches - 1:
            print(f"Warning: reached scan limit ({max_scan_batches} batches) while collecting eval windows.")
            break

    if len(xs) == 0:
        print("Warning: no non-zero eval windows found; falling back to first streamed eval batches.")
        for i, (x, cond) in enumerate(eval_stream):
            meter_id = cond[:, 0, 0].long()
            timestamps = cond[:, 1, 0].long()
            xs.append(x.float())
            ms.append(meter_id)
            ts.append(timestamps)
            if i >= max(1, eval_batches) - 1:
                break

    if len(xs) == 0:
        return None

    combined_test_ds = TensorDataset(torch.cat(xs, 0), torch.cat(ms, 0), torch.cat(ts, 0))
    return DataLoader(
        combined_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _run_evaluation_and_visualization(model, test_loader, normalizer, conditioner, writer, device, log_path, config, phased_mode: bool):
    """Route evaluation by mode and run shared visualizations."""
    if phased_mode:
        phased_evaluator = PhasedEvaluationSuite(
            model,
            test_loader,
            normalizer=normalizer,
            conditioner=conditioner,
            device='cpu',
            window_size=config.window.input_len,
            output_path=log_path,
        )
        phased_evaluator.evaluate()
    else:
        evaluator = EvalFramework(model, test_loader, normalizer,conditioner=conditioner, device='cpu', output_path=log_path)
        evaluator.register_all()
        evaluator.evaluate()
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

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phase", "parquet", "phased_parquet"}

    if phased_mode and not config.skip_validation:
        print("Phased parquet mode detected. Skipping CSV-based pre-validation (run_validation).")
    if (not config.skip_validation) and (not phased_mode):
        validation_passed = run_validation(
            model_type=config.model.type,
            use_test_data=True,  # use small test data for fast validation
            verbose=config.verbose
        )
        if not validation_passed:
            print("\nValidation failed! Please fix issues before training.")
            print("To skip validation, set skip_validation=True in the config (not recommended)")
            return
        print("\nValidation passed! Proceeding to training\n")
    
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

    if model_type in {"cvae", "set_guide_vae"}:
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
                conditioner.add_condition_item("nr_phases_active", ActivePhasesConditioner())
            else:
                print(f"Warning: Unknown condition tag '{condition_tag}' in config. Skipping this conditioner.")
        
        if data_mode in {"phased", "phase", "parquet", "phased_parquet"} and model_type in {"set_guide_vae"}:
            set_cfg = config.model.set
            set_num_features = set_cfg.num_features if set_cfg.num_features is not None else 3
            input_dim = config.window.input_len * int(set_num_features)
        else:
            input_dim = config.window.input_len * len(config.columns.value_cols)
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
        else:
            set_cfg = config.model.set
            set_num_features = set_cfg.num_features if set_cfg.num_features is not None else len(config.columns.value_cols)
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
            )
            print(f"Initialized SetGuideVAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}, conditioner_tags={config.conditioning.tags}")
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

    # === Load Data === 
    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)
    data_dir = config.data_dir

    # === Phased parquet path ===
    if phased_mode:
        if model_type not in {"set_guide_vae"}:
            raise ValueError("Phased parquet mode currently supports only SetGuideVAE (set_guide_vae) model type.")

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
        parquet_paths = parquet_paths[:20]  # temporary limit for testing

        print(f"Phased mode enabled. Using {len(parquet_paths)} parquet file(s).")

        phased_cfg = getattr(config.data, "phased", SimpleNamespace())
        replay_capacity = int(getattr(phased_cfg, "replay_buffer_capacity", 10000))
        min_buffer_size = int(getattr(phased_cfg, "min_buffer_size", 2000))
        phased_workers = int(getattr(phased_cfg, "num_workers", config.dataloader.num_workers))
        steps_per_epoch = int(getattr(phased_cfg, "steps_per_epoch", 100))
        eval_batches = int(getattr(phased_cfg, "eval_batches", 5))

        normalizer = None
        norm_method = str(getattr(config.normalizer, "method", "log1p")).lower()
        if norm_method == "log1p_phase":
            normalizer = Normalizer(method="log1p_phase")
            stats_name = f"normalization_stats_{os.path.basename(os.path.normpath(data_dir))}.json"
            stats_path = os.path.join(config.logging.log_dir, stats_name)
            normalizer.fit(X=None, sm_files=parquet_paths, stats_path=stats_path)
            print("Using phased normalizer: log1p_phase")
        else:
            print(f"Warning: phased_parquet mode currently expects normalization.method='log1p_phase'. Got '{norm_method}'. Continuing without normalization.")
        if conditioner.condition_items.get("user_profile") is not None:
            conditioner.condition_items.get("user_profile", None).fit_on_files(parquet_paths)  # Fit user profiler on all data files for phased mode
        if config.window.aggregated and config.window.input_len != 24:
            print(f"Warning: window.aggregated is True but window.input_len is {config.window.input_len}. For hourly aggregation, consider setting input_len to 24.")
            config.window.input_len = 24
            config.window.stride = 24
        phase_window = PhaseWindowSpec(input_len=config.window.input_len, stride=config.window.stride)
        train_ds = StreamingParquetBuffer(
            sm_files=parquet_paths,
            replay_buffer_capacity=replay_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=phase_window,
            normalizer=normalizer,
            num_workers=phased_workers,
            hourly_aggregation=config.window.aggregated
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.dataloader.batch_size,
            num_workers=phased_workers,
        )

        writer = Writer(log_dir=os.path.join(config.logging.log_dir, model_tag))
        optim = model.get_optimizer(lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

        model = model.to(device)
        model.prior.to(device)

        print(f"Starting phased training for {config.training.num_epochs} epoch(s), {steps_per_epoch} step(s)/epoch")
        for epoch in range(config.training.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n = 0

            for step, batch in enumerate(train_loader, start=1):
                if step > steps_per_epoch:
                    break
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

            if n == 0:
                raise RuntimeError("No batches were produced in phased training. Check parquet files and buffer settings.")

            avg_loss = epoch_loss / n
            avg_recon = epoch_recon / n
            avg_kl = epoch_kl / n
            print(f"Epoch [{epoch+1:4d}/{config.training.num_epochs}] Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

            writer.add_scalar('train_loss_epoch', avg_loss, epoch)
            writer.add_scalar('train_recon_loglikelihood_epoch', avg_recon, epoch)
            writer.add_scalar('train_kl_divergence_epoch', avg_kl, epoch)

            if (epoch + 1) % config.training.checkpointing.save_freq == 0:
                save_dir = os.path.join(config.logging.log_dir, model_tag)
                os.makedirs(save_dir, exist_ok=True)
                model.save_model(save_path=save_dir, model_name=f"{model_tag}_epoch_{epoch+1}", device=device, normalizer=normalizer)

        log_path = os.path.join(config.logging.log_dir, model_tag)
        os.makedirs(log_path, exist_ok=True)
        model.save_model(save_path=log_path, model_name=f"{model_tag}_final", device=device, normalizer=normalizer)

        # Build finite eval loader in (x, meter_id, timestamps) format
        eval_ds = StreamingParquetBuffer(
            sm_files=parquet_paths,
            replay_buffer_capacity=replay_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=phase_window,
            normalizer=normalizer,
            num_workers=phased_workers,
        )
        eval_stream = DataLoader(eval_ds, batch_size=config.dataloader.batch_size, num_workers=phased_workers)
        combined_test_loader = _build_phased_eval_loader(
            eval_stream=eval_stream,
            batch_size=config.dataloader.batch_size,
            num_workers=phased_workers,
            eval_batches=eval_batches,
            normalizer=normalizer,
        )

        if combined_test_loader is not None:
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

        return model

    # Load data sequentially based on what is in the folder and train the model in an online fashion
    all_data_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("test")])
    
    data_files = config.data.data_files if len(config.data.data_files) > 0 else all_data_files
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
    )

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
                normalizer=normalizer
            )

    all_test_tensors = []
    log_path = os.path.join(config.logging.log_dir, model_tag)
    writer = Writer(log_dir=os.path.join(log_path, 'logs'))

    # SAve training config
    with open(os.path.join(log_path, 'training_config.yaml'), 'w') as f:
        yaml.dump(config.config, f)

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
    _run_evaluation_and_visualization(
        model=model,
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
    parser = argparse.ArgumentParser(description="Train VAE/CVAE/SetGuideVAE from YAML config")
    parser.add_argument("--config", type=str, default=os.path.join(parent_dir, "config", "config.yaml"))
    parser.add_argument("--skip-validation", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--load_path_model", type=str, default=None, help="Path to a saved model to load and visualize without training")
    args = parser.parse_args()

    config = _load_train_config(args.config)
    config.skip_validation = bool(args.skip_validation)
    config.verbose = not bool(args.quiet)

    train(config)


