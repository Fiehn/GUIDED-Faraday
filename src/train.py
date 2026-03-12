import os
import sys
from xml.parsers.expat import model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.encoder_decoder_model import *
from data.conditioner import *
from src.models.cvae import *
from src.models.vae import *
from data.dataloader import *
from data.normalizer import *
from src.test.run_validation import run_validation

from src.models.hierarchical.aggregation import *
from src.models.hierarchical.config import *
 
from src.visualization.visualizer import Visualizer
from src.visualization.writer import Writer

from evaluation.eval_framework import EvalFramework

from src.visualization.plot_styling import PlotStyle
import gc
import yaml

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to be 999

    if not config.skip_validation:
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
    if config.model.type == "cvae":
        conditioner = Conditioner()
        for condition_tag in config.conditioning.tags:
            if condition_tag == "day_of_week":
                conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            elif condition_tag == "user_profile":
                if config.conditioning.user_profile.online:
                    decay = config.conditioning.online_user_profile.decay
                    update_frequency = config.conditioning.online_user_profile.update_frequency
                else:
                    decay = 1.0
                    update_frequency = -1

                profiler = UserProfiler(n_topics=config.conditioning.user_profile.n_topics,
                    num_clusters=config.conditioning.user_profile.num_clusters, 
                    path=config.conditioning.user_profile.cache_path, 
                    verbose=config.conditioning.user_profile.verbose, 
                    decay=decay, 
                    update_frequency=update_frequency)
                
                conditioner.add_condition_item("user_profile", profiler)
                    
            elif condition_tag == "holiday":
                conditioner.add_condition_item("holiday", HolidayConditioner(holiday_path='data/holidays/denmark_holidays.json'))
            elif condition_tag == "month_of_year":
                conditioner.add_condition_item("month_of_year", MonthOfYearConditioner())
            else:
                print(f"Warning: Unknown condition tag '{condition_tag}' in config. Skipping this conditioner.")
        
        input_dim = config.data.window.input_len * len(config.data.columns.value_cols)
        model = CVAE(input_dim=input_dim, latent_dim=config.model.latent_dim, 
                     conditioner=conditioner, dictionary_size=config.model.prior.dictionary_size, 
                     vamp_num_components=config.model.prior.vamp_num_components,
                     encoder_type=config.model.encoder.type,
                     decoder_type=config.model.decoder.type,
                     prior_type=config.model.prior.type,
                     hierarchy_config=hierarchy,
                     )
        print(f"Initialized CVAE with input_dim={input_dim}, latent_dim={config.model.latent_dim}, encoder_type={config.model.encoder.type}, decoder_type={config.model.decoder.type}, prior_type={config.model.prior.type}, conditioner_tags={config.conditioning.tags}")
    elif config.model.type == "vae":
        input_dim = config.data.window.input_len * len(config.data.columns.value_cols)
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
    window = WindowSpec(input_len=config.data.window.input_len, stride=config.data.window.stride)
    data_dir = config.data.data_dir
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

    all_test_tensors = []

    writer = Writer(log_dir=os.path.join(config.logging.log_dir, model_tag, 'logs'))
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
        )
        dm.setup()

        all_test_tensors.append((
            dm._test_ds.tensors[0].clone(),  # (N, L, F)
            dm._test_ds.tensors[1].clone(),  # (N,) meter ids
            dm._test_ds.tensors[2].clone(),  # (N,) timestamps
        ))

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        if i == 1 and model.conditioner.condition_items.get("user_profile").is_fitted == False:
            print("Fitting user profile conditioner on first training file...")
            model.conditioner.condition_items["user_profile"].fit(train_loader)

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
    )
    combined_test_loader = DataLoader(
        combined_test_ds,
        batch_size=config.data.dataloader.batch_size,
        shuffle=False,
        num_workers=config.data.dataloader.num_workers,
    )
    log_path = os.path.join(config.logging.log_dir, model_tag)

    # SAve training config
    with open(os.path.join(log_path, 'training_config.yaml'), 'w') as f:
        yaml.dump(config.config, f)

    # === Evaluation ===
    evaluator = EvalFramework(model, combined_test_loader, normalizer, device='cpu')
    evaluator.register_all()
    evaluator.evaluate()
    evaluator.save_results(os.path.join(log_path, 'eval_results_combined.json'))
    
    # === Visualization ===
    visualizer = Visualizer(model = model, writer = writer,
                            plot_style = PlotStyle(),
                            log_dir = os.path.join(log_path, 'plots'),
                            device = device,
                            normalizer = normalizer)
    visualizer.visualize(test_loader = combined_test_loader,
                         num_samples = config.visualize.num_samples,
                         num_recon_per = config.visualize.num_recon_per,
                         num_synthetic = config.visualize.num_synthetic,
                         num_originals = config.visualize.num_originals,
                         num_synthetic_per = config.visualize.num_synthetic_per
                         )


# now we can run either VAE or CVAE test
if __name__ == "__main__":
    from src.config_loader import ConfigLoader
    config_loader = ConfigLoader(config_path='config/config.yaml')
    train(config_loader)





