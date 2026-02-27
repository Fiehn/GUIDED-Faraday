import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Models.encoder_decoder_model import *
from Data.Conditioner import *
from src.Models.CVAE import *
from src.Models.Online_CVAE import *
from src.Models.VAE import *
from Data.dataloader import *
from Data.normalizer import *
from torch.utils.tensorboard import SummaryWriter
from src.train_utils import *
from src.test.run_validation import run_validation

from src.Hierarchical.aggregation import *
from src.Hierarchical.config import *


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test(model_type: str = "vae", skip_validation: bool = False, file_name = "test_data_small.csv"):
    model_type = model_type.lower()
    assert model_type in {"vae", "cvae"}

    print("Using device: ", device)

    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to be 999

    ##############################
    # Validate model before training
    if not skip_validation:
        print("\nStarting model validation before training...\n") 
        validation_passed = run_validation(
            model_type=model_type,
            use_test_data=True,  # use small test data for fast validation
            verbose=True
        )
        
        if not validation_passed:
            print("\n!! Validation failed! Please fix issues before training.")
            print("To skip validation, use: run_test(model_type='cvae', skip_validation=True)")
            return
        
        print("\nValidation passed! Proceeding to training...\n")
    else:
        print("\n!!  Skipping validation (not recommended)\n")

    ##############################
    # load data
    data_dir = "data/" # "data/"
    filename = file_name

    meter_col = "meter_number"
    timestamp_dst_col = "timestamp_dst"
    timestamp_utc_col = "timestamp_utc"
    feature_cols = ("hourly_energy_consumption",)  # , "hourly_energy_production")

    window = WindowSpec(input_len=24, stride=24)  # full day windows

    batch_size = 512 # 256 # tried smaller batch size for CVAE
    num_workers = 0
    shuffle_train = False
    seed = 42
    train_frac = 0.8
    val_frac = 0.1
    normalizer = Normalizer(method='log1p')

    print("Initializing DataModule...")
    dm = SmartMeterDataModule(
        data_dir=data_dir,
        filename=filename,
        meter_col=meter_col,
        timestamp_dst_col=timestamp_dst_col,
        timestamp_utc_col=timestamp_utc_col,
        feature_cols=feature_cols,
        window=window,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=shuffle_train,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_meters=None, # None # tried smaller number for CVAE
        normalizer=normalizer, # normalizer # try without normalizer
    )
    
    dm.setup()
    print("done setup")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    ##############################
    # get one batch
    x, meter_id, timestamps, _ = next(iter(train_loader))  # Now also get timestamps
    print("x shape:", x.shape)
    print("meter_id shape:", meter_id.shape, "example:", meter_id[:5])
    print("timestamps shape:", len(timestamps))

    # flatten for vae and cvae
    x_flat = x.reshape(x.shape[0], -1)
    print("x_flat shape:", x_flat.shape)

    num_features = x.shape[2]
    input_dim = 24 * num_features  # flattened window dim
    print("input_dim:", input_dim)

    if os.path.exists("models/UserProfiles/user_profiler.pkl"):
        print("UserProfiler model found on disk.")
    else:
        print("UserProfiler model not found on disk. Creating..")
        os.makedirs("models/UserProfiles", exist_ok=True)
        UserProfiler(num_clusters=30, path="models/UserProfiles/user_profiler.pkl", verbose=True).fit(train_loader)

    # Test conditioner - pass timestamps instead of x
    conditioner = Conditioner()
    conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
    conditioner.add_condition_item("user_profile", UserProfiler(num_clusters=30, path="models/UserProfiles/user_profiler.pkl", verbose=True))
    conditioner.add_condition_item("holiday", HolidayConditioner(holiday_path='data/holidays/denmark_holidays.json')) # TODO: check that this works and actually adds on the days it says it should
    cond = conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)  # Pass all data (x, meter, timestamps)
    
    print("Condition shape:", cond.shape)


    ##############################
    # use chosen model
    if model_type == "vae":
        print("Initializing VAE model...")
        model = VAE(input_dim=input_dim, latent_dim=32, device=device)
        log_dir = "./runs/VAE_training"
    else:
        print("Initializing CVAE model (condition = day-of-week + user-profile + holiday)...")
        model = CVAE(input_dim=input_dim, latent_dim=32, conditioner=conditioner, 
                     encoder_type="dictionary",decoder_type='dictionary', 
                     prior_type="vampprior", device=device)
        log_dir = "./runs/CVAE_training"

    ##############################
    # Heirarchy latent space
    h_config = HierarchicalConfig(
        levels=[
            LevelConfig(name="Level 1", input_dim=input_dim, latent_dim=32, strategy="linear", top_down=False, bottom_up=True, loss_weight=1.0),
            LevelConfig(name="Level 2", input_dim=32, latent_dim=16, strategy="linear", top_down=True, bottom_up=True, loss_weight=0.5),
            LevelConfig(name="Level 3", input_dim=16, latent_dim=8, strategy="linear", top_down=True, bottom_up=False, loss_weight=0.25),
        ],
        aggregation="concat",
        bidirectional=True,
        residual_connections=True,
        kl_annealing=True,
        kl_warmup_steps=1000,
        free_bits=0.1,
    )


    ##############################
    # Train
    # Use the .fit() method with all its features
    model_tag = model.fit(
        trainloader=train_loader,
        valloader=val_loader,
        lr=1e-3,
        weight_decay=1e-4,
        gradient_clip=1.0,
        gradient_clip_kwargs={"max_norm": 1.0},
        lr_scheduler=True,
        lr_scheduler_kwargs={"step_size": 10, "gamma": 0.9},
        beta=1.0, # beta_max when annealing is on; fixed beta otherwise
        beta_annealing=True,  # cyclical annealing to avoid posterior collapse
        beta_annealing_kwargs={"n_cycles": 4, "ratio": 0.5},
        num_samples=1,
        epochs=20,
        validation_freq=1,
        validation_num_samples=1,
        device=device,
        early_stopping=True,
        early_stopping_kwargs={"patience": 3, "min_delta": 0.001},
        save_epoch_freq=10,
        log_dir=log_dir,
        weighted_loss=15.0, # Set > 0 to give more weight to samples with higher absolute values, which should help with the sparsity issue in smart meter data
        heirarchical_config=h_config, # Pass the hierarchical config to enable hierarchical latent space training
    )

    print("Training complete. Model saved as ", model_tag)

    ##############################
    # Reopen TensorBoard writer for logging visualizations #TODO: figure out this tensorboard stuff it is a mess
    log_dir_tagged = os.path.join(log_dir, model_tag)
    writer = SummaryWriter(log_dir=log_dir_tagged)

    # Move model
    model.to(device)
    model.prior.to(device)
    model.eval()

    # Visualize reconstructions on test set
    visualize_trained_model(model=model, test_loader=test_loader, normalizer=normalizer, device=device,output_dir="Output/Plots", model_tag=model_tag)

    writer.close()
    print("\nDone.")
    print(f"TensorBoard: tensorboard --logdir={log_dir}/{model_tag}")

def online_train(model_type: str = "vae", skip_validation: bool = False):
    if not skip_validation:
        print("\nStarting model validation before online training...\n") 
        validation_passed = run_validation(
            model_type=model_type,
            use_test_data=True,  # use small test data for fast validation
            verbose=True
        )
        
        if not validation_passed:
            print("\n!! Validation failed! Please fix issues before training.")
            print("To skip validation, use: online_train(model_type='cvae', skip_validation=True)")
            return
        
        print("\nValidation passed! Proceeding to online training...\n")
    else:
        print("\n!!  Skipping validation (not recommended)\n")

    # Load data sequentially based on what is in the folder and train the model in an online fashion
    data_files = sorted([f for f in os.listdir("data/") if f.endswith(".csv") and not f.startswith("test")])
    # remove data files that are already trained on (check runs folder for existing models)
    # TODO: Check runs and find and remove already trained files in the list.

    normalizer = Normalizer('online_log1p')
    # Create the model and normalizer and conditioner once
    if model_type == "vae":
        print("Initializing VAE model...")
        model = VAE(input_dim=24 * 1, latent_dim=32, device=device)  # input_dim will be updated based on data
        log_dir = "./runs/VAE_online_training"
    else:
        print("Initializing CVAE model (condition = day-of-week + user-profile)...")
        conditioner = Conditioner()
        conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
        conditioner.add_condition_item("user_profile", OnlineUserProfiler(num_clusters=30, path="models/UserProfiles/user_profiler.pkl", verbose=True))
        #conditioner.add_condition_item("holiday", HolidayConditioner(holiday_path='data/holidays/denmark_holidays.json')) # TODO: check that this works and actually adds on the days it says it should

        heirarchy = HierarchicalConfig(
            levels=[
                LevelConfig(name="Level 1", input_dim=24 * 1, latent_dim=32, strategy="linear", top_down=False, bottom_up=True, loss_weight=1.0),
                LevelConfig(name="Level 2", input_dim=32, latent_dim=16, strategy="linear", top_down=True, bottom_up=True, loss_weight=0.5),
                LevelConfig(name="Level 3", input_dim=16, latent_dim=8, strategy="linear", top_down=True, bottom_up=False, loss_weight=0.25),
            ],
            aggregation="concat",
            bidirectional=True,
            residual_connections=True,
            kl_annealing=True,
            kl_warmup_steps=1000,
            free_bits=0.1,
        )

        model = OnlineCVAE(input_dim=24, latent_dim=32, conditioner=conditioner, 
                     encoder_type="dictionary",decoder_type='dictionary', 
                     prior_type="vampprior", device=device,
                     normalizer=normalizer, hierarchy_config=heirarchy)  # input_dim will be updated based on data
        log_dir = "./runs/CVAE_online_training"
    
    i = 0
    last_dm = None
    last_test_loader = None
    last_data_file = None

    for data_file in data_files:
        print(f"\nProcessing {data_file}")
        i += 1

        # ── Clean up BEFORE loading new data ──────────────────────────
        import gc
        if i > 1:
            # Explicitly clear DataLoaders first (they may hold worker refs)
            if 'train_loader' in dir():
                del train_loader
            if 'val_loader' in dir():
                del val_loader
            if last_test_loader is not None:
                del last_test_loader
                last_test_loader = None
            # Clear the DataModule's internal datasets explicitly
            if last_dm is not None:
                last_dm._train_ds = None
                last_dm._val_ds   = None
                last_dm._test_ds  = None
                del last_dm
                last_dm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Memory cleared before loading {data_file}")

        # Load data for this file
        dm = SmartMeterDataModule(
            data_dir="data/",
            filename=data_file,
            meter_col="meter_number",
            timestamp_dst_col="timestamp_dst",
            timestamp_utc_col="timestamp_utc",
            feature_cols=("hourly_energy_consumption",),
            window=WindowSpec(input_len=24, stride=24),
            batch_size=512,
            num_workers=0,
            shuffle_train=False,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            max_meters=None,
            normalizer=normalizer,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        # Train on this data
        model.fit(
            trainloader=train_loader,
            valloader=val_loader,
            lr=1e-3,
            weight_decay=1e-4,
            gradient_clip=1.0,
            gradient_clip_kwargs={"max_norm": 1.0},
            lr_scheduler=True,
            lr_scheduler_kwargs={"step_size": 10, "gamma": 0.9},
            beta=1.0,
            beta_annealing=True,
            beta_annealing_kwargs={"n_cycles": 4, "ratio": 0.5},
            num_samples=1,
            epochs=5,
            validation_freq=1,
            validation_num_samples=1,
            device=device,
            early_stopping=True,
            early_stopping_kwargs={"patience": 3, "min_delta": 0.001},
            save_epoch_freq=10,
            log_dir=os.path.join(log_dir, data_file.replace(".csv", "")),
        )
        print(f"Finished training on {data_file} advancing to next file...\n")

        # Free train/val loaders but keep dm and test_loader for potential visualization
        del train_loader, val_loader
        gc.collect()

        last_dm          = dm
        last_test_loader = test_loader
        last_data_file   = data_file

    # Visualize using the last file's test set
    if last_test_loader is not None and last_data_file is not None:
        model.to('cpu')
        model.eval()
        visualize_trained_model(
            model=model,
            test_loader=last_test_loader,
            normalizer=normalizer,
            device='cpu',
            output_dir="Output/Plots",
            model_tag="online_" + last_data_file.replace(".csv", ""),
        )
    else:
        print("No data files found — skipping visualization.")

    # Final cleanup
    if last_dm is not None:
        del last_dm, last_test_loader



# now we can run either VAE or CVAE test
if __name__ == "__main__":
    # choose: "vae" or "cvae"
    # run_test(model_type="vae")
    # run_test(model_type="vae", skip_validation=False)  # VAE with validation
    # run_test(model_type="cvae", skip_validation=False)  # CVAE with validation
    # run_test(model_type="cvae", skip_validation=True)  # Skip validation (not recommended)
     online_train(model_type="cvae", skip_validation=False)  # Online training with CVAE and validation
    #run_test(model_type="cvae", skip_validation=False, file_name="test_data_small.csv")  # CVAE with validation on small test data






