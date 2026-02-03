import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.encoder_decoder_model import *
from src.Conditioner import *
from src.Models import *
from src.dataloader import *
from src.normalizer import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.train_utils import *

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test(model_type: str = "vae"):
    model_type = model_type.lower()
    assert model_type in {"vae", "cvae"}

    print("Using device: ", device)

    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to bbe 999

    ##############################
    # load data
    data_dir = "data/" # "data/"
    filename = "hourly_measurements_2023_10.csv"

    meter_col = "meter_number"
    timestamp_dst_col = "timestamp_dst"
    timestamp_utc_col = "timestamp_utc"
    feature_cols = ("hourly_energy_consumption",)  # , "hourly_energy_production")

    window = WindowSpec(input_len=24, stride=24)  # full day windows

    batch_size = 256 # 256 # tried smaller batch size for CVAE
    num_workers = 0
    shuffle_train = False
    seed = 42
    train_frac = 0.8
    val_frac = 0.1
    normalizer = Normalizer(method='zero_preserved_log')

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
        max_meters=10, # None # tried smaller number for CVAE
        normalizer=normalizer,
    )
    
    dm.setup()
    print("done setup")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    ##############################
    # get one batch
    x, meter_id, timestamps = next(iter(train_loader))  # Now also get timestamps
    print("x shape:", x.shape)
    print("meter_id shape:", meter_id.shape, "example:", meter_id[:5])
    print("timestamps shape:", len(timestamps))

    # flatten for vae and cvae
    x_flat = x.reshape(x.shape[0], -1)
    print("x_flat shape:", x_flat.shape)

    num_features = x.shape[2]
    input_dim = 24 * num_features  # flattened window dim
    print("input_dim:", input_dim)

    # Test conditioner - pass timestamps instead of x
    conditioner = Conditioner()
    conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
    cond = conditioner.get_conditions(timestamps)  # Pass timestamps, not x
    print("Condition shape (day of week):", cond.shape)  # should be [batch_size, 7]
    print("Condition example (first 3):\n", cond[:3])

    ##############################
    # use chosen model
    if model_type == "vae":
        print("Initializing VAE model...")
        model = VAE(input_dim=input_dim)
        log_dir = "./runs/VAE_training"
    else:
        print("Initializing CVAE model (condition = day-of-week)...")
        model = CVAE(input_dim=input_dim, condition_dim=7, latent_dim=10, conditioner=conditioner, coder_type="dictionary", prior_type="standard_dictionary")
        log_dir = "./runs/CVAE_training"

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
        beta=1.0,
        num_samples=1,
        epochs=2,
        validation_freq=1,
        validation_num_samples=1,
        device=device,
        early_stopping=True,
        early_stopping_kwargs={"patience": 3, "min_delta": 0.001},
        save_epoch_freq=10,
        log_dir=log_dir,
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


# now we can run either VAE or CVAE test
if __name__ == "__main__":
    # choose: "vae" or "cvae"
    # run_test(model_type="vae")
    run_test(model_type="cvae")






