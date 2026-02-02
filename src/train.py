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
        max_meters=None, # None # tried smaller number for CVAE
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
        epochs=50,
        validation_freq=1,
        validation_num_samples=1,
        device=device,
        early_stopping=True,
        early_stopping_kwargs={"patience": 10, "min_delta": 0.001},
        save_epoch_freq=10,
        log_dir=log_dir,
    )

    print("Training complete. Model saved as ", model_tag)

    ##############################
    # Reopen TensorBoard writer for logging visualizations
    log_dir_tagged = os.path.join(log_dir, model_tag)
    writer = SummaryWriter(log_dir=log_dir_tagged)

    # Make sure output folder exists BEFORE saving figs
    os.makedirs("Output/Plots", exist_ok=True)

    # Move model
    model.to(device)
    model.prior_params = {k: v.to(device) for k, v in model.prior_params.items()}
    model.eval()

    ##############################
    # Reconstruction
    print("\nGenerating reconstruction visualizations...")
    num_recon_samples = 3  # Number of reconstructions per original
    
    with torch.no_grad():
        x, meter_id, timestamps = next(iter(test_loader))
        x = x.to(device)
        timestamps = timestamps.to(device)
        x_original = x  # This is normalized
        x_flat = x.reshape(x.shape[0], -1)

        if model_type == "vae":
            recon_params, _ = model.forward(x_flat, num_samples=num_recon_samples)
            recon_mu = recon_params["mu"]  # [S, B, D]
            condition_labels = None
        else:
            # CVAE: use reconstruct method which handles conditioning internally
            recon_params, _ = model.reconstruct(x_flat, timestamps, num_samples=num_recon_samples)
            recon_mu = recon_params["mu"]  # [S, B, D]
            
            # Extract day-of-week labels for display
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            condition_labels = []
            for ts in timestamps.cpu():
                dt = pd.Timestamp(ts.item(), unit='s')
                day_name = day_names[dt.dayofweek]
                date_str = dt.strftime("%Y-%m-%d")
                condition_labels.append(f"{day_name} ({date_str})")

        # recon_mu shape: [S, B, D] -> reshape to [S, B, L, F]
        S, B, D = recon_mu.shape
        L, F = x_original.shape[1], x_original.shape[2]
        recon_reshaped = recon_mu.reshape(S, B, L, F)  # [num_samples, batch, length, features]
        
        # Inverse transform for plotting in original scale
        # Move to CPU first, then inverse transform (returns torch tensor)
        x_original_denorm = normalizer.inverse_transform(x_original.cpu())
        recon_denorm = normalizer.inverse_transform(recon_reshaped.cpu())

    # Plot original vs multiple reconstructions
    num_samples_to_plot = min(3, x_original.shape[0])
    fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(14, 5 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes]

    # Color palette for reconstructions
    recon_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i in range(num_samples_to_plot):
        # x_original_denorm is a torch.Tensor, convert to numpy for plotting
        axes[i].plot(
            x_original_denorm[i, :, 0].numpy() if isinstance(x_original_denorm, torch.Tensor) else x_original_denorm[i, :, 0],
            label="Original", 
            marker="o", 
            color="black",
            alpha=0.9, 
            linewidth=2,
            markersize=4
        )
        
        # Plot each reconstruction sample
        for s in range(num_recon_samples):
            axes[i].plot(
                recon_denorm[s, i, :, 0].numpy() if isinstance(recon_denorm, torch.Tensor) else recon_denorm[s, i, :, 0],
                label=f"Recon {s+1}", 
                marker="s", 
                color=recon_colors[s % len(recon_colors)],
                alpha=0.6, 
                linewidth=1.5,
                markersize=3
            )
        
        # Build title with condition info
        if condition_labels is not None:
            title = f"Sample {i+1} - Energy Consumption | Condition: {condition_labels[i]}"
        else:
            title = f"Sample {i+1} - Energy Consumption"
        
        axes[i].set_title(title)
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    # Add overall figure title for model type
    fig_title = f"{'CVAE' if model_type == 'cvae' else 'VAE'} Reconstruction ({num_recon_samples} samples per original)"
    if model_type == "cvae":
        fig_title += " - Conditioned on Day of Week"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    reconstruction_path = f"Output/Plots/{model_tag}_reconstruction.png"
    plt.savefig(reconstruction_path, dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction comparison to {reconstruction_path}")
    writer.add_figure("Validation/Reconstructions", fig, global_step=0)
    plt.close(fig)


    ##############################
    # Synthetic sampling
    print("Generating synthetic samples...")
    with torch.no_grad():
        num_synthetic = 5

        if model_type == "vae":
            latent_dim = model.prior_params["mu"].shape[1]
            prior_samples = torch.randn(num_synthetic, latent_dim, device=device) * model.prior_params["sigma"] + model.prior_params["mu"]
            synthetic_params = model.decoder(prior_samples)
            synthetic_mu = synthetic_params["mu"]
            if synthetic_mu.dim() == 3:
                synthetic_mu = synthetic_mu.squeeze(0)
            synthetic_data = synthetic_mu.reshape(num_synthetic, x_original.shape[1], x_original.shape[2])
            synthetic_condition_label = None

        else:  # CVAE
            monday_ts = pd.Timestamp("2023-01-02 12:00:00").timestamp()
            fake_timestamps = torch.tensor([monday_ts] * num_synthetic, dtype=torch.float64)
            fake_conditions = model.conditioner.get_conditions(fake_timestamps)
            fake_conditions = fake_conditions.to(device)
            samples = model.sample(fake_conditions, num_prior_samples=1, num_x_samples=1, device=device)
            synthetic_flat = samples[0, 0]
            synthetic_data = synthetic_flat.reshape(num_synthetic, x_original.shape[1], x_original.shape[2])
            synthetic_condition_label = "Monday"

        # Denormalize synthetic data
        synthetic_data_denorm = normalizer.inverse_transform(synthetic_data.cpu())

    # Plot synthetic data
    fig, axes = plt.subplots(num_synthetic, 1, figsize=(14, 4 * num_synthetic))
    if num_synthetic == 1:
        axes = [axes]

    for i in range(num_synthetic):
        axes[i].plot(
            synthetic_data_denorm[i, :, 0].numpy() if isinstance(synthetic_data_denorm, torch.Tensor) else synthetic_data_denorm[i, :, 0],
            marker="o", 
            alpha=0.7, 
            linewidth=2
        )
        
        # Build title with condition info
        if synthetic_condition_label is not None:
            title = f"Synthetic Sample {i+1} | Condition: {synthetic_condition_label}"
        else:
            title = f"Synthetic Sample {i+1} - Energy Consumption"
        
        axes[i].set_title(title)
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].grid(True, alpha=0.3)

    # Add overall figure title
    fig_title = f"{'CVAE' if model_type == 'cvae' else 'VAE'} Synthetic Samples"
    if model_type == "cvae" and synthetic_condition_label:
        fig_title += f" (Conditioned on {synthetic_condition_label})"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save to file
    synthetic_path = f"Output/Plots/{model_tag}_synthetic.png" 
    plt.savefig(synthetic_path, dpi=150, bbox_inches="tight")
    print(f"Saved synthetic samples to {synthetic_path}")

    # Get the sigma mapper and plot it as a heatmap
    sigma_mapper = model.get_SigmaMapper().cpu().detach().numpy()  # shape: [output_dim, dictionary_size]
    if sigma_mapper is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(sigma_mapper, aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.xlabel('Dictionary Components')
        plt.ylabel('Output Dimensions')
        plt.title(f'{model_type.upper()} Sigma Mapper Heatmap')
        sigma_mapper_path = f'Output/Plots/{model_tag}_sigma_mapper_heatmap.png'
        plt.savefig(sigma_mapper_path, dpi=150, bbox_inches='tight')
        print(f"Saved sigma mapper heatmap to {sigma_mapper_path}")
        plt.close()    

    # Add to TensorBoard
    writer.add_figure("Validation/Synthetic_Samples", fig, global_step=0)
    plt.close(fig)

    writer.close()
    print("\nDone.")
    print(f"TensorBoard: tensorboard --logdir={log_dir}/{model_tag}")


# now we can run either VAE or CVAE test
if __name__ == "__main__":
    # choose: "vae" or "cvae"
    # run_test(model_type="vae")
    run_test(model_type="cvae")






