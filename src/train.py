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

    batch_size = 32 # 256 # tried smaller batch size for CVAE
    num_workers = 0
    shuffle_train = False
    seed = 2
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
        max_meters=3, # None # tried smaller number for CVAE
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
        model = CVAE(input_dim=input_dim, condition_dim=7, latent_dim=10, conditioner=conditioner)
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
        early_stopping_kwargs={"patience": 10, "min_delta": 0.001},
        save_epoch_freq=5,
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











# # old version which was only for VAE 

# # test Vae
# if __name__ == "__main__":
#     print("Using device: ", device)

#     # Up the recursion limit for deep models
#     sys.setrecursionlimit(5000) # default seems to bbe 999

#     # load data
#     data_dir = "data/"
#     filename = "hourly_measurements_2023_10.csv"

#     meter_col="meter_number"
#     timestamp_dst_col="timestamp_dst"
#     timestamp_utc_col="timestamp_utc"
#     feature_cols=("hourly_energy_consumption",) # , "hourly_energy_production")

#     window=WindowSpec(input_len=24, stride=24) # full day windows

#     batch_size=32 # 256
#     num_workers=0
#     shuffle_train=False
#     seed=0
#     train_frac=0.8
#     val_frac=0.1

#     print("Initializing DataModule...")
#     dm = SmartMeterDataModule(
#         data_dir=data_dir,
#         filename=filename,
#         meter_col=meter_col,
#         timestamp_dst_col=timestamp_dst_col,
#         timestamp_utc_col=timestamp_utc_col,
#         feature_cols=feature_cols,
#         window=window,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle_train=shuffle_train,
#         seed=seed,
#         train_frac=train_frac,
#         val_frac=val_frac,
#         max_meters=3, # None
#     )

#     dm.setup()
#     print("done setup")
#     train_loader = dm.train_dataloader()
#     val_loader   = dm.val_dataloader()
#     test_loader  = dm.test_dataloader()

#     # get one batch
#     for x, meter_id, dow in train_loader: 
#         print("x shape:", x.shape)
#         print("meter_id shape:", meter_id.shape, "example:", meter_id[:5])
#         print("dow shape:", dow.shape, "example:", dow[:10])
#         # Flatten for VAE: [batch_size, 24, 1] -> [batch_size, 24]
#         x_flat = x.reshape(x.shape[0], -1)
#         print("x_flat shape:", x_flat.shape) # [B, 24]
#         break
    
#     print("Initializing VAE model...")
#     VAE_model = VAE(input_dim=x_flat.shape[1])
    
#     # Use the .fit() method with all its features
#     model_tag = VAE_model.fit(
#         trainloader=train_loader,
#         valloader=val_loader,
#         lr=1e-3,
#         weight_decay=1e-4,
#         gradient_clip=1.0,
#         gradient_clip_kwargs={"max_norm": 1.0},
#         lr_scheduler=True,
#         lr_scheduler_kwargs={"step_size": 10, "gamma": 0.9},
#         beta=1.0,
#         num_samples=1,
#         epochs=2, # 20
#         validation_freq=1,
#         validation_num_samples=1,
#         device=device,
#         early_stopping=True,
#         early_stopping_kwargs={"patience": 10, "min_delta": 0.001},
#         save_epoch_freq=5,
#         log_dir='./runs/VAE_training',
#     )
    
#     print("Training complete. Model saved as ", model_tag)
    
#     # Reopen TensorBoard writer for logging visualizations
#     log_dir_tagged = os.path.join('./runs/VAE_training', model_tag)
#     writer = SummaryWriter(log_dir=log_dir_tagged)
    
#     # Load the model for visualization
#     VAE_model.to(device)
#     VAE_model.prior_params = {key: val.to(device) for key, val in VAE_model.prior_params.items()}
#     VAE_model.eval()
    
#     # ===== RECONSTRUCTION VISUALIZATION =====
#     print("\nGenerating reconstruction visualizations...")
#     with torch.no_grad():
#         for x in tqdm(test_loader, desc="Reconstruction", leave=False):
#             x = x.to(device)
#             x_original = x
#             x_flat = x.reshape(x.shape[0], -1)
#             recon_params_dict, _ = VAE_model.forward(x_flat, num_samples=1)
#             recon_mu = recon_params_dict['mu']
#             if recon_mu.dim() == 3:
#                 recon_mu = recon_mu.squeeze(0)
#             recon_reshaped = recon_mu.reshape(x_original.shape[0], x_original.shape[1], x_original.shape[2])
#             break
    
#     # Plot original vs reconstructed
#     num_samples_to_plot = min(3, x_original.shape[0])
#     fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(14, 4*num_samples_to_plot))
#     if num_samples_to_plot == 1:
#         axes = [axes]
    
#     for i in range(num_samples_to_plot):
#         axes[i].plot(x_original[i, :, 0].cpu().numpy(), label='Original', marker='o', alpha=0.7, linewidth=2)
#         axes[i].plot(recon_reshaped[i, :, 0].cpu().numpy(), label='Reconstructed', marker='s', alpha=0.7, linewidth=2)
#         axes[i].set_title(f'Sample {i+1} - Energy Consumption', fontsize=12, fontweight='bold')
#         axes[i].set_xlabel('Hour', fontsize=10)
#         axes[i].set_ylabel('Energy (kWh)', fontsize=10)
#         axes[i].legend(fontsize=10)
#         axes[i].grid(True, alpha=0.3)
    
#     os.makedirs('Output/Plots', exist_ok=True)
#     plt.tight_layout()
#     plt.savefig('Output/Plots/reconstruction_comparison.png', dpi=150)
#     print("Saved reconstruction comparison to Output/Plots/reconstruction_comparison.png")
#     plt.close()

#     # Save to file
#     reconstruction_path = f'Output/Plots/{model_tag}_reconstruction.png'
#     plt.savefig(reconstruction_path, dpi=150, bbox_inches='tight')
#     print(f"Saved reconstruction comparison to {reconstruction_path}")
    
#     # Add to TensorBoard
#     writer.add_figure('Validation/Reconstructions', fig, global_step=0)
#     plt.close()
    
#     #use case:
#     # Create 10k synthetic samples to instantiate a new transformer station
#     # some users dont have ev's, so we input them and add condition on ev

#     # ===== SYNTHETIC SAMPLES VISUALIZATION =====
#     print("Generating synthetic samples...")
#     with torch.no_grad():
#         num_synthetic = 5
#         latent_dim = VAE_model.prior_params["mu"].shape[1]
#         prior_samples = torch.randn(num_synthetic, latent_dim).to(device) * VAE_model.prior_params["sigma"] + VAE_model.prior_params["mu"]
#         synthetic_params = VAE_model.decoder(prior_samples)
#         synthetic_mu = synthetic_params['mu']
#         if synthetic_mu.dim() == 3:
#             synthetic_mu = synthetic_mu.squeeze(0)
#         synthetic_data = synthetic_mu.reshape(num_synthetic, x_original.shape[1], x_original.shape[2])
    
#     # Plot synthetic data
#     fig, axes = plt.subplots(num_synthetic, 1, figsize=(14, 4*num_synthetic))
#     if num_synthetic == 1:
#         axes = [axes]
    
#     for i in range(num_synthetic):
#         axes[i].plot(synthetic_data[i, :, 0].cpu().numpy(), marker='o', color='green', alpha=0.7, linewidth=2)
#         axes[i].set_title(f'Synthetic Sample {i+1} - Energy Consumption', fontsize=12, fontweight='bold')
#         axes[i].set_xlabel('Hour', fontsize=10)
#         axes[i].set_ylabel('Energy (kWh)', fontsize=10)
#         axes[i].grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save to file
#     synthetic_path = f'Output/Plots/{model_tag}_synthetic.png'
#     plt.savefig(synthetic_path, dpi=150, bbox_inches='tight')
#     print(f"Saved synthetic samples to {synthetic_path}")
    
#     # Add to TensorBoard
#     writer.add_figure('Validation/Synthetic_Samples', fig, global_step=0)
#     plt.close()

#     # ===== COMPARISON PLOT =====
#     print("Generating comparison visualization...")
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # Plot 1: Original samples
#     for i in range(min(3, x_original.shape[0])):
#         axes[0].plot(x_original[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
#     axes[0].set_title('Original Samples', fontsize=12, fontweight='bold')
#     axes[0].set_xlabel('Hour', fontsize=10)
#     axes[0].set_ylabel('Energy (kWh)', fontsize=10)
#     axes[0].legend(fontsize=8)
#     axes[0].grid(True, alpha=0.3)
    
#     # Plot 2: Reconstructed samples
#     for i in range(min(3, recon_reshaped.shape[0])):
#         axes[1].plot(recon_reshaped[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
#     axes[1].set_title('Reconstructed Samples', fontsize=12, fontweight='bold')
#     axes[1].set_xlabel('Hour', fontsize=10)
#     axes[1].set_ylabel('Energy (kWh)', fontsize=10)
#     axes[1].legend(fontsize=8)
#     axes[1].grid(True, alpha=0.3)
    
#     # Plot 3: Synthetic samples
#     for i in range(min(3, synthetic_data.shape[0])):
#         axes[2].plot(synthetic_data[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}', color='green')
#     axes[2].set_title('Synthetic Samples', fontsize=12, fontweight='bold')
#     axes[2].set_xlabel('Hour', fontsize=10)
#     axes[2].set_ylabel('Energy (kWh)', fontsize=10)
#     axes[2].legend(fontsize=8)
#     axes[2].grid(True, alpha=0.3)
    
#     plt.tight_layout()

#     # Save comparison
#     comparison_path = f'Output/Plots/{model_tag}_comparison.png'
#     plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
#     print(f"Saved comparison plot to {comparison_path}")
    
#     # Add to TensorBoard
#     writer.add_figure('Validation/Comparison', fig, global_step=0)
#     plt.close()
    
#     # ===== STATISTICS =====
#     # Log sample statistics to TensorBoard
#     writer.add_scalar('Validation/Original_Mean', x_original.mean().item(), 0)
#     writer.add_scalar('Validation/Original_Std', x_original.std().item(), 0)
#     writer.add_scalar('Validation/Reconstructed_Mean', recon_reshaped.mean().item(), 0)
#     writer.add_scalar('Validation/Reconstructed_Std', recon_reshaped.std().item(), 0)
#     writer.add_scalar('Validation/Synthetic_Mean', synthetic_data.mean().item(), 0)
#     writer.add_scalar('Validation/Synthetic_Std', synthetic_data.std().item(), 0)
    
#     # Reconstruction error
#     mse = ((x_original - recon_reshaped) ** 2).mean().item()
#     mae = (x_original - recon_reshaped).abs().mean().item()
#     writer.add_scalar('Validation/MSE', mse, 0)
#     writer.add_scalar('Validation/MAE', mae, 0)
    
#     print(f"\nValidation Metrics:")
#     print(f"  Reconstruction MSE: {mse:.4f}")
#     print(f"  Reconstruction MAE: {mae:.4f}")
    
#     # Close writer
#     writer.close()
    
#     print("\nValidation and visualization complete.")
#     print(f"View TensorBoard logs with: tensorboard --logdir=./runs/VAE_training/{model_tag}")
    