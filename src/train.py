import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.encoder_decoder_model import *
from src.Conditioner import *
from src.Models import *
from src.dataloader import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test Vae
if __name__ == "__main__":
    print("Using device: ", device)

    # Up the recursion limit for deep models
    sys.setrecursionlimit(5000) # default seems to bbe 999

    # load data
    data_dir = "data/"
    filename = "hourly_measurements_2023_10.csv"

    meter_col="meter_number"
    timestamp_dst_col="timestamp_dst"
    timestamp_utc_col="timestamp_utc"
    feature_cols=("hourly_energy_consumption",) # , "hourly_energy_production")

    window=WindowSpec(input_len=24, stride=24) # full day windows

    batch_size=256
    num_workers=0
    shuffle_train=False
    seed=0
    train_frac=0.8
    val_frac=0.1

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
        max_meters= None,
    )

    dm.setup()
    print("done setup")
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    # get one batch
    for x in train_loader:
        print("x shape:", x.shape)
        # Flatten for VAE: [batch_size, 24, 1] -> [batch_size, 24]
        x_flat = x.reshape(x.shape[0], -1)
        print("x_flat shape:", x_flat.shape)
        break
    
    print("Initializing VAE model...")
    VAE_model = VAE(input_dim=x_flat.shape[1])
    
    # Use the .fit() method with all its features
    model_tag = VAE_model.fit(
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
        epochs=20,
        validation_freq=1,
        validation_num_samples=1,
        device=device,
        early_stopping=True,
        early_stopping_kwargs={"patience": 10, "min_delta": 0.001},
        save_epoch_freq=5,
        log_dir='./runs/VAE_training',
    )
    
    print("Training complete. Model saved as ", model_tag)
    
    # Reopen TensorBoard writer for logging visualizations
    log_dir_tagged = os.path.join('./runs/VAE_training', model_tag)
    writer = SummaryWriter(log_dir=log_dir_tagged)
    
    # Load the model for visualization
    VAE_model.to(device)
    VAE_model.prior_params = {key: val.to(device) for key, val in VAE_model.prior_params.items()}
    VAE_model.eval()
    
    # ===== RECONSTRUCTION VISUALIZATION =====
    print("\nGenerating reconstruction visualizations...")
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Reconstruction", leave=False):
            x = x.to(device)
            x_original = x
            x_flat = x.reshape(x.shape[0], -1)
            recon_params_dict, _ = VAE_model.forward(x_flat, num_samples=1)
            recon_mu = recon_params_dict['mu']
            if recon_mu.dim() == 3:
                recon_mu = recon_mu.squeeze(0)
            recon_reshaped = recon_mu.reshape(x_original.shape[0], x_original.shape[1], x_original.shape[2])
            break
    
    # Plot original vs reconstructed
    num_samples_to_plot = min(3, x_original.shape[0])
    fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(14, 4*num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes]
    
    for i in range(num_samples_to_plot):
        axes[i].plot(x_original[i, :, 0].cpu().numpy(), label='Original', marker='o', alpha=0.7, linewidth=2)
        axes[i].plot(recon_reshaped[i, :, 0].cpu().numpy(), label='Reconstructed', marker='s', alpha=0.7, linewidth=2)
        axes[i].set_title(f'Sample {i+1} - Energy Consumption', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Hour', fontsize=10)
        axes[i].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Output/Plots/reconstruction_comparison.png', dpi=150)
    print("Saved reconstruction comparison to Output/Plots/reconstruction_comparison.png")
    plt.close()

    # Save to file
    os.makedirs('Output/Plots', exist_ok=True)
    reconstruction_path = f'Output/Plots/{model_tag}_reconstruction.png'
    plt.savefig(reconstruction_path, dpi=150, bbox_inches='tight')
    print(f"Saved reconstruction comparison to {reconstruction_path}")
    
    # Add to TensorBoard
    writer.add_figure('Validation/Reconstructions', fig, global_step=0)
    plt.close()
    
    #use case:
    # Create 10k synthetic samples to instantiate a new transformer station
    # some users dont have ev's, so we input them and add condition on ev

    # ===== SYNTHETIC SAMPLES VISUALIZATION =====
    print("Generating synthetic samples...")
    with torch.no_grad():
        num_synthetic = 5
        latent_dim = VAE_model.prior_params["mu"].shape[1]
        prior_samples = torch.randn(num_synthetic, latent_dim).to(device) * VAE_model.prior_params["sigma"] + VAE_model.prior_params["mu"]
        synthetic_params = VAE_model.decoder(prior_samples)
        synthetic_mu = synthetic_params['mu']
        if synthetic_mu.dim() == 3:
            synthetic_mu = synthetic_mu.squeeze(0)
        synthetic_data = synthetic_mu.reshape(num_synthetic, x_original.shape[1], x_original.shape[2])
    
    # Plot synthetic data
    fig, axes = plt.subplots(num_synthetic, 1, figsize=(14, 4*num_synthetic))
    if num_synthetic == 1:
        axes = [axes]
    
    for i in range(num_synthetic):
        axes[i].plot(synthetic_data[i, :, 0].cpu().numpy(), marker='o', color='green', alpha=0.7, linewidth=2)
        axes[i].set_title(f'Synthetic Sample {i+1} - Energy Consumption', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Hour', fontsize=10)
        axes[i].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file
    synthetic_path = f'Output/Plots/{model_tag}_synthetic.png'
    plt.savefig(synthetic_path, dpi=150, bbox_inches='tight')
    print(f"Saved synthetic samples to {synthetic_path}")
    
    # Add to TensorBoard
    writer.add_figure('Validation/Synthetic_Samples', fig, global_step=0)
    plt.close()

    # ===== COMPARISON PLOT =====
    print("Generating comparison visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Original samples
    for i in range(min(3, x_original.shape[0])):
        axes[0].plot(x_original[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
    axes[0].set_title('Original Samples', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hour', fontsize=10)
    axes[0].set_ylabel('Energy (kWh)', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstructed samples
    for i in range(min(3, recon_reshaped.shape[0])):
        axes[1].plot(recon_reshaped[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}')
    axes[1].set_title('Reconstructed Samples', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour', fontsize=10)
    axes[1].set_ylabel('Energy (kWh)', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Synthetic samples
    for i in range(min(3, synthetic_data.shape[0])):
        axes[2].plot(synthetic_data[i, :, 0].cpu().numpy(), alpha=0.6, label=f'Sample {i+1}', color='green')
    axes[2].set_title('Synthetic Samples', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Hour', fontsize=10)
    axes[2].set_ylabel('Energy (kWh)', fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Save comparison
    comparison_path = f'Output/Plots/{model_tag}_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {comparison_path}")
    
    # Add to TensorBoard
    writer.add_figure('Validation/Comparison', fig, global_step=0)
    plt.close()
    
    # ===== STATISTICS =====
    # Log sample statistics to TensorBoard
    writer.add_scalar('Validation/Original_Mean', x_original.mean().item(), 0)
    writer.add_scalar('Validation/Original_Std', x_original.std().item(), 0)
    writer.add_scalar('Validation/Reconstructed_Mean', recon_reshaped.mean().item(), 0)
    writer.add_scalar('Validation/Reconstructed_Std', recon_reshaped.std().item(), 0)
    writer.add_scalar('Validation/Synthetic_Mean', synthetic_data.mean().item(), 0)
    writer.add_scalar('Validation/Synthetic_Std', synthetic_data.std().item(), 0)
    
    # Reconstruction error
    mse = ((x_original - recon_reshaped) ** 2).mean().item()
    mae = (x_original - recon_reshaped).abs().mean().item()
    writer.add_scalar('Validation/MSE', mse, 0)
    writer.add_scalar('Validation/MAE', mae, 0)
    
    print(f"\nValidation Metrics:")
    print(f"  Reconstruction MSE: {mse:.4f}")
    print(f"  Reconstruction MAE: {mae:.4f}")
    
    # Close writer
    writer.close()
    
    print("\nValidation and visualization complete.")
    print(f"View TensorBoard logs with: tensorboard --logdir=./runs/VAE_training/{model_tag}")