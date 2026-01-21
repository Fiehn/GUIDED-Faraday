import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.encoder_decoder_model import *
from src.Conditioner import *
from src.Models import *
from src.dataloader import *

# test Vae
if __name__ == "__main__":
    # load data
    data_dir = "data/"
    filename = "hourly_measurements_2023_12.csv"

    meter_col="meter_number"
    timestamp_dst_col="timestamp_dst"
    timestamp_utc_col="timestamp_utc"
    feature_cols=("hourly_energy_consumption", "hourly_energy_production")

    window=WindowSpec(input_len=24, stride=1)

    batch_size=24
    num_workers=0
    shuffle_train=False
    seed=0
    train_frac=0.8
    val_frac=0.1

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
        max_meters=10,
    )

    dm.setup()
    print("done setup")
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    test_loader  = dm.test_dataloader()

    # get one batch
    for x in train_loader:
        print("x shape:", x.shape)  # should be (batch_size, input_len, num_features)
        break
    
    print("Initializing VAE model...")
    VAE_model = VAE(input_dim=48)  # 24 timesteps * 2 features = 48
    optim = torch.optim.Adam(VAE_model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 20
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, x in enumerate(train_loader):
            # Flatten the input: [batch_size, 24, 2] -> [batch_size, 48]
            x_flat = x.reshape(x.shape[0], -1)
            loss_dict = VAE_model.train_core(x_flat, optim)
            epoch_loss += loss_dict["loss"].item()
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete.")
    VAE_model.eval()
    
    # Validation
    val_loss_dict = {}
    with torch.no_grad():
        for x in val_loader:
            x_flat = x.reshape(x.shape[0], -1)
            recon_params_dict, posterior_params_dict = VAE_model.forward(x_flat, num_samples=1)
            loss_dict = VAE_model.loss(x_flat, recon_params_dict, posterior_params_dict, beta=1.0, prior_params=VAE_model.prior_params)
            # Accumulate validation loss if needed
            for key, value in loss_dict.items():
                if key not in val_loss_dict:
                    val_loss_dict[key] = 0.0
                val_loss_dict[key] += value.item()
    
    for key in val_loss_dict:
        val_loss_dict[key] /= len(val_loader)
        print(f"Validation {key}: {val_loss_dict[key]:.4f}")
    
    # ===== VISUALIZATIONS =====
    import matplotlib.pyplot as plt
    
    # 1. Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), epoch_losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('VAE Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    print("Saved training loss plot to training_loss.png")
    plt.close()
    
    # 2. Get one batch of test data for reconstruction visualization
    with torch.no_grad():
        for x in test_loader:
            x_original = x  # [batch_size, 24, 2]
            x_flat = x.reshape(x.shape[0], -1)
            recon_params_dict, _ = VAE_model.forward(x_flat, num_samples=1)
            # Handle the num_samples dimension if present
            recon_mu = recon_params_dict['mu']
            if recon_mu.dim() == 3:  # [num_samples, batch_size, input_dim]
                recon_mu = recon_mu.squeeze(0)  # [batch_size, input_dim]
            # Reshape back to [batch_size, 24, 2]
            recon_reshaped = recon_mu.reshape(x_original.shape[0], 24, 2)
            break
    
    # 3. Plot original vs reconstructed for first 3 samples
    num_samples_to_plot = min(3, x_original.shape[0])
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(14, 4*num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples_to_plot):
        # Plot consumption
        axes[i, 0].plot(x_original[i, :, 0].cpu().numpy(), label='Original', marker='o', alpha=0.7)
        axes[i, 0].plot(recon_reshaped[i, :, 0].cpu().numpy(), label='Reconstructed', marker='s', alpha=0.7)
        axes[i, 0].set_title(f'Sample {i+1} - Consumption', fontsize=12)
        axes[i, 0].set_xlabel('Hour', fontsize=10)
        axes[i, 0].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot production
        axes[i, 1].plot(x_original[i, :, 1].cpu().numpy(), label='Original', marker='o', alpha=0.7)
        axes[i, 1].plot(recon_reshaped[i, :, 1].cpu().numpy(), label='Reconstructed', marker='s', alpha=0.7)
        axes[i, 1].set_title(f'Sample {i+1} - Production', fontsize=12)
        axes[i, 1].set_xlabel('Hour', fontsize=10)
        axes[i, 1].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150)
    print("Saved reconstruction comparison to reconstruction_comparison.png")
    plt.close()
    
    # 4. Generate synthetic samples from VAE (sample from prior)
    with torch.no_grad():
        num_synthetic = 5
        # Sample from prior in latent space
        prior_samples = torch.randn(num_synthetic, 10) * VAE_model.prior_params["sigma"] + VAE_model.prior_params["mu"]
        # Decode to get synthetic data
        synthetic_params = VAE_model.decoder(prior_samples)
        synthetic_mu = synthetic_params['mu']
        if synthetic_mu.dim() == 3:
            synthetic_mu = synthetic_mu.squeeze(0)
        # Reshape to [num_synthetic, 24, 2]
        synthetic_data = synthetic_mu.reshape(num_synthetic, 24, 2)
    
    # 5. Plot synthetic data
    fig, axes = plt.subplots(num_synthetic, 2, figsize=(14, 4*num_synthetic))
    if num_synthetic == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_synthetic):
        # Plot consumption
        axes[i, 0].plot(synthetic_data[i, :, 0].cpu().numpy(), marker='o', color='green', alpha=0.7)
        axes[i, 0].set_title(f'Synthetic Sample {i+1} - Consumption', fontsize=12)
        axes[i, 0].set_xlabel('Hour', fontsize=10)
        axes[i, 0].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot production
        axes[i, 1].plot(synthetic_data[i, :, 1].cpu().numpy(), marker='o', color='orange', alpha=0.7)
        axes[i, 1].set_title(f'Synthetic Sample {i+1} - Production', fontsize=12)
        axes[i, 1].set_xlabel('Hour', fontsize=10)
        axes[i, 1].set_ylabel('Energy (kWh)', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthetic_samples.png', dpi=150)
    print("Saved synthetic samples to synthetic_samples.png")
    plt.close()
    
    # 6. Compare one real meter vs one synthetic sample side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Real data
    axes[0].plot(x_original[0, :, 0].cpu().numpy(), label='Consumption', marker='o', alpha=0.7)
    axes[0].plot(x_original[0, :, 1].cpu().numpy(), label='Production', marker='s', alpha=0.7)
    axes[0].set_title('Real Meter Data (24 hours)', fontsize=14)
    axes[0].set_xlabel('Hour', fontsize=12)
    axes[0].set_ylabel('Energy (kWh)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Synthetic data
    axes[1].plot(synthetic_data[0, :, 0].cpu().numpy(), label='Consumption', marker='o', alpha=0.7)
    axes[1].plot(synthetic_data[0, :, 1].cpu().numpy(), label='Production', marker='s', alpha=0.7)
    axes[1].set_title('VAE Generated Synthetic Data (24 hours)', fontsize=14)
    axes[1].set_xlabel('Hour', fontsize=12)
    axes[1].set_ylabel('Energy (kWh)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_vs_synthetic.png', dpi=150)
    print("Saved real vs synthetic comparison to real_vs_synthetic.png")
    plt.close()

    print("Validation and visualization complete.")
