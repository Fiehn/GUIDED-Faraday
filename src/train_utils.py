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
import matplotlib.pyplot as plt


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_visualize(
        model_path: str,
        data_config: dict = None,
        output_dir: str = "Output/Plots",
        device: str = "auto",
        num_reconstructions: int = 3,
        num_synthetic_samples: int = 5,
        num_samples_per_original: int = 3,
    ):
    """
    Loads a trained model from disk and generates reconstruction and synthetic data visualizations.
    model_path: Path to the saved model file.
    data_config: Configuration dictionary for the data module (if needed).
    output_dir: Directory to save the output plots.
    device: Device to run the model on ("cpu", "cuda", or "auto").
    num_reconstructions: Number of reconstruction samples per original data point.
    num_synthetic_samples: Number of synthetic samples to generate.
    num_samples_per_original: Number of reconstruction samples per original data point.
    """
    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(model_path, map_location=device)

    model_kwargs = checkpoint.get("model_kwargs", {})
    model_tag = checkpoint.get("model_tag", "loaded_model")

    is_cvae = 'condition_dim' in model_kwargs or 'conditioner' in model_kwargs

    if data_config is None:
        data_config = {
            "data_dir": "data/",
            "filename": "hourly_measurements_2023_10.csv",
            "meter_col": "meter_number",
            "timestamp_dst_col": "timestamp_dst",
            "timestamp_utc_col": "timestamp_utc",
            "feature_cols": ("hourly_energy_consumption",),
            "window": WindowSpec(input_len=24, stride=24),
            "batch_size": 256,
            "num_workers": 0,
            "seed": 42,
            "train_frac": 0.8,
            "val_frac": 0.1,
        }
        normalizer = Normalizer(method='zero_preserved_log') # THIS SHOULD MATCH TRAINING NORMALIZER TODO

        # load data
        dm = SmartMeterDataModule(
            normalizer=normalizer,
            **data_config
        )
        dm.setup()
        test_loader = dm.test_dataloader()

        if is_cvae:
            conditioner = Conditioner()
            conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            
            input_dim = checkpoint['model_state_dict']['encoder.parameterizer.shared_block.0.weight'].shape[1] - conditioner.get_condition_dim()

            model = CVAE(
                input_dim=input_dim,
                condition_dim=conditioner.get_condition_dim(),
                conditioner=conditioner,
                **{k: v for k, v in model_kwargs.items() if k not in ['self', '__class__', 'conditioner']}
            )
        else:
            input_dim = checkpoint['model_state_dict']['encoder.parameterizer.shared_block.0.weight'].shape[1]
            model = VAE(
                input_dim=input_dim,
                **{k: v for k, v in model_kwargs.items() if k not in ['self', '__class__']}
            )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.prior = checkpoint.get('prior', model.prior)
    model.to(device)
    model.prior.to(device)
    model.eval()

    print(f"Model loaded: {model.__class__.__name__} with tag '{model_tag}'")
    
    viz_path = {}

    recon_path = _visualize_reconstructions(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        num_samples=num_reconstructions,
        num_recon_per=num_samples_per_original,
        output_dir=output_dir,
        model_tag=model_tag,
        is_cvae=is_cvae,
    )
    viz_path['reconstructions'] = recon_path
    synth_path = _visualize_synthetic_samples(
        model=model,
        normalizer=normalizer,
        device=device,
        num_samples=num_synthetic_samples,
        output_dir=output_dir,
        model_tag=model_tag,
        is_cvae=is_cvae,
    )
    viz_path['synthetic_samples'] = synth_path
    if hasattr(model, 'get_SigmaMapper'):
        sigma_mapper_path = _visualize_sigma_mapper(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            model_tag=model_tag,
        )
        viz_path.update(sigma_mapper_path)
    
    latent_space_path = _visualize_latent_space(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(latent_space_path)

    summary_path = _save_model_summary(
        model=model,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path['model_summary'] = summary_path

    print("All visualizations saved.")
    print(viz_path)
    return viz_path

def visualize_trained_model(
        model,
        test_loader,
        normalizer,
        device,
        output_dir: str,
        model_tag: str,
        num_reconstructions: int = 3,
        num_synthetic_samples: int = 5,
        num_samples_per_original: int = 3,
    ) -> dict:
    """
    Generates reconstruction and synthetic data visualizations for a trained model.
    model: Trained model instance.
    test_loader: DataLoader for the test dataset.
    normalizer: Normalizer instance used for data normalization.
    device: Device to run the model on.
    output_dir: Directory to save the output plots.
    model_tag: Tag/name for the model (used in filenames).
    num_reconstructions: Number of reconstruction samples per original data point.
    num_synthetic_samples: Number of synthetic samples to generate.
    num_samples_per_original: Number of reconstruction samples per original data point.
    """
    os.makedirs(output_dir, exist_ok=True)
    is_cvae = hasattr(model, 'condition_dim') and model.condition_dim > 0
    viz_path = {}
    recon_path = _visualize_reconstructions(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        num_samples=num_reconstructions,
        num_recon_per=num_samples_per_original,
        output_dir=output_dir,
        model_tag=model_tag,
        is_cvae=is_cvae,
    )
    viz_path['reconstructions'] = recon_path
    synth_path = _visualize_synthetic_samples(
        model=model,
        normalizer=normalizer,
        device=device,
        num_samples=num_synthetic_samples,
        output_dir=output_dir,
        model_tag=model_tag,
        is_cvae=is_cvae,
    )
    viz_path['synthetic_samples'] = synth_path
    if hasattr(model, 'get_SigmaMapper'):
        sigma_mapper_path = _visualize_sigma_mapper(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            model_tag=model_tag,
        )
        viz_path.update(sigma_mapper_path)
    latent_space_path = _visualize_latent_space(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(latent_space_path)
    summary_path = _save_model_summary(
        model=model,
        model_kwargs={},
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path['model_summary'] = summary_path
    print("All visualizations saved.")
    print(viz_path)
    return viz_path


def _visualize_reconstructions(
        model,
        test_loader,
        normalizer,
        device,
        num_samples: int,
        num_recon_per: int,
        output_dir: str,
        model_tag: str,
        is_cvae: bool,
    ) -> str:
    with torch.no_grad():
        x, meter_id, timestamps = next(iter(test_loader))
        x = x.to(device)
        timestamps = timestamps.to(device)
        x_original = x  # This is normalized
        x_flat = x.reshape(x.shape[0], -1)
        if is_cvae:
            recon_params, _ = model.reconstruct(x_flat, timestamps, num_samples=num_samples)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            condition_labels = []
            for ts in timestamps.cpu():
                dt = pd.Timestamp(ts.item(), unit='s')
                day_name = day_names[dt.dayofweek]
                date_str = dt.strftime("%Y-%m-%d")
                condition_labels.append(f"{day_name} ({date_str})")
        else:
            recon_params, _ = model.forward(x_flat, num_samples=num_samples)
            condition_labels = None
        recon_mu = recon_params["mu"]  # [S, B, D]
        S, B, D = recon_mu.shape
        L, F = x_original.shape[1], x_original.shape[2]
        recon_reshaped = recon_mu.reshape(S, B, L, F)  # [num_samples, batch, length, features]
        x_original_denorm = normalizer.inverse_transform(x_original.cpu())
        recon_denorm = normalizer.inverse_transform(recon_reshaped.cpu())

    num_to_plot = min(num_samples, B)
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(14, 5 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    recon_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i in range(num_to_plot):
        # original
        axes[i].plot(
            x_original_denorm[i, :, 0].numpy() if isinstance(x_original_denorm, torch.Tensor) else x_original_denorm[i, :, 0],
            label="Original",
            marker="o",
            color="black",
            alpha=0.9,
            linewidth=2,
            markersize=4
        )
        # reconstructions
        for s in range(num_samples):
            axes[i].plot(
                recon_denorm[s, i, :, 0].numpy() if isinstance(recon_denorm, torch.Tensor) else recon_denorm[s, i, :, 0],
                label=f"Recon {s+1}",
                marker="s",
                color=recon_colors[s % len(recon_colors)],
                alpha=0.6,
                linewidth=1.5,
                markersize=3
            )
        if condition_labels is not None:
            title = f"Sample {i+1} - Energy Consumption | Condition: {condition_labels[i]}"
        else:
            title = f"Sample {i+1} - Energy Consumption"
        axes[i].set_title(title)
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Energy (kWh)")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
    fig_title = f"{'CVAE' if is_cvae else 'VAE'} Reconstruction ({num_samples} samples per original)"
    if is_cvae:
        fig_title += " - Conditioned on Day of Week"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    reconstruction_path = os.path.join(output_dir, f"{model_tag}_reconstruction.png")
    plt.savefig(reconstruction_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reconstruction comparison to {reconstruction_path}")
    return reconstruction_path

def _visualize_synthetic_samples(
        model,
        normalizer,
        device,
        num_samples: int,
        output_dir: str,
        model_tag: str,
        is_cvae: bool,
    ) -> str:
    with torch.no_grad():
        if is_cvae:
            # Generate samples for each day of the week
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            samples_per_day = max(1, num_samples // 7)
            
            # Create timestamps for a full week
            base_date = pd.Timestamp("2023-01-02")  # A Monday
            all_synthetic_data = []
            day_labels = []
            
            for day_idx in range(7):
                target_date = base_date + pd.Timedelta(days=day_idx)
                fake_ts = torch.tensor([target_date.timestamp()] * samples_per_day, dtype=torch.float64)
                fake_conditions = model.conditioner.get_conditions(fake_ts).to(device)
                
                # Sample from model
                samples = model.sample(fake_conditions, num_prior_samples=1, num_x_samples=1, device=device)
                synthetic_flat = samples[0, 0]  # [samples_per_day, input_dim]
                
                all_synthetic_data.append(synthetic_flat)
                day_labels.extend([day_names[day_idx]] * samples_per_day)
            
            # Concatenate all samples
            synthetic_flat = torch.cat(all_synthetic_data, dim=0)  # [total_samples, input_dim]
            total_samples = synthetic_flat.shape[0]
            
            # Reshape to [num_samples, length, features]
            # input_dim = length * features, so we need to figure out the split
            # Assuming data was flattened from [batch, length, features]
            # For hourly data with 24 hours, length=24
            window_len = 24  # This should match your data's window length
            num_features = synthetic_flat.shape[1] // window_len
            synthetic_data = synthetic_flat.reshape(total_samples, window_len, num_features)
            
        else:
            # VAE: sample from prior
            samples_dict = model.sample(num_samples_prior=num_samples, num_samples_posterior=1, device=device)
            synthetic_flat = samples_dict["samples"][:, 0, 0, :]  # [num_samples, input_dim]
            
            # Reshape
            window_len = 24
            num_features = synthetic_flat.shape[1] // window_len
            synthetic_data = synthetic_flat.reshape(num_samples, window_len, num_features)
            day_labels = [None] * num_samples
        
        # Denormalize
        synthetic_data_denorm = normalizer.inverse_transform(synthetic_data.cpu())
    
    # Plot
    num_to_plot = synthetic_data.shape[0]
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(14, 3 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    
    for i in range(num_to_plot):
        axes[i].plot(
            synthetic_data_denorm[i, :, 0].numpy() if isinstance(synthetic_data_denorm, torch.Tensor) else synthetic_data_denorm[i, :, 0],
            label="Synthetic Sample",
            marker="o",
            color="#8e44ad",
            alpha=0.8,
            linewidth=2,
            markersize=4
        )
        
        if day_labels[i] is not None:
            title = f"Synthetic Sample {i+1} | Condition: {day_labels[i]}"
        else:
            title = f"Synthetic Sample {i+1}"
        
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Hour", fontsize=9)
        axes[i].set_ylabel("Energy (kWh)", fontsize=9)
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    fig_title = f"{'CVAE' if is_cvae else 'VAE'} Synthetic Samples"
    if is_cvae:
        fig_title += " - By Day of Week"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    synthetic_path = os.path.join(output_dir, f"{model_tag}_synthetic_samples.png")
    plt.savefig(synthetic_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved synthetic samples to {synthetic_path}")
    return synthetic_path

def _visualize_sigma_mapper(
        model,
        test_loader,
        device,
        output_dir: str,
        model_tag: str,
    ) -> dict:
    sigma_mapper = model.get_SigmaMapper().cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(sigma_mapper, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title("Sigma Mapper Matrix")
    sigma_mapper_path = os.path.join(output_dir, f"{model_tag}_sigma_mapper.png")
    plt.savefig(sigma_mapper_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Sigma Mapper visualization to {sigma_mapper_path}")
    return {"sigma_mapper": sigma_mapper_path}

def _visualize_latent_space(model, test_loader, device, output_dir, model_tag):
    """Visualize latent space via PCA/t-SNE."""
    from sklearn.decomposition import PCA
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for x, meter_id, timestamps in test_loader:
            x = x.to(device)
            x_flat = x.reshape(x.shape[0], -1)
            
            if hasattr(model, 'conditioner'):
                c = model.conditioner.get_conditions(timestamps).to(device)
                enc_in = torch.cat([x_flat, c], dim=-1)
            else:
                enc_in = x_flat
            
            q_params = model.encoder(enc_in)
            z = model.encoder.reparameterize(q_params, num_samples=1).squeeze(0)
            
            latents.append(z.cpu().numpy())
            
            # Extract day of week for coloring
            for ts in timestamps:
                dt = pd.Timestamp(ts.item(), unit='s')
                labels.append(dt.dayofweek)
    
    latents = np.concatenate(latents, axis=0)
    labels = np.array(labels)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    for day_idx in range(7):
        mask = labels == day_idx
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[colors[day_idx]],
            label=day_names[day_idx],
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Latent Space Visualization (PCA)', fontsize=13, fontweight='bold')
    ax.legend(title='Day of Week', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_tag}_latent_space.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)
    
    return {'latent_space': save_path}

def _save_model_summary(
        model,
        model_kwargs,
        output_dir: str,
        model_tag: str,
    ) -> str:
    import json

    summary = {
        "model_tag": model_tag,
        "model_class": model.__class__.__name__,
        "model_kwargs": model_kwargs,
        "architecture": {
            "encoder": model.encoder.__class__.__name__,
            "decoder": model.decoder.__class__.__name__,
            "prior": model.prior.__class__.__name__,
        },
        "dimensions": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "condition_dim": model.condition_dim if hasattr(model, 'condition_dim') else None,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "hyperparameters": {k: v for k, v in model_kwargs.items() if not k.startswith('_')},
    }
    summary_path = os.path.join(output_dir, f"{model_tag}_model_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved model summary to {summary_path}")
    return summary_path


if __name__ == "__main__":
    # Example usage
    model_path = "models/CVAE_DictionaryCoder_20260202_145755_final.pt"
    load_model_and_visualize(
        model_path=model_path,
        output_dir="Output/Plots",
        num_reconstructions=3,
        num_synthetic_samples=5,
        num_samples_per_original=3,
    )