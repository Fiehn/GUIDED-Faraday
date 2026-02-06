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

    state_dict = checkpoint['model_state_dict']

    # === Infer dimensions from state_dict ===
    # Decoder output dimension
    if 'decoder.SigmaMapper' in state_dict:
        input_dim = state_dict['decoder.SigmaMapper'].shape[0]
    elif 'decoder.parameterizer.param_heads.mu.3.bias' in state_dict:
        input_dim = state_dict['decoder.parameterizer.param_heads.mu.3.bias'].shape[0]
    else:
        raise ValueError("Cannot infer input_dim from state_dict")
    
    # Encoder and decoder input dimensions
    encoder_input_dim = state_dict['encoder.parameterizer.shared_block.0.weight'].shape[1]
    decoder_input_dim = state_dict['decoder.parameterizer.shared_block.0.weight'].shape[1]
    
    # Latent dimension (from model_kwargs or infer)
    latent_dim = model_kwargs.get('latent_dim', 24)
    
    # Detect CVAE: decoder_input_dim = latent_dim + condition_dim
    # For VAE: decoder_input_dim = latent_dim
    is_cvae = (decoder_input_dim > latent_dim)

    if is_cvae:
        condition_dim = decoder_input_dim - latent_dim
        # Verify with encoder: encoder_input_dim should = input_dim + condition_dim
        expected_encoder_input = input_dim + condition_dim
        if encoder_input_dim != expected_encoder_input:
            print(f"Warning: encoder_input_dim ({encoder_input_dim}) != "
                  f"input_dim + condition_dim ({expected_encoder_input})")
    else:
        condition_dim = 0
    
    print(f"Inferred dimensions:")
    print(f"  - input_dim: {input_dim}")
    print(f"  - latent_dim: {latent_dim}")
    print(f"  - condition_dim: {condition_dim}")
    print(f"  - is_cvae: {is_cvae}")

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
        #normalizer = Normalizer(method='zero_preserved_log') # THIS SHOULD MATCH TRAINING NORMALIZER TODO
        normalizer = Normalizer(method='log1p') # THIS SHOULD MATCH TRAINING NORMALIZER TODO
        # load data
        dm = SmartMeterDataModule(
            normalizer=normalizer,
            **data_config
        )
        dm.setup()
        test_loader = dm.test_dataloader()

        if is_cvae:
            print("\nCreating CVAE model...")
        
            # Recreate conditioner (must match training!)
            conditioner = Conditioner()
            conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            conditioner.add_condition_item("user_profile", UserProfiler(num_clusters=30, path="models/UserProfiles/user_profiler.pkl", verbose=False))
            condition_dim = conditioner.get_condition_dim()
            
            # For CVAE: encoder_input_dim = input_dim + condition_dim
            # So: input_dim = encoder_input_dim - condition_dim
            input_dim = encoder_input_dim - condition_dim
            
            print(f"  - input_dim: {input_dim}")
            print(f"  - condition_dim: {condition_dim}")
            print(f"  - encoder sees: {input_dim} + {condition_dim} = {encoder_input_dim}")
            
            # Extract model creation kwargs
            encoder_type = model_kwargs.get('encoder_type', 'dictionary')
            decoder_type = model_kwargs.get('decoder_type', 'dictionary')
            prior_type = model_kwargs.get('prior_type', 'standard_dictionary')
            dictionary_size = model_kwargs.get('dictionary_size', 100)
            
            model = CVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                conditioner=conditioner,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                prior_type=prior_type,
                dictionary_size=dictionary_size,
                # Pass other kwargs but exclude specific ones
                **{k: v for k, v in model_kwargs.items() 
                if k not in ['self', '__class__', 'input_dim', 'latent_dim', 
                            'conditioner', 'condition_dim', 'encoder_type','decoder_type', 'prior_type']}
            )
        else:
            print("\nCreating VAE model...")
        
            # For VAE: encoder_input_dim = input_dim (no conditions)
            input_dim = encoder_input_dim
            
            print(f"  - input_dim: {input_dim}")
            
            encoder_type = model_kwargs.get('encoder_type', 'gaussian')
            decoder_type = model_kwargs.get('decoder_type', 'gaussian')
            prior_type = model_kwargs.get('prior_type', 'standard_gaussian')
            dictionary_size = model_kwargs.get('dictionary_size', 100)
            
            model = VAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                prior_type=prior_type,
                dictionary_size=dictionary_size,
                **{k: v for k, v in model_kwargs.items() 
                if k not in ['self', '__class__', 'input_dim', 'latent_dim', 
                            'encoder_type','decoder_type' 'prior_type']}
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

    user_dict_path = _visualize_user_dictionary(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(user_dict_path)
    synth_recreation_path = _visualize_synthetic_recreations(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(synth_recreation_path)
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
    synth_recreation_path = _visualize_synthetic_recreations(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(synth_recreation_path)
    
    
    user_profile_path = _visualize_user_dictionary(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_tag=model_tag,
    )
    viz_path.update(user_profile_path)

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
            recon_params, _ = model.reconstruct(x_flat,meter_ids=meter_id, timestamps=timestamps, num_samples=num_samples)
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
        if normalizer is not None:
            x_original_denorm = normalizer.inverse_transform(x_original.cpu())
            recon_denorm = normalizer.inverse_transform(recon_reshaped.cpu())
        else:
            x_original_denorm = x_original.cpu()
            recon_denorm = recon_reshaped.cpu()

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
            all_synthetic_data = []
            day_labels = []
            
            fake_conditions = model.conditioner.get_random_conditions(batch_size=samples_per_day).to(device)
            
            # Extract day of week from one-hot encoded conditions
            # Assuming conditions are [batch_size, 7] one-hot vectors
            if fake_conditions.shape[1] == 7:  # day-of-week only
                day_indices = fake_conditions.argmax(dim=1).cpu().numpy()
                day_labels = [day_names[idx] for idx in day_indices]
            else:
                # If multiple conditions concatenated, extract first 7 dims
                dow_part = fake_conditions[:, :7]
                day_indices = dow_part.argmax(dim=1).cpu().numpy()
                day_labels = [day_names[idx] for idx in day_indices]


            # Sample from model
            samples = model.sample(fake_conditions, num_prior_samples=1, num_x_samples=1, device=device)
            synthetic_flat = samples[0, 0]  # [samples_per_day, input_dim]
                
            all_synthetic_data.append(synthetic_flat)
            
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
        if normalizer is not None:
            synthetic_data_denorm = normalizer.inverse_transform(synthetic_data.cpu())
        else:
            synthetic_data_denorm = synthetic_data.cpu()
    
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
    
    sigma_mapper = model.get_SigmaMapper()
    if sigma_mapper is None:
        print("Model does not have a Sigma Mapper, skipping visualization.")
        return {}
    sigma_mapper = sigma_mapper.cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(sigma_mapper, cmap='RdBu_r', vmin=-np.abs(sigma_mapper).max(), vmax=np.abs(sigma_mapper).max())
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
                c = model.conditioner.get_conditions(x,meter_id,timestamps).to(device)
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

def _visualize_synthetic_recreations(model, test_loader, normalizer, device, output_dir, model_tag):
    """
    Pick a few test samples, generate multiple synthetic samples with the same conditions, 
    and plot them together. Also plot the original for comparison and a mean synthetic 
    recreation to see how well the model captures the original.
    """
    if not hasattr(model, 'conditioner'):
        print("Model does not have a conditioner, skipping synthetic recreation visualization.")
        return {}
    if "day_of_week" not in model.conditioner.get_condition_tags():
        print("Model conditioner does not include 'day_of_week', skipping synthetic recreation visualization.")
        return {}
    if "user_profile" not in model.conditioner.get_condition_tags():
        print("Model conditioner does not include 'user_profile', skipping synthetic recreation visualization.")
        return {}
    
    model.eval()
    num_originals = 4  # Number of original samples to visualize
    num_synthetic_per = 50  # Number of synthetic samples per original
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    with torch.no_grad():
        # Get a batch from test loader
        x, meter_id, timestamps = next(iter(test_loader))
        x = x.to(device)
        timestamps = timestamps.to(device)
        
        # Select a few samples
        num_originals = min(num_originals, x.shape[0])
        x_selected = x[:num_originals]
        meter_id_selected = meter_id[:num_originals]
        timestamps_selected = timestamps[:num_originals]
        
        # Get conditions for selected samples
        c = model.conditioner.get_conditions(
            x_selected, 
            meter_id=meter_id_selected, 
            timestamps=timestamps_selected
        )
        c = model.move_to_device(c, device=device)
        
        # Generate synthetic samples with the SAME conditions
        # Shape: [num_prior_samples, num_x_samples, B, D]
        synthetic_samples = model.sample(
            conditions=c,
            num_prior_samples=num_synthetic_per,
            num_x_samples=1,
            device=device
        )
        
        # Reshape: [num_synthetic_per, 1, num_originals, D] -> [num_synthetic_per, num_originals, D]
        synthetic_samples = synthetic_samples.squeeze(1)  # [num_synthetic_per, num_originals, D]
        
        # Reshape to [num_synthetic_per, num_originals, L, F]
        L, F = x_selected.shape[1], x_selected.shape[2]
        synthetic_reshaped = synthetic_samples.reshape(num_synthetic_per, num_originals, L, F)
        
        # Compute mean synthetic sample
        mean_synthetic = synthetic_reshaped.mean(dim=0)  # [num_originals, L, F]
        
        # Denormalize
        if normalizer is not None:
            x_denorm = normalizer.inverse_transform(x_selected.cpu())
            synthetic_denorm = normalizer.inverse_transform(synthetic_reshaped.cpu())
            mean_synthetic_denorm = normalizer.inverse_transform(mean_synthetic.cpu())
        else:
            x_denorm = x_selected.cpu()
            synthetic_denorm = synthetic_reshaped.cpu()
            mean_synthetic_denorm = mean_synthetic.cpu()
        
        # Get condition labels
        condition_labels = []
        for i in range(num_originals):
            ts = timestamps_selected[i].cpu().item()
            dt = pd.Timestamp(ts, unit='s')
            day_name = day_names[dt.dayofweek]
            date_str = dt.strftime("%Y-%m-%d")
            m_id = meter_id_selected[i].item()
            condition_labels.append(f"Meter {m_id} | {day_name} ({date_str})")
    
    # Create figure
    fig, axes = plt.subplots(num_originals, 1, figsize=(14, 5 * num_originals))
    if num_originals == 1:
        axes = [axes]
    
    hours = np.arange(L)
    synthetic_colors = plt.cm.Blues(np.linspace(0.3, 0.7, num_synthetic_per))
    
    for i in range(num_originals):
        ax = axes[i]
        
        # Plot synthetic samples (thin, semi-transparent)
        for s in range(num_synthetic_per):
            synth_data = synthetic_denorm[s, i, :, 0]
            if isinstance(synth_data, torch.Tensor):
                synth_data = synth_data.numpy()
            ax.plot(
                hours, synth_data,
                color=synthetic_colors[s],
                alpha=0.4,
                linewidth=1,
                label='Synthetic' if s == 0 else None
            )
        
        # Plot mean synthetic (dashed, prominent)
        mean_data = mean_synthetic_denorm[i, :, 0]
        if isinstance(mean_data, torch.Tensor):
            mean_data = mean_data.numpy()
        ax.plot(
            hours, mean_data,
            color='#e74c3c',
            linewidth=2.5,
            linestyle='--',
            label='Mean Synthetic',
            marker='s',
            markersize=4,
            alpha=0.9
        )
        
        # Plot original (solid, on top)
        orig_data = x_denorm[i, :, 0]
        if isinstance(orig_data, torch.Tensor):
            orig_data = orig_data.numpy()
        ax.plot(
            hours, orig_data,
            color='black',
            linewidth=2.5,
            label='Original',
            marker='o',
            markersize=5,
            alpha=1.0
        )
        
        # Compute reconstruction error (RMSE between original and mean synthetic)
        rmse = np.sqrt(np.mean((orig_data - mean_data) ** 2))
        
        ax.set_title(
            f"{condition_labels[i]} | RMSE: {rmse:.3f} kWh",
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("Hour", fontsize=10)
        ax.set_ylabel("Energy Consumption (kWh)", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(hours[::2])
        ax.set_xlim(0, L - 1)
    
    fig.suptitle(
        f"Synthetic Recreation with Same Conditions ({num_synthetic_per} samples per original)",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{model_tag}_synthetic_recreations.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved synthetic recreations to {save_path}")
    
    return {"synthetic_recreations": save_path}
    


def _visualize_user_dictionary(model, test_loader, device, output_dir, model_tag):
    """Plot the cached user profiles in the UserProfiler conditioner."""
    if not hasattr(model, 'conditioner'):
        return {}
    if "user_profile" not in model.conditioner.get_condition_tags():
        return {}
    # Access the UserProfiler from condition_items dictionary
    user_profiler: UserProfiler = model.conditioner.condition_items["user_profile"]
    
    # Get profiles from cache
    if not user_profiler.user_profiles_cache:
        print("No user profiles cached, skipping visualization.")
        return {}
    # Convert cache to matrix
    meter_ids = sorted(user_profiler.user_profiles_cache.keys())
    profiles = np.array([user_profiler.user_profiles_cache[m_id] for m_id in meter_ids])
    
    # [num_users, n_topics]
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(profiles, cmap='plasma', aspect='auto')
    fig.colorbar(cax, label='Topic Probability')
    ax.set_title(f"User Profiles (Topic Distributions) - {len(meter_ids)} users", fontsize=13, fontweight='bold')
    ax.set_xlabel("Topic Index", fontsize=11)
    ax.set_ylabel("User (Meter ID)", fontsize=11)
    
    # Optionally add some meter IDs as y-tick labels (sample if too many)
    if len(meter_ids) <= 20:
        ax.set_yticks(range(len(meter_ids)))
        ax.set_yticklabels(meter_ids, fontsize=8)
    
    user_dict_path = os.path.join(output_dir, f"{model_tag}_user_profiles.png")
    plt.tight_layout()
    plt.savefig(user_dict_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved User Profiles visualization to {user_dict_path}")
    return {"user_profiles": user_dict_path}


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
    model_path = "models/CVAE_DictionaryCoder_20260205_203329_best.pt"
    load_model_and_visualize(
        model_path=model_path,
        output_dir="Output/Plots",
        num_reconstructions=3,
        num_synthetic_samples=5,
        num_samples_per_original=3,
    )