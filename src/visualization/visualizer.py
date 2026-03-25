from .plot_styling import PlotStyle
from .writer import Writer
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE

from src.model.generators import SyntheticGenerator
from src.data.conditioner import UserProfiler
 
class Visualizer:
    """Class for visualizing training results and model performance."""
    
    def __init__(self, plot_style: PlotStyle, log_dir: str, 
                 model = None, normalizer = None,  writer = None, 
                 conditioner = None,
                 device = "cpu"):
        self.plot_style = plot_style
        self.log_dir = log_dir
        self.model = model
        self.normalizer = normalizer
        self.writer = writer
        self.device = device
        self.generator = SyntheticGenerator(model=model, normalizer=normalizer, conditioner=conditioner, device=device)
    
    def _is_cvae(self) -> bool:
        """Check if model is a CVAE based on presence of conditioner on the model."""
        return hasattr(self.model, 'conditioner') and self.model.conditioner is not None

    def visualize(self, test_loader = None, num_samples: int = 5, 
                  num_recon_per: int = 3, num_synthetic: int = 100,
                  num_originals: int = 4, num_synthetic_per: int = 50):
        """Main method to visualize results for an entire model experiment.
        Args:
            test_loader: The test data loader for reconstruction and latent space visualization.
            num_samples: Number of samples to visualize for reconstructions and synthetic samples.
            num_recon_per: Number of reconstructions to show per original sample.
            num_synthetic: Number of synthetic samples to generate for visualization.
            num_originals: Number of original samples to select for synthetic recreation visualization.
            num_synthetic_per: Number of synthetic samples to generate per original sample for synthetic recreation visualization.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        if self.model is not None:
            self.model.to(self.device)
        try:
            print("Visualizing training progress.")
            self.visualize_training_progress()
        except Exception as e:
            print(f"Error visualizing training progress: {e}")
        
        if test_loader is not None:
            print("Visualizing reconstructions.")
            self.visualize_reconstructions(num_samples=num_samples, num_recon_per=num_recon_per, test_loader=test_loader)
    
        try:
            print("Visualizing synthetic samples.")
            self.visualize_synthetic(num_samples=num_synthetic)
        except Exception as e:
            print(f"Error visualizing synthetic samples: {e}")
        try:
            print("Visualizing sigma mapper.")
            self.visualize_sigma_mapper()
        except Exception as e:
            print(f"Error visualizing sigma mapper: {e}")
        try:
            print("Visualizing latent space.")
            #self.visualize_latent_space(test_loader=test_loader)
        except Exception as e:
            print(f"Error visualizing latent space: {e}")
        try:
            print("Visualizing synthetic recreations.")
            self.visualize_synthetic_recreations(test_loader=test_loader, num_originals=num_originals, num_synthetic_per=num_synthetic_per)
        except Exception as e:
            print(f"Error visualizing synthetic recreations: {e}")
        try:
            print("Visualizing user profiles.")
            self.visualize_user_dict()
        except Exception as e:
            print(f"Error visualizing user profiles: {e}")
        try:
            print("Visualizing PCA comparison.")
            self.pca_compared(test_loader=test_loader)
        except Exception as e:
            print(f"Error visualizing PCA comparison: {e}")
        try:
            print("Visualizing t-SNE comparison.")
            #self.tsne_compared(test_loader=test_loader)
        except Exception as e:
            print(f"Error visualizing t-SNE comparison: {e}")
    
    def visualize_training_progress(self):
        """Visualize training progress using scalars from the writer."""
        if self.writer is None:
            print("No writer available for visualization.")
            return
        
        # Visualize training and validation loss
        train_loss = self.writer.get_scalars("train_loss_epoch")
        val_loss = self.writer.get_scalars("val_loss_epoch")
        print(f"Retrieved {len(train_loss)} training loss entries and {len(val_loss)} validation loss entries from writer.")
        
        if train_loss and val_loss:
            plt.figure(figsize=self.plot_style.get_figsize())
            plt.plot(*zip(*train_loss), label="Train Loss", color=self.plot_style.get_color(0), marker=self.plot_style.marker, linestyle=self.plot_style.linestyle, linewidth=self.plot_style.linewidth)
            plt.plot(*zip(*val_loss), label="Val Loss", color=self.plot_style.get_color(1), marker=self.plot_style.marker, linestyle=self.plot_style.linestyle, linewidth=self.plot_style.linewidth)
            plt.title("Training Progress")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"training_progress.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            print("No loss data available for visualization.")

    def visualize_reconstructions(self, test_loader, num_samples: int,
        num_recon_per: int):
        """Visualize original vs reconstructed samples.
        Args:
            test_loader: The test data loader.
            num_samples: Total number of samples to visualize.
            num_recon_per: Number of reconstructions to show per original sample.
        """
        if self.model is None or self.normalizer is None:
            print("Model and normalizer are required for reconstruction visualization.")
            return
        
        self.model.eval()
        input_data = next(iter(test_loader))
        if len(input_data) == 4:
            x_original, c, meter_id, timestamps = input_data
        else:
            x_original, meter_id, timestamps = input_data
        x_original = x_original.to(self.device)
        
        # If x_original is [B, F, L] (e.g. 3, 96), transpose it to [B, L, F] so it acts consistently
        if x_original.dim() == 3 and x_original.shape[1] < x_original.shape[2]:
            x_original = x_original.transpose(1, 2)

        recon_samples = self.generator.reconstruct(x_original, num_recon_per, meter_id=meter_id, timestamp=timestamps, is_normalized=True)
        
        S, B, D = recon_samples.shape
        L, F = x_original.shape[1], x_original.shape[2]
        recon_reshaped = recon_samples.reshape(S, B, L, F)  # [num_recon_per, batch, length, features]

        if self.normalizer is not None:
            x_original_denorm = self.normalizer.inverse_transform(x_original.cpu())
            recon_denorm = self.normalizer.inverse_transform(recon_reshaped.cpu())
        else:
            x_original_denorm = x_original.cpu()
            recon_denorm = recon_reshaped.cpu()

        num_to_plot = min(num_samples, B)
        condition_labels = None
        fig, axes = plt.subplots(num_to_plot, 1, figsize=self.plot_style.get_figsize(num_to_plot))
        if num_to_plot == 1:
            axes = [axes]
        for i in range(num_to_plot):
            # original
            axes[i].plot(
                x_original_denorm[i, :, 0].detach().cpu().numpy() if isinstance(x_original_denorm, torch.Tensor) else x_original_denorm[i, :, 0],
                label="Original",
                marker=self.plot_style.marker,
                color=self.plot_style.color_original,
                alpha=0.9,
                linewidth=self.plot_style.linewidth,
                linestyle=self.plot_style.linestyle,
                markersize=4
            )
            # reconstructed samples from decoder distribution
            for s in range(num_recon_per):
                axes[i].plot(
                    recon_denorm[s, i, :, 0].detach().cpu().numpy() if isinstance(recon_denorm, torch.Tensor) else recon_denorm[s, i, :, 0],
                    label=f"Recon Sample {s+1}",
                    marker=self.plot_style.marker,
                    color=self.plot_style.get_color(s),
                    alpha=0.6,
                    linewidth=self.plot_style.linewidth,
                    linestyle=self.plot_style.linestyle,
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
        fig_title = f"{'CVAE' if self._is_cvae() else 'VAE'} Reconstruction ({num_recon_per} decoder samples per original)"
        if self._is_cvae():
            fig_title += " - Conditioned on Day of Week"
        fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        reconstruction_path = os.path.join(self.log_dir, f"reconstruction.pdf")
        plt.savefig(reconstruction_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    def visualize_synthetic(self, num_samples: int):
        """Visualize synthetic samples generated by the model."""
        if self.model is None or self.normalizer is None:
            print("Model and normalizer are required for synthetic sample visualization.")
            return
        
        synthetic_data, conditions = self.generator.generate(num_samples=num_samples)

        day_labels = None
        if self._is_cvae() and conditions is not None:
            if "day_of_week" in self.model.conditioner.get_condition_tags():
                day_of_week_idx = self.model.conditioner.get_condition_indexes("day_of_week")
                if day_of_week_idx is not None:
                    day_labels = []
                    for c in conditions.cpu():
                        if c.dim() > 0:
                            day_onehot = c[day_of_week_idx[0]:day_of_week_idx[1]]
                            day_idx = torch.argmax(day_onehot).item()
                            day_name = self.plot_style.get_day_names(short=False)[day_idx]
                            day_labels.append(day_name)
                        else:
                            day_labels.append("Unknown")

        # Plot all samples on a single plot with different colors
        num_to_plot = synthetic_data.shape[0]
        fig, ax = plt.subplots(figsize=self.plot_style.figsize)

        for i in range(num_to_plot):
            label = f"Sample {i+1}"
            if day_labels is not None and len(day_labels) > 0:
                label += f" ({day_labels[min(i, len(day_labels)-1)]})"
            
            ax.plot(
                synthetic_data[i, :, 0].cpu().numpy() if isinstance(synthetic_data, torch.Tensor) else synthetic_data[i, :, 0],
                label=label,
                marker=self.plot_style.marker,
                color=self.plot_style.get_color(i),
                alpha=0.8,
                linewidth=self.plot_style.linewidth,
                linestyle=self.plot_style.linestyle,
                markersize=4
            )
        
        ax.set_xlabel("Hour", fontsize=11)
        ax.set_ylabel("Energy (kWh)", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig_title = f"{'CVAE' if self._is_cvae() else 'VAE'} Synthetic Samples"
        if self._is_cvae():
            fig_title += " - By Day of Week"
        fig.suptitle(fig_title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        synthetic_path = os.path.join(self.log_dir, f"synthetic_samples.pdf")
        plt.savefig(synthetic_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
            
    def visualize_sigma_mapper(self):
        """Visualize the learned sigma mapper function."""
        sigma_mapper = self.model.get_SigmaMapper()
        if sigma_mapper is None:
            print("Model does not have a Sigma Mapper, skipping visualization.")
            return {}
        sigma_mapper = sigma_mapper.cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=self.plot_style.figsize)
        cax = ax.matshow(sigma_mapper, cmap='RdBu_r', vmin=-np.abs(sigma_mapper).max(), vmax=np.abs(sigma_mapper).max())
        fig.colorbar(cax)
        ax.set_title("Sigma Mapper Matrix")
        sigma_mapper_path = os.path.join(self.log_dir, f"sigma_mapper.pdf")
        plt.savefig(sigma_mapper_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def visualize_latent_space(self, test_loader, PCA: bool = True, tSNE: bool = True):
        """Visualize the latent space using t-SNE or PCA."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        latents = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for x, meter_id, timestamps in test_loader:
                x = x.to(self.device)
                x_flat = x.reshape(x.shape[0], -1)
                if self._is_cvae():
                    c = self.model.conditioner.get_conditions(x,meter_id,timestamps).to(self.device)
                    enc_in = torch.cat([x_flat, c], dim=-1)
                else:
                    enc_in = x_flat

                latent = self.model.encoder(enc_in)
                q_params = self.model.encoder(enc_in)
                z = self.model.encoder.reparameterize(q_params, num_samples=1).squeeze(0)
                
                latents.append(z.cpu().numpy())
                if len(latents) >= 1000:  # Limit to 1000 samples for visualization
                    break
        
        latents = np.concatenate(latents, axis=0)
        labels = np.array(labels)

        if PCA:
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
            fig, ax = plt.subplots(figsize=self.plot_style.figsize)
            ax.scatter(
                latents_2d[:, 0],
                latents_2d[:, 1],
                alpha=0.6,
                s=100
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            ax.set_title('Latent Space Visualization (PCA)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"PCA_latent_space.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        if tSNE:
            perplexity = min(30, (latents.shape[0] - 1) // 3)  # Rule of thumb for perplexity
            tsne = TSNE(n_components=2, perplexity=perplexity)
            latents_2d = tsne.fit_transform(latents)
            fig, ax = plt.subplots(figsize=self.plot_style.figsize)
            ax.scatter(
                latents_2d[:, 0],
                latents_2d[:, 1],
                alpha=0.6,
                s=100
            )
            ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
            ax.set_title('Latent Space Visualization (t-SNE)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"tSNE_latent_space.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def visualize_synthetic_recreations(self, test_loader, num_originals: int = 4, num_synthetic_per: int = 50):
        """
        Pick a few test samples, generate multiple synthetic samples with the same conditions, 
        and plot them together. Also plot the original for comparison and a mean synthetic 
        recreation to see how well the model captures the original.
        Arg:
            test_loader: The test data loader.
            num_originals: Number of original samples to select for visualization.
            num_synthetic_per: Number of synthetic samples to generate per original sample.
        """
        if not hasattr(self.model, 'conditioner'):
            print("Model does not have a conditioner, skipping synthetic recreation visualization.")
            return
        if "day_of_week" not in self.model.conditioner.get_condition_tags():
            print("Model conditioner does not include 'day_of_week', skipping synthetic recreation visualization.")
            return
        if "user_profile" not in self.model.conditioner.get_condition_tags():
            print("Model conditioner does not include 'user_profile', skipping synthetic recreation visualization.")
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get a batch from test loader
            input_data = next(iter(test_loader))
            if len(input_data) == 4:
                x, c, meter_id, timestamps = input_data
            else:
                x, meter_id, timestamps = input_data
                if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                    c = self.model.conditioner.get_conditions(x, meter_id, timestamps)
                else:
                    c = None
            x = x.to(self.device)
            c = c.to(self.device)
            timestamps = timestamps.to(self.device)
            
            # Select a few samples
            num_originals = min(num_originals, x.shape[0])
            x_selected = x[:num_originals]
            meter_id_selected = meter_id[:num_originals]
            timestamps_selected = timestamps[:num_originals]
            
            # Get conditions for selected samples
            c = self.model.conditioner.get_conditions(
                x_selected, 
                meter_id=meter_id_selected, 
                timestamps=timestamps_selected
            )
            c = self.model.move_to_device(c, device=self.device)
            
            synthetic_samples, _ = self.generator.generate(num_samples=num_synthetic_per, conditions=c)

            x_selected_tf = x_selected.transpose(1, 2) if x_selected.dim() == 3 and x_selected.shape[1] < x_selected.shape[2] else x_selected
            L, F = x_selected_tf.shape[1], x_selected_tf.shape[2]

            if synthetic_samples.dim() >= 3:
                # [N, L, F]
                synthetic_reshaped = synthetic_samples.reshape(num_synthetic_per, num_originals, L, F)
            else:
                synthetic_reshaped = synthetic_samples.reshape(num_synthetic_per, num_originals, L, F)
            
            # Compute mean synthetic sample
            mean_synthetic = synthetic_reshaped.mean(dim=0)  # [num_originals, L, F]
            
            if self.normalizer is not None:
                x_selected_tf = x_selected.transpose(1, 2) if x_selected.dim() == 3 and x_selected.shape[1] < x_selected.shape[2] else x_selected
                x_denorm = self.normalizer.inverse_transform(x_selected_tf.cpu())
            else:
                x_denorm = x_selected.transpose(1, 2).cpu() if x_selected.dim() == 3 and x_selected.shape[1] < x_selected.shape[2] else x_selected.cpu()
            synthetic_denorm = synthetic_reshaped.cpu()
            mean_synthetic_denorm = mean_synthetic.cpu()
            
            # Get condition labels
            condition_labels = []
            for i in range(num_originals):
                ts = timestamps_selected[i].cpu().item()
                dt = pd.Timestamp(ts, unit='s')
                day_name = self.plot_style.get_day_names(short=False)[dt.dayofweek]
                date_str = dt.strftime("%Y-%m-%d")
                m_id = meter_id_selected[i].item()
                condition_labels.append(f"Meter {m_id} | {day_name} ({date_str})")
        
        # Create figure
        fig, axes = plt.subplots(num_originals, 1, figsize=self.plot_style.get_figsize(num_originals))
        if num_originals == 1:
            axes = [axes]
        
        hours = np.arange(L)
        # Use PALETTE_COLORS for synthetic samples with slight alpha variation
        synthetic_alpha_values = np.linspace(0.3, 0.7, num_synthetic_per)
        
        for i in range(num_originals):
            ax = axes[i]
            
            # Plot synthetic samples (thin, semi-transparent)
            for s in range(num_synthetic_per):
                synth_data = synthetic_denorm[s, i, :, 0]
                if isinstance(synth_data, torch.Tensor):
                    synth_data = synth_data.numpy()
                ax.plot(
                    hours, synth_data,
                    color=self.plot_style.get_color(s),
                    alpha=synthetic_alpha_values[s],
                    linewidth=self.plot_style.linewidth,
                    linestyle=self.plot_style.linestyle,
                    label='Synthetic' if s == 0 else None
                )
            
            # Plot mean synthetic (dashed, prominent)
            mean_data = mean_synthetic_denorm[i, :, 0]
            if isinstance(mean_data, torch.Tensor):
                mean_data = mean_data.numpy()
            ax.plot(
                hours, mean_data,
                color=self.plot_style.get_color(0),
                linewidth=self.plot_style.linewidth,
                linestyle='--',
                label='Mean Synthetic',
                marker=self.plot_style.marker,
                markersize=4,
                alpha=0.95
            )
            
            # Plot original (solid, on top)
            orig_data = x_denorm[i, :, 0]
            if isinstance(orig_data, torch.Tensor):
                orig_data = orig_data.numpy()
            ax.plot(
                hours, orig_data,
                color=self.plot_style.get_color_original(),
                linewidth=self.plot_style.linewidth,
                label='Original',
                marker=self.plot_style.marker,
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
        
        save_path = os.path.join(self.log_dir, f"synthetic_recreations.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def visualize_user_dict(self):
        """Plot the cached user profiles in the UserProfiler conditioner."""
        if not self._is_cvae():
            print("Model is not a CVAE, skipping user profiles visualization.")
            return {}
        if "user_profile" not in self.model.conditioner.get_condition_tags():
            print("Model conditioner does not include 'user_profile', skipping user profiles visualization.")
            return {}
        # Access the UserProfiler from condition_items dictionary
        user_profiler: UserProfiler = self.model.conditioner.condition_items["user_profile"]

        # Get profiles from cache
        if not user_profiler.user_profiles_cache:
            print("No user profiles cached, skipping visualization.")
            return {}
        # Convert cache to matrix
        meter_ids = sorted(user_profiler.user_profiles_cache.keys())
        profiles = np.array([user_profiler.user_profiles_cache[m_id] for m_id in meter_ids])
        
        # [num_users, n_topics]
        fig, ax = plt.subplots(figsize=self.plot_style.figsize)
        cax = ax.matshow(profiles, cmap='RdBu_r', aspect='auto')
        fig.colorbar(cax, label='Topic Probability')
        ax.set_title(f"User Profiles (Topic Distributions) - {len(meter_ids)} users", fontsize=13, fontweight='bold')
        ax.set_xlabel("Topic Index", fontsize=11)
        ax.set_ylabel("User (Meter ID)", fontsize=11)
        
        # Optionally add some meter IDs as y-tick labels (sample if too many)
        if len(meter_ids) <= 20:
            ax.set_yticks(range(len(meter_ids)))
            ax.set_yticklabels(meter_ids, fontsize=8)
        
        user_dict_path = os.path.join(self.log_dir, f"user_profiles.pdf")
        plt.tight_layout()
        plt.savefig(user_dict_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    def pca_compared(self, test_loader):
        """
        Generate syntetic samples and compare with real values in PCA space.
        """
        # Get the conditions of real data:
        input_data = next(iter(test_loader))
        if len(input_data) == 4:
            x, c, meter_id, timestamps = input_data
        else:
            x, meter_id, timestamps = input_data
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                c = self.model.conditioner.get_conditions(x, meter_id, timestamps)
            else:
                c = None
        x = x.to(self.device)
        c = c.to(self.device)
        timestamps = timestamps.to(self.device)

        # Generate synthetic samples with the SAME conditions
        synthetic_samples, _ = self.generator.generate(
            num_samples=x.shape[-1],
            conditions=c
        )

        if self.normalizer is not None:
            x_tf = x.transpose(1, 2) if x.dim() == 3 and x.shape[1] < x.shape[2] else x
            x_denorm = self.normalizer.inverse_transform(x_tf.cpu())
        else:
            x_tf = x.transpose(1, 2) if x.dim() == 3 and x.shape[1] < x.shape[2] else x
            x_denorm = x_tf.cpu().numpy()

        if isinstance(x_denorm, torch.Tensor):
            x_denorm = x_denorm.numpy()
        if isinstance(synthetic_samples, torch.Tensor):
            synthetic_samples = synthetic_samples.numpy()

        x_denorm_flat = x_denorm.reshape(x_denorm.shape[0], -1)
        synthetic_flat = synthetic_samples.reshape(synthetic_samples.shape[0], -1)

        # Reshape synthetic samples to match original data shape
        num_samples = synthetic_samples.shape[0]
        pca = PCA(n_components=min(3, x_denorm_flat.shape[1], synthetic_flat.shape[1]))
        pca_real = pca.fit_transform(x_denorm_flat)
        pca_synthetic = pca.transform(synthetic_flat)

        fig = plt.figure(figsize=(16, 6))
        
        # Real data
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pca_real[:, 0], pca_real[:, 1], pca_real[:, 2], 
                   alpha=0.5, label='Real', s=20, c='blue')
        ax1.set_title('Real Data (PCA)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        
        # Synthetic data
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], pca_synthetic[:, 2], 
                   alpha=0.5, label='Synthetic', s=20, c='red')
        ax2.set_title('Synthetic Data (PCA)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        plt.tight_layout()

        pca_comparison_path = os.path.join(self.log_dir, f"pca_comparison.pdf")
        plt.savefig(pca_comparison_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def tsne_compared(self,test_loader):
        """Generate t-SNE plots for real vs synthetic data."""
        # Get conditions of real data
        input_data = next(iter(test_loader))
        if len(input_data) == 4:
            x, c, meter_id, timestamps = input_data
        else:
            x, meter_id, timestamps = input_data
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                c = self.model.conditioner.get_conditions(x, meter_id, timestamps)
            else:
                c = None
        x = x.to(self.device)
        c = c.to(self.device)
        timestamps = timestamps.to(self.device)
        # Generate synthetic samples with the SAME conditions
        synthetic_samples, _ = self.generator.generate(
            num_samples=x.shape[-1],
            conditions=c
        )
        if self.normalizer is not None:
            x_tf = x.transpose(1, 2) if x.dim() == 3 and x.shape[1] < x.shape[2] else x
            x_denorm = self.normalizer.inverse_transform(x_tf.cpu())
        else:
            x_tf = x.transpose(1, 2) if x.dim() == 3 and x.shape[1] < x.shape[2] else x
            x_denorm = x_tf.cpu().numpy()

        if isinstance(x_denorm, torch.Tensor):
            x_denorm = x_denorm.numpy()
        if isinstance(synthetic_samples, torch.Tensor):
            synthetic_samples = synthetic_samples.numpy()

        x_denorm_flat = x_denorm.reshape(x_denorm.shape[0], -1)
        synthetic_flat = synthetic_samples.reshape(synthetic_samples.shape[0], -1)

        # Combine for t-SNE
        combined = np.vstack([x_denorm_flat, synthetic_flat])
        labels = np.hstack([np.zeros(len(x_denorm_flat)), np.ones(len(synthetic_flat))])

        n_samples = combined.shape[0]
        if n_samples < 3:
            raise ValueError(f"Need at least 3 samples for t-SNE, got {n_samples}.")

        # Perplexity must be strictly less than n_samples
        perplexity = min(30, n_samples - 1)
        # Keep a sensible lower bound for stability when possible
        if perplexity < 5 and n_samples >= 6:
            perplexity = 5
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, learning_rate='auto', random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(combined)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        mask_real = labels == 0
        mask_synthetic = labels == 1
        ax.scatter(tsne_result[mask_real, 0], tsne_result[mask_real, 1], 
                  alpha=0.6, label='Real', s=30, c='blue')
        ax.scatter(tsne_result[mask_synthetic, 0], tsne_result[mask_synthetic, 1], 
                  alpha=0.6, label='Synthetic', s=30, c='red')
        ax.set_title('t-SNE: Real vs Synthetic Data')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        plt.tight_layout()
        tsne_comparison_path = os.path.join(self.log_dir, f"tsne_comparison.pdf")
        plt.savefig(tsne_comparison_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    

                                                                            