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
 
class Visualizer:
    """Class for visualizing training results and model performance."""
    
    def __init__(self, plot_style: PlotStyle, log_dir: str, 
                 model = None, normalizer = None,  writer = None, 
                 device = "cpu", conditioner = None, secondary_model = None):
        self.plot_style = plot_style
        self.log_dir = log_dir
        self.model = model
        self.normalizer = normalizer
        self.writer = writer
        self.device = device
        self.conditioner = conditioner

        self.generator = SyntheticGenerator(model=model, normalizer=normalizer, conditioner=conditioner, secondary_model=secondary_model, device=device)
    
    def _is_cvae(self) -> bool:
        """Check if model is a CVAE based on presence of conditioner on the model."""
        return hasattr(self.model, 'conditioner') and self.model.conditioner is not None

    def _to_time_feature(self, x):
        """Ensure tensor is in [..., L, F] layout (feature-last).

        Supports:
        - [B, L, F] or [B, F, L]
        - [S, B, L, F] or [S, B, F, L]
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        seq_len = getattr(self.model, 'sequence_length', 24)
        n_feat = getattr(self.model, 'num_features', None)

        if x.dim() == 3:
            # [B, L, F]
            if x.shape[1] == seq_len:
                return x
            # [B, F, L]
            if x.shape[2] == seq_len:
                return x.transpose(1, 2)
            # fallback: infer by smaller axis as feature axis
            return x if x.shape[1] >= x.shape[2] else x.transpose(1, 2)

        if x.dim() == 4:
            # [S, B, L, F]
            if x.shape[2] == seq_len:
                return x
            # [S, B, F, L]
            if x.shape[3] == seq_len:
                return x.transpose(2, 3)
            return x if x.shape[2] >= x.shape[3] else x.transpose(2, 3)

        return x
    def _unpack_batch(self, batch):
        """Strict unpack contract: (x, meter_id, timestamps, agg)."""
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            raise ValueError(
                "Expected batch as a 4-tuple: (x, meter_id, timestamps, agg)."
            )

        x, meter_id, timestamps, agg = batch

        if torch.is_tensor(meter_id):
            meter_id = meter_id.long()
        if torch.is_tensor(timestamps):
            timestamps = timestamps.long()

        return x, meter_id, timestamps, agg

    def _safe_inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Run inverse transform on feature-last tensors, preserving shape."""
        if self.normalizer is None:
            return x
        x_cpu = x.detach().cpu()

        # Robust path for phased log1p normalizers where stats may be numpy arrays.
        # Expected layout: [..., L, F] with feature axis last.
        method = getattr(self.normalizer, 'method', None)
        if method in {'log1p_phase', 'log1p'}:
            eps = float(getattr(self.normalizer, 'epsilon', 1e-8))
            mean = torch.as_tensor(getattr(self.normalizer, 'mean'), dtype=x_cpu.dtype)
            std = torch.as_tensor(getattr(self.normalizer, 'std'), dtype=x_cpu.dtype)

            if mean.ndim == 2 and mean.shape[0] == 1:
                mean = mean.squeeze(0)
            if std.ndim == 2 and std.shape[0] == 1:
                std = std.squeeze(0)

            if mean.ndim == 1:
                mean = mean.view(*([1] * (x_cpu.ndim - 1)), -1)
            if std.ndim == 1:
                std = std.view(*([1] * (x_cpu.ndim - 1)), -1)

            restored = torch.expm1(x_cpu * (std + eps) + mean)
            return torch.clamp(restored, min=0.0)

        return self.normalizer.inverse_transform(x_cpu)

    def _to_numpy(self, sample):
        """Convert tensor-like input to numpy array."""
        if isinstance(sample, torch.Tensor):
            return sample.detach().cpu().numpy()
        return np.asarray(sample)

    def _feature_series(self, sample_2d, feature_idx: int = 0):
        """Return a 1D hourly series for one feature/phase from a [L, F] sample."""
        arr = self._to_numpy(sample_2d)
        if arr.ndim == 1:
            return arr
        return arr[:, feature_idx]

    def _feature_labels(self, n_features: int):
        """Return labels for phases/features."""
        if n_features == 3:
            return ["Phase L1", "Phase L2", "Phase L3"]
        return [f"Feature {idx + 1}" for idx in range(n_features)]

    def _collect_nonzero_samples(self, test_loader, min_samples: int, max_batches: int = 100):
        """Collect at least min_samples non-zero examples from loader; fallback to first batch if needed."""
        xs, mids, tss = [], [], []
        total = 0
        first_batch = None

        if test_loader is None:
            raise RuntimeError("No test_loader provided for visualization.")

        for batch_idx, batch in enumerate(test_loader):
            if not isinstance(batch, (tuple, list)):
                continue

            x, meter_id, timestamps, _ = self._unpack_batch(batch)
            if x is None:
                continue

            if meter_id is None:
                meter_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            elif not torch.is_tensor(meter_id):
                meter_id = torch.as_tensor(np.asarray(meter_id), dtype=torch.long, device=x.device)
            else:
                meter_id = meter_id.to(device=x.device, dtype=torch.long)

            if timestamps is None:
                timestamps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            elif not torch.is_tensor(timestamps):
                timestamps = torch.as_tensor(np.asarray(timestamps), dtype=torch.long, device=x.device)
            else:
                timestamps = timestamps.to(device=x.device, dtype=torch.long)

            if first_batch is None:
                first_batch = (x, meter_id, timestamps)

            # Simple non-zero mask in raw space
            x_tf = self._to_time_feature(x)
            x_raw = x_tf
            mask = x_raw.reshape(x_raw.shape[0], -1).abs().sum(dim=1) > 1e-8

            if mask.any():
                xs.append(x[mask])
                mids.append(meter_id[mask])
                tss.append(timestamps[mask])
                total += int(mask.sum().item())

            if total >= int(min_samples):
                break
            if batch_idx >= max_batches - 1:
                break

        if total == 0:
            if first_batch is None:
                raise RuntimeError("No batches available in test_loader.")
            x, meter_id, timestamps = first_batch
            return x, meter_id, timestamps

        x_cat = torch.cat(xs, dim=0)
        mid_cat = torch.cat(mids, dim=0)
        ts_cat = torch.cat(tss, dim=0)
        n = min(int(min_samples), x_cat.shape[0])
        return x_cat[:n], mid_cat[:n], ts_cat[:n]

    def _rank_features_by_total(self, x):
        """Rank feature axis by descending total consumption, preserving shape.

        Supports:
        - [B, L, F]
        - [S, B, L, F]
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        if x.dim() == 3:  # [B, L, F]
            order = torch.argsort(x.sum(dim=1), dim=-1, descending=True)  # [B, F]
            return torch.gather(x, 2, order.unsqueeze(1).expand(-1, x.shape[1], -1))

        if x.dim() == 4:  # [S, B, L, F]
            order = torch.argsort(x.sum(dim=2), dim=-1, descending=True)  # [S, B, F]
            return torch.gather(x, 3, order.unsqueeze(2).expand(-1, -1, x.shape[2], -1))

        return x

    def visualize(self, test_loader = None, num_samples: int = 5, 
                  num_recon_per: int = 3, num_synthetic: int = 100,
                  num_originals: int = 4, num_synthetic_per: int = 10):
        """Main method to visualize results for an entire model experiment.
        Args:
            test_loader: The test data loader for reconstruction and latent space visualization.
            num_samples: Number of samples to visualize for reconstructions and synthetic samples.
            num_recon_per: Number of reconstructions to show per original sample.
            num_synthetic: Number of synthetic samples to generate for visualization.
            num_originals: Number of original samples to select for synthetic recreation visualization.
            num_synthetic_per: Number of synthetic samples to generate per original sample for synthetic recreation visualization.
        """
        try:
            print("Visualizing training progress.")
            self.visualize_training_progress()
        except Exception as e:
            print(f"Error visualizing training progress: {e}")

        self.visualize_reconstructions(num_samples=num_samples, num_recon_per=num_recon_per, test_loader=test_loader)

        self.visualize_synthetic(num_samples=num_synthetic)

        try:
            print("Visualizing sigma mapper.")
            self.visualize_sigma_mapper()
        except Exception as e:
            print(f"Error visualizing sigma mapper: {e}")
        try:
            print("Visualizing latent space.")
            self.visualize_latent_space(test_loader=test_loader)
        except Exception as e:
            print(f"Error visualizing latent space: {e}")
        
        self.visualize_synthetic_recreations(test_loader=test_loader, num_originals=num_originals, num_synthetic_per=num_synthetic_per)
        
        try:
            print("Visualizing user profiles.")
            self.visualize_user_dict()
        except Exception as e:
            print(f"Error visualizing user profiles: {e}")
    
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
        x_original, m_ids, t_ids = self._collect_nonzero_samples(test_loader, min_samples=num_samples)
        x_original_tf = self._to_time_feature(x_original)

        x_original_denorm_dbg = self._safe_inverse_transform(x_original_tf)


        # test_loader values are normalized in phased mode; reconstruct expects raw input
        # when is_denormalized=False (it will normalize once internally).
        recon_samples = self.generator.reconstruct(
            x_original_denorm_dbg,
            num_reconstructions=num_recon_per,
            is_denormalized=False,
            meter_id=m_ids,
            timestamp=t_ids,
        )

        if recon_samples is None or recon_samples.dim() != 3:
            print("Unexpected reconstruction output; expected tensor with shape [S, B, D].")
            return

        if x_original_tf.dim() != 3:
            print(f"Unexpected original sample shape: {tuple(x_original_tf.shape)}")
            return

        S, B, D = recon_samples.shape
        L, F = x_original_tf.shape[1], x_original_tf.shape[2]
        expected_D = L * F
        if D != expected_D:
            print(
                f"Shape mismatch: reconstruction last dim D={D} does not match original L*F={expected_D} "
                f"(L={L}, F={F})."
            )
            return

        # SyntheticGenerator.reconstruct flattens [L, F] before returning [S, B, L*F].
        recon_reshaped = recon_samples.reshape(S, B, L, F)

        if B != x_original_tf.shape[0]:
            b_common = min(B, x_original_tf.shape[0])
            x_original_tf = x_original_tf[:b_common]
            recon_reshaped = recon_reshaped[:, :b_common]
            B = b_common

        x_original_denorm = x_original_denorm_dbg.cpu()
        recon_denorm = recon_reshaped.cpu()

        num_to_plot = min(num_samples, B)

        view_specs = []
        if F > 1:
            ranked_orig = self._rank_features_by_total(x_original_denorm)
            ranked_recon = self._rank_features_by_total(recon_denorm)
            view_specs.append((
                "aggregated",
                x_original_denorm.sum(dim=-1, keepdim=True),
                recon_denorm.sum(dim=-1, keepdim=True),
                "Aggregated 3-Phase",
            ))
            for phase_idx in range(F):
                phase_no = phase_idx + 1
                view_specs.append((
                    f"ranked_phase_{phase_no}",
                    ranked_orig[..., phase_idx:phase_idx+1],
                    ranked_recon[..., phase_idx:phase_idx+1],
                    f"Ranked Phase {phase_no}",
                ))
        else:
            view_specs.append(("single_phase", x_original_denorm, recon_denorm, "Single Phase"))

        for view_name, orig_view, recon_view, view_label in view_specs:
            fig, axes = plt.subplots(num_to_plot, 1, figsize=(8, 3.2 * num_to_plot), squeeze=False, sharex=True)

            for i in range(num_to_plot):
                ax = axes[i][0]
                original_series = self._feature_series(orig_view[i], 0)
                ax.plot(
                    original_series,
                    label="Original",
                    marker=self.plot_style.marker,
                    color=self.plot_style.color_original,
                    alpha=0.9,
                    linewidth=self.plot_style.linewidth,
                    linestyle=self.plot_style.linestyle,
                    markersize=4
                )
                num_recon_available = recon_view.shape[0]
                for s in range(min(num_recon_per, num_recon_available)):
                    recon_series = self._feature_series(recon_view[s, i], 0)
                    ax.plot(
                        recon_series,
                        label=f"Recon Sample {s+1}",
                        marker=self.plot_style.marker,
                        color=self.plot_style.get_color(s),
                        alpha=0.6,
                        linewidth=self.plot_style.linewidth,
                        linestyle=self.plot_style.linestyle,
                        markersize=3
                    )

                title = f"Sample {i+1} - {view_label}"
                ax.set_title(title)
                ax.set_xlabel("Hour")
                ax.set_ylabel("Energy (kWh)")
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)
                if view_name.startswith("ranked_phase_"):
                    samples_per_hour = max(1, L // 24)
                    hour_ticks = np.arange(0, L, samples_per_hour * 2)  # every 2 hours
                    hour_labels = (hour_ticks // samples_per_hour).astype(int)
                    ax.set_xticks(hour_ticks)
                    ax.set_xticklabels(hour_labels)
                    ax.tick_params(axis='x', labelsize=8)

            fig_title = f"{self.model.__class__.__name__} Reconstruction ({view_label}, {num_recon_per} decoder samples per original)"
            if self._is_cvae():
                fig_title += " - Conditioned on Day of Week"
            fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            if view_name == "aggregated":
                target_dir = os.path.join(self.log_dir, "aggregated_phase_eval")
            elif view_name.startswith("ranked_phase_"):
                target_dir = os.path.join(self.log_dir, f"{view_name}_eval")
            else:
                target_dir = self.log_dir
            os.makedirs(target_dir, exist_ok=True)
            reconstruction_path = os.path.join(target_dir, f"reconstruction_{view_name}.pdf")
            plt.savefig(reconstruction_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
    
    def visualize_synthetic(self, num_samples: int):
        """Visualize synthetic samples generated by the model."""
        if self.model is None or self.normalizer is None:
            print("Model and normalizer are required for synthetic sample visualization.")
            return
        
        synthetic_data, conditions = self.generator.generate(num_samples=num_samples)
    
        num_to_plot = synthetic_data.shape[0]
        num_features = synthetic_data.shape[2] if synthetic_data.dim() == 3 else 1
        feature_labels = self._feature_labels(num_features)
        hours = np.arange(synthetic_data.shape[1])
        fig, axes = plt.subplots(num_features, 1, figsize=(8, 3.2 * num_features), squeeze=False, sharex=True)

        for feature_idx in range(num_features):
            ax = axes[feature_idx][0]
            for i in range(num_to_plot):
                label = f"Sample {i+1}"
                ax.plot(
                    hours,
                    self._feature_series(synthetic_data[i], feature_idx),
                    label=label,
                    marker=self.plot_style.marker,
                    color=self.plot_style.get_color(i),
                    alpha=0.8,
                    linewidth=self.plot_style.linewidth,
                    linestyle=self.plot_style.linestyle,
                    markersize=4
                )
            ax.set_title(feature_labels[feature_idx])
            ax.set_xlabel("Hour", fontsize=11)
            ax.set_ylabel("Energy (kWh)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        
        fig_title = f"{self.model.__class__.__name__} Synthetic Samples"
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

    def visualize_latent_space(self, test_loader, PCA: bool = True, tSNE: bool = True, max_samples: int = 1000):
        """Visualize the latent space using t-SNE or PCA."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        latents = []

        # we cut the 

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                
                x, meter_id, timestamps, agg = self._unpack_batch(batch)
                
                x = x.to(self.device)

                if self._is_cvae():
                    # SetGuideVAE path: encoder expects [set_repr, c], not [x_flat, c]
                    if hasattr(self.model, "_set_encode") and hasattr(self.model, "_to_phase_first"):
                        x_phase_first = self.model._to_phase_first(x).float()
                        x_for_conditions = self.model._build_condition_input(x_phase_first) if hasattr(self.model, "_build_condition_input") else x_phase_first.transpose(1, 2)
                        c = self.model.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps).to(self.device)
                        set_repr = self.model._set_encode(x_phase_first)
                        enc_in = torch.cat([set_repr, c], dim=-1)
                    else:
                        x_flat = x.reshape(x.shape[0], -1)
                        c = self.model.conditioner.get_conditions(x, meter_id, timestamps).to(self.device)
                        enc_in = torch.cat([x_flat, c], dim=-1)
                else:
                    x_flat = x.reshape(x.shape[0], -1)
                    enc_in = x_flat

                q_params = self.model.encoder(enc_in)
                z = self.model.encoder.reparameterize(q_params, num_samples=1).squeeze(0)
                
                latents.append(z.cpu().numpy())
                if len(latents) >= max_samples // x.shape[0]:
                    break
        
        latents = np.concatenate(latents, axis=0)

        if PCA:
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
            fig, ax = plt.subplots(figsize=self.plot_style.figsize)
            ax.scatter(
                latents_2d[:, 0],
                latents_2d[:, 1],
                alpha=0.6,
                s=100,
                label='Samples'
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            ax.set_title('Latent Space Visualization (PCA)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"PCA_latent_space.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        if tSNE:
            tsne = TSNE(n_components=2)
            latents_2d = tsne.fit_transform(latents)
            fig, ax = plt.subplots(figsize=self.plot_style.figsize)
            ax.scatter(
                latents_2d[:, 0],
                latents_2d[:, 1],
                alpha=0.6,
                s=100,
                label='Samples'
            )
            ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
            ax.set_title('Latent Space Visualization (t-SNE)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
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
        
        x, meter_id, timestamps = self._collect_nonzero_samples(
                test_loader,
                min_samples=max(1, int(num_originals)),
                max_batches=100,
            )
        
        x_tf = self._to_time_feature(x)
        num_originals = min(int(num_originals), x_tf.shape[0])
        x_selected_tf = x_tf[:num_originals]
        x_selected_denorm = self._safe_inverse_transform(x_selected_tf)
        meter_id_selected = meter_id[:num_originals]
        timestamps_selected = timestamps[:num_originals]
        
        # Build labels and generate synthetic recreations for each original condition.
        # Collect as [S, B, L, F] where B=num_originals.
        condition_labels = []
        synthetic_per_original = []
        for i in range(num_originals):
            ts = timestamps_selected[i].cpu().item()
            dt = pd.Timestamp(ts, unit='s')
            day_name = self.plot_style.get_day_names(short=False)[dt.dayofweek]
            date_str = dt.strftime("%Y-%m-%d")
            m_id = meter_id_selected[i].item()
            synthetic_samples_i, _ = self.generator.generate(num_synthetic_per, meter_id=m_id, date=dt)
            synthetic_samples_i = self._to_time_feature(synthetic_samples_i)
            synthetic_per_original.append(synthetic_samples_i)
            condition_labels.append(f"Meter {m_id} | {day_name} ({date_str})")

        synthetic_samples = torch.stack(synthetic_per_original, dim=1)  # [S, B, L, F]
        num_synth_available = synthetic_samples.shape[0]
        L = x_selected_denorm.shape[1]
        F = x_selected_denorm.shape[2] if x_selected_denorm.dim() == 3 else 1
    
        hours = np.arange(L)
        synthetic_alpha_values = np.linspace(0.3, 0.7, max(1, num_synth_available))
        mean_synthetic = synthetic_samples.mean(dim=0)
        view_specs = []
        if F > 1:
            ranked_orig = self._rank_features_by_total(x_selected_denorm)
            ranked_synth = self._rank_features_by_total(synthetic_samples)
            ranked_mean = self._rank_features_by_total(mean_synthetic)

            view_specs.append((
                "aggregated",
                x_selected_denorm.sum(dim=-1, keepdim=True),
                synthetic_samples.sum(dim=-1, keepdim=True),
                mean_synthetic.sum(dim=-1, keepdim=True),
                "Aggregated 3-Phase",
            ))
            for phase_idx in range(F):
                phase_no = phase_idx + 1
                view_specs.append((
                    f"ranked_phase_{phase_no}",
                    ranked_orig[..., phase_idx:phase_idx+1],
                    ranked_synth[..., phase_idx:phase_idx+1],
                    ranked_mean[..., phase_idx:phase_idx+1],
                    f"Ranked Phase {phase_no}",
                ))
        else:
            view_specs.append(("single_phase", x_selected_denorm, synthetic_samples, mean_synthetic, "Single Phase"))

        for view_name, orig_view, synth_view, mean_view, view_label in view_specs:
            fig, axes = plt.subplots(num_originals, 1, figsize=(8, 3.2 * num_originals), squeeze=False, sharex=True)

            for i in range(num_originals):
                ax = axes[i][0]
                num_plot_synth = min(num_synthetic_per, synth_view.shape[0])
                for s in range(num_plot_synth):
                    synth_data = self._feature_series(synth_view[s, i], 0)
                    ax.plot(
                        hours, synth_data,
                        color=self.plot_style.get_color(s),
                        alpha=synthetic_alpha_values[s],
                        linewidth=self.plot_style.linewidth,
                        linestyle=self.plot_style.linestyle,
                        label='Synthetic' if s == 0 else None
                    )

                mean_data = self._feature_series(mean_view[i], 0)
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

                orig_data = self._feature_series(orig_view[i], 0)
                ax.plot(
                    hours, orig_data,
                    color=self.plot_style.get_color_original(),
                    linewidth=self.plot_style.linewidth,
                    label='Original',
                    marker=self.plot_style.marker,
                    markersize=5,
                    alpha=1.0
                )

                rmse = np.sqrt(np.mean((orig_data - mean_data) ** 2))
                ax.set_title(
                    f"{condition_labels[i]} | {view_label} | RMSE: {rmse:.3f} kWh",
                    fontsize=12,
                    fontweight='bold'
                )
                L = orig_view.shape[1]
                ax.set_xlabel("Hour", fontsize=10)
                ax.set_ylabel("Energy Consumption (kWh)", fontsize=10)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
                if view_name.startswith("ranked_phase_"):
                    samples_per_hour = max(1, L // 24)
                    hour_ticks = np.arange(0, L, samples_per_hour) 
                    hour_labels = (hour_ticks // samples_per_hour).astype(int)
                    ax.set_xticks(hour_ticks)
                    ax.set_xticklabels(hour_labels)
                    ax.tick_params(axis='x', labelsize=8)
                else:
                    ax.set_xticks(hours[::2])
                    ax.tick_params(axis='x', labelsize=10)
                ax.set_xlim(0, L - 1)

            fig.suptitle(
                f"Synthetic Recreation ({view_label}) with Same Conditions ({num_synthetic_per} samples per original)",
                fontsize=14,
                fontweight='bold',
                y=1.02
            )
            plt.tight_layout()

            if view_name == "aggregated":
                target_dir = os.path.join(self.log_dir, "aggregated_phase_eval")
            elif view_name.startswith("ranked_phase_"):
                target_dir = os.path.join(self.log_dir, f"{view_name}_eval")
            else:
                target_dir = self.log_dir
            os.makedirs(target_dir, exist_ok=True)
            save_path = os.path.join(target_dir, f"synthetic_recreations_{view_name}.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def visualize_user_dict(self):
        """Plot the cached user profiles in the UserProfiler conditioner."""
        if not hasattr(self.model, 'conditioner'):
            print("Model does not have a conditioner, skipping user profiles visualization.")
            return {}
        if "user_profile" not in self.model.conditioner.get_condition_tags():
            print("Model conditioner does not include 'user_profile', skipping user profiles visualization.")
            return {}
        # Access the UserProfiler from condition_items dictionary
        user_profiler = self.model.conditioner.condition_items["user_profile"]

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