import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from src.model.model import BaseModel

class SplitterCNN(nn.Module):
    def __init__(self, condition_dim=32, base_channels=64, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        
        self.init_conv = nn.Conv1d(1 + condition_dim, base_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1)
        ])
        
        self.out_conv = nn.Conv1d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x_agg, c):
        # x_agg: [B, 1, 24]
        # c: [B, condition_dim]
        
        # 1. Expand conditions across all 24 hours
        # c_expanded shape: [B, condition_dim, 24]
        c_expanded = c.unsqueeze(-1).expand(-1, -1, x_agg.size(2))
        
        # 2. Concatenate the Aggregated curve with the Conditions
        # h shape: [B, 1 + condition_dim, 24]
        h = torch.cat([x_agg, c_expanded], dim=1)
        
        # 3. Process
        h = F.gelu(self.init_conv(h))
        
        for block in self.blocks:
            h_new = F.gelu(block(h))
            h = h + h_new # Residual connection
            
        logits = self.out_conv(h) # [B, 3, 24]
        
        # Divide by a temperature (e.g., 2.0 to 5.0) to prevent logit explosion
        # This keeps the probabilities from snapping to 1.0/0.0 too early in training
        proportions = F.softmax(logits / self.temperature, dim=1)
        
        return proportions
    

class HierarchicalSplitter(BaseModel):
    def __init__(self, conditioner, sequence_length=24, condition_dim=32, base_channels=64, temperature=2.0, phased_mode=True):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length)
        self.conditioner = conditioner
        self.sequence_length = sequence_length
        self.num_features = 3
        self.condition_dim = int(condition_dim)
        
        self.model = SplitterCNN(condition_dim=condition_dim, base_channels=base_channels, temperature=temperature)
        
        # The 6 possible ways to assign 3 anonymous slots to 3 physical wires
        self.permutations = [
            [0, 1, 2], [0, 2, 1], 
            [1, 0, 2], [1, 2, 0], 
            [2, 0, 1], [2, 1, 0]
        ]

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def train_core(self, batch, optim, condition_dropout = False, **_):
        self.train()

        # 1. Unpack Real Data
        x_raw, meter_id, timestamps, agg = self._unpack_batch(batch)
        x_true_phases = self._to_phase_first(x_raw).float() # [B, 3, 24]
        
        # 2. Create the Aggregated Input from the real data
        x_agg = x_true_phases.sum(dim=1, keepdim=True) # [B, 1, 24]

        # 3. Get Conditions
        if (
            torch.is_tensor(agg)
            and agg.dim() == 2
            and agg.shape[-1] == self.condition_dim
        ):
            c = self.move_to_device(agg, device=x_raw.device).float()
            if c.dim() == 1:
                c = c.unsqueeze(0)
        else:
            x_for_cond = self._build_condition_input(x_true_phases)
            c = self.conditioner.get_conditions(x_for_cond, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=x_raw.device).float()

        if condition_dropout and self.training and torch.rand(1).item() < 0.2:
            c = torch.zeros_like(c)

        # 4. Predict Proportions
        proportions = self.model(x_agg, c) # [B, 3, 24], sums to 1

        # 5. Multiply proportions by the total to get the generated phases
        x_pred_phases = proportions * x_agg # [B, 3, 24]

        # 6. Permutation-Invariant Loss
        loss_dict = self.loss(x_pred_phases, x_true_phases)

        optim.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        optim.step()
        
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, **_):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        n = 0

        device = next(self.parameters()).device
        for batch in valloader:
            batch = self.move_to_device(batch, device)
            x_raw, meter_id, timestamps, _ = self._unpack_batch(batch)
            x_true_phases = self._to_phase_first(x_raw).float()
            x_agg = x_true_phases.sum(dim=1, keepdim=True)  # [B, 1, 24]

            x_for_cond = self._build_condition_input(x_true_phases)
            c = self.conditioner.get_conditions(x_for_cond, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device).float()

            proportions = self.model(x_agg, c)
            x_pred_phases = proportions * x_agg
            loss_dict = self.loss(x_pred_phases, x_true_phases)

            for k in agg:
                agg[k] += float(loss_dict[k].detach())
            n += 1

        for k in agg:
            agg[k] /= max(n, 1)
        return agg

    @torch.no_grad()
    def evaluate_split_quality(
        self,
        dataloader,
        device: Optional[str] = None,
        max_batches: Optional[int] = None,
        normalizer=None,
        plot_dir: Optional[str] = None,
        num_plot_samples: int = 6,
    ) -> Dict[str, float]:
        """Evaluate permutation-invariant split quality against ground truth phases."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        total_samples = 0
        pi_mse_sum = 0.0
        pi_mae_sum = 0.0
        phase_energy_mae_sum = 0.0
        agg_mae_sum = 0.0
        r2_sum = 0.0
        plot_true = []
        plot_pred = []
        phase_mae_samples = []
        agg_mae_samples = []

        eps = 1e-8
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = self.move_to_device(batch, device)
            x_raw, meter_id, timestamps, x_agg = self._unpack_batch(batch)
            x_true = self._to_phase_first(x_raw).float()  # [B, 3, L]

            x_for_cond = self._build_condition_input(x_true)
            c = self.conditioner.get_conditions(x_for_cond, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device).float()

            proportions = self.model(x_agg, c)
            x_pred = proportions * x_agg

            per_perm_mse = []
            per_perm_mae = []
            for perm in self.permutations:
                p = x_pred[:, perm, :]
                per_perm_mse.append(((p - x_true) ** 2).mean(dim=(1, 2)))
                per_perm_mae.append((p - x_true).abs().mean(dim=(1, 2)))

            per_perm_mse = torch.stack(per_perm_mse, dim=1)  # [B, 6]
            per_perm_mae = torch.stack(per_perm_mae, dim=1)  # [B, 6]
            best_mse, best_idx = per_perm_mse.min(dim=1)
            best_mae = per_perm_mae.gather(1, best_idx.unsqueeze(1)).squeeze(1)

            aligned_proportions = torch.empty_like(proportions).to(device)
            for i in range(proportions.size(0)):
                perm = self.permutations[int(best_idx[i].item())]
                aligned_proportions[i] = proportions[i, perm, :]

            x_true_eval = x_true
            if normalizer is not None and hasattr(normalizer, "inverse_transform"):
                true_lf = x_true.transpose(1, 2)
                bsz, seq_len, n_feat = true_lf.shape
                
                true_denorm = normalizer.inverse_transform(true_lf.reshape(-1, n_feat)).reshape(bsz, seq_len, n_feat)
                true_denorm = true_denorm.to(device=x_true.device, dtype=x_true.dtype)
                x_true_eval = true_denorm.transpose(1, 2)
            x_agg_eval = x_true_eval.sum(dim=1, keepdim=True)
            aligned_pred_eval = aligned_proportions * x_agg_eval

            phase_energy_mae = (aligned_pred_eval.sum(dim=2) - x_true_eval.sum(dim=2)).abs().mean(dim=1)
            agg_mae = (
                aligned_pred_eval.sum(dim=1, keepdim=True) - x_true_eval.sum(dim=1, keepdim=True)
            ).abs().mean(dim=(1, 2))

            phase_pointwise_mae = (aligned_pred_eval - x_true_eval).abs().mean(dim=2)  # [B, F]
            phase_mae_samples.append(phase_pointwise_mae.detach().cpu())
            agg_mae_samples.append(agg_mae.detach().cpu())

            flat_true = x_true_eval.reshape(x_true_eval.shape[0], -1)
            flat_pred = aligned_pred_eval.reshape(aligned_pred_eval.shape[0], -1)
            ss_res = ((flat_true - flat_pred) ** 2).sum(dim=1)
            true_mean = flat_true.mean(dim=1, keepdim=True)
            ss_tot = ((flat_true - true_mean) ** 2).sum(dim=1).clamp_min(eps)
            r2 = 1.0 - (ss_res / ss_tot)

            remaining = max(0, num_plot_samples - len(plot_true))
            if remaining > 0:
                take = min(remaining, x_true_eval.shape[0])
                plot_true.extend([x_true_eval[i].detach().cpu() for i in range(take)])
                plot_pred.extend([aligned_pred_eval[i].detach().cpu() for i in range(take)])

            b = int(x_true.size(0))
            total_samples += b
            pi_mse_sum += float(best_mse.sum().item())
            pi_mae_sum += float(best_mae.sum().item())
            phase_energy_mae_sum += float(phase_energy_mae.sum().item())
            agg_mae_sum += float(agg_mae.sum().item())
            r2_sum += float(r2.sum().item())

        if total_samples == 0:
            raise RuntimeError("No samples available to evaluate splitter quality.")

        metrics = {
            "num_samples": float(total_samples),
            "pi_mse": pi_mse_sum / total_samples,
            "pi_mae": pi_mae_sum / total_samples,
            "phase_energy_mae": phase_energy_mae_sum / total_samples,
            "aggregate_conservation_mae": agg_mae_sum / total_samples,
            "pi_r2": r2_sum / total_samples,
        }

        if plot_dir is not None and len(plot_true) > 0:
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "phase_splitter_eval_examples.pdf")

            n = len(plot_true)
            fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.5 * n)), squeeze=False, sharex=True)
            for i in range(n):
                ax = axes[i][0]
                true_sample = plot_true[i]  # [F, L]
                pred_sample = plot_pred[i]  # [F, L]
                hours = np.arange(true_sample.shape[1])

                ax.plot(hours, true_sample.sum(dim=0).numpy(), color="black", linewidth=2.0, label="True aggregate")
                ax.plot(hours, pred_sample.sum(dim=0).numpy(), color="gray", linewidth=1.8, linestyle="--", label="Pred aggregate")

                colors = ["tab:blue", "tab:orange", "tab:green"]
                for p in range(min(3, true_sample.shape[0])):
                    ax.plot(hours, true_sample[p].numpy(), color=colors[p], alpha=0.8, linewidth=1.4, label=f"True L{p+1}")
                    ax.plot(hours, pred_sample[p].numpy(), color=colors[p], alpha=0.8, linewidth=1.2, linestyle=":", label=f"Pred L{p+1}")

                if i == 0:
                    ax.legend(loc="upper right", ncol=2, fontsize=8)
                ax.set_ylabel("Energy")
                ax.grid(alpha=0.25)
                ax.set_title(f"Sample {i+1}")

            axes[-1][0].set_xlabel("Hour")
            fig.suptitle("Phase Splitter Evaluation Samples (Permutation-Aligned)", fontsize=12)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

        if plot_dir is not None and len(phase_mae_samples) > 0:
            os.makedirs(plot_dir, exist_ok=True)
            boxplot_path = os.path.join(plot_dir, "phase_splitter_phase_mae_boxplot.pdf")

            phase_mae_tensor = torch.cat(phase_mae_samples, dim=0)  # [N, F]
            agg_mae_tensor = torch.cat(agg_mae_samples, dim=0)      # [N]

            n_phase = phase_mae_tensor.shape[1]
            labels = [f"Phase L{i+1}" for i in range(n_phase)] + ["Aggregate"]
            box_data = [phase_mae_tensor[:, i].numpy() for i in range(n_phase)] + [agg_mae_tensor.numpy()]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            bp = ax.boxplot(box_data, label=labels, patch_artist=True, showfliers=False)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#444444"]
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[min(i, len(colors) - 1)])
                patch.set_alpha(0.55)

            ax.set_ylabel("MAE")
            ax.set_title("Phase Splitter MAE Distribution (Permutation-Aligned)")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(boxplot_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

            metrics["phase_mae_median"] = {
                f"phase_{i+1}": float(torch.median(phase_mae_tensor[:, i]).item())
                for i in range(n_phase)
            }
            metrics["aggregate_mae_median"] = float(torch.median(agg_mae_tensor).item())

        return metrics

    def loss(self, x_pred, x_true):
        # x_pred: [B, 3, 24], x_true: [B, 3, 24]
        B = x_pred.size(0)
        
        losses = []
        for perm in self.permutations:
            # Shuffle the output slots to match the current target permutation
            permuted_pred = x_pred[:, perm, :]
            
            # Calculate MSE per batch item: shape [B]
            mse = F.mse_loss(permuted_pred, x_true, reduction='none').mean(dim=[1, 2])
            losses.append(mse)
            
        # Stack all 6 permutation losses: [B, 6]
        losses = torch.stack(losses, dim=1)
        
        # For each item in the batch, pick the permutation that had the lowest error
        best_loss_per_item, _ = losses.min(dim=1) # [B]
        
        # Average over the batch
        final_loss = best_loss_per_item.mean()
        
        return {
            "loss": final_loss,
            "recon_loglikelihood": final_loss, # Keeping dictionary format consistent
            "kl_divergence": torch.zeros_like(final_loss)
        }

    @torch.no_grad()
    def sample(
        self,
        x_agg_generated,
        conditions,
        normalizer=None,
        x_phased_real=None,
        meter_id=None,
        timestamps=None,
        timestamp=None,
        device="cpu",
        **_,
    ):
        """Split aggregated generated data into 3 phases.

        Conditions are always built fresh from the splitter's own conditioner so
        that the condition dimension matches what the SplitterCNN was trained on.

        Priority for building conditions:
          1. ``x_phased_real`` — real phased input (reconstruction path).
          2. ``meter_id`` / ``timestamps`` — metadata only (generation path).
          3. ``conditions`` — fallback only when the splitter has no conditioner.

        ``meter_id=None`` is not supported when the splitter has a conditioner; a
        clear ValueError is raised rather than silently producing garbage outputs.
        """
        self.eval()
        self.to(device)

        x_agg = self.move_to_device(x_agg_generated, device)
        if x_agg.dim() == 3 and x_agg.shape[2] == 1:
            x_agg = x_agg.transpose(1, 2)

        ts = timestamps if timestamps is not None else timestamp

        if self.conditioner is not None:
            # Build conditions using the splitter's own conditioner so the
            # condition dimension always matches SplitterCNN's expectation.
            has_signal = x_phased_real is not None
            has_meta   = meter_id is not None or ts is not None

            if not has_signal and not has_meta:
                raise ValueError(
                    "HierarchicalSplitter.sample: conditioner is set but neither "
                    "x_phased_real nor meter_id/timestamps were provided.  "
                    "Cannot build conditions — pass at least one."
                )

            cond_x = None
            if has_signal:
                xp = self.move_to_device(x_phased_real, device).float()
                cond_x = self._build_condition_input(self._to_phase_first(xp))  # [B, T, F]

            c = self.conditioner.get_conditions(cond_x, meter_id=meter_id, timestamps=ts)
            c = self.move_to_device(c, device).float()
        else:
            # No conditioner — use whatever was passed in (e.g. zero vector).
            c = self.move_to_device(conditions, device).float()

        # Predict proportions and scale by aggregate.
        proportions = self.model(x_agg, c)
        x_pred = proportions * x_agg

        if normalizer is not None and hasattr(normalizer, "inverse_transform"):
            try:
                return normalizer.inverse_transform(x_pred)
            except Exception:
                print("[Warning]: Normalizer inverse_transform failed. Returning phase values in input scale.")

        return x_pred

    def reconstruct(self, x_agg_generated, conditions, normalizer=None, x_phased_real=None, device="cpu", **kwargs):
        """Reconstruct by splitting; forwards x_phased_real and metadata to sample."""
        return self.sample(
            x_agg_generated,
            conditions,
            normalizer=normalizer,
            x_phased_real=x_phased_real,
            device=device,
            **kwargs,
        )