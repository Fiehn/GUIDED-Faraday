import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

class SplitterCNN(nn.Module):
    def __init__(self, condition_dim=32, base_channels=64):
        super().__init__()
        
        # INCREASED INPUT CHANNELS: 1 (Aggregated) + condition_dim
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
        temperature = 3.0 
        proportions = F.softmax(logits / temperature, dim=1)
        
        return proportions
    

class HierarchicalSplitter(nn.Module):
    def __init__(self, conditioner, sequence_length=24, condition_dim=32):
        super().__init__()
        self.conditioner = conditioner
        self.sequence_length = sequence_length
        self.num_features = 3
        self.condition_dim = int(condition_dim)
        
        self.model = SplitterCNN(condition_dim=condition_dim)
        
        # The 6 possible ways to assign 3 anonymous slots to 3 physical wires
        self.permutations = [
            [0, 1, 2], [0, 2, 1], 
            [1, 0, 2], [1, 2, 0], 
            [2, 0, 1], [2, 1, 0]
        ]

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_tag(self):
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.__class__.__name__}_{self.model.__class__.__name__}_{time_stamp}"

    def train_core(self, batch, optim, condition_dropout=False, **_):
        self.train()

        # 1. Unpack Real Data
        x_raw, meter_id, timestamps, provided_conditions = self._unpack_batch(batch)
        x_true_phases = self._to_phase_first(x_raw).float() # [B, 3, 24]
        
        # 2. Create the Aggregated Input from the real data
        x_agg = x_true_phases.sum(dim=1, keepdim=True) # [B, 1, 24]

        # 3. Get Conditions
        if provided_conditions is not None:
            c = self.move_to_device(provided_conditions, device=x_raw.device).float()
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
            x_raw, meter_id, timestamps, provided_conditions = self._unpack_batch(batch)
            x_true_phases = self._to_phase_first(x_raw).float()
            x_agg = x_true_phases.sum(dim=1, keepdim=True)

            if provided_conditions is not None:
                c = self.move_to_device(provided_conditions, device=device).float()
                if c.dim() == 1:
                    c = c.unsqueeze(0)
            else:
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
            if max_batches is not None and batch_idx >= int(max_batches):
                break

            batch = self.move_to_device(batch, device)
            x_raw, meter_id, timestamps, provided_conditions = self._unpack_batch(batch)
            x_true = self._to_phase_first(x_raw).float()  # [B, 3, L]
            x_agg = x_true.sum(dim=1, keepdim=True)

            if provided_conditions is not None:
                c = self.move_to_device(provided_conditions, device=device).float()
                if c.dim() == 1:
                    c = c.unsqueeze(0)
            else:
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

            aligned_pred = torch.empty_like(x_pred)
            for i in range(x_pred.size(0)):
                perm = self.permutations[int(best_idx[i].item())]
                aligned_pred[i] = x_pred[i, perm, :]

            x_true_eval = x_true
            aligned_pred_eval = aligned_pred
            if normalizer is not None and hasattr(normalizer, "inverse_transform"):
                # Normalizer expects feature-last input; convert [B, F, L] -> [B, L, F].
                true_lf = x_true.transpose(1, 2)
                pred_lf = aligned_pred.transpose(1, 2)
                bsz, seq_len, n_feat = true_lf.shape

                true_denorm = normalizer.inverse_transform(true_lf.reshape(-1, n_feat)).reshape(bsz, seq_len, n_feat)
                pred_denorm = normalizer.inverse_transform(pred_lf.reshape(-1, n_feat)).reshape(bsz, seq_len, n_feat)

                x_true_eval = true_denorm.transpose(1, 2)
                aligned_pred_eval = pred_denorm.transpose(1, 2)

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

            remaining = max(0, int(num_plot_samples) - len(plot_true))
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
            metrics["examples_plot_path"] = plot_path

        if plot_dir is not None and len(phase_mae_samples) > 0:
            os.makedirs(plot_dir, exist_ok=True)
            boxplot_path = os.path.join(plot_dir, "phase_splitter_phase_mae_boxplot.pdf")

            phase_mae_tensor = torch.cat(phase_mae_samples, dim=0)  # [N, F]
            agg_mae_tensor = torch.cat(agg_mae_samples, dim=0)      # [N]

            n_phase = int(phase_mae_tensor.shape[1])
            labels = [f"Phase L{i+1}" for i in range(n_phase)] + ["Aggregate"]
            box_data = [phase_mae_tensor[:, i].numpy() for i in range(n_phase)] + [agg_mae_tensor.numpy()]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
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

            metrics["phase_mae_boxplot_path"] = boxplot_path
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
    def sample(self, x_agg_generated, conditions, device="cpu", **_):
        """
        Inference time! 
        Pass your Aggregated Synthetic data (from your VAE/Diffusion model) into here.
        """
        self.eval()
        self.to(device)
        
        x_agg = self.move_to_device(x_agg_generated, device)
        c = self.move_to_device(conditions, device).float()
        
        # Predict Proportions
        proportions = self.model(x_agg, c)
        
        # Apply constraint
        x_final_phases = proportions * x_agg
        
        return x_final_phases
    
    def _build_condition_input(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Convert [B, F, L] -> [B, L, F] for conditioner compatibility."""
        return x_phase_first.transpose(1, 2)

    def _unpack_batch(self, batch):
        """Support (x, meter_id, timestamps), (x, cond), (x, meter_id), and (x, c, meter_id, timestamps)."""
        provided_conditions = None
        expected_cond_dim = self.condition_dim

        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            x, second, third, fourth = batch
            # Regular CSV explicit tuple: (x, c, meter_id, timestamps)
            if (
                torch.is_tensor(second)
                and second.dim() >= 2
                and expected_cond_dim is not None
                and int(expected_cond_dim) > 0
                and second.shape[-1] == int(expected_cond_dim)
            ):
                return x, third, fourth, second
            # Extended eval tuple: (x, meter_id, timestamps, agg)
            return x, second, third, None

        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, second, third = batch
            if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
                meter_id = second[:, 0, 0].long()
                timestamps = second[:, 1, 0].long()
                return x, meter_id, timestamps, None
            return x, second, third, None

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "HierarchicalSplitter expects batches as (x, meter_id, timestamps), (x, meter_id), (x, cond), or (x, c, meter_id, timestamps)."
            )

        x, second = batch
        meter_id = second
        timestamps = None

        if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
            meter_id = second[:, 0, 0].long()
            timestamps = second[:, 1, 0].long()
        elif torch.is_tensor(second) and second.dim() == 2:
            if second.shape[1] >= 2 and not torch.is_floating_point(second):
                meter_id = second[:, 0].long()
                timestamps = second[:, 1].long()
            elif expected_cond_dim is not None and second.shape[1] == expected_cond_dim:
                provided_conditions = second

        return x, meter_id, timestamps, provided_conditions

    def save_model(self, save_path=None, model_name="trained_model", device="cpu", normalizer=None):
        if save_path is None:
            save_path = "./models/"

        model_path = os.path.join(save_path, f"{model_name}.pt")
        os.makedirs(save_path, exist_ok=True)

        original_device = next(self.parameters()).device
        self.eval()
        self.to("cpu")

        normalizer_state = None
        if normalizer is not None and hasattr(normalizer, "fitted") and normalizer.fitted:
            normalizer_state = {
                "method": normalizer.method,
                "epsilon": normalizer.epsilon,
                "zero_id": normalizer.zero_id,
                "shift": normalizer.shift,
                "non_zero_mean": normalizer.non_zero_mean,
                "non_zero_std": normalizer.non_zero_std,
                "mean": normalizer.mean,
                "std": normalizer.std,
            }

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "constructor_args": {
                "sequence_length": self.sequence_length,
                "condition_dim": self.condition_dim,
            },
            "normalizer_state": normalizer_state,
            "model_info": {
                "model_class": self.__class__.__name__,
                "save_format_version": 1,
            },
        }

        temp_path = f"{model_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, model_path)

        self.to(device)
        self.to(original_device)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model with {num_params} parameters saved to {model_path}")

    def move_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                b.to(device, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch
            )
        if isinstance(batch, dict):
            return {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        return batch

    def _to_phase_first(self, x: torch.Tensor) -> torch.Tensor:
        """Return `x` as [B, F, L], supporting [B, D], [B, F, L], or [B, L, F]."""
        if x.dim() == 2:
            if x.shape[1] != self.sequence_length * self.num_features:
                raise ValueError(
                    f"Expected flattened input dim {self.sequence_length * self.num_features}, got {x.shape[1]}"
                )
            return x.reshape(x.shape[0], self.num_features, self.sequence_length)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D/3D input, got shape {tuple(x.shape)}")

        # [B, F, L]
        if x.shape[1] == self.num_features and x.shape[2] == self.sequence_length:
            return x

        # [B, L, F] -> [B, F, L]
        if x.shape[1] == self.sequence_length and x.shape[2] == self.num_features:
            return x.transpose(1, 2)

        raise ValueError(
            f"Could not infer phase/time axes for input shape {tuple(x.shape)} with "
            f"num_features={self.num_features}, sequence_length={self.sequence_length}"
        ) 
    