"""
Sequential Phase VAE for 3-phase electrical load generation.

Autoregressive decomposition: Phase 1 independent, Phase 2 conditioned on Phase 1,
Phase 3 conditioned on Phases 1 and 2. All three models share the same Conditioner.
"""

import torch
import torch.nn as nn
from typing import Dict
from src.model.cvae import CVAE
from src.model.vae import HierarchicalConfig
from src.data.conditioner import Conditioner
from src.model.model import BaseModel


class SequentialPhaseVAE(BaseModel):
    """Autoregressive sequential model for 3-phase electrical load."""
    
    def __init__(
        self,
        input_dim: int,
        *,
        conditioner: Conditioner,
        latent_dim: int = 32,
        distribution_dict: dict = None,
        prior_type: str = None,
        hierarchy_config: HierarchicalConfig = None,
        phased_mode: bool = False,
        sequence_length: int = 24,
        **cvae_kwargs,
    ):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length)
        
        # In phased mode, input_dim is for all 3 phases combined (e.g., 24*3=72)
        # We need individual phase dimensions accounting for concatenation
        if phased_mode:
            input_dim_per_phase = input_dim // 3
        else:
            input_dim_per_phase = input_dim
        
        self.input_dim = input_dim_per_phase
        self.latent_dim = latent_dim
        self.conditioner = conditioner
        self.hierarchy_config = hierarchy_config
        self.sequence_length = sequence_length
        self.cvae_kwargs = cvae_kwargs
        self.distribution_dict = distribution_dict or {}
        self.prior_type = prior_type
        
        # Phase 1: Independent (input_dim = 24)
        self.phase_1_model = CVAE(
            input_dim=input_dim_per_phase,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            phased_mode=False,
            sequence_length=sequence_length,
            **cvae_kwargs,
        )
        
        # Phase 2: Conditioned on Phase 1 (input_dim = 24 + 24 = 48)
        # Receives concatenated [phase_2, phase_1]
        self.phase_2_model = CVAE(
            input_dim=input_dim_per_phase * 2,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            phased_mode=False,
            sequence_length=sequence_length,
            **cvae_kwargs,
        )
        
        # Phase 3: Conditioned on Phases 1 and 2 (input_dim = 24 + 24 + 24 = 72)
        # Receives concatenated [phase_3, phase_1, phase_2]
        self.phase_3_model = CVAE(
            input_dim=input_dim_per_phase * 3,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            phased_mode=False,
            sequence_length=sequence_length,
            **cvae_kwargs,
        )
        
        self.phase_models = nn.ModuleList([
            self.phase_1_model, self.phase_2_model, self.phase_3_model
        ])
        self.num_parameters = sum(m.num_parameters for m in self.phase_models)
    
    @property
    def encoder(self):
        """Provide access to phase_1_model encoder for compatibility."""
        return self.phase_1_model.encoder
    
    @property
    def decoder(self):
        """Provide access to phase_1_model decoder for compatibility."""
        return self.phase_1_model.decoder
    
    @property
    def prior(self):
        """Provide access to phase_1_model prior for compatibility."""
        return self.phase_1_model.prior
    
    @property
    def hierarchy(self):
        """Provide access to phase_1_model hierarchy for compatibility."""
        return self.phase_1_model.hierarchy
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, num_samples: int = 1) -> Dict:
        """Forward pass through all three phase models (teacher forcing)."""
        if x.dim() == 2:
            x = x.reshape(x.shape[0], 3, self.input_dim)
        
        phase_1 = x[:, 0, :]
        phase_2 = x[:, 1, :]
        phase_3 = x[:, 2, :]
        
        p1_out = self.phase_1_model(phase_1, c, num_samples=num_samples)
        
        # Concatenate phase_2 with phase_1 for phase_2 model input
        p2_input = torch.cat([phase_2, phase_1], dim=-1)
        p2_out = self.phase_2_model(p2_input, c, num_samples=num_samples)
        
        # Concatenate phase_3 with phases_1 and 2 for phase_3 model input
        p3_input = torch.cat([phase_3, phase_1, phase_2], dim=-1)
        p3_out = self.phase_3_model(p3_input, c, num_samples=num_samples)
        
        recon_loss = (
            p1_out.get("recon_loss", 0) +
            p2_out.get("recon_loss", 0) +
            p3_out.get("recon_loss", 0)
        )
        kl_loss = (
            p1_out.get("kl_loss", 0) +
            p2_out.get("kl_loss", 0) +
            p3_out.get("kl_loss", 0)
        )
        
        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "phase_outputs": [p1_out, p2_out, p3_out],
            "phases": [phase_1, phase_2, phase_3],
        }
    
    def generate(
        self,
        conditions: torch.Tensor,
        num_samples: int = 1,
        return_all_phases: bool = False,
    ) -> torch.Tensor:
        """Generate 3-phase data autoregressively: Phase 1 → Phase 2|1 → Phase 3|1,2."""
        with torch.no_grad():
            # Generate Phase 1 (independent) using CVAE.sample()
            # Returns [S=1, num_x_samples=num_samples, B, D]
            phase_1_samples = self.phase_1_model.sample(conditions, num_prior_samples=1, num_x_samples=num_samples)
            # Extract to [num_samples, B, D]
            phase_1 = phase_1_samples[0]  # [num_samples, B, D]
            
            B = conditions.size(0)
            if num_samples > 1:
                conditions_exp = conditions.repeat_interleave(num_samples, dim=0)
                phase_1_flat = phase_1.reshape(num_samples * B, -1)
            else:
                conditions_exp = conditions
                phase_1_flat = phase_1.squeeze(0)  # [1, B, D] -> [B, D]
            
            # Generate Phase 2 conditioned on Phase 1
            # Create input by concatenating phase_2_sample with phase_1
            # For generation, we create a dummy phase_2 and concatenate with phase_1
            dummy_phase_2 = torch.zeros_like(phase_1_flat)  # [num_samples*B, D]
            p2_x_input = torch.cat([dummy_phase_2, phase_1_flat], dim=-1)  # [num_samples*B, 2*D]
            
            # Sample from phase_2 model: this samples x given z and conditions
            # The model will generate phase_2 data conditioned on phase_1
            phase_2_samples = self.phase_2_model.sample(conditions_exp, num_prior_samples=1, num_x_samples=1)
            # Extract [num_samples*B, D_combined=48]
            phase_2_full = phase_2_samples[0, 0]  # [num_samples*B, 48]
            # Extract phase_2 part (first self.input_dim = 24)
            phase_2_flat = phase_2_full[:, :self.input_dim]  # [num_samples*B, D]
            if num_samples > 1:
                phase_2 = phase_2_flat.reshape(num_samples, B, -1)
            else:
                phase_2 = phase_2_flat  # [B, D]
            
            # Generate Phase 3 conditioned on Phases 1 and 2
            # Phase 3 samples conditioned on concatenated phases 1 and 2
            phase_3_samples = self.phase_3_model.sample(conditions_exp, num_prior_samples=1, num_x_samples=1)
            # Extract [num_samples*B, D_combined=72]
            phase_3_full = phase_3_samples[0, 0]  # [num_samples*B, 72]
            # Extract phase_3 part (first self.input_dim = 24)
            phase_3_flat = phase_3_full[:, :self.input_dim]  # [num_samples*B, D]
            if num_samples > 1:
                phase_3 = phase_3_flat.reshape(num_samples, B, -1)
            else:
                phase_3 = phase_3_flat  # [B, D]
            
            # Ensure consistent dimensions
            if num_samples == 1:
                phase_1 = phase_1.squeeze(0)  # [1, B, D] -> [B, D]
        
        if return_all_phases:
            if num_samples > 1:
                return torch.stack([phase_1, phase_2, phase_3], dim=1)  # [num_samples, 3, B, D]
            else:
                return torch.stack([phase_1, phase_2, phase_3], dim=0).unsqueeze(0)  # [1, 3, B, D]
        else:
            if num_samples > 1:
                return torch.cat([phase_1, phase_2, phase_3], dim=2)  # [num_samples, B, 3*D]
            else:
                return torch.cat([phase_1, phase_2, phase_3], dim=1)  # [B, 3*D]
    
    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hierarchy_config": self.hierarchy_config.to_dict() if self.hierarchy_config else None,
            "phased_mode": self.phased_mode,
            "sequence_length": self.sequence_length,
            "cvae_kwargs": self.cvae_kwargs,
            "distribution_dict": self.distribution_dict,
            "prior_type": self.prior_type,
        }
    
    @classmethod
    def from_dict(cls, config: dict, conditioner: Conditioner) -> "SequentialPhaseVAE":
        hierarchy_config = None
        if config.get("hierarchy_config"):
            hierarchy_config = HierarchicalConfig.from_dict(config["hierarchy_config"])
        
        return cls(
            input_dim=config["input_dim"],
            conditioner=conditioner,
            latent_dim=config["latent_dim"],
            hierarchy_config=hierarchy_config,
            phased_mode=config.get("phased_mode", False),
            sequence_length=config.get("sequence_length", 24),
            distribution_dict=config.get("distribution_dict"),
            prior_type=config.get("prior_type"),
            **config.get("cvae_kwargs", {}),
        )
    
    def train_core(self, batch, optim, beta=1.0, weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        """Train all three phase models with teacher forcing (ground truth phase conditioning)."""
        self.train()
        
        # Unpack batch
        x, meter_id, timestamps, agg = self.phase_1_model._unpack_batch(batch)
        
        # x is in phased format [B, 72] or [B, 3, 24]
        # Reshape to [B, 3, 24]
        if x.dim() == 2:
            x = x.reshape(x.shape[0], 3, self.input_dim)
        
        phase_1_data = x[:, 0, :]  # [B, 24]
        phase_2_data = x[:, 1, :]
        phase_3_data = x[:, 2, :]
        
        # Build base conditions from phase 1 data
        x_phase_first = self.phase_1_model._to_phase_first(phase_1_data)
        c = self.phase_1_model._build_conditions(
            x_phase_first=x_phase_first,
            meter_id=meter_id,
            timestamps=timestamps
        )
        
        optim.zero_grad(set_to_none=True)
        
        # Train Phase 1 (independent)
        p1_recon, p1_q_params = self.phase_1_model.forward(phase_1_data, c, num_samples=1)
        p1_loss_dict = self.phase_1_model.loss(phase_1_data, p1_recon, p1_q_params, beta=beta, weighted_loss=weighted_loss)
        
        # Train Phase 2 conditioned on Phase 1 (teacher forcing with ground truth phase_1)
        p2_input = torch.cat([phase_2_data, phase_1_data], dim=-1)  # [B, 48]
        p2_recon, p2_q_params = self.phase_2_model.forward(p2_input, c, num_samples=1)
        p2_loss_dict = self.phase_2_model.loss(p2_input, p2_recon, p2_q_params, beta=beta, weighted_loss=weighted_loss)
        
        # Train Phase 3 conditioned on Phases 1 and 2 (teacher forcing with ground truth)
        p3_input = torch.cat([phase_3_data, phase_1_data, phase_2_data], dim=-1)  # [B, 72]
        p3_recon, p3_q_params = self.phase_3_model.forward(p3_input, c, num_samples=1)
        p3_loss_dict = self.phase_3_model.loss(p3_input, p3_recon, p3_q_params, beta=beta, weighted_loss=weighted_loss)
        
        # Aggregate losses
        total_loss = p1_loss_dict["loss"] + p2_loss_dict["loss"] + p3_loss_dict["loss"]
        
        total_loss.backward()
        
        if gradient_clip is not None:
            gradient_clip_kwargs = gradient_clip_kwargs or {}
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))
        
        optim.step()
        
        # Return losses as tensors
        return {
            "loss": total_loss / 3,  # Average across phases
            "recon_loglikelihood": (p1_loss_dict.get("recon_loglikelihood", 0) +
                                   p2_loss_dict.get("recon_loglikelihood", 0) +
                                   p3_loss_dict.get("recon_loglikelihood", 0)) / 3,
            "kl_divergence": (p1_loss_dict.get("kl_divergence", 0) +
                             p2_loss_dict.get("kl_divergence", 0) +
                             p3_loss_dict.get("kl_divergence", 0)) / 3,
        }
    
    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        """Validate all three phase models with teacher forcing."""
        self.eval()
        metrics = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0
        
        for batch in valloader:
            x, meter_id, timestamps, _agg = self.phase_1_model._unpack_batch(batch)
            
            # Reshape x to [B, 3, 24]
            if x.dim() == 2:
                x = x.reshape(x.shape[0], 3, self.input_dim)
            
            x_3d = x.to(device)
            phase_1_data = x_3d[:, 0, :]
            phase_2_data = x_3d[:, 1, :]
            phase_3_data = x_3d[:, 2, :]
            
            # Build conditions
            x_phase_first = self.phase_1_model._to_phase_first(phase_1_data)
            c = self.phase_1_model._build_conditions(
                x_phase_first=x_phase_first,
                meter_id=meter_id,
                timestamps=timestamps
            )
            
            # Validate Phase 1
            p1_recon, p1_q_params = self.phase_1_model.forward(phase_1_data, c, num_samples=1)
            p1_loss_dict = self.phase_1_model.loss(phase_1_data, p1_recon, p1_q_params, beta=beta)
            
            # Validate Phase 2 with ground truth Phase 1
            p2_input = torch.cat([phase_2_data, phase_1_data], dim=-1)
            p2_recon, p2_q_params = self.phase_2_model.forward(p2_input, c, num_samples=1)
            p2_loss_dict = self.phase_2_model.loss(p2_input, p2_recon, p2_q_params, beta=beta)
            
            # Validate Phase 3 with ground truth Phases 1 and 2
            p3_input = torch.cat([phase_3_data, phase_1_data, phase_2_data], dim=-1)
            p3_recon, p3_q_params = self.phase_3_model.forward(p3_input, c, num_samples=1)
            p3_loss_dict = self.phase_3_model.loss(p3_input, p3_recon, p3_q_params, beta=beta)
            
            # Aggregate
            metrics["loss"] += (p1_loss_dict["loss"] + p2_loss_dict["loss"] + p3_loss_dict["loss"]).item() / 3
            metrics["recon_loglikelihood"] += (
                p1_loss_dict.get("recon_loglikelihood", 0) +
                p2_loss_dict.get("recon_loglikelihood", 0) +
                p3_loss_dict.get("recon_loglikelihood", 0)
            ) / 3
            metrics["kl_divergence"] += (
                p1_loss_dict.get("kl_divergence", 0) +
                p2_loss_dict.get("kl_divergence", 0) +
                p3_loss_dict.get("kl_divergence", 0)
            ) / 3
            
            num_batches += 1
        
        for k in metrics:
            metrics[k] /= max(num_batches, 1)
        
        return metrics
    
    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu", **kwargs):
        """Sample 3-phase data autoregressively."""
        self.eval()
        original_device = next(self.parameters()).device
        self.to(device)
        
        # Generate samples using the generate method
        samples = self.generate(conditions, num_samples=num_prior_samples, return_all_phases=False)
        
        self.to(original_device)
        return samples
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, meter_ids: torch.Tensor, timestamps: torch.Tensor, num_samples: int = 1):
        """Reconstruct 3-phase input data with teacher forcing using mean reconstruction."""
        self.eval()
        
        # Handle different input formats
        B = x.shape[0]
        
        if x.dim() == 3:
            # Check if shape is [B, 24, 3] (timesteps, phases) or [B, 3, 24] (phases, timesteps)
            if x.shape[2] == 3 and x.shape[1] == self.input_dim:
                # Format: [B, seq_len, num_phases] - extract phases from last dimension
                phase_1_data = x[:, :, 0]  # [B, 24]
                phase_2_data = x[:, :, 1]  # [B, 24]
                phase_3_data = x[:, :, 2]  # [B, 24]
            elif x.shape[1] == 3 and x.shape[2] == self.input_dim:
                # Format: [B, 3, seq_len] - extract phases from second dimension
                phase_1_data = x[:, 0, :]  # [B, 24]
                phase_2_data = x[:, 1, :]  # [B, 24]
                phase_3_data = x[:, 2, :]  # [B, 24]
            else:
                raise ValueError(f"Unexpected 3D shape: {x.shape}. Expected [B, {self.input_dim}, 3] or [B, 3, {self.input_dim}]")
        elif x.dim() == 2 and x.shape[1] == 3 * self.input_dim:
            # In [B, 72] flattened format - reshape to [B, 3, 24]
            x = x.reshape(B, 3, self.input_dim)
            phase_1_data = x[:, 0, :]
            phase_2_data = x[:, 1, :]
            phase_3_data = x[:, 2, :]
        elif x.dim() == 2 and x.shape[1] == self.input_dim:
            # Single phase [B, 24] - treat as phase 1 only (shouldn't happen but handle gracefully)
            phase_1_data = x
            phase_2_data = x.clone()  # Use same data for other phases
            phase_3_data = x.clone()
        else:
            raise ValueError(f"Unexpected input shape for reconstruct: {x.shape}. Expected [B, {self.input_dim}, 3] or [B, 3, {self.input_dim}] or [B, {3*self.input_dim}]")
        
        # Build conditions using conditioner directly
        x_phase_first = self.phase_1_model._to_phase_first(phase_1_data)
        c = self.conditioner.get_conditions(x_phase_first, meter_id=meter_ids, timestamps=timestamps)
        c = c.to(phase_1_data.device)
        
        # Reconstruct Phase 1 (independent) - use mean reconstruction
        p1_recon_params, _ = self.phase_1_model.forward(phase_1_data, c, num_samples=1)
        # Extract mean [B, D] from [1, B, D]
        p1_decoded = p1_recon_params['mu'][0]
        
        # Reconstruct Phase 2 (with ground truth Phase 1) - use mean reconstruction
        p2_input = torch.cat([phase_2_data, phase_1_data], dim=-1)
        p2_recon_params, _ = self.phase_2_model.forward(p2_input, c, num_samples=1)
        # Extract mean and get phase_2 part (first self.input_dim)
        p2_decoded_full = p2_recon_params['mu'][0]  # [B, D_combined=48]
        p2_decoded = p2_decoded_full[:, :self.input_dim]  # [B, D]
        
        # Reconstruct Phase 3 (with ground truth Phases 1 and 2) - use mean reconstruction
        p3_input = torch.cat([phase_3_data, phase_1_data, phase_2_data], dim=-1)
        p3_recon_params, _ = self.phase_3_model.forward(p3_input, c, num_samples=1)
        # Extract mean and get phase_3 part (first self.input_dim)
        p3_decoded_full = p3_recon_params['mu'][0]  # [B, D_combined=72]
        p3_decoded = p3_decoded_full[:, :self.input_dim]  # [B, D]
        
        # Stack reconstructions: return in same format as input
        # If input was [B, 24, 3], return [B, 24, 3]; if [B, 3, 24], return [B, 3, 24]
        if x.shape[2] == 3:
            return torch.stack([p1_decoded, p2_decoded, p3_decoded], dim=2)  # [B, 24, 3]
        else:
            return torch.stack([p1_decoded, p2_decoded, p3_decoded], dim=1)  # [B, 3, 24]
    
    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        """Return Adam optimizer for all model parameters."""
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def loss(self, x, recon_params, q_params, beta=1.0, weighted_loss=0.0):
        """Compute loss for compatibility (delegates to phase_1_model)."""
        return self.phase_1_model.loss(x, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
