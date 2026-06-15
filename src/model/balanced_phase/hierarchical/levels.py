"""Hierarchical latent level strategies and base interface."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
 
from .registry import StrategyRegistry

class LatentLevelBase(ABC,nn.Module):
    """Abstract base class for one hierarchical latent-processing level.

    Subclasses implement different latent strategies (deterministic,
    stochastic, quantized, etc.) while exposing a common output contract.
    """

    def __init__(self, input_dim: int, latent_dim: int, context_dim: int = 0, **kwargs):
        """Initialize base level dimensions.

        Args:
            input_dim: Dimension of input latent tensor.
            latent_dim: Dimension of output latent representation.
            context_dim: Optional top-down context dimension.
            **kwargs: Unused strategy-specific kwargs accepted for interface
                compatibility.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim

    @abstractmethod
    def forward(self, z_in: torch.Tensor, context: Optional[torch.Tensor] = None) -> dict:
        """Run level transformation and return latent output + regularizer.

        Args:
            z_in: Input latent tensor of shape ``[batch_size, input_dim]``.
            context: Optional top-down context tensor of shape
                ``[batch_size, context_dim]``.

        Returns:
            Dictionary with keys:
            - ``"z"``: Level latent tensor ``[batch_size, latent_dim]``.
            - ``"loss"``: Scalar regularization loss for this level.
            - ``"params"``: Optional strategy-specific diagnostics.
        """
        ...

# === Strategies ===
@StrategyRegistry.register("linear")
class LinearLevel(LatentLevelBase):
    """Deterministic linear latent projection with zero regularization loss."""

    def __init__(self, input_dim: int, latent_dim: int, context_dim: int = 0, **kwargs):
        """Initialize linear strategy.

        Args:
            input_dim: Dimension of input latent tensor.
            latent_dim: Dimension of output latent representation.
            context_dim: Optional top-down context dimension.
            **kwargs: Unused strategy-specific kwargs.
        """
        super().__init__(input_dim, latent_dim, context_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim + context_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, z_in: torch.Tensor, context: Optional[torch.Tensor] = None) -> dict:
        """Project input latent (and optional context) to deterministic output.

        Args:
            z_in: Input latent tensor of shape ``[batch_size, input_dim]``.
            context: Optional context tensor of shape
                ``[batch_size, context_dim]``.

        Returns:
            Dictionary containing projected latent ``"z"``, zero ``"loss"``,
            and empty ``"params"``.
        """
        inp = torch.cat([z_in, context], dim=-1) if context is not None else z_in
        z = self.proj(inp)
        return {"z": z, "loss": torch.tensor(0.0, device=z.device), "params": {}}
    
@StrategyRegistry.register("stochastic")
class StochasticLevel(LatentLevelBase):
    """Gaussian stochastic latent level with KL regularization.

    The level predicts ``mu`` and ``logvar`` and uses reparameterized sampling
    during training; at evaluation, it outputs ``mu`` directly.
    """

    def __init__(self, input_dim: int, latent_dim: int, context_dim: int = 0, free_bits: float = 0.0, **kwargs):
        """Initialize stochastic strategy.

        Args:
            input_dim: Dimension of input latent tensor.
            latent_dim: Dimension of output latent representation.
            context_dim: Optional top-down context dimension.
            free_bits: Per-dimension minimum KL contribution.
            **kwargs: Unused strategy-specific kwargs.
        """
        super().__init__(input_dim, latent_dim, context_dim)
        self.free_bits = free_bits
        hidden = max(latent_dim * 2, input_dim)
        total_in = input_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, z_in: torch.Tensor, context: Optional[torch.Tensor] = None) -> dict:
        """Compute stochastic latent sample and KL regularization.

        Args:
            z_in: Input latent tensor of shape ``[batch_size, input_dim]``.
            context: Optional context tensor of shape
                ``[batch_size, context_dim]``.

        Returns:
            Dictionary with sampled latent ``"z"``, scalar ``"loss"`` (KL),
            and ``"params"`` containing ``mu`` and ``logvar``.
        """
        inp = torch.cat([z_in, context], dim=-1) if context is not None else z_in
        h = self.net(inp)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # KL Divergence to standard normal
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()
        return {"z": z, "loss": kl_loss, "params": {"mu": mu, "logvar": logvar}}
    
# https://proceedings.neurips.cc/paper/2020/file/309fee4e541e51de2e41f21bebb342aa-Paper.pdf
@StrategyRegistry.register("vq")
class VQLevel(LatentLevelBase):
    """Vector-quantized latent level with straight-through estimator.

    The level projects to continuous latent space, snaps to nearest codebook
    vector, and optimizes codebook/commitment losses.
    """

    def __init__(self, input_dim: int, latent_dim: int, context_dim: int = 0, num_codes: int = 512, commitment_weight=0.25, **kwargs):
        """Initialize vector-quantized strategy.

        Args:
            input_dim: Dimension of input latent tensor.
            latent_dim: Dimension of output latent representation.
            context_dim: Optional top-down context dimension.
            num_codes: Number of codebook vectors.
            commitment_weight: Weight for commitment loss term.
            **kwargs: Unused strategy-specific kwargs.
        """
        super().__init__(input_dim, latent_dim, context_dim)
        self.commitment_weight = commitment_weight
        total_in = input_dim + context_dim

        self.proj = nn.Sequential(
            nn.Linear(total_in, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1/num_codes, 1/num_codes)
    
    def forward(self, z_in: torch.Tensor, context: Optional[torch.Tensor] = None) -> dict:
        """Quantize projected latent using nearest codebook vector.

        Args:
            z_in: Input latent tensor of shape ``[batch_size, input_dim]``.
            context: Optional context tensor of shape
                ``[batch_size, context_dim]``.

        Returns:
            Dictionary containing quantized latent ``"z"``, scalar ``"loss"``,
            and ``"params"`` with code indices/projections.
        """
        inp = torch.cat([z_in, context], dim=-1) if context is not None else z_in
        z_proj = self.proj(inp) # [B, latent_dim]
        
        # Find nearest codebook entry
        dists = torch.cdist(z_proj.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0) # [B, num_codes]
        indices = torch.argmin(dists, dim=-1) # [B]
        z_q = self.codebook(indices) # [B, latent_dim]

        # Losses
        commitment = F.mse_loss(z_proj.detach(), z_q)
        codebook_loss = F.mse_loss(z_proj, z_q.detach())
        loss = commitment * self.commitment_weight + codebook_loss

        z = z_proj + (z_q - z_proj).detach() # Straight-through estimator: forward uses z_q, backward flows to z_proj
        return {"z": z, "loss": loss, "params": {"indices": indices, "z_proj": z_proj, "z_q": z_q}}

@StrategyRegistry.register("information_bottleneck")
class InformationBottleneckLevel(LatentLevelBase):
    """Information bottleneck level with learned temperature scaling.

    A learnable temperature controls information throughput and induces a
    compression-oriented regularization objective.
    """

    def __init__(self, input_dim: int, latent_dim: int, context_dim: int = 0, initial_temp: float = 1.0, **kwargs):
        """Initialize information-bottleneck strategy.

        Args:
            input_dim: Dimension of input latent tensor.
            latent_dim: Dimension of output latent representation.
            context_dim: Optional top-down context dimension.
            initial_temp: Initial temperature value before training updates.
            **kwargs: Unused strategy-specific kwargs.
        """
        super().__init__(input_dim, latent_dim, context_dim)
        total_in = input_dim + context_dim

        self.proj = nn.Sequential(
            nn.Linear(total_in, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        # Learnable temperature parameter (log scale for stability)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temp)))
    
    def forward(self, z_in: torch.Tensor, context: Optional[torch.Tensor] = None) -> dict:
        """Apply learned temperature bottleneck and return regularization.

        Args:
            z_in: Input latent tensor of shape ``[batch_size, input_dim]``.
            context: Optional context tensor of shape
                ``[batch_size, context_dim]``.

        Returns:
            Dictionary with bottlenecked latent ``"z"``, scalar ``"loss"``,
            and ``"params"`` containing temperature diagnostics.
        """
        inp = torch.cat([z_in, context], dim=-1) if context is not None else z_in
        z_proj = self.proj(inp) # [B, latent_dim]
        
        temperature = self.log_temperature.exp()
        # Soft gating: scale features by learned temperature
        z = z_proj / (temperature + 1e-8)

        # Regularization loss to encourage lower temperature (more compression)
        rate_loss = 0.5 * (z_proj.pow(2) / (temperature + 1e-8)).sum(dim=-1).mean()
        return {"z": z, "loss": rate_loss, "params": {"temperature": temperature, "z_proj": z_proj}}
    
