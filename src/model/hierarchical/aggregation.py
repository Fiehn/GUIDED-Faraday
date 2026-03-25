"""Aggregation modules for combining hierarchical latent representations."""

import torch
import torch.nn as nn


class ConcatAggregation(nn.Module):
    """Aggregate level-wise latents by concatenation and projection.

    This strategy concatenates all latent tensors along the feature axis and
    projects the combined representation into a shared output embedding.
    """

    def __init__(self, level_dims: list[int], output_dim: int):
        """Initialize concatenation-based aggregation.

        Args:
            level_dims: Feature dimensions for each hierarchy level latent.
            output_dim: Target embedding dimension after aggregation.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sum(level_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate latent tensors into a single representation.

        Args:
            latents: List of tensors shaped ``[batch_size, level_dim]``.

        Returns:
            Tensor of shape ``[batch_size, output_dim]``.
        """
        concat = torch.cat(latents, dim=-1)
        return self.proj(concat)


class SumAggregation(nn.Module):
    """Aggregate level-wise latents by projection and summation.

    Each level latent is projected to a common output dimension and the
    projected vectors are summed element-wise.
    """

    def __init__(self, level_dims: list[int], output_dim: int):
        """Initialize sum-based aggregation.

        Args:
            level_dims: Feature dimensions for each hierarchy level latent.
            output_dim: Target embedding dimension after aggregation.
        """
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate latent tensors by summing projected levels.

        Args:
            latents: List of tensors shaped ``[batch_size, level_dim]``.

        Returns:
            Tensor of shape ``[batch_size, output_dim]``.
        """
        projected = [proj(latent) for proj, latent in zip(self.projs, latents)]
        return torch.stack(projected, dim=0).sum(dim=0)


class GatedAggregation(nn.Module):
    """Aggregate latents using learned per-level gates.

    A gating network predicts soft weights across levels from concatenated
    latents, and the final embedding is a weighted sum of projected levels.
    """

    def __init__(self, level_dims: list[int], output_dim: int):
        """Initialize gated aggregation.

        Args:
            level_dims: Feature dimensions for each hierarchy level latent.
            output_dim: Target embedding dimension after aggregation.
        """
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])
        self.gate_net = nn.Sequential(
            nn.Linear(sum(level_dims), len(level_dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate latent tensors using learned gates.

        Args:
            latents: List of tensors shaped ``[batch_size, level_dim]``.

        Returns:
            Tensor of shape ``[batch_size, output_dim]``.
        """
        gates = self.gate_net(torch.cat(latents, dim=-1))
        projected = [proj(latent) for proj, latent in zip(self.projs, latents)]
        stacked = torch.stack(projected, dim=1)
        return (gates.unsqueeze(-1) * stacked).sum(dim=1)


class AttentionAggregation(nn.Module):
    """Aggregate level-wise latents with multi-head attention.

    Projected level latents are treated as key/value tokens, and a learnable
    query attends to them to produce a single output embedding.
    """

    def __init__(self, level_dims: list[int], output_dim: int, num_heads: int = 4):
        """Initialize attention-based aggregation.

        Args:
            level_dims: Feature dimensions for each hierarchy level latent.
            output_dim: Target embedding dimension after aggregation.
            num_heads: Number of attention heads in the aggregator.
        """
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        self.attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate latent tensors using attention over hierarchy levels.

        Args:
            latents: List of tensors shaped ``[batch_size, level_dim]``.

        Returns:
            Tensor of shape ``[batch_size, output_dim]``.
        """
        batch_size = latents[0].shape[0]
        projected = torch.stack(
            [proj(z) for proj, z in zip(self.projs, latents)],
            dim=1,
        )
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attn(query, projected, projected)
        return self.norm(attn_output.squeeze(1))


_AGGREGATION_STRATEGIES = {
    "concat": ConcatAggregation,
    "sum": SumAggregation,
    "gated": GatedAggregation,
    "attention": AttentionAggregation,
}
