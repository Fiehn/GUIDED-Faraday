import torch
import torch.nn as nn
from typing import Optional

class ConcatAggregation(nn.Module):
    """
    Concatenate latent representations from all levels and project to output_dim.
    """
    def __init__(self, level_dims: list[int], output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sum(level_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(latents, dim=-1)
        return self.proj(concat)

class SumAggregation(nn.Module):
    """
    Project each level to output_dim and sum the results.
    """
    def __init__(self, level_dims: list[int], output_dim: int):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        projected = [proj(latent) for proj, latent in zip(self.projs, latents)]
        return torch.stack(projected, dim=0).sum(dim=0)
    
class GatedAggregation(nn.Module):
    """
    Use a learned gating mechanism to weight each level's contribution.
    """
    def __init__(self, level_dims: list[int], output_dim: int):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])
        self.gate_net = nn.Sequential(
            nn.Linear(sum(level_dims), len(level_dims)),
            nn.Softmax(dim=-1),
        )

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        gates = self.gate_net(torch.cat(latents, dim=-1))  # [B, num_levels]
        projected = [proj(latent) for proj, latent in zip(self.projs, latents)]
        stacked = torch.stack(projected, dim=1)  # [B, num_levels, output_dim]
        return (gates.unsqueeze(-1) * stacked).sum(dim=1)  # [B, output_dim]
    
class AttentionAggregation(nn.Module):
    """
    Use multi-head attention to combine latent representations from all levels.
    """
    def __init__(self, level_dims: list[int], output_dim: int, num_heads: int = 4):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in level_dims])
        self.query = nn.Parameter(torch.randn(1,1,output_dim)) # Learnable query for attention
        self.attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        B = latents[0].shape[0]
        projected = torch.stack([projs(z) for projs, z in zip(self.projs, latents)], dim=1) # [B, num_levels, output_dim]
        query = self.query.expand(B, -1, -1) # [B, 1, output_dim]
        attn_output, _ = self.attn(query, projected, projected) # [B, 1, output_dim]
        return self.norm(attn_output.squeeze(1)) # [B, output_dim]
    
_AGGREGATION_STRATEGIES = {
    "concat": ConcatAggregation,
    "sum": SumAggregation,
    "gated": GatedAggregation,
    "attention": AttentionAggregation,
}