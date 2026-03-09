import torch
import torch.nn as nn
from typing import Optional

from .config import HierarchicalConfig
from .levels import StrategyRegistry
from .aggregation import ConcatAggregation, SumAggregation, GatedAggregation, AttentionAggregation, _AGGREGATION_STRATEGIES

class HierarchicalLatentModule(nn.Module):
    """
    Modular hierarchical latent variable module that can be flexibly configured with different strategies at each level,
    optional top-down context, and various aggregation methods to combine the levels into a single representation.

    usage:
        # In VAE model __init__:
        if hierarchy_config is not None:
            self.hierarchical_latent = HierarchicalLatentModule(hierarchy_config, encoder_latent_dim)
        
        # In VAE model forward:
        z = self.encoder.reparameterize(q_params, num_samples).sqeeze(1)  # [B, encoder_latent_dim]
        if self.hierarchical_latent is not None:
            hier_results = self.hierarchical_latent(z)
            z = hier_results["z"]  # [B, output_dim]
            hierachy_loss = hier_results["loss"]  # scalar
        else:
            hier_results = None
            hierachy_loss = torch.tensor(0.0, device=z.device)
    """
    def __init__(self, config: HierarchicalConfig, encoder_latent_dim: int):
        super().__init__()
        self.config = config
        self.encoder_latent_dim = encoder_latent_dim
        self.num_levels = len(config.levels)

        if self.num_levels == 0:
            raise ValueError("HierarchicalLatentModule requires at least one level in the config."
                             " Add levels using config.add_level().")
        
        # Build Levels
        self.levels = nn.ModuleList()
        for i, lc in enumerate(config.levels):
            # Every level takes the encoders z as input
            # Plus optional top-down context from the previous level
            context_dim = 0
            if lc.top_down and i > 0:
                context_dim = config.levels[i-1].latent_dim

            strategy_cls = StrategyRegistry.get(lc.strategy)
            level = strategy_cls(
                input_dim=encoder_latent_dim,
                latent_dim=lc.latent_dim,
                context_dim=context_dim,
                **lc.strategy_kwargs,
            )
            self.levels.append(level)

        # Residual projections between adjacent levels
        if config.residual_connections:
            self.residual_projs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                d_from = config.levels[i].latent_dim
                d_to = config.levels[i+1].latent_dim
                if d_from != d_to:
                    self.residual_projs.append(nn.Linear(d_from, d_to))
                else:
                    self.residual_projs.append(nn.Identity())
        else: 
            self.residual_projs = None

        # Aggregation: combine all level latents into a single representation
        level_dims = [lc.latent_dim for lc in config.levels]
        out_dim = getattr(config, 'aggregation_output_dim', None) or encoder_latent_dim
        self.output_dim = out_dim

        agg_cls = _AGGREGATION_STRATEGIES.get(config.aggregation)
        if agg_cls is None:
            raise ValueError(f"Unknown aggregation strategy '{config.aggregation}'. "
                                f"Available: {list(_AGGREGATION_STRATEGIES.keys())}")
        self.aggregator = agg_cls(level_dims, out_dim)

    def forward(self, z_enc: torch.Tensor, active_levels: Optional[list] = None,) -> dict:
        """
        Args:
            z_enc: [B, encoder_latent_dim] Latent representation from the encoder
            active_levels: Optional list of level indices to activate (for ablation). If None, use all levels.
        Returns:
            dict with keys:
            - "z": [B, output_dim] Aggregated latent representation from all levels
            - "loss": Total Heirarchical latent loss (scalar)
            - "level_outputs": List of dicts for each level with keys:
                - "z": [B, latent_dim] This level's latent representation
                - "loss": This level's regularization loss (scalar)
                - "params": Any distributional parameters for this level (for logging/analysis)
        """
        if active_levels is None:
            active_levels = list(range(self.num_levels))

        z_levels = [None] * self.num_levels
        losses = [torch.tensor(0.0, device=z_enc.device)] * self.num_levels
        params_all = [{}] * self.num_levels

        # Top down pass through levels
        for i in active_levels:
            lc = self.config.levels[i]
            
            # Build context from coarser level
            context = None
            if lc.top_down and i > 0 and z_levels[i - 1] is not None:
                context = z_levels[i-1]

            # Each level sees the Same encoder output z (shared input)
            # But with the same top-down context
            results  = self.levels[i](z_enc, context=context)
            z_levels[i] = results["z"]
            losses[i] = results["loss"]
            params_all[i] = results["params"]

            # Residuals from previous level
            if self.residual_projs is not None and i > 0 and z_levels[i - 1] is not None:
                res = self.residual_projs[i-1](z_levels[i - 1])
                z_levels[i] = z_levels[i] + res

        # Aggregate active levels
        active_z = [z_levels[i] for i in active_levels if z_levels[i] is not None]
        z_out = self.aggregator(active_z)

        # Weigthed total loss
        total_loss = sum(
            self.config.levels[i].loss_weight * losses[i] for i in active_levels
        )

        return {
            "z": z_out,
            "loss": total_loss,
            "level_outputs": [
                {"z": z_levels[i], "loss": losses[i], "params": params_all[i]} for i in active_levels
            ]
        }
    
    def get_level_info(self):
        """
        Utility for logging/debugging: returns a list of dicts with info about each level's configuration.
        """
        return [
            {
                "level_index": i,
                "name": lc.name,
                "strategy": lc.strategy,
                "latent_dim": lc.latent_dim,
                "receives_top_down": lc.top_down,
                "loss_weight": lc.loss_weight,
                "params": sum(p.numel() for p in self.levels[i].parameters()),
            }
            for i, lc in enumerate(self.config.levels)
        ]

        

