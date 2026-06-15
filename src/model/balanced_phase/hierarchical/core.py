import torch
import torch.nn as nn
from typing import Optional
 
from .config import HierarchicalConfig
from .levels import StrategyRegistry
from .aggregation import ConcatAggregation, SumAggregation, GatedAggregation, AttentionAggregation, _AGGREGATION_STRATEGIES

class HierarchicalLatentModule(nn.Module):
    """Hierarchical latent block with configurable level strategies.

    The module builds multiple latent-processing levels from
    ``HierarchicalConfig``, optionally applies top-down context and residual
    links, and aggregates active levels into a single latent representation.
    """

    def __init__(self, config: HierarchicalConfig, encoder_latent_dim: int):
        """Initialize hierarchical latent module.

        Args:
            config: Hierarchical model configuration describing all levels and
                global aggregation/training options.
            encoder_latent_dim: Dimension of encoder latent input passed to each
                level.

        Raises:
            ValueError: If no levels are provided or aggregation strategy is
                unknown.
        """
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
        """Run hierarchical processing and aggregate active levels.

        Args:
            z_enc: Latent tensor from encoder with shape
                ``[batch_size, encoder_latent_dim]``.
            active_levels: Optional list of level indices to execute (e.g., for
                ablations). If ``None``, all levels are used.

        Returns:
            Dictionary with keys:
            - ``"z"``: Aggregated latent tensor of shape
              ``[batch_size, output_dim]``.
            - ``"loss"``: Weighted scalar hierarchical loss.
            - ``"level_outputs"``: Per-level dictionaries for active levels,
              each containing ``"z"``, ``"loss"``, and ``"params"``.
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
        """Return per-level configuration and parameter-count summary.

        Returns:
            List of dictionaries with level index, strategy metadata, and
            parameter counts for logging/debugging.
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

        

