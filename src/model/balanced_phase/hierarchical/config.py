from dataclasses import dataclass, field, asdict
from typing import Optional
 

@dataclass
class LevelConfig:
    """Configuration for one level in a hierarchical latent model.

    Attributes:
        name: Human-readable level identifier.
        input_dim: Input feature dimension consumed by the level.
        latent_dim: Latent embedding dimension produced by the level.
        strategy: Name of the level strategy registered in the strategy registry.
        strategy_kwargs: Optional keyword arguments forwarded to the strategy.
        top_down: Whether this level receives top-down input from the level above.
        bottom_up: Whether this level receives bottom-up input from the level below.
        hidden_dim: Optional override for hidden dimension.
        loss_weight: Relative contribution of this level in total loss.
    """
    name: str
    input_dim: int
    latent_dim: int
    strategy: str = "linear" # Key in StrategyRegistry
    strategy_kwargs: dict = field(default_factory=dict)
    # Wether this leve recieves top-down input from the level above
    top_down: bool = False
    # Wether this level recieves bottom-up input from the level below
    bottom_up: bool = False
    # Optional: override hidden dimension for this level (if different from latent_dim)
    hidden_dim: Optional[int] = None
    # Weighting for this level's loss contribution
    loss_weight: float = 1.0

    def to_dict(self) -> dict:
        """Serialize level configuration to a plain dictionary.

        Returns:
            Dictionary representation of the dataclass fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LevelConfig":
        """Create ``LevelConfig`` from a dictionary.

        Args:
            d: Dictionary containing level configuration fields.

        Returns:
            Instantiated ``LevelConfig`` object.
        """
        return cls(**d)


@dataclass
class HierarchicalConfig:
    """Configuration for the complete hierarchical model stack.

    Attributes:
        levels: Ordered list of per-level configurations.
        aggregation: Strategy for combining latent representations across levels.
        bidirectional: Whether cross-level connections are bidirectional.
        residual_connections: Whether residual links are added between levels.
        kl_annealing: Whether KL annealing is enabled during training.
        kl_warmup_steps: Number of steps used for KL warmup.
        free_bits: Minimum KL contribution per latent dimension.
    """
    levels: list[LevelConfig] = field(default_factory=list)
    # Global settings
    aggregation: str = "concat" # How to combine latent representations across levels (e.g. "concat", "sum", "attention")
    bidirectional: bool = False # Whether to allow bidirectional connections between levels (top-down and bottom-up)
    residual_connections: bool = False # Whether to add residual connections between levels
    # Training settings
    kl_annealing: bool = True # Whether to use KL annealing during training
    kl_warmup_steps: int = 1000 # Number of steps to warm up KL weight
    free_bits: float = 0.0 # Minimum KL divergence per latent dimension

    def add_level(self, level_config: LevelConfig):
        """Append a level configuration to ``levels``.

        Args:
            level_config: Level definition to append.

        Returns:
            ``self`` for fluent chaining.
        """
        self.levels.append(level_config)
        return self
    
    def to_dict(self) -> dict:
        """Serialize hierarchical configuration to a plain dictionary.

        Returns:
            Dictionary with level configs and global settings.
        """
        return {
            "levels": [level.to_dict() for level in self.levels],
            "aggregation": self.aggregation,
            "bidirectional": self.bidirectional,
            "residual_connections": self.residual_connections,
            "kl_annealing": self.kl_annealing,
            "kl_warmup_steps": self.kl_warmup_steps,
            "free_bits": self.free_bits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchicalConfig":
        """Create ``HierarchicalConfig`` from a dictionary payload.

        Args:
            d: Dictionary containing serialized hierarchical configuration.

        Returns:
            Instantiated ``HierarchicalConfig`` object.
        """
        d = dict(d)
        levels = [LevelConfig.from_dict(ld) for ld in d.pop("levels", [])]
        return cls(
            levels=levels,
            **d)

