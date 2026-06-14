from dataclasses import dataclass, field, asdict
from typing import Optional
 
@dataclass
class LevelConfig:
    """Configuration for a single level in the hierarchical model."""
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
        return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "LevelConfig":
        return cls(**d)

@dataclass
class HierarchicalConfig:
    """Configuration for the entire hierarchical model."""
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
        """Add a new level to the hierarchical model."""
        self.levels.append(level_config)
        return self
    
    def to_dict(self) -> dict:
        """ Serialize to a plain dict"""
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
        d = dict(d)
        levels = [LevelConfig.from_dict(ld) for ld in d.pop("levels", [])]
        return cls(
            levels=levels,
            **d)

