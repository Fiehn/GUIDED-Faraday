from typing import Callable, Type

class StrategyRegistry:
    """
    Registry for latent-level strategies. Allows plg-in of custom
    encoding/decoding strategies for each level in the hierarchical model.
    
    usage:
        @StrategyRegistry.register("my_strategy")
        class MyStrategy(LatentLevel):
            ...
    """
    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a new strategy."""
        def decorator(strategy_cls: Type) -> Type:
            if name in cls._registry:
                raise ValueError(
                    f"Strategy '{name}' is already registered "
                    f"({cls._registry[name].__name__}). "
                    f"Use force_register() to override."
                )
            cls._registry[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def force_register(cls, name: str) -> Callable:
        """Decorator to force register a new strategy, overriding any existing one."""
        def decorator(strategy_cls: Type) -> Type:
            cls._registry[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: [{available}]"
            )
        return cls._registry[name]
    
    @classmethod
    def available(cls) -> list[str]:
        """Return a list of available strategy names."""
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registered strategies (useful for testing)."""
        cls._registry.clear()

