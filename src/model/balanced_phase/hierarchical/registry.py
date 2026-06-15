from typing import Callable, Type


class StrategyRegistry:
    """Registry for hierarchical latent-level strategy classes.

    This registry maps string keys to strategy class objects and is used by
    hierarchical modules to instantiate levels from configuration names.
    """

    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Create a decorator that registers a new strategy class.

        Args:
            name: Registry key for the strategy implementation.

        Returns:
            Decorator that stores the decorated class under ``name``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
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
        """Create a decorator that registers and overwrites existing entries.

        Args:
            name: Registry key for the strategy implementation.

        Returns:
            Decorator that stores the decorated class under ``name``.
        """
        def decorator(strategy_cls: Type) -> Type:
            cls._registry[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type:
        """Retrieve a registered strategy class by name.

        Args:
            name: Registry key to resolve.

        Returns:
            Strategy class associated with ``name``.

        Raises:
            KeyError: If ``name`` is not present in the registry.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: [{available}]"
            )
        return cls._registry[name]
    
    @classmethod
    def available(cls) -> list[str]:
        """Return all currently registered strategy names.

        Returns:
            List of registry keys.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """Remove all registered strategy mappings.

        Returns:
            None.
        """
        cls._registry.clear()

