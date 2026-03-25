
from typing import Any

# ==============
# Cyclical Beta Annealing Scheduler
# Reference: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
# (Fu et al., 2019) https://arxiv.org/abs/1903.10145
# ==============
class CyclicalBetaScheduler:
    """Cyclical annealing schedule for the KL weight (beta) in VAE training.

    Each cycle consists of a linear warmup from 0 to *beta_max* followed by a
    constant phase at *beta_max*.  The proportion of each cycle spent warming
    up is controlled by *ratio* (default 0.5 = half warmup, half constant).

    Parameters
    ----------
    total_steps : int
        Total number of training steps (epochs * batches_per_epoch).
    n_cycles : int
        Number of full annealing cycles over training (default 4).
    ratio : float
        Fraction of each cycle devoted to the linear warmup (0-1, default 0.5).
    beta_max : float
        Maximum beta value reached at the top of each cycle (default 1.0).
    """

    def __init__(self, total_steps: int, n_cycles: int = 4, ratio: float = 0.5, beta_max: float = 1.0):
        assert total_steps > 0, "total_steps must be > 0"
        assert n_cycles >= 1, "n_cycles must be >= 1"
        assert 0.0 < ratio <= 1.0, "ratio must be in (0, 1]"
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.beta_max = beta_max
        self._step = 0

    def step(self) -> float:
        """Advance one step and return the current beta value."""
        beta = self._get_beta(self._step)
        self._step += 1
        return beta

    def _get_beta(self, step: int) -> float:
        """Compute beta for a given global step."""
        # How far through training are we (0..1)
        tau = step / self.total_steps
        # Which cycle are we in, and where inside that cycle (0..1)
        cycle_progress = (tau * self.n_cycles) % 1.0
        if cycle_progress < self.ratio:
            # Linear warmup phase
            return self.beta_max * (cycle_progress / self.ratio)
        else:
            # Constant phase
            return self.beta_max

    def state_dict(self) -> dict[str, int | float]:
        """Return serializable scheduler state.

        Returns:
            Dictionary containing step counter and schedule hyperparameters.
        """
        return {"step": self._step, "total_steps": self.total_steps,
                "n_cycles": self.n_cycles, "ratio": self.ratio, "beta_max": self.beta_max}

    @classmethod
    def from_state_dict(cls, d: dict[str, int | float]) -> "CyclicalBetaScheduler":
        """Create scheduler from serialized state.

        Args:
            d: State dictionary as produced by `state_dict()`.

        Returns:
            Restored `CyclicalBetaScheduler` instance.
        """
        obj = cls(d["total_steps"], d["n_cycles"], d["ratio"], d["beta_max"])
        obj._step = d["step"]
        return obj

    def __repr__(self) -> str:
        """Return concise string representation of scheduler state."""
        return (f"CyclicalBetaScheduler(total_steps={self.total_steps}, n_cycles={self.n_cycles}, "
                f"ratio={self.ratio}, beta_max={self.beta_max}, current_step={self._step})")


class EarlyStopping:
    """Early stopping helper with optional overfitting guard.

    Training stops when validation improvement stagnates for `patience` checks,
    or when train/validation metrics become too close under a simple overfitting
    heuristic.

    Attributes:
        patience: Maximum non-improving checks before stopping.
        min_delta: Minimum required metric improvement.
        counter: Number of consecutive non-improving checks.
        best_score: Best transformed validation score observed so far.
        early_stop: Stop flag indicating training should terminate.
        best_epoch: Epoch index where best score was observed.
        mode: Optimization direction (`min` or `max`).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min') -> None:
        """Initialize early stopping state.

        Args:
            patience: Number of non-improving checks allowed.
            min_delta: Minimum improvement threshold.
            mode: Whether lower (`min`) or higher (`max`) val metric is better.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.mode = mode

    def __call__(self,
                 train_metric: float | None,
                 val_metric: float | None,
                 epoch: int | None = None) -> bool:
        """Update stopper with latest metrics.

        Args:
            train_metric: Current training metric.
            val_metric: Current validation metric.
            epoch: Optional epoch index.

        Returns:
            `True` if current point is considered an improvement, else `False`.
        """
        score = -val_metric if self.mode == 'min' else val_metric

        if train_metric is not None and val_metric is not None:
            if abs(train_metric - val_metric) < self.min_delta * 10:
                # Overfitting detected - Stop training
                self.early_stop = True
                return False

        if self.best_score is None:
            # First time call
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False                
        else:
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            return True

    def reset(self) -> None:
        """Reset internal state for a fresh training run."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

