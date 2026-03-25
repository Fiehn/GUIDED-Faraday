import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

# === Normalizer class that will use the entire data ===
class Normalizer:
    """
    Normalizes data using specified methods.
    Supports both numpy arrays and torch tensors.

    Methods:
      - zero_preserved_log: Log-transform non-zero values, mark zeros with special sentinel
      - standard:           Z-score normalisation
      - log1p:              (log1p(x) - mean) / std 

    The normalizer can be fitted once across multiple files using
    fit_incremental() / finalise_fit(), and persisted to disk with
    save() / load() so stats do not need to be recalculated each run.

    Args:
        method (str): Normalisation method to use ('zero_preserved_log', 'standard', or 'log1p'). Defaults to 'log1p'.

    Attributes:
        method (str): The normalisation method in use.
        fitted (bool): Indicates if the normalizer has been fitted.
        device (str | torch.device): The device tensors are stored on.
        epsilon (float): Small value to prevent division by zero. Defaults to 1e-8.
        non_zero_mean (float | None): Mean of non-zero values (for 'zero_preserved_log').
        non_zero_std (float | None): Standard deviation of non-zero values (for 'zero_preserved_log').
        zero_id (int): Sentinel value for zeros (for 'zero_preserved_log').
        shift (float): Shift applied after normalisation (for 'zero_preserved_log').
        mean (torch.Tensor | None): Mean of the data (for 'standard' and 'log1p').
        std (torch.Tensor | None): Standard deviation of the data (for 'standard' and 'log1p').
    """
    def __init__(self, method: str = 'log1p'):
        self.method  = method
        self.fitted  = False
        self.device  = 'cpu'
        self.epsilon = 1e-8

        # zero_preserved_log
        self.non_zero_mean: Optional[float] = None
        self.non_zero_std:  Optional[float] = None
        self.zero_id = -3
        self.shift   = 1.0

        # standard / log1p
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None
    
    # === Helpers ===
    def to(self, device: str | torch.device) -> 'Normalizer':
        """Move tensor stats to device."""
        self.device = torch.device(device)

        def _move(t):
            return t.to(self.device) if isinstance(t, torch.Tensor) else t

        self.mean = _move(self.mean)
        self.std  = _move(self.std)
        return self

    def _to_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert X to float32 on self.device."""
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)
        return X.float().to(self.device)

    def fit(self, X: np.ndarray | torch.Tensor) -> 'Normalizer':
        """
        Fit the normalizer on a full dataset at once.

        For large datasets spread across multiple files prefer the incremental
        API (fit_incremental() per chunk, then finalise_fit()) which keeps peak
        RAM at O(one chunk) and is numerically equivalent.

        Args:
            X (np.ndarray | torch.Tensor): The data to fit. Shape (N, F).

        Returns:
            Normalizer: The fitted normalizer instance.
        """
        Xt = self._to_tensor(X)

        if self.method == 'zero_preserved_log':
            nz = Xt[Xt != 0]
            log_nz = torch.log(nz)
            self.non_zero_mean = log_nz.mean().item()
            self.non_zero_std  = log_nz.std().item()
        elif self.method == 'standard':
            self.mean = Xt.mean(dim=0, keepdim=True)
            self.std  = Xt.std(dim=0,  keepdim=True)
            self.std[self.std == 0] = 1.0
        elif self.method == 'log1p':
            Xl = torch.log1p(Xt.clamp(min=0.0))
            self.mean = Xl.mean(dim=0, keepdim=True)
            self.std  = Xl.std(dim=0,  keepdim=True)
            self.std[self.std == 0] = 1.0
        else:
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

        self.fitted = True
        return self
    
    # ------------------------------------------------------------------
    # Incremental (streaming) fit via batch Welford algorithm.
    # Processes one meter at a time so peak RAM is O(one meter) instead
    # of O(full training split).  Call fit_incremental() per chunk, then
    # finalise_fit() once to commit the stats, identically to fit().
    # ------------------------------------------------------------------

    def _welford_merge(self, n_a: int, mean_a: torch.Tensor, M2_a: torch.Tensor,
                       chunk: torch.Tensor):
        """Merge a new data chunk into running Welford accumulators.

        Returns updated (n, mean, M2).
        """
        n_b   = chunk.shape[0]
        mean_b = chunk.mean(dim=0, keepdim=True)
        M2_b   = ((chunk - mean_b) ** 2).sum(dim=0, keepdim=True)

        n = n_a + n_b
        if n_a == 0:
            return n_b, mean_b, M2_b

        delta    = mean_b - mean_a
        new_mean = mean_a + delta * (n_b / n)
        new_M2   = M2_a + M2_b + delta ** 2 * (n_a * n_b / n)
        return n, new_mean, new_M2

    def fit_incremental(self, X_chunk: np.ndarray | torch.Tensor) -> None:
        """
        Stream one chunk of data through the Welford online algorithm.

        Call finalise_fit() once all chunks have been streamed to commit
        the stats.  This is numerically equivalent to fit() but keeps
        peak RAM at O(one chunk) regardless of dataset size.

        Args:
            X_chunk (np.ndarray | torch.Tensor): A chunk of data to process. Shape (N, F).
        """
        Xt = self._to_tensor(X_chunk)   # (N, F)
        F  = Xt.shape[1]

        # Initialise accumulators on first call
        if not hasattr(self, '_wf_n'):
            self._wf_n    = 0
            self._wf_mean = torch.zeros(1, F, device=self.device)
            self._wf_M2   = torch.zeros(1, F, device=self.device)

        if self.method == 'log1p':
            Xt = torch.log1p(Xt.clamp(min=0.0))
        elif self.method == 'zero_preserved_log':
            nz = Xt[Xt != 0]
            if nz.numel() > 0:
                log_nz = torch.log(nz).unsqueeze(1)    # treat as (N, 1)
                self._wf_n, self._wf_mean, self._wf_M2 = self._welford_merge(
                    self._wf_n, self._wf_mean, self._wf_M2, log_nz)
            return  # handled above; no global row-level update needed

        # Global Welford update
        self._wf_n, self._wf_mean, self._wf_M2 = self._welford_merge(
            self._wf_n, self._wf_mean, self._wf_M2, Xt)

    def finalise_fit(self) -> None:
        """Commit Welford accumulators to the normalizer's parameters.

        Must be called after all fit_incremental() calls.
        Raises ValueError if no data was streamed.
        """
        if not hasattr(self, '_wf_n') or self._wf_n == 0:
            raise ValueError("No data streamed — call fit_incremental() first.")

        def _safe_std(M2: torch.Tensor, n: int) -> torch.Tensor:
            std = torch.sqrt(M2 / max(n - 1, 1))
            std[std == 0] = 1.0
            return std

        if self.method in ('standard', 'log1p'):
            self.mean  = self._wf_mean
            self.std   = _safe_std(self._wf_M2, self._wf_n)
        elif self.method == 'zero_preserved_log':
            self.non_zero_mean = float(self._wf_mean.squeeze().item())
            self.non_zero_std  = float(_safe_std(self._wf_M2, self._wf_n).squeeze().item())
        else:
            raise ValueError(f"finalise_fit() not implemented for method '{self.method}'")

        self.fitted = True

        # Release temporary Welford state
        for attr in ('_wf_n', '_wf_mean', '_wf_M2'):
            self.__dict__.pop(attr, None)

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save fitted stats to disk.

        The file is a plain torch.save dict and can be re-loaded with
        Normalizer.load(path).  It is safe to save before moving tensors
        to GPU; the file always stores stats on CPU.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            'method':        self.method,
            'fitted':        self.fitted,
            'epsilon':       self.epsilon,
            'zero_id':       self.zero_id,
            'shift':         self.shift,
            'non_zero_mean': self.non_zero_mean,
            'non_zero_std':  self.non_zero_std,
            # store tensors on CPU so the file is device-agnostic
            'mean': self.mean.cpu() if isinstance(self.mean, torch.Tensor) else self.mean,
            'std':  self.std.cpu()  if isinstance(self.std,  torch.Tensor) else self.std,
        }
        torch.save(state, path)
 
    @classmethod
    def load(cls, path: str, device: str | torch.device = 'cpu') -> 'Normalizer':
        """Load a previously saved Normalizer from disk."""
        state = torch.load(path, map_location='cpu', weights_only=False)
        norm = cls(method=state['method'])
        norm.fitted        = state['fitted']
        norm.epsilon       = state['epsilon']
        norm.zero_id       = state['zero_id']
        norm.shift         = state['shift']
        norm.non_zero_mean = state['non_zero_mean']
        norm.non_zero_std  = state['non_zero_std']
        norm.mean          = state['mean']
        norm.std           = state['std']
        norm.to(device)
        return norm

    def _zero_preserved_log_transform(self, X: torch.Tensor) -> torch.Tensor:
        is_zero = X == 0
        Y = X.clone()
        if is_zero.any():
            Y[~is_zero] = (torch.log(Y[~is_zero]) - self.non_zero_mean) / (self.non_zero_std + self.epsilon) + self.shift
            Y[is_zero]  = self.zero_id
        else:
            Y = (torch.log(Y) - self.non_zero_mean) / (self.non_zero_std + self.epsilon) + self.shift
        return Y

    def _zero_preserved_log_inverse(self, X: torch.Tensor) -> torch.Tensor:
        Y       = X.clone()
        is_zero = torch.isclose(Y, torch.tensor(float(self.zero_id), dtype=Y.dtype, device=Y.device), atol=0.1)
        Y[~is_zero] = torch.exp((Y[~is_zero] - self.shift) * (self.non_zero_std + self.epsilon) + self.non_zero_mean)
        Y[is_zero]  = 0.0
        return Y
    
    def transform(self, X) -> torch.Tensor:
        """
        Normalise X.  Always returns a float32 torch.Tensor on self.device.

        Args:
            X (np.ndarray | torch.Tensor): The data to transform. Shape (N, F).

        Returns:
            torch.Tensor: The normalised data.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. call fit() or finalise_fit() first.")
        Xt = self._to_tensor(X)

        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_transform(Xt)
        elif self.method == 'standard':
            return (Xt - self.mean) / (self.std + self.epsilon)
        elif self.method == 'log1p':
            return (torch.log1p(Xt.clamp(min=0.0)) - self.mean) / (self.std + self.epsilon)
        else:
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

    def inverse_transform(self, X) -> torch.Tensor:
        """
        Denormalise X back to the original scale.

        Args:
            X (np.ndarray | torch.Tensor): The normalised data. Shape (N, F).

        Returns:
            torch.Tensor: The denormalised data on the original scale.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. call fit() or finalise_fit() first.")
        Xt = self._to_tensor(X)

        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_inverse(Xt)
        elif self.method == 'standard':
            return Xt * self.std + self.mean
        elif self.method == 'log1p':
            return torch.clamp(
                torch.expm1(Xt * (self.std + self.epsilon) + self.mean), min=0.0
            )
        else:
            raise ValueError(f"Unknown normalisation method: '{self.method}'")
