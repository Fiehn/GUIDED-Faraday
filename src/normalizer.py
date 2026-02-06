import numpy as np
import torch

class Normalizer:
    """
    Normalizes data using specified methods.
    Supports both numpy arrays and torch tensors.
    
    Methods:
      - zero_preserved_log: Log-transform non-zero values, mark zeros with special ID
      - standard: Standard z-score normalization
    """
    def __init__(self, method='zero_preserved_log'):
        self.method = method
        self.fitted = False
        self.device = 'cpu'

        # For zero preserved log normalization
        self.non_zero_mean = None
        self.non_zero_std = None
        self.zero_id = -3
        self.shift = 1.0

        # for log1p normalization, we can use standard mean/std but with log1p transform
        self.mean = None
        self.std = None
        self.epsilon = 1e-8  # for numerical stability in standard normalization
    
    def _zero_preserved_log_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Zero-preserved log normalization."""
        is_zero = (X == 0)
        Y = X.clone()
        # Apply log transform to non-zero values
        if is_zero.any():

            Y_nonzero = Y[~is_zero]
            Y_log = torch.log(Y_nonzero)
            Y_normalized = (Y_log - self.non_zero_mean) / self.non_zero_std + self.shift
        
            # Put back
            Y[~is_zero] = Y_normalized
            Y[is_zero] = self.zero_id
        else:
            Y = (torch.log(Y) - self.non_zero_mean) / self.non_zero_std + self.shift
        
        return Y

    def _zero_preserved_log_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse of zero-preserved log normalization. Returns torch.Tensor."""
        Y = X.clone()
        is_zero = torch.isclose(Y, torch.tensor(self.zero_id, dtype=Y.dtype, device=Y.device), atol=0.1)
        
        # Inverse transform non-zero values
        Y_nonzero = Y[~is_zero]
        Y_log = (Y_nonzero - self.shift) * self.non_zero_std + self.non_zero_mean
        Y_original = torch.exp(Y_log)
        
        # Put back
        Y[~is_zero] = Y_original
        Y[is_zero] = 0.0
        
        return Y  # Returns torch.Tensor, not numpy
    
    def _log1p_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Log1p normalization."""
        return (torch.log1p(X) - self.mean) / (self.std + self.epsilon)
    def _log1p_inverse(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse of log1p normalization."""
        return torch.clamp(torch.expm1(X * (self.std + self.epsilon) + self.mean),min=0.0)

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        if self.method == 'zero_preserved_log':
            non_zero_values = X[X != 0]
            log_values = torch.log(non_zero_values)
            self.non_zero_mean = log_values.mean().item()
            self.non_zero_std = log_values.std().item()
            self.fitted = True
        elif self.method == 'standard':
            self.mean = torch.mean(X, dim=0, keepdim=True)
            self.std = torch.std(X, dim=0, keepdim=True)
            self.fitted = True
        elif self.method == 'log1p':
            log1p_values = torch.log1p(X)
            self.mean = log1p_values.mean(dim=0, keepdim=True)
            self.std = log1p_values.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0 # Avoid division by zero
            self.fitted = True
        else:
            raise ValueError(f"Normalization method {self.method} not recognized.")
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        
        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_transform(X)
        elif self.method == 'standard':
            return (X - self.mean) / self.std
        elif self.method == 'log1p':
            return self._log1p_transform(X)
        else:
            raise ValueError(f"Normalization method '{self.method}' not recognized.")

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse transform to original scale. Returns torch.Tensor."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # Convert to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            
        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_inverse(X)
        elif self.method == 'standard':
            return X * self.std + self.mean
        elif self.method == 'log1p':
            return self._log1p_inverse(X)
        else:
            raise ValueError(f"Normalization method '{self.method}' not recognized.")