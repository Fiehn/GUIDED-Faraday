import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# === Normalizer class that will use the entire data ===
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
        

# === RevIN (reversible instance normalization) for online normalization ===
# Refrence: https://openreview.net/forum?id=cGDAkQo1C0p
# With refrence dict to dates seen 
class IndexedLogRevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, device='cpu'):
        """
        Reversible Instance Normalization (RevIN) for time series data.
        Args:
            num_features: Number of features in the input data (e.g., number of channels).
            eps: Small constant for numerical stability.
            affine: If True, applies learnable affine transformation after normalization.
        """
        super(IndexedLogRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = device # TODO: get everything to device 

        if self.affine:
            self.__init_params()

        # === Indexing for storing stats per date ===
        # Will store the mean and std for each date (or other time period) to allow for proper denormalization later
        self.daily_stats = defaultdict(lambda: {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0})

        # Global stats for fallback (if we encounter a date we haven't seen before during denormalization)
        self.global_stats = {'mean': 0.0, 'std': 0.0, 'count': 0}


    def __init_params(self):
        # Learnable parameters for affine transformation
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str = 'norm', timestamp=None, month=None):
        """
        Forward pass for RevIN.
        Args:
            x: Input tensor of shape (batch_size, num_features, sequence_length).
            mode: 'norm' for normalization, 'denorm' for denormalization.
        Returns:
            Normalized or denormalized tensor of the same shape as input.
        """
        if mode == 'norm':
            # For online normalization, we compute stats on the fly for each batch
            self._get_stats(x)
            x = self._normalize(x)

            if timestamp is not None:
                self._update_index(x, timestamp)
            return x
        elif mode == 'denorm':
            # Ensure we have stats for denormalization
            if timestamp is not None:
                mean, std = self._get_scale_for_generation(timestamp, month)
                self.mean = mean.view(1, -1, 1).to(x.device)
                self.std = std.view(1, -1, 1).to(x.device)
            elif not hasattr(self, 'mean') or not hasattr(self, 'std'):
                # Use default stats if no timestamp and no cached stats
                self.mean = torch.zeros(1, self.num_features, 1).to(x.device)
                self.std = torch.ones(1, self.num_features, 1).to(x.device)
            
            x = self._denormalize(x)
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")
        return x
    
    def _update_index(self, x, timestamp):
        """ Update the index with the mean and std for the given timestamp (e.g., date).
        This allows us to store the normalization parameters for each date, which can be used later for proper denormalization.
        """
        # Handle case where timestamp is a tensor with multiple values (batch)
        if isinstance(timestamp, torch.Tensor):
            timestamps_cpu = timestamp.detach().cpu().numpy()
        else:
            timestamps_cpu = np.array(timestamp) if not isinstance(timestamp, np.ndarray) else timestamp
        
        # Ensure timestamps_cpu is 1D
        if timestamps_cpu.ndim == 0:
            timestamps_cpu = np.array([timestamps_cpu])
        elif timestamps_cpu.ndim > 1:
            timestamps_cpu = timestamps_cpu.flatten()
        
        # Convert timestamps to dates (remove time component)
        dates = timestamps_cpu // (24 * 3600)
        
        # Get unique dates in this batch
        unique_dates = np.unique(dates)
        
        # Process each unique date separately
        for date_val in unique_dates:
            # Find indices for this date
            date_indices = np.where(dates == date_val)[0]
            
            # Extract data for this date only
            if isinstance(x, torch.Tensor):
                x_date = x[date_indices]
                date_mean = torch.mean(x_date, dim=[0, 2]).detach().cpu().numpy()  # Mean per feature
                date_std = torch.sqrt(torch.var(x_date, dim=[0, 2], unbiased=False) + self.eps).detach().cpu().numpy()  # Std per feature
            else:
                x_date = x[date_indices]
                date_mean = np.mean(x_date, axis=(0, 2))  # Mean per feature
                date_std = np.sqrt(np.var(x_date, axis=(0, 2), ddof=0) + self.eps)  # Std per feature
            
            # Update the index for this specific date
            stats = self.daily_stats[int(date_val)]
            stats['sum_mean'] += date_mean
            stats['sum_std'] += date_std
            stats['count'] += 1
            
            # Update global stats as well
            self.global_stats['mean'] += date_mean
            self.global_stats['std'] += date_std
            self.global_stats['count'] += 1

    def _get_scale_for_generation(self, timestamp, month=None):
        """ Get the mean and std for the given timestamp (date) to use for proper denormalization during generation.
        If we haven't seen this date before, we can fall back to global mean/std or some default values.
        """
        if month is None:
            target_date = timestamp // (24 * 3600)
            if target_date in self.daily_stats and self.daily_stats[target_date]['count'] > 0:
                stats = self.daily_stats[target_date]
                mean = stats['sum_mean'] / stats['count']
                std = stats['sum_std'] / stats['count']
            elif self.global_stats['count'] > 0:
                mean = self.global_stats['mean'] / self.global_stats['count']
                std = self.global_stats['std'] / self.global_stats['count']
            else:
                # If we have no data at all, return default values (e.g., mean=0, std=1)
                mean = np.zeros(self.num_features)
                std = np.ones(self.num_features)
            return torch.tensor(mean), torch.tensor(std)
        else:
            # Average over all dates in the same month
            month_dates = [date for date in self.daily_stats if (date // 30) == month]
            if month_dates:
                mean = np.mean([self.daily_stats[date]['sum_mean'] / self.daily_stats[date]['count'] for date in month_dates], axis=0)
                std = np.mean([self.daily_stats[date]['sum_std'] / self.daily_stats[date]['count'] for date in month_dates], axis=0)
            elif self.global_stats['count'] > 0:
                mean = self.global_stats['mean'] / self.global_stats['count']
                std = self.global_stats['std'] / self.global_stats['count']
            else:
                mean = np.zeros(self.num_features)
                std = np.ones(self.num_features)
            return torch.tensor(mean), torch.tensor(std)

    def _get_stats(self, x):
        # For log1p normalization, we need to compute stats on log-transformed data
        x_log = torch.log1p(torch.clamp(x, min=0.0))  # Ensure non-negative input for log1p
        self.mean = torch.mean(x_log, dim=1, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x_log, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
    
    def _normalize(self, x):
        # Apply log1p transformation first, then normalize
        x_log = torch.log1p(torch.clamp(x, min=0.0))  # Ensure non-negative input
        x_norm = (x_log - self.mean) / (self.std + self.eps)  # Add eps to avoid division by zero
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x_norm
    
    def _denormalize(self, x):
        if self.affine:
            x = (x - self.bias.view(1, -1, 1)) / (self.weight.view(1, -1, 1) + self.eps)
        # Denormalize then apply expm1 to inverse log1p
        x_denorm = x * (self.std + self.eps) + self.mean
        x_original = torch.expm1(x_denorm)
        return torch.clamp(x_original, min=0.0)  # Ensure non-negative output

