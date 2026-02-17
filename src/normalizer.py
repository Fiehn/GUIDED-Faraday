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
    """
    Reversible Instance Normalization with log1p transform for non-negative time series.
    
    Input shape: [B, L, F]  (batch, sequence_length, num_features)
    
    Normalization pipeline:
        1. x_log = log1p(clamp(x, min=0))
        2. mean, std = per-instance stats over time dim (dim=1)
        3. x_norm = (x_log - mean) / std
        4. x_norm = x_norm * gamma + beta  (if affine)
    
    Denormalization pipeline:
        1. x = (x - beta) / gamma  (if affine)
        2. x_log = x * std + mean
        3. x = expm1(x_log)
        4. x = clamp(x, min=0)
    
    Index:
        Stores running mean/std of the log1p-transformed data per calendar date,
        enabling denormalization during generation when we don't have the original
        instance stats.
    """
    def __init__(self, num_features, eps=1e-5, affine=True, device='cpu', 
                 ema_alpha=0.05, min_std=0.1):
        """
        Args:
            num_features: Number of features (F) in [B, L, F] input.
            eps: Numerical stability constant.
            affine: If True, applies learnable affine transform after normalization.
            device: Device for parameters.
            ema_alpha: EMA smoothing factor for daily stats (0 = no update, 1 = full replace).
                       Higher values give more weight to recent observations.
            min_std: Minimum standard deviation to prevent division by near-zero.
                     This is critical for constant or near-constant windows.
        """
        super(IndexedLogRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = device # TODO: get everything to device 
        self.ema_alpha = ema_alpha
        self.min_std = min_std

        if self.affine:
            # Learnable affine parameters: gamma (scale) and beta (shift)
            # Initialized to identity transform (gamma=1, beta=0)
            self.weight = nn.Parameter(torch.ones(num_features, device=device))
            self.bias = nn.Parameter(torch.zeros(num_features, device=device))

        # === Per-date index for generation-time denormalization ===
        # Stores EMA of mean/std of log1p-transformed data per calendar month
        self.monthly_stats = {}  # Key: int (month 1-12 or year*12+month), Value: dict with mean, std, count

        # Global running stats as fallback
        self._global_mean_sum = np.zeros(num_features, dtype=np.float64)
        self._global_std_sum = np.zeros(num_features, dtype=np.float64)
        self._global_count = 0

        # Per-instance stats cached for denormalization in the same forward pass
        # These are set during _normalize and used during _denormalize
        self._cached_mean = None  # [B, 1, F]
        self._cached_std = None   # [B, 1, F]

    def forward(self, x, mode='norm', timestamp=None, month=None, month_key=None):
        """
        Args:
            x: [B, L, F] tensor.
            mode: 'norm' to normalize, 'denorm' to denormalize.
            timestamp: Optional timestamps for stat retrieval during generation.
            month: Optional month (1-12) for generation denorm.
            month_key: Optional int (e.g., 202310) to select which monthly stats 
                       to normalize/denormalize with. If None, uses the most recent.
        Returns:
            Transformed tensor of same shape [B, L, F].
        """
        if mode == 'norm':
            return self._normalize(x, month_key=month_key)
        elif mode == 'denorm':
            return self._denormalize(x, timestamp=timestamp, month=month, month_key=month_key)
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got '{mode}'")

    def _normalize(self, x, month_key=None):
        """
        Normalize: log1p -> z-score using monthly population stats -> affine.
        
        Uses monthly stats instead of per-instance stats so the model
        can see relative scale differences between meters.
        """
        # 1. Log1p transform
        x_log = torch.log1p(torch.clamp(x, min=0.0))  # [B, L, F]

        # 2. Get population-level stats (monthly or global)
        mean, std = self._get_normalization_stats(month_key)  # [1, 1, F]
        mean = mean.to(x.device)
        std = std.to(x.device)

        # 3. Cache for denormalization in same pass
        self._cached_mean = mean
        self._cached_std = std

        # 4. Z-score with population stats
        x_norm = (x_log - mean) / std

        # 5. Learnable affine
        if self.affine:
            x_norm = x_norm * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)

        return x_norm

    def _denormalize(self, x, timestamp=None, month=None, month_key=None):
        """
        Inverse of _normalize.
        
        If month_key is provided or cached stats exist, uses those.
        Otherwise falls back to timestamp/month lookup -> global.
        """
        # 1. Determine which mean/std to use
        if month_key is not None:
            mean, std = self._get_normalization_stats(month_key)
            mean = mean.to(x.device)
            std = std.to(x.device)
        elif self._cached_mean is not None and timestamp is None and month is None:
            # Same-pass denormalization
            mean = self._cached_mean.to(x.device)
            std = self._cached_std.to(x.device)
        elif timestamp is not None or month is not None:
            mean, std = self._get_stats_for_generation(timestamp, month)
            mean = mean.to(x.device)
            std = std.to(x.device)
        else:
            mean, std = self._get_global_stats()
            mean = mean.to(x.device)
            std = std.to(x.device)

        # 2. Inverse affine
        if self.affine:
            x = (x - self.bias.view(1, 1, -1)) / (self.weight.view(1, 1, -1) + self.eps)

        # 3. Inverse z-score
        x_log = x * std + mean

        # 4. Inverse log1p
        x_original = torch.expm1(x_log)

        # 5. Clamp to non-negative
        return torch.clamp(x_original, min=0.0)

    def _get_normalization_stats(self, month_key=None):
        """
        Get the stats to use for normalization.
        
        Priority:
            1. Specific month_key if provided
            2. Most recently added monthly stats
            3. Global stats
            4. Default (mean=0, std=1)
            
        Returns:
            mean, std: tensors of shape [1, 1, F]
        """
        # Try specific month
        if month_key is not None and month_key in self.monthly_stats:
            stats = self.monthly_stats[month_key]
            mean = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 1, -1)
            std = torch.tensor(np.maximum(stats['std'], self.min_std), dtype=torch.float32).view(1, 1, -1)
            return mean, std

        # Try most recent month
        if len(self.monthly_stats) > 0:
            latest_key = max(self.monthly_stats.keys())
            stats = self.monthly_stats[latest_key]
            mean = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 1, -1)
            std = torch.tensor(np.maximum(stats['std'], self.min_std), dtype=torch.float32).view(1, 1, -1)
            return mean, std

        # Global fallback
        return self._get_global_stats()

    def update_monthly_stats(self, data_loader, month_key=None):
        """
        Compute and store normalization stats for an entire month of data.
        Call this once per monthly data file, e.g., during DataModule setup.
        
        Args:
            data_loader: DataLoader yielding (x, meter_id, timestamps) batches
            month_key: Optional int identifier (e.g., 202310 for Oct 2023).
                       If None, inferred from first batch timestamps.
        """
        # Collect ALL log1p values to compute true population mean/std
        all_log_values = []
        total_samples = 0

        for x, meter_id, timestamps in data_loader:
            if x.dim() == 3:
                # x: [B, L, F] -> log1p transform
                x_log = torch.log1p(torch.clamp(x, min=0.0))
                # Flatten time dim: [B, L, F] -> [B*L, F]
                x_log_flat = x_log.reshape(-1, x_log.shape[-1])
                all_log_values.append(x_log_flat.detach().cpu().numpy())
                total_samples += x.shape[0]

                # Infer month_key from timestamps if not provided
                if month_key is None and timestamps is not None:
                    from datetime import datetime, timezone
                    ts_val = timestamps[0].item() if isinstance(timestamps, torch.Tensor) else timestamps[0]
                    dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    month_key = dt.year * 100 + dt.month  # e.g., 202310

        if total_samples == 0:
            return

        # Concatenate all log1p values: [N_total, F]
        all_log_values = np.concatenate(all_log_values, axis=0)

        # Compute TRUE population mean and std across all observations
        month_mean = all_log_values.mean(axis=0)  # [F]
        month_std = np.maximum(all_log_values.std(axis=0), self.min_std)  # [F]

        self.monthly_stats[month_key] = {
            'mean': month_mean,
            'std': month_std,
            'count': total_samples,
        }

        # Update global running stats
        self._global_mean_sum += month_mean * total_samples
        self._global_std_sum += month_std * total_samples
        self._global_count += total_samples

    def _get_stats_for_generation(self, timestamp, month=None):
        """Retrieve stats: month key → global fallback."""
        if month is not None:
            # Direct month lookup (1-12)
            for key, stats in self.monthly_stats.items():
                if key % 100 == month:
                    mean = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 1, -1)
                    std = torch.tensor(stats['std'], dtype=torch.float32).view(1, 1, -1)
                    return mean, std

        # Infer month_key from timestamp
        if timestamp is not None:
            from datetime import datetime, timezone
            if isinstance(timestamp, torch.Tensor):
                ts_val = timestamp.detach().cpu().item() if timestamp.numel() == 1 else timestamp[0].detach().cpu().item()
            else:
                ts_val = int(timestamp[0]) if hasattr(timestamp, '__len__') else int(timestamp)

            try:
                dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                month_key = dt.year * 100 + dt.month
                if month_key in self.monthly_stats:
                    stats = self.monthly_stats[month_key]
                    mean = torch.tensor(stats['mean'], dtype=torch.float32).view(1, 1, -1)
                    std = torch.tensor(stats['std'], dtype=torch.float32).view(1, 1, -1)
                    return mean, std
            except (ValueError, OSError, OverflowError):
                pass

        return self._get_global_stats()

    def _get_monthly_stats(self, month):
        """
        Compute average mean/std across all daily stats that fall in the given month.
        
        Args:
            month: int (1-12) representing the calendar month.
        
        Returns:
            mean, std: tensors of shape [1, 1, F], or falls back to global stats.
        """
        from datetime import datetime, timezone
        
        monthly_means = []
        monthly_stds = []
        
        for date_key, stats in self.daily_stats.items():
            try:
                # Convert date_key (days since epoch) back to datetime
                dt = datetime.fromtimestamp(date_key * 24 * 3600, tz=timezone.utc)
                if dt.month == month:
                    monthly_means.append(stats['mean'])
                    monthly_stds.append(stats['std'])
            except (ValueError, OSError, OverflowError):
                continue
        
        if len(monthly_means) > 0:
            avg_mean = np.mean(monthly_means, axis=0)  # [F]
            avg_std = np.mean(monthly_stds, axis=0)     # [F]
            mean = torch.tensor(avg_mean, dtype=torch.float32).view(1, 1, -1)
            std = torch.tensor(np.maximum(avg_std, self.min_std), dtype=torch.float32).view(1, 1, -1)
            return mean, std
        else:
            # No data for this month, fall back to global
            return self._get_global_stats()

    def _get_global_stats(self):
        """
        Compute global average mean/std from all observed data.
        
        Returns:
            mean, std: tensors of shape [1, 1, F].
            Falls back to (mean=0, std=1) if no data has been observed.
        """
        if self._global_count > 0:
            global_mean = self._global_mean_sum / self._global_count  # [F]
            global_std = self._global_std_sum / self._global_count    # [F]
            # Clamp std to minimum
            global_std = np.maximum(global_std, self.min_std)
            mean = torch.tensor(global_mean, dtype=torch.float32).view(1, 1, -1)
            std = torch.tensor(global_std, dtype=torch.float32).view(1, 1, -1)
        else:
            # No data observed yet — identity transform (no-op denormalization)
            mean = torch.zeros(1, 1, self.num_features, dtype=torch.float32)
            std = torch.ones(1, 1, self.num_features, dtype=torch.float32)
        return mean, std
        