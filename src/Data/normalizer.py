import numpy as np
import torch
import torch.nn as nn
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import os

def compute_normalization_stats(sm_files, feature_cols, save_path = "normalization_stats.json"):
    """Compute the mean and standard deviation for normalization across all Parquet files."""
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
            mean = np.array(stats['log_mean'])
            std = np.array(stats['log_std'])
            return mean, std
        
    if sm_files is None:
        # Find all Parquet files in the default data directory
        data_dir = 'F:\Data\processed_data\second_batch\individual_sm_datasets\cleaned_profiles'
        sm_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not sm_files:
        raise ValueError("No Parquet files found for normalization stats computation.")
    
    if not feature_cols:
        raise ValueError("No feature columns specified for normalization stats computation.")
    
    # Feature columns in the Parquet files have a suffix like _meterid, so we need to remove that to get the base feature name
    df = df.rename(columns={
            col: '_'.join(col.split('_')[:-1])  # keep everything except the last part
            for col in df.columns
        })

    total_sum = np.zeros(len(feature_cols))
    total_sq_sum = np.zeros(len(feature_cols))
    total_count = 0

    for file in tqdm(sm_files, desc="Computing normalization stats"):
        dataset = pq.ParquetDataset(file)
        for batch in dataset.iter_batches(columns=feature_cols, batch_size=10000):
            df =  batch.to_pandas()
            log_df = np.log1p(df[feature_cols])  # Apply log transformation to the features
            total_sum += log_df[feature_cols].sum().values
            total_sq_sum += (log_df[feature_cols] ** 2).sum().values
            total_count += len(log_df)

    mean = total_sum / total_count
    std = np.sqrt((total_sq_sum / total_count) - (mean ** 2))

    # Save the computed mean and std to a JSON file for later use
    with open(save_path, 'w') as f:
        json.dump({'log_mean': mean.tolist(), 'log_std': std.tolist()}, f)

    return mean, std

# === Normalizer class that will use the entire data ===
class Normalizer:
    """
    Normalizes data using specified methods.
    Supports both numpy arrays and torch tensors.
    
    Methods:
      - zero_preserved_log: Log-transform non-zero values, mark zeros with special ID
      - standard: Standard z-score normalization
      - log1p: log1p(x) - mean / std
      - online_log1p: stores and recalculates normalization stats per month
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

        # for online log1p normalization
        self.monthly_stats = {}
        self._global_mean_sum = 0
        self._global_std_sum = 0
        self._global_count = 0
    
    # === Helpers ===
    def to(self, device: str | torch.device) -> 'Normalizer':
        """Move everything to device"""
        self.device = torch.device(device)

        def _move(t):
            return t.to(self.device) if isinstance(t, torch.Tensor) else t
        
        self.non_zero_mean = _move(self.non_zero_mean)
        self.non_zero_std = _move(self.non_zero_std)
        self.mean = _move(self.mean)
        self.std = _move(self.std)
        self._global_mean = _move(self._global_mean)
        self._global_std = _move(self._global_std)

        for m, stats in self.monthly_stats.items():
            stats['mean'] = _move(stats['mean'])
            stats['std']  = _move(stats['std'])

        return self

    def _to_tensor(self, X) -> torch.Tensor:
        """ Convert X to float32 on self.device"""
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)
        return X.float().to(self.device)
    
    def _mk_to_tensor(self, month_key, n_rows: int) -> torch.Tensor | None:
        """Convert month_key to a LongTensor of shape (n_rows,) on self.device.
        Returns None when keys are none."""
        if month_key is None:
            return None
        if isinstance(month_key, torch.Tensor):
            mk = month_key.long().to(self.device)
        else:
            mk = torch.from_numpy(np.asarray(month_key, dtype=np.int64)).to(self.device)
        if mk.ndim == 0:
            mk = mk.expand(n_rows)
        if mk.shape[0] != n_rows:
            raise ValueError(f"month_key length {mk.shape[0]} != X rows {n_rows}")
        return mk 

    def fit(self, X, month_key = None):
        """Fit the normalizer.
        Parameters:
        X : array-like (N,F)
        month_key : 1-D int array (N,) of YYYYMM values
                    Only required for method=='online_log1p'"""
        Xt = self._to_tensor(X) # (N,F)

        if self.method == 'zero_preserved_log':
            non_zero_values = Xt[Xt != 0]
            log_values = torch.log(non_zero_values)
            self.non_zero_mean = log_values.mean().item()
            self.non_zero_std = log_values.std().item()
            self.fitted = True
        elif self.method == 'standard':
            self.mean = torch.mean(Xt, dim=0, keepdim=True)
            self.std = torch.std(Xt, dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0
            self.fitted = True
        elif self.method == 'log1p':
            log1p_values = torch.log1p(Xt)
            self.mean = log1p_values.mean(dim=0, keepdim=True)
            self.std = log1p_values.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0 # Avoid division by zero
            self.fitted = True
        elif self.method == 'online_log1p':
            X_log = torch.log1p(Xt.clamp(min=0.0))
            # Store values on monthly basis
            if month_key is not None:
                mk = self._mk_to_tensor(month_key, Xt.shape[0])
                for m in mk.unique():
                    m_int = int(m.item())
                    mask  = mk == m
                    vals  = X_log[mask]                     # (K, F)
                    mean_m = vals.mean(dim=0, keepdim=True) # (1, F)
                    std_m  = vals.std(dim=0,  keepdim=True) # (1, F)
                    std_m[std_m == 0] = 1.0
                    self.monthly_stats[m_int] = {
                        'mean':  mean_m,
                        'std':   std_m,
                        'count': int(mask.sum().item()),
                    }

            # global fallback 
            self._global_mean = X_log.mean(dim=0, keepdim=True)  # (1, F)
            self._global_std  = X_log.std(dim=0,  keepdim=True)  # (1, F)
            self._global_std[self._global_std == 0] = 1.0
            self.fitted = True
        elif self.method == 'log1p_phase':
            self.mean, self.std = compute_normalization_stats(sm_files=None, feature_cols=['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3'], save_path='normalization_stats.json')
        else:
            raise ValueError(f"Normalization method {self.method} not recognized.")
    
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
    
    def _get_month_stats(self, m_int: int):
        """Return (mean (1,F), std (1,F)) tensors for a single YYYYMM key."""
        stats = self.monthly_stats.get(m_int)
        if stats is not None:
            return stats['mean'], stats['std']
        return self._global_mean, self._global_std
    
    def _apply_online_log1p(self, X: torch.Tensor, mk: torch.Tensor | None, inverse: bool) -> torch.Tensor:
        """
        Per-row normalise (inverse=False) or denormalise (inverse=True).
        X is already log1p-transformed when inverse=False.
        """
        out = torch.empty_like(X)

        if mk is None:
            mean, std = self._global_mean, self._global_std
            out = torch.expm1(X * (std + self.epsilon) + mean) if inverse \
                  else (torch.log1p(X) - mean) / (std + self.epsilon)
            return out

        for m in mk.unique():
            m_int   = int(m.item())
            mask    = mk == m
            mean, std = self._get_month_stats(m_int)
            if inverse:
                out[mask] = torch.expm1(X[mask] * (std + self.epsilon) + mean)
            else:
                out[mask] = (torch.log1p(X[mask]) - mean) / (std + self.epsilon)
        return out
    
    def transform(self, X, month_key = None) -> torch.Tensor:
        """
        Normalise X.  Always returns a torch.Tensor on self.device.

        Parameters
        ----------
        X          : array-like or Tensor (N, F)
        month_key  : 1-D int array/Tensor (N,) of YYYYMM.
                     Only used for method=='online_log1p'.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        Xt = self._to_tensor(X)
        
        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_transform(Xt)

        elif self.method == 'standard':
            return (Xt - self.mean) / (self.std + self.epsilon)

        elif self.method == 'log1p':
            return (torch.log1p(Xt) - self.mean) / (self.std + self.epsilon)

        elif self.method == 'online_log1p':
            mk    = self._mk_to_tensor(month_key, Xt.shape[0])
            return self._apply_online_log1p(Xt, mk, inverse=False)
        
        elif self.method == 'log1p_phase':
            # For phase-aware log1p, we can just apply the standard log1p normalization using pre-computed mean/std from the entire dataset
            if self.mean is None or self.std is None:
                raise RuntimeError("log1p_phase normalization requires pre-computed mean and std. Call fit() with method='log1p_phase' first.")
            return (torch.log1p(Xt) - self.mean) / (self.std + self.epsilon)
        
        else:
            raise ValueError(f"Normalization method '{self.method}' not recognized.")

    def inverse_transform(self, X, month_key=None) -> torch.Tensor:
        """
        Denormalise X back to original scale. Always returns a torch.Tensor on self.device.

        Parameters
        ----------
        X          : array-like or Tensor (N, F)
        month_key  : 1-D int array/Tensor (N,) of YYYYMM.
                     Only used for method=='online_log1p'.
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted – call fit() first.")

        Xt = self._to_tensor(X)

        if self.method == 'zero_preserved_log':
            return self._zero_preserved_log_inverse(Xt)

        elif self.method == 'standard':
            return Xt * self.std + self.mean

        elif self.method == 'log1p':
            return torch.clamp(torch.expm1(Xt * (self.std + self.epsilon) + self.mean), min=0.0)

        elif self.method == 'online_log1p':
            mk  = self._mk_to_tensor(month_key, Xt.shape[0])
            out = self._apply_online_log1p(Xt, mk, inverse=True)
            return out.clamp(min=0.0)

        elif self.method == 'log1p_phase':
            if self.mean is None or self.std is None:
                raise RuntimeError("log1p_phase normalization requires pre-computed mean and std. Call fit() with method='log1p_phase' first.")
            return torch.clamp(torch.expm1(Xt * (self.std + self.epsilon) + self.mean), min=0.0)
        
        else:
            raise ValueError(f"Unknown normalization method: '{self.method}'")
    
    def _get_monthly_stats(self, month_key: int):
        """Public alias kept for backward compatibility."""
        mean, std = self._get_month_stats(int(month_key))
        return {'mean': mean, 'std': std}

# === RevIN (reversible instance normalization) for online normalization ===
# Refrence: https://openreview.net/forum?id=cGDAkQo1C0p
# With refrence dict to dates seen
class IndexedLogRevIN(nn.Module):
    """
    DEPRECATED: moved back to Normalizer class since Affine parameters have a negative influence on the model performance.

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
        