import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import os

def compute_normalization_stats(sm_files, 
                                feature_cols, 
                                save_path = "normalization_stats.json", 
                                valid_ids_json_path = "data\Timeseries_info\Timeseries_info.json"):
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

    # === Filter by Timeseries_info JSON if provided ===
    print("Loading valid IDs...")
    with open(valid_ids_json_path, 'r') as f:
        valid_data = json.load(f)
        # Convert to a set for O(1) lightning-fast lookups
        valid_ids = set(valid_data['id_list_of_sms_with_data']) 
        
    filtered_sm_files = []
    for file in sm_files:
        # Extract the ID from the filename (e.g., "sm_100025.parquet" -> "100025")
        basename = os.path.basename(file)
        file_id = basename.replace('sm_', '').replace('.parquet', '')
        
        if file_id in valid_ids:
            filtered_sm_files.append(file)
            
    print(f"Filtered files: Kept {len(filtered_sm_files)} out of {len(sm_files)} original files.")


    total_sum = np.zeros(len(feature_cols))
    total_sq_sum = np.zeros(len(feature_cols))
    total_count = np.zeros(len(feature_cols))

    users_missing_columns = 0

    for file in tqdm(filtered_sm_files, desc="Computing stats"):
        dataset = pq.ParquetFile(file)
        file_cols = dataset.schema.names
        
        cols_to_load = []
        rename_map = {}
        found_features = []
        
        for base_feat in feature_cols:
            for actual_col in file_cols:
                if actual_col.startswith(base_feat):
                    cols_to_load.append(actual_col)
                    rename_map[actual_col] = base_feat
                    found_features.append(base_feat)
                    break
        
        # Track how many users are missing at least one column
        if len(cols_to_load) != len(feature_cols):
            users_missing_columns += 1
            
        # If the file has literally none of the features, skip it entirely
        if not cols_to_load:
            continue

        for batch in dataset.iter_batches(columns=cols_to_load, batch_size=10000):
            df = batch.to_pandas()
            df = df.rename(columns=rename_map)

            # Apply log only to the features we actually found
            log_df = np.log1p(df[found_features])  
            
            # Add to the sums independently based on what exists
            for i, feat in enumerate(feature_cols):
                if feat in found_features:
                    # Drop NaNs just in case there are internal empty rows
                    valid_data = log_df[feat].dropna()
                    total_sum[i] += valid_data.sum()
                    total_sq_sum[i] += (valid_data ** 2).sum()
                    total_count[i] += len(valid_data)


    print(f"\n--- DATA QUALITY REPORT ---")
    print(f"Total files processed: {len(filtered_sm_files)}")
    print(f"Users missing one or more columns: {users_missing_columns}")
    for i, feat in enumerate(feature_cols):
        print(f"Valid rows found for {feat}: {int(total_count[i])}")
    print(f"---------------------------\n")


    mean = np.zeros(len(feature_cols))
    std = np.ones(len(feature_cols)) 
    
    for i in range(len(feature_cols)):
        if total_count[i] > 0:
            mean[i] = total_sum[i] / total_count[i]
            # max(0, ...) prevents math domain errors from floating point inaccuracies
            var = max(0, (total_sq_sum[i] / total_count[i]) - (mean[i] ** 2))
            std[i] = np.sqrt(var)
            
            # If standard deviation is exactly 0 (constant value), force to 1 to prevent division by zero during training
            if std[i] == 0:
                std[i] = 1.0

    with open(save_path, 'w') as f:
        json.dump({'log_mean': mean.tolist(), 'log_std': std.tolist()}, f)

    return mean, std

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

    def _to_tensor(self, X) -> torch.Tensor:
        """Convert X to float32 on self.device."""
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
        if X is None:
            # For log1p_phase, we can compute the stats from the entire dataset using the compute_normalization_stats function
            if self.method == 'log1p_phase':
                self.mean, self.std = compute_normalization_stats(sm_files=None, feature_cols=['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3'], save_path='normalization_stats.json')
                self.fitted = True
                return
            else:
                raise ValueError("X cannot be None for methods other than 'log1p_phase'.")
        Xt = self._to_tensor(X) # (N,F)

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
        """Normalise X.  Always returns a float32 torch.Tensor on self.device.

        Parameters
        ----------
        X : array-like or Tensor (N, F)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. call fit() or finalise_fit() first.")
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
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

    def inverse_transform(self, X) -> torch.Tensor:
        """Denormalise X back to the original scale.

        Parameters
        ----------
        X : array-like or Tensor (N, F)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. call fit() or finalise_fit() first.")
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
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

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
        