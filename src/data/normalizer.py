import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import os

def compute_normalization_stats(sm_files,
                                feature_cols,
                                save_path="normalization_stats.json",
                                valid_ids_json_path=None,
                                use_cache=True,
                                hourly_aggregation: bool = False,
                                aggregation_divisor: float = 1.0):
    """Compute the mean and standard deviation for normalization across all Parquet files."""
    if use_cache and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
            mean = np.array(stats['log_mean'])
            std = np.array(stats['log_std'])
            return mean, std
        
    if sm_files is None:
        # Portable defaults for local project layouts
        cwd = os.getcwd()
        candidate_dirs = [
            os.path.join(cwd, 'data', 'raw_profiles'),
            os.path.join(cwd, 'data', 'cleaned_profiles'),
            os.path.join(cwd, 'data'),
        ]
        sm_files = []
        for data_dir in candidate_dirs:
            if os.path.isdir(data_dir):
                for root, _, files in os.walk(data_dir):
                    for f in files:
                        if f.endswith('.parquet'):
                            sm_files.append(os.path.join(root, f))
                if sm_files:
                    break
    
    if not sm_files:
        raise ValueError("No Parquet files found for normalization stats computation.")
    
    if not feature_cols:
        raise ValueError("No feature columns specified for normalization stats computation.")

    # === Optional filter by Timeseries_info JSON ===
    filtered_sm_files = list(sm_files)
    if valid_ids_json_path is not None and os.path.exists(valid_ids_json_path):
        print("Loading valid IDs...")
        with open(valid_ids_json_path, 'r') as f:
            valid_data = json.load(f)
            valid_ids = set(valid_data.get('id_list_of_sms_with_data', []))

        filtered_sm_files = []
        for file in sm_files:
            basename = os.path.basename(file)
            file_id = basename.replace('sm_', '').replace('.parquet', '')
            if file_id in valid_ids:
                filtered_sm_files.append(file)

        print(f"Filtered files: Kept {len(filtered_sm_files)} out of {len(sm_files)} original files.")
    else:
        print(f"No valid_ids filter applied. Using {len(filtered_sm_files)} files.")


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

        time_col_name = next((col for col in file_cols if col.lower() in ('timestamp_dst', 'timestamp', 'timestamp_utc')), None)
        if hourly_aggregation and time_col_name is not None and time_col_name not in cols_to_load:
            cols_to_load.append(time_col_name)
        
        # Track how many users are missing at least one column
        if len(cols_to_load) != len(feature_cols):
            users_missing_columns += 1
            
        # If the file has literally none of the features, skip it entirely
        if not cols_to_load:
            continue

        if hourly_aggregation:
            # Compute stats on the same hourly scale used by phased aggregated training.
            df = pq.read_table(file, columns=cols_to_load).to_pandas()
            df = df.rename(columns=rename_map)
            if time_col_name is not None and time_col_name in df.columns:
                time_values = df[time_col_name]
                df = df.drop(columns=[time_col_name])
            else:
                time_values = df.index

            time_index = pd.to_datetime(time_values, errors='coerce', utc=True)
            for feat in feature_cols:
                if feat not in df.columns:
                    df[feat] = np.nan
            phase_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
            phase_df.index = time_index
            phase_df = phase_df[~phase_df.index.isna()]
            phase_df = phase_df[~phase_df.index.duplicated(keep='last')]
            phase_df = phase_df.sort_index()
            phase_df = phase_df.resample('1h').sum(min_count=1)
            if float(aggregation_divisor) > 0 and float(aggregation_divisor) != 1.0:
                phase_df = phase_df / float(aggregation_divisor)

            log_df = np.log1p(phase_df[found_features])
            for i, feat in enumerate(feature_cols):
                if feat in found_features:
                    valid_data = log_df[feat].dropna()
                    total_sum[i] += valid_data.sum()
                    total_sq_sum[i] += (valid_data ** 2).sum()
                    total_count[i] += len(valid_data)
        else:
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

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
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

    def fit(self, 
            X: np.ndarray | torch.Tensor, 
            sm_files=None, 
            stats_path='normalization_stats.json',
            hourly_aggregation: bool = False,
            aggregation_divisor: float = 1.0) -> 'Normalizer':
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
        
        if X is None:
            # For log1p_phase, we can compute the stats from the entire dataset using the compute_normalization_stats function
            if self.method == 'log1p_phase':
                self.mean, self.std = compute_normalization_stats(
                    sm_files=sm_files,
                    feature_cols=['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3'],
                    save_path=stats_path,
                    valid_ids_json_path=None,
                    use_cache=True,
                    hourly_aggregation=hourly_aggregation,
                    aggregation_divisor=aggregation_divisor,
                )
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
        else:
            raise ValueError(f"finalise_fit() not implemented for method '{self.method}'")

        self.fitted = True

        # Release temporary Welford state
        for attr in ('_wf_n', '_wf_mean', '_wf_M2'):
            self.__dict__.pop(attr, None)
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
