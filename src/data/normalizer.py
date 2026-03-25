import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import os

def compute_normalization_stats(sm_files: Optional[list[str]] = None,
                                feature_cols: Optional[list[str]] = None,
                                save_path: str = "normalization_stats.json",
                                valid_ids_json_path: Optional[str] = None,
                                use_cache: bool =True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation for normalization across all Parquet files.
    
    Args: 
        sm_files: Optional list of Parquet file paths to process. If None, will search common data directories.
        feature_cols: List of base feature column names to compute stats for (e.g. ['active_power_p14_l1', ...]).
                      The function will look for columns that start with these base names in each file.
        save_path: Path to save the computed stats as JSON. The file will contain 'log_mean' and 'log_std' lists.
        valid_ids_json_path: Optional path to a JSON file containing a list of valid meter IDs to include in the stats computation.
                            If provided, only files corresponding to these IDs will be processed.
        use_cache: If True, will attempt to load stats from save_path if it exists before computing.
    
    Returns:
        mean: Numpy array of means for each feature column
        std: Numpy array of standard deviations for each feature column
    """
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

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'log_mean': mean.tolist(), 'log_std': std.tolist()}, f)

    return mean, std

# === Normalizer class that will use the entire data ===
class Normalizer:
    """
    Normalizes data using specified methods.
    Supports both numpy arrays and torch tensors.

    Attributes:
        zero_preserved_log: Log-transform non-zero values, mark zeros with special sentinel
        standard: Z-score normalisation
        log1p: (log1p(x) - mean) / std 

    The normalizer can be fitted once across multiple files using
    fit_incremental() / finalise_fit(), and persisted to disk with
    save() / load() so stats do not need to be recalculated each run.
    """
    def __init__(self, 
                 method: str = 'log1p'):
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
    def to(self, 
           device: str | torch.device) -> 'Normalizer':
        """
        Move tensor stats to device.
        
        Args: 
            device: Target device (e.g. 'cpu' or 'cuda')
        
        Returns:
            self: with stats moved to target device
        """
        self.device = torch.device(device)

        def _move(t):
            return t.to(self.device) if isinstance(t, torch.Tensor) else t

        self.mean = _move(self.mean)
        self.std  = _move(self.std)
        return self

    def _to_tensor(self, 
                   X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Convert X to float32 on self.device.
        
        Args: 
            X: Input data as numpy array or torch tensor
        
        Returns:
            Tensor version of X on self.device
        """
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)
        return X.float().to(self.device)
    
    def _mk_to_tensor(self, 
                      month_key: Optional[np.ndarray | torch.Tensor], 
                      n_rows: int) -> torch.Tensor | None:
        """
        Convert month_key to a LongTensor of shape (n_rows,) on self.device.
        
        Args: 
            month_key: Optional array of month keys (e.g. YYYYMM integers) corresponding to each row in X. Required for 'online_log1p' method to apply monthly stats. Can be None for other methods.

        Returns: 
            None when keys are none.
        """
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

    def fit(self, 
            X: np.ndarray | torch.Tensor, 
            month_key: Optional[np.ndarray | torch.Tensor] = None, 
            sm_files: Optional[list[str]] = None, 
            stats_path: str ='normalization_stats.json'):
        """
        Fit the normalizer.
        
        Args: 
            X: array-like (N,F)
            month_key: 1-D int array (N,) of YYYYMM values. Only required for method=='online_log1p'
            sm_files: Optional list of Parquet file paths to compute stats from when method=='log1p_phase'. If None, will search common data directories.
            stats_path: Path to save computed stats when method=='log1p_phase'
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

    def save(self, 
             path: str):
        """
        Save fitted stats to disk.
        The file is a plain torch.save dict and can be re-loaded with
        Normalizer.load(path).  It is safe to save before moving tensors
        to GPU; the file always stores stats on CPU.

        Args:
            path: File path to save the normalizer state (e.g. 'normalizer.pt')
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
    def load(cls: type, 
             path: str, 
             device: str | torch.device = 'cpu') -> 'Normalizer':
        """
        Load a previously saved Normalizer from disk.
        
        Args: 
            cls: The Normalizer class (use Normalizer.load(...))
            path: File path to load the normalizer state from (e.g. 'normalizer.pt')
            device: Target device to move the loaded stats to (default: 'cpu')

        Returns:
            An instance of Normalizer with stats loaded from the specified file and moved to the target device
        """
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

    def _zero_preserved_log_transform(self,
                                      X: torch.Tensor) -> torch.Tensor:
        """
        Apply zero-preserved log transformation to X.
        Non-zero values are transformed as: (log(x) - non_zero_mean) / non_zero_std + shift
        Zero values are set to zero_id.
        
        Args:
            X: Input tensor (N, F)

        Returns:
            Transformed tensor of the same shape as X
        """
        is_zero = X == 0
        Y = X.clone()
        if is_zero.any():
            Y[~is_zero] = (torch.log(Y[~is_zero]) - self.non_zero_mean) / (self.non_zero_std + self.epsilon) + self.shift
            Y[is_zero]  = self.zero_id
        else:
            Y = (torch.log(Y) - self.non_zero_mean) / (self.non_zero_std + self.epsilon) + self.shift
        return Y

    def _zero_preserved_log_inverse(self, 
                                    X: torch.Tensor) -> torch.Tensor:
        """
        Inverse of zero-preserved log transformation.
        Values equal to zero_id are set to 0. Non-zero values are transformed back as exp((x - shift) * non_zero_std + non_zero_mean)
        
        Args: 
            X: Input tensor (N, F)
        
        Returns:
            Inverse transformed tensor of the same shape as X
        """
        Y = X.clone()
        is_zero = torch.isclose(Y, torch.tensor(float(self.zero_id), dtype=Y.dtype, device=Y.device), atol=0.1)
        Y[~is_zero] = torch.exp((Y[~is_zero] - self.shift) * (self.non_zero_std + self.epsilon) + self.non_zero_mean)
        Y[is_zero]  = 0.0
        return Y
    
    def transform(self, 
                  X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Normalise X.  Always returns a float32 torch.Tensor on self.device.

        Args: 
            X: array-like or Tensor (N, F)
        
        Returns:
            Normalised tensor of shape (N, F) on self.device
        
        Raises:
            RuntimeError: If the normalizer has not been fitted yet
            ValueError: If an unknown normalization method is specified
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

    def inverse_transform(self, 
                          X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Denormalise X back to the original scale.

        Args: 
            X: array-like or Tensor (N, F)
        
        Returns:
            Denormalised tensor of shape (N, F) on self.device
        
        Raises:
            RuntimeError: If the normalizer has not been fitted yet
            ValueError: If an unknown normalization method is specified
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
