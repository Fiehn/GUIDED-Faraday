import os
import json
from typing import Optional
from datetime import datetime, timezone
import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm
import hashlib

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
    LOG_METHODS = {'log1p', 'log1p_phase'}

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

        self.manifest = None  # Dict that stores metadata so the file can be reused
    
    # === Helpers ===
    def to(self, device: str | torch.device) -> 'Normalizer':
        """Move tensor stats to device."""
        self.device = torch.device(device)

        def _move(t):
            if isinstance(t, torch.Tensor):
                return t.to(self.device)
            if isinstance(t, np.ndarray):
                return torch.from_numpy(t.astype(np.float32)).to(self.device)
            if isinstance(t, (list, tuple)):
                return torch.tensor(t, dtype=torch.float32, device=self.device)
            return t

        self.mean = _move(self.mean)
        self.std  = _move(self.std)
        return self

    def _to_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert X to float32 on self.device."""
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)
        return X.float().to(self.device)

    def _feature_dim_hint(self) -> int | None:
        for stat in (self.mean, self.std):
            if stat is None:
                continue
            if isinstance(stat, torch.Tensor):
                return int(stat.reshape(-1).numel())
            return int(np.asarray(stat).reshape(-1).shape[0])
        return 3 if self.method == 'log1p_phase' else None

    def _flatten_feature_matrix(self, X, *, context: str) -> tuple[torch.Tensor, callable]:
        """Normalize 1D/2D/3D input to [N, F] and return a restore closure."""
        Xt = self._to_tensor(X)
        if Xt.dim() not in (1, 2, 3):
            raise ValueError(f"{context}: expected 1D, 2D, or 3D input. Got shape {tuple(Xt.shape)}")

        if Xt.dim() == 1:
            return Xt.reshape(-1, 1), lambda y: y.reshape_as(Xt)

        feat_hint = self._feature_dim_hint()

        if Xt.dim() == 2:
            # Default: feature-last [N, F]. If stats suggest feature-first [F, N], transpose once.
            if feat_hint is not None and Xt.shape[0] == feat_hint and Xt.shape[1] != feat_hint:
                x_tf = Xt.transpose(0, 1).contiguous()
                return x_tf.reshape(-1, x_tf.shape[-1]), lambda y: y.reshape_as(x_tf).transpose(0, 1).contiguous()
            return Xt.reshape(-1, Xt.shape[-1]), lambda y: y.reshape_as(Xt)

        # 3D: accept both [B, L, F] and [B, F, L]
        if feat_hint is not None:
            if Xt.shape[-1] == feat_hint:
                x_lf = Xt
                return x_lf.reshape(-1, x_lf.shape[-1]), lambda y: y.reshape_as(x_lf)
            if Xt.shape[1] == feat_hint:
                x_lf = Xt.transpose(1, 2).contiguous()
                return x_lf.reshape(-1, x_lf.shape[-1]), lambda y: y.reshape_as(x_lf).transpose(1, 2).contiguous()
            raise ValueError(
                f"{context}: could not infer feature axis for shape {tuple(Xt.shape)} with expected feature dim {feat_hint}"
            )

        # No stats yet: treat smaller of axes 1/2 as feature axis.
        if Xt.shape[-1] <= Xt.shape[1]:
            x_lf = Xt
            return x_lf.reshape(-1, x_lf.shape[-1]), lambda y: y.reshape_as(x_lf)
        x_lf = Xt.transpose(1, 2).contiguous()
        return x_lf.reshape(-1, x_lf.shape[-1]), lambda y: y.reshape_as(x_lf).transpose(1, 2).contiguous()

    def _validate_feature_dim(self, X2d: torch.Tensor, *, context: str) -> None:
        expected_f = self._feature_dim_hint()
        if expected_f is not None and int(X2d.shape[1]) != int(expected_f):
            raise ValueError(
                f"{context}: input feature dim {int(X2d.shape[1])} does not match normalizer stats dim {int(expected_f)}"
            )

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
                mean_np, std_np = fit_normalizer_on_files(
                    filenames= sm_files,
                    feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3'],
                    save_path= stats_path,
                    hourly_aggregation=hourly_aggregation,
                    aggregation_divisor=aggregation_divisor,
                )
                # Store as tensors so downstream device moves and arithmetic are well-defined.
                self.mean = torch.as_tensor(mean_np, dtype=torch.float32, device=self.device).reshape(1, -1)
                self.std = torch.as_tensor(std_np, dtype=torch.float32, device=self.device).reshape(1, -1)
                self.fitted = True
                return self
            else:
                raise ValueError("X cannot be None for methods other than 'log1p_phase'.")
        Xt, _ = self._flatten_feature_matrix(X, context='fit')

        if self.method == 'zero_preserved_log':
            nz = Xt[Xt != 0]
            log_nz = torch.log(nz)
            self.non_zero_mean = log_nz.mean().item()
            self.non_zero_std  = log_nz.std().item()
        elif self.method == 'standard':
            self.mean = Xt.mean(dim=0, keepdim=True)
            self.std  = Xt.std(dim=0,  keepdim=True)
            self.std[self.std == 0] = 1.0
        elif self.method in self.LOG_METHODS:
            log1p_values = torch.log1p(torch.clamp(Xt, min=0.0))
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
        Xt, _ = self._flatten_feature_matrix(X_chunk, context='fit_incremental')
        if Xt.numel() == 0:
            return
        F  = Xt.shape[1]

        # Initialise accumulators on first call
        if not hasattr(self, '_wf_n'):
            self._wf_n    = 0
            self._wf_mean = torch.zeros(1, F, device=self.device)
            self._wf_M2   = torch.zeros(1, F, device=self.device)

        if self.method in self.LOG_METHODS:
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

        if self.method == 'standard' or self.method in self.LOG_METHODS:
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
            'manifest': self.manifest,
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
        norm.manifest      = state['manifest']
        norm.to(device)
        return norm

    @classmethod
    def _manifest_value(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._manifest_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [cls._manifest_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _stable_json_dumps(cls, payload: dict) -> str:
        canonical = cls._manifest_value(payload)
        return json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

    @classmethod
    def _build_file_fingerprint(cls, filenames: list[str]) -> str:
        file_meta = []
        for original_path in filenames:
            raw_path = str(original_path)
            norm_path = os.path.normcase(os.path.normpath(raw_path))
            abs_path = os.path.abspath(raw_path)
            try:
                stats = os.stat(abs_path)
                size = int(stats.st_size)
                mtime_ns = int(stats.st_mtime_ns)
            except OSError:
                size = None
                mtime_ns = None
            file_meta.append({
                'path': norm_path,
                'size': size,
                'mtime_ns': mtime_ns,
            })
        file_meta.sort(key=lambda item: item['path'])
        digest_input = {'files': file_meta}
        return hashlib.sha256(cls._stable_json_dumps(digest_input).encode('utf-8')).hexdigest()

    @staticmethod
    def _resolve_cache_target(save_path: str | None, manifest_hash: str) -> tuple[str, str | None]:
        if save_path is None:
            cache_dir = os.path.join('data', 'normalizations')
            return (
                os.path.join(cache_dir, f'normalizer_{manifest_hash}.pt'),
                os.path.join(cache_dir, 'index.json'),
            )

        save_path_abs = os.path.abspath(save_path)
        has_extension = os.path.splitext(save_path_abs)[1] != ''
        if os.path.isdir(save_path_abs) or not has_extension:
            return (
                os.path.join(save_path_abs, f'normalizer_{manifest_hash}.pt'),
                os.path.join(save_path_abs, 'index.json'),
            )

        # Explicit file path -> legacy single-artifact behavior.
        return save_path_abs, None

    @staticmethod
    def _read_json_file(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _update_normalization_index(
        cls,
        *,
        index_path: str,
        manifest_hash: str,
        artifact_path: str,
        manifest: dict,
    ) -> None:
        index = cls._read_json_file(index_path)
        entries = index.get('entries')
        if not isinstance(entries, dict):
            entries = {}

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        existing = entries.get(manifest_hash, {})
        entries[manifest_hash] = {
            'path': os.path.basename(artifact_path),
            'method': manifest.get('method'),
            'num_files': manifest.get('num_files'),
            'created_at': existing.get('created_at', now),
            'last_used_at': now,
            'manifest': manifest,
        }
        index = {
            'version': 1,
            'entries': entries,
        }

        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        tmp_path = f"{index_path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, index_path)

    def _apply_cached_state(self, cached: 'Normalizer') -> None:
        self.mean = cached.mean
        self.std = cached.std
        self.non_zero_mean = cached.non_zero_mean
        self.non_zero_std = cached.non_zero_std
        self.epsilon = cached.epsilon
        self.zero_id = cached.zero_id
        self.shift = cached.shift
        self.fitted = bool(cached.fitted)

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
        x2d, restore = self._flatten_feature_matrix(X, context='transform')
        self._validate_feature_dim(x2d, context='transform')

        if self.method == 'zero_preserved_log':
            y2d = self._zero_preserved_log_transform(x2d)
        elif self.method == 'standard':
            y2d = (x2d - self.mean) / (self.std + self.epsilon)
        elif self.method in self.LOG_METHODS:
            x2d = torch.clamp(x2d, min=0.0)
            y2d = (torch.log1p(x2d) - self.mean) / (self.std + self.epsilon)
        else:
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

        return restore(y2d)

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
        x2d, restore = self._flatten_feature_matrix(X, context='inverse_transform')
        self._validate_feature_dim(x2d, context='inverse_transform')

        if self.method == 'zero_preserved_log':
            y2d = self._zero_preserved_log_inverse(x2d)
        elif self.method == 'standard':
            y2d = torch.clamp(x2d * self.std + self.mean, min=0.0)
        elif self.method in self.LOG_METHODS:
            y2d = torch.clamp(torch.expm1(x2d * (self.std + self.epsilon) + self.mean), min=0.0)
        else:
            raise ValueError(f"Unknown normalisation method: '{self.method}'")

        return restore(y2d)
    
    def fit_normalizer_on_files(
            self,
            filenames: list[str],
            window_spec,
            phased: bool = False,
            aggregation_divisor: float = 1.0,
            hourly_aggregation: bool = False,
            meter_col: str = 'meter_number',
            nan_as_zero: bool = False,
            force_recompute: bool = False,
            feature_cols: list[str] | None = None,
            save_path: str | None = None,
            device: str | torch.device = 'cpu',
    ):
        # === Try Cache ===
        window_spec_dict = {
            'input_len': window_spec.input_len,
            'stride': window_spec.stride,
        }
        file_fingerprint = self._build_file_fingerprint(filenames)
        normalizer_manifest = {
            'manifest_version': 2,
            'method': self.method,
            'phased': phased,
            'aggregation_divisor': aggregation_divisor,
            'meter_col': meter_col,
            'nan_as_zero': nan_as_zero,
            'feature_cols': feature_cols,
            'hourly_aggregation': hourly_aggregation,
            'window_spec': window_spec_dict,
            'num_files': len(filenames),
            'file_fingerprint': file_fingerprint,
        }
        manifest_hash = hashlib.sha256(self._stable_json_dumps(normalizer_manifest).encode('utf-8')).hexdigest()
        cache_path, index_path = self._resolve_cache_target(
            save_path=save_path,
            manifest_hash=manifest_hash,
        )

        if not force_recompute and os.path.exists(cache_path):
            try:
                saved = self.load(cache_path, device=device)
                saved_manifest = getattr(saved, 'manifest', None)
                if normalizer_manifest == saved_manifest:
                    print(f"Loading normalization stats from cache at {cache_path}...")
                    self._apply_cached_state(saved)
                    self.manifest = saved_manifest
                    self.to(self.device)
                    if index_path is not None:
                        self._update_normalization_index(
                            index_path=index_path,
                            manifest_hash=manifest_hash,
                            artifact_path=cache_path,
                            manifest=normalizer_manifest,
                        )
                    return
                else:
                    print(f"Cache manifest does not match current configuration. Recomputing stats.")
            except Exception as e:
                print(f"Failed to load normalization stats from cache: {e}. Recomputing stats.")
        
        # === Compute Stats ===
        from data.dataloader_phased import StreamingParquetBuffer
        from torch.utils.data import DataLoader
        phased_ds = StreamingParquetBuffer(
            sm_files=filenames, 
            replay_buffer_capacity=10000, 
            min_buffer_size=1000,
            window_spec=window_spec,
            num_workers=0,
            feature_cols=feature_cols,
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
            phased_mode=phased,
            meter_col=meter_col,
            nan_as_zero=nan_as_zero,
            batch_size=256,
        )
        train_loader = DataLoader(phased_ds, batch_size=None, num_workers=0, pin_memory=True)
        
        for X_batch in tqdm(train_loader, desc="Fitting normalizer"):
            x_gpu = X_batch[0].to(device)
            self.fit_incremental(x_gpu)
        self.finalise_fit()
        print(f"Fitted normalizer with method '{self.method}' on {len(filenames)} files.")

        # === Save Stats ===
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self.manifest = normalizer_manifest
        self.save(cache_path)
        if index_path is not None:
            self._update_normalization_index(
                index_path=index_path,
                manifest_hash=manifest_hash,
                artifact_path=cache_path,
                manifest=normalizer_manifest,
            )

        print(f"Saved normalization stats to {cache_path}.")


            
