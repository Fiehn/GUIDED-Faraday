from __future__ import annotations

import math
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

from .base import BaseConditioner


class UserProfiler(LatentDirichletAllocation, BaseConditioner):
    """
    User profiler using LDA + KMeans for time series data.

    Each user's time series is clustered into "words", then LDA produces
    topic distributions as user profiles. Supports both batch fitting and
    incremental online updates with exponential decay for inactive users.

    Args:
        n_topics (int): Number of LDA topics. Defaults to 10.
        num_clusters (int): Number of KMeans clusters (vocabulary size). Defaults to 100.
        random_state (int, optional): Random seed. Defaults to None.
        path (str): Path to save/load the model. Defaults to 'models/UserProfiles/user_profiles.pkl'.
        decay (float): Exponential forgetting factor for inactive users (0 < decay <= 1). Defaults to 0.9.
        **kwargs: Additional keyword arguments. Accepts 'verbose' (bool).

    Attributes:
        lambda_matrix (np.ndarray): Learned topic-word distributions.
        cluster_centers_ (np.ndarray): KMeans cluster centers.
        user_profiles_cache (dict): Cache mapping `meter_id` to sampled theta distributions.
        gamma_cache (dict): Cache for concentration parameters `{meter_id: gamma_u}`.
        is_fitted (bool): Indicates if the LDA model has been fitted.
    """
    def __init__(self, n_topics=10, num_clusters=100,
                 random_state=None,
                 path='models/UserProfiles/user_profiles.pkl',
                 decay=0.9,
                 **kwargs):
        super(UserProfiler, self).__init__(n_components=n_topics, random_state=random_state)
        self.n_topics = n_topics
        self.num_clusters = num_clusters
        self.condition_dim = n_topics
        self.decay = decay

        self.lambda_matrix = None
        self.cluster_centers_ = None
        self.user_profiles_cache = {}
        self.gamma_cache = {}
        self.user_last_seen = {}

        self.random_state = random_state
        self.path = path
        self.verbose = kwargs.get('verbose', False)

        self.batch_counter = 0
        self.last_update_batch = 0
        self.kmeans_initialized = False
        self.is_fitted = False
        self._profile_lookup_tensors = {}

        self.kmeans = self._build_kmeans()

        if self.path is not None:
            try:
                self.load(self.path)
            except Exception as e:
                if self.verbose:
                    print(f"Could not load existing UserProfiler model from {self.path}. Will fit a new model. Error: {e}")

    def _build_kmeans(self) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            batch_size=1024,
            n_init='auto'
        )

    def _reset_kmeans_state(self):
        """Reset kmeans + LDA state while preserving the user profile cache."""
        self.lambda_matrix = None
        self.cluster_centers_ = None
        self.kmeans_initialized = False
        self.is_fitted = False
        self._invalidate_profile_lookup()


    def _current_feature_dim(self) -> int | None:
        if self.cluster_centers_ is not None and getattr(self.cluster_centers_, "ndim", 0) == 2:
            return int(self.cluster_centers_.shape[1])
        n_features = getattr(self.kmeans, "n_features_in_", None)
        if n_features is None:
            return None
        return int(n_features)

    def _ensure_kmeans_feature_dim(self, incoming_feature_dim: int, context: str):
        expected_feature_dim = self._current_feature_dim()
        if expected_feature_dim is None or expected_feature_dim == int(incoming_feature_dim):
            return

        if self.verbose:
            print(
                f"[UserProfiler] Feature dimension changed in {context} "
                f"({expected_feature_dim} -> {incoming_feature_dim}). Resetting cached profiler state."
            )

        self.kmeans = self._build_kmeans()
        self._reset_kmeans_state()

    def _invalidate_profile_lookup(self):
        self._profile_lookup_tensors = {}

    def _prepare_kmeans_input(self, batch_data, context: str) -> np.ndarray:
        """Project batch input to [B, D] using current profiler feature-width contract.

        Accepts 2D [B, D] and 3D [B, F, L]/[B, L, F] tensors.
        For 3D input, chooses a projection that matches the fitted KMeans feature
        width when available:
            - D == L: collapse feature/phase axis by summation.
            - D == L*F: flatten after canonicalizing to [B, L, F].

        SHAPER:
          batch_data : [batch_size, feature_dim] or [batch_size, num_phases, seq_len] or [batch_size, seq_len, num_phases]
          ->         : [batch_size, feature_dim]
        """
        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.detach().cpu().numpy()

        x = np.asarray(batch_data)
        if x.ndim == 2:
            return x.astype(np.float32, copy=False)

        if x.ndim != 3:
            raise ValueError(f"{context}: expected 2D or 3D input, got {x.ndim}D")

        bsz, d1, d2 = x.shape
        expected_dim = self._current_feature_dim()

        if expected_dim is not None:
            if expected_dim == d2:
                return x.sum(axis=1, dtype=np.float32)
            if expected_dim == d1:
                return x.sum(axis=2, dtype=np.float32)
            if expected_dim == d1 * d2:
                if d1 in (1, 3) and d2 not in (1, 3):
                    return np.transpose(x, (0, 2, 1)).reshape(bsz, -1).astype(np.float32, copy=False)
                return x.reshape(bsz, -1).astype(np.float32, copy=False)

        if d1 in (1, 3) and d2 not in (1, 3):
            return x.sum(axis=1, dtype=np.float32)
        if d2 in (1, 3) and d1 not in (1, 3):
            return x.sum(axis=2, dtype=np.float32)
        return x.reshape(bsz, -1).astype(np.float32, copy=False)

    def _get_profile_lookup_for_device(self, device: torch.device):
        key = str(device)
        cached = self._profile_lookup_tensors.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        # Cache miss or stale entry (e.g. DeviceUnpickler remapped tensors to a
        # different device than the cache key implies) — rebuild for this device.
        if key in self._profile_lookup_tensors:
            del self._profile_lookup_tensors[key]

        if len(self.user_profiles_cache) == 0:
            empty_lookup = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, self.n_topics, dtype=torch.float32, device=device),
            )
            self._profile_lookup_tensors[key] = empty_lookup
            return empty_lookup

        meter_ids_np = np.array([int(m) for m in self.user_profiles_cache.keys()], dtype=np.int64)
        order = np.argsort(meter_ids_np)
        meter_ids_sorted = meter_ids_np[order]
        profiles_sorted = np.stack(
            [
                np.asarray(self.user_profiles_cache[int(m_id)], dtype=np.float32)
                for m_id in meter_ids_sorted
            ],
            axis=0,
        )

        lookup = (
            torch.from_numpy(meter_ids_sorted).to(device=device, dtype=torch.long),
            torch.from_numpy(profiles_sorted).to(device=device, dtype=torch.float32),
        )
        self._profile_lookup_tensors[key] = lookup
        return lookup

    def fit(self, train_loader):
        """
        Fits the UserProfiler using batches from a PyTorch DataLoader.

        Uses streaming/incremental learning to handle large datasets. Phases include
        incremental clustering (KMeans), corpus creation, and LDA fitting.

        Args:
            train_loader (torch.utils.data.DataLoader): PyTorch DataLoader yielding
                tuples of `(data, meter_id, timestamps)`. `data` must have shape
                `(batch_size, seq_len, num_features)`.

        Returns:
            UserProfiler: The fitted profiler instance.
        """
        if self.verbose:
            print("Fitting UserProfiler model")
            print("Phase 1: Clustering to create 'words")

        self.user_profiles_cache = {}
        self.gamma_cache = {}
        self.user_last_seen = {}
        self._invalidate_profile_lookup()

        # === Phase 1: Incremental clustering to create "words" ===
        batch_count = 0
        total_timesteps = 0

        for data, meter_id, _, _ in tqdm(train_loader, desc="Clustering batches"):
            batch_size, seq_len, num_features = data.shape  # data: [batch_size, seq_len, num_features]
            data_flat = data.cpu().numpy().reshape(batch_size, seq_len * num_features)  # SHAPER: [batch_size, seq_len * num_features]

            valid_mask = ~np.isnan(data_flat).any(axis=1)
            data_fit = data_flat[valid_mask]
            if len(data_fit) == 0:
                continue

            self._ensure_kmeans_feature_dim(data_fit.shape[1], context="fit")
            if data_fit.shape[0] < self.num_clusters:
                repeat_factor = int(math.ceil(self.num_clusters / max(1, data_fit.shape[0])))
                idx = np.tile(np.arange(data_fit.shape[0]), repeat_factor)[:self.num_clusters]
                data_kmeans = data_fit[idx]
            else:
                data_kmeans = data_fit

            self.kmeans.partial_fit(data_kmeans)
            total_timesteps += data_fit.shape[0]
            batch_count += 1

        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True

        if self.verbose:
            print(f"Clustering complete over {batch_count} batches with {total_timesteps} timesteps.")
            print("Phase 2: Creating corpus and fitting LDA model")

        # === Phase 2: Create corpus and fit LDA model ===
        meter_corpus = {}

        for data, meter_id, _, _ in tqdm(train_loader, desc="Creating corpus batches"):
            batch_size, seq_len, num_features = data.shape  # data: [batch_size, seq_len, num_features]
            data_batch = data.cpu().numpy().reshape(batch_size, seq_len * num_features)  # SHAPER: [batch_size, seq_len * num_features]
            meter_id_flat = meter_id.cpu().numpy()

            valid_mask = ~np.isnan(data_batch).any(axis=1)
            data_fit = data_batch[valid_mask]
            meter_id_fit = meter_id_flat[valid_mask]
            if len(data_fit) == 0:
                continue

            labels = cdist(data_fit, self.cluster_centers_).argmin(axis=1)

            for m_id, label in zip(meter_id_fit, labels):
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
                meter_corpus[m_id][label] += 1

        meter_ids_sorted = sorted(meter_corpus.keys())
        X_corpus = np.array([meter_corpus[m_id] for m_id in meter_ids_sorted])

        if self.verbose:
            print(f"Corpus shape: {X_corpus.shape}")
            print("Phase 3: Fitting LDA model")

        # === Phase 3: Fit LDA model ===
        self.doc_lengths = X_corpus.sum(axis=1)
        super().fit(X_corpus)
        self.lambda_matrix = self.components_
        self.is_fitted = True

        # === Phase 4: Cache gamma parameters for all users ===
        if self.verbose:
            print("Phase 4: Caching gamma parameters for all users")

        self.meter_id_to_idx = {m_id: idx for idx, m_id in enumerate(meter_ids_sorted)}
        all_gamma = super().transform(X_corpus, normalize=False)

        for m_id, gamma_u in zip(meter_ids_sorted, all_gamma):
            self.gamma_cache[m_id] = gamma_u
            self.user_last_seen[m_id] = self.batch_counter
            self.resample_user_vector(m_id)

        self._invalidate_profile_lookup()

        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.save(self.path)
            if self.verbose:
                print(f"UserProfiler model saved to {self.path}.")
        return self

    def get_conditions(self, data, meter_id=None, timestamps=None):
        """Gets topic distributions for a batch during inference.

        Looks up cached profiles by `meter_id`. If not found, computes on-the-fly.

        Args:
            data (torch.Tensor | np.ndarray | None): Time series data of shape
                `(batch_size, seq_len, num_features)`. Can be None if profiles are cached.
            meter_id (torch.Tensor | np.ndarray): 1D array/tensor of meter IDs of shape
                `(batch_size,)`. Required.
            timestamps (torch.Tensor | None, optional): Timestamps of shape `(batch_size,)`.
                Unused for user profiles. Defaults to None.

        Returns:
            torch.Tensor: Topic distributions of shape `(batch_size, n_topics)` with
                dtype `torch.float32`.

        Raises:
            ValueError: If `meter_id` is None, model is not fitted, or `cluster_centers_`
                is not 2D.
        """
        if meter_id is None:
            raise ValueError("meter_id is required for UserProfiler.get_conditions()")

        fitted_and_ready = (hasattr(self, 'is_fitted') and self.is_fitted) and self.cluster_centers_ is not None
        if not fitted_and_ready and not self.user_profiles_cache:
            try:
                self.load(self.path)
            except Exception as e:
                raise ValueError("UserProfiler model is not fitted or cluster centers missing.") from e

        if self.cluster_centers_ is not None and self.cluster_centers_.ndim != 2:
            raise ValueError(f"cluster_centers_ must be 2D, got shape {self.cluster_centers_.shape}")

        target_device = torch.device("cpu")
        if isinstance(meter_id, torch.Tensor):
            target_device = meter_id.device
            meter_id = meter_id.detach().cpu().numpy().astype(np.int64, copy=False)
        elif isinstance(meter_id, np.ndarray):
            meter_id = meter_id.astype(np.int64, copy=False)
        else:
            meter_id = np.asarray(meter_id, dtype=np.int64)

        meter_id = np.atleast_1d(meter_id)

        self.batch_counter += 1
        for m_id in meter_id:
            self.user_last_seen[int(m_id)] = self.batch_counter

        return self._get_cached_profiles(data, meter_id, device=target_device)


    def _get_cached_profiles(self, data, meter_ids, device: torch.device = torch.device("cpu")):
        """Assembles profile vectors from cache.

        Computes on-the-fly for unknown meters. Falls back to a uniform distribution
        for meters with no data available.

        Args:
            data (torch.Tensor | np.ndarray | None): Time series data batch.
            meter_ids (np.ndarray): Array of meter IDs.

        Returns:
            torch.Tensor: Profiles of shape `(batch_size, n_topics)`.
        """
        meter_ids = np.ascontiguousarray(np.asarray(meter_ids, dtype=np.int64))

        missing_meters = [
            int(m) for m in np.unique(meter_ids)
            if int(m) not in self.user_profiles_cache
        ]
        if missing_meters:
            if data is not None and self.cluster_centers_ is not None:
                self._compute_profiles_for_batch(data, meter_ids, missing_meters)
            else:
                for m_id in missing_meters:
                    self.user_profiles_cache[int(m_id)] = np.ones(self.n_topics, dtype=np.float32) / self.n_topics
                self._invalidate_profile_lookup()

        meter_ids_t = torch.as_tensor(meter_ids, dtype=torch.long, device=device).contiguous()
        lookup_ids, lookup_profiles = self._get_profile_lookup_for_device(device)

        if lookup_ids.numel() == 0:
            return torch.full(
                (meter_ids_t.shape[0], self.n_topics),
                fill_value=(1.0 / self.n_topics),
                dtype=torch.float32,
                device=device,
            )

        idx = torch.searchsorted(lookup_ids, meter_ids_t)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids_t)

        profiles = torch.full(
            (meter_ids_t.shape[0], self.n_topics),
            fill_value=(1.0 / self.n_topics),
            dtype=torch.float32,
            device=device,
        )
        if valid.any():
            profiles[valid] = lookup_profiles[idx_clamped[valid]]

        return profiles

    def _compute_profiles_for_batch(self, data, meter_id, target_meter_ids):
        """Computes and caches gamma and profile vectors for specific meters.

        Args:
            data (torch.Tensor | np.ndarray): Time series data batch.
            meter_id (torch.Tensor | np.ndarray): Corresponding meter IDs.
            target_meter_ids (list): List of meter IDs that require computation.

        Raises:
            ValueError: If data dimensionality is incorrect or feature sizes mismatch.
        """
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()

        x_batch = self._prepare_kmeans_input(data, context="_compute_profiles_for_batch")

        valid_mask = ~np.isnan(x_batch).any(axis=1)
        x_fit = x_batch[valid_mask]
        meter_id_valid = meter_id[valid_mask]

        if x_fit.shape[1] != self.cluster_centers_.shape[1]:
            print(f"Warning: Feature dimension mismatch (x:{x_fit.shape[1]}, centers:{self.cluster_centers_.shape[1]}). Using uniform init for missing meters.")
            random_profiles = np.random.dirichlet(np.ones(self.n_topics), size=len(target_meter_ids))
            for i, m_id in enumerate(target_meter_ids):
                self.user_profiles_cache[int(m_id)] = random_profiles[i].astype(np.float32)
            self._invalidate_profile_lookup()
            return

        target_set = {int(m) for m in target_meter_ids}
        selected_mask = np.isin(meter_id_valid.astype(np.int64, copy=False), np.fromiter(target_set, dtype=np.int64))
        x_selected = x_fit[selected_mask]
        meter_id_selected = meter_id_valid[selected_mask]

        if x_selected.shape[0] == 0:
            return

        labels = cdist(x_selected, self.cluster_centers_).argmin(axis=1)
        meter_corpus = {}
        for m_id, label in zip(meter_id_selected, labels):
            if m_id not in meter_corpus:
                meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
            meter_corpus[m_id][label] += 1

        computable = [m for m in target_meter_ids if m in meter_corpus]
        if not computable:
            return

        X_corpus = np.array([meter_corpus[m_id] for m_id in computable], dtype=np.int32)
        posterior = super().transform(X_corpus, normalize=False)

        for m_id, gamma_u in zip(computable, posterior):
            self.gamma_cache[int(m_id)] = gamma_u
            self.resample_user_vector(int(m_id))

        self._invalidate_profile_lookup()

    def fit_on_files(self, parquet_paths: list[str],
                     window_spec=None,
                     feature_cols=None,
                     hourly_aggregation: bool = False,
                     aggregation_divisor: float = 1.0,
                     phased: bool = True,
                     meter_col: str = "meter_number",
                     max_hour_value: float | None = None):
        """Fit directly from parquet files without a streaming dataloader.

        Does two explicit passes over files (KMeans then corpus) so the fit
        terminates cleanly regardless of dataset size.
        """
        import pandas as pd

        if self.path and os.path.exists(self.path):
            if self.verbose:
                print(f"[UserProfiler] Loading pre-fitted model from {self.path}")
            self.load(self.path)
            return self

        if not parquet_paths:
            if self.verbose:
                print("[UserProfiler] No parquet paths provided.")
            return self

        if feature_cols is None:
            feature_cols = ["hourly_energy_consumption"]

        window_len = window_spec.input_len if window_spec is not None else 24
        ts_candidates = ["timestamp_dst", "timestamp", "timestamp_utc"]
        agg_div = float(aggregation_divisor) if float(aggregation_divisor) > 0 else 1.0

        def _read_file(fpath):
            """Return (windows [M, D], meter_ids [M]) for one file, or (None, None)."""
            try:
                df = pd.read_parquet(fpath)
            except Exception as e:
                if self.verbose:
                    print(f"[UserProfiler] Skipping {fpath}: {e}")
                return None, None

            if phased:
                stem = os.path.splitext(os.path.basename(fpath))[0].replace("sm_", "")
                try:
                    meter_id = int(stem)
                except ValueError:
                    return None, None

                ts_col = next((c for c in ts_candidates if c in df.columns), None)
                if ts_col is None:
                    if df.index.name in ts_candidates:
                        df = df.reset_index()
                        ts_col = df.columns[0]
                    else:
                        return None, None

                present = [c for c in feature_cols if c in df.columns]
                if not present:
                    return None, None

                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                frame = (
                    df.set_index(ts_col)[present]
                    .apply(pd.to_numeric, errors="coerce")
                    .sort_index()
                )
                if hourly_aggregation:
                    frame = frame.resample("1h").mean()
                    if agg_div != 1.0:
                        frame /= agg_div

                vals = frame.to_numpy(dtype=np.float32)  # [T, n_phases]
                n_windows = len(vals) // window_len
                if n_windows == 0:
                    return None, None
                T = n_windows * window_len
                # [n_windows, window_len * n_phases] — same layout as fit() expects
                windows = vals[:T].reshape(n_windows, window_len, vals.shape[1]).reshape(n_windows, -1)

                if max_hour_value is not None:
                    windows = windows[windows.max(axis=1) <= max_hour_value]
                if len(windows) == 0:
                    return None, None

                return windows, np.full(len(windows), meter_id, dtype=np.int64)

            else:
                present = [c for c in feature_cols if c in df.columns]
                if not present:
                    return None, None

                if isinstance(df.index, pd.MultiIndex):
                    # (meter_id, timestamp) MultiIndex — already sorted by meter
                    meter_flat = df.index.get_level_values(0).to_numpy()
                    ts_idx = df.index.get_level_values(1)
                    if ts_idx.tz is None:
                        ts_idx = ts_idx.tz_localize("UTC")
                else:
                    ts_col = next((c for c in ts_candidates if c in df.columns), None)
                    m_col = next((c for c in df.columns if c.lower() == meter_col.lower()), None)
                    if ts_col is None or m_col is None:
                        return None, None
                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                    df = df.dropna(subset=[ts_col, m_col]).sort_values([m_col, ts_col])
                    meter_flat = df[m_col].to_numpy()

                vals_flat = df[present].to_numpy(dtype=np.float32).sum(axis=1)

                unique_m, starts = np.unique(meter_flat, return_index=True)
                ends = np.append(starts[1:], len(vals_flat))

                win_parts, m_parts = [], []
                for m_id, s, e in zip(unique_m, starts, ends):
                    n_w = (e - s) // window_len
                    if n_w == 0:
                        continue
                    T = n_w * window_len
                    w = vals_flat[s: s + T].reshape(n_w, window_len)
                    if max_hour_value is not None:
                        w = w[w.max(axis=1) <= max_hour_value]
                    if len(w) == 0:
                        continue
                    win_parts.append(w)
                    m_parts.append(np.full(len(w), int(m_id), dtype=np.int64))

                if not win_parts:
                    return None, None
                return np.vstack(win_parts), np.concatenate(m_parts)

        # Reset state
        self.user_profiles_cache = {}
        self.gamma_cache = {}
        self.user_last_seen = {}
        self._invalidate_profile_lookup()

        if self.verbose:
            print(f"[UserProfiler] Fitting across {len(parquet_paths)} parquet files...")

        # === Phase 1: Incremental KMeans ===
        total_windows = 0
        for fpath in tqdm(parquet_paths, desc="[UserProfiler] Phase 1: KMeans"):
            windows, _ = _read_file(fpath)
            if windows is None:
                continue
            valid = ~np.isnan(windows).any(axis=1)
            windows = windows[valid]
            if len(windows) == 0:
                continue
            self._ensure_kmeans_feature_dim(windows.shape[1], context="fit_on_files")
            if windows.shape[0] < self.num_clusters:
                reps = math.ceil(self.num_clusters / max(1, windows.shape[0]))
                idx = np.tile(np.arange(windows.shape[0]), reps)[:self.num_clusters]
                windows = windows[idx]
            self.kmeans.partial_fit(windows)
            total_windows += len(windows)

        if not hasattr(self.kmeans, "cluster_centers_"):
            print("[UserProfiler] WARNING: no valid windows found in any file; aborting fit.")
            return self

        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True
        if self.verbose:
            print(f"[UserProfiler] KMeans done ({total_windows:,} windows).")

        # === Phase 2: Build per-meter word-count corpus ===
        meter_corpus: dict[int, np.ndarray] = {}
        for fpath in tqdm(parquet_paths, desc="[UserProfiler] Phase 2: Corpus"):
            windows, meter_ids = _read_file(fpath)
            if windows is None:
                continue
            valid = ~np.isnan(windows).any(axis=1)
            windows, meter_ids = windows[valid], meter_ids[valid]
            if len(windows) == 0:
                continue
            labels = cdist(windows, self.cluster_centers_).argmin(axis=1)
            for m_id, label in zip(meter_ids, labels):
                m_id = int(m_id)
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
                meter_corpus[m_id][label] += 1

        meter_ids_sorted = sorted(meter_corpus.keys())
        X_corpus = np.array([meter_corpus[m_id] for m_id in meter_ids_sorted])

        # === Phase 3: Fit LDA ===
        if self.verbose:
            print(f"[UserProfiler] Phase 3: LDA on {len(meter_ids_sorted):,} meters")
        self.doc_lengths = X_corpus.sum(axis=1)
        super(UserProfiler, self).fit(X_corpus)
        self.lambda_matrix = self.components_
        self.is_fitted = True

        # === Phase 4: Cache gamma parameters ===
        if self.verbose:
            print("[UserProfiler] Phase 4: Caching gamma")
        self.meter_id_to_idx = {m_id: idx for idx, m_id in enumerate(meter_ids_sorted)}
        all_gamma = super(UserProfiler, self).transform(X_corpus, normalize=False)
        for m_id, gamma_u in zip(meter_ids_sorted, all_gamma):
            self.gamma_cache[int(m_id)] = gamma_u
            self.user_last_seen[int(m_id)] = self.batch_counter
            self.resample_user_vector(int(m_id))
        self._invalidate_profile_lookup()

        if self.path is not None:
            dir_part = os.path.dirname(self.path)
            if dir_part:
                os.makedirs(dir_part, exist_ok=True)
            self.save(self.path)
            if self.verbose:
                print(f"[UserProfiler] Saved to {self.path}")

        return self

    def resample_user_vector(self, meter_id=None):
        """Samples new user profiles from stored gamma parameters using a Dirichlet distribution.

        Args:
            meter_id (str | int, optional): Specific meter ID to resample. If None,
                resamples all cached users. Defaults to None.

        Raises:
            ValueError: If no gamma parameters are cached or `meter_id` is not found.
        """
        if not self.gamma_cache:
            raise ValueError("No gamma parameters cached. Cannot resample user vectors.")
        if meter_id is None:
            for m_id, gamma_u in self.gamma_cache.items():
                self.user_profiles_cache[m_id] = np.random.dirichlet(gamma_u)
        else:
            if meter_id not in self.gamma_cache:
                raise ValueError(f"Meter ID {meter_id} not found in gamma cache.")
            self.user_profiles_cache[meter_id] = np.random.dirichlet(self.gamma_cache[meter_id])
        self._invalidate_profile_lookup()

    def get_random_conditions(self, batch_size):
        """Generates random user profiles for testing and random synthetic profiles.

        Args:
            batch_size (int): Number of profiles to generate.

        Returns:
            torch.Tensor: Random profile tensor of shape `(batch_size, n_topics)`.
        """
        random_profiles = np.random.dirichlet(alpha=np.ones(self.n_topics), size=batch_size)
        return torch.tensor(random_profiles, dtype=torch.float32)

    def save(self, path):
        """Saves the UserProfiler model state to a pickle file.

        Args:
            path (str): Filepath to save the model.
        """
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        """Loads model state from a pickle file and updates the instance state.

        Args:
            path (str): Filepath to load the model from.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        self._invalidate_profile_lookup()
        if self.verbose:
            print(f"Loaded model with {len(self.user_profiles_cache)} cached profiles")