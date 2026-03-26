import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from abc import ABC, abstractmethod
 
# =============
# LDA conditions and user profiles
# =============
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm

class BaseConditioner(ABC):
    """Abstract base class for all conditioners.

    Any custom conditioner must inherit from this class and implement
    `get_conditions` and `get_condition_dim`.
    """

    def __init__(self):
        self.condition_dim: int = 0

    @abstractmethod
    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Generates the conditioning tensor based on input"""
        pass
    
    @abstractmethod
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random conditions for testing and validation"""
        pass


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
    def __init__(self, n_topics = 10, num_clusters = 100, 
                 random_state=None, 
                 path='models/UserProfiles/user_profiles.pkl', 
                 decay=0.9,
                 **kwargs):
        super(UserProfiler, self).__init__(n_components=n_topics, random_state=random_state)
        self.n_topics = n_topics
        self.num_clusters = num_clusters
        self.condition_dim = n_topics
        self.decay = decay

        self.lambda_matrix = None  # To hold the learned topic-word distributions
        self.cluster_centers_ = None  # To hold KMeans cluster centers
        self.user_profiles_cache = {}  # Cache for user profiles {meter_id: sampled_theta}
        self.gamma_cache = {}  # Cache for concentration parameters {meter_id: gamma_u}
        self.user_last_seen = {}  # For tracking last seen batch for decay

        self.random_state = random_state
        self.path = path
        self.verbose = kwargs.get('verbose', False)

        self.batch_counter = 0
        self.last_update_batch = 0
        self.kmeans_initialized = False
        self.is_fitted = False

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            batch_size=1024,
            n_init='auto'
        )

        try:
            self.load(self.path)
        except Exception as e:
            if self.verbose:
                print(f"Could not load existing UserProfiler model from {self.path}. Will fit a new model. Error: {e}")

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
        # TODO: they scale the data here?? or before??
        # TODO: reduce dimensions (svd or pca) before clustering??

        if self.verbose:
            print("Fitting UserProfiler model")
            print("Phase 1: Clustering to create 'words")
        
        # === Phase 1: Incremental clustering to create "words" ===
        batch_count = 0
        total_timesteps = 0

        for data, c, meter_id, timestamps in tqdm(train_loader, desc="Clustering batches"):
            batch_size, seq_len, num_features = data.shape
            data_flat = data.cpu().numpy().reshape(batch_size, seq_len * num_features)  # [B, L*F]

            # Remove missing data
            valid_mask = ~np.isnan(data_flat).any(axis=1)
            data_fit = data_flat[valid_mask]
            if len(data_fit) == 0:
                continue

            self.kmeans.partial_fit(data_fit)
            total_timesteps += data_fit.shape[0]
            batch_count += 1
        
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True

        if self.verbose:
            print(f"Clustering complete over {batch_count} batches with {total_timesteps} timesteps.")
            print("Phase 2: Creating corpus and fitting LDA model")

        # === Phase 2: Create corpus and fit LDA model ===
        meter_corpus = {} # dict: meter_id -> cluster counts

        for data, c, meter_id, timestamps in tqdm(train_loader, desc="Creating corpus batches"):
            batch_size, seq_len, num_features = data.shape
            data_batch = data.cpu().numpy().reshape(batch_size, seq_len * num_features)  # [B*T, F]
            meter_id_flat = meter_id.cpu().numpy() 

            # Remove missing data (reminder: this should not be an issue at any point since it is already handled in data processing, but we keep it here for safety)
            valid_mask = ~np.isnan(data_batch).any(axis=1)
            data_fit = data_batch[valid_mask]
            meter_id_fit = meter_id_flat[valid_mask]
            if len(data_fit) == 0:
                continue

            # Assign clusters
            labels = cdist(data_fit, self.cluster_centers_).argmin(axis=1)

            # count clusters per meter
            for m_id, label in zip(meter_id_fit, labels):
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters,dtype=np.int32)
                meter_corpus[m_id][label] += 1

        # Convert to document term matrix
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

        # Cache gamma values and smaple profiles
        for m_id, gamma_u in zip(meter_ids_sorted, all_gamma):
            self.gamma_cache[m_id] = gamma_u
            self.user_last_seen[m_id] = self.batch_counter
            self.resample_user_vector(m_id)

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

        if not (hasattr(self, 'is_fitted') and self.is_fitted) or self.cluster_centers_ is None:
            try:
                self.load(self.path)
            except Exception as e:
                raise ValueError("UserProfiler model is not fitted or cluster centers missing.") from e

        if self.cluster_centers_.ndim != 2:
            raise ValueError(f"cluster_centers_ must be 2D, got shape {self.cluster_centers_.shape}")

        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()

        self.batch_counter += 1
        for m_id in meter_id:
            self.user_last_seen[m_id] = self.batch_counter

        return self._get_cached_profiles(data, meter_id)

    def update_model_incremental(self, x_batch, meter_ids):
        """Incrementally updates the KMeans clustering and LDA topic model.
        
        New users are added to the cache; existing users are updated with 
        exponential decay.

        Args:
            x_batch (torch.Tensor | np.ndarray): Batch data of shape 
                `(batch_size, seq_len, num_features)` or `(batch_size, features)`.
            meter_ids (torch.Tensor | np.ndarray): Corresponding meter IDs of shape 
                `(batch_size,)`.

        Raises:
            ValueError: If `x_batch` does not have 2 or 3 dimensions.
        """
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.cpu().numpy()
        if isinstance(meter_ids, torch.Tensor):
            meter_ids = meter_ids.cpu().numpy()

        if x_batch.ndim == 3:
            batch_size, seq_len, num_features = x_batch.shape
            x_flat = x_batch.reshape(batch_size, seq_len * num_features)
        elif x_batch.ndim == 2:
            x_flat = x_batch
        else:
            raise ValueError(f"Expected x to have 2 or 3 dimensions, got {x_batch.ndim}")

        valid_mask = ~np.isnan(x_flat).any(axis=1)
        x_valid = x_flat[valid_mask]
        meter_ids_valid = meter_ids[valid_mask]

        if len(x_valid) == 0:
            return
        

        # MiniBatchKMeans requires n_samples >= n_clusters on each partial_fit call.
        # For small batches, upsample with replacement to avoid runtime failure.
        if x_valid.shape[0] < self.num_clusters:
            repeat_factor = int(math.ceil(self.num_clusters / max(1, x_valid.shape[0])))
            idx = np.tile(np.arange(x_valid.shape[0]), repeat_factor)[:self.num_clusters]
            x_kmeans = x_valid[idx]
            meter_ids_kmeans = meter_ids_valid[idx]
        else:
            x_kmeans = x_valid
            meter_ids_kmeans = meter_ids_valid

        self.kmeans.partial_fit(x_kmeans)
        # self.kmeans.partial_fit(x_valid)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True

        labels = cdist(x_kmeans, self.cluster_centers_).argmin(axis=1)
        meter_corpus = {}
        for m_id, label in zip(meter_ids_kmeans, labels):
            if m_id not in meter_corpus:
                meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
            meter_corpus[m_id][label] += 1

        unique_meters = sorted(meter_corpus.keys())
        X_corpus = np.array([meter_corpus[m_id] for m_id in unique_meters])

        if self.is_fitted:
            super().partial_fit(X_corpus)
            new_gamma = super().transform(X_corpus, normalize=False)

            seen_in_update = set(unique_meters)
            for m_id in list(self.gamma_cache):
                if m_id not in seen_in_update:
                    batches_inactive = self.batch_counter - self.user_last_seen.get(m_id, self.batch_counter)
                    decay_factor = self.decay ** batches_inactive
                    if decay_factor < 0.1:
                        self.gamma_cache[m_id] = np.ones(self.n_topics)
                    else:
                        self.gamma_cache[m_id] = (
                            decay_factor * self.gamma_cache[m_id]
                            + (1 - decay_factor) * np.ones(self.n_topics)
                        )
                    self.resample_user_vector(m_id)

            for m_id, gamma_new in zip(unique_meters, new_gamma):
                if m_id in self.gamma_cache:
                    self.gamma_cache[m_id] = self.decay * self.gamma_cache[m_id] + (1 - self.decay) * gamma_new
                else:
                    self.gamma_cache[m_id] = gamma_new
                self.resample_user_vector(m_id)
        else:
            super().fit(X_corpus)
            self.is_fitted = True
            initial_gamma = super().transform(X_corpus, normalize=False)
            for m_id, gamma_u in zip(unique_meters, initial_gamma):
                self.gamma_cache[m_id] = gamma_u
                self.resample_user_vector(m_id)
    
    def _get_cached_profiles(self, data, meter_ids):
        """Assembles profile vectors from cache. 
        
        Computes on-the-fly for unknown meters. Falls back to a uniform distribution 
        for meters with no data available.

        Args:
            data (torch.Tensor | np.ndarray | None): Time series data batch.
            meter_ids (np.ndarray): Array of meter IDs.

        Returns:
            torch.Tensor: Profiles of shape `(batch_size, n_topics)`.
        """
        # FIX 5: compute missing profiles AND populate user_profiles_cache before assembling
        missing_meters = [m for m in meter_ids if m not in self.user_profiles_cache]
        if missing_meters:
            if data is not None and self.cluster_centers_ is not None:
                self._compute_profiles_for_batch(data, meter_ids, missing_meters)
            else:
                # No data to compute from — initialise to uniform
                for m_id in missing_meters:
                    self.user_profiles_cache[m_id] = np.ones(self.n_topics, dtype=np.float32) / self.n_topics

        profiles = np.zeros((len(meter_ids), self.n_topics), dtype=np.float32)
        for i, m_id in enumerate(meter_ids):
            profiles[i] = self.user_profiles_cache[m_id]

        return torch.tensor(profiles, dtype=torch.float32)

    def _compute_profiles_for_batch(self, data, meter_id, target_meter_ids):
        """Computes and caches gamma and profile vectors for specific meters.

        Args:
            data (torch.Tensor | np.ndarray): Time series data batch.
            meter_id (torch.Tensor | np.ndarray): Corresponding meter IDs.
            target_meter_ids (list): List of meter IDs that require computation.

        Raises:
            ValueError: If data dimensionality is incorrect or feature sizes mismatch.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()

        if data.ndim == 3:
            batch_size, seq_len, num_features = data.shape
            x_batch = data.reshape(batch_size, seq_len * num_features)
        elif data.ndim == 2:
            x_batch = data
        else:
            raise ValueError(f"Expected data to have 2 or 3 dimensions, got {data.ndim}")

        valid_mask = ~np.isnan(x_batch).any(axis=1)
        x_fit = x_batch[valid_mask]
        meter_id_valid = meter_id[valid_mask]

        if x_fit.shape[1] != self.cluster_centers_.shape[1]:
            print(f"Warning: Feature dimension mismatch (x:{x_fit.shape[1]}, centers:{self.cluster_centers_.shape[1]}). Using uniform init for missing meters.")
            random_profiles = np.random.dirichlet(np.ones(self.n_topics), size=len(target_meter_ids))
            for i, m_id in enumerate(target_meter_ids):
                self.user_profiles_cache[m_id] = torch.tensor(random_profiles[i], dtype=torch.float32)
            return

        target_set = set(target_meter_ids)
        selected_indices = [i for i, m in enumerate(meter_id_valid) if m in target_set]
        x_selected = x_fit[selected_indices]
        meter_id_selected = meter_id_valid[selected_indices]

        labels = cdist(x_selected, self.cluster_centers_).argmin(axis=1)
        meter_corpus = {}
        for m_id, label in zip(meter_id_selected, labels):
            if m_id not in meter_corpus:
                meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
            meter_corpus[m_id][label] += 1

        # Only compute for meters that actually appeared in this batch
        computable = [m for m in target_meter_ids if m in meter_corpus]
        if not computable:
            return

        X_corpus = np.array([meter_corpus[m_id] for m_id in computable], dtype=np.int32)
        posterior = super().transform(X_corpus, normalize=False)

        for m_id, gamma_u in zip(computable, posterior):
            self.gamma_cache[m_id] = gamma_u
            self.resample_user_vector(m_id)

    
    def fit_on_files(self, parquet_paths: list[str]):
        """Convenience method to fit the model directly on a list of parquet files.

        Args:
            parquet_paths (list[str]): List of file paths to parquet files containing 
                the training data.
        """
        import os
        from data.dataloader_phased import StreamingParquetBuffer, WindowSpec
        from torch.utils.data import DataLoader
        
        if self.path and os.path.exists(self.path):
            if self.verbose:
                print(f"[UserProfiler] Loading pre-fitted model from {self.path}")
            self.load(self.path)
            return self
            
        if not parquet_paths:
            if self.verbose:
                print("[UserProfiler] No parquet paths provided for fitting.")
            return self

        if self.verbose:
            print(f"[UserProfiler] Fitting across {len(parquet_paths)} parquet files...")

        # We use a purely non-overlapping stride (24) to avoid massively redundant clustering
        phased_ds = StreamingParquetBuffer(
            sm_files=parquet_paths, 
            replay_buffer_capacity=10000, 
            min_buffer_size=5184,
            window_spec=WindowSpec(input_len=24, stride=24), 
            num_workers=0
        )
        
        loader = DataLoader(phased_ds, batch_size=1024, num_workers=0)
        
        class WrapperLoader:
            def __init__(self, loader):
                self.loader = loader
            def __iter__(self):
                for x, cond in self.loader:
                    # Transpose x from [B, 3, 24] to [B, 24, 3] 
                    # so it matches the inference shape
                    x_time_first = x.transpose(1, 2)
                    
                    yield x_time_first, None, cond[:, 0, 0], cond[:, 1, 0]
                    
        self.fit(WrapperLoader(loader))
        return self
        

    def resample_user_vector(self, meter_id = None):
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
        if self.verbose:
            print(f"Saved model with {len(self.user_profiles_cache)} cached profiles")

    def load(self, path):
        """Loads model state from a pickle file and updates the instance state.

        Args:
            path (str): Filepath to load the model from.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        if self.verbose:
            print(f"Loaded model with {len(self.user_profiles_cache)} cached profiles")

            
# === Day of week conditioning ===
class DayOfWeekConditioner(BaseConditioner):
    """Generates a one-hot encoded representation of the day of the week.
    
    Attributes:
        condition_dim (int): The dimension of the generated condition (7).
    """
    def __init__(self):
        self.condition_dim = 7

    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Extracts day of the week from timestamps and returns a one-hot encoding.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps. If 
                a tensor, assumes Unix timestamps (float).

        Returns:
            torch.Tensor: One-hot encoded days of shape `(batch_size, 7)`. Respects 
                input tensor device if applicable.
        """
        # Handle both tensor of floats and list of pandas Timestamps
        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            # Convert Unix timestamps back to pandas Timestamps
            days_of_week = ((timestamps / 86400 + 4) % 7).long()  # 1970-01-01 was a Thursday
        else:
            days_of_week = []
            for ts in timestamps:
                if isinstance(ts, (int, float)):
                    ts = pd.Timestamp(ts, unit='s')
                days_of_week.append(ts.dayofweek)  # 0=Monday, 6=Sunday
            days_of_week = torch.tensor(days_of_week, dtype=torch.long)
            device = 'cpu'
        
        dow_onehot = F.one_hot(days_of_week, num_classes=7).float().to(device)
        return dow_onehot
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random day of week conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Random one-hot tensor of shape `(batch_size, 7)`.
        """
        random_days = torch.randint(0,7,(batch_size,))
        dow_onehot = F.one_hot(random_days, num_classes=7).float()
        return dow_onehot

class MonthOfYearConditioner(BaseConditioner):
    """Generates a one-hot encoded representation of the month of the year.
    
    Attributes:
        condition_dim (int): The dimension of the generated condition (12).
    """
    def __init__(self):
        self.condition_dim = 12

    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Extracts month from timestamps and returns a one-hot encoding.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps.

        Returns:
            torch.Tensor: One-hot encoded months of shape `(batch_size, 12)`.
        """
        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            # Convert Unix timestamps to months (0-11)
            months = ((timestamps / 86400 / 30 + 1) % 12).long()  # Approximate month calculation
        else:
            months = []
            for ts in timestamps:
                if isinstance(ts, (int, float)):
                    ts = pd.Timestamp(ts, unit='s')
                months.append(ts.month - 1)  # pandas month is 1-12, we want 0-11
            months = torch.tensor(months, dtype=torch.long)
            device = 'cpu'
        
        moy_onehot = F.one_hot(months, num_classes=12).float().to(device)
        return moy_onehot
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random month conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Random one-hot tensor of shape `(batch_size, 12)`.
        """
        random_months = torch.randint(0,12,(batch_size,))
        moy_onehot = F.one_hot(random_months, num_classes=12).float()
        return moy_onehot

class HolidayConditioner(BaseConditioner):
    """Flags whether a given timestamp falls on a known holiday.
    As defined by: https://helligdage-dk.dk/helligdage-2024/
    
    Args:
        holiday_path (str): Path to JSON file containing holiday dates. 
            Defaults to 'data/holidays/denmark_holidays.json'.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        holiday_dates (set): Set of loaded holiday dates.
    """
    def __init__(self, holiday_path='data/holidays/denmark_holidays.json'):
        self.condition_dim = 1
        with open(holiday_path, 'r') as f:
            holidays = json.load(f)
        self.holiday_dates = set(pd.to_datetime(holidays['dates']).date)
    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Checks if timestamps are holidays.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps.

        Returns:
            torch.Tensor: Binary holiday indicators of shape `(batch_size, 1)`.
        """
        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            dates = pd.to_datetime(timestamps.cpu().numpy(), unit='s').date
            is_holiday = torch.tensor([1.0 if d in self.holiday_dates else 0.0 for d in dates], device=device).unsqueeze(1)
        else:
            is_holiday = []
            for ts in timestamps:
                if isinstance(ts, (int, float)):
                    ts = pd.Timestamp(ts, unit='s')
                is_holiday.append(1.0 if ts.date() in self.holiday_dates else 0.0)
            is_holiday = torch.tensor(is_holiday, dtype=torch.float32).unsqueeze(1)
        return is_holiday
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random holiday conditions for testing.

            Args:
                batch_size (int): Number of conditions to generate.

            Returns:
                torch.Tensor: Binary tensor of shape `(batch_size, 1)`.
        """
        random_holiday = torch.randint(0,2,(batch_size,1)).float()
        return random_holiday
    
class HeatpumpConditioner(BaseConditioner):
    """Flags whether a specific meter is associated with a heat pump.

    Args:
        topology_path (str): Path to CSV file mapping meters to heat pump status.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        topology_df (pd.DataFrame): Loaded topology data.
    """
    def __init__(self, topology_path = '/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        self.topology_df = pd.read_csv(topology_path)
    
    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Retrieves heat pump status for given meter IDs.

        Args:
            x (Any): Unused input data.
            meter_id (torch.Tensor | np.ndarray): Meter IDs.
            timestamps (Any): Unused timestamps.

        Returns:
            torch.Tensor: Binary indicators of shape `(batch_size, 1)`. Defaults to 
                0.0 if meter is not found.
        """
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()
        
        heatpump_status = []
        for m_id in meter_id:
            row = self.topology_df[self.topology_df['meter_number'] == m_id]
            if not row.empty:
                val = row['has_heat_pump'].iloc[0]
                if str(val) == "True":
                    heatpump_status.append(1.0)
                else:
                    heatpump_status.append(0.0)
            else:
                heatpump_status.append(0.0)

        return torch.tensor(heatpump_status, dtype=torch.float32).unsqueeze(1)
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random holiday conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Binary tensor of shape `(batch_size, 1)`.
        """
        random_status = torch.randint(0,2,(batch_size,1)).float()
        return random_status
    
class SolarPanelConditioner(BaseConditioner):
    """Flags whether a specific meter is associated with solar panels.

    Args:
        topology_path (str): Path to CSV file mapping meters to solar panel status.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        topology_df (pd.DataFrame): Loaded topology data.
    """
    def __init__(self, topology_path = '/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        self.topology_df = pd.read_csv(topology_path)
    
    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Retrieves solar panel status for given meter IDs.

        Args:
            x (Any): Unused input data.
            meter_id (torch.Tensor | np.ndarray): Meter IDs.
            timestamps (Any): Unused timestamps.

        Returns:
            torch.Tensor: Binary indicators of shape `(batch_size, 1)`. Defaults to 
                0.0 if meter is not found.
        """
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()
        
        solar_status = []
        for m_id in meter_id:
            row = self.topology_df[self.topology_df['meter_number'] == m_id]
            if not row.empty:
                val = row['has_solar_panel'].iloc[0]
                if str(val) == "True":
                    solar_status.append(1.0)
                else:
                    solar_status.append(0.0)
            else:
                solar_status.append(0.0)

        return torch.tensor(solar_status, dtype=torch.float32).unsqueeze(1)
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random solar panel conditions for testing.
        
        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Binary tensor of shape `(batch_size, 1)`.
        """
        random_status = torch.randint(0,2,(batch_size,1)).float()
        return random_status

class ElectricVehicleConditioner():
    # TODO: WAIT FOR EV data to arrive
    def __init__(self, topology_path = '/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 0

    def get_conditions(self, x, meter_id, timestamps):
        return None
    
    def get_random_conditions(self, batch_size):
        return None
    
    def get_conditions_no_data(self, meter_id, timestamps):
        return None


class ActivePhaseCountConditioner(BaseConditioner):
    """
    Conditions the model on HOW MANY phases are active in the current window.
    Outputs a one-hot vector of length 4 (representing 0, 1, 2, or 3 active phases).
    """
    def __init__(self, activity_threshold=1e-5):
        super().__init__()
        self.condition_dim = 4
        # Anything with an absolute sum below this threshold is considered "dead"
        self.activity_threshold = activity_threshold

    def get_conditions(
        self, 
        x: torch.Tensor | np.ndarray | None, 
        meter_id: torch.Tensor | np.ndarray | None, 
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        
        if x is None:
            raise ValueError("ActivePhaseCountConditioner requires input data 'x' to calculate phase activity.")
            
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
            
        # x arrives from SetGuideVAE._build_condition_input as [B, L, F] (Time-First)
        # 1. Sum the absolute values over the time dimension (L) to get total activity per phase
        # Shape: [B, F]
        phase_activity = x.abs().sum(dim=1)
        
        # 2. Check which phases cross the active threshold
        # Shape: [B, F] (Boolean)
        is_active = phase_activity > self.activity_threshold
        
        # 3. Count how many phases are active per batch
        # Shape: [B] (Values 0, 1, 2, or 3)
        active_count = is_active.sum(dim=1).long()
        
        # 4. Convert to One-Hot encoding [B, 4]
        count_onehot = F.one_hot(active_count, num_classes=self.condition_dim).float()
        
        return count_onehot.to(x.device)
    
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """
        For random synthetic generation, we probabilistically sample the number of active phases.
        (You can adjust these weights based on your real dataset distribution!)
        """
        # Example probabilities: 
        # 0 active: 0%, 1 active: 20%, 2 active: 30%, 3 active: 50%
        probs = torch.tensor([0.0, 0.2, 0.3, 0.5])
        random_counts = torch.multinomial(probs, batch_size, replacement=True)
        
        return F.one_hot(random_counts, num_classes=self.condition_dim).float()

    

# =============
# Conditioner Module - Holds an object that creates and determines type of conditioning to be used in Models
# ============

class Conditioner():
    """Central manager that aggregates multiple condition modules.

    Holds objects that create and determine the type of conditioning used in models.
    Concatenates individual conditions into a single tensor.

    Args:
        tags (list[str], optional): List of active condition tags. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        tags (list[str]): Registered condition tags (e.g., "user_profiles", "days").
        condition_dim (int): Total concatenated dimension of all active conditions.
        condition_items (dict): Mapping of tags to their respective conditioner instances.
    """
    def __init__(self, tags: list[str] | None = None, **kwargs):
        self.tags = tags if tags is not None else []
        self.condition_dim = 0
        self.condition_items = {}
    
    def get_condition_tags(self) -> list[str]:
        """Retrieves list of active condition tags.

        Returns:
            list[str]: Tags registered in the Conditioner.
        """
        return self.tags

    def add_condition_item(self, tag: str, item):
        """Registers a new conditioner item to the Conditioner.

        Args:
            tag (str): Identifier for the condition (e.g., "holidays").
            item (Any): An instantiated conditioner object that implements 
                `get_conditions` and has a `condition_dim` attribute. (e.g. HolidayConditioner())
        """
        # Add the tag to the list of tags
        self.tags.append(tag)
        # Add the item to the dictionary
        self.condition_items[tag] = item
        # Add the condition dimension to the total condition dimension
        self.condition_dim += item.condition_dim

    def get_conditions(self, x: torch.Tensor | np.ndarray | None = None, 
                       meter_id: torch.Tensor | np.ndarray | None = None, 
                       timestamps: torch.Tensor | np.ndarray | None = None): 
        """Gets and concatenates conditions from all registered conditioners.

        Args:
            x (torch.Tensor | np.ndarray): Input data. Used to infer target device.
            meter_id (torch.Tensor | np.ndarray, optional): Meter IDs.
            timestamps (torch.Tensor | list, optional): Timestamps.

        Returns:
            torch.Tensor | None: Concatenated condition tensor of shape 
                `(batch_size, total_condition_dim)`. Returns None if no conditions 
                are registered.
        """
        conditions = []
        target_device = None
        if isinstance(x, torch.Tensor):
            target_device = x.device
        elif isinstance(x, np.ndarray):
            target_device = torch.device('cpu')
        
        for tag in self.tags:
            cond = self.condition_items[tag].get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond, dtype=torch.float32)
            if target_device is not None and cond.device != target_device:
                cond = cond.to(target_device)
            conditions.append(cond)
        
        if len(conditions) == 0:
            return None
        
        # Concatenate all conditions along feature dimension
        return torch.cat(conditions, dim=-1)            
    
    def get_random_conditions(self, batch_size: int):
        """Generates random conditions concatenated from all active items for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Concatenated random conditions of shape 
                `(batch_size, total_condition_dim)`.
        """
        conditions = []
        for tag in self.tags:
            item = self.condition_items[tag]
            conditions.append(item.get_random_conditions(batch_size))
        return torch.cat(conditions, dim=-1)

    def get_condition_dim(self):
        """Returns the total aggregated condition dimension.
        
        Returns:
            int: Size of condition vector
        """
        return int(self.condition_dim)

    def get_condition_indexes(self, tag: str):
        """Calculates the feature indices for a specific condition in the concatenated tensor.

        Args:
            tag (str): The condition tag to locate.

        Returns:
            tuple: Start and end indices `(start_idx, end_idx)`.

        Raises:
            ValueError: If the tag is not found in the registered tags.
        """
        if tag not in self.tags:
            raise ValueError(f"Condition tag '{tag}' not found in Conditioner.")
        idx = 0
        for t in self.tags:
            if t == tag:
                return (idx, idx + self.condition_items[t].condition_dim)
            idx += self.condition_items[t].condition_dim
        raise ValueError(f"Condition tag '{tag}' not found in Conditioner.")

    def save(self, path: str):
        """Saves the entire Conditioner state to a pickle file.

        Args:
            path (str): Filepath to save the model.
        """
        self.path = path
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """Loads a Conditioner instance from a pickle file.

        Args:
            path (str): Filepath to load the model from.

        Returns:
            Conditioner: The unpickled Conditioner instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
