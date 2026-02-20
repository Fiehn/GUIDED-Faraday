import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# =============
# LDA conditions and user profiles
# =============
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm
import os

class UserProfiler(LatentDirichletAllocation):
    """
    Uses LDA to create user profiles from time series data
    Each user's time series data is clustered into "words" using KMeans
    Then LDA is applied to get topic distributions as user profiles
    Parameters:
    - n_topics: Number of topics for LDA
    - num_clusters: Number of clusters for KMeans to create "words"
    - random_state: Random seed for reproducibility
    """
    def __init__(self, n_topics = 10, num_clusters = 100, random_state=None, path='models/UserProfiles/user_profiles.pkl', **kwargs):
        super(UserProfiler, self).__init__(n_components=n_topics, random_state=random_state)
        self.n_topics = n_topics
        self.num_clusters = num_clusters
        self.condition_dim = n_topics

        self.lambda_matrix = None  # To hold the learned topic-word distributions
        self.cluster_centers_ = None  # To hold KMeans cluster centers

        self.user_profiles_cache = {}  # Cache for user profiles {meter_id: sampled_theta}
        self.gamma_cache = {}  # Cache for concentration parameters {meter_id: gamma_u}

        self.random_state = random_state
        self.model_kwargs = locals()
        self.path = path
        self.verbose = kwargs.get('verbose', False)

    def fit(self, train_loader):
        """
        Fit UserProfiler using batches from a PyTorch DataLoader.
        Will use streaming/incremental learning to handle the size of the data.

        train_loader: PyTorch DataLoader yielding batches of shape [B, T, F]
        """
        # TODO: they scale the data here?? or before??
        # TODO: reduce dimensions (svd or pca) before clustering??

        if self.verbose:
            print("Fitting UserProfiler model...")
        
        # === Phase 1: Incremental clustering to create "words" ===
        if self.verbose:
            print("Phase 1: Clustering to create 'words'...")

        cluster_model = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            batch_size=1024,
            n_init='auto'
        )
        batch_count = 0
        total_timesteps = 0

        for x, meter_id, timestamps in tqdm(train_loader, desc="Clustering batches"):
            batch_size, seq_len, num_features = x.shape
            x_daily_patterns = x.cpu().numpy().reshape(batch_size, seq_len * num_features)  # [B, L*F]

            # Remove missing data
            missing_mask = np.isnan(x_daily_patterns).any(axis=1)  # [B]
            x_fit = x_daily_patterns[~missing_mask]  # [valid_samples, L*F]

            if len(x_fit) == 0:
                continue  # Skip batches with all missing data
            cluster_model.partial_fit(x_fit)
            total_timesteps += x_fit.shape[0]
            batch_count += 1
        
        self.cluster_centers_ = cluster_model.cluster_centers_

        if self.verbose:
            print(f"Clustering complete over {batch_count} batches with {total_timesteps} timesteps.")
            print("Phase 2: Creating corpus and fitting LDA model...")

        # === Phase 2: Create corpus and fit LDA model ===
        meter_corpus = {} # dict: meter_id -> cluster counts

        for x, meter_id, timestamps in tqdm(train_loader, desc="Creating corpus batches"):
            batch_size, seq_len, num_features = x.shape
            x_batch = x.cpu().numpy().reshape(batch_size, seq_len * num_features)  # [B*T, F]
            meter_id_flat = meter_id.cpu().numpy() 

            # Remove missing data
            missing_idx = np.isnan(x_batch).any(axis=1)
            x_fit = x_batch[~missing_idx]
            meter_id_fit = meter_id_flat[~missing_idx]
            if len(x_fit) == 0:
                continue  # Skip batches with all missing data

            # Assign clusters
            labels = cdist(x_fit, self.cluster_centers_).argmin(axis=1)

            # count clusters per meter
            for m_id, label in zip(meter_id_fit, labels):
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters,dtype=np.int32)
                meter_corpus[m_id][label] += 1

        # Convert to document term matrix
        meter_ids_sorted = sorted(meter_corpus.keys())
        num_entities = len(meter_ids_sorted)
        X_corpus = np.array([meter_corpus[m_id] for m_id in meter_ids_sorted])

        if self.verbose:
            print(f"Corpus shape: {X_corpus.shape} (num_entities: {num_entities}, num_clusters: {self.num_clusters})")
            print("Phase 3: Fitting LDA model...")
        
        # === Phase 3: Fit LDA model ===
        self.doc_lengths = X_corpus.sum(axis=1)

        if hasattr(self, 'fit_kwargs'):
            for key, value in self.fit_kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        super(UserProfiler, self).fit(X_corpus)
        self.lambda_matrix = self.components_
        self.is_fitted = True

        # === Phase 4: Cache gamma parameters for all users ===
        if self.verbose:
            print("Phase 4: Caching gamma parameters for all users...")

        self.meter_id_to_idx = {m_id: idx for idx, m_id in enumerate(meter_ids_sorted)}
        all_gamma = super(UserProfiler, self).transform(X_corpus, normalize=False)

        # Cache gamma values and smaple profiles
        for m_id, gamma_u in zip(meter_ids_sorted, all_gamma):
            self.gamma_cache[m_id] = gamma_u
            self.user_profiles_cache[m_id] = np.random.dirichlet(gamma_u)

        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.save(self.path)
            if self.verbose:
                print(f"UserProfiler model saved to {self.path}.")
        return self

    def get_conditions(self, x, meter_id=None, timestamps=None):
        """
        Get conditions for a batch during inference.
        Looks up cached profiles by meter_id. If not found, computes on-the-fly.

        Args:
            x: [B, L, F] tensor of time series data (unused if profiles cached)
            meter_id: [B] tensor of meter IDs
            timestamps: [B] timestamps (unused for user profiles)

        Returns:
            [B, n_topics] topic distributions
        """
        if self.cluster_centers_ is None or not hasattr(self, 'is_fitted') or not self.is_fitted:
            try:
                self.load(self.path)
            except Exception as e:
                raise ValueError("UserProfiler model is not fitted or cluster centers missing.") from e
            
         # Validate cluster_centers_ shape
        if self.cluster_centers_.ndim != 2:
            raise ValueError(f"cluster_centers_ must be 2D, got shape {self.cluster_centers_.shape}")
        
        if meter_id is None:
            raise ValueError("meter_id is required for UserProfiler.get_conditions()")

        # Convert single batch to corpus
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()
        
        # === Cached Gamma Parameters Retrieval ===
        missing_meters = [m for m in meter_id if m not in self.gamma_cache]

        if missing_meters:
            self._compute_profiles_for_batch(x, meter_id, missing_meters)
        
        # === Cached Sample Profile Retrieval ===
        profiles = np.zeros((len(meter_id), self.n_topics), dtype=np.float32)
        missing_meters = [m for m in meter_id if m not in self.user_profiles_cache]

        # If some meters are missing, compute them on-the-fly (by sampling from cached gamma params)
        if missing_meters:
            for m_id in missing_meters:
                self.resample_user_vector(meter_id=m_id)
        
        # === Assemble profiles ===
        for i, m_id in enumerate(meter_id):
            profiles[i] = self.user_profiles_cache[m_id]
        
        return profiles

    def _compute_profiles_for_batch(self, x, meter_id, target_meter_ids):
        """
        Compute profiles for specific meters from current batch.
        Only used when a meter is not in cache.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(meter_id, torch.Tensor):
            meter_id = meter_id.cpu().numpy()

        # Handle both 2D [B, D] and 3D [B, L, F] inputs
        if x.ndim == 3:
            batch_size, seq_len, num_features = x.shape
            # Flatten
            x_batch = x.reshape(batch_size, seq_len * num_features)
        elif x.ndim == 2:
            batch_size = x.shape[0]
            x_batch = x
        else:
            raise ValueError(f"Expected x to have 2 or 3 dimensions, got {x.ndim}")
        
        # Remove missing
        missing_mask = np.isnan(x_batch).any(axis=1)
        x_fit = x_batch[~missing_mask]
        meter_id_valid = meter_id[~missing_mask]
        
        if x_fit.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_valid has {x_fit.shape[1]} features, "
                f"but cluster_centers_ has {self.cluster_centers_.shape[1]} features"
            )
        
        # Select rows for target_meter_ids
        selected_indices = [i for i, m_id in enumerate(meter_id_valid) if
                            m_id in target_meter_ids]
        x_selected = x_fit[selected_indices]
        meter_id_selected = meter_id_valid[selected_indices]

        # Cluster into "words"
        labels = cdist(x_selected, self.cluster_centers_).argmin(axis=1)
        
        # Create document-term matrix for target meters
        meter_corpus = {}
        for m_id, label in zip(meter_id_selected, labels):
            if m_id not in meter_corpus:
                meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
            meter_corpus[m_id][label] += 1
        
        X_corpus = np.array([meter_corpus[m_id] for m_id in target_meter_ids], dtype=np.int32)
        
        # Get posterior parameters (gamma) from LDA
        # _unnormalized_transform returns the unnormalized Dirichlet parameters
        posterior = super(UserProfiler, self).transform(X_corpus,normalize=False)
        
        # Cache gamma values (these are the Dirichlet concentration parameters)
        for m_id, gamma_u in zip(target_meter_ids, posterior):
            self.gamma_cache[m_id] = gamma_u
            # TODO: make streaming compatible
        return posterior
    
    def resample_user_vector(self, meter_id=None):
        """Sample new user profiles from stored gamma parameters."""
        if not self.gamma_cache:
            raise ValueError("No gamma parameters cached. Cannot resample user vectors.")
        if meter_id is None:
            # Resample for all cached meters
            for m_id, gamma_u in self.gamma_cache.items():
                sampled_theta = np.random.dirichlet(gamma_u)
                self.user_profiles_cache[m_id] = sampled_theta
        else:
            if meter_id not in self.gamma_cache:
                raise ValueError(f"Meter ID {meter_id} not found in gamma cache.")
            gamma_u = self.gamma_cache[meter_id]
            sampled_theta = np.random.dirichlet(gamma_u)
            self.user_profiles_cache[meter_id] = sampled_theta

    def get_random_conditions(self, batch_size):
        """Generate random user profiles for testing."""
        random_profiles = np.random.dirichlet(alpha=np.ones(self.n_topics), size=batch_size)
        return torch.tensor(random_profiles,dtype=torch.float32)

    def save(self, path):
        """Save model including the cached profiles."""
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
        if self.verbose:
            print(f"Saved model with {len(self.user_profiles_cache)} cached profiles")
            print(f"Cluster centers shape: {self.cluster_centers_.shape}")
    
    @staticmethod
    def load(path):
        """Load model with cached profiles."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
            
        
        if hasattr(obj, 'verbose') and obj.verbose:
            print(f"Loaded model with {len(obj.user_profiles_cache)} cached profiles")
            print(f"Cluster centers shape: {obj.cluster_centers_.shape}")
        
        return obj
    def load(self,path):
        """Load model with cached profiles."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Loaded model with {len(self.user_profiles_cache)} cached profiles")
                print(f"Cluster centers shape: {self.cluster_centers_.shape}")


class OnlineUserProfiler(UserProfiler):
    """
    Online version of UserProfiler that updates profiles incrementally during training.
    """
    def __init__(self, decay=0.9, update_frequency=10, **kwargs):
        super().__init__(**kwargs)
        self.decay = decay  # Exponential forgetting factor (0 < decay <= 1)
        self.update_frequency = update_frequency  # Update model every N batches
        self.batch_counter = 0
        self.last_update_batch = 0
        
        # Initialize MiniBatchKMeans for incremental clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            batch_size=1024,
            n_init='auto'
        )
        self.kmeans_initialized = False
        
        # Track user activity for decay
        self.user_last_seen = {}  # meter_id -> batch_number
        
    def get_conditions(self, x, meter_id=None, timestamps=None):
        """
        Get conditions for a batch, with automatic online updating.
        This maintains compatibility with the Conditioner interface.
        """
        if meter_id is None:
            raise ValueError("meter_id is required for OnlineUserProfiler.get_conditions()")
        
        # Update model if needed
        self._maybe_update_model(x, meter_id)
        
        # Get profiles using parent class logic but with online updates
        return self._get_profiles_with_decay(x, meter_id, timestamps)
    
    def _maybe_update_model(self, x, meter_ids):
        """Update the model if enough batches have passed."""
        self.batch_counter += 1
        
        if (self.batch_counter - self.last_update_batch) >= self.update_frequency:
            self.update_model_incremental(x, meter_ids)
            self.last_update_batch = self.batch_counter
    
    def update_model_incremental(self, x_batch, meter_ids):
        """
        Incrementally update the clustering and topic model with new data.
        This separates the update logic from profile retrieval.
        """
        # Convert inputs
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.cpu().numpy()
        if isinstance(meter_ids, torch.Tensor):
            meter_ids = meter_ids.cpu().numpy()
        
        # Handle both 2D [B, D] and 3D [B, L, F] inputs
        if x_batch.ndim == 3:
            batch_size, seq_len, num_features = x_batch.shape
            x_flat = x_batch.reshape(batch_size, seq_len * num_features)
        elif x_batch.ndim == 2:
            x_flat = x_batch
        else:
            raise ValueError(f"Expected x to have 2 or 3 dimensions, got {x_batch.ndim}")
        
        # Remove invalid data
        valid_mask = ~np.isnan(x_flat).any(axis=1)
        x_valid = x_flat[valid_mask]
        meter_ids_valid = meter_ids[valid_mask]
        
        if len(x_valid) == 0:
            return  # No valid data to update with
        
        # Update cluster centers incrementally
        self.kmeans.partial_fit(x_valid)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True
        
        # Create corpus for this batch
        if self.cluster_centers_ is not None:
            labels = cdist(x_valid, self.cluster_centers_).argmin(axis=1)
            
            # Count clusters per user
            meter_corpus = {}
            for m_id, label in zip(meter_ids_valid, labels):
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
                meter_corpus[m_id][label] += 1
            
            # Update topic model if we have LDA fitted
            if hasattr(self, 'is_fitted') and self.is_fitted:
                # Convert to document-term matrix
                unique_meters = sorted(meter_corpus.keys())
                if len(unique_meters) > 0:
                    X_corpus = np.array([meter_corpus[m_id] for m_id in unique_meters])
                    
                    # Update LDA incrementally
                    super(UserProfiler, self).partial_fit(X_corpus)
                    
                    # Update cached gamma parameters with decay
                    new_gamma = super(UserProfiler, self).transform(X_corpus, normalize=False)
                    for m_id, gamma_new in zip(unique_meters, new_gamma):
                        if m_id in self.gamma_cache:
                            # Apply exponential decay to old gamma values
                            gamma_old = self.gamma_cache[m_id]
                            self.gamma_cache[m_id] = self.decay * gamma_old + (1 - self.decay) * gamma_new
                        else:
                            # New user
                            self.gamma_cache[m_id] = gamma_new
                        
                        # Update user activity tracking
                        self.user_last_seen[m_id] = self.batch_counter
            
            # If LDA not fitted yet, need initial fit
            elif len(meter_corpus) > 0:
                unique_meters = sorted(meter_corpus.keys())
                X_corpus = np.array([meter_corpus[m_id] for m_id in unique_meters])
                
                # Initial fit
                super(UserProfiler, self).fit(X_corpus)
                self.is_fitted = True
                
                # Initialize caches
                initial_gamma = super(UserProfiler, self).transform(X_corpus, normalize=False)
                for m_id, gamma_u in zip(unique_meters, initial_gamma):
                    self.gamma_cache[m_id] = gamma_u
                    self.user_last_seen[m_id] = self.batch_counter
    
    def _get_profiles_with_decay(self, x, meter_ids, timestamps=None):
        """
        Get user profiles with decay applied to inactive users.
        """
        if isinstance(meter_ids, torch.Tensor):
            meter_ids = meter_ids.cpu().numpy()
        
        profiles = np.zeros((len(meter_ids), self.n_topics), dtype=np.float32)
        
        # Apply decay to users based on when they were last seen
        for m_id in set(meter_ids):
            if m_id in self.gamma_cache:
                # Calculate decay factor based on inactivity
                batches_since_seen = self.batch_counter - self.user_last_seen.get(m_id, self.batch_counter)
                decay_factor = self.decay ** batches_since_seen
                
                # Apply decay to gamma parameters
                if decay_factor < 0.1:  # If too much decay, reset to prior
                    self.gamma_cache[m_id] = np.ones(self.n_topics)
                else:
                    self.gamma_cache[m_id] *= decay_factor
                    self.gamma_cache[m_id] += (1 - decay_factor) * np.ones(self.n_topics)  # Add prior
                
                # Resample profile from updated gamma
                self.user_profiles_cache[m_id] = np.random.dirichlet(self.gamma_cache[m_id])
        
        # Handle missing profiles by computing them on-the-fly or using defaults
        missing_meters = [m for m in meter_ids if m not in self.user_profiles_cache]
        if missing_meters and hasattr(self, 'cluster_centers_') and self.cluster_centers_ is not None:
            self._compute_profiles_for_batch(x, meter_ids, missing_meters)
        
        # Assemble final profiles
        for i, m_id in enumerate(meter_ids):
            if m_id in self.user_profiles_cache:
                profiles[i] = self.user_profiles_cache[m_id]
            else:
                # Default to uniform distribution for unknown users (should not really happen if we compute on-the-fly)
                profiles[i] = np.ones(self.n_topics) / self.n_topics
        
        return torch.tensor(profiles, dtype=torch.float32)
    
    def update_and_get_profile(self, x_batch, meter_ids):
        """
        Legacy method for backward compatibility.
        Updates model and returns profiles in one call.
        """
        self.update_model_incremental(x_batch, meter_ids)
        return self._get_profiles_with_decay(x_batch, meter_ids)

            
# === Day of week conditioning ===
class DayOfWeekConditioner():
    def __init__(self):
        self.condition_dim = 7

    def get_conditions(self,x, meter_id, timestamps):
        """
        timestamps: tensor of Unix timestamps (float) or array-like of pandas Timestamps
        returns: [B, 7] one-hot encoded day of week
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
    
    def get_random_conditions(self, batch_size):
        random_days = torch.randint(0,7,(batch_size,))
        dow_onehot = F.one_hot(random_days, num_classes=7).float()
        return dow_onehot

class MonthOfYearConditioner():
    def __init__(self):
        self.condition_dim = 12

    def get_conditions(self,x, meter_id, timestamps):
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
    
    def get_random_conditions(self, batch_size):
        random_months = torch.randint(0,12,(batch_size,))
        moy_onehot = F.one_hot(random_months, num_classes=12).float()
        return moy_onehot

class HolidayConditioner():
    def __init__(self, holiday_path='data/holidays/denmark_holidays.json'):
        self.condition_dim = 1
        with open(holiday_path, 'r') as f:
            holidays = json.load(f)
        self.holiday_dates = set(pd.to_datetime(holidays['date']).date)
    def get_conditions(self,x, meter_id, timestamps):
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
    
    def get_random_conditions(self, batch_size):
        random_holiday = torch.randint(0,2,(batch_size,1)).float()
        return random_holiday

class SchoolHolidayConditioner():
    def __init__(self):
        self.condition_dim = 1
        # Define school holiday periods (example for Denmark)
        # Weeks 7, 27-32, 43
        
    def get_conditions(self,x, meter_id, timestamps):
        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            dates = pd.to_datetime(timestamps.cpu().numpy(), unit='s')
            is_school_holiday = torch.tensor([1.0 if (d.weekofyear in [7,27,28,29,30,31,32,43]) else 0.0 for d in dates], device=device).unsqueeze(1)
        else:
            is_school_holiday = []
            for ts in timestamps:
                if isinstance(ts, (int, float)):
                    ts = pd.Timestamp(ts, unit='s')
                is_school_holiday.append(1.0 if (ts.weekofyear in [7,27,28,29,30,31,32,43]) else 0.0)
            is_school_holiday = torch.tensor(is_school_holiday, dtype=torch.float32).unsqueeze(1)
        return is_school_holiday
    
    def get_random_conditions(self, batch_size):
        random_school_holiday = torch.randint(0,2,(batch_size,1)).float()
        return random_school_holiday

# =============
# Conditioner Module - Holds an object that creates and determines type of conditioning to be used in Models
# ============

class Conditioner():
    def __init__(self, tags = None, **kwargs):
        # Tags is the name of the condition, e.g. "user_profiles", "days", "weeks" and so on
        self.tags = tags if tags is not None else []
        # Dimention is important for the model to know how much conditioning to expect
        self.condition_dim = 0
        # Dictionary to hold the conditioning items
        self.condition_items = {}
    
    def get_condition_tags(self):
        return self.tags

    def add_condition_item(self, tag, item):
        # Add the tag to the list of tags
        self.tags.append(tag)
        # Add the item to the dictionary
        self.condition_items[tag] = item
        # Add the condition dimension to the total condition dimension
        self.condition_dim += item.condition_dim

    def get_conditions(self, x, meter_id=None, timestamps=None): 
        """
        Get conditions from all registered conditioners.
        Returns concatenated tensor of all conditions.
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
    
    def get_random_conditions(self, batch_size):
        """
        Generate random conditions for testing.
        """
        conditions = []
        for tag in self.tags:
            item = self.condition_items[tag]
            conditions.append(item.get_random_conditions(batch_size))
        return torch.cat(conditions, dim=-1)

    def get_condition_dim(self):
        return int(self.condition_dim)

    def save(self, path):
        self.path = path
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
