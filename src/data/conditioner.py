import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
 
# =============
# LDA conditions and user profiles
# =============
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm


class UserProfiler(LatentDirichletAllocation):
    """
    User profiler using LDA + KMeans for time series data.
    Each user's time series is clustered into "words", then LDA produces
    topic distributions as user profiles. Supports both batch fitting and
    incremental online updates with exponential decay for inactive users.

    Parameters:
        n_topics: Number of LDA topics
        num_clusters: Number of KMeans clusters (vocabulary size)
        decay: Exponential forgetting factor for inactive users (0 < decay <= 1)
        update_frequency: Trigger incremental update every N get_conditions calls
        random_state: Random seed
        path: Path to save/load the model
    """
    def __init__(self, n_topics = 10, num_clusters = 100, 
                 random_state=None, 
                 path='models/UserProfiles/user_profiles.pkl', 
                 decay=0.9, update_frequency=10,
                 **kwargs):
        super(UserProfiler, self).__init__(n_components=n_topics, random_state=random_state)
        self.n_topics = n_topics
        self.num_clusters = num_clusters
        self.condition_dim = n_topics
        self.decay = decay
        self.update_frequency = update_frequency

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
        Fit UserProfiler using batches from a PyTorch DataLoader.
        Will use streaming/incremental learning to handle the size of the data.

        train_loader: PyTorch DataLoader yielding batches of shape [B, T, F]
        """
        # TODO: they scale the data here?? or before??
        # TODO: reduce dimensions (svd or pca) before clustering??

        if self.verbose:
            print("Fitting UserProfiler model")
            print("Phase 1: Clustering to create 'words")
        
        # === Phase 1: Incremental clustering to create "words" ===
        batch_count = 0
        total_timesteps = 0

        for data, meter_id, timestamps in tqdm(train_loader, desc="Clustering batches"):
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

        for data, meter_id, timestamps in tqdm(train_loader, desc="Creating corpus batches"):
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
        """
        Get conditions for a batch during inference. Triggers incremental model updates
        Looks up cached profiles by meter_id. If not found, computes on-the-fly.

        Args:
            data: [B, L, F] tensor of time series data (unused if profiles cached)
            meter_id: [B] tensor of meter IDs
            timestamps: [B] timestamps (unused for user profiles)

        Returns:
            [B, n_topics] topic distributions
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

        if data is not None:
            self._maybe_update_model(data, meter_id)

        return self._get_cached_profiles(data, meter_id)
    
    def _maybe_update_model(self, data, meter_ids):
        """Trigger an incremental model update if enough batches have passed."""
        if self.update_frequency <= 0:
            return
        if (self.batch_counter - self.last_update_batch) >= self.update_frequency:
            self.update_model_incremental(data, meter_ids)
            self.last_update_batch = self.batch_counter

    def update_model_incremental(self, x_batch, meter_ids):
        """
        Incrementally update the KMeans clustering and LDA topic model with new data.
        New users are added to the cache; existing users are updated with exponential decay.
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

        self.kmeans.partial_fit(x_valid)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.kmeans_initialized = True

        labels = cdist(x_valid, self.cluster_centers_).argmin(axis=1)
        meter_corpus = {}
        for m_id, label in zip(meter_ids_valid, labels):
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
        """
        Assemble profile vectors from cache. Computes on-the-fly for unknown meters.
        Falls back to uniform for meters with no data available.
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
        """
        Compute and cache gamma + profile for specific meters from the current batch.
        Only called for meters not present in the cache.
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
            raise ValueError(
                f"Feature dimension mismatch: x has {x_fit.shape[1]} features, "
                f"but cluster_centers_ has {self.cluster_centers_.shape[1]} features"
            )

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

    def resample_user_vector(self, meter_id=None):
        """Sample new user profiles from stored gamma parameters."""
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
        """Generate random user profiles for testing."""
        random_profiles = np.random.dirichlet(alpha=np.ones(self.n_topics), size=batch_size)
        return torch.tensor(random_profiles, dtype=torch.float32)

    def save(self, path):
        """Save model to pickle."""
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if self.verbose:
            print(f"Saved model with {len(self.user_profiles_cache)} cached profiles")

    def load(self, path):
        """Load model from pickle and update instance state."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        if self.verbose:
            print(f"Loaded model with {len(self.user_profiles_cache)} cached profiles")

            
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
    # https://helligdage-dk.dk/helligdage-2024/
    def __init__(self, holiday_path='data/holidays/denmark_holidays.json'):
        self.condition_dim = 1
        with open(holiday_path, 'r') as f:
            holidays = json.load(f)
        self.holiday_dates = set(pd.to_datetime(holidays['dates']).date)
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
    
class HeatpumpConditioner():
    def __init__(self, topology_path = '/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        self.topology_df = pd.read_csv(topology_path)
    
    def get_conditions(self, x, meter_id, timestamps):
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
    
    def get_random_conditions(self, batch_size):
        random_status = torch.randint(0,2,(batch_size,1)).float()
        return random_status
    
class SolarPanelConditioner():
    def __init__(self, topology_path = '/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        self.topology_df = pd.read_csv(topology_path)
    
    def get_conditions(self, x, meter_id, timestamps):
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
    
    def get_random_conditions(self, batch_size):
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
