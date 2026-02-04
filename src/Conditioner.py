import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        self.user_profiles_cache = {}  # Cache for user profiles {meter_id: topic_distribution}

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

        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.save(self.path)
            if self.verbose:
                print(f"UserProfiler model saved to {self.path}.")
        return self
        
    
    def transform(self, train_loader):
        """
        Transform data from DataLoader to topic distributions.
        Returns topic distributions for each meter_id and caches them.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() before transform().")
        if self.verbose:
            print("Transforming data to user profiles...")

        meter_corpus = {}
        for x, meter_id, timestamps in tqdm(train_loader, desc="Transforming", disable=not self.verbose):
            batch_size, seq_len, num_features = x.shape
            x_batch = x.cpu().numpy().reshape(batch_size, seq_len * num_features)
            meter_ids = meter_id.cpu().numpy()

            # Remove missing data
            missing_mask = np.isnan(x_batch).any(axis=1)
            x_fit = x_batch[~missing_mask]
            meter_ids_valid = meter_ids[~missing_mask]

            if len(x_fit) == 0:
                continue

            # Assign clusters
            labels = cdist(x_fit, self.cluster_centers_).argmin(axis=1)
            
            for m_id, label in zip(meter_ids_valid, labels):
                if m_id not in meter_corpus:
                    meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.int32)
                meter_corpus[m_id][label] += 1
        
        # Convert to matrix
        meter_ids_sorted = sorted(meter_corpus.keys())
        X_corpus = np.array([meter_corpus[m_id] for m_id in meter_ids_sorted])

        # Transform to topic distributions
        topic_distributions = super(UserProfiler, self)._unnormalized_transform(X_corpus)

        # Cache the profiles by meter_id
        for m_id, profile in zip(meter_ids_sorted, topic_distributions):
            self.user_profiles_cache[m_id] = profile
        
        if self.verbose:
            print(f"Cached profiles for {len(self.user_profiles_cache)} meters")

        return topic_distributions, meter_ids_sorted

    def get_user_profiles(self, X, retrain=False):
        """ Get user profiles from DataLoader X"""
        if os.path.exists(self.path) and not retrain and self.lambda_matrix is None:
            with open(self.path, "rb") as f:
                obj = pickle.load(f)
            return obj.transform(X)
        elif retrain or self.lambda_matrix is None:
            self.fit(X, self.fit_kwargs)
            return self.transform(X)
        else:
            # Lambda matrix is already computed and stored in self.lambda_matrix
            # We can use it to transform the input data without retraining or loading from disk
            return self.transform(X)

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
                if self.verbose:
                    print(f"Loading model from {self.path}")
                loaded_obj = self.load(self.path)
                
                # Copy ALL attributes from loaded object
                self.__dict__.update(loaded_obj.__dict__)
                
                if self.verbose:
                    print(f"Loaded model from {self.path}")
                    print(f"Cluster centers shape: {self.cluster_centers_.shape}")
                    print(f"Cached profiles: {len(self.user_profiles_cache)}")
                    print(f"Has exp_dirichlet_component_: {hasattr(self, 'exp_dirichlet_component_')}")
            except Exception as e:
                raise ValueError(f"Model must be fitted before getting conditions. Error loading: {e}")
            
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

        batch_size = len(meter_id)
        # Look up profiles from cache
        profiles = np.zeros((batch_size, self.n_topics))
        missing_meters = []

        for i, m_id in enumerate(meter_id):
            if m_id in self.user_profiles_cache:
                profiles[i] = self.user_profiles_cache[m_id]
            else:
                missing_meters.append(m_id)
        # If some meters are missing, compute them on-the-fly
        if missing_meters:
            if self.verbose:
                print(f"Warning: {len(missing_meters)} meters not in cache, computing profiles...")
            
            # Compute profiles for missing meters using current batch data
            profiles_computed = self._compute_profiles_for_batch(x, meter_id, missing_meters)
            
            # Fill in the missing profiles and cache them
            for m_id, profile in zip(missing_meters, profiles_computed):
                idx = np.where(meter_id == m_id)[0][0]
                profiles[idx] = profile
                self.user_profiles_cache[m_id] = profile  # Cache for future use
        
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

        # Validate inputs before cdist
        if self.verbose:
            print(f"X_valid shape: {x_fit.shape}")
            print(f"cluster_centers_ shape: {self.cluster_centers_.shape}")
        
        if x_fit.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_valid has {x_fit.shape[1]} features, "
                f"but cluster_centers_ has {self.cluster_centers_.shape[1]} features"
            )
        # Build corpus only for target meters
        meter_corpus = {}
        if len(x_fit) > 0:
            labels = cdist(x_fit, self.cluster_centers_).argmin(axis=1)
            
            for m_id, cluster_label in zip(meter_id_valid, labels):
                if m_id in target_meter_ids:
                    if m_id not in meter_corpus:
                        meter_corpus[m_id] = np.zeros(self.num_clusters, dtype=np.float64)
                    meter_corpus[m_id][cluster_label] += 1
        
        # Create corpus matrix for target meters only
        X_corpus = np.array([meter_corpus.get(m_id, np.zeros(self.num_clusters, dtype=np.float64))
                         for m_id in target_meter_ids], dtype=np.float64)
        
        # Transform to topics
        topic_distributions = super(UserProfiler, self)._unnormalized_transform(X_corpus)
        
        return topic_distributions
    
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
