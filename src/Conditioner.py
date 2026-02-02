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
    def __init__(self, n_topics = 10, num_clusters = 100, random_state=None, **kwargs):
        super(UserProfiler, self).__init__(n_components=n_topics, random_state=random_state)
        self.n_topics = n_topics
        self.num_clusters = num_clusters

        self.random_state = random_state
        self.model_kwargs = locals()
        self.path = None
        self.verbose = kwargs.get('verbose', False)
    
    def create_corpus(self, X, num_entities, missing_idx, num_clusters, cluster_centers):
        # Will cluster an entity's features into num_clusters clusters
        # Then One-hot encode the cluster assignments for each entity
        # Finally sum over entities to get document-term matrix

        if self.verbose:
            print(f"Creating corpus for {num_entities} entities...")

        # TODO: This is bug filled
        # First cluster the entities
        labels = cdist(X, cluster_centers).argmin(axis=1)
        # One-hot encode the cluster labels
        labels_onehot = np.zeros((num_entities, num_clusters))
        labels_onehot[np.arange(num_entities), labels] = 1

        # Now create the corpus with missing entities removed
        X_words = np.zeros((missing_idx.size, num_clusters))
        # Fill in only the non-missing entities
        X_words[~missing_idx] = labels_onehot
        # Then sum over entities to get document-term matrix
        X_words = X_words.reshape(num_entities, -1, num_clusters).sum(axis=1)
        return X_words

    def fit(self, X, fit_kwargs):
        self.fit_kwargs = fit_kwargs
        # Get dimensions from the times series data
        num_entities, _, num_features = X.shape 
        # TODO: they scale the data here?? or before??

        # Find missing entities and create a mask
        X_missing = X.reshape(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_fit = X_missing[~missing_idx]

        # TODO: reduce dimensions (svd or pca) before clustering??

        # Cluster the entities to create "words"
        if self.verbose:
            print(f"Clustering {X_fit.shape[0]} entities into {self.num_clusters} clusters...")
        
        cluster_model = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=self.random_state, verbose=self.verbose) # TODO: change these hyperparameters
        cluster_model.fit(X_fit)
        
        if self.verbose:
            print("Clustering complete.")

        self.cluster_centers_ = cluster_model.cluster_centers_

        # Create corpus for LDA
        X_corpus = self.create_corpus(X_fit, num_entities, missing_idx, self.num_clusters, self.cluster_centers_)

        # Fit LDA model
        self.doc_lengths = X_corpus.sum(axis=1)
        for key, value in self.fit_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        if self.verbose:
            print("Fitting LDA model...")
            
        super(UserProfiler, self).fit(X_corpus)
        self.lambda_matrix = self.components_
        if self.path is not None:
            self.save(self.path)
        return self
    
    def transform(self, X):
        num_entities, _, num_features = X.shape
        X_missing = X.reshape(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_fit = X_missing[~missing_idx]


        # TODO: reduce dimensions (svd or pca) before clustering??

        if self.verbose:
            print("Transforming input data to topic distributions...")

        X_corpus = self.create_corpus(X_fit, num_entities, missing_idx, self.num_clusters, self.cluster_centers_)
        topic_distributions = super(UserProfiler, self)._unnormalized_transform(X_corpus)
        return topic_distributions
    
    def get_user_profiles(self, X, retrain=False):
        # If this model has not been trained and a model already exists on disk then load it and return the user profiles for the input data
        if os.path.exists(self.path) and not retrain and self.lambda_matrix is not None:
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

    def get_conditions(self, X):
        return self.get_user_profiles(X, retrain=False)

    def save(self, path):
        self.path = path
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    
# === Day of week conditioning ===
class DayOfWeekConditioner():
    def __init__(self):
        self.condition_dim = 7

    def get_conditions(self, timestamps):
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


# =============
# Conditioner Module - Holds an object that creates and determines type of conditioning to be used in Models
# ============

class Conditioner():
    def __init__(self, tags = [], **kwargs):
        # Tags is the name of the condition, e.g. "user_profiles", "days", "weeks" and so on
        self.tags = tags
        # Dimention is important for the model to know how much conditioning to expect
        self.condition_dim = 0
        # Dictionary to hold the conditioning items
        self.condition_items = {}
    
    def add_condition_item(self, tag, item):
        # Add the tag to the list of tags
        self.tags.append(tag)
        # Add the item to the dictionary
        self.condition_items[tag] = item
        # Add the condition dimension to the total condition dimension
        self.condition_dim += item.condition_dim

    def get_conditions(self, inputs): 
        """
        Get conditions from all registered conditioners.
        Returns concatenated tensor of all conditions.
        """
        conditions = []
        for tag in self.tags:
            cond = self.condition_items[tag].get_conditions(inputs)
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond, dtype=torch.float32)
            conditions.append(cond)
        
        if len(conditions) == 0:
            return None
        
        # Concatenate all conditions along feature dimension
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
