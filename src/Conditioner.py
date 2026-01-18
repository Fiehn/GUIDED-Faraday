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
        self.num_topics = n_topics
        self.num_clusters = num_clusters

        self.random_state = random_state
        self.model_kwargs = locals()
    
    def create_corpus(self, X, num_entities, missing_idx, num_clusters, cluster_centers):
        # Will cluster an entity's features into num_clusters clusters
        # Then One-hot encode the cluster assignments for each entity
        # Finally sum over entities to get document-term matrix

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
        X_missing = X.reshpae(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_fit = X_missing[~missing_idx]

        # TODO: reduce dimensions (svd or pca) before clustering??

        # Cluster the entities to create "words"
        cluster_model = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=self.random_state) # TODO: change these hyperparameters
        cluster_model.fit(X_fit)
        self.cluster_centers_ = cluster_model.cluster_centers_

        # Create corpus for LDA
        X_corpus = self.create_corpus(X_fit, num_entities, missing_idx, self.num_clusters, self.cluster_centers_)

        # Fit LDA model
        self.doc_lengths = X_corpus.sum(axis=1)
        for key, value in self.fit_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super(UserProfiler, self).fit(X_corpus)
        self.lambda_matrix = self.components_
        return self
    
    def transform(self, X):
        num_entities, _, num_features = X.shape
        X_missing = X.reshape(-1, num_features)
        missing_idx = np.isnan(X_missing.sum(1))
        X_fit = X_missing[~missing_idx]

        # TODO: reduce dimensions (svd or pca) before clustering??

        X_corpus = self.create_corpus(X_fit, num_entities, missing_idx, self.num_clusters, self.cluster_centers_)
        topic_distributions = super(UserProfiler, self)._unnormalized_transform(X_corpus)
        return topic_distributions

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
    

# =============
# Conditioner Module - Holds an object that creates and determines type of conditioning to be used in Models
# ============

class Conditioner():
    def __init__(self, tags, type = "None", **kwargs):
        self.tags = tags
        self.type = type
        self.condition_dim = 0
        # TODO:
    
    def get_conditioning(self, inputs):
        pass #TODO:

