import numpy as np
from scipy.spatial.distance import cdist

from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _sanitize_numeric_array
from datetime import datetime, timezone


class MaximumMeanDiscrepancy(SyntheticMetric):
    """Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures the distance between two distributions using a kernel-based approach.
    Uses RBF (Gaussian) kernel with automatic bandwidth selection via median heuristic.

        Implementation inspired by:
        - https://en.wikipedia.org/wiki/Maximum_mean_discrepancy
        - ??

    Math definition:
    MMD^2(P, Q) = E_{x,x' ~ P}[k(x,x')] + E_{y,y' ~ Q}[k(y,y')] - 2 E_{x ~ P, y ~ Q}[k(x,y)]
     where k is the RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
     and sigma is the kernel bandwidth, which can be set via median heuristic or provided by the user.
     Lower MMD means the synthetic distribution is closer to the real distribution, with 0 indicating identical distributions (under the kernel).
        Note: MMD is a distance metric, not a likelihood, so lower is better.

    Args:
        sigma (float): RBF kernel bandwidth. If None, uses median heuristic.
        seed (int): Random seed for reproducibility
        max_samples (int): Maximum number of samples to use for median heuristic (for efficiency)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, sigma: float | None = None, seed: int = 42, max_samples: int = 2000):
        self.sigma = sigma
        self.seed = seed
        self.max_samples = max_samples
    
    @property
    def name(self) -> str:
        return "Maximum Mean Discrepancy (MMD)"
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix between X and Y.
        
        Args:
            X (np.ndarray): array [n, d]
            Y (np.ndarray): array [m, d]
            sigma (float): kernel bandwidth
            
        Returns:
            np.ndarray: kernel matrix [n, m]
        """
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        sq_dists = np.nan_to_num(sq_dists, nan=1e12, posinf=1e12, neginf=0.0)
        denom = max(2.0 * (sigma ** 2), 1e-12)
        return np.exp(-sq_dists / denom)
    
    def _median_heuristic_sigma(self, Z: np.ndarray, max_points: int =  2000) -> float:
        """Estimate kernel bandwidth using median pairwise distance.
        
        Args:
            Z (np.ndarray): array [n, d] - combined data
            max_points (int): max points to use for efficiency
            
        Returns:
            float: estimated sigma value
        """
        
        rng = np.random.default_rng(self.seed)
        if Z.shape[0] > max_points:
            Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
        
        dists = cdist(Z, Z, metric="euclidean")
        med = np.median(dists[dists > 0])
        return float(med) if med > 0 else 1.0
    
    def _mmd_rbf_unbiased(self, X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
        """Compute MMD^2 statistic using RBF kernel.
        
        Args:
            X (np.ndarray): array [n, d] - real data
            Y (np.ndarray): array [m, d] - synthetic data
            sigma (float | None): kernel bandwidth (auto-estimate if None)
            
        Returns:
            float: MMD^2 statistic (lower is better)
        """
        X = _sanitize_numeric_array(X)
        Y = _sanitize_numeric_array(Y)
        n, m = X.shape[0], Y.shape[0]
        
        if n < 2 or m < 2:
            raise ValueError("Need at least 2 samples in each set.")
        
        if sigma is None:
            sigma = self._median_heuristic_sigma(np.vstack([X, Y]))
        
        # Compute kernel matrices
        Kxx = self._rbf_kernel(X, X, sigma)
        Kyy = self._rbf_kernel(Y, Y, sigma)
        Kxy = self._rbf_kernel(X, Y, sigma)
        
        # Zero out diagonal (unbiased estimator)
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        
        # Unbiased MMD^2
        mmd2 = (Kxx.sum() / (n * (n - 1))
              + Kyy.sum() / (m * (m - 1))
              - 2.0 * Kxy.sum() / (n * m))
        
        return float(mmd2)
    
    def compute(self) -> float:
        """Compute MMD metric.
        
        Returns:
            float: MMD value (lower is better, 0 = identical distributions)
        """
        rng = np.random.default_rng(self.seed)

        real = self.real_data
        synthetic = self.synthetic_data
        real = _sanitize_numeric_array(real)
        synthetic = _sanitize_numeric_array(synthetic)

        # Subsample to avoid OOM (O(n^2) kernel matrix)
        if len(real) > self.max_samples:
            idx = rng.choice(len(real), size=self.max_samples, replace=False)
            real = real[idx]
        if len(synthetic) > self.max_samples:
            idx = rng.choice(len(synthetic), size=self.max_samples, replace=False)
            synthetic = synthetic[idx]

        # Use provided sigma or auto-estimate
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([real, synthetic])
            sigma = self._median_heuristic_sigma(combined)
        
        # Compute MMD^2, return MMD 
        mmd2 = self._mmd_rbf_unbiased(real, synthetic, sigma=sigma)
        mmd_value = float(np.sqrt(np.maximum(mmd2, 0.0)))

        _save_standalone_metric_result(
            metric_key='MaximumMeanDiscrepancy',
            payload={
                'value': mmd_value,
                'mmd2': float(mmd2),
                'sigma': float(sigma),
                'seed': int(self.seed),
                'max_samples': int(self.max_samples),
                'n_real_used': int(len(real)),
                'n_synthetic_used': int(len(synthetic)),
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mmd_value


class SetMaximumMeanDiscrepancy(SyntheticMetric):

    """Permutation-Invariant MMD for 3-Phase Set Data."""
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, sigma: float | None = None, seed: int = 42, max_samples: int = 1000, n_features: int = 3):
        self.sigma = sigma
        self.seed = seed
        self.max_samples = max_samples
        self.n_features = n_features
        # The 6 possible permutations for 3 phases
        self.perms = [
            [0, 1, 2], [0, 2, 1], 
            [1, 0, 2], [1, 2, 0], 
            [2, 0, 1], [2, 1, 0]
        ]

    @property
    def name(self) -> str:
        return "Set-MMD (Permutation Invariant)"

    def _set_sqeuclidean_cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the minimum squared euclidean distance across all 6 permutations.
        X: [N, 3, 24]
        Y: [M, 3, 24]
        Returns: [N, M] distance matrix
        """
        N, F, L = X.shape
        M = Y.shape[0]
        
        # Flatten the temporal dimension for distance calculation -> [N, 3*24]
        X_flat = X.reshape(N, -1)
        
        # We will track the minimum distance across all permutations
        min_dists = np.full((N, M), np.inf)
        
        for p in self.perms:
            # Permute the features of Y
            Y_perm = Y[:, p, :]
            # Flatten -> [M, 3*24]
            Y_perm_flat = Y_perm.reshape(M, -1)
            
            # Compute pairwise squared euclidean distance for this permutation
            dists_p = cdist(X_flat, Y_perm_flat, metric="sqeuclidean")
            
            # Update minimum distances
            min_dists = np.minimum(min_dists, dists_p)
            
        return min_dists

    def _rbf_kernel_set(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        sq_dists = self._set_sqeuclidean_cdist(X, Y)
        sq_dists = np.nan_to_num(sq_dists, nan=1e12, posinf=1e12, neginf=0.0)
        denom = max(2.0 * (sigma ** 2), 1e-12)
        return np.exp(-sq_dists / denom)

    def _median_heuristic_sigma_set(self, Z: np.ndarray, max_points: int = 1000) -> float:
        rng = np.random.default_rng(self.seed)
        if Z.shape[0] > max_points:
            Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
            
        dists = self._set_sqeuclidean_cdist(Z, Z)
        med = np.median(dists[dists > 0])
        return float(med) if med > 0 else 1.0

    def compute(self) -> float:
        # 1. Ensure data is correctly shaped as [N, F, L]
        self.real_data = _sanitize_numeric_array(self.real_data)
        self.synthetic_data = _sanitize_numeric_array(self.synthetic_data)

        N_real = self.real_data.shape[0]
        N_synth = self.synthetic_data.shape[0]
        L = self.real_data.shape[1] // self.n_features
        
        real_3d = self.real_data.reshape(N_real, self.n_features, L)
        synth_3d = self.synthetic_data.reshape(N_synth, self.n_features, L)

        # 2. Subsample for memory efficiency
        rng = np.random.default_rng(self.seed)
        if len(real_3d) > self.max_samples:
            real_3d = real_3d[rng.choice(len(real_3d), size=self.max_samples, replace=False)]
        if len(synth_3d) > self.max_samples:
            synth_3d = synth_3d[rng.choice(len(synth_3d), size=self.max_samples, replace=False)]

        # 3. Compute Auto-Sigma on the combined Set Data
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([real_3d, synth_3d])
            sigma = self._median_heuristic_sigma_set(combined)

        # 4. Compute Kernel Matrices
        n, m = real_3d.shape[0], synth_3d.shape[0]
        Kxx = self._rbf_kernel_set(real_3d, real_3d, sigma)
        Kyy = self._rbf_kernel_set(synth_3d, synth_3d, sigma)
        Kxy = self._rbf_kernel_set(real_3d, synth_3d, sigma)

        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)

        mmd2 = (Kxx.sum() / (n * (n - 1))
              + Kyy.sum() / (m * (m - 1))
              - 2.0 * Kxy.sum() / (n * m))

        mmd_value = float(np.sqrt(np.maximum(mmd2, 0.0)))
        return mmd_value