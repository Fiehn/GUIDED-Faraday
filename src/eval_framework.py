import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import json
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import stats

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate not installed
    def tabulate(data, headers="keys", **kwargs):
        return str(data)


class EvaluationMetric(ABC):
    """Base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric (displayed in table)."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute and return metric value."""
        pass


class ReconstructionMetric(EvaluationMetric):
    """Base class for reconstruction evaluation metrics.
    
    These metrics evaluate how well the model reconstructs real data:
    - Input: real data from dataloader
    - Output: compare original vs reconstructed
    """
    
    @abstractmethod
    def compute(self) -> float:
        """Compute metric on reconstructions."""
        pass


class SyntheticMetric(EvaluationMetric):
    """Base class for synthetic data evaluation metrics.
    
    These metrics evaluate properties of generated synthetic data:
    - Input: model samples from prior (or with conditions)
    - Output: compare synthetic vs real distributions
    """
    
    @abstractmethod
    def compute(self) -> float:
        """Compute metric on synthetic samples."""
        pass


class EvalFramework:
    """
    Evaluation framework that runs multiple evaluation metrics and displays results in a table.
    
    Usage:
        evaluator = EvalFramework(model, dataloader, normalizer, device='cpu')
        evaluator.register_metric(MyMetric(model, dataloader))
        evaluator.register_metric(AnotherMetric(model, dataloader))
        results = evaluator.evaluate()
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        normalizer=None,
        conditioner=None, 
        device: str = 'cpu',
    ):
        """Initialize evaluation framework.
        
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation data
            normalizer: Data normalizer 
            conditioner: Conditioner object (for CVAE)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.conditioner = conditioner
        self.device = device
        
        # Metrics storage 
        self.reconstruction_metrics: Dict[str, ReconstructionMetric] = {}
        self.synthetic_metrics: Dict[str, SyntheticMetric] = {}
        self.results: Dict[str, float] = {}
        
    def register_metric(self, metric: EvaluationMetric) -> None:
        """Register an evaluation metric.
        
        Args:
            metric: Instance of ReconstructionMetric or SyntheticMetric
        """
        if isinstance(metric, ReconstructionMetric):
            self.reconstruction_metrics[metric.name] = metric
        elif isinstance(metric, SyntheticMetric):
            self.synthetic_metrics[metric.name] = metric
        else:
            raise ValueError(f"Metric {metric.name} must inherit from ReconstructionMetric or SyntheticMetric")
    
    def register_metrics(self, metrics: List[EvaluationMetric]) -> None:
        """Register multiple metrics at once.
        
        Args:
            metrics: List of ReconstructionMetric or SyntheticMetric instances
        """
        for metric in metrics:
            self.register_metric(metric)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run all registered evaluation metrics (reconstruction + synthetic) and return results.
        
        Returns:
            Dictionary mapping metric names to values
        """
        print("\n" + "="*70)
        print("EVALUATION FRAMEWORK")
        print("="*70 + "\n")
        
        self.results = {}
        
        # Run reconstruction metrics
        if self.reconstruction_metrics:
            print("RECONSTRUCTION METRICS")
            print("-" * 70)
            for name, metric in self.reconstruction_metrics.items():
                print(f"  {name}...", end=" ", flush=True)
                try:
                    value = metric.compute()
                    self.results[name] = value
                    print(f"{value:.6f}")
                except Exception as e:
                    print(f"Error: {e}")
                    self.results[name] = None
            print()
        
        # Run synthetic metrics
        if self.synthetic_metrics:
            print("SYNTHETIC DATA METRICS")
            print("-" * 70)
            for name, metric in self.synthetic_metrics.items():
                print(f"  {name}...", end=" ", flush=True)
                try:
                    value = metric.compute()
                    self.results[name] = value
                    print(f"{value:.6f}")
                except Exception as e:
                    print(f"Error: {e}")
                    self.results[name] = None
            print()
        
        # Display results table
        self._print_results_table()
        
        return self.results
    
    def evaluate_reconstruction_only(self) -> Dict[str, float]:
        """Run only reconstruction metrics."""
        print("\n" + "="*70)
        print("RECONSTRUCTION EVALUATION")
        print("="*70 + "\n")
        
        self.results = {}
        for name, metric in self.reconstruction_metrics.items():
            print(f"  {name}...", end=" ", flush=True)
            try:
                value = metric.compute()
                self.results[name] = value
                print(f"{value:.6f}")
            except Exception as e:
                print(f"Error: {e}")
                self.results[name] = None
        
        self._print_results_table()
        return self.results
    
    def evaluate_synthetic_only(self) -> Dict[str, float]:
        """Run only synthetic data metrics."""
        print("\n" + "="*70)
        print("SYNTHETIC DATA EVALUATION")
        print("="*70 + "\n")
        
        self.results = {}
        for name, metric in self.synthetic_metrics.items():
            print(f"  {name}...", end=" ", flush=True)
            try:
                value = metric.compute()
                self.results[name] = value
                print(f"{value:.6f}")
            except Exception as e:
                print(f"Error: {e}")
                self.results[name] = None
        
        self._print_results_table()
        return self.results
    
    def _print_results_table(self) -> None:
        """Print results in a formatted table."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70 + "\n")
        
        # Build table data
        table_data = []
        for name, value in self.results.items():
            if value is not None:
                if isinstance(value, float):
                    table_data.append([name, f"{value:.6f}"])
                else:
                    table_data.append([name, str(value)])
            else:
                table_data.append([name, "ERROR"])
        
        # Print table
        headers = ["Metric", "Value"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def get_results(self) -> Dict[str, float]:
        """Get cached results from last evaluation."""
        return self.results.copy()
    
    def save_results(self, output_path: str) -> None:
        """Save results to JSON file.
        
        Args:
            output_path: Path to save JSON results
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Convert None values to string for JSON
        results_serializable = {
            k: v if v is not None else "ERROR"
            for k, v in self.results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {output_path}")


# ============================================================================
# Reconstruction Evaluation Metrics 
# ============================================================================

# TODO: implement this 
class ReconstructionMSE(ReconstructionMetric):
    """Mean squared error on reconstruction."""
    
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    @property
    def name(self) -> str:
        return "Reconstruction MSE"
    
    def compute(self) -> float:
        """Compute reconstruction MSE on test data."""
        # TODO: Implement reconstruction MSE computation
        return 0.0

# TODO: implement this 
class ReconstructionRMSE(ReconstructionMetric):
    """Mean absolute error on reconstruction."""
    
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    @property
    def name(self) -> str:
        return "Reconstruction RMSE"
    
    def compute(self) -> float:
        """Compute reconstruction RMSE on test data."""
        # TODO: Implement reconstruction RMSE computation
        return 0.0


# TODO: implement this 
class ReconstructionLogLikelihood(ReconstructionMetric):
    """Log-likelihood of reconstruction."""
    
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    @property
    def name(self) -> str:
        return "Reconstruction Log-Likelihood"
    
    def compute(self) -> float:
        """Compute reconstruction log-likelihood."""
        # TODO: Implement reconstruction log-likelihood
        return 0.0
    

class FlexibilityDistance(ReconstructionMetric):
    """Flexibility Distance (FD) between real and reconstructed time-series.
    
    Implements the assignment-based FD from FlexibilityDistance2023:
    given two equal-length series, build a cost matrix combining amplitude
    differences and temporal shifts, then solve a linear assignment to obtain
    the minimum reshaping cost.
    """
    
    def __init__(self, model, real_data, device='cpu', tau='auto', normalize=True):
        """
        Args:
            model: Trained VAE/CVAE model
            real_data: Real data array [N, D] (flattened windows) to reconstruct
            device: Device to run on ('cpu' or 'cuda')
            tau: Temporal scaling factor. Use 'auto' to set tau = data_range / D.
            normalize: If True, divide per-sample cost by D (per-hour average).
        """
        self.model = model
        self.real_data = real_data
        self.device = device
        self.tau = tau
        self.normalize = normalize
    
    @property
    def name(self) -> str:
        return "Flexibility Distance"
    
    def compute(self) -> float:
        """Compute flexibility distance metric.
        
        Process:
        1. Pass real data through encoder-decoder (reconstruct)
        2. For each window, build cost matrix and solve linear assignment
        3. Return average FD across windows (optionally normalized per hour)
        
        Returns:
            Flexibility Distance (lower is better, 0 = perfect reconstruction)
        """
        
        # Convert to tensor
        real_tensor = torch.FloatTensor(self.real_data).to(self.device)
        
        # Flatten if needed (handle 3D [batch, length, features] -> [batch, length*features])
        if real_tensor.dim() == 3:
            batch_size, length, features = real_tensor.shape
            real_flat = real_tensor.reshape(batch_size, -1)  # [batch, length*features]
        else:
            real_flat = real_tensor
        
        # Reconstruct data through encoder-decoder
        self.model.eval()
        with torch.no_grad():
            # For CVAE: need to create dummy conditions to match training
            if hasattr(self.model, 'condition_dim') and self.model.condition_dim > 0:
                # CVAE: need conditions
                batch_size = real_flat.shape[0]
                c = torch.zeros(batch_size, self.model.condition_dim).to(self.device)
                enc_input = torch.cat([real_flat, c], dim=-1)
            else:
                # VAE: no conditions needed
                enc_input = real_flat
            
            # Encode
            q_params = self.model.encoder(enc_input)
            z = self.model.encoder.reparameterize(q_params, num_samples=1).squeeze(0)  # [batch, latent_dim]
            
            # Decode
            if hasattr(self.model, 'condition_dim') and self.model.condition_dim > 0:
                # CVAE: need to concat conditions to latent
                dec_input = torch.cat([z, c], dim=-1)
            else:
                # VAE: just use latent
                dec_input = z
            
            # Get reconstruction parameters
            recon_params = self.model.decoder(dec_input)
            
            # Extract mu (mean) from reconstruction parameters
            if isinstance(recon_params, dict):
                if 'mu' in recon_params:
                    reconstructed = recon_params['mu']
                else:
                    # Try other keys
                    for key in ['recon', 'x_recon', 'reconstruction', 'x']:
                        if key in recon_params:
                            reconstructed = recon_params[key]
                            break
                    else:
                        raise ValueError(f"Could not find reconstruction key in decoder output. Keys: {list(recon_params.keys())}")
            else:
                # Decoder output is already the reconstruction
                reconstructed = recon_params
        
        # Convert to numpy
        real_np = real_flat.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()
        
        # Ensure shapes match
        if real_np.shape != recon_np.shape:
            min_samples = min(real_np.shape[0], recon_np.shape[0])
            min_features = min(real_np.shape[1], recon_np.shape[1])
            real_np = real_np[:min_samples, :min_features]
            recon_np = recon_np[:min_samples, :min_features]
        
        # Compute FD per sample using assignment-based cost
        num_samples, series_len = real_np.shape
        indices = np.arange(series_len)
        fd_values = []
        
        for i in range(num_samples):
            x = real_np[i]
            y = recon_np[i]
            
            # Auto tau based on value range and series length
            if self.tau == 'auto' or self.tau is None:
                value_range = float(np.max([x.max(), y.max()]) - np.min([x.min(), y.min()]))
                tau = value_range / series_len if series_len > 0 else 0.0
            else:
                tau = float(self.tau)
            
            # Cost matrix: |X_i - Y_j| + |i - j| * tau
            amplitude_cost = np.abs(x[:, None] - y[None, :])
            temporal_cost = np.abs(indices[:, None] - indices[None, :]) * tau
            cost_matrix = amplitude_cost + temporal_cost
            
            # Linear sum assignment (minimum cost)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = float(cost_matrix[row_ind, col_ind].sum())
            
            if self.normalize and series_len > 0:
                total_cost /= series_len
            
            fd_values.append(total_cost)
        
        return float(np.mean(fd_values))


# ============================================================================
# Synthetic Data Evaluation Metrics
# ============================================================================



class MaximumMeanDiscrepancy(SyntheticMetric):
    """Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures the distance between two distributions using a kernel-based approach.
    Uses RBF (Gaussian) kernel with automatic bandwidth selection via median heuristic.
    """
    
    def __init__(self, real_data, synthetic_data, sigma=None, seed=42):
        """
        Args:
            real_data: numpy array [N, D] where D = window_size * n_features
            synthetic_data: numpy array [M, D] where D = window_size * n_features
            sigma: RBF kernel bandwidth. If None, uses median heuristic.
            seed: Random seed for reproducibility
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.sigma = sigma
        self.seed = seed
    
    @property
    def name(self) -> str:
        return "Maximum Mean Discrepancy (MMD)"
    
    def _rbf_kernel(self, X, Y, sigma):
        """Compute RBF (Gaussian) kernel matrix between X and Y.
        
        Args:
            X: array [n, d]
            Y: array [m, d]
            sigma: kernel bandwidth
            
        Returns:
            RBF kernel matrix [n, m]
        """
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * sigma ** 2))
    
    def _median_heuristic_sigma(self, Z, max_points=2000):
        """Estimate kernel bandwidth using median pairwise distance.
        
        Args:
            Z: array [n, d] - combined data
            max_points: max points to use for efficiency
            
        Returns:
            Estimated bandwidth sigma
        """
        
        rng = np.random.default_rng(self.seed)
        if Z.shape[0] > max_points:
            Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
        
        dists = cdist(Z, Z, metric="euclidean")
        med = np.median(dists[dists > 0])
        return float(med) if med > 0 else 1.0
    
    def _mmd_rbf_unbiased(self, X, Y, sigma=None):
        """Compute MMD^2 statistic using RBF kernel.
        
        Args:
            X: array [n, d] - real data
            Y: array [m, d] - synthetic data
            sigma: kernel bandwidth (auto-estimate if None)
            
        Returns:
            MMD^2 value
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
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
            MMD value (lower is better, 0 = identical distributions)
        """
        # Use provided sigma or auto-estimate
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([self.real_data, self.synthetic_data])
            sigma = self._median_heuristic_sigma(combined)
        
        # Compute MMD^2, return MMD 
        mmd2 = self._mmd_rbf_unbiased(self.real_data, self.synthetic_data, sigma=sigma)
        mmd = np.sqrt(np.maximum(mmd2, 0.0))  # Ensure non-negative before sqrt
        
        return float(mmd)


class ACFDistribution(SyntheticMetric):
    """
    Compare ACF coefficient distributions between real and synthetic time series.
    
    Based on: https://arxiv.org/pdf/2407.11785
    
    For each lag, we compute ACF values across all time series, then compare
    the distributions using Kolmogorov-Smirnov (KS) test. The final metric is the
    average KS statistic across all lags.
    """
    
    def __init__(self, real_data, synthetic_data, window_size=24, max_lag=10):
        """
        Args:
            real_data: numpy array [N, D] where D = window_size * n_features
            synthetic_data: numpy array [M, D] where D = window_size * n_features
            window_size: Length of each time series window (e.g., 24 hours)
            max_lag: Maximum lag to compute ACF for (default: 10)
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.window_size = window_size
        self.max_lag = max_lag
    
    @property
    def name(self) -> str:
        return "ACF Distribution Match (KS)"
    
    def _compute_acf(self, time_series, max_lag):
        """
        Compute autocorrelation function for a single time series.
        
        Args:
            time_series: 1D array of length T
            max_lag: Maximum lag to compute
            
        Returns:
            Array of ACF values for lags 1 to max_lag
        """
        n = len(time_series)
        mean = np.mean(time_series)
        c0 = np.sum((time_series - mean) ** 2) / n
        
        acf_values = []
        for lag in range(1, max_lag + 1):
            if lag >= n:
                acf_values.append(0.0)
                continue
            c_lag = np.sum((time_series[:-lag] - mean) * (time_series[lag:] - mean)) / n
            acf_values.append(c_lag / c0 if c0 > 0 else 0.0)
        
        return np.array(acf_values)
    
    def compute(self) -> float:
        """
        Compute ACF distribution matching metric.
        
        Returns:
            Average KS statistic across all lags (lower is better, 0 = perfect match)
        """
        
        # Reshape data to separate time series
        # Assuming data is [N, window_size * n_features], reshape to [N * n_features, window_size]
        n_features = self.real_data.shape[1] // self.window_size
        
        real_reshaped = self.real_data.reshape(-1, self.window_size, n_features)
        synthetic_reshaped = self.synthetic_data.reshape(-1, self.window_size, n_features)
        
        # Flatten to get all time series across all features
        # Shape: [N * n_features, window_size]
        real_ts = real_reshaped.reshape(-1, self.window_size)
        synthetic_ts = synthetic_reshaped.reshape(-1, self.window_size)
        
        # Compute ACF for each time series
        real_acfs = []
        for i in range(len(real_ts)):
            acf = self._compute_acf(real_ts[i], self.max_lag)
            real_acfs.append(acf)
        
        synthetic_acfs = []
        for i in range(len(synthetic_ts)):
            acf = self._compute_acf(synthetic_ts[i], self.max_lag)
            synthetic_acfs.append(acf)
        
        real_acfs = np.array(real_acfs)  # Shape: [N * n_features, max_lag]
        synthetic_acfs = np.array(synthetic_acfs)  # Shape: [M * n_features, max_lag]
        
        # For each lag, compare distributions using KS test
        ks_statistics = []
        for lag_idx in range(self.max_lag):
            real_lag_dist = real_acfs[:, lag_idx]
            synthetic_lag_dist = synthetic_acfs[:, lag_idx]
            
            # Kolmogorov-Smirnov test
            ks_stat, _ = stats.ks_2samp(real_lag_dist, synthetic_lag_dist)
            ks_statistics.append(ks_stat)
        
        # Average KS statistic across all lags
        avg_ks = np.mean(ks_statistics)
        
        return float(avg_ks)


class VisualEvaluation(SyntheticMetric):
    """Visual evaluation of synthetic data"""
    
    def __init__(self, real_data, synthetic_data, output_dir='src/Output/Plots'):
        """
        Args:
            real_data: numpy array of real data [N, D]
            synthetic_data: numpy array of synthetic data [M, D]
            output_dir: Directory to save plots
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Visual Evaluations"
    

    def pca(self) -> None:
        """Generate PCA plots for real vs synthetic data."""
        real_data = self.real_data[:50] if len(self.real_data) > 50 else self.real_data
        synthetic_data = self.synthetic_data[:50] if len(self.synthetic_data) > 50 else self.synthetic_data
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Apply PCA
        pca = PCA(n_components=min(3, real_data.shape[1], synthetic_data.shape[1]))
        pca_real = pca.fit_transform(real_data)
        pca_synthetic = pca.transform(synthetic_data)
        print(f'  PCA Explained Variance: {pca.explained_variance_ratio_}')
        
        # Plot
        fig = plt.figure(figsize=(16, 6))
        
        # Real data
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pca_real[:, 0], pca_real[:, 1], pca_real[:, 2], 
                   alpha=0.5, label='Real', s=20, c='blue')
        ax1.set_title('Real Data (PCA)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        
        # Synthetic data
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], pca_synthetic[:, 2], 
                   alpha=0.5, label='Synthetic', s=20, c='red')
        ax2.set_title('Synthetic Data (PCA)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_comparison.pdf'), format='pdf')
        plt.close()
    
    def tsne(self) -> None:
        """Generate t-SNE plots for real vs synthetic data."""
        # Use smaller subset for t-SNE
        real_data = self.real_data[:25] if len(self.real_data) > 25 else self.real_data
        synthetic_data = self.synthetic_data[:25] if len(self.synthetic_data) > 25 else self.synthetic_data
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Combine for t-SNE
        combined = np.vstack([real_data, synthetic_data])
        labels = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, learning_rate='auto', random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(combined)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        mask_real = labels == 0
        mask_synthetic = labels == 1
        ax.scatter(tsne_result[mask_real, 0], tsne_result[mask_real, 1], 
                  alpha=0.6, label='Real', s=30, c='blue')
        ax.scatter(tsne_result[mask_synthetic, 0], tsne_result[mask_synthetic, 1], 
                  alpha=0.6, label='Synthetic', s=30, c='red')
        ax.set_title('t-SNE: Real vs Synthetic Data')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_comparison.pdf'), format='pdf')
        plt.close()
    
    def compute(self) -> float:
        """Generate visual evaluation plots and return success indicator."""
        try:
            self.pca()
            self.tsne()
            print(f"  Visual evaluation plots saved to {self.output_dir}")
            return 1.0  # Success
        except Exception as e:
            print(f"  Error generating visualizations: {e}")
            return 0.0  # Failure





class ClassicalStatistics(SyntheticMetric):
    """Classical statistics comparison between real and synthetic data.
    
    Computes and compares basic statistical properties including mean, variance,
    min/max, quantiles, and standard deviation across time for both datasets.
    """
    
    def __init__(self, real_data, synthetic_data, window_size=24, n_features=1):
        """
        Args:
            real_data: numpy array [N, D] where D = window_size * n_features
            synthetic_data: numpy array [M, D] where D = window_size * n_features
            window_size: Length of time series window (default: 24 hours)
            n_features: Number of features (default: 1 for consumption only)
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.window_size = window_size
        self.n_features = n_features
    
    @property
    def name(self) -> str:
        return "Classical Statistics"
    
    def _compute_stats(self, data):
        """Compute classical statistics for a dataset.
        
        Args:
            data: numpy array [N, D] where D = window_size * n_features
            
        Returns:
            Dictionary with statistical measures
        """
        # Use only consumption, so reshape: [N, window_size * 2] -> [N, window_size, 2]
        data_reshaped = data.reshape(-1, self.window_size, 2)
        data = data_reshaped[:, :, 0:1] # [N, window_size, 1]
        
        # Compute mean profile: average value at each timestep across all samples, shape: [window_size, 1]
        mean_profile = np.mean(data, axis=0)
        
        # Compute std profile: std at each timestep across all samples, shape: [window_size, 1]
        std_profile = np.std(data, axis=0)
        
        # Overall statistics (across all timesteps and samples)
        overall_mean = np.mean(data)
        overall_std = np.std(data)
        overall_min = np.min(data)
        overall_max = np.max(data)
        
        stats = {
            # Time-aware statistics (shape: [window_size, 1])
            'mean_profile': mean_profile,  # Average consumption at each hour
            'std_profile': std_profile,    # Variability at each hour
            
            # Overall statistics (single values)
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_min': overall_min,
            'overall_max': overall_max,
            
            # Distribution statistics
            'quantiles': {
                '25%': np.percentile(data, 25),
                '50%': np.percentile(data, 50),
                '75%': np.percentile(data, 75)
            }
        }
        
        return stats
    
    def compute(self) -> Dict[str, Any]:
        """Compute and compare classical statistics between real and synthetic data.
        
        Returns:
            Dictionary containing both real and synthetic statistics
        """
        real_stats = self._compute_stats(self.real_data)
        synthetic_stats = self._compute_stats(self.synthetic_data)
        
        return {
            'real': real_stats,
            'synthetic': synthetic_stats
        }
    
    def print_comparison(self):
        """Print formatted comparison of statistics."""
        stats = self.compute()
        real = stats['real']
        synth = stats['synthetic']
    
        print(f"CLASSICAL STATISTICS COMPARISON")
        
        print(f"Overall Statistics:")
        print(f"  Real:      Mean={real['overall_mean']:.4f}, Std={real['overall_std']:.4f}, "
              f"Min={real['overall_min']:.4f}, Max={real['overall_max']:.4f}")
        print(f"  Synthetic: Mean={synth['overall_mean']:.4f}, Std={synth['overall_std']:.4f}, "
              f"Min={synth['overall_min']:.4f}, Max={synth['overall_max']:.4f}")

        print(f"\nQuantiles:")
        print(f"  Real:      25%={real['quantiles']['25%']:.4f}, "
              f"50%={real['quantiles']['50%']:.4f}, 75%={real['quantiles']['75%']:.4f}")
        print(f"  Synthetic: 25%={synth['quantiles']['25%']:.4f}, "
              f"50%={synth['quantiles']['50%']:.4f}, 75%={synth['quantiles']['75%']:.4f}")
        
        print(f"\nMean Profile (avg at each timestep)")
        print(f"  Real:")
        for i in range(len(real['mean_profile'])):
            print(f"    Hour {i}: {real['mean_profile'][i, 0]:.4f}")
        print(f"  Synthetic:")
        for i in range(len(synth['mean_profile'])):
            print(f"    Hour {i}: {synth['mean_profile'][i, 0]:.4f}")
        
        print(f"\nStd Profile (variability at each timestep)")
        print(f"  Real:")
        for i in range(len(real['std_profile'])):
            print(f"    Hour {i}: {real['std_profile'][i, 0]:.4f}")
        print(f"  Synthetic:")
        for i in range(len(synth['std_profile'])):
            print(f"    Hour {i}: {synth['std_profile'][i, 0]:.4f}")
        
        # Compute differences
        mean_profile_diff = np.mean(np.abs(real['mean_profile'] - synth['mean_profile']))
        std_profile_diff = np.mean(np.abs(real['std_profile'] - synth['std_profile']))
        overall_mean_diff = abs(real['overall_mean'] - synth['overall_mean'])
        
        print(f"\nMean Absolute Differences:")
        print(f"  Overall Mean Difference: {overall_mean_diff:.6f}")
        print(f"  Mean Profile Difference: {mean_profile_diff:.6f}")
        print(f"  Std Profile Difference: {std_profile_diff:.6f}")
        print(f"{'='*70}\n")


class TSTR(SyntheticMetric):
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.
    
    This metric evaluates synthetic data quality by:
    1. Training a predictive model on synthetic data
    2. Training the same model on real data
    3. Testing both models on real test data
    4. Comparing performance - good synthetic data should yield similar performance
    
    Based on: https://arxiv.org/abs/1706.02633
    
    Predicts next timestep given previous timesteps.
    """
    
    def __init__(self, real_train, real_test, synthetic_train, 
                 task='next_step', window_size=24, n_features=1, 
                 predictor_type='linear', epochs=50, random_state=42):
        """
        Args:
            real_train: Real training data [N, D] where D = window_size * n_features
            real_test: Real test data [M, D]
            synthetic_train: Synthetic training data [K, D]
            task: Prediction task - 'next_step' (predict hour 24 from hours 0-23)
            window_size: Length of time series window (default: 24)
            n_features: Number of features (default: 1 for consumption)
            predictor_type: Type of predictor ('linear', 'mlp', 'lstm')
            epochs: Training epochs for predictor
            random_state: Random seed
        """
        self.real_train = real_train
        self.real_test = real_test
        self.synthetic_train = synthetic_train
        self.task = task
        self.window_size = window_size
        self.n_features = n_features
        self.predictor_type = predictor_type
        self.epochs = epochs
        self.random_state = random_state
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    @property
    def name(self) -> str:
        return "TSTR (Train on Synthetic, Test on Real)"
    
    def _prepare_prediction_data(self, data):
        """
        Prepare data for prediction task.
        For 'next_step': X = hours 0-22, y = hour 23 
        For x days ahead: X = ?, y = hour ? 
        
        Args:
            data: [N, window_size * n_features]
            
        Returns:
            X: [N, input_features], y: [N, 1]
        """
        # Reshape to [N, window_size, n_features] and extract consumption
        data_reshaped = data.reshape(-1, self.window_size, 2)
        data_consumption = data_reshaped[:, :, 0]  # [N, window_size]
        
        # TODO: maybe predict several days ahead based on several previous days instead of just hours 

        if self.task == 'next_step':
            # Predict last hour from first 23 hours
            X = data_consumption[:, :-1]  # [N, 23]
            y = data_consumption[:, -1:]  # [N, 1]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return X, y
    
    def _create_predictor(self, input_dim):
        """Create a simple predictor model."""
        if self.predictor_type == 'linear':
            model = nn.Linear(input_dim, 1)
        elif self.predictor_type == 'mlp':
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            raise ValueError(f"Unknown predictor type: {self.predictor_type}")
        
        return model
    
    def _train_predictor(self, X, y, model):
        """Train predictor model."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            y_pred = model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model
    
    def _evaluate_predictor(self, X, y, model):
        """Evaluate predictor and return MSE."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor)
            mse = nn.MSELoss()(y_pred, y_tensor).item()
        
        return mse
    
    def compute(self) -> float:
        """
        Compute TSTR metric.
        
        Returns:
            Ratio of TSTR performance to TRTR performance (closer to 1.0 is better)
        """
        # Prepare data
        X_real_train, y_real_train = self._prepare_prediction_data(self.real_train)
        X_real_test, y_real_test = self._prepare_prediction_data(self.real_test)
        X_synth_train, y_synth_train = self._prepare_prediction_data(self.synthetic_train)
        
        input_dim = X_real_train.shape[1]
        
        # Train model on real data (TRTR - Train Real, Test Real)
        print(f"\n  Training predictor on REAL data ({len(X_real_train)} samples)...")
        model_real = self._create_predictor(input_dim)
        model_real = self._train_predictor(X_real_train, y_real_train, model_real)
        trtr_mse = self._evaluate_predictor(X_real_test, y_real_test, model_real)
        
        # Train model on synthetic data (TSTR - Train Synthetic, Test Real)
        print(f"  Training predictor on SYNTHETIC data ({len(X_synth_train)} samples)...")
        model_synth = self._create_predictor(input_dim)
        model_synth = self._train_predictor(X_synth_train, y_synth_train, model_synth)
        tstr_mse = self._evaluate_predictor(X_real_test, y_real_test, model_synth)
        
        # Compute ratio (ideal: 1.0, higher = synthetic is worse)
        tstr_ratio = tstr_mse / trtr_mse if trtr_mse > 0 else float('inf')
        print(f"  TRTR MSE: {trtr_mse:.6f}")
        print(f"  TSTR MSE: {tstr_mse:.6f}")
        print(f"  TSTR/TRTR Ratio: {tstr_ratio:.6f}")
        
        return tstr_ratio





# ============================================================================
# Usage 
# ============================================================================

if __name__ == "__main__":
    print("Evaluation Framework - Separate Reconstruction & Synthetic Metrics")
    print("\n" + "="*70)
    print("EXAMPLE 1: Run all metrics (reconstruction + synthetic)")
    print("="*70)
    print("""
    from src.eval_framework import EvalFramework
    from src.eval_framework import (
        ReconstructionMSE, ReconstructionMAE, KLDivergence,
        DistributionMatch, TemporalConsistency, PrivacyMetric
    )
    
    # Create evaluator
    evaluator = EvalFramework(model, dataloader, normalizer, device='cpu')
    
    # Register reconstruction metrics
    evaluator.register_metrics([
        ReconstructionMSE(model, dataloader),
        ReconstructionMAE(model, dataloader),
        KLDivergence(model, dataloader),
    ])
    
    # Register synthetic metrics
    evaluator.register_metrics([
        DistributionMatch(model, dataloader),
        TemporalConsistency(model, dataloader),
        PrivacyMetric(model, dataloader),
    ])
    
    # Run all metrics (will print reconstruction section, then synthetic section)
    results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results('Output/eval_results.json')
    """)
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Run only reconstruction metrics")
    print("="*70)
    print("""
    evaluator = EvalFramework(model, dataloader, normalizer, device='cpu')
    evaluator.register_metrics([
        ReconstructionMSE(model, dataloader),
        KLDivergence(model, dataloader),
    ])
    
    # Only evaluate reconstruction
    results = evaluator.evaluate_reconstruction_only()
    """)
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Run only synthetic metrics")
    print("="*70)
    print("""
    evaluator = EvalFramework(model, dataloader, normalizer, device='cpu')
    evaluator.register_metrics([
        DistributionMatch(model, dataloader),
        TemporalConsistency(model, dataloader),
        PrivacyMetric(model, dataloader),
    ])
    
    # Only evaluate synthetic data
    results = evaluator.evaluate_synthetic_only()
    """)