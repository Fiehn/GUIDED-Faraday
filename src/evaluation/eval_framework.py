import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import json
from typing import Dict, List, Any
from abc import ABC, abstractmethod
 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import stats

from src.visualization.plot_styling import PlotStyle

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate not installed
    def tabulate(data, headers="keys", **kwargs):
        return str(data)


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch-like objects into JSON-serializable values."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_standalone_metric_result(metric_key: str, payload: Dict[str, Any],
                                   output_file: str = 'models/eval_results.json') -> None:
    """Save/update standalone metric results in the shared per-condition JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        existing = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        if not isinstance(existing, dict):
            existing = {}

        existing.setdefault('standalone_metrics', {})
        existing['standalone_metrics'][metric_key] = _to_json_serializable(payload)

        with open(output_file, 'w') as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save standalone metric '{metric_key}' to {output_file}: {e}")


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
        window_size: int = 24,
        n_synthetic_samples: int = 1000,
        output_path: str = 'models/eval_results.json'
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

        self.window_size = window_size
        self.n_synthetic_samples = n_synthetic_samples
        
        # Metrics storage 
        self.reconstruction_metrics: Dict[str, ReconstructionMetric] = {}
        self.synthetic_metrics: Dict[str, SyntheticMetric] = {}
        self.results: Dict[str, float] = {}

        self._real_data: np.ndarray = None  # Cache real data for synthetic metrics that need it
        self._reconstructed_data: np.ndarray = None  # Cache reconstructed data for metrics that need it
        self._synthetic_data: np.ndarray = None  # Cache synthetic data for metrics that need it
        self._timestamps: np.ndarray = None  # Cache timestamps for metrics that need it

        self.output_path = output_path

    def register_all(self):
        self.register_metrics([
            ReconstructionRMSE(),
            ReconstructionLogLikelihood(),
            FlexibilityDistance(),
            MaximumMeanDiscrepancy(),
            ACFDistribution(),
            ClassicalStatistics(),
            TSTR(),
            AggregatedProfiles(),
            PerConditionEval()
        ])

    def _is_cvae(self) -> bool:
        """Check if model is a CVAE based on presence of conditioner on the model."""
        return hasattr(self.model, 'conditioner') and self.model.conditioner is not None
    
    def _prepare_data(self) -> None:
        """Extract real data, reconstructed data, and synthetic data for metrics that need it."""
        print("Preparing data for evaluation metrics...")
        real_batches = []
        recon_batches = []
        condition_batches = []
        timestamp = []

        self.model.eval()
        with torch.no_grad():
            for data, c, meter_ids, timestamps in self.dataloader:
                data = data.to(self.device)
                c = c.to(self.device)
                x_flat = data.reshape(data.shape[0], -1) if data.dim() == 3 else data
                real_batches.append(x_flat.cpu().numpy())

                timestamp.append(timestamps.cpu().numpy())

                recon = self._reconstruct(data, meter_id=meter_ids, timestamps=timestamps)
                recon_batches.append(recon.cpu().numpy())

                if self._is_cvae():
                    # Use the model's own conditioner, not self.conditioner
                    condition_batches.append(c.cpu())

        self._timestamps = np.concatenate(timestamp, axis=0) if timestamp else None
        self._real_data = np.concatenate(real_batches, axis=0)
        self._reconstructed_data = np.concatenate(recon_batches, axis=0)

        if self._is_cvae() and condition_batches:
            all_conditions = torch.cat(condition_batches, dim=0)
            n = self.n_synthetic_samples
            if len(all_conditions) >= n:
                idx = torch.randperm(len(all_conditions))[:n]
                self._synthetic_conditions = all_conditions[idx]
            else:
                repeats = (n // len(all_conditions)) + 1
                self._synthetic_conditions = all_conditions.repeat(repeats, 1)[:n]
        else:
            self._synthetic_conditions = None

        self._synthetic_data = self._generate_synthetic(self.n_synthetic_samples)

        print(f"  Real: {self._real_data.shape} | "
              f"Reconstructed: {self._reconstructed_data.shape} | "
              f"Synthetic: {self._synthetic_data.shape}")
    
    def _reconstruct(self, x: torch.Tensor, meter_id=None, timestamps=None) -> torch.Tensor:
        """Reconstruct via encoder-decoder, using real conditions for CVAE."""
        with torch.no_grad():
            x_flat = x.reshape(x.shape[0], -1) if x.dim() == 3 else x

            if self._is_cvae():
                # Use the model's own conditioner with real meter_id and timestamps
                c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                c = self.model.move_to_device(c, device=self.device)
                recon_params, _ = self.model.forward(x_flat, c, num_samples=1)
            else:
                recon_params, _ = self.model.forward(x_flat, num_samples=1)

            if isinstance(recon_params, dict):
                mu = recon_params.get('mu')
                if mu is not None:
                    recon = mu[0] if mu.dim() == 3 else mu
                else:
                    for key in ['recon', 'x_recon', 'reconstruction', 'x']:
                        if key in recon_params:
                            v = recon_params[key]
                            recon = v[0] if v.dim() == 3 else v
                            break
                    else:
                        raise ValueError(f"Cannot find reconstruction in decoder output. Keys: {list(recon_params.keys())}")
            else:
                recon = recon_params[0] if recon_params.dim() == 3 else recon_params

        return recon  # [B, D]
    
    def _generate_synthetic(self, n_samples: int) -> np.ndarray:
        """Sample from prior using real conditions to match the real data distribution."""
        with torch.no_grad():
            if self._is_cvae():
                # Use real conditions collected during _prepare_data
                # TODO: GENERATOR HERE!
                c = self._synthetic_conditions.to(self.device)  # [n_samples, C]

                samples = self.model.sample(
                    conditions=c,
                    num_prior_samples=1,
                    num_x_samples=1,
                    device=self.device,
                )
                # [1, 1, B, D] -> [B, D]
                synthetic_flat = samples[0, 0]

            else:
                result = self.model.sample(
                    num_samples_prior=n_samples,
                    num_samples_posterior=1,
                )
                synthetic_flat = result["samples"][:, 0, 0, :]  # [n_samples, D]

        return synthetic_flat.cpu().numpy()

    def _inject_data_into_metrics(self) -> None:
        """Inject prepared data into metrics that need it."""
        for metric in self.reconstruction_metrics.values():
            metric.real_data = self._real_data
            metric.reconstructed_data = self._reconstructed_data
            metric.model = self.model
            metric.device = self.device
            metric.normalizer = self.normalizer
            metric.output_path = self.output_path
            metric.timestamps = self._timestamps

        for metric in self.synthetic_metrics.values():
            metric.real_data = self._real_data
            metric.synthetic_data = self._synthetic_data
            metric.model = self.model
            metric.device = self.device
            metric.normalizer = self.normalizer
            metric.output_path = self.output_path
            metric.timestamps = self._timestamps

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
        
        # Prepare data for metrics that need it (reconstructions, synthetic samples)
        self._prepare_data()
        self._inject_data_into_metrics()

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

class ReconstructionMSE(ReconstructionMetric):
    """Mean squared error on reconstruction."""
    
    real_data: np.ndarray = None
    reconstructed_data: np.ndarray = None
    output_path: str = None
    
    @property
    def name(self) -> str:
        return "Reconstruction MSE"
    
    def compute(self) -> float:
        """Compute reconstruction MSE on test data."""
        if self.real_data is None or self.reconstructed_data is None:
            raise ValueError("ReconstructionMSE requires injected real_data and reconstructed_data before compute().")

        real = np.asarray(self.real_data)
        recon = np.asarray(self.reconstructed_data)

        if real.ndim != 2 or recon.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real.shape}, reconstructed={recon.shape}")

        # Align shapes safely if they differ
        if real.shape != recon.shape:
            min_samples = min(real.shape[0], recon.shape[0])
            min_features = min(real.shape[1], recon.shape[1])
            if min_samples == 0 or min_features == 0:
                raise ValueError(f"Invalid shapes for MSE: real={real.shape}, reconstructed={recon.shape}")
            real = real[:min_samples, :min_features]
            recon = recon[:min_samples, :min_features]

        diff = real - recon
        mse = float(np.mean(diff ** 2))

        _save_standalone_metric_result(
            metric_key='ReconstructionMSE',
            payload={
                'value': mse,
                'n_samples': int(real.shape[0]),
                'n_features': int(real.shape[1]),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mse


class ReconstructionRMSE(ReconstructionMetric):
    """Mean absolute error on reconstruction."""
    
    real_data: np.ndarray = None
    reconstructed_data: np.ndarray = None
    output_path: str = None
    
    @property
    def name(self) -> str:
        return "Reconstruction RMSE"
    
    def compute(self) -> float:
        """Compute reconstruction RMSE on test data."""
        if self.real_data is None or self.reconstructed_data is None:
            raise ValueError("ReconstructionRMSE requires injected real_data and reconstructed_data before compute().")

        real = np.asarray(self.real_data)
        recon = np.asarray(self.reconstructed_data)

        if real.ndim != 2 or recon.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real.shape}, reconstructed={recon.shape}")

        # Align shapes safely if they differ
        if real.shape != recon.shape:
            min_samples = min(real.shape[0], recon.shape[0])
            min_features = min(real.shape[1], recon.shape[1])
            if min_samples == 0 or min_features == 0:
                raise ValueError(f"Invalid shapes for RMSE: real={real.shape}, reconstructed={recon.shape}")
            real = real[:min_samples, :min_features]
            recon = recon[:min_samples, :min_features]

        diff = real - recon
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        _save_standalone_metric_result(
            metric_key='ReconstructionRMSE',
            payload={
                'value': rmse,
                'n_samples': int(real.shape[0]),
                'n_features': int(real.shape[1]),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return rmse


class ReconstructionLogLikelihood(ReconstructionMetric):
    """Log-likelihood of reconstruction."""

    real_data: np.ndarray = None
    reconstructed_data: np.ndarray = None
    
    @property
    def name(self) -> str:
        return "Reconstruction Log-Likelihood"
    
    def compute(self) -> float:
        """Compute reconstruction log-likelihood.

        Uses a simple isotropic Gaussian likelihood approximation:
        x | x_hat ~ N(x_hat, sigma^2 I), where sigma^2 is estimated from residuals.
        Returns mean log-likelihood per element (higher is better).
        """
        if self.real_data is None or self.reconstructed_data is None:
            raise ValueError(
                "ReconstructionLogLikelihood requires injected real_data and reconstructed_data before compute()."
            )

        real = np.asarray(self.real_data)
        recon = np.asarray(self.reconstructed_data)

        if real.ndim != 2 or recon.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real.shape}, reconstructed={recon.shape}")

        # Align shapes safely if they differ
        if real.shape != recon.shape:
            min_samples = min(real.shape[0], recon.shape[0])
            min_features = min(real.shape[1], recon.shape[1])
            if min_samples == 0 or min_features == 0:
                raise ValueError(
                    f"Invalid shapes for log-likelihood: real={real.shape}, reconstructed={recon.shape}"
                )
            real = real[:min_samples, :min_features]
            recon = recon[:min_samples, :min_features]

        residual = real - recon

        # Estimate isotropic residual variance and guard against degenerate zero variance
        eps = 1e-8
        sigma2 = float(np.var(residual))
        sigma2 = max(sigma2, eps)

        log_const = -0.5 * np.log(2.0 * np.pi * sigma2)
        ll_matrix = log_const - (residual ** 2) / (2.0 * sigma2)

        mean_log_likelihood = float(np.mean(ll_matrix))
        total_log_likelihood = float(np.sum(ll_matrix))

        _save_standalone_metric_result(
            metric_key='ReconstructionLogLikelihood',
            payload={
                'value': mean_log_likelihood,
                'total_log_likelihood': total_log_likelihood,
                'sigma2_estimate': sigma2,
                'n_samples': int(real.shape[0]),
                'n_features': int(real.shape[1]),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mean_log_likelihood
    

class FlexibilityDistance(ReconstructionMetric):
    """Flexibility Distance (FD) between real and reconstructed time-series.
    
    Implements the assignment-based FD from FlexibilityDistance2023:
    given two equal-length series, build a cost matrix combining amplitude
    differences and temporal shifts, then solve a linear assignment to obtain
    the minimum reshaping cost.
    """
    
    real_data: np.ndarray = None
    reconstructed_data: np.ndarray = None
    output_path: str = None

    def __init__(self, tau='auto', normalize=True):
        """
        Args:
            tau: Temporal scaling factor. Use 'auto' to set tau = data_range / D.
            normalize: If True, divide per-sample cost by D (per-hour average).
        """
        self.tau = tau
        self.normalize = normalize
    
    @property
    def name(self) -> str:
        return "Flexibility Distance"
    
    @staticmethod
    def compute_fd_between_series(x: np.ndarray, y: np.ndarray, tau='auto', normalize=True) -> float:
        """Compute flexibility distance between two time series.
        
        Args:
            x: First time series [T]
            y: Second time series [T]
            tau: Temporal scaling factor ('auto' or float)
            normalize: If True, divide cost by series length
            
        Returns:
            Flexibility distance value
        """
        series_len = len(x)
        indices = np.arange(series_len)
        
        # Auto tau based on value range and series length
        if tau == 'auto' or tau is None:
            value_range = float(np.max([x.max(), y.max()]) - np.min([x.min(), y.min()]))
            tau_val = value_range / series_len if series_len > 0 else 0.0
        else:
            tau_val = float(tau)
        
        # Cost matrix: |X_i - Y_j| + |i - j| * tau
        amplitude_cost = np.abs(x[:, None] - y[None, :])
        temporal_cost = np.abs(indices[:, None] - indices[None, :]) * tau_val
        cost_matrix = amplitude_cost + temporal_cost
        
        # Linear sum assignment (minimum cost)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        
        if normalize and series_len > 0:
            total_cost /= series_len
        
        return total_cost
    
    def compute(self) -> float:
        """Compute flexibility distance metric.
        
        Process:
        2. For each window, build cost matrix and solve linear assignment
        3. Return average FD across windows (optionally normalized per hour)
        
        Returns:
            Flexibility Distance (lower is better, 0 = perfect reconstruction)
        """
        if self.real_data is None or self.reconstructed_data is None:
            raise ValueError(
                "FlexibilityDistance requires injected real_data and reconstructed_data before compute()."
            )

        real_np = np.asarray(self.real_data)
        recon_np = np.asarray(self.reconstructed_data)

        if real_np.ndim != 2 or recon_np.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real_np.shape}, reconstructed={recon_np.shape}")
        
        # Ensure shapes match
        if real_np.shape != recon_np.shape:
            min_samples = min(real_np.shape[0], recon_np.shape[0])
            min_features = min(real_np.shape[1], recon_np.shape[1])
            real_np = real_np[:min_samples, :min_features]
            recon_np = recon_np[:min_samples, :min_features]
        
        # Compute FD per sample using assignment-based cost
        num_samples, series_len = real_np.shape
        fd_values = []
        
        for i in range(num_samples):
            x = real_np[i]
            y = recon_np[i]
            fd = self.compute_fd_between_series(x, y, tau=self.tau, normalize=self.normalize)
            fd_values.append(fd)
        
        return float(np.mean(fd_values))


# ============================================================================
# Synthetic Data Evaluation Metrics
# ============================================================================



class MaximumMeanDiscrepancy(SyntheticMetric):
    """Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures the distance between two distributions using a kernel-based approach.
    Uses RBF (Gaussian) kernel with automatic bandwidth selection via median heuristic.
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, sigma=None, seed=42, max_samples=2000):
        """
        Args:
            sigma: RBF kernel bandwidth. If None, uses median heuristic.
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use for median heuristic (for efficiency)
        """
        self.sigma = sigma
        self.seed = seed
        self.max_samples = max_samples
    
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
        rng = np.random.default_rng(self.seed)

        real = self.real_data
        synthetic = self.synthetic_data

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
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mmd_value


class ACFDistribution(SyntheticMetric):
    """
    Compare ACF coefficient distributions between real and synthetic time series.
    
    Based on: https://arxiv.org/pdf/2407.11785
    
    For each lag, we compute ACF values across all time series, then compare
    the distributions using Kolmogorov-Smirnov (KS) test. The final metric is the
    average KS statistic across all lags.
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size=24, max_lag=10):
        """
        Args:
            real_data: numpy array [N, D] where D = window_size * n_features
            synthetic_data: numpy array [M, D] where D = window_size * n_features
            window_size: Length of each time series window (e.g., 24 hours)
            max_lag: Maximum lag to compute ACF for (default: 10)
        """
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
        avg_ks = float(np.mean(ks_statistics))

        _save_standalone_metric_result(
            metric_key='ACFDistribution',
            payload={
                'value': avg_ks,
                'window_size': int(self.window_size),
                'max_lag': int(self.max_lag),
                'ks_by_lag': [float(v) for v in ks_statistics],
                'n_real_series': int(len(real_ts)),
                'n_synthetic_series': int(len(synthetic_ts)),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return avg_ks


class ClassicalStatistics(SyntheticMetric):
    """Classical statistics comparison between real and synthetic data.
    
    Computes and compares basic statistical properties including mean, variance,
    min/max, quantiles, and standard deviation across time for both datasets.
    """
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size=24, n_features=1):
        """
        Args:
            window_size: Length of time series window (default: 24 hours)
            n_features: Number of features (default: 1 for consumption only)
        """
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
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {data.shape}")
        if data.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Data feature dimension ({data.shape[1]}) must be divisible by window_size ({self.window_size})"
            )

        # Support both [N, window_size] (consumption only) and [N, window_size * n_features]
        inferred_n_features = data.shape[1] // self.window_size
        data_reshaped = data.reshape(-1, self.window_size, inferred_n_features)
        data = data_reshaped[:, :, 0:1]  # consumption channel, shape [N, window_size, 1]
        
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
        
        print(f"  Computed classical statistics for real and synthetic data.")
        print(f"  Overall Mean - Real: {real_stats['overall_mean']:.4f}, Synthetic: {synthetic_stats['overall_mean']:.4f}")
        print(f"  Overall Std - Real: {real_stats['overall_std']:.4f}, Synthetic: {synthetic_stats['overall_std']:.4f}")
        print(f"  Overall Min - Real: {real_stats['overall_min']:.4f}, Synthetic: {synthetic_stats['overall_min']:.4f}")
        print(f"  Overall Max - Real: {real_stats['overall_max']:.4f}, Synthetic: {synthetic_stats['overall_max']:.4f}")
        print(f"  Quantiles (25%, 50%, 75%) - Real: {real_stats['quantiles']}, Synthetic: {synthetic_stats['quantiles']}")

        # Scalar summary: mean absolute difference of mean profiles
        mean_profile_diff = float(np.mean(np.abs(
            real_stats['mean_profile'] - synthetic_stats['mean_profile']
        )))
        std_profile_diff = float(np.mean(np.abs(
            real_stats['std_profile'] - synthetic_stats['std_profile']
        )))
        overall_mean_diff = float(abs(real_stats['overall_mean'] - synthetic_stats['overall_mean']))

        _save_standalone_metric_result(
            metric_key='ClassicalStatistics',
            payload={
                'value': mean_profile_diff,
                'differences': {
                    'mean_profile_mae': mean_profile_diff,
                    'std_profile_mae': std_profile_diff,
                    'overall_mean_abs_diff': overall_mean_diff,
                },
                'real': real_stats,
                'synthetic': synthetic_stats,
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mean_profile_diff
    
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

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, task='next_step', window_size=24, n_features=1, 
                 predictor_type='linear', epochs=50, random_state=42):
        """
        Args:
            task: Prediction task - 'next_step' (predict hour 24 from hours 0-23)
            window_size: Length of time series window (default: 24)
            n_features: Number of features (default: 1 for consumption)
            predictor_type: Type of predictor ('linear', 'mlp', 'lstm')
            epochs: Training epochs for predictor
            random_state: Random seed
        """
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
        # Reshape to [N, window_size, n_features] and extract consumption (feature 0)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {data.shape}")

        if data.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Data feature dimension ({data.shape[1]}) must be divisible by window_size ({self.window_size})"
            )

        inferred_n_features = data.shape[1] // self.window_size
        data_reshaped = data.reshape(-1, self.window_size, inferred_n_features)
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
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("TSTR requires injected real_data and synthetic_data before compute().")

        real_data = np.asarray(self.real_data)
        synthetic_data = np.asarray(self.synthetic_data)

        if len(real_data) < 2:
            raise ValueError("Need at least 2 real samples for an 80/20 train/test split.")

        # Split real data: 80% train, 20% test
        split_idx = int(0.8 * len(real_data))
        split_idx = max(1, min(split_idx, len(real_data) - 1))
        real_train = real_data[:split_idx]
        real_test = real_data[split_idx:]

        # Cap synthetic data to real training size (if synthetic is larger)
        if len(synthetic_data) > len(real_train):
            synthetic_train = synthetic_data[:len(real_train)]
        else:
            synthetic_train = synthetic_data

        if len(synthetic_train) == 0:
            raise ValueError("Synthetic data is empty after capping.")

        # Prepare data
        X_real_train, y_real_train = self._prepare_prediction_data(real_train)
        X_real_test, y_real_test = self._prepare_prediction_data(real_test)
        X_synth_train, y_synth_train = self._prepare_prediction_data(synthetic_train)
        
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

        _save_standalone_metric_result(
            metric_key='TSTR',
            payload={
                'value': float(tstr_ratio),
                'trtr_mse': float(trtr_mse),
                'tstr_mse': float(tstr_mse),
                'task': self.task,
                'window_size': int(self.window_size),
                'predictor_type': self.predictor_type,
                'epochs': int(self.epochs),
                'n_real_total': int(len(real_data)),
                'n_real_train': int(len(real_train)),
                'n_real_test': int(len(real_test)),
                'n_synthetic_train': int(len(synthetic_train)),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return tstr_ratio


class AggregatedProfiles(SyntheticMetric):
    """Aggregate samples from real and synthetic data and compute Flexibility Distance between aggregated profiles."""
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    normalizer = None
    output_dir: str = None
    
    def __init__(self, n_samples=100, window_size=24, tau='auto', normalize=True):
        self.n_samples = n_samples
        self.window_size = window_size
        self.tau = tau
        self.normalize = normalize
    
    @property
    def name(self) -> str:
        return "Aggregated Profiles (Flexibility Distance)"
    
    def compute(self) -> float:
        """Aggregate real and synthetic data, then compute Flexibility Distance."""
        real_data = np.asarray(self.real_data)
        synthetic_data = np.asarray(self.synthetic_data)
        
        # Denormalize synthetic data if normalizer is available
        if self.normalizer is not None and hasattr(self.normalizer, 'inverse_transform'):
            synthetic_data = self.normalizer.inverse_transform(synthetic_data).detach().cpu().numpy()

        # Extract consumption only (feature 0)
        real_n_features = real_data.shape[1] // self.window_size
        synth_n_features = synthetic_data.shape[1] // self.window_size
        
        real_data = real_data.reshape(-1, self.window_size, real_n_features)[:, :, 0]
        synthetic_data = synthetic_data.reshape(-1, self.window_size, synth_n_features)[:, :, 0]
        
        # Use the same number of samples for fair comparison
        n_agg = min(self.n_samples, len(real_data), len(synthetic_data))
        real_samples = real_data[:n_agg]
        synth_samples = synthetic_data[:n_agg]
        
        print(f"  Aggregating {n_agg} samples (real: {len(real_data)} available, synth: {len(synthetic_data)} available)")
        
        # Aggregate by summing
        real_aggregated = np.sum(real_samples, axis=0)
        synth_aggregated = np.sum(synth_samples, axis=0)
        
        # Plot aggregated profiles
        self._plot_aggregated_profiles(real_aggregated, synth_aggregated)
        
        # Compute Flexibility Distance
        fd_score = FlexibilityDistance.compute_fd_between_series(
            real_aggregated, synth_aggregated, tau=self.tau, normalize=self.normalize
        )

        _save_standalone_metric_result(
            metric_key='AggregatedProfiles',
            payload={
                'value': float(fd_score),
                'n_samples_requested': int(self.n_samples),
                'n_samples_used': int(n_agg),
                'window_size': int(self.window_size),
                'tau': self.tau,
                'normalize': bool(self.normalize),
                'real_aggregated': real_aggregated,
                'synthetic_aggregated': synth_aggregated,
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return fd_score
    
    def _plot_aggregated_profiles(self, real_aggregated, synth_aggregated):
        """Plot aggregated real and synthetic profiles."""
        fig, ax = plt.subplots(figsize=(14, 6))
        timesteps = np.arange(len(real_aggregated))
        ax.plot(timesteps, real_aggregated, label='Real Aggregated', 
                color='blue', linewidth=2, alpha=0.8)
        ax.plot(timesteps, synth_aggregated, label='Synthetic Aggregated', 
                color='red', linewidth=2, alpha=0.8, linestyle='--')
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Aggregated Consumption [kWh]', fontsize=12)
        ax.set_title(f'Aggregated Consumption Profiles (n={self.n_samples} samples)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Ensure output_path doesn't contain a filename
        base_dir = os.path.dirname(self.output_path) if self.output_path and self.output_path.endswith('.json') else self.output_path
        plot_dir = os.path.join(base_dir, 'Plots') if base_dir else 'Plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.savefig(os.path.join(plot_dir, 'aggregated_profiles_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        

class PerConditionEval(SyntheticMetric):
    """Compute per-condition evaluation for MMD, ACFDistribution, TSTR, ClassicalStatistics, and AggregatedProfiles.

    Includes:
    - MMD per day-of-week and month (using full 24-hour windows)
    - ACFDistribution per day-of-week and month (using full 24-hour windows)
    - ClassicalStatistics per day-of-week and month
    - TSTR per day-of-week and month (train/test on real, train on synthetic, test on real holdout)
    - AggregatedProfiles per day-of-week and month (aggregate samples within each group and compare profiles)

    Can be tested using test_eval.py
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    meter_ids: np.ndarray = None
    feeder_ids: np.ndarray = None
    output_dir: str = None

    def __init__(self, window_size=24, max_lag=10, min_samples_per_group=2, 
                 verbose=False, predictor_type='linear', tstr_epochs=50,
                 feeder_mapping_path='topology/cleaned/Cleaned_SM_Topology_mapping.json',
                 min_samples_per_feeder=5):
        self.window_size = window_size
        self.max_lag = max_lag
        self.min_samples_per_group = min_samples_per_group
        self.verbose = verbose
        self.predictor_type = predictor_type
        self.tstr_epochs = tstr_epochs
        self.feeder_mapping_path = feeder_mapping_path
        self.min_samples_per_feeder = min_samples_per_feeder
        self._meter_to_feeder = None

        self.results_by_condition = {
            'mmd': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'acf': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'classical_statistics': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'aggregated_profiles': {
                'day_of_week': {},
                'month_of_year': {},
                'feeder': {},
            },
            'tstr': {
                'day_of_week': {},
                'month_of_year': {},
            }
        }

        self.mmd_calculator = MaximumMeanDiscrepancy(sigma=None, seed=42, max_samples=2000)
        self.acf_calculator = ACFDistribution(window_size=window_size, max_lag=max_lag)
        self.classical_statistics_calculator = ClassicalStatistics(window_size=window_size, n_features=1)
        self.tstr_calculator = TSTR(
            task='next_step',
            window_size=window_size,
            n_features=1,
            predictor_type=predictor_type,
            epochs=tstr_epochs,
            random_state=42
        )
        self.aggregated_profiles_calculator = AggregatedProfiles(
            n_samples=100,
            window_size=window_size,
            tau='auto',
            normalize=True,
        )

    @property
    def name(self) -> str:
        return "Per-Condition Evaluation"

    def _print_input_debug(self, real_data: np.ndarray, synthetic_data: np.ndarray, timestamps: np.ndarray) -> None:
        """Print information about input formats."""
        print("\n" + "=" * 70)
        print("PER-CONDITION DATA INPUT")
        print("=" * 70)
        print(f"real_data:      shape={real_data.shape}, dtype={real_data.dtype}")
        print(f"synthetic_data: shape={synthetic_data.shape}, dtype={synthetic_data.dtype}")
        print(f"timestamps:     shape={timestamps.shape}, dtype={timestamps.dtype}")

        if len(real_data) > 0:
            print(f"real_data[0] (first window, len={len(real_data[0])}):")
            print(np.array2string(real_data[0], precision=4, separator=", "))

        if len(synthetic_data) > 0:
            print(f"synthetic_data[0] (first window, len={len(synthetic_data[0])}):")
            print(np.array2string(synthetic_data[0], precision=4, separator=", "))

        if len(timestamps) > 0:
            dt0 = datetime.utcfromtimestamp(int(timestamps[0]))
            print(f"timestamps[0]: {int(timestamps[0])} -> {dt0.strftime('%Y-%m-%d %A %H:%M:%S')} UTC")

    def _compute_subset_acf(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute ACF score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None
        self.acf_calculator.real_data = real_subset
        self.acf_calculator.synthetic_data = synthetic_subset
        return float(self.acf_calculator.compute())

    def _compute_subset_tstr(self, real_subset: np.ndarray, synthetic_subset: np.ndarray,
                             group_label: str = None) -> float:
        """Compute TSTR ratio for one subset using the TSTR metric; return None if too few samples."""
        if len(real_subset) < 4 or len(synthetic_subset) < 2:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} -> insufficient samples (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            return None

        try:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            self.tstr_calculator.real_data = real_subset
            self.tstr_calculator.synthetic_data = synthetic_subset
            tstr_ratio = self.tstr_calculator.compute()
            return float(tstr_ratio)
        except Exception as e:
            if self.verbose:
                label_suffix = f" for {group_label}" if group_label is not None else ""
                print(f"Warning: TSTR computation failed{label_suffix}: {e}")
            return None

    def _compute_subset_aggregated_profiles(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute AggregatedProfiles score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            self.aggregated_profiles_calculator.real_data = real_subset
            self.aggregated_profiles_calculator.synthetic_data = synthetic_subset
            self.aggregated_profiles_calculator.output_path = self.output_path
            return float(self.aggregated_profiles_calculator.compute())
        except Exception as e:
            if self.verbose:
                print(f"Warning: AggregatedProfiles computation failed: {e}")
            return None

    def _load_meter_to_feeder_mapping(self) -> Dict[str, str]:
        """Load meter->feeder mapping from Cleaned_SM_Topology_mapping.json."""
        if self._meter_to_feeder is not None:
            return self._meter_to_feeder

        mapping_path = self.feeder_mapping_path
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.join(os.getcwd(), mapping_path)

        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)

        meter_to_feeder = {}
        for meter, topo in raw_mapping.items():
            feeder_id = topo.get('Feeder ID') if isinstance(topo, dict) else None
            if feeder_id is None:
                meter_to_feeder[str(meter)] = None
            else:
                meter_to_feeder[str(meter)] = f"LvFeeder.{feeder_id}"

        self._meter_to_feeder = meter_to_feeder
        return meter_to_feeder

    def _resolve_feeder_ids(self, n_expected: int) -> np.ndarray:
        """Resolve feeder IDs either from injected feeder_ids or from injected meter_ids + mapping file."""
        if self.feeder_ids is not None:
            feeder_ids = np.asarray(self.feeder_ids)
        elif self.meter_ids is not None:
            meter_ids = np.asarray(self.meter_ids)
            if len(meter_ids) != n_expected:
                raise ValueError(f"meter_ids length ({len(meter_ids)}) must match data length ({n_expected})")

            meter_to_feeder = self._load_meter_to_feeder_mapping()
            feeder_ids = []
            for meter in meter_ids:
                if isinstance(meter, np.generic):
                    meter = meter.item()
                meter_key = str(meter)
                feeder_ids.append(meter_to_feeder.get(meter_key, None))
            feeder_ids = np.array(feeder_ids, dtype=object)
        else:
            return None

        if len(feeder_ids) != n_expected:
            raise ValueError(f"feeder_ids length ({len(feeder_ids)}) must match data length ({n_expected})")
        return feeder_ids

    def _compute_aggregated_profiles_per_feeder(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feeder_ids: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute AggregatedProfiles FD per feeder."""
        feeder_scores = {}
        unique_feeders = [f for f in np.unique(feeder_ids) if f is not None and str(f).lower() != 'nan']

        if self.verbose:
            print(f"  Found {len(unique_feeders)} unique feeders")
            print(f"  Min samples threshold per feeder: {self.min_samples_per_feeder}")

        for feeder in unique_feeders:
            n_group = int(np.sum(feeder_ids == feeder))
            if n_group < self.min_samples_per_feeder:
                if self.verbose:
                    print(f"    Skipping {feeder}: only {n_group} samples (need {self.min_samples_per_feeder})")
                continue

            score = self._compute_subset_aggregated_profiles(real_data[feeder_ids == feeder], 
                                                             synthetic_data[feeder_ids == feeder])
            if score is None:
                if self.verbose:
                    print(f"    {feeder}: insufficient samples for FD computation")
                continue

            feeder_scores[str(feeder)] = {
                'fd': float(score),
                'n_samples': n_group,
            }

        return feeder_scores

    def _compute_classical_stats_for_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute classical stats for [N, window_size] consumption windows."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected [N, {self.window_size}] array for classical stats, got {data.shape}"
            )

        mean_profile = np.mean(data, axis=0)
        std_profile = np.std(data, axis=0)

        return {
            'mean_profile': mean_profile.tolist(),
            'std_profile': std_profile.tolist(),
            'overall_mean': float(np.mean(data)),
            'overall_std': float(np.std(data)),
            'overall_min': float(np.min(data)),
            'overall_max': float(np.max(data)),
            'quantiles': {
                '25%': float(np.percentile(data, 25)),
                '50%': float(np.percentile(data, 50)),
                '75%': float(np.percentile(data, 75)),
            }
        }

    def _compute_subset_classical_stats(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute full classical stats for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            real_stats = self._compute_classical_stats_for_data(real_subset)
            synthetic_stats = self._compute_classical_stats_for_data(synthetic_subset)
            real_mean_profile = np.array(real_stats['mean_profile'])
            synth_mean_profile = np.array(synthetic_stats['mean_profile'])
            real_std_profile = np.array(real_stats['std_profile'])
            synth_std_profile = np.array(synthetic_stats['std_profile'])

            return {
                'n_profiles': int(min(len(real_subset), len(synthetic_subset))),
                'real': real_stats,
                'synthetic': synthetic_stats,
                'differences': {
                    'overall_mean_abs_diff': float(abs(real_stats['overall_mean'] - synthetic_stats['overall_mean'])),
                    'overall_std_abs_diff': float(abs(real_stats['overall_std'] - synthetic_stats['overall_std'])),
                    'mean_profile_mae': float(np.mean(np.abs(real_mean_profile - synth_mean_profile))),
                    'std_profile_mae': float(np.mean(np.abs(real_std_profile - synth_std_profile))),
                }
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: ClassicalStatistics computation failed: {e}")
            return None

    def _plot_aggregated_profiles_bars(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        days_of_week: np.ndarray,
        months_of_year: np.ndarray,
        feeder_ids: np.ndarray = None,
    ) -> Dict[str, str]:
        """Create bar charts of aggregated real vs synthetic consumption by day and month."""
        base_dir = os.path.dirname(self.output_path) if self.output_path and self.output_path.endswith('.json') else self.output_path
        plots_dir = os.path.join(base_dir, 'Plots') if base_dir else 'Plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot style
        plot_style = PlotStyle(figsize=(14, 6))
        color_real = plot_style.palette_colors[0]
        color_synthetic = plot_style.palette_colors[4]
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        # Sum total consumption (all windows + all 24 hours) per day-of-week
        day_labels = []
        day_real_totals = []
        day_synth_totals = []
        day_counts = []
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                day_labels.append(day_names[day])
                day_real_totals.append(float(np.sum(real_data[mask])))
                day_synth_totals.append(float(np.sum(synthetic_data[mask])))
                day_counts.append(int(np.sum(mask)))

        day_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_day_of_week.png')
        if day_labels:
            x = np.arange(len(day_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, day_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, day_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Day of Week ({sum(day_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(day_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(day_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption (all windows + all 24 hours) per month
        month_labels = []
        month_real_totals = []
        month_synth_totals = []
        month_counts = []
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                month_labels.append(month_names[month])
                month_real_totals.append(float(np.sum(real_data[mask])))
                month_synth_totals.append(float(np.sum(synthetic_data[mask])))
                month_counts.append(int(np.sum(mask)))

        month_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_month_of_year.png')
        if month_labels:
            x = np.arange(len(month_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, month_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, month_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Month of Year ({sum(month_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month of Year', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(month_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(month_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption per feeder (if feeder_ids provided)
        feeder_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_feeder.png')
        if feeder_ids is not None:
            unique_feeders = [f for f in np.unique(feeder_ids) if f is not None and str(f).lower() != 'nan']
            
            if unique_feeders:
                feeder_labels = []
                feeder_real_totals = []
                feeder_synth_totals = []
                feeder_counts = []
                for feeder in sorted(unique_feeders):
                    mask = feeder_ids == feeder
                    if np.sum(mask) > 0:
                        feeder_labels.append(str(feeder))
                        feeder_real_totals.append(float(np.sum(real_data[mask])))
                        feeder_synth_totals.append(float(np.sum(synthetic_data[mask])))
                        feeder_counts.append(int(np.sum(mask)))
                
                if feeder_labels:
                    x = np.arange(len(feeder_labels))
                    width = 0.38
                    # fig, ax = plt.subplots(figsize=(max(12, len(feeder_labels)), 6))
                    fig, ax = plt.subplots(figsize=plot_style.figsize)
                    ax.bar(x - width / 2, feeder_real_totals, width, label='Real', 
                           color=color_real, alpha=0.9, linewidth=1.2)
                    ax.bar(x + width / 2, feeder_synth_totals, width, label='Synthetic', 
                           color=color_synthetic, alpha=0.9, linewidth=1.2)
                    ax.set_title(f'Aggregated Consumption by Feeder ({sum(feeder_counts)} profiles)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Feeder ID', fontsize=12)
                    ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(feeder_labels, rotation=45, ha='right', fontsize=10)
                    ax.legend(fontsize=11)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    plt.savefig(feeder_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

        return {
            'day_of_week': day_plot_path,
            'month_of_year': month_plot_path,
            'feeder': feeder_plot_path if feeder_ids is not None else None,
        }

    def compute(self) -> float:
        """Run per-condition MMD, ACFDistribution, and TSTR and save results."""
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("PerConditionEval requires injected real_data and synthetic_data.")
        if self.timestamps is None:
            raise ValueError("PerConditionEval requires injected timestamps for per-condition grouping.")

        self.mmd_calculator.output_path = self.output_path
        self.acf_calculator.output_path = self.output_path
        self.classical_statistics_calculator.output_path = self.output_path
        self.tstr_calculator.output_path = self.output_path
        self.aggregated_profiles_calculator.output_path = self.output_path

        real_data = np.asarray(self.real_data)
        synthetic_data = np.asarray(self.synthetic_data)
        timestamps = np.asarray(self.timestamps)

        if len(real_data) != len(synthetic_data):
            synthetic_data = synthetic_data[:len(real_data)]

        # Ensure lengths match before creating masking arrays
        min_len = min(len(real_data), len(timestamps))
        real_data = real_data[:min_len]
        synthetic_data = synthetic_data[:min_len]
        timestamps = timestamps[:min_len]
        
        # Save plots
        self._plot_aggregated_profiles_bars(
            real_data, synthetic_data, 
            np.array([datetime.utcfromtimestamp(ts).weekday() for ts in timestamps]),
            np.array([datetime.utcfromtimestamp(ts).month for ts in timestamps]),
            self._resolve_feeder_ids(len(real_data))
        )
        if real_data.ndim != 2 or synthetic_data.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real_data.shape}, synthetic={synthetic_data.shape}")
        if real_data.shape[1] != self.window_size or synthetic_data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window shape [N, {self.window_size}], got real={real_data.shape}, synthetic={synthetic_data.shape}"
            )
        if self.verbose:
            self._print_input_debug(real_data, synthetic_data, timestamps)

        days_of_week = []
        months_of_year = []
        for ts in timestamps:
            dt = datetime.utcfromtimestamp(ts)
            days_of_week.append(dt.weekday())
            months_of_year.append(dt.month)
        days_of_week = np.array(days_of_week)
        months_of_year = np.array(months_of_year)

        if self.verbose:
            unique_days, day_counts = np.unique(days_of_week, return_counts=True)
            unique_months, month_counts = np.unique(months_of_year, return_counts=True)
            print("\nCondition counts:")
            print("  day_of_week:", {int(k): int(v) for k, v in zip(unique_days, day_counts)})
            print("  month_of_year:", {int(k): int(v) for k, v in zip(unique_months, month_counts)})

        # Resolve feeder IDs if available
        feeder_ids = self._resolve_feeder_ids(len(real_data))

        aggregated_profile_plot_paths = self._plot_aggregated_profiles_bars(
            real_data=real_data,
            synthetic_data=synthetic_data,
            days_of_week=days_of_week,
            months_of_year=months_of_year,
            feeder_ids=feeder_ids,
        )

        # -------------------------
        # MMD 
        # -------------------------
        mmd_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_day[day] = self.mmd_calculator.compute()
                else:
                    mmd_per_day[day] = 0.0

        mmd_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_month[month] = self.mmd_calculator.compute()
                else:
                    mmd_per_month[month] = 0.0

        # -------------------------
        # ACF Distribution 
        # -------------------------
        acf_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                acf_per_day[day] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        acf_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                acf_per_month[month] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        # -------------------------
        # Classical Statistics
        # -------------------------
        classical_stats_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                classical_stats_per_day[day] = self._compute_subset_classical_stats(
                    real_data[days_of_week == day], synthetic_data[days_of_week == day]
                )

        classical_stats_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                classical_stats_per_month[month] = self._compute_subset_classical_stats(
                    real_data[months_of_year == month], synthetic_data[months_of_year == month]
                )

        # -------------------------
        # Aggregated Profiles (Flexibility Distance)
        # -------------------------
        aggregated_profiles_per_day = {}
        for day in range(7):
            if np.sum( days_of_week == day) > 0:
                aggregated_profiles_per_day[day] = self._compute_subset_aggregated_profiles(
                    real_data[ days_of_week == day], synthetic_data[ days_of_week == day]
                )

        aggregated_profiles_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                aggregated_profiles_per_month[month] = self._compute_subset_aggregated_profiles(
                    real_data[months_of_year == month], synthetic_data[months_of_year == month]
                )

        aggregated_profiles_per_feeder = {}
        feeder_ids = self._resolve_feeder_ids(len(real_data))
        if feeder_ids is not None:
            aggregated_profiles_per_feeder = self._compute_aggregated_profiles_per_feeder(
                real_data=real_data,
                synthetic_data=synthetic_data,
                feeder_ids=feeder_ids,
            )

        # -------------------------
        # TSTR (Train on Synthetic, Test on Real)
        # -------------------------
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        tstr_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                tstr_per_day[day] = self._compute_subset_tstr(
                    real_data[days_of_week == day],
                    synthetic_data[days_of_week == day],
                    group_label=f"day_of_week={day_names[day]} (day {day})"
                )

        tstr_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                tstr_per_month[month] = self._compute_subset_tstr(
                    real_data[months_of_year == month],
                    synthetic_data[months_of_year == month],
                    group_label=f"month_of_year={month_names[month]} (month {month})"
                )


        # -------------------------
        # Save results and print
        # -------------------------
        self.results_by_condition['mmd']['day_of_week'] = mmd_per_day
        self.results_by_condition['mmd']['month_of_year'] = mmd_per_month
        self.results_by_condition['acf']['day_of_week'] = acf_per_day
        self.results_by_condition['acf']['month_of_year'] = acf_per_month
        self.results_by_condition['classical_statistics']['day_of_week'] = classical_stats_per_day
        self.results_by_condition['classical_statistics']['month_of_year'] = classical_stats_per_month
        self.results_by_condition['aggregated_profiles']['day_of_week'] = aggregated_profiles_per_day
        self.results_by_condition['aggregated_profiles']['month_of_year'] = aggregated_profiles_per_month
        self.results_by_condition['aggregated_profiles']['feeder'] = aggregated_profiles_per_feeder
        self.results_by_condition['tstr']['day_of_week'] = tstr_per_day
        self.results_by_condition['tstr']['month_of_year'] = tstr_per_month

        if self.verbose:
            print("\n" + "="*70)
            print("MMD PER DAY OF WEEK (24-HOUR WINDOWS)")
            print("="*70)
            for day in sorted(mmd_per_day.keys()):
                print(f"{day_names[day]:10s} (day {day})  ->  MMD = {mmd_per_day[day]:.6f}")

            print("\n" + "="*70)
            print("MMD PER MONTH OF YEAR (24-HOUR WINDOWS)")
            print("="*70)
            for month in sorted(mmd_per_month.keys()):
                print(f"{month_names[month]:10s} (month {month:2d})  ->  MMD = {mmd_per_month[month]:.6f}")

            print("\n" + "="*70)
            print("ACF DISTRIBUTION PER DAY OF WEEK (KS, lower is better)")
            print("="*70)
            for day in sorted(acf_per_day.keys()):
                value = acf_per_day[day]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{day_names[day]:10s} (day {day})  ->  ACF KS = {value_str}")

            print("\n" + "="*70)
            print("ACF DISTRIBUTION PER MONTH") # (KS, lower is better)
            print("="*70)
            for month in sorted(acf_per_month.keys()):
                value = acf_per_month[month]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{month_names[month]:10s} (month {month:2d})  ->  ACF KS = {value_str}")

            print("\n" + "="*70)
            print("CLASSICAL STATISTICS PER DAY OF WEEK")
            print("="*70)
            for day in sorted(classical_stats_per_day.keys()):
                value = classical_stats_per_day[day]
                if value is None:
                    print(f"{day_names[day]:10s} (day {day})  ->  insufficient samples")
                else:
                    mae = value['differences']['mean_profile_mae']
                    omd = value['differences']['overall_mean_abs_diff']
                    print(f"{day_names[day]:10s} (day {day})  ->  mean_profile_mae = {mae:.6f}, overall_mean_abs_diff = {omd:.6f}")

            print("\n" + "="*70)
            print("CLASSICAL STATISTICS PER MONTH")
            print("="*70)
            for month in sorted(classical_stats_per_month.keys()):
                value = classical_stats_per_month[month]
                if value is None:
                    print(f"{month_names[month]:10s} (month {month:2d})  ->  insufficient samples")
                else:
                    mae = value['differences']['mean_profile_mae']
                    omd = value['differences']['overall_mean_abs_diff']
                    print(f"{month_names[month]:10s} (month {month:2d})  ->  mean_profile_mae = {mae:.6f}, overall_mean_abs_diff = {omd:.6f}")

            print("\n" + "="*70)
            print("AGGREGATED PROFILES PER DAY OF WEEK") # (FD, lower is better)
            print("="*70)
            for day in sorted(aggregated_profiles_per_day.keys()):
                value = aggregated_profiles_per_day[day]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{day_names[day]:10s} (day {day})  ->  Aggregated FD = {value_str}")

            print("\n" + "="*70)
            print("AGGREGATED PROFILES PER MONTH") # (FD, lower is better)
            print("="*70)
            for month in sorted(aggregated_profiles_per_month.keys()):
                value = aggregated_profiles_per_month[month]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{month_names[month]:10s} (month {month:2d})  ->  Aggregated FD = {value_str}")

            print("\n" + "="*70)
            print("AGGREGATED PROFILES PER FEEDER")
            print("="*70)
            if not aggregated_profiles_per_feeder:
                print("  No feeders met minimum sample threshold")
            else:
                sorted_feeders = sorted(
                    aggregated_profiles_per_feeder.items(),
                    key=lambda kv: kv[1]['fd']
                )
                for feeder, details in sorted_feeders[:20]:
                    print(
                        f"{feeder:20s} -> Aggregated FD = {details['fd']:.6f} "
                        f"(n={details['n_samples']})"
                    )

            print("\n" + "="*70)
            print("TSTR PER DAY OF WEEK") # (Ratio, 1.0 = perfect, lower is better)
            print("="*70)
            for day in sorted(tstr_per_day.keys()):
                value = tstr_per_day[day]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{day_names[day]:10s} (day {day})  ->  TSTR Ratio = {value_str}")

            print("\n" + "="*70)
            print("TSTR PER MONTH") # (Ratio, 1.0 = perfect, lower is better)
            print("="*70)
            for month in sorted(tstr_per_month.keys()):
                value = tstr_per_month[month]
                value_str = f"{value:.6f}" if value is not None else "insufficient samples"
                print(f"{month_names[month]:10s} (month {month:2d})  ->  TSTR Ratio = {value_str}")

        # Save file
        combined_output_file = os.path.join(self.output_dir, 'eval_results.json')
        existing_payload = {}
        if os.path.exists(combined_output_file):
            try:
                with open(combined_output_file, 'r') as f:
                    existing_payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_payload = {}

        output_payload = {
            'mmd': self.results_by_condition['mmd'],
            'acf': self.results_by_condition['acf'],
            'classical_statistics': self.results_by_condition['classical_statistics'],
            'aggregated_profiles': self.results_by_condition['aggregated_profiles'],
            'tstr': self.results_by_condition['tstr'],
            'aggregated_profile_plots': aggregated_profile_plot_paths,
            'feeder_mapping_path': self.feeder_mapping_path,
            'min_samples_per_feeder': int(self.min_samples_per_feeder),
            'window_size': int(self.window_size),
            'max_lag': int(self.max_lag),
            'min_samples_per_group': int(self.min_samples_per_group),
            'predictor_type': self.predictor_type,
        }

        # Preserve standalone metric entries written by individual metric classes
        if isinstance(existing_payload, dict) and 'standalone_metrics' in existing_payload:
            output_payload['standalone_metrics'] = existing_payload['standalone_metrics']

        with open(combined_output_file, 'w') as f:
            json.dump(_to_json_serializable(output_payload), f, indent=2)

        if self.verbose:
            print(f"Results saved to: {combined_output_file}")
            print(f"Aggregated profile bar plots saved to: {aggregated_profile_plot_paths['day_of_week']} and {aggregated_profile_plot_paths['month_of_year']}")

        all_mmd_scores = list(mmd_per_day.values()) + list(mmd_per_month.values())
        avg_mmd = np.mean(all_mmd_scores) if all_mmd_scores else 0.0
        return float(avg_mmd)
