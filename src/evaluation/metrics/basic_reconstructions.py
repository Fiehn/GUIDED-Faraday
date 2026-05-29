
import numpy as np

from src.evaluation.metrics.base.base import ReconstructionMetric, _save_standalone_metric_result
from datetime import datetime, timezone

# ============================================================================
# Reconstruction Evaluation Metrics 
# ============================================================================

class ReconstructionMSE(ReconstructionMetric):
    """Mean squared error on reconstruction.
        mean((real - recon)^2) over all elements. Lower is better.
    """
    
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
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mse


class ReconstructionRMSE(ReconstructionMetric):
    """Root mean squared error on reconstruction.
        sqrt(mean((real - recon)^2)) over all elements. Lower is better.
    """
    
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
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return rmse


class ReconstructionLogLikelihood(ReconstructionMetric):
    """Log-likelihood of reconstruction.
     Higher is better. Uses a simple isotropic Gaussian likelihood approximation:
     x | x_hat ~ N(x_hat, sigma^2 I), where sigma^2 is estimated from residuals.
     """

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
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mean_log_likelihood