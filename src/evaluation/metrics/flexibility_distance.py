import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.metrics.base.base import EvaluationMetric, ReconstructionMetric, _sanitize_numeric_array

class FlexibilityDistance(ReconstructionMetric):
    """Flexibility Distance (FD) between real and reconstructed time-series.
    
    Implements the assignment-based FD from FlexibilityDistance2023:
    given two equal-length series, build a cost matrix combining amplitude
    differences and temporal shifts, then solve a linear assignment to obtain
    the minimum reshaping cost.

        Cost matrix C(i,j) = |X_i - Y_j| + |i - j| * tau, where tau is a temporal scaling factor.

        The final FD is the average cost per sample (optionally normalized by series length).
    
        Args:
            tau (str): Temporal scaling factor. Use 'auto' to set tau = data_range / D.
            normalize (bool): If True, divide per-sample cost by D (per-hour average).
    """
    
    real_data: np.ndarray = None
    reconstructed_data: np.ndarray = None
    output_path: str = None

    # Maximum number of windows evaluated per compute() call.
    # The Hungarian assignment is O(L³) per window; capping keeps wall-time
    # predictable regardless of dataset size while the score stays stable
    # above ~500 pairs.
    MAX_SAMPLES: int = 2000

    def __init__(self, tau: str = 'auto', normalize: bool = True):
        self.tau = tau
        self.normalize = normalize
    
    @property
    def name(self) -> str:
        return "Flexibility Distance"
    
    @staticmethod
    def compute_fd_between_series(x: np.ndarray, y: np.ndarray, tau: str = 'auto', normalize: bool = True) -> float:
        """Compute flexibility distance between two time series.
        
        Args:
            x (np.ndarray): First time series [T]
            y (np.ndarray): Second time series [T]
            tau (str): Temporal scaling factor ('auto' or float)
            normalize (bool): If True, divide cost by series length
            
        Returns:
            float: Flexibility distance (lower is better)
        """
        x = _sanitize_numeric_array(x).reshape(-1)
        y = _sanitize_numeric_array(y).reshape(-1)

        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

        series_len = len(x)
        if series_len == 0:
            return 0.0

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
        if not np.isfinite(cost_matrix).all():
            cost_matrix = np.nan_to_num(cost_matrix, nan=1e12, posinf=1e12, neginf=0.0)
        
        # Linear sum assignment (minimum cost)
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = float(cost_matrix[row_ind, col_ind].sum())
        except ValueError:
            total_cost = float(np.mean(np.abs(x - y)))
        
        if normalize and series_len > 0:
            total_cost /= series_len
        
        return total_cost
    
    def compute(self) -> float:
        """Compute flexibility distance metric.
        
        Process:
        2. For each window, build cost matrix and solve linear assignment
        3. Return average FD across windows (optionally normalized per hour)
        
        Returns:
            float: Flexibility Distance (lower is better, 0 = perfect reconstruction)
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

        # Cap to MAX_SAMPLES: Hungarian is O(L³) per window so this keeps
        # wall-time bounded; the score is statistically stable well below this.
        if real_np.shape[0] > self.MAX_SAMPLES:
            real_np = real_np[:self.MAX_SAMPLES]
            recon_np = recon_np[:self.MAX_SAMPLES]

        # Compute FD per sample using assignment-based cost
        num_samples, series_len = real_np.shape
        fd_values = []
        
        for i in range(num_samples):
            x = real_np[i]
            y = recon_np[i]
            fd = self.compute_fd_between_series(x, y, tau=self.tau, normalize=self.normalize)
            fd_values.append(fd)
        
        return float(np.mean(fd_values))