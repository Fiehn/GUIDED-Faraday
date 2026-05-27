import numpy as np
from scipy import stats
from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _sanitize_numeric_array
from datetime import datetime

class ACFDistribution(SyntheticMetric):
    """
    Compare ACF coefficient distributions between real and synthetic time series.
    
    Based on: https://arxiv.org/pdf/2407.11785
    
    For each lag, we compute ACF values across all time series, then compare
    the distributions using Kolmogorov-Smirnov (KS) test. The final metric is the
    average KS statistic across all lags.

    mathical definition:
    1. For each time series, compute ACF values for lags 1 to max_l
    2. For each lag, collect ACF values across all series to get two distributions (real vs synthetic)
    3. Compute KS statistic for each lag: KS_lag = sup_x |F_real(x) - F_synthetic(x)|
    4. Average KS statistic across lags: ACFDistribution = (1/max_lag) sum_{lag=1}^{max_lag} KS_lag
    
    Lower ACFDistribution means the synthetic data has a more similar temporal dependence structure to the real data, with 0 indicating identical ACF distributions.

    Args:
    window_size (int): Length of each time series window (e.g., 24 hours)
    max_lag (int): Maximum lag to compute ACF for (default: 10)
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size=24, max_lag=10):
        self.window_size = window_size
        self.max_lag = max_lag
    
    @property
    def name(self) -> str:
        return "ACF Distribution Match (KS)"
    
    def _compute_acf_matrix(self, ts_matrix: np.ndarray, max_lag: int) -> np.ndarray:
        """Vectorized ACF computation for a batch of time series.

        Args:
            ts_matrix: [N, L] array of N time series each of length L
            max_lag: Maximum lag to compute

        Returns:
            [N, max_lag] array of ACF values clipped to [-1, 1].
            Rows for zero-variance series are all zeros.
        """
        ts_matrix = np.asarray(ts_matrix, dtype=np.float64)
        N, L = ts_matrix.shape

        # Centre each series                          [N, L]
        centered = ts_matrix - ts_matrix.mean(axis=1, keepdims=True)

        # Per-series variance                         [N]
        c0 = np.mean(centered ** 2, axis=1)
        valid = np.isfinite(c0) & (c0 > 1e-12)      # boolean mask [N]

        acf_matrix = np.zeros((N, max_lag), dtype=np.float64)

        for lag in range(1, min(max_lag + 1, L)):
            # dot product for all series at once      [N]
            c_lag = np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
            acf_matrix[valid, lag - 1] = np.clip(
                c_lag[valid] / c0[valid], -1.0, 1.0
            )

        # Guard against any remaining non-finite values
        np.nan_to_num(acf_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return acf_matrix

    def compute(self) -> float:
        """
        Compute ACF distribution matching metric.

        Returns:
            float: Average KS statistic across all lags (lower is better, 0 = perfect match)
        """

        # Reshape data to separate time series
        # Assuming data is [N, window_size * n_features], reshape to [N * n_features, window_size]
        self.real_data = _sanitize_numeric_array(self.real_data)
        self.synthetic_data = _sanitize_numeric_array(self.synthetic_data)

        n_features = self.real_data.shape[1] // self.window_size

        real_reshaped = self.real_data.reshape(-1, self.window_size, n_features)
        synthetic_reshaped = self.synthetic_data.reshape(-1, self.window_size, n_features)

        # Flatten to get all time series across all features
        # Shape: [N * n_features, window_size]
        real_ts = real_reshaped.reshape(-1, self.window_size)
        synthetic_ts = synthetic_reshaped.reshape(-1, self.window_size)

        # Vectorised ACF computation — replaces per-series Python loop
        real_acfs = self._compute_acf_matrix(real_ts, self.max_lag)           # [N*F, max_lag]
        synthetic_acfs = self._compute_acf_matrix(synthetic_ts, self.max_lag) # [M*F, max_lag]
        
        # For each lag, compare distributions using KS test
        ks_statistics = []
        for lag_idx in range(self.max_lag):
            real_lag_dist = real_acfs[:, lag_idx]
            synthetic_lag_dist = synthetic_acfs[:, lag_idx]

            real_lag_dist = real_lag_dist[np.isfinite(real_lag_dist)]
            synthetic_lag_dist = synthetic_lag_dist[np.isfinite(synthetic_lag_dist)]

            if len(real_lag_dist) == 0 or len(synthetic_lag_dist) == 0:
                ks_statistics.append(1.0)
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, _ = stats.ks_2samp(real_lag_dist, synthetic_lag_dist)
            if not np.isfinite(ks_stat):
                ks_stat = 1.0
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
