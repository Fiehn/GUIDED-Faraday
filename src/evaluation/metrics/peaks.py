import numpy as np
import os
import matplotlib.pyplot as plt

from src.evaluation.metrics.base.base import (
    SyntheticMetric,
    _save_standalone_metric_result,
    _resolve_output_directory,
    _sanitize_numeric_array,
)
from datetime import datetime


class PeakAnalysis(SyntheticMetric):

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self):
        super().__init__()
        self.sorted_real_data = None
        self.sorted_synthetic_data = None
        self._min_real = None
        self._min_synth = None

    @property
    def name(self) -> str:
        return "Peak Analysis"

    def _infer_window_size(self, arr: np.ndarray) -> int:
        if getattr(self, "window_size", None):
            try:
                return int(self.window_size)
            except (TypeError, ValueError):
                pass
        if arr.ndim == 3:
            return int(arr.shape[-1])
        if arr.ndim == 2:
            if arr.shape[1] % 24 == 0:
                return 24
            return int(arr.shape[1])
        return 1

    def _to_window_matrix(self, data: np.ndarray) -> np.ndarray:
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim == 3:
            # [N, F, L] -> aggregate across features for timing
            return arr.sum(axis=1)
        if arr.ndim == 2:
            window_size = self._infer_window_size(arr)
            if window_size > 0 and arr.shape[1] % window_size == 0:
                n_feat = arr.shape[1] // window_size
                if n_feat > 1:
                    return arr.reshape(-1, n_feat, window_size).sum(axis=1)
                return arr.reshape(-1, window_size)
            return arr
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr.reshape(arr.shape[0], -1)

    def _prepare_duration_series(self, data: np.ndarray) -> np.ndarray:
        windows = self._to_window_matrix(data)
        return windows.reshape(-1)
    
    def plot_load_duration_curve(self) -> None:
        """Plot and save load duration curve."""
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return
            
            # Record total available windows (number of rows) before truncation
            def _n_windows(arr):
                if arr is None:
                    return 0
                a = np.asarray(arr)
                if a.ndim == 0:
                    return 1
                return int(a.shape[0])

            n_real_total = _n_windows(self.real_data)
            n_synth_total = _n_windows(self.synthetic_data)

            # if len(self.sorted_real_data) != len(self.sorted_synthetic_data):
            #     min_len = min(len(self.sorted_real_data), len(self.sorted_synthetic_data))
            #     self.sorted_real_data = self.sorted_real_data[:min_len]
            #     self.sorted_synthetic_data = self.sorted_synthetic_data[:min_len]

            # Number of points used for the plotted curves (after truncation)
            n_points = len(self.sorted_real_data)
            
            # Create the x-axis values (percentage of 24h)
            x_real = np.linspace(0, 100, len(self.sorted_real_data))
            x_synth = np.linspace(0, 100, len(self.sorted_synthetic_data))

            # Plotting the duration curve
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x_real, self.sorted_real_data, label='Real Data', color='blue', linewidth=2)
            ax.plot(x_synth, self.sorted_synthetic_data, label='Synthetic Data', color='orange', linewidth=2)
            # Compute RMSE between the two sorted curves
            try:
                # use the same metric as reported by compute()
                distance = self._EMD1()
            except Exception:
                distance = float(np.sqrt(np.mean((self.sorted_real_data - self.sorted_synthetic_data) ** 2)))

            min_real = float(self._min_real) if self._min_real is not None else float("nan")
            min_synth = float(self._min_synth) if self._min_synth is not None else float("nan")
            title = (
                f"Duration Curve — RMSE={distance:.3f} kWh "
                f"(points={n_points}, min_real={min_real:.3f}, min_synth={min_synth:.3f})"
            )
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Percentage of 24h (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Load (kWh)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'duration_curve.pdf')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
            plt.close(fig)
        except Exception as e:
            import traceback
            traceback.print_exc()
        
    def _EMD1(self):
        """Return RMSE between the sorted real and synthetic curves.

        Both curves are treated as duration functions over [0, 1] and resampled
        to the same quantile grid so arrays of different lengths can be compared.
        """
        assert self.sorted_real_data is not None and self.sorted_synthetic_data is not None
        real = self.sorted_real_data
        synth = self.sorted_synthetic_data

        n_real, n_synth = len(real), len(synth)
        if n_real == n_synth:
            diff = real - synth
        else:
            n_out = min(n_real, n_synth)
            q = np.linspace(0, 1, n_out)
            real_resampled = np.interp(q, np.linspace(0, 1, n_real), real)
            synth_resampled = np.interp(q, np.linspace(0, 1, n_synth), synth)
            diff = real_resampled - synth_resampled

        return float(np.sqrt(np.mean(np.square(diff))))

    def timing_analysis(self):
        """ Will evaluate the timing difference of the peaks between the real and synthetic data.
        The timings are cyclical (23:59 is close to 00:00) so we need to use a circular distance metric.
        We convert peaks hours to angles (theta = 2 * pi * hour / 24)
        Then we compute the circular mean for each dataset (vector averaging)
        Then the measure is just the difference.
        """
        real_windows = self._to_window_matrix(self.real_data)
        synthetic_windows = self._to_window_matrix(self.synthetic_data)

        if real_windows.size == 0 or synthetic_windows.size == 0:
            return float("nan")

        window_size = int(real_windows.shape[1])
        if window_size <= 0:
            return float("nan")

        real_peaks = np.argmax(real_windows, axis=1)
        synth_peaks = np.argmax(synthetic_windows, axis=1)

        # Convert hours to angles
        real_angles = 2 * np.pi * real_peaks / window_size
        synthetic_angles = 2 * np.pi * synth_peaks / window_size

        # Compute circular mean
        real_mean_angle = np.arctan2(np.mean(np.sin(real_angles)), np.mean(np.cos(real_angles)))
        synthetic_mean_angle = np.arctan2(np.mean(np.sin(synthetic_angles)), np.mean(np.cos(synthetic_angles)))

        # Convert to hours
        real_mean_hour = (real_mean_angle * 24) / (2 * np.pi)
        synthetic_mean_hour = (synthetic_mean_angle * 24) / (2 * np.pi)
        # Compute distance
        angle_diff = np.arctan2(
            np.sin(real_mean_angle - synthetic_mean_angle),
            np.cos(real_mean_angle - synthetic_mean_angle),
        )
        circular_distance = np.abs(angle_diff) * window_size / (2 * np.pi)
        return float(circular_distance)

    def compute(self) -> float:
        try:
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError("PeakAnalysis requires injected real_data and synthetic_data before compute().")
            
            real_series = self._prepare_duration_series(self.real_data)
            synth_series = self._prepare_duration_series(self.synthetic_data)

            self.sorted_real_data = np.sort(real_series)[::-1]
            self.sorted_synthetic_data = np.sort(synth_series)[::-1]
            self._min_real = float(real_series.min()) if real_series.size else float("nan")
            self._min_synth = float(synth_series.min()) if synth_series.size else float("nan")

            self.plot_load_duration_curve()
            magnitude_metric = self._EMD1()
            timing_metric = self.timing_analysis()

            window_size = self._infer_window_size(self.real_data)

            timing_metric = timing_metric / (window_size / 2) 
            
            # Save individual metrics
            _save_standalone_metric_result(
                metric_key='PeakAnalysis',
                payload={
                    'magnitude_metric': float(magnitude_metric),
                    'timing_metric': float(timing_metric),
                    'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
                },
                output_file=self.output_path or 'models/eval_results.json'
            )
            
            # Return average of both metrics as the main metric value
            return float(np.mean([magnitude_metric, timing_metric]))
        except Exception as e:
            print(f"[ERROR] PeakAnalysis.compute() failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  # Re-raise so framework sees the error