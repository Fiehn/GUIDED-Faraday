import numpy as np
import os
import matplotlib.pyplot as plt

from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _resolve_output_directory
from datetime import datetime


class PeakAnalysis(SyntheticMetric):

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self):
        super().__init__()
        self.sorted_real_data = None
        self.sorted_synthetic_data = None

    @property
    def name(self) -> str:
        return "Peak Analysis"
    
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

            if len(self.sorted_real_data) != len(self.sorted_synthetic_data):
                min_len = min(len(self.sorted_real_data), len(self.sorted_synthetic_data))
                self.sorted_real_data = self.sorted_real_data[:min_len]
                self.sorted_synthetic_data = self.sorted_synthetic_data[:min_len]

            # Number of windows used for the plotted curves (after truncation)
            n_used = min(n_real_total, n_synth_total)
            
            # Create the x-axis values (percentage of 24h)
            x = np.linspace(0, 100, len(self.sorted_real_data))
            
            # Plotting the duration curve
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, self.sorted_real_data, label='Real Data', color='blue', linewidth=2)
            ax.plot(x, self.sorted_synthetic_data, label='Synthetic Data', color='orange', linewidth=2)
            # Compute RMSE between the two sorted curves
            try:
                # use the same metric as reported by compute()
                distance = self._EMD1()
            except Exception:
                distance = float(np.sqrt(np.mean((self.sorted_real_data - self.sorted_synthetic_data) ** 2)))

            title = f"Duration Curve — RMSE={distance:.3f} kWh (n_real={n_used}, n_synth={n_used})"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Percentage of 24h (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Load (kWh)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=11)
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

        Although historically this computed the mean absolute difference (EMD-like),
        we now report root-mean-square error between the sorted curves as the
        primary magnitude distance.
        """
        diff = self.sorted_real_data - self.sorted_synthetic_data
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        return rmse

    def timing_analysis(self):
        """ Will evaluate the timing difference of the peaks between the real and synthetic data.
        The timings are cyclical (23:59 is close to 00:00) so we need to use a circular distance metric.
        We convert peaks hours to angles (theta = 2 * pi * hour / 24)
        Then we compute the circular mean for each dataset (vector averaging)
        Then the measure is just the difference.
        """
        # Convert hours to angles
        real_angles = 2 * np.pi * self.real_data.flatten() / 24
        synthetic_angles = 2 * np.pi * self.synthetic_data.flatten() / 24

        # Compute circular mean
        real_mean_angle = np.arctan2(np.mean(np.sin(real_angles)), np.mean(np.cos(real_angles)))
        synthetic_mean_angle = np.arctan2(np.mean(np.sin(synthetic_angles)), np.mean(np.cos(synthetic_angles)))

        # Convert to hours
        real_mean_hour = (real_mean_angle * 24) / (2 * np.pi)
        synthetic_mean_hour = (synthetic_mean_angle * 24) / (2 * np.pi)
        # Compute distance
        circular_distance = np.abs(real_mean_hour - synthetic_mean_hour)
        return circular_distance

    def compute(self) -> float:
        try:
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError("PeakAnalysis requires injected real_data and synthetic_data before compute().")
            
            self.sorted_real_data = np.sort(self.real_data.flatten())[::-1]
            self.sorted_synthetic_data = np.sort(self.synthetic_data.flatten())[::-1]

            self.plot_load_duration_curve()
            magnitude_metric = self._EMD1()
            timing_metric = self.timing_analysis()
            
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