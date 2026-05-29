import numpy as np
import os
import matplotlib.pyplot as plt

from src.evaluation.metrics.base.base import (
    SyntheticMetric,
    _save_standalone_metric_result,
    _resolve_output_directory,
    _sanitize_numeric_array,
)
from datetime import datetime, timezone


class _PeakBase(SyntheticMetric):
    """Shared computation for PeakMagnitude and PeakTiming."""

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    window_size: int | None = None

    def __init__(self):
        super().__init__()
        self.sorted_real_data = None
        self.sorted_synthetic_data = None
        self._min_real = None
        self._min_synth = None

    def _infer_window_size(self, arr: np.ndarray) -> int:
        if self.window_size is not None:
            return self.window_size
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

    def _EMD1(self):
        """RMSE between sorted real and synthetic duration curves, resampled to the same quantile grid."""
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
        """Circular distance between mean peak hours of real and synthetic windows, in timesteps."""
        real_windows = self._to_window_matrix(self.real_data)
        synthetic_windows = self._to_window_matrix(self.synthetic_data)

        if real_windows.size == 0 or synthetic_windows.size == 0:
            return float("nan")

        window_size = int(real_windows.shape[1])
        if window_size <= 0:
            return float("nan")

        real_peaks = np.argmax(real_windows, axis=1)
        synth_peaks = np.argmax(synthetic_windows, axis=1)

        real_angles = 2 * np.pi * real_peaks / window_size
        synthetic_angles = 2 * np.pi * synth_peaks / window_size

        real_mean_angle = np.arctan2(np.mean(np.sin(real_angles)), np.mean(np.cos(real_angles)))
        synthetic_mean_angle = np.arctan2(np.mean(np.sin(synthetic_angles)), np.mean(np.cos(synthetic_angles)))

        angle_diff = np.arctan2(
            np.sin(real_mean_angle - synthetic_mean_angle),
            np.cos(real_mean_angle - synthetic_mean_angle),
        )
        circular_distance = np.abs(angle_diff) * window_size / (2 * np.pi)
        return float(circular_distance)


class PeakMagnitude(_PeakBase):

    @property
    def name(self) -> str:
        return "Peak Magnitude"

    def plot_load_duration_curve(self) -> None:
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return

            n_points = len(self.sorted_real_data)
            x_real = np.linspace(0, 100, len(self.sorted_real_data))
            x_synth = np.linspace(0, 100, len(self.sorted_synthetic_data))

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x_real, self.sorted_real_data, label='Real Data', color='blue', linewidth=2)
            ax.plot(x_synth, self.sorted_synthetic_data, label='Synthetic Data', color='orange', linewidth=2)
            try:
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
        except Exception:
            import traceback
            traceback.print_exc()

    def compute(self) -> float:
        try:
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError("PeakMagnitude requires injected real_data and synthetic_data before compute().")

            real_series = self._prepare_duration_series(self.real_data)
            synth_series = self._prepare_duration_series(self.synthetic_data)

            self.sorted_real_data = np.sort(real_series)[::-1]
            self.sorted_synthetic_data = np.sort(synth_series)[::-1]
            self._min_real = float(real_series.min()) if real_series.size else float("nan")
            self._min_synth = float(synth_series.min()) if synth_series.size else float("nan")

            self.plot_load_duration_curve()
            magnitude = self._EMD1()

            _save_standalone_metric_result(
                metric_key='PeakMagnitude',
                payload={
                    'magnitude_metric': float(magnitude),
                    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                },
                output_file=self.output_path or 'models/eval_results.json'
            )

            return float(magnitude)
        except Exception as e:
            print(f"[ERROR] PeakMagnitude.compute() failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise


class PeakTiming(_PeakBase):

    @property
    def name(self) -> str:
        return "Peak Timing"

    def compute(self) -> float:
        try:
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError("PeakTiming requires injected real_data and synthetic_data before compute().")

            window_size = self._infer_window_size(self.real_data)
            timing_norm = self.timing_analysis()

            _save_standalone_metric_result(
                metric_key='PeakTiming',
                payload={
                    'timing_metric': float(timing_norm),
                    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                },
                output_file=self.output_path or 'models/eval_results.json'
            )

            return float(timing_norm)
        except Exception as e:
            print(f"[ERROR] PeakTiming.compute() failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise