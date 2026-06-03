import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _sanitize_numeric_array


class TSTR(SyntheticMetric):
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.

    Trains an OLS predictor on synthetic windows, tests on real windows, and divides
    the resulting MSE by the TRTR MSE (same predictor trained on real windows).
    A ratio of 1.0 means synthetic trains as well as real; higher is worse.

    Predictor task: predict hour 23 (last timestep) from hours 0-22 of the consumption
    channel. OLS is deterministic — identical data always produces identical results.

    See ADR-0013 for why OLS was chosen over a neural net predictor.

    Args:
        task (str): Prediction task. Only 'next_step' is currently supported.
        window_size (int): Length of time series window (default: 24)
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, task: str = 'next_step', window_size: int = 24):
        self.task = task
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "TSTR (Train on Synthetic, Test on Real)"

    def _prepare_prediction_data(self, data: np.ndarray):
        """
        Prepare data for prediction task.
        For 'next_step': X = hours 0-22, y = hour 23

        Args:
            data (np.ndarray): [N, window_size * n_features]

        Returns:
            Tuple[np.ndarray, np.ndarray]: X = [N, window_size-1], y = [N, 1]
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {data.shape}")

        if data.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Data feature dimension ({data.shape[1]}) must be divisible by window_size ({self.window_size})"
            )

        inferred_n_features = data.shape[1] // self.window_size
        data_reshaped = data.reshape(-1, self.window_size, inferred_n_features)
        data_consumption = data_reshaped[:, :, 0]  # [N, window_size]

        if self.task == 'next_step':
            X = data_consumption[:, :-1]   # [N, window_size-1]
            y = data_consumption[:, -1:]   # [N, 1]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return X, y

    def compute(self) -> float:
        """
        Compute TSTR metric.

        Returns:
            float: Ratio of TSTR MSE to TRTR MSE (closer to 1.0 is better)
        """
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("TSTR requires injected real_data and synthetic_data before compute().")

        real_data = _sanitize_numeric_array(np.asarray(self.real_data))
        synthetic_data = _sanitize_numeric_array(np.asarray(self.synthetic_data))

        if len(real_data) < 2:
            raise ValueError("Need at least 2 real samples for an 80/20 train/test split.")

        split_idx = int(0.8 * len(real_data))
        split_idx = max(1, min(split_idx, len(real_data) - 1))
        real_train = real_data[:split_idx]
        real_test = real_data[split_idx:]

        if len(synthetic_data) > len(real_train):
            synthetic_train = synthetic_data[:len(real_train)]
        else:
            synthetic_train = synthetic_data

        if len(synthetic_train) == 0:
            raise ValueError("Synthetic data is empty after capping.")

        X_real_train, y_real_train = self._prepare_prediction_data(real_train)
        X_real_test,  y_real_test  = self._prepare_prediction_data(real_test)
        X_synth_train, y_synth_train = self._prepare_prediction_data(synthetic_train)

        # TRTR: train on real, test on real
        model_real = LinearRegression().fit(X_real_train, y_real_train)
        y_pred_real = model_real.predict(X_real_test)
        trtr_mse = float(np.mean((y_pred_real - y_real_test) ** 2))

        # TSTR: train on synthetic, test on real
        model_synth = LinearRegression().fit(X_synth_train, y_synth_train)
        y_pred_synth = model_synth.predict(X_real_test)
        tstr_mse = float(np.mean((y_pred_synth - y_real_test) ** 2))

        tstr_ratio = tstr_mse / trtr_mse if trtr_mse > 0 else float('inf')

        _save_standalone_metric_result(
            metric_key='TSTR',
            payload={
                'value': float(tstr_ratio),
                'trtr_mse': float(trtr_mse),
                'tstr_mse': float(tstr_mse),
                'task': self.task,
                'window_size': int(self.window_size),
                'predictor': 'OLS',
                'n_real_total': int(len(real_data)),
                'n_real_train': int(len(real_train)),
                'n_real_test': int(len(real_test)),
                'n_synthetic_train': int(len(synthetic_train)),
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return tstr_ratio
