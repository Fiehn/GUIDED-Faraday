import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics.base.base import (
    SyntheticMetric,
    _resolve_output_directory,
    _save_standalone_metric_result,
    _sanitize_numeric_array,
)


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

    Args:
        task (str): Prediction task - 'next_step' (predict hour 24 from hours 0-23) or 'x_steps_ahead' (predict hour ? from previous hours)
        window_size (int): Length of time series window (default: 24)
        n_features (int): Number of features (default: 1 for consumption)
        predictor_type (str): Type of predictor ('linear', 'mlp', 'lstm')
        epochs (int): Training epochs for predictor
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    timestamps: np.ndarray = None
    
    def __init__(self, task: str = 'next_step', window_size: int = 24, n_features: int = 1, 
                 predictor_type: str = 'linear', epochs: int = 50, random_state: int = 42,
                 horizon: int = 24):
        self.task = task
        self.window_size = window_size
        self.n_features = n_features
        self.predictor_type = predictor_type
        self.epochs = epochs
        self.random_state = random_state
        self.horizon = int(horizon)
        self._has_plotted = False

        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    @property
    def name(self) -> str:
        return "TSTR (Train on Synthetic, Test on Real)"
    
    def _prepare_prediction_data(self, data: np.ndarray):
        """
        Prepare data for prediction task.
        For 'next_step': X = hours 0-22, y = hour 23 
        For x days ahead: X = ?, y = hour ? 
        
        Args:
            data (np.ndarray): [N, window_size * n_features]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:  X = [N, input_features], y = [N, 1]
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
        
        # Support multi-step prediction: predict the last `horizon` timesteps
        # from the preceding values in the window. This requires
        # window_size > horizon so there is at least one input feature.
        if self.task == 'next_step':
            target_len = int(self.horizon)
            if target_len <= 0:
                raise ValueError("horizon must be >= 1")
            if target_len >= self.window_size:
                raise ValueError(
                    f"window_size ({self.window_size}) must be greater than horizon ({target_len})."
                )
            # X: all timesteps before the last `target_len` values
            X = data_consumption[:, : self.window_size - target_len]
            # y: the last `target_len` values in the window
            y = data_consumption[:, self.window_size - target_len :]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return X, y
    
    def _create_predictor(self, input_dim: int):
        """Create a simple predictor model.
        Args:
            input_dim (int): Number of input features
        Returns:
            nn.Module: PyTorch model
        """
        if self.predictor_type == 'linear':
            model = nn.Linear(input_dim, self.horizon)
        elif self.predictor_type == 'mlp':
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.horizon)
            )
        else:
            raise ValueError(f"Unknown predictor type: {self.predictor_type}")
        
        return model
    
    def _train_predictor(self, X: np.ndarray, y: np.ndarray, model: nn.Module):
        """Train predictor model.
        Args:
            X (np.ndarray): Training features [N, input_dim]
            y (np.ndarray): Training targets [N, 1]
            model (nn.Module): PyTorch model to train
        Returns:
            nn.Module: Trained model
        """
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
    
    def _evaluate_predictor(self, X: np.ndarray, y: np.ndarray, model: nn.Module):
        """Evaluate predictor and return MSE.
        Args:
            X (np.ndarray): Test features [N, input_dim]
            y (np.ndarray): Test targets [N, 1]
            model (nn.Module): Trained PyTorch model
        Returns:
            float: Mean Squared Error on test set
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor)
            mse = nn.MSELoss()(y_pred, y_tensor).item()
        
        return mse

    def _format_prediction_date(self, index: int) -> str:
        """Format an available timestamp for the plot title."""
        if self.timestamps is None:
            return "unknown"

        ts = np.asarray(self.timestamps)
        if index < 0 or index >= len(ts):
            return "unknown"

        value = ts[index]
        try:
            if isinstance(value, np.datetime64):
                return np.datetime_as_string(value, unit='D')
            if hasattr(value, 'strftime'):
                return value.strftime('%Y-%m-%d')
            if isinstance(value, (np.integer, int, float)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime('%Y-%m-%d')
            parsed = np.datetime64(value)
            return np.datetime_as_string(parsed, unit='D')
        except Exception:
            return str(value)

    def _plot_prediction_comparison(
        self,
        X_real_test: np.ndarray,
        y_real_test: np.ndarray,
        model_real: nn.Module,
        model_synth: nn.Module,
        tstr_ratio: float,
        split_idx: int,
    ) -> None:
        """Plot the predicted horizon for the real-trained and synthetic-trained models."""
        output_dir = _resolve_output_directory(self.output_path)
        if not output_dir:
            return

        os.makedirs(output_dir, exist_ok=True)

        plot_idx = 0
        x_sample = torch.FloatTensor(X_real_test[plot_idx:plot_idx + 1])
        with torch.no_grad():
            y_real_pred = model_real(x_sample).squeeze(0).cpu().numpy()
            y_synth_pred = model_synth(x_sample).squeeze(0).cpu().numpy()

        y_true = np.asarray(y_real_test[plot_idx]).reshape(-1)
        hours = np.arange(1, len(y_true) + 1)
        date_label = self._format_prediction_date(split_idx + plot_idx)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(hours, y_true, color='black', linestyle='--', linewidth=2, marker='o', label='Real data')
        ax.plot(hours, y_real_pred, color='#1f77b4', linewidth=2.5, marker='o', label='Predicted by real model')
        ax.plot(hours, y_synth_pred, color='#ff7f0e', linewidth=2.5, marker='o', label='Predicted by synthetic model')

        ax.set_title(
            f'TSTR Prediction Comparison | TSTR Ratio: {tstr_ratio:.4f} | Predicted data date: {date_label}',
            fontsize=13,
            fontweight='bold',
        )
        ax.set_xlabel('Hours', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consumption [kWh]', fontsize=12, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_xlim(hours[0] - 0.5, hours[-1] + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'tstr_prediction_comparison.pdf')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
        plt.close(fig)
    
    def compute(self) -> float:
        """
        Compute TSTR metric.
        
        Returns:
            float: Ratio of TSTR performance to TRTR performance (closer to 1.0 is better)
        """
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("TSTR requires injected real_data and synthetic_data before compute().")

        real_data = _sanitize_numeric_array(np.asarray(self.real_data))
        synthetic_data = _sanitize_numeric_array(np.asarray(self.synthetic_data))

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
        model_real = self._create_predictor(input_dim)
        model_real = self._train_predictor(X_real_train, y_real_train, model_real)
        trtr_mse = self._evaluate_predictor(X_real_test, y_real_test, model_real)
        
        # Train model on synthetic data (TSTR - Train Synthetic, Test Real)
        model_synth = self._create_predictor(input_dim)
        model_synth = self._train_predictor(X_synth_train, y_synth_train, model_synth)
        tstr_mse = self._evaluate_predictor(X_real_test, y_real_test, model_synth)
        
        # Compute ratio (ideal: 1.0, higher = synthetic is worse)
        tstr_ratio = tstr_mse / trtr_mse if trtr_mse > 0 else float('inf')

        if not self._has_plotted:
            self._has_plotted = True
            self._plot_prediction_comparison(
                X_real_test=X_real_test,
                y_real_test=y_real_test,
                model_real=model_real,
                model_synth=model_synth,
                tstr_ratio=tstr_ratio,
                split_idx=split_idx,
            )
        
        _save_standalone_metric_result(
            metric_key='TSTR',
            payload={
                'value': float(tstr_ratio),
                'trtr_mse': float(trtr_mse),
                'tstr_mse': float(tstr_mse),
                'horizon': int(self.horizon),
                'task': self.task,
                'window_size': int(self.window_size),
                'predictor_type': self.predictor_type,
                'epochs': int(self.epochs),
                'n_real_total': int(len(real_data)),
                'n_real_train': int(len(real_train)),
                'n_real_test': int(len(real_test)),
                'n_synthetic_train': int(len(synthetic_train)),
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return tstr_ratio