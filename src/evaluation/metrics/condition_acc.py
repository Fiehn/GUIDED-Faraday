import numpy as np
from typing import Any

from src.evaluation.metrics.base.base import SyntheticMetric


class EVConditionAccuracy(SyntheticMetric):
    """Evaluate if synthetic profiles correctly represent their generation EV condition.

    This metric:
    1. Reads the conditional tensor used for generation mapping to [No EV, 1ph, 2ph, 3ph]
    2. Modifies scale (kW -> W) if needed so thresholds match original Jupyter notebooks
    3. Runs detect_platos_unified_dP2_std on each profile
    4. Computes accuracy of correct EV classification.
    """

    # Per-call sample cap. Accuracy is stable for a 4-class problem well below
    # this count; the detection function is a pure Python loop so O(N) cost is high.
    MAX_SAMPLES: int = 4000

    synthetic_data: np.ndarray = None
    synthetic_conditions: Any = None

    def __init__(self, window_size: int = 24, interval_minutes: int = 60):
        self.window_size = window_size
        self.interval_minutes = interval_minutes
        
    @property
    def name(self) -> str:
        return "EV Condition Accuracy"
        
    def _unpack_synthetic_profile(self, flat_profile: np.ndarray) -> np.ndarray:
        n_features = len(flat_profile) // self.window_size
        # Try to dynamically fetch Phase first property if model exists
        phase_first = False
        if hasattr(self, 'model') and self.model is not None:
            phase_first = getattr(self.model, '_to_phase_first', False)
            
        if phase_first:
            reshaped = flat_profile.reshape(n_features, self.window_size)
            return reshaped.T 
        else:
            return flat_profile.reshape(self.window_size, n_features)
            
    def compute(self) -> float:
        import pandas as pd
        import torch
        from src.evaluation.ev_detection import detect_platos_unified_dP2_std
        
        if self.synthetic_data is None or self.synthetic_conditions is None:
            return 0.0
            
        if isinstance(self.synthetic_conditions, torch.Tensor):
            conds = self.synthetic_conditions.cpu().numpy()
        else:
            conds = np.asarray(self.synthetic_conditions)
            
        n_samples = min(len(self.synthetic_data), len(conds), self.MAX_SAMPLES)
        if n_samples == 0:
            return 0.0

        scale_factor = 1.0
        if np.max(self.synthetic_data[:100]) < 100.0:
            scale_factor = 1000.0

        def process_sample(i):
            if len(conds[i]) == 0:
                expected_label = 'no'
            else:
                cond_idx = int(np.argmax(conds[i]))
                if cond_idx == 0: expected_label = 'no'
                elif cond_idx == 1: expected_label = '1ph'
                elif cond_idx == 2: expected_label = '2ph'
                elif cond_idx == 3: expected_label = '3ph'
                else: expected_label = 'no'

            try:
                phase_data = self._unpack_synthetic_profile(self.synthetic_data[i]) * scale_factor
                dates = pd.date_range(start='2024-01-01', periods=len(phase_data), freq=f"{self.interval_minutes}min")
                df_profile = pd.DataFrame({
                    "active_power_p14_l1_test": phase_data[:, 0] if phase_data.shape[1] > 0 else 0,
                    "active_power_p14_l2_test": phase_data[:, 1] if phase_data.shape[1] > 1 else 0,
                    "active_power_p14_l3_test": phase_data[:, 2] if phase_data.shape[1] > 2 else 0,
                }, index=dates)

                events = detect_platos_unified_dP2_std(
                    df_profile, "test", start_threshold_min=3000, start_threshold_max=7500, interval_minutes=self.interval_minutes
                )
                detected_label = str(events.iloc[0]['phase_type']).lower() if events is not None and not events.empty else 'no'
            except Exception:
                detected_label = 'no'

            if phase_data.shape[1] == 1:
                return (expected_label in ['1ph', '2ph', '3ph']) == (detected_label != 'no')
            return detected_label == expected_label

        correct_count = 0
        
        for i in range(n_samples):
            if i >= n_samples:
                break
            try:
                if process_sample(i):
                    correct_count += 1
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
        return float(correct_count) / float(n_samples)
