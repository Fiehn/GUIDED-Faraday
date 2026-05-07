import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict
import numpy as np


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
    """Save/update standalone metric results in the shared per-condition JSON file.
    Args:
        metric_key (str): Key for the metric.
        payload (Dict[str, Any]): Metric results to save.
        output_file (str): Path to the output JSON file.
    """
    try:
        resolved_output = output_file or 'models/eval_results.json'
        if os.path.isdir(resolved_output) or not str(resolved_output).lower().endswith('.json'):
            resolved_output = os.path.join(resolved_output, 'eval_results.json')

        os.makedirs(os.path.dirname(resolved_output) or '.', exist_ok=True)

        existing = {}
        if os.path.exists(resolved_output):
            try:
                with open(resolved_output, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        if not isinstance(existing, dict):
            existing = {}

        existing.setdefault('standalone_metrics', {})
        existing['standalone_metrics'][metric_key] = _to_json_serializable(payload)

        with open(resolved_output, 'w') as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        # Keep this helper quiet by default to avoid noisy evaluation logs.
        pass


def _flatten_windows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [N, L], got {arr.shape}")
    return arr.reshape(arr.shape[0], -1)


def _resolve_output_directory(output_path: str | None) -> str:
    """Resolve a writable directory from either a directory path or a JSON file path."""
    if not output_path:
        return '.'
    if output_path.endswith('.json'):
        return os.path.dirname(output_path) or '.'
    return output_path


def _sanitize_numeric_array(data: np.ndarray, clip_abs: float = 1e6, fill_value: float = 0.0) -> np.ndarray:
    """Return a finite float64 array with NaN/Inf handled and optional clipping."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return arr
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=fill_value, posinf=clip_abs, neginf=-clip_abs)
    if clip_abs is not None:
        np.clip(arr, -clip_abs, clip_abs, out=arr)
    return arr

def _rank_phases_by_total_consumption(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [N, F, L], got {arr.shape}")

    totals = arr.sum(axis=2)
    order = np.argsort(-totals, axis=1)
    return np.take_along_axis(arr, order[:, :, None], axis=1)


def _align_phases_by_mse(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = _sanitize_numeric_array(reference)
    cand = _sanitize_numeric_array(candidate)
    if ref.ndim != 3 or cand.ndim != 3:
        raise ValueError(f"Expected 3D arrays [N, F, L], got {ref.shape} and {cand.shape}")
    if ref.shape != cand.shape:
        raise ValueError(f"Reference and candidate shapes must match, got {ref.shape} vs {cand.shape}")

    n, f, _ = ref.shape
    if f <= 1:
        return cand

    aligned = np.empty_like(cand)
    for i in range(n):
        # Cost[j, k] = MSE between reference phase j and candidate phase k.
        diff = ref[i][:, None, :] - cand[i][None, :, :]
        cost = np.mean(diff * diff, axis=2)
        if not np.isfinite(cost).all():
            aligned[i] = _rank_phases_by_total_consumption(cand[i:i + 1])[0]
            continue
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
            aligned[i, row_ind, :] = cand[i, col_ind, :]
        except ValueError:
            aligned[i] = _rank_phases_by_total_consumption(cand[i:i + 1])[0]
    return aligned


def _align_phases_to_reference_mean(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = _sanitize_numeric_array(reference)
    cand = _sanitize_numeric_array(candidate)
    if ref.ndim != 3 or cand.ndim != 3:
        raise ValueError(f"Expected 3D arrays [N, F, L], got {ref.shape} and {cand.shape}")
    if ref.shape[1:] != cand.shape[1:]:
        raise ValueError(f"Reference and candidate feature shapes must match, got {ref.shape} vs {cand.shape}")

    _, f, _ = ref.shape
    if f <= 1:
        return cand

    ref_mean = ref.mean(axis=0)
    cand_mean = cand.mean(axis=0)
    diff = ref_mean[:, None, :] - cand_mean[None, :, :]
    cost = np.mean(diff * diff, axis=2)
    if not np.isfinite(cost).all():
        return _align_phases_by_mse(ref, cand)

    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        return _align_phases_by_mse(ref, cand)

    aligned = np.empty_like(cand)
    aligned[:, row_ind, :] = cand[:, col_ind, :]
    return aligned


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
    """Base class for reconstruction evaluation metrics."""
    
    @abstractmethod
    def compute(self) -> float:
        """Compute metric on reconstructions."""
        pass


class SyntheticMetric(EvaluationMetric):
    """Base class for synthetic data evaluation metrics."""
    
    @abstractmethod
    def compute(self) -> float:
        """Compute metric on synthetic samples."""
        pass
