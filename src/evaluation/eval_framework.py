import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
from abc import ABC, abstractmethod
 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import stats 

from src.visualization.plot_styling import PlotStyle
from src.model.generators import SyntheticGenerator

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not installed: render a simple ASCII grid table.
    def tabulate(data, headers="keys", **kwargs):
        rows = [list(r) for r in data]
        if isinstance(headers, (list, tuple)):
            header_row = [str(h) for h in headers]
        else:
            header_row = []

        str_rows = [["" if c is None else str(c) for c in r] for r in rows]
        if header_row:
            str_rows = [header_row] + str_rows

        if not str_rows:
            return ""

        n_cols = max(len(r) for r in str_rows)
        norm_rows = [r + [""] * (n_cols - len(r)) for r in str_rows]
        col_widths = [max(len(r[c]) for r in norm_rows) for c in range(n_cols)]

        def _hline() -> str:
            return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        def _row(cells) -> str:
            return "| " + " | ".join(cells[c].ljust(col_widths[c]) for c in range(n_cols)) + " |"

        out = [_hline()]
        start_idx = 0
        if header_row:
            out.append(_row(norm_rows[0]))
            out.append(_hline())
            start_idx = 1

        for r in norm_rows[start_idx:]:
            out.append(_row(r))
        out.append(_hline())
        return "\n".join(out)


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


class EvalFramework:
    """
    Evaluation framework that runs multiple evaluation metrics and displays results in a table.
    Will handle all necessary data preparation (reconstructions, synthetic samples) and inject them 
    into metrics that need them, so each metric can focus on its specific computation logic.
    
    Usage:
        evaluator = EvalFramework(model, dataloader, normalizer, device='cpu')
        evaluator.register_metric(MyMetric(model, dataloader))
        evaluator.register_metric(AnotherMetric(model, dataloader))
        results = evaluator.evaluate()
    
    Args:
        model (nn.Module): Trained model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        normalizer: Normalizer object (if needed by metrics)
        conditioner: Conditioner object (for CVAE, if needed by metrics)
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        window_size (int): Window size for reconstruction metrics that need it
        n_synthetic_samples (int): Number of synthetic samples to generate for synthetic metrics
        output_path (str): Path to save JSON results for standalone metrics

    Attributes:
        model (nn.Module): The model being evaluated
        dataloader (DataLoader): DataLoader for evaluation data
        normalizer: Normalizer object (if needed by metrics)
        conditioner: Conditioner object (for CVAE, if needed by metrics)
        device (str): Device to run evaluation on
        window_size (int): Window size for reconstruction metrics that need it
        n_synthetic_samples (int): Number of synthetic samples to generate for synthetic metrics
        output_path (str): Path to save JSON results for standalone metrics
        reconstruction_metrics (Dict[str, ReconstructionMetric]): Registered reconstruction metrics
        synthetic_metrics (Dict[str, SyntheticMetric]): Registered synthetic data metrics
        results (Dict[str, float]): Cached results from last evaluation
        _real_data (np.ndarray): Cached real data for metrics that need it
        _reconstructed_data (np.ndarray): Cached reconstructed data for metrics that need it
        _synthetic_data (np.ndarray): Cached synthetic data for metrics that need it
        _timestamps (np.ndarray): Cached timestamps for metrics that need it

    """
    
    def __init__(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        normalizer = None,
        conditioner = None, 
        device: str = 'cpu',
        window_size: int = 24,
        n_synthetic_samples: int = 1000,
        output_path: str = 'models/eval_results.json',
        prepared_data: Dict[str, Any] = None,
        evaluation_title: str = 'EVALUATION RESULTS',
        verbose: bool = False,
        include_mean_baseline: bool = True,
        baseline_reference_data: np.ndarray = None,
        secondary_model: nn.Module = None,
    ):
        """Initialize evaluation framework.
        
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation data
            normalizer: Data normalizer 
            conditioner: Conditioner object (for CVAE)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.conditioner = conditioner
        self.device = device

        self.generator = SyntheticGenerator(model, conditioner, normalizer, secondary_model, device)

        self.window_size = window_size
        self.n_synthetic_samples = n_synthetic_samples
        self.output_path = output_path
        self.prepared_data = prepared_data
        self.evaluation_title = evaluation_title
        self.verbose = bool(verbose)
        self.include_mean_baseline = bool(include_mean_baseline)
        self.baseline_reference_data = baseline_reference_data
        if self.baseline_reference_data is None and isinstance(prepared_data, dict):
            # Prefer explicit training reference data when provided by caller.
            train_ref = prepared_data.get('train_data', None)
            if train_ref is not None:
                self.baseline_reference_data = train_ref
        
        # Metrics storage 
        self.reconstruction_metrics: Dict[str, ReconstructionMetric] = {}
        self.synthetic_metrics: Dict[str, SyntheticMetric] = {}
        self.results: Dict[str, float] = {}
        self.baseline_results: Dict[str, float] = {}

        self._real_data: np.ndarray = None  # Cache real data for synthetic metrics that need it
        self._reconstructed_data: np.ndarray = None  # Cache reconstructed data for metrics that need it
        self._synthetic_data: np.ndarray = None  # Cache synthetic data for metrics that need it
        self._timestamps: np.ndarray = None  # Cache timestamps for metrics that need it
        self._meter_ids: np.ndarray = None

    def register_all(self, phased: bool = False) -> None:
        """Convenience method to register all default metrics."""
        self.register_metrics([
            ReconstructionRMSE(),
            ReconstructionLogLikelihood(),
            FlexibilityDistance(),
            ACFDistribution(),
            ClassicalStatistics(),
            TSTR(),
            AggregatedProfiles(),
            HourlyProfilesByMonth(),
            HourlyProfilesByDayofWeek(),
            PerConditionEval()
        ])
        if phased:
            self.register_metric(SetMaximumMeanDiscrepancy())
        else:
            self.register_metric(MaximumMeanDiscrepancy())

    def _is_cvae(self) -> bool:
        """Check if model is a CVAE based on presence of conditioner on the model."""
        return hasattr(self.model, 'conditioner') and self.model.conditioner is not None

    def _prepare_condition_input(self, x: torch.Tensor) -> torch.Tensor:
        """Match training input layout for conditioners when available."""
        if hasattr(self.model, "_to_phase_first") and hasattr(self.model, "_build_condition_input"):
            try:
                x_phase = self.model._to_phase_first(x)
                return self.model._build_condition_input(x_phase)
            except Exception:
                return x
        return x
    def _unpack_batch(self, batch):
        """Strict unpack contract: (x, meter_id, timestamps, agg)."""
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            raise ValueError(
                "Expected batch as a 4-tuple: (x, meter_id, timestamps, agg)."
            )

        x, meter_id, timestamps, agg = batch

        if torch.is_tensor(meter_id):
            meter_id = meter_id.long()
        if torch.is_tensor(timestamps):
            timestamps = timestamps.long()

        return x, meter_id, timestamps, agg
    def _flatten_for_metrics(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten in the same order used during training when possible."""
        if x.dim() == 3:
            if hasattr(self.model, "_to_phase_first"):
                try:
                    x_phase = self.model._to_phase_first(x)
                    return x_phase.reshape(x_phase.shape[0], -1)
                except Exception:
                    return x.reshape(x.shape[0], -1)
            return x.reshape(x.shape[0], -1)
        return x

    def _normalizer_feature_dim(self) -> int | None:
        """Infer feature dimension expected by the normalizer stats."""
        if self.normalizer is None:
            return None
        stats = getattr(self.normalizer, 'mean', None)
        if stats is None:
            return None
        if isinstance(stats, torch.Tensor):
            if stats.numel() == 0:
                return None
            return int(stats.reshape(-1).shape[0])
        if isinstance(stats, np.ndarray):
            if stats.size == 0:
                return None
            return int(stats.reshape(-1).shape[0])
        try:
            return int(np.asarray(stats).reshape(-1).shape[0])
        except Exception:
            return None

    def _inverse_transform_for_metrics(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform 2D/3D tensors while preserving [B, L, F] or [B, F, L] layout."""
        if self.normalizer is None or not hasattr(self.normalizer, 'inverse_transform'):
            return x

        inv = self.normalizer.inverse_transform
        
        # 1. Handle 2D tensors [B, D]
        if x.dim() == 2:
            # For 2D tensors, we need to know if they are already flattened.
            # If so, they must be unflattened to apply normalization correctly to phases.
            seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
            n_feat = int(getattr(self.model, 'num_features', 0) or 0)
            
            if seq_len > 0 and n_feat > 0 and x.shape[1] == seq_len * n_feat:
                # Assume time-first flattening for unflattening [B, L, F]
                # (This is consistent with how we handle recon in _reconstruct)
                x_3d = x.reshape(-1, seq_len, n_feat)
                y_3d = self._inverse_transform_for_metrics(x_3d)
                return y_3d.reshape_as(x)
            
            # Fallback for other 2D tensors
            return inv(x)

        # 2. Handle 3D tensors
        if x.dim() != 3:
            return inv(x)

        feature_dim = self._normalizer_feature_dim()
        if feature_dim is None:
             return inv(x)

        # Case [B, L, F]: Features are the last dimension
        if x.shape[-1] == feature_dim and x.shape[1] != feature_dim:
            return inv(x.reshape(-1, feature_dim)).reshape_as(x)

        # Case [B, F, L]: Features are the second dimension (phased dataloader output)
        if x.shape[1] == feature_dim:
            x_last = x.movedim(1, -1).contiguous()
            y_last = inv(x_last.reshape(-1, feature_dim)).reshape_as(x_last)
            return y_last.movedim(-1, 1).contiguous()

        # Fallback: let the normalizer handle it or assume last axis
        return inv(x)
    
    def _prepare_data(self) -> None:
        """Extract real data, reconstructed data, and synthetic data for metrics that need it."""
        if self.prepared_data is not None:
            self._real_data = _sanitize_numeric_array(np.asarray(self.prepared_data.get('real_data')))
            self._reconstructed_data = _sanitize_numeric_array(np.asarray(self.prepared_data.get('reconstructed_data')))
            self._synthetic_data = _sanitize_numeric_array(np.asarray(self.prepared_data.get('synthetic_data')))
            self._timestamps = self.prepared_data.get('timestamps')
            self._meter_ids = self.prepared_data.get('meter_ids')

            return

        if self.verbose:
            print("Preparing data for evaluation metrics...")
        real_batches = []
        recon_batches = []
        condition_batches = []
        meter_id_batches = []
        timestamp_batches = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                data, meter_ids, timestamps, agg = self._unpack_batch(batch)
                
                data = data.to(self.device)
                if self._is_cvae():
                    cond_input = self._prepare_condition_input(data)
                    
                    c = self.conditioner.get_conditions(cond_input, meter_id=meter_ids, timestamps=timestamps)
                    if isinstance(c, np.ndarray):
                        c = torch.tensor(c, dtype=torch.float32)
                    c = c.to(self.device)
                    condition_batches.append(c.cpu())

                # Denormalize with robust axis handling for [B, L, F] and [B, F, L].
                data_denorm = self._inverse_transform_for_metrics(data)
                
                x_flat = self._flatten_for_metrics(data_denorm)
                real_batches.append(x_flat.cpu().numpy())

                if timestamps is not None:
                    if isinstance(timestamps, torch.Tensor):
                        timestamp_batches.append(np.asarray(timestamps.detach().cpu().numpy()).reshape(-1))
                    else:
                        timestamp_batches.append(np.asarray(timestamps).reshape(-1))
                if meter_ids is not None:
                    if isinstance(meter_ids, torch.Tensor):
                        meter_id_batches.append(np.asarray(meter_ids.detach().cpu().numpy()).reshape(-1))
                    else:
                        meter_id_batches.append(np.asarray(meter_ids).reshape(-1))

                recon = self._reconstruct(data, meter_id=meter_ids, timestamps=timestamps)
                recon_batches.append(recon.cpu().numpy())

        self._real_data = _sanitize_numeric_array(np.concatenate(real_batches, axis=0))
        self._reconstructed_data = _sanitize_numeric_array(np.concatenate(recon_batches, axis=0))
        self._meter_ids = np.concatenate(meter_id_batches, axis=0) if meter_id_batches else None
        self._timestamps = np.concatenate(timestamp_batches, axis=0) if timestamp_batches else None

        if self._is_cvae() and condition_batches:
            all_conditions = torch.cat(condition_batches, dim=0)
            n = self.n_synthetic_samples
            if len(all_conditions) >= n:
                idx = torch.randperm(len(all_conditions))[:n]
                self._synthetic_conditions = all_conditions[idx]
            else:
                repeats = (n // len(all_conditions)) + 1
                self._synthetic_conditions = all_conditions.repeat(repeats, 1)[:n]
        else:
            self._synthetic_conditions = None

        self._synthetic_data = _sanitize_numeric_array(self._generate_synthetic(self.n_synthetic_samples))

    
    def _reconstruct(self, x: torch.Tensor, 
                     meter_id: torch.Tensor | int | None = None, 
                     timestamps: torch.Tensor | None=None) -> torch.Tensor:
        """Reconstruct via encoder-decoder, using real conditions for CVAE.
        Args:
            x (torch.Tensor): Input data batch `(bathc_size, seq_len, features)` or `(batch_size, features)`
            meter_id (torch.Tensor | int | None): Meter IDs for the batch (for CVAE conditioning)
            timestamps (torch.Tensor | None): Timestamps for the batch (for CVAE conditioning)
            
        Returns:
            torch.Tensor: Reconstructed data of shape `(batch_size, features)`
        """
        recon = self.generator.reconstruct(x, meter_id=meter_id, timestamp=timestamps)
        if recon is None:
            raise ValueError("Generator returned None for reconstruction")

        # Expected from generator: [S, B, D]. Keep one reconstruction sample for metrics.
        if recon.dim() == 3:
            recon = recon[0]

        # If still time/feature shaped [B, L, F], flatten to [B, D].
        if recon.dim() == 3:
            recon = self._flatten_for_metrics(recon)
        elif recon.dim() == 2:
            # For 2D tensors, if they are time-first flattened, re-flatten to phase-first
            seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
            n_feat = int(getattr(self.model, 'num_features', 0) or 0)
            if seq_len > 0 and n_feat > 0 and recon.shape[1] == seq_len * n_feat:
                recon_3d = recon.reshape(recon.shape[0], seq_len, n_feat)
                recon = self._flatten_for_metrics(recon_3d)
        elif recon.dim() != 2:
            raise ValueError(f"Unexpected reconstruction shape: {tuple(recon.shape)}")

        # For models with known [L, F], enforce the same phase-first flattening
        # convention used for real data so phased extraction is consistent.
        seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
        n_feat = int(getattr(self.model, 'num_features', 0) or 0)
        if recon.dim() == 2 and seq_len > 0 and n_feat > 0 and recon.shape[1] == seq_len * n_feat:
            recon_lf = recon.reshape(recon.shape[0], seq_len, n_feat)
            recon = self._flatten_for_metrics(recon_lf)

        return recon  # [B, D]
    
    def _generate_synthetic(self, n_samples: int) -> np.ndarray:
        """Sample from prior using real conditions to match the real data distribution.
         For CVAE, conditions are sampled from the real conditions collected during reconstruction preparation.
         For non-CVAE, conditions are not used.
         Args:
            n_samples (int): Number of synthetic samples to generate
         Returns:
             np.ndarray: Synthetic samples of shape (n_samples, features)
        """
        if self._is_cvae():
            if self._synthetic_conditions is None:
                raise ValueError("CVAE synthetic generation requires prepared conditions, but none were found.")
            conditions = self._synthetic_conditions.to(self.device)
            synthetic, _ = self.generator.generate(n_samples, conditions=conditions)
        else:
            synthetic, _ = self.generator.generate(n_samples)

        if synthetic is None:
            raise ValueError("Generator returned None for synthetic samples")

        # Flatten samples and batches into a single sample dimension [Samples, Features]
        if synthetic.dim() == 4:
             # Handle [N, B, L, F] or [N, B, F, L]
             N, B, S1, S2 = synthetic.shape
             synthetic = synthetic.reshape(N * B, S1, S2)
             synthetic = self._flatten_for_metrics(synthetic)
        elif synthetic.dim() == 3:
             # Check if it is [N, L, F], [N, F, L], or [N, B, D]
             n_feat = int(getattr(self.model, 'num_features', 0) or 0)
             seq_len = int(getattr(self.model, 'sequence_length', 0) or self.window_size or 0)

             if seq_len > 0 and synthetic.shape[1] == seq_len:
                 # [N, L, F] -> [N, F, L]
                 synthetic = synthetic.transpose(1, 2).reshape(synthetic.shape[0], -1)
             elif seq_len > 0 and synthetic.shape[2] == seq_len:
                 # [N, F, L]
                 synthetic = synthetic.reshape(synthetic.shape[0], -1)
             else:
                 # Assume [N, B, D]. Flatten to [N*B, D].
                 synthetic = synthetic.reshape(-1, synthetic.shape[-1])
                 # If the resulting 2D is time-first, re-flatten to phase-first
                 if n_feat > 0 and seq_len > 0 and synthetic.shape[1] == seq_len * n_feat:
                     synth_3d = synthetic.reshape(-1, seq_len, n_feat)
                     synthetic = self._flatten_for_metrics(synth_3d)

        res = synthetic.cpu().numpy()
        return res

    def _inject_data_into_metrics(self) -> None:
        """Inject prepared data into metrics that need it."""
        for metric in self.reconstruction_metrics.values():
            metric.real_data = self._real_data
            metric.reconstructed_data = self._reconstructed_data
            metric.model = self.model
            metric.device = self.device
            metric.normalizer = self.normalizer
            metric.output_path = self.output_path
            metric.timestamps = self._timestamps

        for metric in self.synthetic_metrics.values():
            metric.real_data = self._real_data
            metric.synthetic_data = self._synthetic_data
            metric.model = self.model
            metric.device = self.device
            metric.normalizer = self.normalizer
            metric.output_path = self.output_path
            metric.timestamps = self._timestamps
            metric.meter_ids = self._meter_ids
            metric.synthetic_conditions = getattr(self, '_synthetic_conditions', None)
            
            # PerConditionEval needs aligned timestamps and 24h windows (consumption feature only)
            if isinstance(metric, PerConditionEval):
                n = min(len(self._real_data), len(self._synthetic_data))
                real = np.asarray(self._real_data)[:n]
                synth = np.asarray(self._synthetic_data)[:n]
                phase_first_flatten = hasattr(self.model, '_to_phase_first')

                # Convert flattened [N, L*F] -> [N, L] by selecting feature 0
                if real.ndim == 2 and real.shape[1] != metric.window_size and real.shape[1] % metric.window_size == 0:
                    f_real = real.shape[1] // metric.window_size
                    if phase_first_flatten:
                        real = real.reshape(-1, f_real, metric.window_size)[:, 0, :]
                    else:
                        real = real.reshape(-1, metric.window_size, f_real)[:, :, 0]
                if synth.ndim == 2 and synth.shape[1] != metric.window_size and synth.shape[1] % metric.window_size == 0:
                    f_synth = synth.shape[1] // metric.window_size
                    if phase_first_flatten:
                        synth = synth.reshape(-1, f_synth, metric.window_size)[:, 0, :]
                    else:
                        synth = synth.reshape(-1, metric.window_size, f_synth)[:, :, 0]

                metric.real_data = real
                metric.synthetic_data = synth
                metric.timestamps = (np.asarray(self._timestamps)[:n] if self._timestamps is not None else None)
                metric.meter_ids = (np.asarray(self._meter_ids)[:n] if self._meter_ids is not None else None)

    def register_metric(self, metric: EvaluationMetric) -> None:
        """Register an evaluation metric. 
        Args:
            metric: An instance of ReconstructionMetric or SyntheticMetric
        """
        # Accept metric classes for convenience (e.g., from register_all)
        if isinstance(metric, type):
            if issubclass(metric, EvaluationMetric):
                metric = metric()
            else:
                raise ValueError(f"Metric class {metric} must inherit from EvaluationMetric")

        if isinstance(metric, ReconstructionMetric):
            self.reconstruction_metrics[metric.name] = metric
        elif isinstance(metric, SyntheticMetric):
            self.synthetic_metrics[metric.name] = metric
        else:
            metric_label = getattr(metric, 'name', metric.__class__.__name__)
            raise ValueError(f"Metric {metric_label} must inherit from ReconstructionMetric or SyntheticMetric")
    
    def register_metrics(self, metrics: List[EvaluationMetric]) -> None:
        """Register multiple metrics at once.
        Args:
            metrics: List of ReconstructionMetric or SyntheticMetric instances
        """
        for metric in metrics:
            self.register_metric(metric)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run all registered evaluation metrics (reconstruction + synthetic) and return results.
        
        Returns:
            Dict[str, float]: Dictionary mapping metric names to their computed values
        """
        print(f"[{self.evaluation_title}] Running evaluation...")
        
        # Prepare data for metrics that need it (reconstructions, synthetic samples)
        self._prepare_data()
        self._inject_data_into_metrics()

        if self.verbose:
            n_real = 0 if self._real_data is None else int(len(self._real_data))
            n_synth = 0 if self._synthetic_data is None else int(len(self._synthetic_data))
            print(f"Prepared data: real={n_real}, synthetic={n_synth}")

        self.results = {}
        
        # Run reconstruction metrics
        if self.reconstruction_metrics:
            for name, metric in self.reconstruction_metrics.items():
                try:
                    value = metric.compute()
                    self.results[name] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: metric '{name}' failed: {e}")
                    self.results[name] = None
        
        # Run synthetic metrics
        if self.synthetic_metrics:
            for name, metric in self.synthetic_metrics.items():
                try:
                    value = metric.compute()
                    self.results[name] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: metric '{name}' failed: {e}")
                    self.results[name] = None

        self._print_results_table()
        
        return self.results

    def evaluate_reconstruction_only(self) -> Dict[str, float]:
        """Run only reconstruction metrics.
        Returns:
            Dict[str, float]: Dictionary mapping reconstruction metric names to their computed values
        """

        self.results = {}
        for name, metric in self.reconstruction_metrics.items():
            try:
                value = metric.compute()
                self.results[name] = value
            except Exception as e:
                if self.verbose:
                    print(f"Warning: metric '{name}' failed: {e}")
                self.results[name] = None

        self.baseline_results = self._compute_mean_baseline_results(reconstruction_only=True) if self.include_mean_baseline else {}

        self._print_interpretation_guide()
        self._print_results_table()
        return self.results

    def evaluate_synthetic_only(self) -> Dict[str, float]:
        """Run only synthetic data metrics.
        Returns:
            Dict[str, float]: Dictionary mapping synthetic metric names to their computed values
        """

        self.results = {}
        for name, metric in self.synthetic_metrics.items():
            try:
                value = metric.compute()
                self.results[name] = value
            except Exception as e:
                if self.verbose:
                    print(f"Warning: metric '{name}' failed: {e}")
                self.results[name] = None

        self.baseline_results = self._compute_mean_baseline_results(synthetic_only=True) if self.include_mean_baseline else {}

        self._print_interpretation_guide()
        self._print_results_table()
        return self.results

    def _metric_direction(self, metric_name: str) -> str:
        """Return optimization direction for display/benchmarking."""
        if metric_name == "TSTR (Train on Synthetic, Test on Real)":
            return "target_1"
        if "Log-Likelihood" in metric_name or metric_name == "Visual Evaluations":
            return "higher"
        return "lower"

    def _print_results_table(self) -> None:
        """Print results in a formatted table."""
        print("\nEvaluation summary")

        table_data = []
        for name, value in self.results.items():
            model_str = "ERROR" if value is None else (f"{float(value):.6f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value))
                    
            goal = self._metric_direction(name)
            goal_label = "target=1" if goal == "target_1" else ("higher" if goal == "higher" else "lower")
            table_data.append([name, goal_label, model_str])

        headers = ["Metric", "Goal", "Model"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()

    def get_results(self) -> Dict[str, float]:
        """Get cached results from last evaluation."""
        return self.results.copy()

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file.
        
        Args:
            output_path (str): Path to save JSON results
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        results_serializable = {
            k: v if v is not None else "ERROR"
            for k, v in self.results.items()
        }
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        if self.verbose:
            print(f"Results saved to {output_path}")


class PhasedEvaluationSuite:
    """Run separate evaluation views for multi-phase data while preserving the original one-phase evaluator."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        normalizer=None,
        conditioner=None,
        device: str = 'cpu',
        window_size: int = 96,
        n_synthetic_samples: int = 1000,
        output_path: str = 'output',
        secondary_model = None,
    ):
        self.model = model
        self.secondary_model = secondary_model if secondary_model is not None else None
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.conditioner = conditioner
        self.device = device
        self.window_size = int(window_size)
        self.n_synthetic_samples = int(n_synthetic_samples)
        self.output_path = output_path

    def _prepare_base_framework(self) -> EvalFramework:
        framework = EvalFramework(
            self.model,
            self.dataloader,
            normalizer=self.normalizer,
            conditioner=self.conditioner,
            device=self.device,
            window_size=self.window_size,
            n_synthetic_samples=self.n_synthetic_samples,
            output_path=self.output_path,
            evaluation_title='PHASED DATA PREPARATION',
            verbose=False,
            secondary_model=self.secondary_model,
        )
        framework._prepare_data()
        return framework

    def _extract_phase_first(self, flat_data: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(flat_data)
        if arr.ndim == 3:
            # If already 3D, ensure it is [N, F, L]
            if arr.shape[1] == 3 and arr.shape[2] == self.window_size:
                return arr
            if arr.shape[1] == self.window_size and arr.shape[2] == 3:
                return arr.transpose(0, 2, 1)
            raise ValueError(f"Unexpected 3D shape for {label}: {arr.shape}")
            
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array for {label}, got {arr.shape}")
        
        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened {label} dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
                f"data shape: {arr.shape}, window_size: {self.window_size}"
            )
        n_features = arr.shape[1] // self.window_size
        return arr.reshape(arr.shape[0], n_features, self.window_size)

    def _run_prepared_eval(self, prepared_data: Dict[str, Any], output_path: str, title: str) -> Dict[str, float]:
        evaluator = EvalFramework(
            self.model,
            self.dataloader,
            normalizer=self.normalizer,
            conditioner=self.conditioner,
            device=self.device,
            window_size=self.window_size,
            n_synthetic_samples=self.n_synthetic_samples,
            output_path=output_path,
            prepared_data=prepared_data,
            evaluation_title=title,
            verbose=False,
        )
        evaluator.register_all(phased=True)
        results = evaluator.evaluate()
        evaluator.save_results(output_path)
        return results

    def evaluate(self) -> Dict[str, Any]:
        output_dir = _resolve_output_directory(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        base = self._prepare_base_framework()

        real_phase = self._extract_phase_first(base._real_data, 'real data')
        recon_phase = self._extract_phase_first(base._reconstructed_data, 'reconstructed data')
        synthetic_phase = self._extract_phase_first(base._synthetic_data, 'synthetic data')

        aggregated_dir = os.path.join(output_dir, 'aggregated_phase_eval')
        os.makedirs(aggregated_dir, exist_ok=True)
        aggregated_prepared = {
            'real_data': _flatten_windows(real_phase.sum(axis=1)),
            'reconstructed_data': _flatten_windows(recon_phase.sum(axis=1)),
            'synthetic_data': _flatten_windows(synthetic_phase.sum(axis=1)),
            'timestamps': base._timestamps,
            'meter_ids': base._meter_ids,
        }
        aggregate_results = self._run_prepared_eval(
            prepared_data=aggregated_prepared,
            title='AGGREGATED 3-PHASE EVALUATION',
            output_path=os.path.join(aggregated_dir, 'eval_results.json'),
        )

        ranked_real = real_phase
        ranked_recon = _align_phases_by_mse(real_phase, recon_phase)
        ranked_synth = _align_phases_to_reference_mean(real_phase, synthetic_phase)

        ranked_results = {}
        num_phases = ranked_real.shape[1]
        for phase_idx in range(num_phases):
            phase_number = phase_idx + 1
            phase_dir = os.path.join(output_dir, f'ranked_phase_{phase_number}_eval')
            os.makedirs(phase_dir, exist_ok=True)
            prepared = {
                'real_data': _flatten_windows(ranked_real[:, phase_idx, :]),
                'reconstructed_data': _flatten_windows(ranked_recon[:, phase_idx, :]),
                'synthetic_data': _flatten_windows(ranked_synth[:, phase_idx, :]),
                'timestamps': base._timestamps,
                'meter_ids': base._meter_ids,
            }
            ranked_results[f'phase_{phase_number}'] = self._run_prepared_eval(
                prepared_data=prepared,
                title=f'ALIGNED PHASE {phase_number} EVALUATION',
                output_path=os.path.join(phase_dir, 'eval_results.json'),
            )

        summary = {
            'aggregated_phase': aggregate_results,
            'ranked_phases': ranked_results,
        }

        summary_path = os.path.join(output_dir, 'phased_eval_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(_to_json_serializable(summary), f, indent=2)
        return summary


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
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
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
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
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
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mean_log_likelihood
    

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
        
        # Compute FD per sample using assignment-based cost
        num_samples, series_len = real_np.shape
        fd_values = []
        
        for i in range(num_samples):
            x = real_np[i]
            y = recon_np[i]
            fd = self.compute_fd_between_series(x, y, tau=self.tau, normalize=self.normalize)
            fd_values.append(fd)
        
        return float(np.mean(fd_values))


# ============================================================================
# Synthetic Data Evaluation Metrics
# ============================================================================


class MaximumMeanDiscrepancy(SyntheticMetric):
    """Maximum Mean Discrepancy (MMD) between real and synthetic data.
    
    MMD measures the distance between two distributions using a kernel-based approach.
    Uses RBF (Gaussian) kernel with automatic bandwidth selection via median heuristic.

        Implementation inspired by:
        - https://en.wikipedia.org/wiki/Maximum_mean_discrepancy
        - ??

    Math definition:
    MMD^2(P, Q) = E_{x,x' ~ P}[k(x,x')] + E_{y,y' ~ Q}[k(y,y')] - 2 E_{x ~ P, y ~ Q}[k(x,y)]
     where k is the RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
     and sigma is the kernel bandwidth, which can be set via median heuristic or provided by the user.
     Lower MMD means the synthetic distribution is closer to the real distribution, with 0 indicating identical distributions (under the kernel).
        Note: MMD is a distance metric, not a likelihood, so lower is better.

    Args:
        sigma (float): RBF kernel bandwidth. If None, uses median heuristic.
        seed (int): Random seed for reproducibility
        max_samples (int): Maximum number of samples to use for median heuristic (for efficiency)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, sigma: float | None = None, seed: int = 42, max_samples: int = 2000):
        self.sigma = sigma
        self.seed = seed
        self.max_samples = max_samples
    
    @property
    def name(self) -> str:
        return "Maximum Mean Discrepancy (MMD)"
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix between X and Y.
        
        Args:
            X (np.ndarray): array [n, d]
            Y (np.ndarray): array [m, d]
            sigma (float): kernel bandwidth
            
        Returns:
            np.ndarray: kernel matrix [n, m]
        """
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        sq_dists = np.nan_to_num(sq_dists, nan=1e12, posinf=1e12, neginf=0.0)
        denom = max(2.0 * (sigma ** 2), 1e-12)
        return np.exp(-sq_dists / denom)
    
    def _median_heuristic_sigma(self, Z: np.ndarray, max_points: int =  2000) -> float:
        """Estimate kernel bandwidth using median pairwise distance.
        
        Args:
            Z (np.ndarray): array [n, d] - combined data
            max_points (int): max points to use for efficiency
            
        Returns:
            float: estimated sigma value
        """
        
        rng = np.random.default_rng(self.seed)
        if Z.shape[0] > max_points:
            Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
        
        dists = cdist(Z, Z, metric="euclidean")
        med = np.median(dists[dists > 0])
        return float(med) if med > 0 else 1.0
    
    def _mmd_rbf_unbiased(self, X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
        """Compute MMD^2 statistic using RBF kernel.
        
        Args:
            X (np.ndarray): array [n, d] - real data
            Y (np.ndarray): array [m, d] - synthetic data
            sigma (float | None): kernel bandwidth (auto-estimate if None)
            
        Returns:
            float: MMD^2 statistic (lower is better)
        """
        X = _sanitize_numeric_array(X)
        Y = _sanitize_numeric_array(Y)
        n, m = X.shape[0], Y.shape[0]
        
        if n < 2 or m < 2:
            raise ValueError("Need at least 2 samples in each set.")
        
        if sigma is None:
            sigma = self._median_heuristic_sigma(np.vstack([X, Y]))
        
        # Compute kernel matrices
        Kxx = self._rbf_kernel(X, X, sigma)
        Kyy = self._rbf_kernel(Y, Y, sigma)
        Kxy = self._rbf_kernel(X, Y, sigma)
        
        # Zero out diagonal (unbiased estimator)
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        
        # Unbiased MMD^2
        mmd2 = (Kxx.sum() / (n * (n - 1))
              + Kyy.sum() / (m * (m - 1))
              - 2.0 * Kxy.sum() / (n * m))
        
        return float(mmd2)
    
    def compute(self) -> float:
        """Compute MMD metric.
        
        Returns:
            float: MMD value (lower is better, 0 = identical distributions)
        """
        rng = np.random.default_rng(self.seed)

        real = self.real_data
        synthetic = self.synthetic_data
        real = _sanitize_numeric_array(real)
        synthetic = _sanitize_numeric_array(synthetic)

        # Subsample to avoid OOM (O(n^2) kernel matrix)
        if len(real) > self.max_samples:
            idx = rng.choice(len(real), size=self.max_samples, replace=False)
            real = real[idx]
        if len(synthetic) > self.max_samples:
            idx = rng.choice(len(synthetic), size=self.max_samples, replace=False)
            synthetic = synthetic[idx]

        # Use provided sigma or auto-estimate
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([real, synthetic])
            sigma = self._median_heuristic_sigma(combined)
        
        # Compute MMD^2, return MMD 
        mmd2 = self._mmd_rbf_unbiased(real, synthetic, sigma=sigma)
        mmd_value = float(np.sqrt(np.maximum(mmd2, 0.0)))

        _save_standalone_metric_result(
            metric_key='MaximumMeanDiscrepancy',
            payload={
                'value': mmd_value,
                'mmd2': float(mmd2),
                'sigma': float(sigma),
                'seed': int(self.seed),
                'max_samples': int(self.max_samples),
                'n_real_used': int(len(real)),
                'n_synthetic_used': int(len(synthetic)),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mmd_value


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
    
    def _compute_acf(self, time_series, max_lag):
        """
        Compute autocorrelation function for a single time series.
        
        Args:
            time_series (np.ndarray): 1D array of length T
            max_lag (int): Maximum lag to compute
            
        Returns:
            np.ndarray: Array of ACF values for lags 1 to max_lag
        """
        time_series = _sanitize_numeric_array(time_series).reshape(-1)
        n = len(time_series)
        if n == 0:
            return np.zeros(max_lag, dtype=np.float64)

        mean = float(np.mean(time_series))
        centered = time_series - mean
        c0 = float(np.mean(centered * centered))
        if not np.isfinite(c0) or c0 <= 1e-12:
            return np.zeros(max_lag, dtype=np.float64)
        
        acf_values = []
        for lag in range(1, max_lag + 1):
            if lag >= n:
                acf_values.append(0.0)
                continue
            c_lag = float(np.mean(centered[:-lag] * centered[lag:]))
            if not np.isfinite(c_lag):
                acf_values.append(0.0)
            else:
                acf_values.append(float(np.clip(c_lag / c0, -1.0, 1.0)))
        
        return np.array(acf_values)
    
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
        
        # Compute ACF for each time series
        real_acfs = []
        for i in range(len(real_ts)):
            acf = self._compute_acf(real_ts[i], self.max_lag)
            real_acfs.append(acf)
        
        synthetic_acfs = []
        for i in range(len(synthetic_ts)):
            acf = self._compute_acf(synthetic_ts[i], self.max_lag)
            synthetic_acfs.append(acf)
        
        real_acfs = np.array(real_acfs)  # Shape: [N * n_features, max_lag]
        synthetic_acfs = np.array(synthetic_acfs)  # Shape: [M * n_features, max_lag]
        
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


class VisualEvaluation(SyntheticMetric):
    """Visual evaluation of synthetic data"""
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None   

    def __init__(self, output_path='output'):
        """
        Args:
            real_data: numpy array of real data [N, D]
            synthetic_data: numpy array of synthetic data [M, D]
            output_path: Directory to save plots
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Visual Evaluations"
    
    def pca(self) -> None:
        """Generate PCA plots for real vs synthetic data."""
        real_data = self.real_data[:500] if len(self.real_data) > 500 else self.real_data
        synthetic_data = self.synthetic_data[:500] if len(self.synthetic_data) > 500 else self.synthetic_data
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Apply PCA
        pca = PCA(n_components=min(3, real_data.shape[1], synthetic_data.shape[1]))
        pca_real = pca.fit_transform(real_data)
        pca_synthetic = pca.transform(synthetic_data)
        # Plot
        fig = plt.figure(figsize=(16, 6))
        
        # Real data
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pca_real[:, 0], pca_real[:, 1], pca_real[:, 2], 
                   alpha=0.5, label='Real', s=20, c='blue')
        ax1.set_title('Real Data (PCA)')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        
        # Synthetic data
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], pca_synthetic[:, 2], 
                   alpha=0.5, label='Synthetic', s=20, c='red')
        ax2.set_title('Synthetic Data (PCA)')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'pca_comparison.pdf'), format='pdf')
        plt.close()
    
    def tsne(self) -> None:
        """Generate t-SNE plots for real vs synthetic data."""
        # Use smaller subset for t-SNE
        real_data = self.real_data[:500] if len(self.real_data) > 500 else self.real_data
        synthetic_data = self.synthetic_data[:500] if len(self.synthetic_data) > 500 else self.synthetic_data
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Combine for t-SNE
        combined = np.vstack([real_data, synthetic_data])
        labels = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])

        n_samples = combined.shape[0]
        if n_samples < 3:
            raise ValueError(f"Need at least 3 samples for t-SNE, got {n_samples}.")

        # Perplexity must be strictly less than n_samples
        perplexity = min(30, n_samples - 1)
        # Keep a sensible lower bound for stability when possible
        if perplexity < 5 and n_samples >= 6:
            perplexity = 5
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, learning_rate='auto', random_state=42, perplexity=perplexity)
        tsne_result = tsne.fit_transform(combined)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        mask_real = labels == 0
        mask_synthetic = labels == 1
        ax.scatter(tsne_result[mask_real, 0], tsne_result[mask_real, 1], 
                  alpha=0.6, label='Real', s=30, c='blue')
        ax.scatter(tsne_result[mask_synthetic, 0], tsne_result[mask_synthetic, 1], 
                  alpha=0.6, label='Synthetic', s=30, c='red')
        ax.set_title('t-SNE: Real vs Synthetic Data')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'tsne_comparison.pdf'), format='pdf')
        plt.close()
    
    def compute(self) -> float:
        """Generate visual evaluation plots and return success indicator."""
        try:
            self.pca()
            self.tsne()
            return 1.0  # Success
        except Exception as e:
            if getattr(self, 'verbose', False):
                print(f"Warning: visual evaluation plotting failed: {e}")
            return 0.0  # Failure


class ClassicalStatistics(SyntheticMetric):
    """Classical statistics comparison between real and synthetic data.
    
    Computes and compares basic statistical properties including mean, variance,
    min/max, quantiles, and standard deviation across time for both datasets.
        Provides a comprehensive overview of how well the synthetic data captures the 
        central tendencies and variability of the real data, both overall and at each time step.
    """
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size=24, n_features=1):
        self.window_size = window_size
        self.n_features = n_features
    
    @property
    def name(self) -> str:
        return "Classical Statistics"
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute classical statistics for a dataset.
        
        Args:
            data (np.ndarray): numpy array [N, D] where D = window_size * n_features
            
        Returns:
            Dict[str, Any]: Dictionary with statistical measures
        """
        data = _sanitize_numeric_array(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {data.shape}")
        if data.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Data feature dimension ({data.shape[1]}) must be divisible by window_size ({self.window_size})"
            )

        # Support both [N, window_size] (consumption only) and [N, window_size * n_features]
        inferred_n_features = data.shape[1] // self.window_size
        data_reshaped = data.reshape(-1, self.window_size, inferred_n_features)
        data = data_reshaped[:, :, 0:1]  # consumption channel, shape [N, window_size, 1]
        
        # Compute mean profile: average value at each timestep across all samples, shape: [window_size, 1]
        mean_profile = np.mean(data, axis=0)
        
        # Compute std profile: std at each timestep across all samples, shape: [window_size, 1]
        std_profile = np.std(data, axis=0)
        
        # Overall statistics (across all timesteps and samples)
        overall_mean = np.mean(data)
        overall_std = np.std(data)
        overall_min = np.min(data)
        overall_max = np.max(data)
        
        stats = {
            # Time-aware statistics (shape: [window_size, 1])
            'mean_profile': mean_profile,  # Average consumption at each hour
            'std_profile': std_profile,    # Variability at each hour
            
            # Overall statistics (single values)
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'overall_min': overall_min,
            'overall_max': overall_max,
            
            # Distribution statistics
            'quantiles': {
                '25%': np.percentile(data, 25),
                '50%': np.percentile(data, 50),
                '75%': np.percentile(data, 75)
            }
        }
        
        return stats
    
    def compute(self) -> Dict[str, Any]:
        """Compute and compare classical statistics between real and synthetic data.
        
        Returns: TODO: THIS IS NOT TRUE IT IS CURRENTLY A FLOAT VALUE (MAE OF MEAN PROFILES)
            Dict[str, Any]: Dictionary containing both real and synthetic statistics
        """
        real_stats = self._compute_stats(self.real_data)
        synthetic_stats = self._compute_stats(self.synthetic_data)
        
        # Scalar summary: mean absolute difference of mean profiles
        mean_profile_diff = float(np.mean(np.abs(
            real_stats['mean_profile'] - synthetic_stats['mean_profile']
        )))
        std_profile_diff = float(np.mean(np.abs(
            real_stats['std_profile'] - synthetic_stats['std_profile']
        )))
        overall_mean_diff = float(abs(real_stats['overall_mean'] - synthetic_stats['overall_mean']))

        _save_standalone_metric_result(
            metric_key='ClassicalStatistics',
            payload={
                'value': mean_profile_diff,
                'differences': {
                    'mean_profile_mae': mean_profile_diff,
                    'std_profile_mae': std_profile_diff,
                    'overall_mean_abs_diff': overall_mean_diff,
                },
                'real': real_stats,
                'synthetic': synthetic_stats,
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )

        return mean_profile_diff
    
    def print_comparison(self):
        """Print formatted comparison of statistics."""
        stats = self.compute()
        real = stats['real']
        synth = stats['synthetic']
    
        print(f"CLASSICAL STATISTICS COMPARISON")
        
        print(f"Overall Statistics:")
        print(f"  Real:      Mean={real['overall_mean']:.4f}, Std={real['overall_std']:.4f}, "
              f"Min={real['overall_min']:.4f}, Max={real['overall_max']:.4f}")
        print(f"  Synthetic: Mean={synth['overall_mean']:.4f}, Std={synth['overall_std']:.4f}, "
              f"Min={synth['overall_min']:.4f}, Max={synth['overall_max']:.4f}")

        print(f"\nQuantiles:")
        print(f"  Real:      25%={real['quantiles']['25%']:.4f}, "
              f"50%={real['quantiles']['50%']:.4f}, 75%={real['quantiles']['75%']:.4f}")
        print(f"  Synthetic: 25%={synth['quantiles']['25%']:.4f}, "
              f"50%={synth['quantiles']['50%']:.4f}, 75%={synth['quantiles']['75%']:.4f}")
        
        print(f"\nMean Profile (avg at each timestep)")
        print(f"  Real:")
        for i in range(len(real['mean_profile'])):
            print(f"    Hour {i}: {real['mean_profile'][i, 0]:.4f}")
        print(f"  Synthetic:")
        for i in range(len(synth['mean_profile'])):
            print(f"    Hour {i}: {synth['mean_profile'][i, 0]:.4f}")
        
        print(f"\nStd Profile (variability at each timestep)")
        print(f"  Real:")
        for i in range(len(real['std_profile'])):
            print(f"    Hour {i}: {real['std_profile'][i, 0]:.4f}")
        print(f"  Synthetic:")
        for i in range(len(synth['std_profile'])):
            print(f"    Hour {i}: {synth['std_profile'][i, 0]:.4f}")
        
        # Compute differences
        mean_profile_diff = np.mean(np.abs(real['mean_profile'] - synth['mean_profile']))
        std_profile_diff = np.mean(np.abs(real['std_profile'] - synth['std_profile']))
        overall_mean_diff = abs(real['overall_mean'] - synth['overall_mean'])
        
        print(f"\nMean Absolute Differences:")
        print(f"  Overall Mean Difference: {overall_mean_diff:.6f}")
        print(f"  Mean Profile Difference: {mean_profile_diff:.6f}")
        print(f"  Std Profile Difference: {std_profile_diff:.6f}")
        print(f"{'='*70}\n")


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
    
    def __init__(self, task: str = 'next_step', window_size: int = 24, n_features: int = 1, 
                 predictor_type: str = 'linear', epochs: int = 50, random_state: int = 42):
        self.task = task
        self.window_size = window_size
        self.n_features = n_features
        self.predictor_type = predictor_type
        self.epochs = epochs
        self.random_state = random_state
        
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
        
        # TODO: maybe predict several days ahead based on several previous days instead of just hours 

        if self.task == 'next_step':
            # Predict last hour from first 23 hours
            X = data_consumption[:, :-1]  # [N, 23]
            y = data_consumption[:, -1:]  # [N, 1]
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
            model = nn.Linear(input_dim, 1)
        elif self.predictor_type == 'mlp':
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
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
        
        _save_standalone_metric_result(
            metric_key='TSTR',
            payload={
                'value': float(tstr_ratio),
                'trtr_mse': float(trtr_mse),
                'tstr_mse': float(tstr_mse),
                'task': self.task,
                'window_size': int(self.window_size),
                'predictor_type': self.predictor_type,
                'epochs': int(self.epochs),
                'n_real_total': int(len(real_data)),
                'n_real_train': int(len(real_train)),
                'n_real_test': int(len(real_test)),
                'n_synthetic_train': int(len(synthetic_train)),
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return tstr_ratio


class AggregatedProfiles(SyntheticMetric):
    """Aggregate samples from real and synthetic data and compute Flexibility Distance between aggregated profiles.
     This metric evaluates how well the synthetic data captures the overall consumption patterns when multiple samples are aggregated together, 
     which is important for understanding collective behavior and flexibility in energy consumption.
     Steps:
     1. Randomly sample N time series from real and synthetic datasets (e.g.,
     2. Aggregate by summing consumption across the sampled series at each time step to create an "aggregated profile" for real and synthetic data.
     3. Compute Flexibility Distance between the two aggregated profiles to assess similarity in overall patterns and flexibility.
     4. Optionally, plot the aggregated profiles for visual comparison.
     Lower Flexibility Distance indicates that the synthetic data captures the collective consumption patterns of the real data more closely, with 0 indicating identical aggregated profiles.
    
    Args:
    n_samples (int): Number of time series to sample and aggregate (default: 100)
    window_size (int): Length of each time series window (default: 24)
    tau (int or 'auto'): Time delay for Flexibility Distance (default: 'auto, which uses median heuristic)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, n_samples: int = 100, window_size: int = 24, tau: str = 'auto',
                 plot_filename: str = 'aggregated_profiles_comparison.pdf'):
        self.n_samples = n_samples
        self.window_size = window_size
        self.tau = tau
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Aggregated Profiles (Flexibility Distance)"

    def _uses_phase_first_flatten(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')

    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        if self._uses_phase_first_flatten():
            # Flatten order is [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous.
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]

        # Fallback for time-first flatten order [t0_f0, t0_f1, ..., t1_f0, ...].
        return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Aggregate real and synthetic data, then compute Flexibility Distance."""
        real_data = self._extract_consumption_feature(self.real_data)
        synthetic_data = self._extract_consumption_feature(self.synthetic_data)
        
        # Use the same number of samples for fair comparison
        n_agg = min(self.n_samples, len(real_data), len(synthetic_data))
        if n_agg < 1:
            raise ValueError("AggregatedProfiles requires at least one sample in both real and synthetic data")

        real_samples = real_data[:n_agg]
        synth_samples = synthetic_data[:n_agg]
        
        # Aggregate by summing
        real_aggregated = np.sum(real_samples, axis=0)
        synth_aggregated = np.sum(synth_samples, axis=0)
        
        # Plot aggregated profiles
        self._plot_aggregated_profiles(real_aggregated, synth_aggregated)
        
        # Compute Flexibility Distance
        fd_score = FlexibilityDistance.compute_fd_between_series(
            real_aggregated, synth_aggregated, tau=self.tau
        )

        _save_standalone_metric_result(
            metric_key='AggregatedProfiles',
            payload={
                'value': float(fd_score),
                'n_samples_requested': int(self.n_samples),
                'n_samples_used': int(n_agg),
                'window_size': int(self.window_size),
                'tau': self.tau,
                'real_aggregated': real_aggregated,
                'synthetic_aggregated': synth_aggregated,
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return fd_score
    
    def _plot_aggregated_profiles(self, real_aggregated: np.ndarray, synth_aggregated: np.ndarray):
        """Plot aggregated real and synthetic profiles.
        Args:
            real_aggregated (np.ndarray): Aggregated profile from real data
            synth_aggregated (np.ndarray): Aggregated profile from synthetic data
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        timesteps = np.arange(len(real_aggregated))
        ax.plot(timesteps, real_aggregated, label='Real Aggregated', 
                color='blue', linewidth=2, alpha=0.8)
        ax.plot(timesteps, synth_aggregated, label='Synthetic Aggregated', 
                color='red', linewidth=2, alpha=0.8, linestyle='--')
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Aggregated Consumption [kWh]', fontsize=12)
        ax.set_title(f'Aggregated Consumption Profiles (n={self.n_samples} samples)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        base_dir = _resolve_output_directory(self.output_path)
        os.makedirs(base_dir, exist_ok=True)
        plot_filename = self.plot_filename or 'aggregated_profiles_comparison.pdf'
        plt.savefig(os.path.join(base_dir, plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        

class PerConditionEval(SyntheticMetric):
    """Compute per-condition evaluation for MMD, ACFDistribution, TSTR, ClassicalStatistics, and AggregatedProfiles.

    Includes:
    - MMD per day-of-week and month (using full 24-hour windows)
    - ACFDistribution per day-of-week and month (using full 24-hour windows)
    - ClassicalStatistics per day-of-week and month
    - TSTR per day-of-week and month (train/test on real, train on synthetic, test on real holdout)
    - AggregatedProfiles per day-of-week and month (aggregate samples within each group and compare profiles)

        Also includes AggregatedProfiles per feeder (aggregate samples by feeder and compare profiles), which is important for understanding how well the synthetic data captures patterns at the feeder level, which can be critical for grid management and planning.

        This comprehensive per-condition evaluation framework allows us to assess synthetic data quality across multiple dimensions and conditions, providing insights into where the synthetic data performs well and where it may need improvement.

    Args:
    window_size (int): Length of time series windows (default: 24 for hourly data)
    max_lag (int): Maximum lag for ACFDistribution (default: 10)
    min_samples_per_group (int): Minimum samples required in each group to compute metrics (default: 2)
    verbose (bool): Whether to print detailed debug information during computation (default: False)
    predictor_type (str): Predictor type for TSTR ('linear', 'mlp', etc.) (default: 'linear')
    tstr_epochs (int): Training epochs for TSTR predictor (default: 50)
    feeder_mapping_path (str): Path to JSON file mapping meter IDs to feeder IDs (default: 'topology/cleaned/Cleaned_SM_Topology_mapping.json')
    min_samples_per_feeder (int): Minimum samples required per feeder to compute AggregatedProfiles (default: 5)
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    meter_ids: np.ndarray = None
    feeder_ids: np.ndarray = None
    output_path: str = None

    def __init__(self, window_size: int = 24, max_lag: int = 10, min_samples_per_group: int = 2,
                 verbose: bool = False, predictor_type: str = 'linear', tstr_epochs: int = 50,
                 feeder_mapping_path: str = 'data/topology/cleaned/Cleaned_SM_Topology_mapping.json',
                 min_samples_per_feeder: int = 5):
        self.window_size = window_size
        self.max_lag = max_lag
        self.min_samples_per_group = min_samples_per_group
        self.verbose = verbose
        self.predictor_type = predictor_type
        self.tstr_epochs = tstr_epochs
        self.feeder_mapping_path = feeder_mapping_path
        self.min_samples_per_feeder = min_samples_per_feeder
        self._meter_to_feeder = None

        self.results_by_condition = {
            'mmd': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'acf': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'classical_statistics': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'aggregated_profiles': {
                'day_of_week': {},
                'month_of_year': {},
                'feeder': {},
            },
            'tstr': {
                'day_of_week': {},
                'month_of_year': {},
            }
        }

        self.mmd_calculator = MaximumMeanDiscrepancy(sigma=None, seed=42, max_samples=2000)
        self.acf_calculator = ACFDistribution(window_size=window_size, max_lag=max_lag)
        self.classical_statistics_calculator = ClassicalStatistics(window_size=window_size, n_features=1)
        self.tstr_calculator = TSTR(
            task='next_step',
            window_size=window_size,
            n_features=1,
            predictor_type=predictor_type,
            epochs=tstr_epochs,
            random_state=42
        )
        self.aggregated_profiles_calculator = AggregatedProfiles(
            n_samples=100,
            window_size=window_size,
            tau='auto',
        )

    @property
    def name(self) -> str:
        return "Per-Condition Evaluation"

    def _compute_subset_acf(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute ACF score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None
        self.acf_calculator.real_data = real_subset
        self.acf_calculator.synthetic_data = synthetic_subset
        return float(self.acf_calculator.compute())

    def _compute_subset_tstr(self, real_subset: np.ndarray, synthetic_subset: np.ndarray,
                             group_label: str = None) -> float:
        """Compute TSTR ratio for one subset using the TSTR metric; return None if too few samples."""
        if len(real_subset) < 4 or len(synthetic_subset) < 2:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} -> insufficient samples (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            return None

        try:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            self.tstr_calculator.real_data = real_subset
            self.tstr_calculator.synthetic_data = synthetic_subset
            tstr_ratio = self.tstr_calculator.compute()
            return float(tstr_ratio)
        except Exception as e:
            if self.verbose:
                label_suffix = f" for {group_label}" if group_label is not None else ""
                print(f"Warning: TSTR computation failed{label_suffix}: {e}")
            return None

    def _compute_subset_aggregated_profiles(self, real_subset: np.ndarray, synthetic_subset: np.ndarray,
                                            group_type: str = 'all', group_value: Any = 'all'):
        """Compute AggregatedProfiles score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            group_key = self._sanitize_label_for_filename(group_type)
            group_val = self._sanitize_label_for_filename(group_value)
            plot_filename = f"aggregated_profiles_comparison_{group_key}_{group_val}.pdf"

            metric = AggregatedProfiles(
                n_samples=self.aggregated_profiles_calculator.n_samples,
                window_size=self.aggregated_profiles_calculator.window_size,
                tau=self.aggregated_profiles_calculator.tau,
                plot_filename=plot_filename,
            )
            metric.model = getattr(self, 'model', None)
            metric.real_data = real_subset
            metric.synthetic_data = synthetic_subset
            metric.output_path = self.output_path
            return float(metric.compute())
        except Exception as e:
            if self.verbose:
                print(f"Warning: AggregatedProfiles computation failed: {e}")
            return None

    def _load_meter_to_feeder_mapping(self) -> Dict[str, str]:
        """Load meter->feeder mapping from Cleaned_SM_Topology_mapping.json."""
        if self._meter_to_feeder is not None:
            return self._meter_to_feeder

        mapping_path = self.feeder_mapping_path
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.join(os.getcwd(), mapping_path)

        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)

        meter_to_feeder = {}
        for meter, topo in raw_mapping.items():
            feeder_id = topo.get('Feeder ID') if isinstance(topo, dict) else None
            if feeder_id is None:
                meter_to_feeder[str(meter)] = None
            else:
                meter_to_feeder[str(meter)] = f"LvFeeder.{feeder_id}"

        self._meter_to_feeder = meter_to_feeder
        return meter_to_feeder

    def _resolve_feeder_ids(self, n_expected: int) -> np.ndarray:
        """Resolve feeder IDs either from injected feeder_ids or from injected meter_ids + mapping file."""
        if self.feeder_ids is not None:
            feeder_ids = np.asarray(self.feeder_ids)
        elif self.meter_ids is not None:
            meter_ids = np.asarray(self.meter_ids)
            if len(meter_ids) != n_expected:
                raise ValueError(f"meter_ids length ({len(meter_ids)}) must match data length ({n_expected})")

            meter_to_feeder = self._load_meter_to_feeder_mapping()
            feeder_ids = []
            for meter in meter_ids:
                if isinstance(meter, np.generic):
                    meter = meter.item()
                meter_key = str(meter)
                feeder_ids.append(meter_to_feeder.get(meter_key, None))
            feeder_ids = np.array(feeder_ids, dtype=object)
        else:
            return None

        if len(feeder_ids) != n_expected:
            raise ValueError(f"feeder_ids length ({len(feeder_ids)}) must match data length ({n_expected})")
        return self._normalize_feeder_ids(feeder_ids)

    @staticmethod
    def _normalize_feeder_ids(feeder_ids: np.ndarray) -> np.ndarray:
        labels = []
        for raw in np.asarray(feeder_ids).reshape(-1):
            if isinstance(raw, np.generic):
                raw = raw.item()
            if raw is None:
                labels.append(None)
                continue
            sval = str(raw).strip()
            if sval == "" or sval.lower() in {"nan", "none"}:
                labels.append(None)
            else:
                labels.append(sval)
        return np.array(labels, dtype=object)

    @staticmethod
    def _unique_valid_labels(labels: np.ndarray) -> List[str]:
        unique = []
        seen = set()
        for label in np.asarray(labels).reshape(-1):
            if label is None:
                continue
            sval = str(label)
            if sval not in seen:
                seen.add(sval)
                unique.append(sval)
        return unique

    @staticmethod
    def _sanitize_label_for_filename(label: Any) -> str:
        text = str(label).strip().lower()
        safe = []
        for ch in text:
            if ch.isalnum():
                safe.append(ch)
            else:
                safe.append('_')
        out = ''.join(safe)
        while '__' in out:
            out = out.replace('__', '_')
        out = out.strip('_')
        return out or 'group'

    def _compute_aggregated_profiles_per_feeder(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feeder_ids: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute AggregatedProfiles FD per feeder."""
        feeder_scores = {}
        feeder_ids = self._normalize_feeder_ids(feeder_ids)
        unique_feeders = self._unique_valid_labels(feeder_ids)

        for feeder in unique_feeders:
            n_group = int(np.sum(feeder_ids == feeder))
            if n_group < self.min_samples_per_feeder:
                continue

            score = self._compute_subset_aggregated_profiles(
                real_data[feeder_ids == feeder],
                synthetic_data[feeder_ids == feeder],
                group_type='feeder',
                group_value=str(feeder),
            )
            if score is None:
                continue

            feeder_scores[str(feeder)] = {
                'fd': float(score),
                'n_samples': n_group,
            }

        return feeder_scores

    def _compute_classical_stats_for_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute classical stats for [N, window_size] consumption windows."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected [N, {self.window_size}] array for classical stats, got {data.shape}"
            )

        mean_profile = np.mean(data, axis=0)
        std_profile = np.std(data, axis=0)

        return {
            'mean_profile': mean_profile.tolist(),
            'std_profile': std_profile.tolist(),
            'overall_mean': float(np.mean(data)),
            'overall_std': float(np.std(data)),
            'overall_min': float(np.min(data)),
            'overall_max': float(np.max(data)),
            'quantiles': {
                '25%': float(np.percentile(data, 25)),
                '50%': float(np.percentile(data, 50)),
                '75%': float(np.percentile(data, 75)),
            }
        }

    def _compute_subset_classical_stats(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute full classical stats for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            real_stats = self._compute_classical_stats_for_data(real_subset)
            synthetic_stats = self._compute_classical_stats_for_data(synthetic_subset)
            real_mean_profile = np.array(real_stats['mean_profile'])
            synth_mean_profile = np.array(synthetic_stats['mean_profile'])
            real_std_profile = np.array(real_stats['std_profile'])
            synth_std_profile = np.array(synthetic_stats['std_profile'])

            return {
                'n_profiles': int(min(len(real_subset), len(synthetic_subset))),
                'real': real_stats,
                'synthetic': synthetic_stats,
                'differences': {
                    'overall_mean_abs_diff': float(abs(real_stats['overall_mean'] - synthetic_stats['overall_mean'])),
                    'overall_std_abs_diff': float(abs(real_stats['overall_std'] - synthetic_stats['overall_std'])),
                    'mean_profile_mae': float(np.mean(np.abs(real_mean_profile - synth_mean_profile))),
                    'std_profile_mae': float(np.mean(np.abs(real_std_profile - synth_std_profile))),
                }
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: ClassicalStatistics computation failed: {e}")
            return None

    def _plot_aggregated_profiles_bars(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        days_of_week: np.ndarray,
        months_of_year: np.ndarray,
        feeder_ids: np.ndarray = None,
    ) -> Dict[str, str]:
        """Create bar charts of aggregated real vs synthetic consumption by day and month."""
        real_data = _sanitize_numeric_array(real_data)
        synthetic_data = _sanitize_numeric_array(synthetic_data)

        base_dir = _resolve_output_directory(self.output_path)
        # plots_dir = os.path.join(base_dir, 'Plots') if base_dir else 'Plots'
        plots_dir = base_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot style
        plot_style = PlotStyle(figsize=(14, 6))
        color_real = plot_style.palette_colors[0]
        color_synthetic = plot_style.palette_colors[4]
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        # Sum total consumption (all windows + all 24 hours) per day-of-week
        day_labels = []
        day_real_totals = []
        day_synth_totals = []
        day_counts = []
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                day_labels.append(day_names[day])
                day_real_totals.append(float(np.sum(real_data[mask])))
                day_synth_totals.append(float(np.sum(synthetic_data[mask])))
                day_counts.append(int(np.sum(mask)))

        day_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_day_of_week.pdf')
        if day_labels:
            x = np.arange(len(day_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, day_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, day_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Day of Week ({sum(day_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(day_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(day_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption (all windows + all 24 hours) per month
        month_labels = []
        month_real_totals = []
        month_synth_totals = []
        month_counts = []
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                month_labels.append(month_names[month])
                month_real_totals.append(float(np.sum(real_data[mask])))
                month_synth_totals.append(float(np.sum(synthetic_data[mask])))
                month_counts.append(int(np.sum(mask)))

        month_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_month_of_year.pdf')
        if month_labels:
            x = np.arange(len(month_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, month_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, month_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Month of Year ({sum(month_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month of Year', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(month_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(month_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption per feeder (if feeder_ids provided)
        feeder_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_feeder.pdf')
        if feeder_ids is not None:
            feeder_ids = self._normalize_feeder_ids(feeder_ids)
            unique_feeders = self._unique_valid_labels(feeder_ids)
            
            if unique_feeders:
                feeder_labels = []
                feeder_real_totals = []
                feeder_synth_totals = []
                feeder_counts = []
                for feeder in unique_feeders:
                    mask = feeder_ids == feeder
                    if np.sum(mask) > 0:
                        feeder_labels.append(str(feeder))
                        feeder_real_totals.append(float(np.sum(real_data[mask])))
                        feeder_synth_totals.append(float(np.sum(synthetic_data[mask])))
                        feeder_counts.append(int(np.sum(mask)))
                
                if feeder_labels:
                    x = np.arange(len(feeder_labels))
                    width = 0.38
                    # fig, ax = plt.subplots(figsize=(max(12, len(feeder_labels)), 6))
                    fig, ax = plt.subplots(figsize=plot_style.figsize)
                    ax.bar(x - width / 2, feeder_real_totals, width, label='Real', 
                           color=color_real, alpha=0.9, linewidth=1.2)
                    ax.bar(x + width / 2, feeder_synth_totals, width, label='Synthetic', 
                           color=color_synthetic, alpha=0.9, linewidth=1.2)
                    ax.set_title(f'Aggregated Consumption by Feeder ({sum(feeder_counts)} profiles)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Feeder ID', fontsize=12)
                    ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(feeder_labels, rotation=45, ha='right', fontsize=10)
                    ax.legend(fontsize=11)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    plt.savefig(feeder_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

        # Split into input timesteps to see if it has an easier time predicting certain hours of the day (e.g., morning vs evening) 
        hour_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_hour_of_day.pdf')

        input_length = real_data.shape[1]
        if input_length == self.window_size:
            hour_labels = [f"Hour {i}" for i in range(self.window_size)]
            hour_real_totals = np.sum(real_data, axis=0)
            hour_synth_totals = np.sum(synthetic_data, axis=0)

            x = np.arange(len(hour_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, hour_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, hour_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Hour of Day ({real_data.shape[0]} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(hour_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(hour_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        return {
            'day_of_week': day_plot_path,
            'month_of_year': month_plot_path,
            'feeder': feeder_plot_path if feeder_ids is not None else None,
            'hour_of_day': hour_plot_path,
        }

    def compute(self) -> float:
        """Run per-condition MMD, ACFDistribution, and TSTR and save results."""
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("PerConditionEval requires injected real_data and synthetic_data.")
        if self.timestamps is None:
            raise ValueError("PerConditionEval requires injected timestamps for per-condition grouping.")

        self.mmd_calculator.output_path = self.output_path
        self.acf_calculator.output_path = self.output_path
        self.classical_statistics_calculator.output_path = self.output_path
        self.tstr_calculator.output_path = self.output_path
        self.aggregated_profiles_calculator.output_path = self.output_path

        real_data = _sanitize_numeric_array(np.asarray(self.real_data))
        synthetic_data = _sanitize_numeric_array(np.asarray(self.synthetic_data))
        timestamps = np.asarray(self.timestamps)

        if len(real_data) != len(synthetic_data):
            synthetic_data = synthetic_data[:len(real_data)]

        # Ensure lengths match before creating masking arrays
        min_len = min(len(real_data), len(timestamps))
        real_data = real_data[:min_len]
        synthetic_data = synthetic_data[:min_len]
        timestamps = timestamps[:min_len]

        if real_data.ndim != 2 or synthetic_data.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real_data.shape}, synthetic={synthetic_data.shape}")
        if real_data.shape[1] != self.window_size or synthetic_data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window shape [N, {self.window_size}], got real={real_data.shape}, synthetic={synthetic_data.shape}"
            )
        if self.verbose:
            self._print_input_debug(real_data, synthetic_data, timestamps)

        days_of_week = []
        months_of_year = []
        for ts in timestamps:
            dt = datetime.utcfromtimestamp(ts)
            days_of_week.append(dt.weekday())
            months_of_year.append(dt.month)
        days_of_week = np.array(days_of_week)
        months_of_year = np.array(months_of_year)

        # Resolve feeder IDs if available
        feeder_ids = self._resolve_feeder_ids(len(real_data))

        aggregated_profile_plot_paths = self._plot_aggregated_profiles_bars(
            real_data=real_data,
            synthetic_data=synthetic_data,
            days_of_week=days_of_week,
            months_of_year=months_of_year,
            feeder_ids=feeder_ids,
        )

        # -------------------------
        # MMD 
        # -------------------------
        mmd_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_day[day] = self.mmd_calculator.compute()
                else:
                    mmd_per_day[day] = 0.0

        mmd_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_month[month] = self.mmd_calculator.compute()
                else:
                    mmd_per_month[month] = 0.0

        # -------------------------
        # ACF Distribution 
        # -------------------------
        acf_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                acf_per_day[day] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        acf_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                acf_per_month[month] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        # -------------------------
        # Classical Statistics
        # -------------------------
        classical_stats_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                classical_stats_per_day[day] = self._compute_subset_classical_stats(
                    real_data[days_of_week == day], synthetic_data[days_of_week == day]
                )

        classical_stats_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                classical_stats_per_month[month] = self._compute_subset_classical_stats(
                    real_data[months_of_year == month], synthetic_data[months_of_year == month]
                )

        # -------------------------
        # Aggregated Profiles (Flexibility Distance)
        # -------------------------
        aggregated_profiles_per_day = {}
        for day in range(7):
            if np.sum( days_of_week == day) > 0:
                aggregated_profiles_per_day[day] = self._compute_subset_aggregated_profiles(
                    real_data[ days_of_week == day],
                    synthetic_data[ days_of_week == day],
                    group_type='day_of_week',
                    group_value=day,
                )

        aggregated_profiles_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                aggregated_profiles_per_month[month] = self._compute_subset_aggregated_profiles(
                    real_data[months_of_year == month],
                    synthetic_data[months_of_year == month],
                    group_type='month_of_year',
                    group_value=month,
                )

        aggregated_profiles_per_feeder = {}
        feeder_ids = self._resolve_feeder_ids(len(real_data))
        if feeder_ids is not None:
            aggregated_profiles_per_feeder = self._compute_aggregated_profiles_per_feeder(
                real_data=real_data,
                synthetic_data=synthetic_data,
                feeder_ids=feeder_ids,
            )

        # -------------------------
        # TSTR (Train on Synthetic, Test on Real)
        # -------------------------
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        tstr_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                tstr_per_day[day] = self._compute_subset_tstr(
                    real_data[days_of_week == day],
                    synthetic_data[days_of_week == day],
                    group_label=f"day_of_week={day_names[day]} (day {day})"
                )

        tstr_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                tstr_per_month[month] = self._compute_subset_tstr(
                    real_data[months_of_year == month],
                    synthetic_data[months_of_year == month],
                    group_label=f"month_of_year={month_names[month]} (month {month})"
                )


        # -------------------------
        # Save results and print
        # -------------------------
        self.results_by_condition['mmd']['day_of_week'] = mmd_per_day
        self.results_by_condition['mmd']['month_of_year'] = mmd_per_month
        self.results_by_condition['acf']['day_of_week'] = acf_per_day
        self.results_by_condition['acf']['month_of_year'] = acf_per_month
        self.results_by_condition['classical_statistics']['day_of_week'] = classical_stats_per_day
        self.results_by_condition['classical_statistics']['month_of_year'] = classical_stats_per_month
        self.results_by_condition['aggregated_profiles']['day_of_week'] = aggregated_profiles_per_day
        self.results_by_condition['aggregated_profiles']['month_of_year'] = aggregated_profiles_per_month
        self.results_by_condition['aggregated_profiles']['feeder'] = aggregated_profiles_per_feeder
        self.results_by_condition['tstr']['day_of_week'] = tstr_per_day
        self.results_by_condition['tstr']['month_of_year'] = tstr_per_month

        # Save file
        output_dir = _resolve_output_directory(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        combined_output_file = os.path.join(output_dir, 'eval_results_condition.json')
        existing_payload = {}
        if os.path.exists(combined_output_file):
            try:
                with open(combined_output_file, 'r') as f:
                    existing_payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_payload = {}

        output_payload = {
            'mmd': self.results_by_condition['mmd'],
            'acf': self.results_by_condition['acf'],
            'classical_statistics': self.results_by_condition['classical_statistics'],
            'aggregated_profiles': self.results_by_condition['aggregated_profiles'],
            'tstr': self.results_by_condition['tstr'],
            'aggregated_profile_plots': aggregated_profile_plot_paths,
            'feeder_mapping_path': self.feeder_mapping_path,
            'min_samples_per_feeder': int(self.min_samples_per_feeder),
            'window_size': int(self.window_size),
            'max_lag': int(self.max_lag),
            'min_samples_per_group': int(self.min_samples_per_group),
            'predictor_type': self.predictor_type,
        }

        # Preserve standalone metric entries written by individual metric classes
        if isinstance(existing_payload, dict) and 'standalone_metrics' in existing_payload:
            output_payload['standalone_metrics'] = existing_payload['standalone_metrics']

        with open(combined_output_file, 'w') as f:
            json.dump(_to_json_serializable(output_payload), f, indent=2)

        all_mmd_scores = list(mmd_per_day.values()) + list(mmd_per_month.values())
        avg_mmd = np.mean(all_mmd_scores) if all_mmd_scores else 0.0
        return float(avg_mmd)


class SetMaximumMeanDiscrepancy(SyntheticMetric):

    """Permutation-Invariant MMD for 3-Phase Set Data."""
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, sigma: float | None = None, seed: int = 42, max_samples: int = 1000, n_features: int = 3):
        self.sigma = sigma
        self.seed = seed
        self.max_samples = max_samples
        self.n_features = n_features
        # The 6 possible permutations for 3 phases
        self.perms = [
            [0, 1, 2], [0, 2, 1], 
            [1, 0, 2], [1, 2, 0], 
            [2, 0, 1], [2, 1, 0]
        ]

    @property
    def name(self) -> str:
        return "Set-MMD (Permutation Invariant)"

    def _set_sqeuclidean_cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the minimum squared euclidean distance across all 6 permutations.
        X: [N, 3, 24]
        Y: [M, 3, 24]
        Returns: [N, M] distance matrix
        """
        N, F, L = X.shape
        M = Y.shape[0]
        
        # Flatten the temporal dimension for distance calculation -> [N, 3*24]
        X_flat = X.reshape(N, -1)
        
        # We will track the minimum distance across all permutations
        min_dists = np.full((N, M), np.inf)
        
        for p in self.perms:
            # Permute the features of Y
            Y_perm = Y[:, p, :]
            # Flatten -> [M, 3*24]
            Y_perm_flat = Y_perm.reshape(M, -1)
            
            # Compute pairwise squared euclidean distance for this permutation
            dists_p = cdist(X_flat, Y_perm_flat, metric="sqeuclidean")
            
            # Update minimum distances
            min_dists = np.minimum(min_dists, dists_p)
            
        return min_dists

    def _rbf_kernel_set(self, X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        sq_dists = self._set_sqeuclidean_cdist(X, Y)
        sq_dists = np.nan_to_num(sq_dists, nan=1e12, posinf=1e12, neginf=0.0)
        denom = max(2.0 * (sigma ** 2), 1e-12)
        return np.exp(-sq_dists / denom)

    def _median_heuristic_sigma_set(self, Z: np.ndarray, max_points: int = 1000) -> float:
        rng = np.random.default_rng(self.seed)
        if Z.shape[0] > max_points:
            Z = Z[rng.choice(Z.shape[0], size=max_points, replace=False)]
            
        dists = self._set_sqeuclidean_cdist(Z, Z)
        med = np.median(dists[dists > 0])
        return float(med) if med > 0 else 1.0

    def compute(self) -> float:
        # 1. Ensure data is correctly shaped as [N, F, L]
        self.real_data = _sanitize_numeric_array(self.real_data)
        self.synthetic_data = _sanitize_numeric_array(self.synthetic_data)

        N_real = self.real_data.shape[0]
        N_synth = self.synthetic_data.shape[0]
        L = self.real_data.shape[1] // self.n_features
        
        real_3d = self.real_data.reshape(N_real, self.n_features, L)
        synth_3d = self.synthetic_data.reshape(N_synth, self.n_features, L)

        # 2. Subsample for memory efficiency
        rng = np.random.default_rng(self.seed)
        if len(real_3d) > self.max_samples:
            real_3d = real_3d[rng.choice(len(real_3d), size=self.max_samples, replace=False)]
        if len(synth_3d) > self.max_samples:
            synth_3d = synth_3d[rng.choice(len(synth_3d), size=self.max_samples, replace=False)]

        # 3. Compute Auto-Sigma on the combined Set Data
        sigma = self.sigma
        if sigma is None:
            combined = np.vstack([real_3d, synth_3d])
            sigma = self._median_heuristic_sigma_set(combined)

        # 4. Compute Kernel Matrices
        n, m = real_3d.shape[0], synth_3d.shape[0]
        Kxx = self._rbf_kernel_set(real_3d, real_3d, sigma)
        Kyy = self._rbf_kernel_set(synth_3d, synth_3d, sigma)
        Kxy = self._rbf_kernel_set(real_3d, synth_3d, sigma)

        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)

        mmd2 = (Kxx.sum() / (n * (n - 1))
              + Kyy.sum() / (m * (m - 1))
              - 2.0 * Kxy.sum() / (n * m))

        mmd_value = float(np.sqrt(np.maximum(mmd2, 0.0)))
        return mmd_value

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

        if len(self.sort_real_data) != len(self.sorted_synthetic_data):
            min_len = min(len(self.sorted_real_data), len(self.sorted_synthetic_data))
            self.sorted_real_data = self.sorted_real_data[:min_len]
            self.sorted_synthetic_data = self.sorted_synthetic_data[:min_len]
        # Create the x-axis values (time steps)
        x = np.arange(1, len(self.sort_real_data) + 1)
        # Plotting the load duration curve
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.sorted_real_data, label='Real Data', color='blue')
        plt.plot(x, self.sorted_synthetic_data, label='Synthetic Data', color='orange')
        plt.title('Load Duration Curve')
        plt.xlabel('Time Steps')
        plt.ylabel('Load')
        plt.legend()
        plt.grid()
        save_path = self.output_path[:-17]
        plt.savefig(f"{save_path}/load_duration_curve.png")
        plt.close()

    def _EMD1(self):
        """ Will evaluate the magnitude difference of the peaks between the real and synthetic data.
        Compute the Earth Mover's Distance (EMD) between the sorted real and synthetic data.
        This is equivilant to Wasserstein 1 distance.
        
        PeakMagEMD = 1/N ∑ |P_real(i)↑​ - P_syn(i)↑|
        We calculate it by sorting the real and synthetic peaks in descending order and then computing the average absolute difference between the corresponding values.
        However, when we take the average we loose the information of which directions the peaks are wrong.
        """
        emd = np.mean(np.abs(self.sorted_real_data - self.sorted_synthetic_data))
        return emd

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

    def compute(self):
        self.sorted_real_data = np.sort(self.real_data.flatten())[::-1]
        self.sorted_synthetic_data = np.sort(self.synthetic_data.flatten())[::-1]

        self.plot_load_duration_curve()
        magnitude_metric = self._EMD1()
        timing_metric = self.timing_analysis()
        return {
            "magnitude_metric": magnitude_metric,
            "timing_metric": timing_metric
        }


class HourlyProfilesByMonth(SyntheticMetric):
    """
    Analyze hourly consumption profiles by month and compute flexibility distances.
    
    For each month, computes the average hourly consumption profile (24-hour) for both real 
    and synthetic data by averaging all windows within that month. Then computes the 
    Flexibility Distance between the two average profiles (1-to-1 pairing). Visualizes both 
    average profiles with 10th--90th percentile bands derived from all windows.
    
    Returns mean flexibility distance across all months as the metric value.
    
    Args:
        window_size (int): Hours per window (default: 24)
        plot_filename (str): Name of plot file (default: hourly_profiles_by_month.png)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size: int = 24, plot_filename: str = 'hourly_profiles_by_month.pdf'):
        self.window_size = window_size
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Hourly Profiles by Month (Flexibility Distance)"
    
    def _uses_phase_first_flatten(self) -> bool:
        """Check if model uses phase-first flattening for multi-phase data."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')
    
    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        """Extract consumption feature (first feature) from potentially multi-phase flattened data.
        
        Handles:
        - [N, 24] single-feature data
        - [N, 24*F] multi-feature flattened data (phase-first or time-first)
        
        Returns:
            np.ndarray: [N, 24] consumption feature array
        """
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr  # Already consumption only

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        # Extract first feature (consumption) depending on flatten order
        if self._uses_phase_first_flatten():
            # Phase-first: [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]
        else:
            # Time-first: [t0_f0, t0_f1, ..., t1_f0, ...]
            return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Compute hourly profiles by month and return mean flexibility distance.
        
        For each month:
        1. Average all real windows in that month into one profile
        2. Average all synthetic windows in that month into one profile
        3. Compute FlexibilityDistance between the two average profiles
        4. Visualize with percentile bands from all windows
        """
        try:
            # Validate inputs
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError(f"Data not injected: real_data={self.real_data is not None}, synthetic_data={self.synthetic_data is not None}")
            
            # Extract consumption features
            real_data = self._extract_consumption_feature(self.real_data)
            synthetic_data = self._extract_consumption_feature(self.synthetic_data)
            
            # Equalize lengths between real and synthetic data
            n = min(len(real_data), len(synthetic_data))
            real_data = real_data[:n]
            synthetic_data = synthetic_data[:n]
            
            # Handle timestamps
            if self.timestamps is None:
                # Fallback: generate dummy timestamps
                self.timestamps = np.array([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)])
            else:
                # Make sure timestamps match data length
                self.timestamps = np.asarray(self.timestamps)[:n]
            
            # Extract month from timestamps
            months = self._extract_months()
            
            # Verify all arrays have matching lengths
            if len(months) != n:
                raise ValueError(f"Array length mismatch: months={len(months)}, data={n}")
            
            # Group by month and compute flexibility distances
            unique_months = sorted(np.unique(months[months > 0]))
            if len(unique_months) == 0:
                # If no valid months found, use month 1 for all
                months = np.ones_like(months)
                unique_months = [1]
            
            flexibility_distances_by_month = {}
            
            for month in unique_months:
                month_mask = (months == month)
                real_month = real_data[month_mask]  # All windows with timestamp in this month
                synth_month = synthetic_data[month_mask]  # All windows with timestamp in this month
                
                if len(real_month) < 1 or len(synth_month) < 1:
                    continue
                
                # Average the windows into single profiles
                real_avg = np.mean(real_month, axis=0)  # [24]
                synth_avg = np.mean(synth_month, axis=0)  # [24]
                
                # Compute FlexibilityDistance between the two average profiles (1-to-1)
                try:
                    fd = FlexibilityDistance.compute_fd_between_series(
                        real_avg, synth_avg, tau='auto', normalize=True
                    )
                    flexibility_distances_by_month[month] = fd
                except Exception:
                    flexibility_distances_by_month[month] = np.nan
            
            if not flexibility_distances_by_month:
                raise ValueError("No flexibility distances computed")
            
            # Compute mean FD across months
            all_fds = np.array([v for v in flexibility_distances_by_month.values()])
            mean_fd = float(np.nanmean(all_fds))
            
            # Plot
            self._plot_hourly_profiles(real_data, synthetic_data, months, flexibility_distances_by_month)
            
            # Save results
            _save_standalone_metric_result(
                metric_key='HourlyProfilesByMonth',
                payload={
                    'value': mean_fd,
                    'unique_months': list(unique_months),
                    'n_months': len(unique_months),
                    'mean_fd': mean_fd,
                    'fd_by_month': {int(m): float(v) for m, v in flexibility_distances_by_month.items()},
                    'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
                },
                output_file=self.output_path or 'models/eval_results.json'
            )
            
            return mean_fd
        except Exception as e:
            print(f"[ERROR] HourlyProfilesByMonth.compute() failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  # Re-raise so framework sees the error
    
    
    def _extract_months(self) -> np.ndarray:
            """Extract month from timestamps."""
            months = []
            
            if self.timestamps is None:
                raise ValueError("Timestamps required for HourlyProfilesByMonth")
            
            for ts in self.timestamps:
                try:
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts)
                    elif isinstance(ts, str):
                        dt = datetime.fromisoformat(ts)
                    elif isinstance(ts, datetime):
                        dt = ts
                    else:
                        dt = datetime.fromtimestamp(int(ts))
                    months.append(dt.month)
                except Exception:
                    months.append(0)
            
            return np.array(months)
    

    def _plot_hourly_profiles(self, real_data, synthetic_data, months, 
                             flexibility_distances_by_month):
        """Plot hourly profiles by month.
        
        For each month:
        - Compute average profile from all real windows
        - Compute average profile from all synthetic windows
        - Plot both with 10th-90th percentile bands from raw windows
        """
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return
            
            unique_months = sorted(flexibility_distances_by_month.keys())
            num_months = len(unique_months)
            
            fig, axes = plt.subplots(num_months, 1, figsize=(12, 4 * num_months))
            if num_months == 1:
                axes = [axes]
            
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            for month_idx, month in enumerate(unique_months):
                ax = axes[month_idx]
                month_mask = (months == month)
                real_m = real_data[month_mask]
                synth_m = synthetic_data[month_mask]
                
                # Compute average profiles (1-to-1)
                real_avg = np.mean(real_m, axis=0)  # [24]
                synth_avg = np.mean(synth_m, axis=0)  # [24]
                
                # Compute percentile bands from all windows
                hours_list = np.arange(self.window_size)
                real_p10 = np.percentile(real_m, 10, axis=0)
                real_p90 = np.percentile(real_m, 90, axis=0)
                synth_p10 = np.percentile(synth_m, 10, axis=0)
                synth_p90 = np.percentile(synth_m, 90, axis=0)
                
                # Plot percentile bands first (background)
                ax.fill_between(hours_list, real_p10, real_p90, alpha=0.2, color='#1f77b4', label='Real (10th-90th %ile)')
                ax.fill_between(hours_list, synth_p10, synth_p90, alpha=0.2, color='#ff7f0e', label='Synthetic (10th-90th %ile)')
                
                # Plot average profiles
                ax.plot(hours_list, real_avg, '-o', label='Real (avg)', color='#1f77b4', linewidth=2, markersize=6)
                ax.plot(hours_list, synth_avg, '-o', label='Synthetic (avg)', color='#ff7f0e', linewidth=2, markersize=6)
                
                # Add FD value to title
                fd_value = flexibility_distances_by_month[month]
                month_name = month_names.get(month, f'Month {month}')
                ax.set_title(
                    f'{month_name} - Hourly Consumption Profiles (FD={fd_value:.4f}, n_real={len(real_m)}, n_synthetic={len(synth_m)})',
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                ax.set_ylabel('Consumption (kWh)', fontsize=12, fontweight='bold')
                ax.set_xticks(hours_list)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
                ax.set_xlim(-0.5, 23.5)
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, self.plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
            plt.close(fig)
        except Exception as e:
            import traceback
            traceback.print_exc()


class HourlyProfilesByDayofWeek(SyntheticMetric):
    """
    Analyze hourly consumption profiles by day of week and compute flexibility distances.
    
    For each day of week (Monday-Sunday), computes the average hourly consumption profile 
    (24-hour) for both real and synthetic data by averaging all windows within that day. 
    Then computes the Flexibility Distance between the two average profiles (1-to-1 pairing). 
    Visualizes both average profiles with 10th--90th percentile bands derived from all windows.
    
    Returns mean flexibility distance across all days of week as the metric value.
    
    Args:
        window_size (int): Hours per window (default: 24)
        plot_filename (str): Name of plot file (default: hourly_profiles_by_dayofweek.pdf)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    output_path: str = None
    
    def __init__(self, window_size: int = 24, plot_filename: str = 'hourly_profiles_by_dayofweek.pdf'):
        self.window_size = window_size
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Hourly Profiles by Day of Week (Flexibility Distance)"
    
    def _uses_phase_first_flatten(self) -> bool:
        """Check if model uses phase-first flattening for multi-phase data."""
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')
    
    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        """Extract consumption feature (first feature) from potentially multi-phase flattened data.
        
        Handles:
        - [N, 24] single-feature data
        - [N, 24*F] multi-feature flattened data (phase-first or time-first)
        
        Returns:
            np.ndarray: [N, 24] consumption feature array
        """
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr  # Already consumption only

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        # Extract first feature (consumption) depending on flatten order
        if self._uses_phase_first_flatten():
            # Phase-first: [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]
        else:
            # Time-first: [t0_f0, t0_f1, ..., t1_f0, ...]
            return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Compute hourly profiles by day of week and return mean flexibility distance.
        
        For each day of week (0-6):
        1. Average all real windows on that day into one profile
        2. Average all synthetic windows on that day into one profile
        3. Compute FlexibilityDistance between the two average profiles
        4. Visualize with percentile bands from all windows
        """
        try:
            # Validate inputs
            if self.real_data is None or self.synthetic_data is None:
                raise ValueError(f"Data not injected: real_data={self.real_data is not None}, synthetic_data={self.synthetic_data is not None}")
            
            # Extract consumption features
            real_data = self._extract_consumption_feature(self.real_data)
            synthetic_data = self._extract_consumption_feature(self.synthetic_data)
            
            # Equalize lengths between real and synthetic data
            n = min(len(real_data), len(synthetic_data))
            real_data = real_data[:n]
            synthetic_data = synthetic_data[:n]
            
            # Handle timestamps
            if self.timestamps is None:
                # Fallback: generate dummy timestamps
                self.timestamps = np.array([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)])
            else:
                # Make sure timestamps match data length
                self.timestamps = np.asarray(self.timestamps)[:n]
            
            # Extract day of week from timestamps
            days_of_week = self._extract_days_of_week()
            
            # Verify all arrays have matching lengths
            if len(days_of_week) != n:
                raise ValueError(f"Array length mismatch: days_of_week={len(days_of_week)}, data={n}")
            
            # Group by day of week and compute flexibility distances
            unique_days = sorted(np.unique(days_of_week[days_of_week >= 0]))
            if len(unique_days) == 0:
                # If no valid days found, use day 0 for all
                days_of_week = np.zeros_like(days_of_week)
                unique_days = [0]
            
            flexibility_distances_by_day = {}
            
            for day in unique_days:
                day_mask = (days_of_week == day)
                real_day = real_data[day_mask]  # All windows with timestamp on this day
                synth_day = synthetic_data[day_mask]  # All windows with timestamp on this day
                
                if len(real_day) < 1 or len(synth_day) < 1:
                    continue
                
                # Average the windows into single profiles
                real_avg = np.mean(real_day, axis=0)  # [24]
                synth_avg = np.mean(synth_day, axis=0)  # [24]
                
                # Compute FlexibilityDistance between the two average profiles (1-to-1)
                try:
                    fd = FlexibilityDistance.compute_fd_between_series(
                        real_avg, synth_avg, tau='auto', normalize=True
                    )
                    flexibility_distances_by_day[day] = fd
                except Exception:
                    flexibility_distances_by_day[day] = np.nan
            
            if not flexibility_distances_by_day:
                raise ValueError("No flexibility distances computed")
            
            # Compute mean FD across days
            all_fds = np.array([v for v in flexibility_distances_by_day.values()])
            mean_fd = float(np.nanmean(all_fds))
            
            # Plot
            self._plot_hourly_profiles(real_data, synthetic_data, days_of_week, flexibility_distances_by_day)
            
            # Save results
            _save_standalone_metric_result(
                metric_key='HourlyProfilesByDayofWeek',
                payload={
                    'value': mean_fd,
                    'unique_days': list(unique_days),
                    'n_days': len(unique_days),
                    'mean_fd': mean_fd,
                    'fd_by_day': {int(d): float(v) for d, v in flexibility_distances_by_day.items()},
                    'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
                },
                output_file=self.output_path or 'models/eval_results.json'
            )
            
            return mean_fd
        except Exception as e:
            print(f"[ERROR] HourlyProfilesByDayofWeek.compute() failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise  # Re-raise so framework sees the error
    
    def _extract_days_of_week(self) -> np.ndarray:
        """Extract day of week from timestamps (0=Monday, 6=Sunday)."""
        days = []
        
        if self.timestamps is None:
            raise ValueError("Timestamps required for HourlyProfilesByDayofWeek")
        
        for ts in self.timestamps:
            try:
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts)
                elif isinstance(ts, str):
                    dt = datetime.fromisoformat(ts)
                elif isinstance(ts, datetime):
                    dt = ts
                else:
                    dt = datetime.fromtimestamp(int(ts))
                days.append(dt.weekday())  # 0=Monday, 6=Sunday
            except Exception:
                days.append(-1)
        
        return np.array(days)
    
    def _plot_hourly_profiles(self, real_data, synthetic_data, days_of_week, 
                             flexibility_distances_by_day):
        """Plot hourly profiles by day of week.
        
        For each day:
        - Compute average profile from all real windows
        - Compute average profile from all synthetic windows
        - Plot both with 10th-90th percentile bands from raw windows
        """
        try:
            output_dir = _resolve_output_directory(self.output_path)
            if not output_dir:
                return
            
            unique_days = sorted(flexibility_distances_by_day.keys())
            num_days = len(unique_days)
            
            fig, axes = plt.subplots(num_days, 1, figsize=(12, 4 * num_days))
            if num_days == 1:
                axes = [axes]
            
            day_names = {
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            }
            
            for day_idx, day in enumerate(unique_days):
                ax = axes[day_idx]
                day_mask = (days_of_week == day)
                real_d = real_data[day_mask]
                synth_d = synthetic_data[day_mask]
                
                # Compute average profiles (1-to-1)
                real_avg = np.mean(real_d, axis=0)  # [24]
                synth_avg = np.mean(synth_d, axis=0)  # [24]
                
                # Compute percentile bands from all windows
                hours_list = np.arange(self.window_size)
                real_p10 = np.percentile(real_d, 10, axis=0)
                real_p90 = np.percentile(real_d, 90, axis=0)
                synth_p10 = np.percentile(synth_d, 10, axis=0)
                synth_p90 = np.percentile(synth_d, 90, axis=0)
                
                # Plot percentile bands first (background)
                ax.fill_between(hours_list, real_p10, real_p90, alpha=0.2, color='#1f77b4', label='Real (10th-90th %ile)')
                ax.fill_between(hours_list, synth_p10, synth_p90, alpha=0.2, color='#ff7f0e', label='Synthetic (10th-90th %ile)')
                
                # Plot average profiles
                ax.plot(hours_list, real_avg, '-o', label='Real (avg)', color='#1f77b4', linewidth=2, markersize=6)
                ax.plot(hours_list, synth_avg, '-o', label='Synthetic (avg)', color='#ff7f0e', linewidth=2, markersize=6)
                
                # Add FD value to title
                fd_value = flexibility_distances_by_day[day]
                day_name = day_names.get(day, f'Day {day}')
                ax.set_title(
                    f'{day_name} - Hourly Consumption Profiles (FD={fd_value:.4f}, n_real={len(real_d)}, n_synthetic={len(synth_d)})',
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                ax.set_ylabel('Consumption (kWh)', fontsize=12, fontweight='bold')
                ax.set_xticks(hours_list)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
                ax.set_xlim(-0.5, 23.5)
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, self.plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', format='pdf')
            plt.close(fig)
        except Exception as e:
            import traceback
            traceback.print_exc()


class EVConditionAccuracy(SyntheticMetric):
    """Evaluate if synthetic profiles correctly represent their generation EV condition.
    
    This metric:
    1. Reads the conditional tensor used for generation mapping to [No EV, 1ph, 2ph, 3ph]
    2. Modifies scale (kW -> W) if needed so thresholds match original Jupyter notebooks
    3. Runs detect_platos_unified_dP2_std on each profile
    4. Computes accuracy of correct EV classification.
    """
    
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
            
        n_samples = min(len(self.synthetic_data), len(conds))
        if n_samples == 0:
            return 0.0

        # Adjust scale to Watts if data is likely in kW
        # Max hourly consumption in houses > 500 is extremely rare, likely kW if < 100
        scale_factor = 1.0
        if np.max(self.synthetic_data[:100]) < 100.0:
            scale_factor = 1000.0

        correct_count = 0

        for i in range(n_samples):
            # Conditions are [No EV, 1ph, 2ph, 3ph]
            cond_idx = int(np.argmax(conds[i]))
            if cond_idx == 0:
                expected_label = 'no'
            elif cond_idx == 1:
                expected_label = '1ph'
            elif cond_idx == 2:
                expected_label = '2ph'
            elif cond_idx == 3:
                expected_label = '3ph'
            else:
                expected_label = 'no'

            try:
                # Shape: [L, F]
                phase_data = self._unpack_synthetic_profile(self.synthetic_data[i]) * scale_factor
                
                dates = pd.date_range(start='2024-01-01', periods=len(phase_data), freq=f"{self.interval_minutes}min")
                df_profile = pd.DataFrame({
                    "active_power_p14_l1_test": phase_data[:, 0] if phase_data.shape[1] > 0 else 0,
                    "active_power_p14_l2_test": phase_data[:, 1] if phase_data.shape[1] > 1 else 0,
                    "active_power_p14_l3_test": phase_data[:, 2] if phase_data.shape[1] > 2 else 0,
                }, index=dates)

                events = detect_platos_unified_dP2_std(
                    df_profile,
                    "test",
                    start_threshold_min=3000,
                    start_threshold_max=7500,
                    interval_minutes=self.interval_minutes
                )

                if events is not None and not events.empty:
                    detected_label = str(events.iloc[0]['phase_type']).lower()
                else:
                    detected_label = 'no'

            except Exception:
                detected_label = 'no'

            if phase_data.shape[1] == 1:
                # Relaxed match for aggregated (binary presence match)
                is_expected_ev = expected_label in ['1ph', '2ph', '3ph']
                is_detected_ev = detected_label != 'no'
                if is_expected_ev == is_detected_ev:
                    correct_count += 1
            else:
                if detected_label == expected_label:
                    correct_count += 1
                
        return float(correct_count) / float(n_samples)
