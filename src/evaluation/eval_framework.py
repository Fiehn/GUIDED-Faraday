import numpy as np
import torch
import torch.nn as nn
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.evaluation.metrics.base.base import EvaluationMetric, ReconstructionMetric, SyntheticMetric, PrivacyMetric, _save_standalone_metric_result

import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.model.generators import SyntheticGenerator

from src.evaluation.metrics.flexibility_distance import FlexibilityDistance
from src.evaluation.metrics.acf import ACFDistribution
from src.evaluation.metrics.tstr import TSTR
from src.evaluation.metrics.mmd import MaximumMeanDiscrepancy, SetMaximumMeanDiscrepancy
from src.evaluation.metrics.hourly_avg import HourlyProfilesByMonth, HourlyProfilesByDayofWeek
from src.evaluation.metrics.aggregates import AggregatedProfiles, PerConditionEval
from src.evaluation.metrics.peaks import PeakMagnitude, PeakTiming
from src.evaluation.metrics.condition_acc import EVConditionAccuracy
from src.evaluation.metrics.basic_reconstructions import ReconstructionRMSE, ReconstructionLogLikelihood

from src.evaluation.metrics.base.base import _sanitize_numeric_array, _align_phases_by_mse, _align_phases_to_reference_mean, _to_json_serializable, _save_standalone_metric_result, _flatten_windows, _resolve_output_directory

from src.evaluation.metrics.mia import MembershipInferenceAttack
from tabulate import tabulate

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
        secondary_model: Optional[nn.Module] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        n_member_samples: int = 1000,
        mia_reconstruction_samples: int = 1,
        preserve_condition_order: bool = False,
    ):
        """Initialize evaluation framework.

        Args:
            model: Trained model
            dataloader: DataLoader for evaluation data (test split)
            normalizer: Data normalizer
            conditioner: Conditioner object (for CVAE)
            device: Device to run on ('cpu' or 'cuda')
            train_dataloader: DataLoader for training data (member windows for MIA).
                When provided, MIA is automatically registered in register_all().
            n_member_samples: Maximum number of member windows to collect for MIA.
            mia_reconstruction_samples: Reconstruction passes averaged per member
                window.  Set to 1 for VAE-family, higher (e.g. 5) for diffusion.
            preserve_condition_order: When True, synthetic samples are generated
                using the exact same condition vectors in the same order as the
                collected real windows (no shuffling). Use for holdout eval where
                direct 1-to-1 condition matching is required.
        """
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.conditioner = conditioner
        self.device = device

        # Preserve secondary model normalizer (if provided) so real data can be
        # denormalized with the correct feature-dimension when evaluating
        # phase-aware models.
        self.secondary_normalizer = getattr(secondary_model, 'normalizer', None) if secondary_model is not None else None

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

        self.train_dataloader = train_dataloader
        self.n_member_samples = int(n_member_samples)
        self.mia_reconstruction_samples = int(mia_reconstruction_samples)
        self.preserve_condition_order = bool(preserve_condition_order)

        # Metrics storage
        self.reconstruction_metrics: Dict[str, ReconstructionMetric] = {}
        self.synthetic_metrics: Dict[str, SyntheticMetric] = {}
        self.privacy_metrics: Dict[str, PrivacyMetric] = {}
        self.results: Dict[str, float] = {}
        self.baseline_results: Dict[str, float] = {}

        self._real_data: np.ndarray = None
        self._reconstructed_data: np.ndarray = None
        self._synthetic_data: np.ndarray = None
        self._timestamps: np.ndarray = None
        self._meter_ids: np.ndarray = None
        self._train_data: Optional[np.ndarray] = None
        self._train_reconstructed_data: Optional[np.ndarray] = None

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
            PeakMagnitude(),
            PeakTiming(),
            HourlyProfilesByMonth(),
            HourlyProfilesByDayofWeek(),
            PerConditionEval(),
            EVConditionAccuracy(),
        ])
        if phased:
            self.register_metric(SetMaximumMeanDiscrepancy())
        else:
            self.register_metric(MaximumMeanDiscrepancy())

        has_train = (
            self.train_dataloader is not None
            or (isinstance(self.prepared_data, dict) and self.prepared_data.get('train_data') is not None)
        )
        if has_train:
            k = self.mia_reconstruction_samples if self.mia_reconstruction_samples > 1 else (
                5 if self.generator._is_diffusion else 1
            )
            self.register_metric(MembershipInferenceAttack(n_reconstruction_samples=k))

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
        """Flatten in the same order used during training when possible.

        SHAPER:
          x  : [batch_size, seq_len, n_features]  (3D) or [batch_size, flat_dim]  (2D pass-through)
          ->  : [batch_size, n_features * seq_len]  (phase-first) or [batch_size, flat_dim]
        """
        if x.dim() == 3:
            if hasattr(self.model, "_to_phase_first"):
                try:
                    x_phase = self.model._to_phase_first(x)  # SHAPER: [batch_size, n_features, seq_len]
                    return x_phase.reshape(x_phase.shape[0], -1)  # SHAPER: [batch_size, n_features, seq_len] -> [batch_size, n_features * seq_len]
                except Exception:
                    return x.reshape(x.shape[0], -1)  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size, seq_len * n_features]
            return x.reshape(x.shape[0], -1)  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size, seq_len * n_features]
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
        """Inverse transform 2D/3D tensors while preserving [B, L, F] or [B, F, L] layout.

        SHAPER:
          x  : [batch_size, seq_len, n_features]  (time-first 3D)
               | [batch_size, n_features, seq_len]  (phase-first 3D)
               | [batch_size, seq_len * n_features]  (flattened 2D)
          ->  : same shape as x (denormalized)
        """
        # Prefer a normalizer whose feature-dimension matches the tensor shape.
        # Try secondary model normalizer first (if available), then fall back to
        # the primary normalizer provided to the evaluator.
        preferred_norm = None
        for candidate in (self.secondary_normalizer, self.normalizer):
            if candidate is None or not hasattr(candidate, 'inverse_transform'):
                continue
            stats = getattr(candidate, 'mean', None)
            feat_dim = None
            try:
                if stats is not None:
                    import numpy as _np
                    feat_dim = int(_np.asarray(stats).reshape(-1).shape[0])
            except Exception:
                feat_dim = None
            # If we can infer a feature dim and it matches the tensor last/second axis, prefer this normalizer.
            if x.dim() == 3 and feat_dim is not None and (x.shape[-1] == feat_dim or x.shape[1] == feat_dim):
                preferred_norm = candidate
                break

        # If none matched by feature-dim, fall back to primary normalizer if present.
        if preferred_norm is None:
            if self.normalizer is None or not hasattr(self.normalizer, 'inverse_transform'):
                return x
            preferred_norm = self.normalizer

        inv = preferred_norm.inverse_transform
        
        # 1. Handle 2D tensors [B, D]
        if x.dim() == 2:
            # For 2D tensors, we need to know if they are already flattened.
            # If so, they must be unflattened to apply normalization correctly to phases.
            seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
            n_feat = int(getattr(self.model, 'num_features', 0) or 0)
            
            if seq_len > 0 and n_feat > 0 and x.shape[1] == seq_len * n_feat:
                # Assume time-first flattening for unflattening [B, L, F]
                # (This is consistent with how we handle recon in _reconstruct)
                x_3d = x.reshape(-1, seq_len, n_feat)  # SHAPER: [batch_size, seq_len * n_features] -> [batch_size, seq_len, n_features]
                y_3d = self._inverse_transform_for_metrics(x_3d)  # SHAPER: [batch_size, seq_len, n_features] (denormalized)
                return y_3d.reshape_as(x)  # SHAPER: [batch_size, seq_len * n_features]
            
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
            return inv(x.reshape(-1, feature_dim)).reshape_as(x)  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size * seq_len, n_features] -> [batch_size, seq_len, n_features]

        # Case [B, F, L]: Features are the second dimension (phased dataloader output)
        if x.shape[1] == feature_dim:
            x_last = x.movedim(1, -1).contiguous()  # SHAPER: [batch_size, n_features, seq_len] -> [batch_size, seq_len, n_features]
            y_last = inv(x_last.reshape(-1, feature_dim)).reshape_as(x_last)  # SHAPER: [batch_size, seq_len, n_features] (denormalized)
            return y_last.movedim(-1, 1).contiguous()  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size, n_features, seq_len]

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
            train_data = self.prepared_data.get('train_data')
            train_recon = self.prepared_data.get('train_reconstructed_data')
            if train_data is not None:
                self._train_data = _sanitize_numeric_array(np.asarray(train_data))
            if train_recon is not None:
                self._train_reconstructed_data = _sanitize_numeric_array(np.asarray(train_recon))
            return

        print("  [1/3] Collecting reconstructions from dataloader...", end='', flush=True)
        t_recon = time.time()
        real_batches = []
        recon_batches = []
        condition_batches = []
        meter_id_batches = []
        timestamp_batches = []

        # Cap real-data collection at n_synthetic_samples so we never run more
        # encoder forward-passes than we have synthetic samples to compare against.
        n_to_collect = self.n_synthetic_samples
        n_collected = 0

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                if n_collected >= n_to_collect:
                    break

                data, meter_ids, timestamps, agg = self._unpack_batch(batch)

                data = data.to(self.device)
                if self._is_cvae() and self.conditioner is not None:
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
                n_collected += len(data)

        # Trim to exactly n_to_collect (last batch may overshoot by up to batch_size - 1)
        self._real_data = _sanitize_numeric_array(np.concatenate(real_batches, axis=0)[:n_to_collect])  # SHAPER: [n_to_collect, n_features * seq_len]
        self._reconstructed_data = _sanitize_numeric_array(np.concatenate(recon_batches, axis=0)[:n_to_collect])  # SHAPER: [n_to_collect, n_features * seq_len]
        self._meter_ids = np.concatenate(meter_id_batches, axis=0)[:n_to_collect] if meter_id_batches else None  # SHAPER: [n_to_collect]
        self._timestamps = np.concatenate(timestamp_batches, axis=0)[:n_to_collect] if timestamp_batches else None  # SHAPER: [n_to_collect]
        print(f" done — {len(self._real_data)} windows ({time.time() - t_recon:.1f}s)")

        if self._is_cvae() and condition_batches:
            all_conditions = torch.cat(condition_batches, dim=0)  # SHAPER: [n_windows, condition_dim]
            # Build parallel metadata arrays so secondary models (e.g. HierarchicalSplitter)
            # can construct their own conditions from the same real-data windows.
            all_meter_ids  = np.concatenate(meter_id_batches,  axis=0) if meter_id_batches  else None
            all_timestamps = np.concatenate(timestamp_batches, axis=0) if timestamp_batches else None
            n = self.n_synthetic_samples
            if self.preserve_condition_order:
                # Holdout eval: use conditions in collection order so synthetic sample i
                # is generated from the same condition as real sample i.
                if len(all_conditions) >= n:
                    self._synthetic_conditions = all_conditions[:n]
                    self._synthetic_meter_ids  = all_meter_ids[:n]  if all_meter_ids  is not None else None
                    self._synthetic_timestamps = all_timestamps[:n] if all_timestamps is not None else None
                else:
                    repeats = (n // len(all_conditions)) + 1
                    self._synthetic_conditions = all_conditions.repeat(repeats, 1)[:n]
                    self._synthetic_meter_ids  = np.tile(all_meter_ids,  repeats)[:n] if all_meter_ids  is not None else None
                    self._synthetic_timestamps = np.tile(all_timestamps, repeats)[:n] if all_timestamps is not None else None
            elif len(all_conditions) >= n:
                idx = torch.randperm(len(all_conditions))[:n]
                self._synthetic_conditions = all_conditions[idx]  # SHAPER: [n_synthetic_samples, condition_dim]
                idx_np = idx.numpy()
                self._synthetic_meter_ids  = all_meter_ids[idx_np]  if all_meter_ids  is not None else None
                self._synthetic_timestamps = all_timestamps[idx_np] if all_timestamps is not None else None
            else:
                repeats = (n // len(all_conditions)) + 1
                self._synthetic_conditions = all_conditions.repeat(repeats, 1)[:n]  # SHAPER: [n_synthetic_samples, condition_dim]
                self._synthetic_meter_ids  = np.tile(all_meter_ids,  repeats)[:n] if all_meter_ids  is not None else None
                self._synthetic_timestamps = np.tile(all_timestamps, repeats)[:n] if all_timestamps is not None else None
        else:
            self._synthetic_conditions = None
            self._synthetic_meter_ids  = None
            self._synthetic_timestamps = None

        print(f"  [2/3] Generating {self.n_synthetic_samples} synthetic samples...", end='', flush=True)
        t_synth = time.time()
        self._synthetic_data = _sanitize_numeric_array(self._generate_synthetic(self.n_synthetic_samples))
        print(f" done ({time.time() - t_synth:.1f}s)")

        # Collect member windows (training data) for MIA privacy metrics.
        if self.train_dataloader is not None and self.privacy_metrics:
            print("  [3/3] Collecting member windows for MIA...", end='', flush=True)
            t_mia = time.time()
            train_real_batches = []
            train_recon_batches = []
            n_collected = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.train_dataloader:
                    if n_collected >= self.n_member_samples:
                        break
                    try:
                        data, meter_ids, timestamps, _agg = self._unpack_batch(batch)
                    except Exception:
                        continue
                    data = data.to(self.device)
                    data_denorm = self._inverse_transform_for_metrics(data)
                    x_flat = self._flatten_for_metrics(data_denorm)
                    train_real_batches.append(x_flat.cpu().numpy())
                    recon = self._reconstruct_mean(
                        data, self.mia_reconstruction_samples,
                        meter_id=meter_ids, timestamps=timestamps,
                    )
                    train_recon_batches.append(recon.cpu().numpy())
                    n_collected += len(data)
            if train_real_batches:
                self._train_data = _sanitize_numeric_array(
                    np.concatenate(train_real_batches, axis=0)[:self.n_member_samples]
                )
                self._train_reconstructed_data = _sanitize_numeric_array(
                    np.concatenate(train_recon_batches, axis=0)[:self.n_member_samples]
                )
            print(f" done — {n_collected} member windows ({time.time() - t_mia:.1f}s)")

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

        SHAPER:
          x     : [batch_size, seq_len, n_features]
          recon : [n_recon_samples, batch_size, n_features * seq_len]  (from generator)
          ->    : [batch_size, n_features * seq_len]
        """
        recon = self.generator.reconstruct(x, meter_id=meter_id, timestamp=timestamps)  # SHAPER: [n_recon_samples, batch_size, n_features * seq_len]
        if recon is None:
            raise ValueError("Generator returned None for reconstruction")

        # Expected from generator: [S, B, D]. Keep one reconstruction sample for metrics.
        if recon.dim() == 3:
            recon = recon[0]  # SHAPER: [n_recon_samples, batch_size, n_features * seq_len] -> [batch_size, n_features * seq_len]

        # If still time/feature shaped [B, L, F], flatten to [B, D].
        if recon.dim() == 3:
            recon = self._flatten_for_metrics(recon)
        elif recon.dim() == 2:
            # For 2D tensors, if they are time-first flattened, re-flatten to phase-first
            seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
            n_feat = int(getattr(self.model, 'num_features', 0) or 0)
            if seq_len > 0 and n_feat > 0 and recon.shape[1] == seq_len * n_feat:
                recon_3d = recon.reshape(recon.shape[0], seq_len, n_feat)  # SHAPER: [batch_size, seq_len * n_features] -> [batch_size, seq_len, n_features]
                recon = self._flatten_for_metrics(recon_3d)  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size, n_features * seq_len]
        elif recon.dim() != 2:
            raise ValueError(f"Unexpected reconstruction shape: {tuple(recon.shape)}")

        # For models with known [L, F], enforce the same phase-first flattening
        # convention used for real data so phased extraction is consistent.
        seq_len = int(getattr(self.model, 'sequence_length', 0) or 0)
        n_feat = int(getattr(self.model, 'num_features', 0) or 0)
        if recon.dim() == 2 and seq_len > 0 and n_feat > 0 and recon.shape[1] == seq_len * n_feat:
            recon_lf = recon.reshape(recon.shape[0], seq_len, n_feat)  # SHAPER: [batch_size, seq_len * n_features] -> [batch_size, seq_len, n_features]
            recon = self._flatten_for_metrics(recon_lf)  # SHAPER: [batch_size, seq_len, n_features] -> [batch_size, n_features * seq_len]

        return recon  # SHAPER: [batch_size, n_features * seq_len]

    def _reconstruct_mean(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
        meter_id=None,
        timestamps=None,
    ) -> torch.Tensor:
        """Reconstruct x and average over n_samples passes.

        For VAE-family (n_samples=1) this is identical to _reconstruct.
        For diffusion (n_samples>1) the stochastic reconstructions are averaged
        to produce a more stable per-window error estimate for MIA.

        SHAPER:
          x      : [batch_size, seq_len, n_features]
          recons : [n_samples, batch_size, n_features * seq_len]
          ->     : [batch_size, n_features * seq_len]
        """
        if n_samples <= 1:
            return self._reconstruct(x, meter_id=meter_id, timestamps=timestamps)
        recons = torch.stack([  # SHAPER: [n_samples, batch_size, n_features * seq_len]
            self._reconstruct(x, meter_id=meter_id, timestamps=timestamps)
            for _ in range(n_samples)
        ])
        return recons.mean(dim=0)  # SHAPER: [n_samples, batch_size, n_features * seq_len] -> [batch_size, n_features * seq_len]

    def _generate_synthetic(self, n_samples: int) -> np.ndarray:
        """Sample from prior using real conditions to match the real data distribution.
         For CVAE, conditions are sampled from the real conditions collected during reconstruction preparation.
         For non-CVAE, conditions are not used.
         Args:
            n_samples (int): Number of synthetic samples to generate
         Returns:
             np.ndarray: Synthetic samples of shape (n_samples, features)

        SHAPER:
          synthetic (4D) : [n_generator_samples, batch_size, S1, S2]  where S1*S2 = seq_len * n_features
          synthetic (3D) : [n_samples, seq_len, n_features]  or  [n_samples, n_features, seq_len]  or  [n_samples, batch_size, flat_dim]
          ->             : [n_samples, n_features * seq_len]
        """
        if self._is_cvae():
            if self._synthetic_conditions is None:
                raise ValueError("CVAE synthetic generation requires prepared conditions, but none were found.")
            conditions = self._synthetic_conditions.to(self.device)
            synthetic, _ = self.generator.generate(
                n_samples,
                conditions=conditions,
                # Pass window-aligned metadata so secondary models (e.g. HierarchicalSplitter)
                # can build their own conditions without sharing the primary's condition space.
                meter_ids=getattr(self, '_synthetic_meter_ids', None),
                timestamps=getattr(self, '_synthetic_timestamps', None),
            )
        else:
            synthetic, _ = self.generator.generate(n_samples)

        if synthetic is None:
            raise ValueError("Generator returned None for synthetic samples")

        # Flatten samples and batches into a single sample dimension [Samples, Features]
        if synthetic.dim() == 4:
             # Handle [N, B, L, F] or [N, B, F, L]
             N, B, S1, S2 = synthetic.shape
             synthetic = synthetic.reshape(N * B, S1, S2)  # SHAPER: [n_generator_samples, batch_size, S1, S2] -> [n_generator_samples * batch_size, S1, S2]
             synthetic = self._flatten_for_metrics(synthetic)  # SHAPER: [n_generator_samples * batch_size, S1, S2] -> [n_generator_samples * batch_size, S1 * S2]
        elif synthetic.dim() == 3:
             # Check if it is [N, L, F], [N, F, L], or [N, B, D]
             n_feat = int(getattr(self.model, 'num_features', 0) or 0)
             seq_len = int(getattr(self.model, 'sequence_length', 0) or self.window_size or 0)

             if seq_len > 0 and synthetic.shape[1] == seq_len:
                 # [N, L, F] -> [N, F, L]
                 synthetic = synthetic.transpose(1, 2).reshape(synthetic.shape[0], -1)  # SHAPER: [n_samples, seq_len, n_features] -> [n_samples, n_features * seq_len]
             elif seq_len > 0 and synthetic.shape[2] == seq_len:
                 # [N, F, L]
                 synthetic = synthetic.reshape(synthetic.shape[0], -1)  # SHAPER: [n_samples, n_features, seq_len] -> [n_samples, n_features * seq_len]
             else:
                 # Assume [N, B, D]. Flatten to [N*B, D].
                 synthetic = synthetic.reshape(-1, synthetic.shape[-1])  # SHAPER: [n_samples, batch_size, flat_dim] -> [n_samples * batch_size, flat_dim]
                 # If the resulting 2D is time-first, re-flatten to phase-first
                 if n_feat > 0 and seq_len > 0 and synthetic.shape[1] == seq_len * n_feat:
                     synth_3d = synthetic.reshape(-1, seq_len, n_feat)  # SHAPER: [n_samples * batch_size, seq_len * n_features] -> [n_samples * batch_size, seq_len, n_features]
                     synthetic = self._flatten_for_metrics(synth_3d)  # SHAPER: [n_samples * batch_size, seq_len, n_features] -> [n_samples * batch_size, n_features * seq_len]

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

        for metric in self.privacy_metrics.values():
            metric.member_data = self._train_data
            metric.non_member_data = self._real_data
            metric.member_reconstructed_data = self._train_reconstructed_data
            metric.non_member_reconstructed_data = self._reconstructed_data
            metric.model = self.model
            metric.device = self.device
            metric.normalizer = self.normalizer
            metric.conditioner = self.conditioner
            metric.output_path = self.output_path

    def register_metric(self, metric: EvaluationMetric) -> None:
        """Register an evaluation metric.
        Args:
            metric: An instance of ReconstructionMetric, SyntheticMetric, or PrivacyMetric
        """
        if isinstance(metric, type):
            if issubclass(metric, EvaluationMetric):
                metric = metric()
            else:
                raise ValueError(f"Metric class {metric} must inherit from EvaluationMetric")

        if isinstance(metric, ReconstructionMetric):
            self.reconstruction_metrics[metric.name] = metric
        elif isinstance(metric, SyntheticMetric):
            self.synthetic_metrics[metric.name] = metric
        elif isinstance(metric, PrivacyMetric):
            self.privacy_metrics[metric.name] = metric
        else:
            metric_label = getattr(metric, 'name', metric.__class__.__name__)
            raise ValueError(
                f"Metric {metric_label} must inherit from ReconstructionMetric, SyntheticMetric, or PrivacyMetric"
            )
    
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
        t_total = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.evaluation_title}")
        print(f"{'='*60}")

        # Prepare data for metrics that need it (reconstructions, synthetic samples)
        self._prepare_data()
        self._inject_data_into_metrics()

        n_real  = 0 if self._real_data  is None else int(len(self._real_data))
        n_synth = 0 if self._synthetic_data is None else int(len(self._synthetic_data))
        print(f"  Data ready: {n_real} real windows, {n_synth} synthetic samples\n")

        self.results = {}

        all_metrics = (
            list(self.reconstruction_metrics.items())
            + list(self.synthetic_metrics.items())
            + list(self.privacy_metrics.items())
        )
        n_total = len(all_metrics)

        for i, (name, metric) in enumerate(all_metrics, 1):
            print(f"  [{i:>{len(str(n_total))}}/{n_total}] {name}...", end='', flush=True)
            t_metric = time.time()
            try:
                value = metric.compute()
                self.results[name] = value
                elapsed = time.time() - t_metric
                value_str = f"{float(value):.4f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value)
                print(f" {value_str}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t_metric
                print(f" ERROR ({elapsed:.1f}s): {e}")
                if self.verbose:
                    import traceback; traceback.print_exc()
                self.results[name] = None

        print(f"\n  Total evaluation time: {time.time() - t_total:.1f}s")
        self._print_results_table()

        return self.results

    def evaluate_averaged(self, n_draws: int = 1) -> Dict[str, float]:
        """Like evaluate(), but averages n_draws independent synthetic draws before
        running metrics. All metrics — not just the profile plot — see the averaged
        synthetic. n_draws=1 is identical to evaluate().
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")

        t_total = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.evaluation_title}")
        print(f"{'='*60}")

        self._prepare_data()

        actual_draws = 1
        if n_draws > 1 and self._synthetic_data is not None:
            print(f"  Averaging {n_draws} synthetic draws...", end='', flush=True)
            draws = [np.asarray(self._synthetic_data)]
            with torch.no_grad():
                for draw_idx in range(1, n_draws):
                    try:
                        extra = self._generate_synthetic(self.n_synthetic_samples)
                        draws.append(np.asarray(extra))
                    except Exception as e:
                        print(f"\n  WARNING: draw {draw_idx + 1}/{n_draws} failed: {e}. Using {len(draws)} draw(s).")
                        import traceback; traceback.print_exc()
                        break
            actual_draws = len(draws)
            self._synthetic_data = _sanitize_numeric_array(np.mean(np.stack(draws, axis=0), axis=0))
            print(f" done — averaged {actual_draws} draw(s).")
        elif n_draws > 1:
            print(f"  WARNING: n_draws={n_draws} requested but no synthetic data was generated; cannot average.")

        self._inject_data_into_metrics()

        n_real  = 0 if self._real_data  is None else int(len(self._real_data))
        n_synth = 0 if self._synthetic_data is None else int(len(self._synthetic_data))
        print(f"  Data ready: {n_real} real windows, {n_synth} synthetic samples (avg of {actual_draws} draw(s))\n")

        self.results = {}
        all_metrics = (
            list(self.reconstruction_metrics.items())
            + list(self.synthetic_metrics.items())
            + list(self.privacy_metrics.items())
        )
        n_total = len(all_metrics)
        for i, (name, metric) in enumerate(all_metrics, 1):
            print(f"  [{i:>{len(str(n_total))}}/{n_total}] {name}...", end='', flush=True)
            t_metric = time.time()
            try:
                value = metric.compute()
                self.results[name] = value
                elapsed = time.time() - t_metric
                value_str = f"{float(value):.4f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value)
                print(f" {value_str}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t_metric
                print(f" ERROR ({elapsed:.1f}s): {e}")
                if self.verbose:
                    import traceback; traceback.print_exc()
                self.results[name] = None

        print(f"\n  Total evaluation time: {time.time() - t_total:.1f}s")
        self._print_results_table()
        return self.results

    def evaluate_reconstruction_only(self) -> Dict[str, float]:
        """Run only reconstruction metrics.
        Returns:
            Dict[str, float]: Dictionary mapping reconstruction metric names to their computed values
        """
        t_total = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.evaluation_title}  [reconstruction only]")
        print(f"{'='*60}")

        self.results = {}
        n_total = len(self.reconstruction_metrics)
        for i, (name, metric) in enumerate(self.reconstruction_metrics.items(), 1):
            print(f"  [{i:>{len(str(n_total))}}/{n_total}] {name}...", end='', flush=True)
            t_metric = time.time()
            try:
                value = metric.compute()
                self.results[name] = value
                elapsed = time.time() - t_metric
                value_str = f"{float(value):.4f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value)
                print(f" {value_str}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t_metric
                print(f" ERROR ({elapsed:.1f}s): {e}")
                if self.verbose:
                    import traceback; traceback.print_exc()
                self.results[name] = None

        self.baseline_results = self._compute_mean_baseline_results(reconstruction_only=True) if self.include_mean_baseline else {}

        print(f"\n  Total evaluation time: {time.time() - t_total:.1f}s")
        self._print_interpretation_guide()
        self._print_results_table()
        return self.results

    def evaluate_synthetic_only(self) -> Dict[str, float]:
        """Run only synthetic data metrics.
        Returns:
            Dict[str, float]: Dictionary mapping synthetic metric names to their computed values
        """
        t_total = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.evaluation_title}  [synthetic only]")
        print(f"{'='*60}")

        self.results = {}
        n_total = len(self.synthetic_metrics)
        for i, (name, metric) in enumerate(self.synthetic_metrics.items(), 1):
            print(f"  [{i:>{len(str(n_total))}}/{n_total}] {name}...", end='', flush=True)
            t_metric = time.time()
            try:
                value = metric.compute()
                self.results[name] = value
                elapsed = time.time() - t_metric
                value_str = f"{float(value):.4f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value)
                print(f" {value_str}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t_metric
                print(f" ERROR ({elapsed:.1f}s): {e}")
                if self.verbose:
                    import traceback; traceback.print_exc()
                self.results[name] = None

        self.baseline_results = self._compute_mean_baseline_results(synthetic_only=True) if self.include_mean_baseline else {}

        print(f"\n  Total evaluation time: {time.time() - t_total:.1f}s")
        self._print_interpretation_guide()
        self._print_results_table()
        return self.results

    def _metric_direction(self, metric_name: str) -> str:
        """Return optimization direction for display/benchmarking."""
        if metric_name == "TSTR (Train on Synthetic, Test on Real)":
            return "target_1"
        if "MIA" in metric_name:
            return "target_0.5"
        if "Log-Likelihood" in metric_name or metric_name == "Visual Evaluations":
            return "higher"
        return "lower"

    def _print_results_table(self) -> None:
        """Print results in a formatted table."""
        print("\nEvaluation summary")

        table_data = []
        for name, value in self.results.items():
            model_str = "ERROR" if value is None else (
                "N/A (no train data)" if value is None and name in self.privacy_metrics
                else (f"{float(value):.6f}" if isinstance(value, (float, int, np.floating, np.integer)) else str(value))
            )
            goal = self._metric_direction(name)
            if goal == "target_1":
                goal_label = "target=1"
            elif goal == "target_0.5":
                goal_label = "target=0.5 (privacy)"
            elif goal == "higher":
                goal_label = "higher"
            else:
                goal_label = "lower"
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
        secondary_model: Optional[nn.Module] = None,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
        n_member_samples: int = 1000,
        mia_reconstruction_samples: int = 1,
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
        self.train_dataloader = train_dataloader
        self.n_member_samples = int(n_member_samples)
        self.mia_reconstruction_samples = int(mia_reconstruction_samples)

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
            train_dataloader=self.train_dataloader,
            n_member_samples=self.n_member_samples,
            mia_reconstruction_samples=self.mia_reconstruction_samples,
        )
        # Register privacy metrics so _prepare_data collects member windows.
        if self.train_dataloader is not None:
            k = self.mia_reconstruction_samples if self.mia_reconstruction_samples > 1 else (
                5 if framework.generator._is_diffusion else 1
            )
            framework.register_metric(MembershipInferenceAttack(n_reconstruction_samples=k))
        framework._prepare_data()
        return framework

    def _extract_phase_first(self, flat_data: np.ndarray, label: str) -> np.ndarray:
        """
        SHAPER:
          flat_data : [n_samples, n_features * window_size]  (2D) or [n_samples, n_features, window_size]  (3D pass-through)
          ->        : [n_samples, n_features, window_size]
        """
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
        return arr.reshape(arr.shape[0], n_features, self.window_size)  # SHAPER: [n_samples, n_features * window_size] -> [n_samples, n_features, window_size]

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
            'train_data': base._train_data,
            'train_reconstructed_data': base._train_reconstructed_data,
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
# Metrics not worthy of their own file
# ============================================================================

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
        """Generate PCA plots for real vs synthetic data.

        SHAPER:
          real_data      : [min(n_real, 500), feature_dim]
          synthetic_data : [min(n_synth, 500), feature_dim]
          pca_real       : [min(n_real, 500), n_components]  n_components <= 3
          pca_synthetic  : [min(n_synth, 500), n_components]
        """
        real_data = self.real_data[:500] if len(self.real_data) > 500 else self.real_data  # SHAPER: [min(n_real, 500), feature_dim]
        synthetic_data = self.synthetic_data[:500] if len(self.synthetic_data) > 500 else self.synthetic_data  # SHAPER: [min(n_synth, 500), feature_dim]
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Apply PCA
        pca = PCA(n_components=min(3, real_data.shape[1], synthetic_data.shape[1]))
        pca_real = pca.fit_transform(real_data)  # SHAPER: [min(n_real, 500), feature_dim] -> [min(n_real, 500), n_components]
        pca_synthetic = pca.transform(synthetic_data)  # SHAPER: [min(n_synth, 500), feature_dim] -> [min(n_synth, 500), n_components]
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
        """Generate t-SNE plots for real vs synthetic data.

        SHAPER:
          real_data      : [min(n_real, 500), feature_dim]
          synthetic_data : [min(n_synth, 500), feature_dim]
          combined       : [min(n_real, 500) + min(n_synth, 500), feature_dim]
          labels         : [min(n_real, 500) + min(n_synth, 500)]
          tsne_result    : [min(n_real, 500) + min(n_synth, 500), 2]
        """
        # Use smaller subset for t-SNE
        real_data = self.real_data[:500] if len(self.real_data) > 500 else self.real_data  # SHAPER: [min(n_real, 500), feature_dim]
        synthetic_data = self.synthetic_data[:500] if len(self.synthetic_data) > 500 else self.synthetic_data  # SHAPER: [min(n_synth, 500), feature_dim]
        
        # Verify same number of features
        if real_data.shape[1] != synthetic_data.shape[1]:
            min_features = min(real_data.shape[1], synthetic_data.shape[1])
            real_data = real_data[:, :min_features]
            synthetic_data = synthetic_data[:, :min_features]
        
        # Combine for t-SNE
        combined = np.vstack([real_data, synthetic_data])  # SHAPER: [min(n_real, 500) + min(n_synth, 500), feature_dim]
        labels = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])  # SHAPER: [min(n_real, 500) + min(n_synth, 500)]

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
        tsne_result = tsne.fit_transform(combined)  # SHAPER: [min(n_real, 500) + min(n_synth, 500), feature_dim] -> [min(n_real, 500) + min(n_synth, 500), 2]
        
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

        SHAPER:
          data (input)   : [n_samples, window_size * n_features]
          data_reshaped  : [n_samples, window_size, inferred_n_features]
          data (sliced)  : [n_samples, window_size, 1]  consumption channel only
          mean_profile   : [window_size, 1]
          std_profile    : [window_size, 1]
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
        data_reshaped = data.reshape(-1, self.window_size, inferred_n_features)  # SHAPER: [n_samples, window_size * inferred_n_features] -> [n_samples, window_size, inferred_n_features]
        data = data_reshaped[:, :, 0:1]  # SHAPER: [n_samples, window_size, inferred_n_features] -> [n_samples, window_size, 1]  consumption channel

        mean_profile = np.mean(data, axis=0)  # SHAPER: [n_samples, window_size, 1] -> [window_size, 1]
        std_profile = np.std(data, axis=0)  # SHAPER: [n_samples, window_size, 1] -> [window_size, 1]
        
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
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
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

