"""Reconstruction-based Membership Inference Attack (Hilprecht et al., PoPETS 2019).

Attack signal: per-window MSE between the original window and its reconstruction.
Members (training windows) are expected to reconstruct with lower error than
non-members (test windows).

Metrics reported (following arXiv:2407.11785 where applicable):
  - AUC-ROC      : standard MIA metric; 0.5 = random guess, 1.0 = perfect attack.
  - Precision    : fraction of predicted members that are true members on a
                   balanced dataset; 0.5 = random guess, 1.0 = perfect attack.
                   Threshold is the median attack score (top-50th percentile),
                   matching the balanced-dataset spirit of 2407.11785.

Known limitation: 24-point windows from smart meter data have heavily
overlapping train/test distributions, so AUC ≈ 0.5 does not necessarily
indicate strong privacy — it may simply reflect distributional similarity.
See arXiv:2407.11785 for a deeper discussion.

Extension point: replace ``member_data`` / ``member_reconstructed_data`` with
outlier-injected arrays (as in 2407.11785) before calling ``compute()`` to
run the improved attack without any code changes to this class.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.metrics.base.base import PrivacyMetric, _resolve_output_directory, _save_standalone_metric_result, _sanitize_numeric_array


class MembershipInferenceAttack(PrivacyMetric):
    """Reconstruction-based MIA (Hilprecht et al., PoPETS 2019).

    Args:
        n_reconstruction_samples: Number of reconstruction passes to average
            per window when computing member errors.  Use 1 for VAE-family
            models (deterministic decoder mean) and a higher value (e.g. 5)
            for diffusion models whose reconstruction is stochastic.  Only
            relevant when the framework is configured with
            ``mia_reconstruction_samples > 1``; the averaged reconstructions
            are pre-computed by EvalFramework and injected here.
    """

    def __init__(self, n_reconstruction_samples: int = 1):
        self.n_reconstruction_samples = n_reconstruction_samples

    @property
    def name(self) -> str:
        return "MIA AUC-ROC"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _per_window_mse(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        diff = _sanitize_numeric_array(original) - _sanitize_numeric_array(reconstructed)
        return np.mean(diff ** 2, axis=1)

    @staticmethod
    def _precision_at_median(labels: np.ndarray, scores: np.ndarray) -> float:
        """Precision when classifying top-50th-percentile scores as members.

        Follows the balanced-dataset precision reporting in arXiv:2407.11785:
        0.5 = random guess, 1.0 = perfect.
        """
        threshold = np.percentile(scores, 50)
        predicted = (scores >= threshold).astype(int)
        tp = int(np.sum((predicted == 1) & (labels == 1)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.5

    @staticmethod
    def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.1) -> float:
        idx = int(np.searchsorted(fpr, target_fpr))
        return float(tpr[min(idx, len(tpr) - 1)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Optional[float]:
        """Return AUC-ROC; save full results to the JSON sidecar.

        Returns None when member data has not been injected (e.g. no train
        loader was provided to EvalFramework).
        """
        if self.member_data is None or self.member_reconstructed_data is None:
            return None
        if self.non_member_data is None or self.non_member_reconstructed_data is None:
            return None

        member_errors = self._per_window_mse(
            self.member_data, self.member_reconstructed_data
        )
        non_member_errors = self._per_window_mse(
            self.non_member_data, self.non_member_reconstructed_data
        )

        # Balance the two classes so precision is interpretable.
        n = min(len(member_errors), len(non_member_errors))
        member_errors = member_errors[:n]
        non_member_errors = non_member_errors[:n]

        # Lower reconstruction error → higher attack score → predicted member.
        scores = np.concatenate([-member_errors, -non_member_errors])
        labels = np.concatenate([np.ones(n), np.zeros(n)])

        auc = float(roc_auc_score(labels, scores))
        precision = self._precision_at_median(labels, scores)

        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        tpr_at_01 = self._tpr_at_fpr(fpr_arr, tpr_arr, target_fpr=0.1)

        # Save ROC curve arrays for compare_models plotting.
        _output_dir = _resolve_output_directory(self.output_path or "models/eval_results.json")
        if _output_dir:
            try:
                _sidecar = os.path.join(_output_dir, "eval_mia_curve.json")
                _curve_data = {
                    "fpr": fpr_arr.tolist(),
                    "tpr": tpr_arr.tolist(),
                    "auc_roc": auc,
                    "precision_balanced": precision,
                    "tpr_at_fpr_0.1": tpr_at_01,
                    "n_members": n,
                    "n_non_members": n,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(_sidecar, "w") as _f:
                    json.dump(_curve_data, _f, indent=2)
            except Exception as _e:
                print(f"  [MIA] Warning: could not save curve sidecar: {_e}")

        _save_standalone_metric_result(
            metric_key="MembershipInferenceAttack",
            payload={
                "auc_roc": auc,
                "precision_balanced": precision,
                "tpr_at_fpr_0.1": tpr_at_01,
                "n_members": n,
                "n_non_members": n,
                "n_reconstruction_samples": self.n_reconstruction_samples,
                "member_error_mean": float(member_errors.mean()),
                "member_error_std": float(member_errors.std()),
                "non_member_error_mean": float(non_member_errors.mean()),
                "non_member_error_std": float(non_member_errors.std()),
                "note": (
                    "AUC near 0.5 may reflect distributional overlap between "
                    "train/test windows rather than genuine privacy. "
                    "See arXiv:2407.11785."
                ),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            output_file=self.output_path or "models/eval_results.json",
        )

        print(
            f"  MIA | AUC-ROC: {auc:.4f}  Precision: {precision:.4f}"
            f"  TPR@FPR=0.1: {tpr_at_01:.4f}"
            f"  (members={n}, non-members={n})"
        )

        return auc