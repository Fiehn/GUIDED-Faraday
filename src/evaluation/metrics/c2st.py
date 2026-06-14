import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.metrics.base.base import _resolve_output_directory, _to_json_serializable

_COLOR_REAL = "#1f77b4"
_COLOR_SYN  = "#ff7f0e"
_CHANNEL_PALETTE = ["#5f88b0", "#c6ad8f", "#7ea06b", "#d9a066"]
_PHASE_LABELS = ["Phase 1 (low)", "Phase 2 (mid)", "Phase 3 (high)"]


class _C2ST_1D(nn.Module):
    """Lightweight 1D CNN for classifying single-channel real (1) vs synthetic (0) windows."""

    def __init__(self, window_size: int = 24):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(4)
        self.fc    = nn.Linear(64 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1)).squeeze(-1)


class C2STSuite:
    """Classifier Two-Sample Test analysis on single-channel load windows.

    For phased [N, 3, 24] input: trains four independent 1D-CNN classifiers —
    one on each ranked phase and one on the aggregated (summed) signal.

    For single-channel [N, 24] or [N, 1, 24] input: trains one classifier.

    Each draw gets a fresh train/test split (seeded with seed+draw_idx) so no
    sample appears in both training and evaluation.  train_fraction of each
    class is used for training; the remainder for AUC evaluation.

    Saves
    -----
    <output_dir>/c2st_results.json          – AUC scalars (one per channel)
    <output_dir>/c2st/<channel>.pdf         – score distribution + ROC per channel
    <output_dir>/c2st/c2st_overview.pdf     – all channels in one figure
    """

    def __init__(
        self,
        output_path: str,
        n_draws: int = 1,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = 'cpu',
        seed: int = 42,
        train_fraction: float = 0.25,
    ):
        self.output_path    = output_path
        self.n_draws        = max(1, int(n_draws))
        self.epochs         = epochs
        self.batch          = batch_size
        self.lr             = lr
        self.device         = device
        self.seed           = seed
        self.train_fraction = train_fraction

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, real: np.ndarray, synth_draws: list) -> dict:
        """Train C2ST classifiers on draw 0, infer on all draws.

        Parameters
        ----------
        real:         [N, 24] or [N, 1, 24] or [N, 3, 24]
        synth_draws:  list of K arrays matching real's shape

        Returns
        -------
        dict  ``"C2ST AUC - <channel>": float``  (K=1) or
              ``{mean, std, values}``             (K>1)
        """
        real = np.array(real, dtype=np.float32)
        synth_draws = [np.array(s, dtype=np.float32) for s in synth_draws]

        channels = self._extract_channels(real)

        output_dir = _resolve_output_directory(self.output_path)
        plot_dir: str | None = None
        if output_dir:
            plot_dir = os.path.join(output_dir, 'c2st')
            os.makedirs(plot_dir, exist_ok=True)

        draw_results: list = []
        ch0_scores: dict = {}  # kept for overview plot

        for draw_idx, synth in enumerate(synth_draws):
            synth_channels = self._extract_channels(synth)
            print(f"\n[C2STSuite] Training classifiers (draw {draw_idx + 1}/{len(synth_draws)})...")
            draw_result: dict = {}
            for ch_name in channels:
                real_ch = channels[ch_name]
                syn_ch  = synth_channels[ch_name]
                y_true, y_score = self._fit_and_eval(real_ch, syn_ch, draw_idx)
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                except Exception:
                    auc = 0.5
                draw_result[f"C2ST AUC - {ch_name}"] = auc
                print(f"  [{ch_name}] AUC = {auc:.4f}")

                if draw_idx == 0:
                    ch0_scores[ch_name] = (y_true, y_score)
                    if plot_dir:
                        plot_path = os.path.join(
                            plot_dir,
                            f"c2st_{ch_name.replace(' ', '_').lower()}.pdf",
                        )
                        try:
                            self._plot_channel(ch_name, y_true, y_score, auc, plot_path)
                        except Exception as e:
                            print(f"  [C2STSuite] per-channel plot failed ({ch_name}): {e}")

            draw_results.append(draw_result)

        # Generate combined overview on draw 0
        if plot_dir:
            overview_path = os.path.join(plot_dir, "c2st_overview.pdf")
            try:
                self._plot_overview(list(channels.keys()), ch0_scores, overview_path)
            except Exception as e:
                print(f"  [C2STSuite] overview plot failed: {e}")

        # Aggregate across draws
        if len(draw_results) == 1:
            result: dict = draw_results[0]
        else:
            result = {}
            for k in draw_results[0]:
                vals = [float(dr.get(k, float('nan'))) for dr in draw_results]
                arr  = np.array(vals, dtype=float)
                result[k] = {
                    "mean":   float(np.nanmean(arr)),
                    "std":    float(np.nanstd(arr)),
                    "values": vals,
                }

        if output_dir:
            out_file = os.path.join(output_dir, 'c2st_results.json')
            with open(out_file, 'w') as f:
                json.dump(_to_json_serializable(result), f, indent=2)
            print(f"  Saved to {out_file}")

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_channels(self, data: np.ndarray) -> dict:
        """Return ordered {channel_name: [N, 24]} from [N,24], [N,1,24], or [N,3,24]."""
        if data.ndim == 2:
            return {"Single Channel": data}
        if data.shape[1] == 1:
            return {"Single Channel": data[:, 0, :]}
        return {
            "Aggregate":      data.sum(axis=1),        # [N, 24]
            _PHASE_LABELS[0]: data[:, 0, :],
            _PHASE_LABELS[1]: data[:, 1, :],
            _PHASE_LABELS[2]: data[:, 2, :],
        }

    def _fit_and_eval(
        self,
        real_ch: np.ndarray,
        syn_ch: np.ndarray,
        draw_idx: int,
    ) -> tuple:
        """Split train_fraction/rest, train CNN, evaluate on held-out set. Returns (y_true, y_score)."""
        seed = self.seed + draw_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        n = min(len(real_ch), len(syn_ch))
        real_ch = real_ch[rng.permutation(len(real_ch))[:n]]
        syn_ch  = syn_ch [rng.permutation(len(syn_ch)) [:n]]

        n_train = max(1, int(n * self.train_fraction))

        real_train, real_test = real_ch[:n_train], real_ch[n_train:]
        syn_train,  syn_test  = syn_ch [:n_train], syn_ch [n_train:]

        # Train
        X_tr = np.concatenate(
            [real_train[:, None, :], syn_train[:, None, :]], axis=0
        ).astype(np.float32)
        y_tr = np.concatenate([np.ones(n_train, dtype=np.float32), np.zeros(n_train, dtype=np.float32)])
        perm = rng.permutation(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        X_t = torch.tensor(X_tr, device=self.device)
        y_t = torch.tensor(y_tr, device=self.device)

        cnn = _C2ST_1D(window_size=X_tr.shape[2]).to(self.device)
        opt = torch.optim.Adam(cnn.parameters(), lr=self.lr)

        cnn.train()
        for _ in range(self.epochs):
            perm_t = torch.randperm(len(X_t), device=self.device)
            for start in range(0, len(X_t), self.batch):
                idx  = perm_t[start:start + self.batch]
                loss = F.binary_cross_entropy_with_logits(cnn(X_t[idx]), y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        cnn.eval()

        # Evaluate on held-out test set
        n_test = len(real_test)
        X_te = np.concatenate(
            [real_test[:, None, :], syn_test[:, None, :]], axis=0
        ).astype(np.float32)
        y_true = np.concatenate([np.ones(n_test, dtype=int), np.zeros(n_test, dtype=int)])

        X_te_t = torch.tensor(X_te, device=self.device)
        logits = []
        with torch.no_grad():
            for start in range(0, len(X_te_t), self.batch):
                logits.append(cnn(X_te_t[start:start + self.batch]).cpu().numpy())
        y_score = torch.sigmoid(torch.tensor(np.concatenate(logits))).numpy()
        return y_true, y_score

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_channel(
        self,
        ch_name: str,
        y_true: np.ndarray,
        y_score: np.ndarray,
        auc: float,
        path: str,
    ) -> None:
        real_sc = y_score[y_true == 1]
        syn_sc  = y_score[y_true == 0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        x  = np.linspace(0.0, 1.0, 300)
        for scores, color, label in [
            (real_sc, _COLOR_REAL, "Real"),
            (syn_sc,  _COLOR_SYN,  "Synthetic"),
        ]:
            clipped = np.clip(scores, 1e-6, 1 - 1e-6)
            if len(clipped) > 3:
                kde = gaussian_kde(clipped)
                y   = kde(x)
                ax.plot(x, y, color=color, lw=2, label=label)
                ax.fill_between(x, y, alpha=0.15, color=color)
        ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7, label="Random (0.5)")
        ax.set_xlabel("P(Synthetic)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Classifier Score Distributions", fontsize=13)
        ax.legend(fontsize=11)

        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=_COLOR_SYN, lw=2, label=f"AUC = {auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color=_COLOR_SYN)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Receiver Operating Characteristic", fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle("Classifier Two-Sample Test Diagnostics", fontsize=15)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_overview(
        self,
        channels: list,
        ch_scores: dict,
        path: str,
    ) -> None:
        """n_channels × 2 figure: score distribution + ROC per channel row."""
        n = len(channels)
        fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

        for row, ch_name in enumerate(channels):
            y_true, y_score = ch_scores[ch_name]
            real_sc = y_score[y_true == 1]
            syn_sc  = y_score[y_true == 0]
            try:
                auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc = 0.5

            ax = axes[row, 0]
            x  = np.linspace(0.0, 1.0, 300)
            for scores, color, label in [
                (real_sc, _COLOR_REAL, "Real"),
                (syn_sc,  _COLOR_SYN,  "Synthetic"),
            ]:
                clipped = np.clip(scores, 1e-6, 1 - 1e-6)
                if len(clipped) > 3:
                    kde = gaussian_kde(clipped)
                    y   = kde(x)
                    ax.plot(x, y, color=color, lw=2, label=label)
                    ax.fill_between(x, y, alpha=0.15, color=color)
            ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
            ax.set_xlabel("P(Synthetic)", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title("Score Distributions", fontsize=12)
            ax.legend(fontsize=10)

            ax = axes[row, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            color = _CHANNEL_PALETTE[row % len(_CHANNEL_PALETTE)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.10, color=color)
            ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_title("ROC Curve", fontsize=12)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle("Classifier Two-Sample Test Analysis", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)