import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.metrics.base.base import SyntheticMetric, _sanitize_numeric_array, _resolve_output_directory

_COLOR_REAL = "#111111"
_COLOR_SYN  = "#5f88b0"
_PHASE_LABELS = ["P1 (low)", "P2 (mid)", "P3 (high)"]


class CrossPhaseCorrelationError(SyntheticMetric):
    """Frobenius norm of the difference between [3,3] empirical cross-phase correlation matrices.

    Correlation (not covariance) is used so the metric is dimensionless and
    scale-invariant under any Normalizer; the diagonal is always 1 so the norm
    measures only off-diagonal cross-phase structure.

    Expects real_data / synthetic_data as [N, 3, 24] or [N, 72] (phase-first flat).
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    @property
    def name(self) -> str:
        return "Cross-Phase Correlation Error"

    def _correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        N, F, L = data.shape
        obs = data.transpose(0, 2, 1).reshape(N * L, F)
        return np.corrcoef(obs.T)

    def _to_3d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], 3, -1)

    def compute(self) -> float:
        real = self._to_3d(_sanitize_numeric_array(self.real_data))
        syn  = self._to_3d(_sanitize_numeric_array(self.synthetic_data))
        R_real = self._correlation_matrix(real)
        R_syn  = self._correlation_matrix(syn)

        output_dir = _resolve_output_directory(self.output_path)
        if output_dir:
            plot_path = os.path.join(output_dir, "cross_phase_correlation_matrices.pdf")
            if not os.path.exists(plot_path):
                try:
                    self._plot(R_real, R_syn, plot_path)
                except Exception as e:
                    print(f"[CrossPhaseCorrelationError] plot failed: {e}")

        return float(np.linalg.norm(R_real - R_syn, 'fro'))

    def _plot(self, R_real: np.ndarray, R_syn: np.ndarray, path: str) -> None:
        diff = R_real - R_syn
        lim = max(abs(diff).max(), 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = _PHASE_LABELS

        for ax, mat, title, vmin, vmax, cmap in [
            (axes[0], R_real, "Real",      -1,    1,    "RdBu_r"),
            (axes[1], R_syn,  "Synthetic", -1,    1,    "RdBu_r"),
            (axes[2], diff,   "Real - Syn",-lim,  lim,  "RdBu_r"),
        ]:
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
            ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=9)
            ax.set_title(title, fontsize=11)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if abs(mat[i, j]) > 0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Cross-Phase Correlation Matrices", fontsize=13)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


class CrossCorrelationDistribution(SyntheticMetric):
    """KS test on pairwise CCF value distributions at lags 0..max_lag, averaged across 3 phase-pairs.

    Analogous to ACFDistribution but for cross-phase pairs instead of within-phase
    autocorrelation.  At lag=0 this tests whether the distribution of instantaneous
    cross-phase correlations is preserved — complementary to CrossPhaseCorrelationError,
    which tests only the mean.

    Expects real_data / synthetic_data as [N, 3, 24] or [N, 72] (phase-first flat).
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    PAIRS     = [(0, 1), (0, 2), (1, 2)]
    PAIR_LBLS = ["P1 x P2", "P1 x P3", "P2 x P3"]

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    @property
    def name(self) -> str:
        return "Cross-Correlation Distribution (KS)"

    def _to_3d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], 3, -1)

    def _ccf_values(self, data: np.ndarray) -> np.ndarray:
        # returns [N, n_pairs, max_lag+1]
        N, _, L = data.shape
        result = np.zeros((N, len(self.PAIRS), self.max_lag + 1), dtype=np.float64)
        for pi, (i, j) in enumerate(self.PAIRS):
            xi = data[:, i, :].astype(np.float64)
            xj = data[:, j, :].astype(np.float64)
            xi_n = (xi - xi.mean(1, keepdims=True)) / xi.std(1, keepdims=True).clip(1e-12)
            xj_n = (xj - xj.mean(1, keepdims=True)) / xj.std(1, keepdims=True).clip(1e-12)
            for tau in range(self.max_lag + 1):
                if tau == 0:
                    result[:, pi, tau] = np.mean(xi_n * xj_n, axis=1)
                else:
                    result[:, pi, tau] = np.mean(xi_n[:, :-tau] * xj_n[:, tau:], axis=1)
        np.nan_to_num(result, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return result

    def compute(self) -> float:
        real = self._to_3d(_sanitize_numeric_array(self.real_data))
        syn  = self._to_3d(_sanitize_numeric_array(self.synthetic_data))

        real_ccf = self._ccf_values(real)  # [N, 3, max_lag+1]
        syn_ccf  = self._ccf_values(syn)   # [M, 3, max_lag+1]

        ks_stats = []
        for pi in range(len(self.PAIRS)):
            for tau in range(self.max_lag + 1):
                r_vals = real_ccf[:, pi, tau]
                s_vals = syn_ccf[:, pi, tau]
                r_vals = r_vals[np.isfinite(r_vals)]
                s_vals = s_vals[np.isfinite(s_vals)]
                if len(r_vals) == 0 or len(s_vals) == 0:
                    ks_stats.append(1.0)
                    continue
                ks_stat, _ = stats.ks_2samp(r_vals, s_vals)
                ks_stats.append(float(ks_stat) if np.isfinite(ks_stat) else 1.0)

        output_dir = _resolve_output_directory(self.output_path)
        if output_dir:
            plot_path = os.path.join(output_dir, "cross_phase_ccf_profile.pdf")
            if not os.path.exists(plot_path):
                try:
                    self._plot(real_ccf, syn_ccf, plot_path)
                except Exception as e:
                    print(f"[CrossCorrelationDistribution] plot failed: {e}")
            dist_path = os.path.join(output_dir, "cross_phase_ccf_distributions.pdf")
            if not os.path.exists(dist_path):
                try:
                    self._plot_distributions(real_ccf, syn_ccf, dist_path)
                except Exception as e:
                    print(f"[CrossCorrelationDistribution] distribution plot failed: {e}")

        return float(np.mean(ks_stats))

    def _plot(self, real_ccf: np.ndarray, syn_ccf: np.ndarray, path: str) -> None:
        lags = np.arange(self.max_lag + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for pi, (ax, lbl) in enumerate(zip(axes, self.PAIR_LBLS)):
            r = real_ccf[:, pi, :]   # [N, max_lag+1]
            s = syn_ccf[:, pi, :]    # [M, max_lag+1]

            r_mean = r.mean(0); r_p25 = np.percentile(r, 25, axis=0); r_p75 = np.percentile(r, 75, axis=0)
            s_mean = s.mean(0); s_p25 = np.percentile(s, 25, axis=0); s_p75 = np.percentile(s, 75, axis=0)

            ax.plot(lags, r_mean, color=_COLOR_REAL, lw=2,   label="Real")
            ax.fill_between(lags, r_p25, r_p75, color=_COLOR_REAL, alpha=0.15)
            ax.plot(lags, s_mean, color=_COLOR_SYN,  lw=2,   label="Synthetic")
            ax.fill_between(lags, s_p25, s_p75, color=_COLOR_SYN,  alpha=0.20)

            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_title(lbl, fontsize=11)
            ax.set_xlabel("Lag τ (hours)", fontsize=10)
            ax.set_xticks(lags)
            if pi == 0:
                ax.set_ylabel("Cross-Correlation", fontsize=10)
            ax.legend(fontsize=9)

        fig.suptitle("Cross-Phase CCF Profile  (mean ± p25/p75 across windows)", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_distributions(self, real_ccf: np.ndarray, syn_ccf: np.ndarray, path: str) -> None:
        """Grouped box plots showing the full distribution of CCF values per lag and phase pair."""
        n_lags = self.max_lag + 1
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for pi, (ax, lbl) in enumerate(zip(axes, self.PAIR_LBLS)):
            r = real_ccf[:, pi, :]  # [N, n_lags]
            s = syn_ccf[:, pi, :]   # [M, n_lags]

            positions_r = np.arange(n_lags) * 2.5 - 0.42
            positions_s = np.arange(n_lags) * 2.5 + 0.42

            bp_r = ax.boxplot(
                [r[:, tau][np.isfinite(r[:, tau])] for tau in range(n_lags)],
                positions=positions_r, widths=0.72, patch_artist=True,
                whis=(5, 95), showfliers=False,
                medianprops=dict(color='white', lw=2),
                whiskerprops=dict(color=_COLOR_REAL, lw=1.2),
                capprops=dict(color=_COLOR_REAL, lw=1.2),
                boxprops=dict(facecolor=_COLOR_REAL, alpha=0.75),
            )
            bp_s = ax.boxplot(
                [s[:, tau][np.isfinite(s[:, tau])] for tau in range(n_lags)],
                positions=positions_s, widths=0.72, patch_artist=True,
                whis=(5, 95), showfliers=False,
                medianprops=dict(color='white', lw=2),
                whiskerprops=dict(color=_COLOR_SYN, lw=1.2),
                capprops=dict(color=_COLOR_SYN, lw=1.2),
                boxprops=dict(facecolor=_COLOR_SYN, alpha=0.75),
            )

            ax.axhline(0, color='gray', lw=0.8, ls='--')
            ax.set_xticks(np.arange(n_lags) * 2.5)
            ax.set_xticklabels([f"τ={tau}" for tau in range(n_lags)], fontsize=9)
            ax.set_title(lbl, fontsize=11)
            if pi == 0:
                ax.set_ylabel("CCF Value", fontsize=10)
            ax.legend(handles=[
                Patch(facecolor=_COLOR_REAL, alpha=0.75, label='Real'),
                Patch(facecolor=_COLOR_SYN,  alpha=0.75, label='Synthetic'),
            ], fontsize=9)

        fig.suptitle("Cross-Phase CCF Distributions per Lag  (box: IQR, whiskers: 5th–95th pct)", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


class SlicedWassersteinDistance(SyntheticMetric):
    """Mean 1D Wasserstein distance over n_projections random unit-vector projections of [72]-dim windows.

    Each [3, 24] window is flattened phase-first to [72] before projection.
    Captures joint distribution topology including non-linear cross-phase structure
    invisible to correlation metrics.

    Expects real_data / synthetic_data as [N, 3, 24] or [N, 72] (phase-first flat).
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(self, n_projections: int = 512, seed: int = 42):
        self.n_projections = n_projections
        self.seed = seed

    @property
    def name(self) -> str:
        return "Sliced Wasserstein Distance"

    def compute(self) -> float:
        real = _sanitize_numeric_array(self.real_data)
        syn  = _sanitize_numeric_array(self.synthetic_data)

        if real.ndim == 3:
            real = real.reshape(real.shape[0], -1)
        if syn.ndim == 3:
            syn = syn.reshape(syn.shape[0], -1)

        D = real.shape[1]
        rng = np.random.RandomState(self.seed)
        directions = rng.randn(self.n_projections, D)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)

        real_proj = real @ directions.T
        syn_proj  = syn  @ directions.T

        real_sorted = np.sort(real_proj, axis=0)
        syn_sorted  = np.sort(syn_proj,  axis=0)

        N, M = real_sorted.shape[0], syn_sorted.shape[0]
        if N != M:
            t   = np.arange(M, dtype=float)
            idx = np.linspace(0, M - 1, N)
            syn_sorted = np.column_stack([
                np.interp(idx, t, syn_sorted[:, k]) for k in range(self.n_projections)
            ])

        output_dir = _resolve_output_directory(self.output_path)
        if output_dir:
            pca_path = os.path.join(output_dir, "cross_phase_pca_joint.pdf")
            if not os.path.exists(pca_path):
                try:
                    self._plot_pca(real, syn, pca_path)
                except Exception as e:
                    print(f"[SlicedWassersteinDistance] PCA plot failed: {e}")

        return float(np.mean(np.abs(real_sorted - syn_sorted)))

    def _plot_pca(self, real: np.ndarray, syn: np.ndarray, path: str) -> None:
        """2D PCA scatter of the 72-dim joint phase windows (real vs synthetic)."""
        from sklearn.decomposition import PCA as _PCA

        n = min(3000, len(real), len(syn))
        rng = np.random.RandomState(self.seed)
        r_idx = rng.choice(len(real), n, replace=False)
        s_idx = rng.choice(len(syn),  n, replace=False)
        real_s, syn_s = real[r_idx], syn[s_idx]

        combined = np.vstack([real_s, syn_s])
        pca = _PCA(n_components=2, random_state=self.seed)
        proj = pca.fit_transform(combined)
        real_2d, syn_2d = proj[:n], proj[n:]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(real_2d[:, 0], real_2d[:, 1], c=_COLOR_REAL, alpha=0.25, s=6,
                   label=f"Real (n={n})")
        ax.scatter(syn_2d[:, 0], syn_2d[:, 1], c=_COLOR_SYN, alpha=0.25, s=6,
                   label=f"Synthetic (n={n})")
        ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=10)
        ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=10)
        ax.set_title("PCA of Joint 3-Phase Windows (72-dim)", fontsize=12)
        ax.legend(markerscale=3, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


class _C2ST_CNN(nn.Module):
    def __init__(self, window_size: int = 24):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(4)
        self.fc    = nn.Linear(64 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1)).squeeze(-1)


class ClassifierTwoSampleTest(SyntheticMetric):
    """AUC-ROC of a 1D-CNN (3 input channels) trained to classify real (1) vs synthetic (0) windows.

    fit(draw_idx=) splits each class train_fraction/1-train_fraction, trains on the train
    portion, and stores the held-out test split.  compute() evaluates on the stored test
    split — no sample appears in both training and evaluation.
    AUC = 0.5 means indistinguishable (good); AUC = 1.0 means perfectly separable (bad).
    Lower is better.

    Expects real_data / synthetic_data as [N, 3, 24] or [N, 72] (phase-first flat).
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None

    def __init__(
        self,
        window_size: int = 24,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = 'cpu',
        seed: int = 42,
        train_fraction: float = 0.25,
    ):
        self.window_size    = window_size
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.device         = device
        self.seed           = seed
        self.train_fraction = train_fraction
        self._cnn: _C2ST_CNN | None       = None
        self._test_X: np.ndarray | None   = None
        self._test_y: np.ndarray | None   = None

    @property
    def name(self) -> str:
        return "Classifier Two-Sample Test (AUC)"

    def _to_3d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr
        return arr.reshape(arr.shape[0], 3, -1)

    def fit(self, real_ref: np.ndarray, syn_ref: np.ndarray, draw_idx: int = 0) -> None:
        """Split train_fraction/rest per class, train, store held-out test split for compute()."""
        seed = self.seed + draw_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        real_ref = self._to_3d(_sanitize_numeric_array(real_ref))
        syn_ref  = self._to_3d(_sanitize_numeric_array(syn_ref))

        n   = min(len(real_ref), len(syn_ref))
        rng = np.random.RandomState(seed)

        real_ref = real_ref[rng.permutation(len(real_ref))[:n]]
        syn_ref  = syn_ref [rng.permutation(len(syn_ref)) [:n]]

        n_train = max(1, int(n * self.train_fraction))

        real_train, real_test = real_ref[:n_train], real_ref[n_train:]
        syn_train,  syn_test  = syn_ref [:n_train], syn_ref [n_train:]

        n_test = len(real_test)
        self._test_X = np.concatenate([real_test, syn_test], axis=0).astype(np.float32)
        self._test_y = np.concatenate([np.ones(n_test, dtype=int), np.zeros(n_test, dtype=int)])

        X_tr = np.concatenate([real_train, syn_train], axis=0).astype(np.float32)
        y_tr = np.concatenate([np.ones(n_train, dtype=np.float32), np.zeros(n_train, dtype=np.float32)])
        perm = rng.permutation(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        X_t = torch.tensor(X_tr, device=self.device)
        y_t = torch.tensor(y_tr, device=self.device)

        self._cnn = _C2ST_CNN(self.window_size).to(self.device)
        opt = torch.optim.Adam(self._cnn.parameters(), lr=self.lr)

        self._cnn.train()
        N_total = len(X_t)
        for _ in range(self.epochs):
            perm_t = torch.randperm(N_total, device=self.device)
            for start in range(0, N_total, self.batch_size):
                idx  = perm_t[start:start + self.batch_size]
                loss = F.binary_cross_entropy_with_logits(self._cnn(X_t[idx]), y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()

        self._cnn.eval()

    def compute(self) -> float:
        if self._cnn is None or self._test_X is None:
            print("[ClassifierTwoSampleTest] WARNING: fit() was not called before compute(); returning 0.5.")
            return 0.5

        X_t = torch.tensor(self._test_X, device=self.device)
        logits_list = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                logits_list.append(self._cnn(X_t[start:start + self.batch_size]).cpu().numpy())

        y_score = torch.sigmoid(torch.tensor(np.concatenate(logits_list))).numpy()
        y_true  = self._test_y

        output_dir = _resolve_output_directory(self.output_path)
        if output_dir:
            score_path = os.path.join(output_dir, "cross_phase_c2st_scores.pdf")
            if not os.path.exists(score_path):
                try:
                    self._plot_scores(y_true, y_score, score_path)
                except Exception as e:
                    print(f"[ClassifierTwoSampleTest] score plot failed: {e}")

        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception as e:
            print(f"[ClassifierTwoSampleTest] roc_auc_score failed: {e}; returning 0.5.")
            return 0.5

    def _plot_scores(self, y_true: np.ndarray, y_score: np.ndarray, path: str) -> None:
        """Score-distribution KDE and ROC curve side by side."""
        from scipy.stats import gaussian_kde

        real_sc = y_score[y_true == 1]
        syn_sc  = y_score[y_true == 0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: score distributions
        ax = axes[0]
        x = np.linspace(0.0, 1.0, 300)
        for scores, color, label in [(real_sc, _COLOR_REAL, "Real"), (syn_sc, _COLOR_SYN, "Synthetic")]:
            clipped = np.clip(scores, 1e-6, 1 - 1e-6)
            if len(clipped) > 3:
                kde = gaussian_kde(clipped)
                y = kde(x)
                ax.plot(x, y, color=color, lw=2, label=label)
                ax.fill_between(x, y, alpha=0.15, color=color)
        ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7, label="Random (0.5)")
        ax.set_xlabel("P(Synthetic)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Classifier Score Distributions", fontsize=11)
        ax.legend(fontsize=9)

        # Right panel: ROC curve
        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=_COLOR_SYN, lw=2, label=f"AUC = {auc_val:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.10, color=_COLOR_SYN)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title("Receiver Operating Characteristic", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle("Classifier Two-Sample Test Diagnostics", fontsize=13)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)