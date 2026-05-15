"""GPU-native Gaussian Mixture Model as a PyTorch nn.Module.

Ported from OpenSynth (Apache-2.0): https://github.com/OpenSynth-energy/OpenSynth
All parameters stored as registered buffers so the GMM follows the model
device automatically (GPU training, CPU inference, etc.).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pure utility functions (operate on tensors, device-agnostic)
# ---------------------------------------------------------------------------

def _compute_precision_cholesky(
    covariances: torch.Tensor, reg: float = 1e-6
) -> torch.Tensor:
    """Compute precision matrices via Cholesky decomposition.

    Port of sklearn's _compute_precision_cholesky for full covariance type.
    Falls back to adding extra regularisation if a component is degenerate.
    """
    n_components, n_features, _ = covariances.shape
    precisions_chol = torch.empty_like(covariances)
    eye = torch.eye(n_features, device=covariances.device, dtype=covariances.dtype)

    for k, cov in enumerate(covariances):
        try:
            L = torch.linalg.cholesky(cov, upper=False)
        except torch.linalg.LinAlgError:
            try:
                L = torch.linalg.cholesky(cov + eye * reg, upper=False)
            except torch.linalg.LinAlgError:
                raise ValueError(
                    f"Component {k}: ill-defined covariance. "
                    "Reduce n_components or increase reg_covar."
                )
        precisions_chol[k] = torch.linalg.solve_triangular(L, eye, upper=False).T

    return precisions_chol


def _estimate_gaussian_parameters(
    X: torch.Tensor,
    responsibilities: torch.Tensor,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """M-step statistics: weights, means, covariances from responsibilities.

    Port of sklearn's _estimate_gaussian_parameters for full covariance.
    """
    eps = torch.finfo(responsibilities.dtype).eps
    weights = responsibilities.sum(dim=0) + eps        # [K]
    means = torch.matmul(responsibilities.T, X) / weights.unsqueeze(1)  # [K, D]

    n_components, n_features = means.shape
    covariances = torch.empty(n_components, n_features, n_features, device=X.device, dtype=X.dtype)
    means_safe = means + eps
    for k in range(n_components):
        diff = X - means_safe[k]                       # [N, D]
        covariances[k] = torch.matmul(
            (responsibilities[:, k].unsqueeze(1) * diff).T, diff
        ) / weights[k]
        covariances[k] += torch.eye(n_features, device=X.device, dtype=X.dtype) * reg_covar

    return weights / weights.sum(), means, covariances


# ---------------------------------------------------------------------------
# GPU-native GMM nn.Module
# ---------------------------------------------------------------------------

class GaussianMixtureModel(nn.Module):
    """Gaussian Mixture Model with EM fitting, stored as PyTorch buffers.

    Because all parameters are registered buffers, calling model.to(device)
    moves the GMM to that device. Sampling via torch.distributions works
    on the same device without any CPU round-trips.
    """

    def __init__(self, n_components: int, n_features: int, reg_covar: float = 1e-6):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.reg_covar = reg_covar

        self.register_buffer("weights", torch.empty(n_components))
        self.register_buffer("means", torch.empty(n_components, n_features))
        self.register_buffer("covariances", torch.empty(n_components, n_features, n_features))
        self.register_buffer("precision_cholesky", torch.empty(n_components, n_features, n_features))
        self.register_buffer("_fitted", torch.zeros(1, dtype=torch.bool))

    @property
    def fitted(self) -> bool:
        return bool(self._fitted.item())

    # ------------------------------------------------------------------
    # Initialisation (called once before EM)
    # ------------------------------------------------------------------

    def _kmeans_init(self, X_np: np.ndarray) -> torch.Tensor:
        """Initialise responsibilities via k-means (CPU, one-shot)."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_components, random_state=0, n_init="auto")
        km.fit(X_np)
        n = len(X_np)
        resp = np.zeros((n, self.n_components), dtype=np.float32)
        resp[np.arange(n), km.labels_] = 1.0
        return torch.from_numpy(resp)

    def _initialise(self, X: torch.Tensor):
        """K-means init then compute initial GMM parameters."""
        X_np = X.detach().cpu().numpy().astype(np.float64)
        resp = self._kmeans_init(X_np).to(X.device, X.dtype)
        weights, means, covariances = _estimate_gaussian_parameters(X, resp, self.reg_covar)
        prec_chol = _compute_precision_cholesky(covariances, self.reg_covar)
        self.weights.data.copy_(weights)
        self.means.data.copy_(means)
        self.covariances.data.copy_(covariances)
        self.precision_cholesky.data.copy_(prec_chol)

    # ------------------------------------------------------------------
    # EM steps
    # ------------------------------------------------------------------

    def _log_det_cholesky(self) -> torch.Tensor:
        """Log-determinant of each component's Cholesky precision."""
        stride = self.n_features + 1
        diag = self.precision_cholesky.reshape(self.n_components, -1)[:, ::stride]
        return diag.log().sum(dim=1)

    def _log_gaussian_prob(self, X: torch.Tensor) -> torch.Tensor:
        """Log N(x | mu_k, Sigma_k) for every sample and component. [N, K]"""
        n_samples = X.shape[0]
        log_det = self._log_det_cholesky()           # [K]
        log_prob = torch.empty(n_samples, self.n_components, device=X.device, dtype=X.dtype)
        for k in range(self.n_components):
            y = X @ self.precision_cholesky[k] - self.means[k] @ self.precision_cholesky[k]
            log_prob[:, k] = y.square().sum(dim=1)
        pi = torch.tensor(torch.pi, device=X.device, dtype=X.dtype)
        return -0.5 * (self.n_features * torch.log(2 * pi) + log_prob) + log_det

    def e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """E step: returns (mean_log_likelihood, log_responsibilities)."""
        weighted = self._log_gaussian_prob(X) + self.weights.log()  # [N, K]
        log_norm = torch.logsumexp(weighted, dim=1)                  # [N]
        log_resp = weighted - log_norm.unsqueeze(1)
        return log_norm.mean(), log_resp

    def m_step(self, X: torch.Tensor, log_resp: torch.Tensor):
        """M step: update parameters from responsibilities."""
        weights, means, covariances = _estimate_gaussian_parameters(
            X, log_resp.exp(), self.reg_covar
        )
        prec_chol = _compute_precision_cholesky(covariances, self.reg_covar)
        self.weights.data.copy_(weights)
        self.means.data.copy_(means)
        self.covariances.data.copy_(covariances)
        self.precision_cholesky.data.copy_(prec_chol)

    # ------------------------------------------------------------------
    # Fit (called once after VAE training)
    # ------------------------------------------------------------------

    def fit(self, X: torch.Tensor, n_iter: int = 100, tol: float = 1e-3):
        """Run EM on X (can be on any device, including CUDA)."""
        self._initialise(X)
        prev_ll = -torch.inf
        for _ in range(n_iter):
            ll, log_resp = self.e_step(X)
            if abs(float(ll) - float(prev_ll)) < tol:
                break
            prev_ll = ll
            self.m_step(X, log_resp)
        self._fitted.fill_(True)

    # ------------------------------------------------------------------
    # Sampling (fully on-device, no CPU round-trip)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw n_samples from the mixture. Returns tensor on self.device."""
        counts = torch.multinomial(
            self.weights, n_samples, replacement=True
        ).bincount(minlength=self.n_components)

        chunks: list[torch.Tensor] = []
        for k, (mean, cov, count) in enumerate(
            zip(self.means, self.covariances, counts)
        ):
            if count > 0:
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                chunks.append(dist.sample((int(count),)))

        return torch.cat(chunks, dim=0)  # [n_samples, n_features]