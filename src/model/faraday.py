"""Faraday model: Conditional VAE with MMD loss and GMM latent-space sampler.

Architecture from: Chai & Chadney, ICLR 2024 Workshop.
Ported from OpenSynth (Apache-2.0): https://github.com/OpenSynth-energy/OpenSynth
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.model import BaseModel
from model.gmm import GaussianMixtureModel


_MMD_BANDWIDTHS = [10, 15, 20, 50]


def mmd_loss(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multi-scale Gaussian kernel MMD between two batches of latent codes."""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    zz = torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(zz)
    for bw in _MMD_BANDWIDTHS:
        XX += torch.exp(-0.5 * dxx / bw)
        YY += torch.exp(-0.5 * dyy / bw)
        XY += torch.exp(-0.5 * dxy / bw)
    return torch.mean(XX + YY - 2.0 * XY)


def quantile_loss(y_pred: torch.Tensor, y_real: torch.Tensor, quantile: float) -> torch.Tensor:
    """Pinball (quantile) loss for a single quantile level q in (0, 1)."""
    errors = y_real - y_pred
    return torch.mean(torch.max(quantile * errors, (quantile - 1.0) * errors))


def mse_loss(y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return torch.mean((y_pred - y_real) ** 2)


def calculate_training_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    mse_weight: float = 1.0,
    upper_quantile_weight: float = 1.0,
    lower_quantile_weight: float = 1.0,
    median_quantile_weight: float = 1.0,
    mmd_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined Faraday training loss: MSE + quantile losses + MMD."""
    prior_samples = torch.randn_like(z)
    l_mse = mse_loss(x_hat, x)
    l_mmd = mmd_loss(z, prior_samples)
    l_q05 = quantile_loss(x_hat, x, 0.05)
    l_q50 = quantile_loss(x_hat, x, 0.50)
    l_q95 = quantile_loss(x_hat, x, 0.95)
    total = (
        mse_weight * l_mse
        + mmd_weight * l_mmd
        + lower_quantile_weight * l_q05
        + median_quantile_weight * l_q50
        + upper_quantile_weight * l_q95
    )
    return {"loss": total, "mse": l_mse, "mmd": l_mmd, "q05": l_q05, "q50": l_q50, "q95": l_q95}


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FaradayEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.trunk = _MLP(input_dim, hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims, hidden_dims[-1])
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu_head(h), self.log_var_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


class FaradayModel(BaseModel):
    """BaseModel subclass implementing the Faraday CVAE+MMD+GMM architecture."""

    def __init__(
        self,
        sequence_length: int = 24,
        latent_dim: int = 32,
        conditioner=None,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        mse_weight: float = 1.0,
        mmd_weight: float = 100.0,
        upper_quantile_weight: float = 1.0,
        lower_quantile_weight: float = 1.0,
        median_quantile_weight: float = 1.0,
        gmm_n_components: int = 300,
        gmm_max_iter: int = 100,
        **kwargs,
    ):
        super().__init__(phased_mode=False, sequence_length=sequence_length)
        self.conditioner = conditioner
        self.latent_dim = latent_dim
        self.cond_dim = conditioner.get_condition_dim() if conditioner is not None else 0

        enc_dims = encoder_hidden_dims or [512, 256, 128, 64, 32]
        dec_dims = decoder_hidden_dims or [32, 64, 128, 256, 512]

        self.encoder = _FaradayEncoder(
            input_dim=sequence_length + self.cond_dim,
            hidden_dims=enc_dims,
            latent_dim=latent_dim,
        )
        self.decoder = _MLP(
            in_dim=latent_dim + self.cond_dim,
            hidden_dims=dec_dims,
            out_dim=sequence_length,
        )

        # GMM is a registered submodule — its buffers are part of state_dict
        # and follow the model's device automatically via .to(device).
        self.gmm = GaussianMixtureModel(
            n_components=gmm_n_components,
            n_features=latent_dim,
        )

        self.mse_weight = mse_weight
        self.mmd_weight = mmd_weight
        self.upper_quantile_weight = upper_quantile_weight
        self.lower_quantile_weight = lower_quantile_weight
        self.median_quantile_weight = median_quantile_weight
        self.gmm_max_iter = gmm_max_iter
        self.gmm_fit_batches = kwargs.get("gmm_fit_batches", 500)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_conditions(self, x: torch.Tensor, meter_id=None, timestamps=None, device="cpu") -> torch.Tensor:
        if self.conditioner is None:
            return torch.zeros(x.shape[0], 0, device=device)
        x_pf = self._to_phase_first(x)
        return self._build_conditions(x_pf, meter_id=meter_id, timestamps=timestamps, device=device)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train_core(self, batch, optim, gradient_clip=None, gradient_clip_max_norm=1.0, **kwargs):
        self.train()
        x, meter_id, timestamps, _ = self._unpack_batch(batch)
        x = self.move_to_device(x, next(self.parameters()).device)
        x_flat = self._prep_inputs(x)
        c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=x_flat.device)

        mu, log_var = self.encoder(torch.cat([x_flat, c], dim=-1))
        z = _FaradayEncoder.reparameterize(mu, log_var)
        x_hat = self.decoder(torch.cat([z, c], dim=-1))

        loss_dict = calculate_training_loss(
            x_hat=x_hat, x=x_flat, z=z,
            mse_weight=self.mse_weight,
            mmd_weight=self.mmd_weight,
            upper_quantile_weight=self.upper_quantile_weight,
            lower_quantile_weight=self.lower_quantile_weight,
            median_quantile_weight=self.median_quantile_weight,
        )

        optim.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        if gradient_clip:
            nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_max_norm)
        optim.step()

        loss_dict["recon_loglikelihood"] = loss_dict["mse"].detach()
        loss_dict["kl_divergence"] = loss_dict["mmd"].detach()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, beta: float = 1.0, device: str = "cpu", **kwargs):
        self.eval()
        metrics = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        n = 0
        for batch in valloader:
            batch = self.move_to_device(batch, device)
            x, meter_id, timestamps, _ = self._unpack_batch(batch)
            x_flat = self._prep_inputs(x)
            c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)

            mu, log_var = self.encoder(torch.cat([x_flat, c], dim=-1))
            z = _FaradayEncoder.reparameterize(mu, log_var)
            x_hat = self.decoder(torch.cat([z, c], dim=-1))

            ld = calculate_training_loss(
                x_hat=x_hat, x=x_flat, z=z,
                mse_weight=self.mse_weight, mmd_weight=self.mmd_weight,
                upper_quantile_weight=self.upper_quantile_weight,
                lower_quantile_weight=self.lower_quantile_weight,
                median_quantile_weight=self.median_quantile_weight,
            )
            metrics["loss"] += ld["loss"].item()
            metrics["recon_loglikelihood"] += ld["mse"].item()
            metrics["kl_divergence"] += ld["mmd"].item()
            n += 1
        if n:
            for k in metrics:
                metrics[k] /= n
        return metrics

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,
        num_prior_samples: int = 1,
        num_x_samples: int = 1,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """Draw samples from the fitted GMM and decode.

        Returns tensor of shape [num_x_samples, num_prior_samples, B, D].
        Falls back to N(0,I) sampling if GMM has not been fitted yet.
        """
        self.eval()
        self.to(device)
        c = conditions.to(device)
        B = c.shape[0]

        if self.gmm.fitted:
            z_flat = self.gmm.sample(num_prior_samples * B)[:, : self.latent_dim]
        else:
            z_flat = torch.randn(num_prior_samples * B, self.latent_dim, device=device)

        z = z_flat.to(device).reshape(num_prior_samples, B, self.latent_dim)
        c_exp = c.unsqueeze(0).expand(num_prior_samples, -1, -1)
        dec_in = torch.cat([z, c_exp], dim=-1).reshape(num_prior_samples * B, -1)
        x_hat = self.decoder(dec_in).reshape(num_prior_samples, B, self.sequence_length)

        # [num_x_samples, num_prior_samples, B, D]
        return x_hat.unsqueeze(0).expand(num_x_samples, -1, -1, -1).contiguous()

    @torch.no_grad()
    def reconstruct(self, batch) -> dict[str, torch.Tensor]:
        """Encode then decode via mu (no sampling). Returns {"mu": x_hat}."""
        x, meter_id, timestamps, _ = self._unpack_batch(batch)
        device = next(self.parameters()).device
        x = self.move_to_device(x, device)
        x_flat = self._prep_inputs(x)
        c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)
        mu, _ = self.encoder(torch.cat([x_flat, c], dim=-1))
        x_hat = self.decoder(torch.cat([mu, c], dim=-1))
        return {"mu": x_hat}

    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # GMM fitting — called after the VAE training loop completes
    # ------------------------------------------------------------------

    def _fit_gmm(self, data_loader, device: str = "cpu"):
        """Encode the entire training set and fit the GMM on latent codes."""
        from data.dataloader_phased import PhasedReplayDataloaders

        self.eval()
        self.to(device)
        z_list: list[torch.Tensor] = []

        if isinstance(data_loader, PhasedReplayDataloaders):
            loader = data_loader._get_train_stream_loader()
            max_batches = self.gmm_fit_batches
        else:
            loader = data_loader
            max_batches = None

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = self.move_to_device(batch, device)
                x, meter_id, timestamps, _ = self._unpack_batch(batch)
                x_flat = self._prep_inputs(x)
                c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)
                mu, _ = self.encoder(torch.cat([x_flat, c], dim=-1))
                z_list.append(mu)

        if not z_list:
            return

        Z = torch.cat(z_list, dim=0)   # [N, latent_dim] — on `device`
        self.gmm.to(device)
        self.gmm.fit(Z, n_iter=self.gmm_max_iter)

    def fit(self, data_loader, device, logging_dir, writer, data_cfg, checkpoint_cfg, train_cfg, prior_cfg):
        super().fit(data_loader, device, logging_dir, writer, data_cfg, checkpoint_cfg, train_cfg, prior_cfg)
        self._fit_gmm(data_loader, device=device)