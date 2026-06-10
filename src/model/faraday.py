"""Faraday model: Conditional VAE with MMD loss and GMM latent-space sampler.

Architecture from: Chai & Chadney, ICLR 2024 Workshop.
Ported from OpenSynth (Apache-2.0): https://github.com/OpenSynth-energy/OpenSynth
"""
from __future__ import annotations

import os
import re
import sys
import glob as _glob

if __name__ == "__main__":
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    while _script_dir in sys.path:
        sys.path.remove(_script_dir)

import torch
import torch.nn as nn

from model.model import BaseModel
from model.gmm import GaussianMixtureModel


def _find_best_checkpoint(model_dir: str) -> str:
    """Return the best-model checkpoint with the highest epoch, or _final as fallback."""
    candidates = _glob.glob(os.path.join(model_dir, "best_model_*.pt"))
    if candidates:
        def _epoch(p: str) -> int:
            m = re.search(r"_(\d+)\.pt$", p)
            return int(m.group(1)) if m else -1
        return max(candidates, key=_epoch)
    finals = _glob.glob(os.path.join(model_dir, "*_final.pt"))
    if finals:
        return finals[0]
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


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


class _DeterministicDecoderWrapper(nn.Module):
    """Adapts _MLP to the dict/sample interface expected by Generator's CVAE path."""

    def __init__(self, mlp: _MLP):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mu": self.mlp(x)}

    def sample(self, param_dict: dict[str, torch.Tensor], num_samples: int = 1, temperature: float = 1.0, **_) -> torch.Tensor:
        """Return [num_samples, B, D] by repeating the mean (deterministic decoder)."""
        mu = param_dict["mu"]  # [B, D]
        return mu.unsqueeze(0).expand(num_samples, -1, -1).contiguous()


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
    def reparameterize(mu_or_params, log_var=None, num_samples: int = 1, **_) -> torch.Tensor:
        """Sample z via reparameterization. Returns [num_samples, B, D].

        Accepts either (mu, log_var) tensors or a (mu, log_var) tuple as
        first arg so the generic generator interface works with Faraday.
        """
        if isinstance(mu_or_params, tuple):
            mu, log_var = mu_or_params
        else:
            mu = mu_or_params
        std = torch.exp(0.5 * log_var)
        B, D = mu.shape
        eps = torch.randn(num_samples, B, D, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + std.unsqueeze(0) * eps


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
        gmm_reg_covar: float = 1e-6,
        gmm_include_tags: list[str] | None = None,
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
        self.decoder = _DeterministicDecoderWrapper(
            _MLP(
                in_dim=latent_dim + self.cond_dim,
                hidden_dims=dec_dims,
                out_dim=sequence_length,
            )
        )

        self.hierarchy = None

        # Resolve household condition indices for joint GMM (OpenSynth approach).
        gmm_hh_indices: list[int] = []
        if conditioner is not None and gmm_include_tags:
            for tag in gmm_include_tags:
                try:
                    start, end = conditioner.get_condition_indexes(tag)
                    gmm_hh_indices.extend(range(start, end))
                except ValueError:
                    pass
        _gmm_hh_idx_tensor = (
            torch.tensor(gmm_hh_indices, dtype=torch.long)
            if gmm_hh_indices
            else torch.tensor([], dtype=torch.long)
        )
        self.register_buffer("_gmm_hh_indices", _gmm_hh_idx_tensor)
        self._gmm_hh_indices: torch.Tensor = _gmm_hh_idx_tensor

        # GMM is a registered submodule — its buffers are part of state_dict
        # and follow the model's device automatically via .to(device).
        # When gmm_include_tags is set, GMM is fitted on [z, c_hh] jointly
        # (n_features = latent_dim + hh_cond_dim, latent_dim=0 → unconditional sampling).
        gmm_n_features = latent_dim + len(gmm_hh_indices)
        self.gmm = GaussianMixtureModel(
            n_components=gmm_n_components,
            n_features=gmm_n_features,
            reg_covar=gmm_reg_covar,
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
        z = _FaradayEncoder.reparameterize(mu, log_var).squeeze(0)
        x_hat = self.decoder(torch.cat([z, c], dim=-1))["mu"]

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
            z = _FaradayEncoder.reparameterize(mu, log_var).squeeze(0)
            x_hat = self.decoder(torch.cat([z, c], dim=-1))["mu"]

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

        c_hh_gmm: torch.Tensor | None = None
        if self.gmm.fitted:
            joint = self.gmm.sample(num_prior_samples * B).to(device)
            z_flat = joint[:, : self.latent_dim]
            if self._gmm_hh_indices.numel() > 0:
                c_hh_gmm = joint[:, self.latent_dim:]  # [num_prior_samples*B, hh_cond_dim]
        else:
            z_flat = torch.randn(num_prior_samples * B, self.latent_dim, device=device)

        z = z_flat.reshape(num_prior_samples, B, self.latent_dim)
        c_exp = c.unsqueeze(0).expand(num_prior_samples, -1, -1).clone()  # [P, B, cond_dim]
        if c_hh_gmm is not None:
            c_exp[:, :, self._gmm_hh_indices] = c_hh_gmm.reshape(num_prior_samples, B, -1)
        dec_in = torch.cat([z, c_exp], dim=-1).reshape(num_prior_samples * B, -1)
        x_hat = self.decoder(dec_in)["mu"].reshape(num_prior_samples, B, self.sequence_length)

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
        x_hat = self.decoder(torch.cat([mu, c], dim=-1))["mu"]
        return {"mu": x_hat}

    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # GMM fitting — called after the VAE training loop completes
    # ------------------------------------------------------------------

    def _fit_gmm(self, data_loader, device: str = "cpu"):
        """Encode the entire training set and fit the GMM on latent codes."""
        from data.dataloader_phased import PhasedReplayDataloaders, InMemoryCSVDataloaders

        self.eval()
        self.to(device)
        z_list: list[torch.Tensor] = []

        if isinstance(data_loader, PhasedReplayDataloaders):
            loader = data_loader._get_train_stream_loader()
            max_batches = self.gmm_fit_batches
        elif isinstance(data_loader, InMemoryCSVDataloaders):
            loader = data_loader.get_train_epoch_loader(self.gmm_fit_batches)
            max_batches = None  # loader is already capped at gmm_fit_batches steps
        else:
            loader = data_loader
            max_batches = None

        if loader is None:
            return

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = self.move_to_device(batch, device)
                x, meter_id, timestamps, _ = self._unpack_batch(batch)
                x_flat = self._prep_inputs(x)
                c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)
                mu, _ = self.encoder(torch.cat([x_flat, c], dim=-1))
                if self._gmm_hh_indices.numel() > 0:
                    c_hh = c[:, self._gmm_hh_indices]
                    z_list.append(torch.cat([mu, c_hh], dim=-1))
                else:
                    z_list.append(mu)

        if not z_list:
            return

        Z = torch.cat(z_list, dim=0)   # [N, latent_dim (+ hh_cond_dim)] — on `device`
        self.gmm.to(device)
        self.gmm.fit(Z, n_iter=self.gmm_max_iter)

    def refit_gmm(self, checkpoint_path: str, data_loader, device: str = "cpu"):
        """Load a saved checkpoint, refit the GMM, and overwrite the checkpoint.

        Use this when the VAE has already been trained but the saved checkpoint
        has an unfitted GMM (e.g. it was a best-model checkpoint saved mid-training).
        """
        import os
        self.load_model(checkpoint_path, device=device)
        print(f"Loaded checkpoint: {checkpoint_path}")
        self._fit_gmm(data_loader, device=device)
        self.save_model(
            save_path=os.path.dirname(os.path.abspath(checkpoint_path)),
            model_name=os.path.splitext(os.path.basename(checkpoint_path))[0],
            device=device,
            normalizer=getattr(self, "normalizer", None),
        )
        print(f"Checkpoint re-saved with fitted GMM: {checkpoint_path}")

    def fit(self, data_loader, device, logging_dir, writer, data_cfg, checkpoint_cfg, train_cfg, prior_cfg):
        super().fit(data_loader, device, logging_dir, writer, data_cfg, checkpoint_cfg, train_cfg, prior_cfg)

        best_path = getattr(self, "_best_checkpoint_path", None)
        if best_path is not None:
            # Restore best VAE weights before fitting the GMM so the latent
            # space matches the best checkpoint rather than the final epoch.
            print(f"Loading best checkpoint for GMM fitting: {best_path}")
            self.load_model(best_path, device=device)

        self._fit_gmm(data_loader, device=device)

        save_path = best_path if best_path is not None else None
        if save_path is not None:
            import os
            self.save_model(
                save_path=os.path.dirname(save_path),
                model_name=os.path.splitext(os.path.basename(save_path))[0],
                device=device,
                normalizer=getattr(self, "normalizer", None),
            )
            print(f"Best model re-saved with fitted GMM: {save_path}")
        else:
            # No best model was saved (checkpointing disabled or too few epochs)
            self.save_model(
                save_path=logging_dir,
                model_name=f"{self.tag}_final",
                device=device,
                normalizer=getattr(self, "normalizer", None),
            )


if __name__ == "__main__":
    import sys
    import argparse
    import copy

    _this_dir = os.path.dirname(os.path.abspath(__file__))   # src/model/
    _src_dir = os.path.dirname(_this_dir)                    # src/
    _project_dir = os.path.dirname(_src_dir)                 # project root
    for _p in (_src_dir, _project_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from src.model.factory import _load_train_config, create_model_from_config
    from data.dataloader_phased import PhasedReplayDataloaders
    from data.normalizer import Normalizer
    from data.dataloader_phased import WindowSpec

    parser = argparse.ArgumentParser(description="Refit Faraday GMM on a saved checkpoint")
    parser.add_argument(
        "--config",
        default=os.path.join(_project_dir, "config", "tests", "agg_runs", "faraday.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(_project_dir, "models", "FaradayModel_20260518_163222"),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit .pt path; auto-selects best_model with highest epoch if omitted",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or _find_best_checkpoint(args.model_dir)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {args.device}")

    config = _load_train_config(args.config)
    data_cfg = getattr(config, "data", None)

    # Load the normalizer saved alongside the model (same stats used at training time).
    normalizer_path = os.path.join(args.model_dir, "normalizer.pt")
    normalizer = Normalizer.load(normalizer_path) if os.path.exists(normalizer_path) else None
    if normalizer is None:
        print("Warning: normalizer.pt not found — data will not be normalised before GMM fitting")

    # Discover data files (same logic as train.py).
    data_dir = config.data_dir
    configured_files = list(config.data_files) if len(config.data_files) > 0 else []
    if configured_files:
        parquet_paths = [f if os.path.isabs(f) else os.path.join(data_dir, f) for f in configured_files]
    else:
        parquet_paths = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files
            if f.endswith(".parquet") and not f.startswith("test")
        )
    parquet_paths = [p for p in parquet_paths if os.path.exists(p)]
    if not parquet_paths:
        raise ValueError(f"No .parquet files found under {data_dir}")

    window = WindowSpec(input_len=config.window.input_len, stride=config.window.stride)

    data_loader = PhasedReplayDataloaders(
        sm_files=parquet_paths,
        replay_buffer_capacity=int(getattr(data_cfg, "replay_buffer_capacity", 10000)),
        min_buffer_size=int(getattr(data_cfg, "min_buffer_size", 2000)),
        window_spec=window,
        normalizer=normalizer,
        num_workers=int(getattr(data_cfg, "num_workers", 4)),
        hourly_aggregation=data_cfg.window.aggregated,
        aggregation_divisor=1.0,
        batch_size=data_cfg.batch_size,
        eval_cache_batches=0,
        pin_memory=(args.device != "cpu"),
        output_aggregate_stream=False,
        skip_corrupt_files=bool(getattr(data_cfg, "skip_corrupt_files", True)),
        phased_mode=str(getattr(data_cfg, "mode", "csv")).lower() in {"phased", "phased_parquet"},
        feature_cols=data_cfg.columns.value_cols,
        meter_col=data_cfg.columns.meter_id,
        split_seed=int(getattr(data_cfg.splits, "random_seed", 0)),
        split_ratios={
            "train_ratio": float(getattr(data_cfg.splits, "train_ratio", 1.0)),
            "val_ratio": float(getattr(data_cfg.splits, "val_ratio", 0.0)),
            "test_ratio": float(getattr(data_cfg.splits, "test_ratio", 0.0)),
        },
        max_value=float(v) if (v := getattr(data_cfg, "max_value", None)) is not None else None,
        max_hour_value=float(v) if (v := getattr(data_cfg, "max_hour_value", None)) is not None else None,
    )

    model = create_model_from_config(config, device=args.device)
    if normalizer is not None:
        model.normalizer = copy.deepcopy(normalizer)

    model.refit_gmm(checkpoint_path, data_loader, device=args.device)