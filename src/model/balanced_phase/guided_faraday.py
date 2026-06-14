"""GuidedFaraday: PDCC decoder + MMD latent regularizer + CNN encoder.

Architecture decision: ADR-0006.
"""
from __future__ import annotations

import os
import re
import glob as _glob
import sys

# When executed as a script (python -m model.guided_faraday or uv run ...),
# ensure src/ is on sys.path before the model-package imports below.
if __name__ == "__main__":
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.dirname(_this_dir)
    _project_dir = os.path.dirname(_src_dir)
    # Remove the script directory: it contains model.py which shadows the model/ package.
    while _this_dir in sys.path:
        sys.path.remove(_this_dir)
    for _p in (_src_dir, _project_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

import torch
import torch.nn as nn

from model.base.model import BaseModel
from model.base.gmm import GaussianMixtureModel
from model.balanced_phase.faraday import mmd_loss, quantile_loss, mse_loss, _FaradayEncoder
from model.base.encoder_decoder_model import TemporalCNNEncoder, DictionaryCoder


class _GuidedFaradayCNNEncoder(nn.Module):
    """CNN encoder with full-input condition injection.

    Conditions are concatenated to the flattened signal BEFORE the CNN trunk
    (``cat([x_flat, c])``).  Every convolutional layer therefore sees condition
    information, allowing the model to learn condition-specific temporal feature
    detectors.

    The original bottleneck-injection design — conditions concatenated only
    after CNN downsampling — was reverted after empirical evidence showed the
    CNN failing to outperform an MLP encoder across nearly all metrics.  With
    bottleneck injection the CNN trunk receives zero condition gradient; the
    parameter heads alone cannot compensate.  See ADR-0006 amendment.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        latent_dim: int,
        cond_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        full_dim = num_features * seq_len + cond_dim
        # TemporalCNNEncoder receives the full [signal, conditions] flat vector.
        # num_features=1, seq_len=full_dim treats the entire flat vector as a
        # single-channel sequence — no cond stripping, no bottleneck injection.
        self.cnn = TemporalCNNEncoder(
            input_dim=full_dim,
            output_dim=latent_dim,
            seq_len=full_dim,
            num_features=1,
            dist_parameters=["mu", "log_var"],
            param_output_dims={"mu": latent_dim, "log_var": latent_dim},
            hidden_dims=hidden_dims,
            cond_dim=0,
        )

    def forward(self, x_flat: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xc = torch.cat([x_flat, c], dim=-1)         # [B, signal_dim + cond_dim]
        out = self.cnn(xc)
        return out["mu"], out["log_var"]


class _GuidedFaradayMLPEncoder(nn.Module):
    """Adapts _FaradayEncoder to the (x_flat, c) → (mu, log_var) interface.

    Conditions are concatenated to the flattened input before the MLP trunk,
    making this the canonical baseline for the CNN-vs-MLP ablation.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        latent_dim: int,
        cond_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        dims = hidden_dims or [64, 128, 256, 256]
        self.encoder = _FaradayEncoder(
            input_dim=num_features * seq_len + cond_dim,
            hidden_dims=dims,
            latent_dim=latent_dim,
        )

    def forward(self, x_flat: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(torch.cat([x_flat, c], dim=-1))



class GuidedFaradayModel(BaseModel):
    """Faraday with PDCC decoder and bottleneck-injecting CNN encoder."""

    def __init__(
        self,
        sequence_length: int = 24,
        latent_dim: int = 32,
        conditioner=None,
        encoder_type: str = "cnn",
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dictionary_size: int = 100,
        pdcc_average_max_std: float = 1.0,
        mse_weight: float = 0.0,
        mmd_weight: float = 100.0,
        upper_quantile_weight: float = 0.0,
        lower_quantile_weight: float = 0.0,
        median_quantile_weight: float = 0.0,
        gmm_n_components: int = 300,
        gmm_max_iter: int = 100,
        gmm_reg_covar: float = 1e-6,
        phased_mode: bool = False,
        stochastic_loss: bool = True,
        encoder_exclude_tags: list[str] | None = None,
        decoder_mu_activation: str = "softplus",
        gmm_include_tags: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length)
        self.conditioner = conditioner
        self.latent_dim = latent_dim
        # self.num_features is set by BaseModel: 1 for aggregated, 3 for phased
        self.cond_dim = conditioner.get_condition_dim() if conditioner is not None else 0

        # Compute which condition indices enter the encoder (and therefore the GMM latent
        # space) vs those that bypass the encoder and reach only the decoder.
        # `encoder_exclude_tags` names condition tags (e.g. "user_profiles") that should
        # stay out of the encoder to keep the GMM distribution tighter.
        if conditioner is not None and encoder_exclude_tags:
            exclude = set()
            for tag in encoder_exclude_tags:
                try:
                    start, end = conditioner.get_condition_indexes(tag)
                    exclude.update(range(start, end))
                except ValueError:
                    pass
            enc_indices = [i for i in range(self.cond_dim) if i not in exclude]
        else:
            enc_indices = list(range(self.cond_dim))

        self.register_buffer("_enc_cond_indices", torch.tensor(enc_indices, dtype=torch.long))
        self.gmm_cond_dim = len(enc_indices)

        self.stochastic_loss = stochastic_loss

        encoder_cls = _GuidedFaradayMLPEncoder if encoder_type == "mlp" else _GuidedFaradayCNNEncoder
        self.encoder = encoder_cls(
            seq_len=sequence_length,
            num_features=self.num_features,
            latent_dim=latent_dim,
            cond_dim=self.gmm_cond_dim,
            hidden_dims=encoder_hidden_dims,
        )

        dec_dims = decoder_hidden_dims or [64, 128, 256, 512]
        self.decoder = DictionaryCoder(
            input_dim=latent_dim + self.cond_dim,
            output_dim=sequence_length,
            dictionary_size=dictionary_size,
            hidden_dims=dec_dims,
            layer_type="MLP",
            average_max_std=pdcc_average_max_std,
            mu_activation=decoder_mu_activation if decoder_mu_activation is not None else "softplus",
        )

        self.hierarchy = None

        # Resolve household condition indices for conditional GMM prior.
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
        self._gmm_hh_indices: torch.Tensor = _gmm_hh_idx_tensor  # typed alias for static analysis

        gmm_n_features = latent_dim + len(gmm_hh_indices)
        self.gmm = GaussianMixtureModel(
            n_components=gmm_n_components,
            n_features=gmm_n_features,
            reg_covar=gmm_reg_covar,
            latent_dim=latent_dim if gmm_hh_indices else 0,
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

    def _get_conditions(self, x, meter_id=None, timestamps=None, device="cpu") -> torch.Tensor:
        if self.conditioner is None:
            return torch.zeros(x.shape[0], 0, device=device)
        x_pf = self._to_phase_first(x)
        return self._build_conditions(x_pf, meter_id=meter_id, timestamps=timestamps, device=device)

    def _enc_conditions(self, c: torch.Tensor) -> torch.Tensor:
        """Slice conditions to the subset that enters the encoder (and GMM)."""
        if self.gmm_cond_dim == self.cond_dim:
            return c
        return c[:, self._enc_cond_indices]

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train_core(self, batch, optim, gradient_clip=None, gradient_clip_max_norm=1.0, **kwargs):
        self.train()
        x, meter_id, timestamps, _ = self._unpack_batch(batch)
        device = next(self.parameters()).device
        x = self.move_to_device(x, device)
        x_flat = self._prep_inputs(x)
        c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)

        mu, log_var = self.encoder(x_flat, self._enc_conditions(c))
        z = _FaradayEncoder.reparameterize(mu, log_var).squeeze(0)   # [B, latent_dim]

        dec_in = torch.cat([z, c], dim=-1)
        param_dict = self.decoder(dec_in)                             # {mu, sigma}
        loglik = self.decoder.loglikelihood(x_flat, param_dict)       # [B]
        l_loglik = -loglik.mean()

        prior_samples = torch.randn_like(z)
        l_mmd = mmd_loss(z, prior_samples)

        total = l_loglik + self.mmd_weight * l_mmd

        loss_dict = {"loglik": l_loglik.detach(), "mmd": l_mmd.detach()}

        if self.mse_weight > 0.0:
            mse = mse_loss(param_dict["mu"], x_flat)
            total = total + self.mse_weight * mse
            loss_dict["mse"] = mse.detach()

        any_quantile = self.upper_quantile_weight > 0.0 or self.lower_quantile_weight > 0.0 or self.median_quantile_weight > 0.0
        if any_quantile:
            # Draw one PDCC sample so the quantile losses train the full output
            # distribution (σ, U) — not just mu.  All three losses share this
            # draw; the pinball asymmetry directs each toward its own tail.
            if not self.stochastic_loss:
                # For a non-stochastic loss, use the mean (mu) directly without sampling.
                x_samp = param_dict["mu"]
            else:
                x_samp = self.decoder.sample(param_dict, num_samples=1).squeeze(0)  # [B, D]

        if self.upper_quantile_weight > 0.0:
            quantile_95 = quantile_loss(x_samp, x_flat, 0.95)
            total = total + self.upper_quantile_weight * quantile_95
            loss_dict["quantile_95"] = quantile_95.detach()
        if self.lower_quantile_weight > 0.0:
            quantile_05 = quantile_loss(x_samp, x_flat, 0.05)
            total = total + self.lower_quantile_weight * quantile_05
            loss_dict["quantile_05"] = quantile_05.detach()
        if self.median_quantile_weight > 0.0:
            quantile_50 = quantile_loss(x_samp, x_flat, 0.50)
            total = total + self.median_quantile_weight * quantile_50
            loss_dict["quantile_50"] = quantile_50.detach()

        optim.zero_grad(set_to_none=True)
        total.backward()
        if gradient_clip:
            nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_max_norm)
        optim.step()

        loss_dict["loss"] = total.detach()

        return loss_dict
    
    @torch.no_grad()
    def validate(self, valloader, beta: float = 1.0, device: str = "cpu", **kwargs):
        self.eval()
        metrics = {"loss": 0.0, "recon_loglikelihood": 0.0, "mmd": 0.0, "mse": 0.0, "quantile_95": 0.0, "quantile_05": 0.0, "quantile_50": 0.0}
        n = 0
        for batch in valloader:
            batch = self.move_to_device(batch, device)
            x, meter_id, timestamps, _ = self._unpack_batch(batch)
            x_flat = self._prep_inputs(x)
            c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)

            mu, log_var = self.encoder(x_flat, self._enc_conditions(c))
            z = _FaradayEncoder.reparameterize(mu, log_var).squeeze(0)
            dec_in = torch.cat([z, c], dim=-1)
            param_dict = self.decoder(dec_in)
            loglik = self.decoder.loglikelihood(x_flat, param_dict)
            l_loglik = -loglik.mean()
            l_mmd = mmd_loss(z, torch.randn_like(z))

            metrics["loss"] += (l_loglik + self.mmd_weight * l_mmd).item()
            metrics["recon_loglikelihood"] += l_loglik.item()
            metrics["mmd"] += l_mmd.item()
            if self.mse_weight > 0.0:
                mse = mse_loss(param_dict["mu"], x_flat)
                metrics["loss"] += self.mse_weight * mse.item()
                metrics["mse"] += mse.item()
            if self.upper_quantile_weight > 0.0:
                quantile_95 = quantile_loss(param_dict["mu"], x_flat, 0.95)
                metrics["loss"] += self.upper_quantile_weight * quantile_95.item()
                metrics["quantile_95"] += quantile_95.item()
            if self.lower_quantile_weight > 0.0:
                quantile_05 = quantile_loss(param_dict["mu"], x_flat, 0.05)
                metrics["loss"] += self.lower_quantile_weight * quantile_05.item()
                metrics["quantile_05"] += quantile_05.item()
            if self.median_quantile_weight > 0.0:
                quantile_50 = quantile_loss(param_dict["mu"], x_flat, 0.50)
                metrics["loss"] += self.median_quantile_weight * quantile_50.item()
                metrics["quantile_50"] += quantile_50.item() 
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
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Returns [num_x_samples, num_prior_samples, B, seq_len]."""
        self.eval()
        self.to(device)
        c = conditions.to(device)
        B = c.shape[0]

        if self.gmm.fitted:
            if self.gmm.latent_dim > 0 and self._gmm_hh_indices.numel() > 0:
                c_hh = c[:, self._gmm_hh_indices]  # [B, hh_cond_dim]
                c_hh_exp = c_hh.unsqueeze(0).expand(num_prior_samples, -1, -1).reshape(num_prior_samples * B, -1)
                z_flat = self.gmm.sample_conditional(c_hh_exp)  # [num_prior_samples*B, latent_dim]
            else:
                z_flat = self.gmm.sample(num_prior_samples * B)[:, : self.latent_dim]
        else:
            z_flat = torch.randn(num_prior_samples * B, self.latent_dim, device=device)

        z = z_flat.to(device).reshape(num_prior_samples, B, self.latent_dim)
        c_exp = c.unsqueeze(0).expand(num_prior_samples, -1, -1)
        dec_in = torch.cat([z, c_exp], dim=-1).reshape(num_prior_samples * B, -1)
        param_dict = self.decoder(dec_in)

        # [num_x_samples, num_prior_samples*B, seq_len]
        x_samples = self.decoder.sample(param_dict, num_samples=num_x_samples, temperature=temperature)
        # x_samples: [num_x_samples, num_prior_samples*B, seq_len]
        x_out = x_samples.reshape(num_x_samples, num_prior_samples, B, self.sequence_length)
        return torch.clamp(x_out.contiguous(),min=0.0)

    @torch.no_grad()
    def reconstruct(self, batch) -> dict[str, torch.Tensor]:
        x, meter_id, timestamps, _ = self._unpack_batch(batch)
        device = next(self.parameters()).device
        x = self.move_to_device(x, device)
        x_flat = self._prep_inputs(x)
        c = self._get_conditions(x, meter_id=meter_id, timestamps=timestamps, device=device)
        mu, _ = self.encoder(x_flat, self._enc_conditions(c))
        param_dict = self.decoder(torch.cat([mu, c], dim=-1))
        return {"mu": param_dict["mu"]}

    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # GMM fitting — identical to FaradayModel
    # ------------------------------------------------------------------

    def _fit_gmm(self, data_loader, device: str = "cpu"):
        from data.dataloader_phased import PhasedReplayDataloaders, InMemoryCSVDataloaders

        self.eval()
        self.to(device)
        z_list: list[torch.Tensor] = []

        if isinstance(data_loader, PhasedReplayDataloaders):
            loader = data_loader._get_train_stream_loader()
            max_batches = self.gmm_fit_batches
        elif isinstance(data_loader, InMemoryCSVDataloaders):
            loader = data_loader.get_train_epoch_loader(self.gmm_fit_batches)
            max_batches = None
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
                mu, _ = self.encoder(x_flat, self._enc_conditions(c))
                if self._gmm_hh_indices.numel() > 0:
                    c_hh = c[:, self._gmm_hh_indices]
                    z_list.append(torch.cat([mu, c_hh], dim=-1))
                else:
                    z_list.append(mu)

        if not z_list:
            return

        Z = torch.cat(z_list, dim=0)
        self.gmm.to(device)
        self.gmm.fit(Z, n_iter=self.gmm_max_iter)

    def refit_gmm(self, checkpoint_path: str, data_loader, device: str = "cpu"):
        """Load a saved checkpoint, refit the GMM, and overwrite the checkpoint."""
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
            self.load_model(best_path, device=device)

        self._fit_gmm(data_loader, device=device)

        save_path = best_path
        if save_path is not None:
            self.save_model(
                save_path=os.path.dirname(save_path),
                model_name=os.path.splitext(os.path.basename(save_path))[0],
                device=device,
                normalizer=getattr(self, "normalizer", None),
            )
        else:
            self.save_model(
                save_path=logging_dir,
                model_name=f"{self.tag}_final",
                device=device,
                normalizer=getattr(self, "normalizer", None),
            )
    def get_SigmaMapper(self):
        return self.decoder.get_SigmaMapper()


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


if __name__ == "__main__":
    import sys
    import argparse
    import copy

    _this_dir = os.path.dirname(os.path.abspath(__file__))   # src/model/balanced_phase/
    _src_dir = os.path.dirname(_this_dir)                    # src/model/
    _project_dir = os.path.dirname(_src_dir)                 # src/
    for _p in (_src_dir, _project_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from model.base.factory import _load_train_config, create_model_from_config
    from data.dataloader_phased import PhasedReplayDataloaders
    from data.normalizer import Normalizer
    from data.dataloader_phased import WindowSpec

    parser = argparse.ArgumentParser(description="Refit GuidedFaraday GMM on a saved checkpoint")
    parser.add_argument(
        "--config",
        default=os.path.join(_project_dir, "config", "tests", "agg", "guided_faraday.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the saved model checkpoint (e.g. models/GuidedFaradayModel_...)",
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

    normalizer_path = os.path.join(args.model_dir, "normalizer.pt")
    normalizer = Normalizer.load(normalizer_path) if os.path.exists(normalizer_path) else None
    if normalizer is None:
        print("Warning: normalizer.pt not found — data will not be normalised before GMM fitting")

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
