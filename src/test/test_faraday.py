from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.faraday import mmd_loss, quantile_loss, mse_loss
from pydantic import ValidationError
from config.loader import load_config


# ---------------------------------------------------------------------------
# Cycle 1 — mmd_loss(x, x) is near zero
# ---------------------------------------------------------------------------

def test_mmd_loss_same_input_is_near_zero():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    loss = mmd_loss(x, x)
    assert loss.ndim == 0, "mmd_loss must return a scalar"
    assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# Cycle 2 — quantile_loss at q=0.5 equals MAE
# ---------------------------------------------------------------------------

def test_quantile_loss_at_0_5_equals_mae():
    torch.manual_seed(1)
    pred = torch.randn(64, 24)
    real = torch.randn(64, 24)
    q_loss = quantile_loss(pred, real, 0.5)
    mae = torch.mean(torch.abs(real - pred))
    assert q_loss.ndim == 0
    assert torch.allclose(q_loss, 0.5 * mae, atol=1e-5)


# ---------------------------------------------------------------------------
# Cycle 3 — mse_loss returns a scalar
# ---------------------------------------------------------------------------

def test_mse_loss_returns_scalar():
    torch.manual_seed(2)
    pred = torch.randn(16, 24)
    real = torch.zeros(16, 24)
    loss = mse_loss(pred, real)
    assert loss.ndim == 0
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> Path:
    import yaml
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


_MINIMAL_FARADAY_MODEL = {
    "type": "faraday",
    "latent_dim": 32,
    "encoder": {"type": "mlp", "hidden_dims": [64, 32]},
    "decoder": {"type": "mlp", "hidden_dims": [32, 64]},
    "faraday": {
        "mse_weight": 1.0,
        "mmd_weight": 100.0,
        "upper_quantile_weight": 1.0,
        "lower_quantile_weight": 1.0,
        "median_quantile_weight": 1.0,
    },
    "gmm": {"n_components": 300, "covariance_type": "full", "max_iter": 100},
}


# ---------------------------------------------------------------------------
# Cycle 4 — valid Faraday config parses
# ---------------------------------------------------------------------------

def test_faraday_config_valid_dict_parses(tmp_path):
    cfg_path = _write_yaml(tmp_path / "faraday.yaml", {"model": _MINIMAL_FARADAY_MODEL})
    cfg = load_config(cfg_path)
    assert cfg.model.type == "faraday"
    assert cfg.model.gmm.n_components == 300


# ---------------------------------------------------------------------------
# Cycle 5 — missing required field raises ValidationError
# ---------------------------------------------------------------------------

def test_faraday_config_missing_latent_dim_raises(tmp_path):
    bad_model = {k: v for k, v in _MINIMAL_FARADAY_MODEL.items() if k != "latent_dim"}
    cfg_path = _write_yaml(tmp_path / "bad.yaml", {"model": bad_model})
    with pytest.raises(ValidationError):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# Cycle 6 — FaradayModel.sample() returns correct shape [S, P, B, D]
# ---------------------------------------------------------------------------

class _StubConditioner:
    """Minimal conditioner stub: returns a fixed-dim zero vector per window."""
    def __init__(self, cond_dim: int = 8):
        self._dim = cond_dim

    def get_condition_dim(self) -> int:
        return self._dim

    def get_conditions(self, x, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        return torch.zeros(B, self._dim)


def test_faraday_sample_output_shape():
    from model.faraday import FaradayModel
    from model.base.gmm import GaussianMixtureModel

    sequence_length = 24
    latent_dim = 16
    cond_dim = 8
    B = 4
    num_prior_samples = 3
    num_x_samples = 2

    conditioner = _StubConditioner(cond_dim)
    model = FaradayModel(
        sequence_length=sequence_length,
        latent_dim=latent_dim,
        conditioner=conditioner,
        encoder_hidden_dims=[32, 16],
        decoder_hidden_dims=[16, 32],
        gmm_n_components=5,
    )

    # Fit the GPU-native GMM on synthetic latent codes
    Z = torch.randn(50, latent_dim)
    model.gmm.fit(Z, n_iter=5)

    conditions = torch.zeros(B, cond_dim)
    out = model.sample(conditions, num_prior_samples=num_prior_samples, num_x_samples=num_x_samples, device="cpu")

    assert out.shape == (num_x_samples, num_prior_samples, B, sequence_length), (
        f"Expected ({num_x_samples}, {num_prior_samples}, {B}, {sequence_length}), got {tuple(out.shape)}"
    )