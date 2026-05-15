from __future__ import annotations

import sys
from pathlib import Path

import pytest

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import warnings

from pydantic import ValidationError

from config.loader import load_config
from model.factory import _load_train_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_ART_MODEL = {
    "type": "art",
    "latent_dim": 128,
    "encoder": {"type": "dictionary", "hidden_dims": [32, 64], "dropout": 0.2},
    "decoder": {"type": "dictionary", "hidden_dims": [64, 32], "dropout": 0.2},
    "prior": {"type": "standard_gaussian"},
    "art": {"num_bins": 64, "transformer_num_heads": 8, "transformer_num_layers": 4, "d_model": 64},
}

_MINIMAL_DIFFUSION_MODEL = {
    "type": "diffusion",
    "diffusion": {"num_timesteps": 200, "schedule_type": "cosine"},
}


def _write_yaml(path: Path, data: dict) -> Path:
    import yaml
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: standalone art config
# ---------------------------------------------------------------------------

def test_load_standalone_art_config(tmp_path):
    cfg_path = _write_yaml(tmp_path / "art.yaml", {"model": _MINIMAL_ART_MODEL})
    cfg = load_config(cfg_path)
    assert cfg.model.type == "art"
    assert cfg.model.art.num_bins == 64


# ---------------------------------------------------------------------------
# Cycle 2 — base + override merge
# ---------------------------------------------------------------------------

def test_override_with_base_key_includes_base_fields(tmp_path):
    base = {"data": {"batch_size": 32}, "model": _MINIMAL_ART_MODEL}
    override = {"base": "base.yaml", "model": {**_MINIMAL_ART_MODEL, "art": {**_MINIMAL_ART_MODEL["art"]}}}
    _write_yaml(tmp_path / "base.yaml", base)
    _write_yaml(tmp_path / "override.yaml", override)
    cfg = load_config(tmp_path / "override.yaml")
    assert cfg.data["batch_size"] == 32


# ---------------------------------------------------------------------------
# Cycle 3 — override value wins at leaf level
# ---------------------------------------------------------------------------

def test_override_value_wins_over_base(tmp_path):
    base = {"model": {**_MINIMAL_ART_MODEL, "art": {**_MINIMAL_ART_MODEL["art"], "num_bins": 32}}}
    override = {"base": "base.yaml", "model": {**_MINIMAL_ART_MODEL, "art": {**_MINIMAL_ART_MODEL["art"], "num_bins": 128}}}
    _write_yaml(tmp_path / "base.yaml", base)
    _write_yaml(tmp_path / "override.yaml", override)
    cfg = load_config(tmp_path / "override.yaml")
    assert cfg.model.art.num_bins == 128


# ---------------------------------------------------------------------------
# Cycle 4 — base: key stripped from result
# ---------------------------------------------------------------------------

def test_base_key_is_not_present_on_result(tmp_path):
    base = {"model": _MINIMAL_ART_MODEL}
    override = {"base": "base.yaml", "model": _MINIMAL_ART_MODEL}
    _write_yaml(tmp_path / "base.yaml", base)
    _write_yaml(tmp_path / "override.yaml", override)
    cfg = load_config(tmp_path / "override.yaml")
    assert not hasattr(cfg, "base")


# ---------------------------------------------------------------------------
# Cycle 5 — diffusion union: no encoder/decoder attributes
# ---------------------------------------------------------------------------

def test_diffusion_model_has_diffusion_config_and_no_encoder(tmp_path):
    cfg_path = _write_yaml(tmp_path / "diffusion.yaml", {"model": _MINIMAL_DIFFUSION_MODEL})
    cfg = load_config(cfg_path)
    assert cfg.model.type == "diffusion"
    assert cfg.model.diffusion.num_timesteps == 200
    assert not hasattr(cfg.model, "encoder")


# ---------------------------------------------------------------------------
# Cycle 6 — VAEBase fields present on art model
# ---------------------------------------------------------------------------

def test_art_model_has_vae_base_fields(tmp_path):
    cfg_path = _write_yaml(tmp_path / "art.yaml", {"model": _MINIMAL_ART_MODEL})
    cfg = load_config(cfg_path)
    assert cfg.model.encoder.type == "dictionary"
    assert cfg.model.hierarchy.enabled is False


# ---------------------------------------------------------------------------
# Cycle 7 — top-level hierarchy is auto-migrated into model.hierarchy
# ---------------------------------------------------------------------------

def test_top_level_hierarchy_is_migrated_into_model(tmp_path):
    data = {
        "model": _MINIMAL_ART_MODEL,
        "hierarchy": {"enabled": True, "levels": ["linear"]},
    }
    cfg_path = _write_yaml(tmp_path / "legacy.yaml", data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(cfg_path)
    assert cfg.model.hierarchy.enabled is True
    assert any("hierarchy" in str(w.message).lower() for w in caught)
    assert not hasattr(cfg, "hierarchy")


# ---------------------------------------------------------------------------
# Cycle 8 — missing model.type raises ValidationError
# ---------------------------------------------------------------------------

def test_missing_model_type_raises_validation_error(tmp_path):
    data = {"model": {"latent_dim": 128}}
    cfg_path = _write_yaml(tmp_path / "bad.yaml", data)
    with pytest.raises(ValidationError):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# Factory integration — _load_train_config uses load_config internally
# ---------------------------------------------------------------------------

_FULL_BASE_CONFIG = {
    "data": {
        "mode": "csv",
        "data_dir": "./data",
        "data_files": [],
        "columns": {
            "meter_id": "meter_number",
            "timestamp_dst": "timestamp_dst",
            "timestamp_utc": "timestamp_utc",
            "value_cols": ["hourly_energy_consumption"],
        },
        "nan_as_zero": False,
        "max_value": 200.0,
        "splits": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1, "random_seed": 42},
        "window": {"aggregated": True, "input_len": 24, "stride": 24},
        "batch_size": 32,
        "shuffle": True,
        "replay_buffer_capacity": 2000,
        "min_buffer_size": 0,
        "num_workers": 0,
        "steps_per_epoch": 100,
        "eval_batches": 5,
        "eval_cache_on_gpu": False,
        "eval_cache_refresh": False,
        "cache_scan_batches": 0,
        "skip_corrupt_files": True,
        "max_skip_warnings": 20,
        "aggregation_energy_unit": "kwh",
        "output_aggregate_stream": False,
    },
    "normalization": {
        "method": "log1p",
        "cache_dir": None,
        "force_recompute": False,
        "log1p": {"epsilon": 1e-6},
    },
    "conditioning": {"tags": []},
    "training": {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": True,
        "gradient_clip_max_norm": 1.0,
        "lr_scheduler": {"enabled": True, "scheduler_type": "cosine", "step_size": 10, "gamma": 0.9},
        "num_epochs": 10,
        "beta": 1.0,
        "beta_annealing": {"enabled": True, "n_cycles": 1, "ratio": 0.25},
        "weighted_loss": 0.0,
        "early_stopping": {"enabled": False, "patience": 7, "min_delta": 0.001},
        "checkpointing": {"save_freq": 5, "save_dir": "./models", "save_best_model": True,
                          "save_best_after_fraction": 0.25, "save_final_model": True},
    },
    "logging": {"log_dir": "./models", "save_config": True, "config_save_dir": "./models"},
    "visualize": {"num_samples": 5, "num_recon_per": 5, "num_synthetic": 10,
                  "num_originals": 4, "num_synthetic_per": 5},
    "model": _MINIMAL_ART_MODEL,
}


# ---------------------------------------------------------------------------
# Phased config migration — all 7 override files load and validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override,expected_type", [
    ("config/tests/phased/art.yaml",              "art"),
    ("config/tests/phased/diffusion.yaml",        "diffusion"),
    ("config/tests/phased/baseline_average.yaml", "baseline_average"),
    ("config/tests/phased/baseline_quantile.yaml","baseline_quantile"),
    ("config/tests/phased/phase_splitter.yaml",   "phase_splitter"),
    ("config/tests/phased/sequential.yaml",       "sequential_vae"),
    ("config/tests/phased/set_guide.yaml",        "set_guide_vae"),
])
def test_phased_override_loads_with_correct_model_type(override, expected_type):
    cfg = load_config(ROOT_DIR / override)
    assert cfg.model.type == expected_type
    # base must have been merged in: training is only declared in base.yaml
    assert cfg.training["num_epochs"] is not None


def test_load_train_config_uses_loader_and_sets_aliases(tmp_path):
    """_load_train_config calls load_config internally, so base: merge works and aliases are set."""
    base = {**_FULL_BASE_CONFIG}
    override = {
        "base": "base.yaml",
        "model": {**_MINIMAL_ART_MODEL, "latent_dim": 256},
    }
    _write_yaml(tmp_path / "base.yaml", base)
    _write_yaml(tmp_path / "override.yaml", override)

    cfg = _load_train_config(str(tmp_path / "override.yaml"))

    assert cfg.model_type == "art"
    assert cfg.window.input_len == 24       # alias → data.window.input_len from base
    assert cfg.model.latent_dim == 256      # override wins
    assert cfg.normalizer.method == "log1p" # normalizer alias from normalization
    assert cfg.data_dir == "./data"         # data_dir alias