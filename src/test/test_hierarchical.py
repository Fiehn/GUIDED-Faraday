"""
Unit tests for the Hierarchical Latent Module.

Tests cover:
    - Registry (register, get, clear, force_register)
    - Config (LevelConfig, HierarchicalConfig serialization)
    - Individual level strategies (linear, stochastic, vq, information_bottleneck)
    - Aggregation strategies (concat, sum, gated, attention)
    - HierarchicalLatentModule (construction, forward, active_levels, residual, top-down)
    - Save/Load of VAE with hierarchy through checkpoint
    - Integration with real data from data/test_data_small.csv
"""

import os
import sys
import copy
import json
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------
# Path setup (same pattern as the rest of the repo)
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from src.Hierarchical.registry import StrategyRegistry
from src.Hierarchical.config import LevelConfig, HierarchicalConfig
from src.Hierarchical.levels import (
    LinearLevel,
    StochasticLevel,
    VQLevel,
    InformationBottleneckLevel,
)
from src.Hierarchical.aggregation import (
    ConcatAggregation,
    SumAggregation,
    GatedAggregation,
    AttentionAggregation,
    _AGGREGATION_STRATEGIES,
)
from src.Hierarchical.core import HierarchicalLatentModule

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
BATCH = 8
ENCODER_DIM = 32
LATENT_DIM_A = 16
LATENT_DIM_B = 24

DATA_DIR = os.path.join(grandparent_dir, "data")
TEST_CSV = os.path.join(DATA_DIR, "test_data_small.csv")


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z_enc(device):
    """Fake encoder output [B, ENCODER_DIM]."""
    return torch.randn(BATCH, ENCODER_DIM, device=device)


def _make_two_level_config(aggregation="concat", residual=False, top_down=False):
    """Helper: 2-level config with linear + stochastic."""
    cfg = HierarchicalConfig(
        aggregation=aggregation,
        residual_connections=residual,
    )
    cfg.add_level(LevelConfig(
        name="coarse",
        input_dim=ENCODER_DIM,
        latent_dim=LATENT_DIM_A,
        strategy="linear",
        loss_weight=1.0,
    ))
    cfg.add_level(LevelConfig(
        name="fine",
        input_dim=ENCODER_DIM,
        latent_dim=LATENT_DIM_B,
        strategy="stochastic",
        top_down=top_down,
        loss_weight=0.5,
    ))
    return cfg


# =====================================================================
#  1. Registry Tests
# =====================================================================
class TestStrategyRegistry:
    def test_builtin_strategies_registered(self):
        for name in ("linear", "stochastic", "vq", "information_bottleneck"):
            assert name in StrategyRegistry.available(), f"{name} not registered"

    def test_get_returns_class(self):
        cls = StrategyRegistry.get("linear")
        assert cls is LinearLevel

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            StrategyRegistry.get("nonexistent_strat")

    def test_register_and_force_register(self):
        @StrategyRegistry.force_register("_test_dummy")
        class _DummyLevel(nn.Module):
            pass

        assert StrategyRegistry.get("_test_dummy") is _DummyLevel
        # Cleanup
        StrategyRegistry._registry.pop("_test_dummy", None)

    def test_duplicate_register_raises(self):
        """Registering the same name twice (without force) should raise."""
        @StrategyRegistry.force_register("_test_dup")
        class A(nn.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @StrategyRegistry.register("_test_dup")
            class B(nn.Module):
                pass

        StrategyRegistry._registry.pop("_test_dup", None)

    def test_available_returns_list(self):
        avail = StrategyRegistry.available()
        assert isinstance(avail, list)
        assert len(avail) >= 4


# =====================================================================
#  2. Config Tests
# =====================================================================
class TestLevelConfig:
    def test_defaults(self):
        lc = LevelConfig(name="test", input_dim=32, latent_dim=16)
        assert lc.strategy == "linear"
        assert lc.top_down is False
        assert lc.loss_weight == 1.0
        assert lc.strategy_kwargs == {}

    def test_round_trip_dict(self):
        lc = LevelConfig(
            name="fine", input_dim=64, latent_dim=32,
            strategy="stochastic", top_down=True, loss_weight=0.5,
            strategy_kwargs={"free_bits": 0.1},
        )
        d = lc.to_dict()
        lc2 = LevelConfig.from_dict(d)
        assert lc2.name == lc.name
        assert lc2.strategy == lc.strategy
        assert lc2.top_down is True
        assert lc2.strategy_kwargs == {"free_bits": 0.1}


class TestHierarchicalConfig:
    def test_add_level(self):
        cfg = HierarchicalConfig()
        assert len(cfg.levels) == 0
        cfg.add_level(LevelConfig(name="a", input_dim=32, latent_dim=16))
        assert len(cfg.levels) == 1

    def test_round_trip_dict(self):
        cfg = _make_two_level_config(aggregation="sum", residual=True, top_down=True)
        d = cfg.to_dict()
        cfg2 = HierarchicalConfig.from_dict(d)
        assert len(cfg2.levels) == 2
        assert cfg2.aggregation == "sum"
        assert cfg2.residual_connections is True
        assert cfg2.levels[1].top_down is True

    def test_round_trip_json(self):
        cfg = _make_two_level_config()
        json_str = json.dumps(cfg.to_dict())
        cfg2 = HierarchicalConfig.from_dict(json.loads(json_str))
        assert len(cfg2.levels) == len(cfg.levels)
        assert cfg2.aggregation == cfg.aggregation


# =====================================================================
#  3. Individual Level Strategy Tests
# =====================================================================
class TestLinearLevel:
    def test_output_shape(self, z_enc):
        level = LinearLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A)
        out = level(z_enc)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)
        assert out["loss"].item() == 0.0

    def test_with_context(self, z_enc):
        ctx_dim = 8
        level = LinearLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, context_dim=ctx_dim)
        ctx = torch.randn(BATCH, ctx_dim)
        out = level(z_enc, context=ctx)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)

    def test_deterministic(self, z_enc):
        level = LinearLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A)
        level.eval()
        o1 = level(z_enc)["z"]
        o2 = level(z_enc)["z"]
        assert torch.allclose(o1, o2)


class TestStochasticLevel:
    def test_output_shape_and_loss(self, z_enc):
        level = StochasticLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A)
        level.train()
        out = level(z_enc)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)
        assert out["loss"].item() >= 0.0
        assert "mu" in out["params"] and "logvar" in out["params"]

    def test_eval_uses_mu(self, z_enc):
        level = StochasticLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A)
        level.eval()
        o1 = level(z_enc)["z"]
        o2 = level(z_enc)["z"]
        assert torch.allclose(o1, o2), "Eval mode should be deterministic (return mu)"

    def test_free_bits(self, z_enc):
        level = StochasticLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, free_bits=1.0)
        level.train()
        out = level(z_enc)
        # With free_bits=1.0, KL per dim is clamped to at least 1.0
        assert out["loss"].item() >= LATENT_DIM_A * 1.0 - 1e-3

    def test_with_context(self, z_enc):
        ctx_dim = 8
        level = StochasticLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, context_dim=ctx_dim)
        ctx = torch.randn(BATCH, ctx_dim)
        out = level(z_enc, context=ctx)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)


class TestVQLevel:
    def test_output_shape(self, z_enc):
        level = VQLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, num_codes=64)
        out = level(z_enc)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)
        assert out["loss"].item() >= 0.0
        assert "indices" in out["params"]

    def test_indices_valid(self, z_enc):
        num_codes = 64
        level = VQLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, num_codes=num_codes)
        out = level(z_enc)
        assert out["params"]["indices"].max().item() < num_codes
        assert out["params"]["indices"].min().item() >= 0

    def test_straight_through_gradient(self, z_enc):
        """Straight-through estimator should allow gradients to flow through level params."""
        level = VQLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, num_codes=64)
        out = level(z_enc)
        out["z"].sum().backward()
        # The straight-through gradient flows through the projection weights
        proj_has_grad = any(p.grad is not None for p in level.proj.parameters() if p.requires_grad)
        assert proj_has_grad, "VQ straight-through should propagate gradients to proj params"


class TestInformationBottleneckLevel:
    def test_output_shape(self, z_enc):
        level = InformationBottleneckLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A)
        out = level(z_enc)
        assert out["z"].shape == (BATCH, LATENT_DIM_A)
        assert out["loss"].item() >= 0.0
        assert "temperature" in out["params"]

    def test_temperature_is_learnable(self):
        level = InformationBottleneckLevel(input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, initial_temp=2.0)
        # Check temperature parameter exists and is close to initial value
        temp = level.log_temperature.exp().item()
        assert abs(temp - 2.0) < 1e-5


# =====================================================================
#  4. Aggregation Strategy Tests
# =====================================================================
LEVEL_DIMS = [LATENT_DIM_A, LATENT_DIM_B]
OUTPUT_DIM = 20


def _make_latents():
    return [torch.randn(BATCH, d) for d in LEVEL_DIMS]


class TestConcatAggregation:
    def test_output_shape(self):
        agg = ConcatAggregation(LEVEL_DIMS, OUTPUT_DIM)
        out = agg(_make_latents())
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_gradient_flow(self):
        latents = [t.requires_grad_(True) for t in _make_latents()]
        agg = ConcatAggregation(LEVEL_DIMS, OUTPUT_DIM)
        out = agg(latents)
        out.sum().backward()
        for t in latents:
            assert t.grad is not None


class TestSumAggregation:
    def test_output_shape(self):
        agg = SumAggregation(LEVEL_DIMS, OUTPUT_DIM)
        out = agg(_make_latents())
        assert out.shape == (BATCH, OUTPUT_DIM)


class TestGatedAggregation:
    def test_output_shape(self):
        agg = GatedAggregation(LEVEL_DIMS, OUTPUT_DIM)
        out = agg(_make_latents())
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_gates_sum_to_one(self):
        agg = GatedAggregation(LEVEL_DIMS, OUTPUT_DIM)
        latents = _make_latents()
        gates = agg.gate_net(torch.cat(latents, dim=-1))
        assert torch.allclose(gates.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


class TestAttentionAggregation:
    def test_output_shape(self):
        agg = AttentionAggregation(LEVEL_DIMS, OUTPUT_DIM, num_heads=4)
        out = agg(_make_latents())
        assert out.shape == (BATCH, OUTPUT_DIM)


class TestAggregationRegistry:
    def test_all_keys_present(self):
        for key in ("concat", "sum", "gated", "attention"):
            assert key in _AGGREGATION_STRATEGIES


# =====================================================================
#  5. HierarchicalLatentModule Tests
# =====================================================================
class TestHierarchicalLatentModule:
    # --- Construction ---
    def test_construct_basic(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        assert module.num_levels == 2
        assert module.output_dim == ENCODER_DIM  # default aggregation_output_dim falls back to encoder_latent_dim

    def test_construct_empty_config_raises(self):
        cfg = HierarchicalConfig()
        with pytest.raises(ValueError, match="at least one level"):
            HierarchicalLatentModule(cfg, ENCODER_DIM)

    def test_construct_unknown_aggregation_raises(self):
        cfg = HierarchicalConfig(aggregation="unknown_agg")
        cfg.add_level(LevelConfig(name="a", input_dim=ENCODER_DIM, latent_dim=16))
        with pytest.raises(ValueError, match="Unknown aggregation"):
            HierarchicalLatentModule(cfg, ENCODER_DIM)

    # --- Forward ---
    def test_forward_output_keys(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z_enc)
        assert "z" in out
        assert "loss" in out
        assert "level_outputs" in out
        assert len(out["level_outputs"]) == 2

    def test_forward_output_shape(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    def test_forward_loss_is_scalar(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.train()
        out = module(z_enc)
        assert out["loss"].dim() == 0

    def test_forward_gradient_flow(self, z_enc):
        z = z_enc.clone().requires_grad_(True)
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.train()
        out = module(z)
        (out["z"].sum() + out["loss"]).backward()
        assert z.grad is not None

    # --- Active Levels (ablation) ---
    def test_active_levels_subset(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        # Only activate the first level — aggregator expects variable number of inputs
        # but ConcatAggregation requires all dims. Use SumAggregation for this test.
        cfg_sum = _make_two_level_config(aggregation="sum")
        module_sum = HierarchicalLatentModule(cfg_sum, ENCODER_DIM)
        out = module_sum(z_enc, active_levels=[0])
        assert len(out["level_outputs"]) == 1
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    # --- Top-Down Context ---
    def test_top_down_context(self, z_enc):
        cfg = _make_two_level_config(top_down=True)
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    # --- Residual Connections ---
    def test_residual_connections(self, z_enc):
        cfg = _make_two_level_config(residual=True)
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        assert module.residual_projs is not None
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    def test_no_residual_by_default(self, z_enc):
        cfg = _make_two_level_config(residual=False)
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        assert module.residual_projs is None

    # --- Loss Weighting ---
    def test_loss_weighting(self, z_enc):
        """Second level has loss_weight=0.5, so total should reflect that."""
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.train()
        out = module(z_enc)
        # Level 0 (linear) loss is 0; Level 1 (stochastic) loss * 0.5
        level1_loss = out["level_outputs"][1]["loss"]
        expected = 0.5 * level1_loss  # level 0 contributes 0
        assert torch.allclose(out["loss"], expected, atol=1e-5)

    # --- get_level_info ---
    def test_get_level_info(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        info = module.get_level_info()
        assert len(info) == 2
        assert info[0]["name"] == "coarse"
        assert info[1]["strategy"] == "stochastic"
        assert info[0]["params"] > 0

    # --- All Aggregation Types ---
    @pytest.mark.parametrize("agg", ["concat", "sum", "gated", "attention"])
    def test_all_aggregation_strategies(self, z_enc, agg):
        cfg = _make_two_level_config(aggregation=agg)
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    # --- All Level Strategies ---
    @pytest.mark.parametrize("strategy", ["linear", "stochastic", "vq", "information_bottleneck"])
    def test_all_level_strategies_in_module(self, z_enc, strategy):
        cfg = HierarchicalConfig(aggregation="sum")
        kwargs = {}
        if strategy == "vq":
            kwargs = {"num_codes": 32}
        cfg.add_level(LevelConfig(
            name="single",
            input_dim=ENCODER_DIM,
            latent_dim=LATENT_DIM_A,
            strategy=strategy,
            strategy_kwargs=kwargs,
        ))
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.train()
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    # --- Eval / Train Mode ---
    def test_eval_mode_deterministic(self, z_enc):
        cfg = _make_two_level_config()
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.eval()
        o1 = module(z_enc)["z"]
        o2 = module(z_enc)["z"]
        assert torch.allclose(o1, o2)


# =====================================================================
#  6. Save / Load Tests
# =====================================================================
class TestHierarchySaveLoad:
    """Test that HierarchicalConfig survives serialization through the VAE save/load path."""

    def _make_vae_with_hierarchy(self):
        from src.Models import VAE
        cfg = _make_two_level_config(aggregation="sum", residual=True, top_down=True)
        model = VAE(
            input_dim=24,
            distribution_dict={"latent_dim": ENCODER_DIM},
            encoder_type="dictionary",
            decoder_type="dictionary",
            hierarchy_config=cfg,
            device="cpu",
        )
        return model

    def test_config_in_constructor_args(self):
        model = self._make_vae_with_hierarchy()
        args = model._extract_constructor_args()
        assert args["hierarchy_config"] is not None
        assert isinstance(args["hierarchy_config"], dict)
        assert len(args["hierarchy_config"]["levels"]) == 2

    def test_save_and_load_model(self):
        model = self._make_vae_with_hierarchy()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_model(save_path=tmpdir, model_name="test_hier", device="cpu")
            model_path = os.path.join(tmpdir, "test_hier.pt")
            assert os.path.exists(model_path)

            # Load checkpoint and inspect
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            hier_raw = ckpt["constructor_args"]["hierarchy_config"]
            assert hier_raw is not None
            assert len(hier_raw["levels"]) == 2
            assert hier_raw["residual_connections"] is True

    def test_load_from_checkpoint(self):
        from src.Models import VAE
        model = self._make_vae_with_hierarchy()
        model.eval()
        # Run a forward pass so the model is fully initialized
        x = torch.randn(4, 24)
        _ = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_model(save_path=tmpdir, model_name="test_hier_full", device="cpu")
            model_path = os.path.join(tmpdir, "test_hier_full.pt")

            loaded = VAE.load_from_checkpoint(model_path, device="cpu")
            assert loaded.hierarchy is not None
            assert loaded.hierarchy.num_levels == 2
            assert loaded.hierarchy.config.residual_connections is True

            # Verify forward pass works on loaded model
            loaded.eval()
            out = loaded(x)
            assert out[0] is not None  # recon_params_dict

    def test_round_trip_state_dict(self):
        """Verify that the saved hierarchy config is faithfully reconstructed."""
        model = self._make_vae_with_hierarchy()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_model(save_path=tmpdir, model_name="test_rt", device="cpu")
            model_path = os.path.join(tmpdir, "test_rt.pt")

            from src.Models import VAE
            loaded = VAE.load_from_checkpoint(model_path, device="cpu")
            assert loaded.hierarchy is not None
            assert loaded.hierarchy.num_levels == model.hierarchy.num_levels
            assert loaded.hierarchy.config.aggregation == model.hierarchy.config.aggregation
            assert loaded.hierarchy.config.residual_connections == model.hierarchy.config.residual_connections

            # Verify hierarchy state_dict keys match
            orig_keys = set(k for k in model.state_dict() if k.startswith("hierarchy."))
            loaded_keys = set(k for k in loaded.state_dict() if k.startswith("hierarchy."))
            assert orig_keys == loaded_keys, f"Hierarchy state_dict keys differ: {orig_keys ^ loaded_keys}"

    def test_save_load_no_hierarchy(self):
        """A VAE without hierarchy should save/load cleanly (regression guard)."""
        from src.Models import VAE
        model = VAE(
            input_dim=24,
            distribution_dict={"latent_dim": 16, "dictionary_size": 50},
            encoder_type="dictionary",
            decoder_type="dictionary",
            device="cpu",
        )
        model.eval()
        x = torch.randn(4, 24)
        _ = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_model(save_path=tmpdir, model_name="no_hier", device="cpu")
            loaded = VAE.load_from_checkpoint(
                os.path.join(tmpdir, "no_hier.pt"), device="cpu"
            )
            assert loaded.hierarchy is None


# =====================================================================
#  7. Integration with Real Data (test_data_small.csv)
# =====================================================================
@pytest.mark.skipif(
    not os.path.exists(TEST_CSV),
    reason="data/test_data_small.csv not found — run src/test/create_test_data.py first",
)
class TestHierarchyWithRealData:
    """End-to-end tests using actual CSV smart-meter data."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Load a small slice of real data via the dataloader."""
        from Data.dataloader import SmartMeterDataModule, WindowSpec
        from Data.normalizer import Normalizer

        self.normalizer = Normalizer(method="log1p")
        self.dm = SmartMeterDataModule(
            data_dir=DATA_DIR,
            filename="test_data_small.csv",
            meter_col="meter_number",
            timestamp_dst_col="timestamp_dst",
            timestamp_utc_col="timestamp_utc",
            feature_cols=("hourly_energy_consumption",),
            window=WindowSpec(input_len=24, stride=24),
            batch_size=32,
            num_workers=0,
            shuffle_train=False,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            max_meters=10,
            normalizer=self.normalizer,
        )
        self.dm.setup()
        self.train_loader = self.dm.train_dataloader()

    def _get_batch_flat(self):
        x, meter_id, timestamps, month_key = next(iter(self.train_loader))
        return x.reshape(x.shape[0], -1), meter_id, timestamps

    def test_vae_with_hierarchy_forward(self):
        """VAE + hierarchy forward pass on real data."""
        from src.Models import VAE

        x_flat, _, _ = self._get_batch_flat()
        input_dim = x_flat.shape[1]

        cfg = HierarchicalConfig(aggregation="concat")
        cfg.add_level(LevelConfig(name="coarse", input_dim=input_dim, latent_dim=16, strategy="linear"))
        cfg.add_level(LevelConfig(name="fine", input_dim=input_dim, latent_dim=16, strategy="stochastic"))

        model = VAE(
            input_dim=input_dim,
            distribution_dict={"latent_dim": 32},
            encoder_type="dictionary",
            decoder_type="dictionary",
            hierarchy_config=cfg,
            device="cpu",
        )
        model.train()
        recon_params, posterior_params = model(x_flat)
        assert recon_params["mu"].shape[1] == x_flat.shape[0]  # [S, B, D]

    def test_vae_hierarchy_loss_backprop(self):
        """Full loss computation and backward pass with real data."""
        from src.Models import VAE

        x_flat, _, _ = self._get_batch_flat()
        input_dim = x_flat.shape[1]

        cfg = HierarchicalConfig(aggregation="sum")
        cfg.add_level(LevelConfig(name="l0", input_dim=input_dim, latent_dim=16, strategy="stochastic"))

        model = VAE(
            input_dim=input_dim,
            distribution_dict={"latent_dim": 32},
            encoder_type="dictionary",
            decoder_type="dictionary",
            hierarchy_config=cfg,
            device="cpu",
        )
        model.train()
        recon_params, posterior_params = model(x_flat)
        loss_dict = model.loss(x_flat, recon_params, posterior_params, beta=1.0)

        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad
        loss_dict["loss"].backward()

        # Check hierarchy params received gradients
        for p in model.hierarchy.parameters():
            if p.requires_grad:
                assert p.grad is not None, "Hierarchy param has no gradient"

    def test_vae_hierarchy_train_step(self):
        """Full train_core step with real data and hierarchy."""
        from src.Models import VAE

        x_flat, _, _ = self._get_batch_flat()
        input_dim = x_flat.shape[1]

        cfg = HierarchicalConfig(aggregation="sum")
        cfg.add_level(LevelConfig(name="l0", input_dim=input_dim, latent_dim=16, strategy="linear"))

        model = VAE(
            input_dim=input_dim,
            distribution_dict={"latent_dim": 32},
            encoder_type="dictionary",
            decoder_type="dictionary",
            hierarchy_config=cfg,
            device="cpu",
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run one training step
        batch = next(iter(self.train_loader))
        loss_dict = model.train_core(batch, optim, beta=1.0)
        assert loss_dict["loss"].item() > 0

    def test_save_load_with_real_data(self):
        """Save and reload a hierarchy-equipped VAE, verify structure on real data."""
        from src.Models import VAE

        x_flat, _, _ = self._get_batch_flat()
        input_dim = x_flat.shape[1]

        cfg = HierarchicalConfig(aggregation="sum", residual_connections=True)
        cfg.add_level(LevelConfig(name="coarse", input_dim=input_dim, latent_dim=16, strategy="linear"))
        cfg.add_level(LevelConfig(name="fine", input_dim=input_dim, latent_dim=16, strategy="stochastic", top_down=True))

        model = VAE(
            input_dim=input_dim,
            distribution_dict={"latent_dim": 32},
            encoder_type="dictionary",
            decoder_type="dictionary",
            hierarchy_config=cfg,
            device="cpu",
        )
        model.eval()
        _ = model(x_flat)  # ensure forward works before save

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_model(save_path=tmpdir, model_name="hier_real", device="cpu")
            loaded = VAE.load_from_checkpoint(
                os.path.join(tmpdir, "hier_real.pt"), device="cpu"
            )
            assert loaded.hierarchy is not None
            assert loaded.hierarchy.num_levels == 2
            assert loaded.hierarchy.config.residual_connections is True
            assert loaded.hierarchy.config.levels[1].top_down is True

            # Verify loaded model can do forward pass on real data
            loaded.eval()
            out_after = loaded(x_flat)
            assert out_after[0]["mu"].shape[-1] == input_dim


# =====================================================================
#  8. Edge Cases
# =====================================================================
class TestEdgeCases:
    def test_single_level_module(self, z_enc):
        cfg = HierarchicalConfig(aggregation="sum")
        cfg.add_level(LevelConfig(name="only", input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A, strategy="linear"))
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)

    def test_three_level_mixed_strategies(self, z_enc):
        cfg = HierarchicalConfig(aggregation="sum")
        for name, strat, kwargs in [
            ("coarse", "information_bottleneck", {}),
            ("mid", "stochastic", {"free_bits": 0.5}),
            ("fine", "vq", {"num_codes": 32}),
        ]:
            cfg.add_level(LevelConfig(
                name=name, input_dim=ENCODER_DIM, latent_dim=LATENT_DIM_A,
                strategy=strat, strategy_kwargs=kwargs,
            ))
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        module.train()
        out = module(z_enc)
        assert out["z"].shape == (BATCH, ENCODER_DIM)
        assert len(out["level_outputs"]) == 3

    def test_batch_size_one(self, device):
        z = torch.randn(1, ENCODER_DIM, device=device)
        cfg = _make_two_level_config(aggregation="attention")
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z)
        assert out["z"].shape == (1, ENCODER_DIM)

    def test_large_batch(self, device):
        z = torch.randn(256, ENCODER_DIM, device=device)
        cfg = _make_two_level_config(aggregation="gated")
        module = HierarchicalLatentModule(cfg, ENCODER_DIM)
        out = module(z)
        assert out["z"].shape == (256, ENCODER_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
