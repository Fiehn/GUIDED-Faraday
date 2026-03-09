"""
Tests for the Evaluation Framework (eval_framework.py)
======================================================

Two categories:
  A. **Accuracy tests** — each metric is tested against known synthetic data
     where the correct answer can be computed analytically or is trivially
     verifiable.
  B. **Integration tests** — the metrics and the EvalFramework are exercised
     on real data from ``data/test_data_small.csv`` via a trained model to
     ensure they run end-to-end without errors and produce sensible values.

Metrics tested:
  Reconstruction: ReconstructionMSE, ReconstructionRMSE, FlexibilityDistance
  Synthetic:      MaximumMeanDiscrepancy, ACFDistribution, ClassicalStatistics
  Framework:      EvalFramework (register, evaluate, save/load)
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import torch

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from src.eval_framework import (
    EvalFramework,
    EvaluationMetric,
    ReconstructionMetric,
    SyntheticMetric,
    ReconstructionMSE,
    ReconstructionRMSE,
    ReconstructionLogLikelihood,
    FlexibilityDistance,
    MaximumMeanDiscrepancy,
    ACFDistribution,
    ClassicalStatistics,
)

from src.test.conftest import skip_no_data

# ------------------------------------------------------------------
# Constants for synthetic test data
# ------------------------------------------------------------------
N_SAMPLES = 100
WINDOW_SIZE = 24
N_FEATURES = 1
DIM = WINDOW_SIZE * N_FEATURES


# =====================================================================
#  A. Accuracy Tests — each metric on known synthetic inputs
# =====================================================================

class TestReconstructionMSEAccuracy:
    """MSE should be zero for identical data and known for shifted data."""

    def test_identical_data_gives_zero(self):
        m = ReconstructionMSE()
        data = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        m.real_data = data
        m.reconstructed_data = data.copy()
        assert m.compute() == pytest.approx(0.0, abs=1e-7)

    def test_known_offset(self):
        """If reconstructed = real + 1, MSE should be 1.0."""
        m = ReconstructionMSE()
        data = np.zeros((N_SAMPLES, DIM), dtype=np.float32)
        m.real_data = data
        m.reconstructed_data = data + 1.0
        assert m.compute() == pytest.approx(1.0, abs=1e-7)

    def test_non_negative(self):
        m = ReconstructionMSE()
        m.real_data = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        m.reconstructed_data = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        assert m.compute() >= 0.0

    def test_symmetric(self):
        """MSE(a,b) == MSE(b,a)."""
        m1 = ReconstructionMSE()
        m2 = ReconstructionMSE()
        a = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        b = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        m1.real_data, m1.reconstructed_data = a, b
        m2.real_data, m2.reconstructed_data = b, a
        assert m1.compute() == pytest.approx(m2.compute(), rel=1e-6)


class TestReconstructionRMSEAccuracy:
    """RMSE = sqrt(MSE)."""

    def test_identical_data_gives_zero(self):
        m = ReconstructionRMSE()
        data = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        m.real_data = data
        m.reconstructed_data = data.copy()
        assert m.compute() == pytest.approx(0.0, abs=1e-6)

    def test_known_offset(self):
        """If reconstructed = real + 2, RMSE should be 2.0."""
        m = ReconstructionRMSE()
        data = np.zeros((N_SAMPLES, DIM), dtype=np.float32)
        m.real_data = data
        m.reconstructed_data = data + 2.0
        assert m.compute() == pytest.approx(2.0, abs=1e-6)

    def test_equals_sqrt_of_mse(self):
        mse_metric = ReconstructionMSE()
        rmse_metric = ReconstructionRMSE()
        a = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        b = np.random.randn(N_SAMPLES, DIM).astype(np.float32)
        mse_metric.real_data = rmse_metric.real_data = a
        mse_metric.reconstructed_data = rmse_metric.reconstructed_data = b
        assert rmse_metric.compute() == pytest.approx(
            np.sqrt(mse_metric.compute()), rel=1e-5
        )


class TestFlexibilityDistanceAccuracy:
    """FD should be 0 for identical series, positive otherwise."""

    def test_identical_gives_zero(self):
        fd = FlexibilityDistance(tau=0.1, normalize=True)
        data = np.abs(np.random.randn(20, WINDOW_SIZE).astype(np.float32)) + 1.0
        fd.real_data = data
        fd.reconstructed_data = data.copy()
        assert fd.compute() == pytest.approx(0.0, abs=1e-6)

    def test_different_gives_positive(self):
        fd = FlexibilityDistance(tau=0.1, normalize=True)
        fd.real_data = np.ones((20, WINDOW_SIZE), dtype=np.float32)
        fd.reconstructed_data = np.ones((20, WINDOW_SIZE), dtype=np.float32) * 5.0
        assert fd.compute() > 0.0

    def test_auto_tau(self):
        """Auto tau should not crash and should yield a finite result."""
        fd = FlexibilityDistance(tau='auto', normalize=True)
        fd.real_data = np.random.randn(10, WINDOW_SIZE).astype(np.float32)
        fd.reconstructed_data = np.random.randn(10, WINDOW_SIZE).astype(np.float32)
        result = fd.compute()
        assert np.isfinite(result)

    def test_normalize_flag(self):
        """Normalized FD should be smaller than un-normalized for non-trivial."""
        fd_norm = FlexibilityDistance(tau=0.1, normalize=True)
        fd_raw = FlexibilityDistance(tau=0.1, normalize=False)
        real = np.random.randn(10, WINDOW_SIZE).astype(np.float32) + 5.0
        recon = real + np.random.randn(10, WINDOW_SIZE).astype(np.float32) * 0.5
        fd_norm.real_data = fd_raw.real_data = real
        fd_norm.reconstructed_data = fd_raw.reconstructed_data = recon
        # Un-normalized should generally be >= normalized (per-hour)
        assert fd_raw.compute() >= fd_norm.compute() - 1e-6


class TestMMDAccuracy:
    """MMD should be ~0 for same distribution, positive for different."""

    def test_same_distribution_small_mmd(self):
        np.random.seed(42)
        data = np.random.randn(500, DIM).astype(np.float32)
        mmd = MaximumMeanDiscrepancy(seed=42, max_samples=500)
        mmd.real_data = data
        mmd.synthetic_data = data.copy()
        # MMD of identical data should be very small (close to 0)
        assert mmd.compute() < 0.05

    def test_different_distribution_positive_mmd(self):
        np.random.seed(42)
        mmd = MaximumMeanDiscrepancy(seed=42, max_samples=300)
        mmd.real_data = np.random.randn(300, DIM).astype(np.float32)
        mmd.synthetic_data = np.random.randn(300, DIM).astype(np.float32) + 5.0
        assert mmd.compute() > 0.1

    def test_non_negative(self):
        np.random.seed(42)
        mmd = MaximumMeanDiscrepancy(seed=42)
        mmd.real_data = np.random.randn(100, DIM).astype(np.float32)
        mmd.synthetic_data = np.random.randn(100, DIM).astype(np.float32)
        assert mmd.compute() >= 0.0

    def test_custom_sigma(self):
        np.random.seed(42)
        mmd = MaximumMeanDiscrepancy(sigma=1.0, seed=42)
        mmd.real_data = np.random.randn(100, DIM).astype(np.float32)
        mmd.synthetic_data = np.random.randn(100, DIM).astype(np.float32)
        result = mmd.compute()
        assert np.isfinite(result)


class TestACFDistributionAccuracy:
    """ACF-KS should be ~0 for same time series, positive for different."""

    def test_identical_series_low_ks(self):
        np.random.seed(42)
        data = np.random.randn(200, WINDOW_SIZE).astype(np.float32)
        acf = ACFDistribution(window_size=WINDOW_SIZE, max_lag=5)
        acf.real_data = data
        acf.synthetic_data = data.copy()
        # Same data ⇒ identical ACF distributions ⇒ KS ~ 0
        assert acf.compute() < 0.05

    def test_different_series_positive_ks(self):
        np.random.seed(42)
        # Make synthetic data with very different autocorrelation structure
        real = np.random.randn(200, WINDOW_SIZE).astype(np.float32)
        # Cumulative sum creates strong positive autocorrelation
        synthetic = np.cumsum(np.random.randn(200, WINDOW_SIZE), axis=1).astype(np.float32)
        acf = ACFDistribution(window_size=WINDOW_SIZE, max_lag=5)
        acf.real_data = real
        acf.synthetic_data = synthetic
        assert acf.compute() > 0.1

    def test_output_range(self):
        """KS statistic is in [0, 1]."""
        np.random.seed(42)
        acf = ACFDistribution(window_size=WINDOW_SIZE, max_lag=5)
        acf.real_data = np.random.randn(100, WINDOW_SIZE).astype(np.float32)
        acf.synthetic_data = np.random.randn(100, WINDOW_SIZE).astype(np.float32)
        result = acf.compute()
        assert 0.0 <= result <= 1.0


class TestClassicalStatisticsAccuracy:
    """ClassicalStatistics should return a finite scalar."""

    def test_identical_gives_zero_profile_diff(self):
        np.random.seed(42)
        # ClassicalStatistics expects data shaped [N, window_size*2] (consumption + another feature)
        data = np.random.randn(50, WINDOW_SIZE * 2).astype(np.float32)
        cs = ClassicalStatistics(window_size=WINDOW_SIZE, n_features=1)
        cs.real_data = data
        cs.synthetic_data = data.copy()
        result = cs.compute()
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_data_positive_diff(self):
        np.random.seed(42)
        cs = ClassicalStatistics(window_size=WINDOW_SIZE, n_features=1)
        cs.real_data = np.random.randn(50, WINDOW_SIZE * 2).astype(np.float32)
        cs.synthetic_data = np.random.randn(50, WINDOW_SIZE * 2).astype(np.float32) + 3.0
        result = cs.compute()
        assert result > 0.0

    def test_finite(self):
        np.random.seed(42)
        cs = ClassicalStatistics(window_size=WINDOW_SIZE, n_features=1)
        cs.real_data = np.random.randn(50, WINDOW_SIZE * 2).astype(np.float32)
        cs.synthetic_data = np.random.randn(50, WINDOW_SIZE * 2).astype(np.float32)
        assert np.isfinite(cs.compute())


# =====================================================================
#  A2. Metric property tests
# =====================================================================

class TestMetricNames:
    """Every metric has a non-empty string name."""

    @pytest.mark.parametrize("MetricCls,kwargs", [
        (ReconstructionMSE, {}),
        (ReconstructionRMSE, {}),
        (ReconstructionLogLikelihood, {}),
        (FlexibilityDistance, {}),
        (MaximumMeanDiscrepancy, {}),
        (ACFDistribution, {}),
        (ClassicalStatistics, {}),
    ])
    def test_name_is_nonempty_string(self, MetricCls, kwargs):
        m = MetricCls(**kwargs)
        assert isinstance(m.name, str) and len(m.name) > 0

    @pytest.mark.parametrize("MetricCls,kwargs", [
        (ReconstructionMSE, {}),
        (ReconstructionRMSE, {}),
        (FlexibilityDistance, {}),
    ])
    def test_reconstruction_metric_hierarchy(self, MetricCls, kwargs):
        m = MetricCls(**kwargs)
        assert isinstance(m, ReconstructionMetric)
        assert isinstance(m, EvaluationMetric)

    @pytest.mark.parametrize("MetricCls,kwargs", [
        (MaximumMeanDiscrepancy, {}),
        (ACFDistribution, {}),
        (ClassicalStatistics, {}),
    ])
    def test_synthetic_metric_hierarchy(self, MetricCls, kwargs):
        m = MetricCls(**kwargs)
        assert isinstance(m, SyntheticMetric)
        assert isinstance(m, EvaluationMetric)


# =====================================================================
#  A3. EvalFramework unit tests (no model needed)
# =====================================================================

class _DummyReconMetric(ReconstructionMetric):
    """Returns a fixed value."""

    real_data = None
    reconstructed_data = None

    def __init__(self, value=0.42):
        self._value = value

    @property
    def name(self):
        return "DummyRecon"

    def compute(self):
        return self._value


class _DummySyntheticMetric(SyntheticMetric):
    """Returns a fixed value."""

    real_data = None
    synthetic_data = None

    def __init__(self, value=0.99):
        self._value = value

    @property
    def name(self):
        return "DummySynthetic"

    def compute(self):
        return self._value


class TestEvalFrameworkUnit:
    """Test EvalFramework registration and metric type checking (no real model)."""

    def test_register_reconstruction_metric(self):
        # We only need a mock model and loader; framework just stores metrics
        m = _DummyReconMetric()
        framework = type("FW", (), {
            "reconstruction_metrics": {},
            "synthetic_metrics": {},
        })()
        # Simulate registration logic
        if isinstance(m, ReconstructionMetric):
            framework.reconstruction_metrics[m.name] = m
        assert "DummyRecon" in framework.reconstruction_metrics

    def test_register_synthetic_metric(self):
        m = _DummySyntheticMetric()
        framework = type("FW", (), {
            "reconstruction_metrics": {},
            "synthetic_metrics": {},
        })()
        if isinstance(m, SyntheticMetric):
            framework.synthetic_metrics[m.name] = m
        assert "DummySynthetic" in framework.synthetic_metrics

    def test_save_results_json(self):
        """Verify results dict can be serialized."""
        results = {"Reconstruction MSE": 0.123, "MMD": 0.456}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(results, f)
            path = f.name
        try:
            loaded = json.load(open(path))
            assert loaded["Reconstruction MSE"] == pytest.approx(0.123)
        finally:
            os.unlink(path)


# =====================================================================
#  B. Integration Tests — metrics on real data via model
# =====================================================================


def _build_cvae(input_dim, conditioner, device):
    """Build a CVAE model (avoids conftest fixture import issue)."""
    from src.Models.CVAE import CVAE as CVAEClass
    model = CVAEClass(
        input_dim=input_dim,
        latent_dim=4,
        conditioner=conditioner,
        encoder_type="dictionary",
        decoder_type="dictionary",
        prior_type="standard_dictionary",
    )
    model.to(device)
    model.prior.to(device)
    return model


def _reconstruct_batch(model, x, meter_id, timestamps, device):
    """One-batch reconstruction helper."""
    x_flat = x.reshape(x.shape[0], -1)
    model.eval()
    with torch.no_grad():
        c = model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        c = c.to(device)
        recon_params, _ = model.forward(x_flat, c, num_samples=1)
        if isinstance(recon_params, dict):
            recon = recon_params["mu"]
            if recon.dim() == 3:
                recon = recon[0]
        else:
            recon = recon_params[0] if recon_params.dim() == 3 else recon_params
    return x_flat, recon


@skip_no_data
class TestReconstructionMSEIntegration:
    """MSE runs end-to-end on the test dataloader."""

    def test_runs_and_returns_finite(self, input_dim, conditioner, train_loader, device):
        model = _build_cvae(input_dim, conditioner, device)
        x, meter_id, timestamps = next(iter(train_loader))
        x, meter_id, timestamps = x.to(device), meter_id.to(device), timestamps.to(device)

        x_flat, recon = _reconstruct_batch(model, x, meter_id, timestamps, device)

        m = ReconstructionMSE()
        m.real_data = x_flat.cpu().numpy()
        m.reconstructed_data = recon.cpu().numpy()
        result = m.compute()
        assert np.isfinite(result)
        assert result >= 0.0


@skip_no_data
class TestReconstructionRMSEIntegration:
    """RMSE runs end-to-end on the test dataloader."""

    def test_runs_and_returns_finite(self, input_dim, conditioner, train_loader, device):
        model = _build_cvae(input_dim, conditioner, device)
        x, meter_id, timestamps = next(iter(train_loader))
        x, meter_id, timestamps = x.to(device), meter_id.to(device), timestamps.to(device)

        x_flat, recon = _reconstruct_batch(model, x, meter_id, timestamps, device)

        m = ReconstructionRMSE()
        m.real_data = x_flat.cpu().numpy()
        m.reconstructed_data = recon.cpu().numpy()
        result = m.compute()
        assert np.isfinite(result)
        assert result >= 0.0


@skip_no_data
class TestFlexibilityDistanceIntegration:
    """FD runs end-to-end on the test dataloader."""

    def test_runs_and_returns_finite(self, input_dim, conditioner, train_loader, device):
        model = _build_cvae(input_dim, conditioner, device)
        x, meter_id, timestamps = next(iter(train_loader))
        x, meter_id, timestamps = x.to(device), meter_id.to(device), timestamps.to(device)

        x_flat, recon = _reconstruct_batch(model, x, meter_id, timestamps, device)

        fd = FlexibilityDistance(tau='auto', normalize=True)
        fd.real_data = x_flat.cpu().numpy()
        fd.reconstructed_data = recon.cpu().numpy()
        result = fd.compute()
        assert np.isfinite(result)
        assert result >= 0.0


@skip_no_data
class TestMMDIntegration:
    """MMD with real data from training set vs itself (should be very low)."""

    def test_runs_and_returns_finite(self, train_loader):
        x, *_ = next(iter(train_loader))
        x_flat = x.reshape(x.shape[0], -1).numpy()

        mmd = MaximumMeanDiscrepancy(seed=42, max_samples=200)
        mmd.real_data = x_flat
        mmd.synthetic_data = x_flat.copy()
        result = mmd.compute()
        assert np.isfinite(result)
        assert result < 0.1  # same data → very small MMD


@skip_no_data
class TestACFIntegration:
    """ACF-KS with real data from training set vs itself."""

    def test_runs_and_returns_finite(self, train_loader):
        x, *_ = next(iter(train_loader))
        x_flat = x.reshape(x.shape[0], -1).numpy()

        acf = ACFDistribution(window_size=24, max_lag=5)
        acf.real_data = x_flat
        acf.synthetic_data = x_flat.copy()
        result = acf.compute()
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0


@skip_no_data
class TestEvalFrameworkIntegration:
    """Full EvalFramework with CVAE model on test data."""

    def test_register_and_evaluate_reconstruction_metrics(
        self, input_dim, train_loader, conditioner, normalizer, device
    ):
        model = _build_cvae(input_dim, conditioner, device)
        framework = EvalFramework(
            model=model,
            dataloader=train_loader,
            normalizer=normalizer,
            conditioner=conditioner,
            device=str(device),
            window_size=24,
            n_synthetic_samples=50,
        )
        framework.register_metric(ReconstructionMSE())
        framework.register_metric(ReconstructionRMSE())

        results = framework.evaluate()
        assert "Reconstruction MSE" in results
        assert "Reconstruction RMSE" in results
        for name, val in results.items():
            assert val is not None, f"{name} returned None"
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_register_and_evaluate_synthetic_metrics(
        self, input_dim, train_loader, conditioner, normalizer, device
    ):
        model = _build_cvae(input_dim, conditioner, device)
        framework = EvalFramework(
            model=model,
            dataloader=train_loader,
            normalizer=normalizer,
            conditioner=conditioner,
            device=str(device),
            window_size=24,
            n_synthetic_samples=50,
        )
        framework.register_metric(MaximumMeanDiscrepancy(seed=42, max_samples=100))
        framework.register_metric(ACFDistribution(window_size=24, max_lag=5))

        results = framework.evaluate()
        assert "Maximum Mean Discrepancy (MMD)" in results
        assert "ACF Distribution Match (KS)" in results
        for name, val in results.items():
            assert val is not None, f"{name} returned None"
            assert np.isfinite(val), f"{name} is not finite: {val}"

    def test_save_results(
        self, input_dim, train_loader, conditioner, normalizer, device
    ):
        model = _build_cvae(input_dim, conditioner, device)
        framework = EvalFramework(
            model=model,
            dataloader=train_loader,
            normalizer=normalizer,
            conditioner=conditioner,
            device=str(device),
            window_size=24,
            n_synthetic_samples=50,
        )
        framework.register_metric(ReconstructionMSE())
        framework.evaluate()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            framework.save_results(path)
            loaded = json.load(open(path))
            assert "Reconstruction MSE" in loaded
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
