from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.conditioner import UserProfiler  # noqa: E402 (import used by all tests)


def _fitted_profiler(n_topics=3, num_clusters=5, feature_dim=10) -> UserProfiler:
    """Return a UserProfiler in a simulated post-fit state with cluster_centers_ set."""
    p = UserProfiler(n_topics=n_topics, num_clusters=num_clusters, path=None)
    p.cluster_centers_ = np.random.default_rng(0).standard_normal((num_clusters, feature_dim)).astype(np.float32)
    p.kmeans_initialized = True
    p.is_fitted = True
    return p


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: dimension change resets kmeans
# ---------------------------------------------------------------------------

def test_ensure_feature_dim_resets_kmeans_on_change():
    p = _fitted_profiler(feature_dim=10)
    old_kmeans = p.kmeans

    p._ensure_kmeans_feature_dim(20, context="test")

    assert p.kmeans is not old_kmeans
    assert p.cluster_centers_ is None
    assert not p.kmeans_initialized


# ---------------------------------------------------------------------------
# Cycle 2 — no-op when no dimension is known yet
# ---------------------------------------------------------------------------

def test_ensure_feature_dim_noop_when_no_dim_known():
    p = UserProfiler(n_topics=3, num_clusters=5, path=None)
    old_kmeans = p.kmeans

    p._ensure_kmeans_feature_dim(10, context="test")

    assert p.kmeans is old_kmeans


# ---------------------------------------------------------------------------
# Cycle 3 — no-op when incoming dimension matches current
# ---------------------------------------------------------------------------

def test_ensure_feature_dim_noop_on_same_dim():
    p = _fitted_profiler(feature_dim=10)
    old_kmeans = p.kmeans

    p._ensure_kmeans_feature_dim(10, context="test")

    assert p.kmeans is old_kmeans


# ---------------------------------------------------------------------------
# Cycle 4 — cached profile preserved after dimension change; not uniform
# ---------------------------------------------------------------------------

def test_cached_profile_preserved_and_not_uniform_after_dimension_change():
    p = _fitted_profiler(feature_dim=10)
    stored = np.array([0.8, 0.15, 0.05], dtype=np.float32)
    p.user_profiles_cache = {42: stored.copy()}
    p.gamma_cache = {42: np.array([8.0, 1.5, 0.5])}

    # Trigger dimension change: 10 → 20
    p._ensure_kmeans_feature_dim(20, context="test")

    # Profile for meter 42 must still be in cache
    assert 42 in p.user_profiles_cache, "Cached profile was wiped on dimension change"
    np.testing.assert_array_equal(p.user_profiles_cache[42], stored)

    # get_conditions must return the preserved profile, not the uniform fallback
    result = p.get_conditions(data=None, meter_id=np.array([42], dtype=np.int64))
    uniform = np.full(3, 1.0 / 3, dtype=np.float32)
    assert result.shape == (1, 3)
    assert not np.allclose(result.numpy()[0], uniform, atol=1e-4), "get_conditions returned uniform"
    np.testing.assert_allclose(result.numpy()[0], stored, atol=1e-6)


# ---------------------------------------------------------------------------
# Cycle 5 — all public conditioner classes importable from data.conditioner
# ---------------------------------------------------------------------------

def test_all_public_conditioner_classes_importable():
    from data.conditioner import (  # noqa: F401
        BaseConditioner,
        UserProfiler,
        Conditioner,
        DayOfWeekConditioner,
        MonthOfYearConditioner,
        HolidayConditioner,
        HeatpumpConditioner,
        SolarPanelConditioner,
        ElectricVehicleConditioner,
        ElectricVehicleConditionerBool,
        ActivePhaseCountConditioner,
        HeatpumpPhaseConditioner,
    )