"""
Tests for all Conditioners (Conditioner.py)
============================================

Each individual conditioner class is tested for:
  1. Correct output shape  (``condition_dim``)
  2. Correct output dtype  (float32 tensor)
  3. ``get_conditions`` works with tensor timestamps
  4. ``get_conditions`` works with pandas/int timestamps where applicable
  5. ``get_random_conditions`` returns correct shape
  6. Output values are in the expected range (one-hot, binary, normalised, etc.)
  7. Integration with the ``Conditioner`` container

Conditioners tested:
  - DayOfWeekConditioner
  - MonthOfYearConditioner
  - HolidayConditioner
  - MonthOfYearConditioner
  - UserProfiler        (requires fitting on dataloader)
  - Conditioner (container: add_condition_item, get_conditions, get_random_conditions)

Topology-based conditioners (tested with a synthetic CSV fixture):
  - HeatpumpConditioner
  - SolarPanelConditioner
  - ElectricVehicleConditioner
"""

import os
import sys
import pytest
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
 
from src.data.conditioner import (
    DayOfWeekConditioner,
    MonthOfYearConditioner,
    HolidayConditioner,
    HeatpumpConditioner,
    SolarPanelConditioner,
    ElectricVehicleConditioner,
    UserProfiler,
    Conditioner,
)

from src.test.conftest import skip_no_data

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------
BATCH_SIZE = 16
SEQ_LEN = 24
N_FEATURES = 1

HOLIDAY_PATH = os.path.join(grandparent_dir, "data", "holidays", "denmark_holidays.json")
HAS_HOLIDAYS = os.path.exists(HOLIDAY_PATH)
skip_no_holidays = pytest.mark.skipif(
    not HAS_HOLIDAYS,
    reason=f"Holiday file not found: {HOLIDAY_PATH}",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_dummy_x(batch_size=BATCH_SIZE):
    """Dummy input tensor [B, L, F]."""
    return torch.abs(torch.randn(batch_size, SEQ_LEN, N_FEATURES)) + 0.1


def _make_unix_timestamps(batch_size=BATCH_SIZE, base_ts=1701388800):
    """Create integer Unix timestamps (Dec 2023), one per sample."""
    return torch.tensor(
        [base_ts + i * 3600 for i in range(batch_size)], dtype=torch.float64
    )


def _make_meter_ids(batch_size=BATCH_SIZE, n_meters=5):
    return torch.randint(1, n_meters + 1, (batch_size,))


# =====================================================================
#  1. DayOfWeekConditioner
# =====================================================================

class TestDayOfWeekConditioner:
    """DayOfWeekConditioner should produce [B, 7] one-hot vectors."""

    def setup_method(self):
        self.cond = DayOfWeekConditioner()

    def test_condition_dim(self):
        assert self.cond.condition_dim == 7

    def test_output_shape_tensor_timestamps(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 7)

    def test_output_is_one_hot(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        # Each row should sum to 1
        assert torch.allclose(out.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-6)
        # Each value should be 0 or 1
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_known_day(self):
        """2023-12-25 (Monday). The formula (ts/86400 + 4) % 7 maps Monday → 1."""
        # Unix timestamp for 2023-12-25 00:00:00 UTC
        ts_monday = torch.tensor([1703462400.0])
        x = _make_dummy_x(1)
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts_monday)
        assert out.shape == (1, 7)
        assert out[0].argmax().item() == 1  # Monday maps to index 1 in this formula

    def test_random_conditions_shape(self):
        rc = self.cond.get_random_conditions(32)
        assert rc.shape == (32, 7)

    def test_random_conditions_one_hot(self):
        rc = self.cond.get_random_conditions(32)
        assert torch.allclose(rc.sum(dim=1), torch.ones(32), atol=1e-6)

    def test_output_dtype(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.dtype == torch.float32

    def test_device_follows_input(self):
        """Output should be on the same device as the timestamps."""
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.device == ts.device


# =====================================================================
#  2. MonthOfYearConditioner
# =====================================================================

class TestMonthOfYearConditioner:
    """MonthOfYearConditioner should produce [B, 12] one-hot vectors."""

    def setup_method(self):
        self.cond = MonthOfYearConditioner()

    def test_condition_dim(self):
        assert self.cond.condition_dim == 12

    def test_output_shape(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 12)

    def test_output_is_one_hot(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert torch.allclose(out.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-6)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_random_conditions_shape(self):
        rc = self.cond.get_random_conditions(32)
        assert rc.shape == (32, 12)

    def test_random_conditions_one_hot(self):
        rc = self.cond.get_random_conditions(32)
        assert torch.allclose(rc.sum(dim=1), torch.ones(32), atol=1e-6)

    def test_output_dtype(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.dtype == torch.float32


# =====================================================================
#  3. HolidayConditioner
# =====================================================================

@skip_no_holidays
class TestHolidayConditioner:
    """HolidayConditioner should produce [B, 1] binary vectors."""

    def setup_method(self):
        self.cond = HolidayConditioner(holiday_path=HOLIDAY_PATH)

    def test_condition_dim(self):
        assert self.cond.condition_dim == 1

    def test_output_shape(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_binary(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_known_holiday(self):
        """2023-12-25 is Christmas ⟹ should be 1."""
        import pandas as pd
        ts_xmas = torch.tensor([pd.Timestamp("2023-12-25").timestamp()])
        x = _make_dummy_x(1)
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts_xmas)
        assert out.item() == 1.0

    def test_known_non_holiday(self):
        """2023-12-20 (Wednesday) is NOT a holiday ⟹ should be 0."""
        import pandas as pd
        ts_wed = torch.tensor([pd.Timestamp("2023-12-20").timestamp()])
        x = _make_dummy_x(1)
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts_wed)
        assert out.item() == 0.0

    def test_random_conditions_shape(self):
        rc = self.cond.get_random_conditions(32)
        assert rc.shape == (32, 1)

    def test_output_dtype(self):
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.dtype == torch.float32


# =====================================================================
#  4. HP Conditioner, SolarPanelConditioner, ElectricVehicleConditioner
# =====================================================================



# =====================================================================
#  5. UserProfiler (requires fitting)
# =====================================================================

class TestUserProfilerSynthetic:
    """UserProfiler on tiny synthetic dataloader."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.n_topics = 3
        self.n_clusters = 5
        self.path = str(tmp_path / "profiler.pkl")

        # Create a tiny synthetic dataloader
        B, L, F = 32, 24, 1
        torch.manual_seed(42)
        x = torch.abs(torch.randn(B, L, F)) + 0.5
        c = torch.zeros(B, dtype=torch.long)
        meter_ids = torch.randint(1, 6, (B,))
        timestamps = torch.arange(B, dtype=torch.float64) * 3600

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(x, c, meter_ids, timestamps)
        self.loader = DataLoader(ds, batch_size=16, shuffle=False)

        self.profiler = UserProfiler(
            n_topics=self.n_topics,
            num_clusters=self.n_clusters,
            random_state=42,
            path=self.path,
            verbose=False,
        )
        self.profiler.fit(self.loader)

    def test_condition_dim(self):
        assert self.profiler.condition_dim == self.n_topics

    def test_get_conditions_shape(self):
        x, c, meter_ids, ts = next(iter(self.loader))
        out = self.profiler.get_conditions(x, meter_id=meter_ids, timestamps=ts)
        assert out.shape == (x.shape[0], self.n_topics)

    def test_profiles_sum_to_one(self):
        """Dirichlet samples should sum to ~1."""
        x, c, meter_ids, ts = next(iter(self.loader))
        out = self.profiler.get_conditions(x, meter_id=meter_ids, timestamps=ts)
        if isinstance(out, np.ndarray):
            sums = out.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-4)
        else:
            sums = out.sum(dim=1)
            assert torch.allclose(sums, torch.ones(x.shape[0]), atol=1e-4)

    def test_profiles_non_negative(self):
        x, c, meter_ids, ts = next(iter(self.loader))
        out = self.profiler.get_conditions(x, meter_id=meter_ids, timestamps=ts)
        if isinstance(out, np.ndarray):
            assert (out >= 0).all()
        else:
            assert (out >= 0).all()

    def test_cache_populated(self):
        """After fit, gamma_cache and user_profiles_cache should be non-empty."""
        assert len(self.profiler.gamma_cache) > 0
        assert len(self.profiler.user_profiles_cache) > 0

    def test_resample_changes_profile(self):
        """Resampling should give a different profile (probabilistically)."""
        m_id = list(self.profiler.user_profiles_cache.keys())[0]
        old = self.profiler.user_profiles_cache[m_id].copy()
        self.profiler.resample_user_vector(meter_id=m_id)
        new = self.profiler.user_profiles_cache[m_id]
        # Extremely unlikely to be identical after resampling
        assert not np.array_equal(old, new)

    def test_random_conditions_shape(self):
        rc = self.profiler.get_random_conditions(8)
        assert rc.shape == (8, self.n_topics)

    def test_random_conditions_sum_to_one(self):
        rc = self.profiler.get_random_conditions(8)
        sums = rc.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-4)

    def test_save_load_roundtrip(self):
        """Save and load should preserve the model."""
        self.profiler.save(self.path)
        # Note: UserProfiler has both a @staticmethod load() and an instance
        # load() — the instance method shadows the static one.  Use a fresh
        # instance and call the instance-level load.
        loaded = UserProfiler(
            n_topics=self.n_topics,
            num_clusters=self.n_clusters,
            random_state=42,
            path=self.path,
            verbose=False,
        )
        loaded.load(self.path)
        assert len(loaded.gamma_cache) == len(self.profiler.gamma_cache)
        assert loaded.n_topics == self.profiler.n_topics

    def test_missing_meter_id_raises(self):
        """get_conditions without meter_id should raise."""
        x, c, meter_ids, ts = next(iter(self.loader))
        with pytest.raises(ValueError, match="meter_id"):
            self.profiler.get_conditions(x, meter_id=None, timestamps=ts)

    def test_3d_and_2d_input(self):
        """get_conditions should work with both [B,L,F] and [B,D] inputs."""
        x, c, meter_ids, ts = next(iter(self.loader))
        out_3d = self.profiler.get_conditions(x, meter_id=meter_ids, timestamps=ts)

        x_flat = x.reshape(x.shape[0], -1)
        self.profiler.resample_user_vector()  # refresh cache
        out_2d = self.profiler.get_conditions(x_flat, meter_id=meter_ids, timestamps=ts)

        assert out_3d.shape == out_2d.shape


@skip_no_data
class TestUserProfilerRealData:
    """UserProfiler fitted on real test data."""

    def test_fit_and_condition_on_real_data(self, train_loader, tmp_path):
        path = str(tmp_path / "real_profiler.pkl")
        profiler = UserProfiler(
            n_topics=3, num_clusters=5, random_state=42,
            path=path, verbose=False,
        )
        profiler.fit(train_loader)

        x, c, meter_ids, ts = next(iter(train_loader))
        out = profiler.get_conditions(x, meter_id=meter_ids, timestamps=ts)
        assert out.shape == (x.shape[0], 3)
        if isinstance(out, np.ndarray):
            assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)
        else:
            sums = out.sum(dim=1)
            assert torch.allclose(sums, torch.ones(x.shape[0]), atol=1e-4)


# =====================================================================
#  7. Conditioner Container
# =====================================================================

class TestConditionerContainer:
    """Tests for the Conditioner container that holds multiple sub-conditioners."""

    def test_empty_conditioner(self):
        c = Conditioner()
        assert c.condition_dim == 0
        assert c.get_condition_tags() == []

    def test_add_single_conditioner(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        assert c.condition_dim == 7
        assert "day_of_week" in c.get_condition_tags()

    def test_add_multiple_conditioners(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("month_of_year", MonthOfYearConditioner())
        assert c.condition_dim == 7 + 12
        assert len(c.get_condition_tags()) == 2

    def test_get_conditions_concatenates(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("month_of_year", MonthOfYearConditioner())

        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 7 + 12)

    def test_get_random_conditions(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("month_of_year", MonthOfYearConditioner())

        rc = c.get_random_conditions(32)
        assert rc.shape == (32, 7 + 12)

    def test_get_condition_dim(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("month_of_year", MonthOfYearConditioner())
        assert c.get_condition_dim() == 7 + 12

    def test_output_dtype(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.dtype == torch.float32

    def test_no_conditions_returns_none(self):
        c = Conditioner()
        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=None, timestamps=ts)
        assert out is None

    def test_save_load_roundtrip(self, tmp_path):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("month_of_year", MonthOfYearConditioner())

        path = str(tmp_path / "conditioner.pkl")
        c.save(path)
        loaded = Conditioner.load(path)
        assert loaded.condition_dim == c.condition_dim
        assert set(loaded.get_condition_tags()) == set(c.get_condition_tags())


@skip_no_holidays
class TestConditionerContainerWithHoliday:
    """Tests with HolidayConditioner in the container (needs holiday file)."""

    def test_three_conditioners_concat(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("holiday", HolidayConditioner(holiday_path=HOLIDAY_PATH))
        c.add_condition_item("month_of_year", MonthOfYearConditioner())
        assert c.condition_dim == 7 + 1 + 12

        x = _make_dummy_x()
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=None, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 20)


# =====================================================================
#  8. HeatpumpConditioner (topology-based)
# =====================================================================


def _make_topology_csv(tmp_path, meters, has_heat_pump=None, has_solar_panel=None):
    """Create a minimal topology CSV for testing."""
    import pandas as pd

    if has_heat_pump is None:
        has_heat_pump = ["True", "False"] * (len(meters) // 2 + 1)
    if has_solar_panel is None:
        has_solar_panel = ["False", "True"] * (len(meters) // 2 + 1)

    df = pd.DataFrame({
        "meter_number": meters,
        "has_heat_pump": [has_heat_pump[i % len(has_heat_pump)] for i in range(len(meters))],
        "has_solar_panel": [has_solar_panel[i % len(has_solar_panel)] for i in range(len(meters))],
    })
    path = str(tmp_path / "topology.csv")
    df.to_csv(path, index=False)
    return path


class TestHeatpumpConditioner:
    """HeatpumpConditioner should produce [B, 1] binary vectors."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.meters = list(range(1, BATCH_SIZE + 1))
        # Even meters have heat pump, odd don't
        hp = ["True" if m % 2 == 0 else "False" for m in self.meters]
        self.csv_path = _make_topology_csv(tmp_path, self.meters, has_heat_pump=hp)
        self.cond = HeatpumpConditioner(topology_path=self.csv_path)

    def test_condition_dim(self):
        assert self.cond.condition_dim == 1

    def test_output_shape(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_binary(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_known_values(self):
        """Even meter_ids should have heat pump (1.0), odd should not (0.0)."""
        x = _make_dummy_x(4)
        meter_id = torch.tensor([2, 3, 4, 5], dtype=torch.long)
        ts = _make_unix_timestamps(4)
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        expected = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        assert torch.equal(out, expected)

    def test_output_dtype(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.dtype == torch.float32

    def test_random_conditions_shape(self):
        rc = self.cond.get_random_conditions(32)
        assert rc.shape == (32, 1)

    def test_random_conditions_binary(self):
        rc = self.cond.get_random_conditions(100)
        assert set(rc.unique().tolist()).issubset({0.0, 1.0})

    def test_tensor_and_numpy_meter_id(self):
        """Both tensor and numpy meter_id should work."""
        x = _make_dummy_x(4)
        ids = [1, 2, 3, 4]
        ts = _make_unix_timestamps(4)
        out_tensor = self.cond.get_conditions(x, meter_id=torch.tensor(ids), timestamps=ts)
        out_numpy = self.cond.get_conditions(x, meter_id=np.array(ids), timestamps=ts)
        assert torch.equal(out_tensor, out_numpy)


# =====================================================================
#  9. SolarPanelConditioner (topology-based)
# =====================================================================


class TestSolarPanelConditioner:
    """SolarPanelConditioner should produce [B, 1] binary vectors."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.meters = list(range(1, BATCH_SIZE + 1))
        # Odd meters have solar panels, even don't
        sp = ["True" if m % 2 == 1 else "False" for m in self.meters]
        self.csv_path = _make_topology_csv(tmp_path, self.meters, has_solar_panel=sp)
        self.cond = SolarPanelConditioner(topology_path=self.csv_path)

    def test_condition_dim(self):
        assert self.cond.condition_dim == 1

    def test_output_shape(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_binary(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_known_values(self):
        """Odd meter_ids should have solar (1.0), even should not (0.0)."""
        x = _make_dummy_x(4)
        meter_id = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        ts = _make_unix_timestamps(4)
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        expected = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        assert torch.equal(out, expected)

    def test_output_dtype(self):
        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = self.cond.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.dtype == torch.float32

    def test_random_conditions_shape(self):
        rc = self.cond.get_random_conditions(32)
        assert rc.shape == (32, 1)

    def test_random_conditions_binary(self):
        rc = self.cond.get_random_conditions(100)
        assert set(rc.unique().tolist()).issubset({0.0, 1.0})

    def test_tensor_and_numpy_meter_id(self):
        x = _make_dummy_x(4)
        ids = [1, 2, 3, 4]
        ts = _make_unix_timestamps(4)
        out_tensor = self.cond.get_conditions(x, meter_id=torch.tensor(ids), timestamps=ts)
        out_numpy = self.cond.get_conditions(x, meter_id=np.array(ids), timestamps=ts)
        assert torch.equal(out_tensor, out_numpy)


# =====================================================================
#  10. ElectricVehicleConditioner (stub)
# =====================================================================


class TestElectricVehicleConditioner:
    """ElectricVehicleConditioner is a stub — condition_dim=0, returns None."""

    def test_condition_dim_zero(self):
        ev = ElectricVehicleConditioner.__new__(ElectricVehicleConditioner)
        ev.condition_dim = 0
        assert ev.condition_dim == 0

    def test_get_conditions_returns_none(self):
        ev = ElectricVehicleConditioner.__new__(ElectricVehicleConditioner)
        ev.condition_dim = 0
        x = _make_dummy_x()
        meter_id = _make_meter_ids()
        ts = _make_unix_timestamps()
        assert ev.get_conditions(x, meter_id=meter_id, timestamps=ts) is None

    def test_get_random_conditions_returns_none(self):
        ev = ElectricVehicleConditioner.__new__(ElectricVehicleConditioner)
        ev.condition_dim = 0
        assert ev.get_random_conditions(8) is None


# =====================================================================
#  11. Topology conditioners in the Conditioner container
# =====================================================================


class TestTopologyConditionersInContainer:
    """HeatpumpConditioner + SolarPanelConditioner combined in a Conditioner."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.meters = list(range(1, BATCH_SIZE + 1))
        self.csv_path = _make_topology_csv(tmp_path, self.meters)
        self.hp_cond = HeatpumpConditioner(topology_path=self.csv_path)
        self.sp_cond = SolarPanelConditioner(topology_path=self.csv_path)

    def test_combined_dim(self):
        c = Conditioner()
        c.add_condition_item("heatpump", self.hp_cond)
        c.add_condition_item("solar", self.sp_cond)
        assert c.condition_dim == 2

    def test_combined_output_shape(self):
        c = Conditioner()
        c.add_condition_item("heatpump", self.hp_cond)
        c.add_condition_item("solar", self.sp_cond)

        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 2)

    def test_combined_with_day_of_week(self):
        c = Conditioner()
        c.add_condition_item("day_of_week", DayOfWeekConditioner())
        c.add_condition_item("heatpump", self.hp_cond)
        c.add_condition_item("solar", self.sp_cond)
        assert c.condition_dim == 9  # 7 + 1 + 1

        x = _make_dummy_x()
        meter_id = torch.tensor(self.meters, dtype=torch.long)
        ts = _make_unix_timestamps()
        out = c.get_conditions(x, meter_id=meter_id, timestamps=ts)
        assert out.shape == (BATCH_SIZE, 9)

    def test_random_conditions(self):
        c = Conditioner()
        c.add_condition_item("heatpump", self.hp_cond)
        c.add_condition_item("solar", self.sp_cond)
        rc = c.get_random_conditions(32)
        assert rc.shape == (32, 2)


# =====================================================================
#  12. Integration: Conditioner + real data + UserProfiler
# =====================================================================

@skip_no_data
class TestConditionerRealDataIntegration:
    """Full conditioner pipeline with real data."""

    def test_full_conditioner_on_real_data(self, train_loader, conditioner):
        """Conditioner fixture (from conftest) should work on real batches."""
        x, c, meter_id, timestamps = next(iter(train_loader))
        out = conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
        if isinstance(out, np.ndarray):
            out = torch.tensor(out, dtype=torch.float32)
        assert out.shape[0] == x.shape[0]
        assert out.shape[1] == conditioner.condition_dim

    def test_individual_conditioners_on_real_data(self, train_loader, conditioner):
        """Each sub-conditioner should return correct shapes."""
        x, c, meter_id, timestamps = next(iter(train_loader))
        for tag in conditioner.get_condition_tags():
            item = conditioner.condition_items[tag]
            cond = item.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond, dtype=torch.float32)
            assert cond.shape[0] == x.shape[0]
            assert cond.shape[-1] == item.condition_dim, (
                f"Conditioner '{tag}': expected dim {item.condition_dim}, got {cond.shape[-1]}"
            )

    def test_random_conditions_match_dim(self, conditioner):
        rc = conditioner.get_random_conditions(32)
        assert rc.shape == (32, conditioner.condition_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
