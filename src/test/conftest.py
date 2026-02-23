"""
Shared pytest fixtures for model validation tests.

Provides data loading, model creation, and model validator fixtures
using data/test_data_small.csv.
"""
import os
import sys
import pytest
import torch

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from src.dataloader import SmartMeterDataModule, WindowSpec
from src.normalizer import Normalizer
from src.Conditioner import Conditioner, DayOfWeekConditioner, UserProfiler
from src.Models import VAE, CVAE

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
PROJECT_ROOT = grandparent_dir
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DATA_FILE = "test_data_small.csv"
TEST_DATA_PATH = os.path.join(DATA_DIR, TEST_DATA_FILE)
USER_PROFILE_DIR = os.path.join(PROJECT_ROOT, "models", "UserProfiles")
USER_PROFILE_PATH = os.path.join(USER_PROFILE_DIR, "user_profiler_test.pkl")

HAS_TEST_DATA = os.path.exists(TEST_DATA_PATH)
skip_no_data = pytest.mark.skipif(
    not HAS_TEST_DATA,
    reason=f"Test data not found: {TEST_DATA_PATH}",
)


# ------------------------------------------------------------------
# Data fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def normalizer():
    """Create a log1p normalizer."""
    return Normalizer(method="log1p")


@pytest.fixture(scope="session")
def data_module(normalizer):
    """Load test data via SmartMeterDataModule (session-scoped for speed)."""
    if not HAS_TEST_DATA:
        pytest.skip(f"Test data not found: {TEST_DATA_PATH}")

    dm = SmartMeterDataModule(
        data_dir=DATA_DIR,
        filename=TEST_DATA_FILE,
        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        timestamp_utc_col="timestamp_utc",
        feature_cols=("hourly_energy_consumption",),
        window=WindowSpec(input_len=24, stride=24),
        batch_size=256,
        num_workers=0,
        shuffle_train=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        max_meters=10,
        normalizer=normalizer,
    )
    dm.setup()
    return dm


@pytest.fixture(scope="session")
def train_loader(data_module):
    return data_module.train_dataloader()


@pytest.fixture(scope="session")
def val_loader(data_module):
    return data_module.val_dataloader()


@pytest.fixture(scope="session")
def test_loader(data_module):
    return data_module.test_dataloader()


@pytest.fixture(scope="session")
def input_dim(train_loader):
    """Derive input_dim from first batch."""
    x, *_ = next(iter(train_loader))
    return 24 * x.shape[2]  # L * F


# ------------------------------------------------------------------
# Conditioner fixture (CVAE)
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def conditioner(train_loader):
    """Create and fit Conditioner with DayOfWeek + UserProfiler."""
    os.makedirs(USER_PROFILE_DIR, exist_ok=True)

    profiler = UserProfiler(
        num_clusters=3,
        path=USER_PROFILE_PATH,
        verbose=False,
    )
    if not os.path.exists(USER_PROFILE_PATH):
        profiler.fit(train_loader)

    cond = Conditioner()
    cond.add_condition_item("day_of_week", DayOfWeekConditioner())
    cond.add_condition_item(
        "user_profile",
        UserProfiler(num_clusters=3, path=USER_PROFILE_PATH, verbose=False),
    )
    return cond


# ------------------------------------------------------------------
# Model fixtures
# ------------------------------------------------------------------
@pytest.fixture
def vae_model(input_dim, device):
    """Fresh VAE model on the test device."""
    model = VAE(input_dim=input_dim)
    model.to(device)
    model.prior.to(device)
    return model


@pytest.fixture
def cvae_model(input_dim, conditioner, device):
    """Fresh CVAE model on the test device."""
    model = CVAE(
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
