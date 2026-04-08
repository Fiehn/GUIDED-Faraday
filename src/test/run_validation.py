"""
Standalone validation runner
============================
Creates model + data from config and runs the full validation suite.
Can also be called from the training pipeline or pytest.

Usage:
    python -m src.test.run_validation --model cvae --test-data --verbose
    python -m src.test.run_validation --model vae  --test-data
"""

import os
import sys
import argparse
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from data.dataloader import SmartMeterDataModule, WindowSpec
from data.normalizer import Normalizer
from data.conditioner import Conditioner, DayOfWeekConditioner, UserProfiler
from src.model.cvae import CVAE
from src.model.vae import VAE 

from src.test.validate_model import validate_before_training


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = "data/"
_DEFAULT_FILENAME = "test_data_small.csv"
_DEFAULT_MAX_METERS = 10
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_INPUT_LEN = 24
_DEFAULT_STRIDE = 24


def build_data_module(
    use_test_data: bool = True,
    data_dir: str = _DEFAULT_DATA_DIR,
    filename: str | None = None,
    max_meters: int = _DEFAULT_MAX_METERS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> SmartMeterDataModule:
    """Build a data module, optionally using the small test csv."""
    if filename is None:
        filename = _DEFAULT_FILENAME if use_test_data else "hourly_measurements_2023_12.csv"

    dm = SmartMeterDataModule(
        data_dir=data_dir,
        filename=filename,
        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        timestamp_utc_col="timestamp_utc",
        feature_cols=("hourly_energy_consumption",),
        window=WindowSpec(input_len=_DEFAULT_INPUT_LEN, stride=_DEFAULT_STRIDE),
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=False,
        seed=42,
        train_frac=0.8,
        val_frac=0.1,
        max_meters=max_meters,
    )
    dm.setup()
    return dm


def build_model(
    model_type: str,
    input_dim: int,
    conditioner: Conditioner | None = None,
    device: str = "cpu",
):
    """Return a fresh model ready for validation."""
    latent_dim = 16
    prior_type = "standard_dictionary"

    if model_type.lower() == "cvae":
        if conditioner is None:
            conditioner = Conditioner()
            conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            conditioner.add_condition_item(
                "user_profile",
                UserProfiler(n_topics=5, num_clusters=30, random_state=42, path=None),
            )

        model = CVAE(
            input_dim=input_dim,
            conditioner=conditioner,
            latent_dim=latent_dim,
            prior_type=prior_type,
            device=device,
        )
    else:
        model = VAE(
            input_dim=input_dim,
            prior_type=prior_type,
            device=device,
        )

    return model.to(device)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_validation(
    model_type: str = "cvae",
    use_test_data: bool = True,
    verbose: bool = True,
) -> bool:
    """End-to-end validation.  Returns True when all checks pass."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n=== Validation Runner ===")
        print(f"  model_type : {model_type}")
        print(f"  device     : {device}")
        print(f"  test_data  : {use_test_data}\n")

    # ---- data ----
    dm = build_data_module(use_test_data=use_test_data)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # input dim
    x, *_ = next(iter(train_loader))
    input_dim = x.shape[1] * x.shape[2]  # L * F

    # ---- normalizer ----
    normalizer = Normalizer("log1p")
    normalizer.fit(x)

    # ---- conditioner (CVAE only) ----
    conditioner = None
    if model_type.lower() == "cvae":
        profiler = UserProfiler(n_topics=5, num_clusters=3, random_state=42, path=None)
        profiler.fit(train_loader)

        conditioner = Conditioner()
        conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
        conditioner.add_condition_item("user_profile", profiler)
        if verbose:
            print(f"  Conditioner tags: {conditioner.get_condition_tags()}")

    # ---- model ----
    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        conditioner=conditioner,
        device=device,
    )

    # ---- validate ----
    return validate_before_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-training model validation")
    parser.add_argument("--model", choices=["vae", "cvae"], default="cvae",
                        help="Model type to validate")
    parser.add_argument("--test-data", action="store_true", default=True,
                        help="Use test_data_small.csv")
    parser.add_argument("--full-data", action="store_true",
                        help="Use full dataset")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    use_test = not args.full_data

    success = run_validation(
        model_type=args.model,
        use_test_data=use_test,
        verbose=verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
