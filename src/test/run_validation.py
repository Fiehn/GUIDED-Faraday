"""
Standalone validation script - can be run independently to validate model setup.
Usage: python src/test/run_validation.py [--model-type vae|cvae]
"""
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from src.encoder_decoder_model import *
from src.Conditioner import *
from src.Models import *
from src.dataloader import *
from src.normalizer import *
from src.test.validate_model import validate_before_training
import torch


def run_validation(model_type: str = "cvae", use_test_data: bool = True, verbose: bool = True):
    """Run validation on specified model type.
    
    Args:
        model_type: 'vae' or 'cvae'
        use_test_data: If True, use small test dataset (fast). If False, use full data.
    """
    model_type = model_type.lower()
    assert model_type in {"vae", "cvae"}, f"Invalid model_type: {model_type}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"\nUsing device: {device}")
        print(f"Model type: {model_type.upper()}\n")
    
    # Load data
    data_dir = "data/"
    
    # Use small test dataset if available for faster validation
    test_data_file = "test_data_small.csv"
    full_data_file = "hourly_measurements_2023_10.csv"
    
    if use_test_data and os.path.exists(os.path.join(data_dir, test_data_file)):
        filename = test_data_file
        if verbose:
            print(f"[TEST DATA] Using small test dataset: {filename}")
            print("            (For full data validation, use --full-data flag)\n")
    else:
        filename = full_data_file
        if use_test_data:
            if verbose:
                print(f"[WARNING] Small test dataset not found. Using full dataset: {filename}")
                print(f"          Create test data with: python src/create_test_data.py\n")
        else:
            if verbose:
                print(f"[FULL DATA] Using full dataset: {filename}\n")
    meter_col = "meter_number"
    timestamp_dst_col = "timestamp_dst"
    timestamp_utc_col = "timestamp_utc"
    feature_cols = ("hourly_energy_consumption",)
    window = WindowSpec(input_len=24, stride=24)
    batch_size = 256
    num_workers = 0
    shuffle_train = False
    seed = 42
    train_frac = 0.8
    val_frac = 0.1
    normalizer = Normalizer(method='log1p') # THIS SHOULD MATCH TRAINING NORMALIZER TODO
    
    if verbose: 
        print("Loading data...")
    dm = SmartMeterDataModule(
        data_dir=data_dir,
        filename=filename,
        meter_col=meter_col,
        timestamp_dst_col=timestamp_dst_col,
        timestamp_utc_col=timestamp_utc_col,
        feature_cols=feature_cols,
        window=window,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=shuffle_train,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        max_meters=10,
        normalizer=normalizer,
    )
    
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    # Get dimensions
    x, meter_id, timestamps = next(iter(train_loader))
    num_features = x.shape[2]
    input_dim = 24 * num_features
    
    # Create or load UserProfiler if needed
    if model_type == "cvae":
        if not os.path.exists("models/UserProfiles/user_profiler_test.pkl"):
            if verbose:
                print("UserProfiler not found, creating...")
            os.makedirs("models/UserProfiles", exist_ok=True)
            UserProfiler(num_clusters=3, path="models/UserProfiles/user_profiler_test.pkl", verbose=verbose).fit(train_loader)
        
        # Create conditioner
        conditioner = Conditioner()
        conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
        conditioner.add_condition_item("user_profile", UserProfiler(num_clusters=3, path="models/UserProfiles/user_profiler_test.pkl", verbose=False))
        
    
    # Create model
    print(f"Creating {model_type.upper()} model...")
    if model_type == "vae":
        model = VAE(input_dim=input_dim)
    else:
        model = CVAE(
            input_dim=input_dim,
            latent_dim=10,
            conditioner=conditioner,
            encoder_type="dictionary",
            decoder_type="dictionary",
            prior_type="standard_dictionary"
        )
    
    # Move to device
    model.to(device)
    model.prior.to(device)
    
    if verbose:
        # Run validation
        print("\n" + "="*80)
        print("STARTING VALIDATION")
        print("="*80 + "\n")
    
    validation_passed = validate_before_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        verbose=verbose,
    )
    
    if validation_passed:
        if verbose:
            print("\n" + "="*80)
            print("✓ ALL VALIDATION TESTS PASSED!")
            print("="*80)
            print("Your model is ready for training.")
            print("Run: python src/train.py")
            print("="*80 + "\n")
        else:
            print("Model validation passed.")
        return True
    else:
        if verbose:
            print("\n" + "="*80)
            print("✗ VALIDATION FAILED")
            print("="*80)
            print("Please fix the issues above before training.")
            print("="*80 + "\n")
        else:
            print("Validation failed. Run with verbose=True for details.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VAE/CVAE model setup")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cvae",
        choices=["vae", "cvae"],
        help="Model type to validate (default: cvae)"
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use full dataset instead of small test data (slower)"
    )
    args = parser.parse_args()
    
    exit_code = run_validation(model_type=args.model_type, use_test_data=not args.full_data)
    sys.exit(exit_code)
