import torch
from eval_framework import FlexibilityDistance # for reconstruction evaluation
from eval_framework import VisualEvaluation, ACFDistribution, ClassicalStatistics, TSTR, MaximumMeanDiscrepancy # for distributional evaluation
from src.Models.VAE import VAE
from src.Models.CVAE import CVAE
import glob
import os
import sys
import numpy as np
import pandas as pd
from config_loader import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test'))
from src.test.create_test_data import create_test_dataset

print("="*70)
print("EVALUATION FRAMEWORK")
print("="*70)

# Create small test dataset
print("\nCreating test dataset...")
test_data_path = create_test_dataset(
    input_file="data/hourly_measurements_2023_10.csv",
    output_file="data/test_data_eval.csv",
    num_meters=5, 
    num_days=3 
)

# Load test data with pandas
print("\n Loading test data...")
df = pd.read_csv(test_data_path)
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample data:\n{df.head()}")

# Load model and config
print("\nLoading model...")
model_files = glob.glob("models/VAE_*.pt")
if not model_files:
    print("No saved models found in models/ directory")
    exit(1)
model_path = sorted(model_files)[-1]
checkpoint = torch.load(model_path, map_location='cpu')
print(f"Model: {model_path}")

# Load config
config_path = "config/config.yaml"
config = load_config(config_path)

# Get parameters
latent_dim = config.latent_dim
model_type = config.model_type
encoder_type = config.encoder_type
decoder_type = config.decoder_type
prior_type = config.prior_type
window_size = config.window_size

# Get feature columns
data_config = config.loader.get_section('data')
feature_cols = data_config['columns']['value_cols']
print(f"Features: {feature_cols}")

# Create windowed data
print("\nCreating windows...")

# Group by meter and create windows
all_windows = []
for meter_id in df["meter_number"].unique():
    meter_data = df[df["meter_number"] == meter_id].sort_values("timestamp_dst")
    
    # Extract feature columns
    print(f"\nMeter {meter_id}:")
    print(f"  Columns available: {meter_data.columns.tolist()}")
    print(f"  Looking for feature_cols: {feature_cols}")
    
    # Check if columns exist
    missing_cols = [col for col in feature_cols if col not in meter_data.columns]
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        continue
    
    values = meter_data[feature_cols].values  # Shape: [T, n_features]
    
    # Fill NaN values with 0
    values = np.nan_to_num(values, nan=0.0)
    
    print(f"  Values shape: {values.shape}")
    print(f"  NaN count after filling: {np.isnan(values).sum()}")
    print(f"  Sample values:\n{values[:5]}")
    
    # Create sliding windows
    for i in range(len(values) - window_size + 1):
        window = values[i:i+window_size]  # Shape: [window_size, n_features]
        all_windows.append(window.flatten())  # Flatten to [window_size * n_features]

real_data = np.array(all_windows)
print(f"Real data: {real_data.shape} (expected: [N, {window_size * len(feature_cols)}])")

# Check for NaN values and remove them
nan_mask = np.isnan(real_data).any(axis=1)
num_nan = nan_mask.sum()
if num_nan > 0:
    print(f"Warning: Found {num_nan} windows with NaN values, removing them...")
    real_data = real_data[~nan_mask]
    print(f"Real data after NaN removal: {real_data.shape}")

if len(real_data) == 0:
    print("ERROR: All windows contain NaN values!")
    exit(1)

# Create model and generate synthetic data
print("\nCreating model and generating synthetic data...")
input_dim = window_size * len(feature_cols)

# Create model
if model_type.upper() == 'CVAE':
    from src.Data.Conditioner import Conditioner
    conditioner = Conditioner(config=config)
    model = CVAE(input_dim=input_dim, latent_dim=latent_dim, encoder_type=encoder_type, 
                 decoder_type=decoder_type, prior_type=prior_type, conditioner=conditioner)
else:
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, encoder_type=encoder_type, 
                decoder_type=decoder_type, prior_type=prior_type)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded: {model.__class__.__name__}")

# Generate synthetic data
print("Generating 100 synthetic samples...")
synthetic_data = []
with torch.no_grad():
    for i in range(0, 100, 16):
        batch_size = min(16, 100 - i)
        z = torch.randn(batch_size, latent_dim)
        recon = model.decoder(z)['mu']
        synthetic_data.append(recon.cpu().numpy())

synthetic_data = np.vstack(synthetic_data)
print(f"Synthetic data: {synthetic_data.shape}")

# Take subset of real data
real_data_subset = real_data[:100]
print(f"Using {len(real_data_subset)} real samples")


# # Run reconstruction evaluation
# print("\n" + "="*70)
# print("RECONSTRUCTION EVALUATION METRICS")
# print("="*70)

# fd_metric = FlexibilityDistance(model, real_data_subset, device='cpu')
# fd_score = fd_metric.compute()
# print(f"Flexibility Distance: {fd_score:.4f}") # (lower is better, 0 = perfect reconstruction)
 


# # Run distributional evaluation
# print("\n" + "="*70)
# print("DISTRIBUTIONAL EVALUATION METRICS")
# print("="*70)

# # Visual evaluation
# print("\nVisual Evaluation:")
# visual_eval = VisualEvaluation(real_data_subset, synthetic_data, output_dir='Output/Plots')
# visual_eval.pca()
# visual_eval.tsne()
# print("PCA and t-SNE plots saved to Output/Plots")

# # ACF Distribution metric
# acf_metric = ACFDistribution(real_data_subset, synthetic_data, window_size=window_size, max_lag=10)
# acf_score = acf_metric.compute()
# print("\nACF Distribution Match:")
# print(f"  Average Kolmogorov-Smirnov (KS) test statistic across lags: {acf_score:.4f}") # (lower is better: 0 = perfect match, 1 = completely different)

# # Maximum Mean Discrepancy
# print("\nMaximum Mean Discrepancy (MMD):")
# mmd_metric = MaximumMeanDiscrepancy(real_data_subset, synthetic_data)
# mmd_score = mmd_metric.compute()
# print(f"  MMD score: {mmd_score:.6f}") # (lower is better, 0 = identical distributions)

# # Classicl statistics match
# print("\nClassical Statistics Match:")
# stats_metric = ClassicalStatistics(real_data, synthetic_data, window_size=24, n_features=2)
# stats_metric.print_comparison()

# TSTR evaluation (Train on Synthetic, Test on Real)
tstr_metric = TSTR(
    task='next_step',
    window_size=window_size,
    n_features=1,
    predictor_type='mlp',  # mlp (multilayer perceptron) / linear
    epochs=50
)
tstr_metric.real_data = real_data
tstr_metric.synthetic_data = synthetic_data
ratio = tstr_metric.compute()
print(f"\nTSTR Ratio: {ratio:.6f}")
