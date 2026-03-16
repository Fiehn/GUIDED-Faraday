import torch
from evaluation.eval_framework import FlexibilityDistance # for reconstruction evaluation
from evaluation.eval_framework import ACFDistribution, ClassicalStatistics, TSTR, MaximumMeanDiscrepancy, AggregatedProfiles, PerConditionEval # for distributional evaluation
from src.models.vae import VAE
from src.models.cvae import CVAE
import glob
import os
import sys
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from config_loader import load_config
from src.data.dataloader import load_meter_to_feeder_mapping, map_meter_ids_to_feeder_ids
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
consumption_col = feature_cols[0]
print(f"Features in config: {feature_cols}")
print(f"Using only consumption feature for evaluation: {consumption_col}")

# Create 24-hour windows (one per day) from the data
print("\nCreating daily windows from test data...")
real_data = []
timestamps = []
window_meter_ids = []

for meter_id in sorted(df["meter_number"].unique()):
    meter_data = df[df["meter_number"] == meter_id].sort_values("timestamp_utc").reset_index(drop=True)
    # Parse timestamps robustly:
    # - numeric unix epoch columns are interpreted as seconds
    # - string datetime columns are parsed directly
    ts_col = meter_data["timestamp_utc"]
    if is_numeric_dtype(ts_col):
        meter_timestamps = pd.to_datetime(ts_col, unit='s', utc=True)
    else:
        meter_timestamps = pd.to_datetime(ts_col, utc=True)
    
    print(f"  Meter {meter_id}: {len(meter_data)} records, date range {meter_timestamps.iloc[0]} to {meter_timestamps.iloc[-1]}")
    
    # Group into consecutive 24-hour blocks
    for start_idx in range(0, len(meter_data) - window_size + 1, window_size):
        end_idx = start_idx + window_size
        window = meter_data.iloc[start_idx:end_idx]
        
        # Extract consumption column and convert to numpy
        window_values = window[consumption_col].values.astype(float)
        real_data.append(window_values)
        
        # Use the first timestamp of this 24-hour window
        start_ts = meter_timestamps.iloc[start_idx]
        from datetime import datetime
        dt = start_ts.to_pydatetime()
        print(f"    Window {start_idx}-{end_idx}: {dt.strftime('%Y-%m-%d %A %H:%M')}")
        
        timestamps.append(int(start_ts.timestamp()))
        window_meter_ids.append(int(meter_id))

real_data = np.array(real_data)
timestamps = np.array(timestamps)
window_meter_ids = np.array(window_meter_ids)

print(f"\nCreated {len(real_data)} daily windows from {len(df)} hourly records")
print(f"Real data shape: {real_data.shape}")

# Resolve feeder ids for each 24-hour window using topology utilities from dataloader
meter_to_feeder = load_meter_to_feeder_mapping('topology/cleaned/Cleaned_SM_Topology_mapping.json')
window_feeder_ids = map_meter_ids_to_feeder_ids(window_meter_ids, meter_to_feeder)

mapped_count = int(np.sum([f is not None for f in window_feeder_ids]))
print(f"Feeder mapping: {mapped_count}/{len(window_feeder_ids)} windows mapped")

# Check temporal coverage
if len(timestamps) > 0:
    days = set()
    hours = set()
    months = set()
    from datetime import datetime
    for ts in timestamps:
        dt = datetime.utcfromtimestamp(ts)
        days.add(dt.weekday())
        hours.add(dt.hour)
        months.add(dt.month)
    print(f"Temporal coverage: Days={sorted(days)}, Hours={sorted(hours)}, Months={sorted(months)}")

# Create model and generate synthetic data
print("\nCreating model and generating synthetic data...")
# Keep model input dim aligned with training/checkpoint feature count
input_dim = window_size * len(feature_cols)

# Create model
if model_type.upper() == 'CVAE':
    from src.data.conditioner import Conditioner
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
print(f"Generating {len(real_data)} synthetic samples...")
synthetic_data = []
with torch.no_grad():
    for i in range(0, len(real_data), 16):
        batch_size = min(16, len(real_data) - i)
        z = torch.randn(batch_size, latent_dim)
        recon = model.decoder(z)['mu']
        synthetic_data.append(recon.cpu().numpy())

synthetic_data = np.vstack(synthetic_data)

# Convert synthetic outputs to consumption-only [N, window_size]
if synthetic_data.shape[1] % window_size != 0:
    raise ValueError(
        f"Synthetic data feature dimension ({synthetic_data.shape[1]}) is not divisible by window_size ({window_size})."
    )
synthetic_n_features = synthetic_data.shape[1] // window_size
synthetic_data = synthetic_data.reshape(-1, window_size, synthetic_n_features)[:, :, 0]

# Use all real data
real_data_subset = real_data


# Run reconstruction evaluation
print("\n" + "="*70)
print("RECONSTRUCTION EVALUATION METRICS")
print("="*70)

fd_metric = FlexibilityDistance(tau='auto', normalize=True)
fd_metric.real_data = real_data_subset
fd_metric.reconstructed_data = synthetic_data
fd_score = fd_metric.compute()
print(f"Flexibility Distance: {fd_score:.4f}") # (lower is better, 0 = perfect reconstruction)
 


# Run distributional evaluation
print("\n" + "="*70)
print("DISTRIBUTIONAL EVALUATION METRICS")
print("="*70)

# Visual evaluation
print("\nVisual Evaluation:")
visual_eval.real_data = real_data_subset
visual_eval.synthetic_data = synthetic_data
visual_eval.pca()
visual_eval.tsne()
print("PCA and t-SNE plots saved to Output/Plots")

# ACF Distribution metric
acf_metric = ACFDistribution(window_size=window_size, max_lag=10)
acf_metric.real_data = real_data_subset
acf_metric.synthetic_data = synthetic_data
acf_score = acf_metric.compute()
print("\nACF Distribution Match:")
print(f"  Average Kolmogorov-Smirnov (KS) test statistic across lags: {acf_score:.4f}") # (lower is better: 0 = perfect match, 1 = completely different)

# Maximum Mean Discrepancy
print("\nMaximum Mean Discrepancy (MMD):")
mmd_metric = MaximumMeanDiscrepancy(sigma=None, seed=42, max_samples=2000)
mmd_metric.real_data = real_data_subset
mmd_metric.synthetic_data = synthetic_data
mmd_score = mmd_metric.compute()
print(f"  MMD score: {mmd_score:.6f}") # (lower is better, 0 = identical distributions)

# Classicl statistics match
print("\nClassical Statistics Match:")
stats_metric = ClassicalStatistics(window_size=window_size, n_features=1)
stats_metric.real_data = real_data_subset
stats_metric.synthetic_data = synthetic_data
stats_score = stats_metric.compute()
print(f"  Mean profile MAE: {stats_score:.6f}")

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

# Aggregated Profiles evaluation
print("\n" + "="*70)
print("AGGREGATED PROFILES EVALUATION")
print("="*70)
agg_metric = AggregatedProfiles(
    n_samples=100,
    window_size=window_size,
    tau='auto',
    normalize=True
)
agg_metric.real_data = real_data
agg_metric.synthetic_data = synthetic_data
agg_score = agg_metric.compute()
print(f"\nAggregated Profiles (Flexibility Distance): {agg_score:.6f}") # (lower is better, 0 = perfect match between summed profiles)


# Per condition evaluation 
print("\n" + "="*70)
print("PER-CONDITION EVALUATION")
print("="*70)
per_condition_eval = PerConditionEval(
    window_size=window_size, 
    max_lag=10,
    min_samples_per_group=2,
    output_dir='src/Output/',
    verbose=True,
    predictor_type='linear',
    tstr_epochs=50,
    min_samples_per_feeder=2  # Lower threshold for test data with few samples
)
per_condition_eval.real_data = real_data
per_condition_eval.synthetic_data = synthetic_data
per_condition_eval.timestamps = timestamps
per_condition_eval.feeder_ids = window_feeder_ids
per_condition_eval.compute()
