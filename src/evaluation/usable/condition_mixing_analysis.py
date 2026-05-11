import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec
from src.data.conditioner import (
    Conditioner, 
    DayOfWeekConditioner, 
    MonthOfYearConditioner,
    ElectricVehicleConditioner, 
    HeatpumpPhaseConditioner,
    SolarPanelConditioner
)
from src.data.normalizer import Normalizer
from src.model.factory import _load_train_config

def build_features(x_windows):
    """
    Focuses entirely on SHAPE by normalizing each 24h profile individually.
    """
    features = []
    
    # 1. Row-wise Normalization (Crucial Step!)
    # Divide each hour by the maximum hour in that specific day.
    # This bounds every profile between 0.0 and 1.0.
    daily_max = x_windows.max(axis=1, keepdims=True)
    # Add epsilon to avoid division by zero
    x_normalized = x_windows / (daily_max + 1e-6) 
    
    features.append(x_normalized)
    
    return np.hstack(features)


def get_metadata_labels(x, timestamps, meter_ids, conditioners_dict):
    """
    Extract categorical labels directly from conditioners rather than getting tensors,
    or use the conditions tensors to reverse-engineer the categories.
    """
    labels = {}
    
    # We can just run the conditioners and argmax them to get the category
    # (Because they are mostly one-hot encoded)
    
    for name, cond_obj in conditioners_dict.items():
        cond_tensor = cond_obj.get_conditions(x, meter_ids, timestamps)
        if cond_tensor is not None:
            if cond_tensor.shape[1] > 1:
                # One-hot: argmax to get the class index
                idx = cond_tensor.argmax(dim=1).cpu().numpy()
                labels[name] = idx
            else:
                # Binary: just float value
                labels[name] = cond_tensor.squeeze().cpu().numpy()
                
    return labels

def season_from_month(month_idx):
    """Map 0-based month (0=Jan) to Season string."""
    m = month_idx + 1
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]: return 'Spring'
    if m in [6, 7, 8]: return 'Summer'
    if m in [9, 10, 11]: return 'Autumn'
    return 'Unknown'

def day_type_from_day(day_idx):
    """Map 0-based day (0=Mon) to Weekday/Weekend."""
    if day_idx < 5: return 'Weekday'
    return 'Weekend'

def ev_from_idx(idx):
    if idx == 0: return 'No_EV'
    return 'Yes_EV'

def hp_from_idx(idx):
    if idx == 0: return 'No_HP'
    return 'Yes_HP'


def extract_real_data(dataloader_val, conditioners_dict, num_batches=0):
    all_x = []
    metadata = {k: [] for k in conditioners_dict.keys()}
    all_meters = []
    months_in_timestamps = set()
    months_in_condition = set()
    month_debug_samples = []
    dows_in_timestamps = set()
    dows_in_condition = set()
    dow_debug_samples = []

    def _months_from_timestamps(ts):
        """
        Returns calendar months in range 1..12 from a timestamps batch.
        Handles torch tensors (unix seconds) and lists of unix seconds / pd.Timestamp.
        """
        if ts is None:
            return []
        if isinstance(ts, torch.Tensor):
            ts_np = ts.detach().to('cpu')
            if ts_np.is_floating_point():
                ts_np = torch.floor(ts_np)
            ts_np = ts_np.to(dtype=torch.long).numpy()
            dt = pd.to_datetime(ts_np, unit='s', utc=True, errors='coerce')
            return [int(m) for m in dt.month if not pd.isna(m)]

        months = []
        for t in ts:
            if isinstance(t, (int, float, np.integer, np.floating)):
                t = pd.Timestamp(t, unit='s', tz='UTC')
            if isinstance(t, pd.Timestamp):
                months.append(int(t.month))
        return months

    def _dows_from_timestamps(ts):
        """
        Returns day-of-week in range 0..6 where 0=Mon, 6=Sun.
        Handles torch tensors (unix seconds) and lists of unix seconds / pd.Timestamp.
        """
        if ts is None:
            return []
        if isinstance(ts, torch.Tensor):
            ts_np = ts.detach().to("cpu")
            if ts_np.is_floating_point():
                ts_np = torch.floor(ts_np)
            ts_np = ts_np.to(dtype=torch.long).numpy()
            dt = pd.to_datetime(ts_np, unit="s", utc=True, errors="coerce")
            return [int(d) for d in dt.dayofweek if not pd.isna(d)]

        dows = []
        for t in ts:
            if isinstance(t, (int, float, np.integer, np.floating)):
                t = pd.Timestamp(t, unit="s", tz="UTC")
            if isinstance(t, pd.Timestamp):
                dows.append(int(t.dayofweek))
        return dows
    
    
    if num_batches > 0:
        print(f"Extracting up to {num_batches} batches from streaming loader...")
    else:
        print("Extracting from streaming loader (no batch limit)...")
        
    for i, batch in enumerate(tqdm(dataloader_val)):
        if num_batches > 0 and i >= num_batches:
            break
            
        x, meter_id, timestamps, agg = batch
        
        labels = get_metadata_labels(x, timestamps, meter_id, conditioners_dict)
        raw_months = _months_from_timestamps(timestamps)
        for m in raw_months:
            months_in_timestamps.add(m)
        if "month_of_year" in labels:
            cond_months = (labels["month_of_year"] + 1).astype(int).tolist()
            months_in_condition |= set(cond_months)
            if len(month_debug_samples) < 5 and len(raw_months) > 0 and len(cond_months) > 0:
                # Store a few per-sample comparisons for debugging if sets disagree.
                ts_cpu = timestamps.detach().to("cpu").long().tolist() if isinstance(timestamps, torch.Tensor) else list(timestamps)
                for t, rm, cm in zip(ts_cpu[:5], raw_months[:5], cond_months[:5]):
                    if len(month_debug_samples) >= 5:
                        break
                    month_debug_samples.append((t, rm, cm))

        raw_dows = _dows_from_timestamps(timestamps)
        for d in raw_dows:
            dows_in_timestamps.add(d)
        if "day_of_week" in labels:
            cond_dows = labels["day_of_week"].astype(int).tolist()
            dows_in_condition |= set(cond_dows)
            if len(dow_debug_samples) < 5 and len(raw_dows) > 0 and len(cond_dows) > 0:
                ts_cpu = timestamps.detach().to("cpu").long().tolist() if isinstance(timestamps, torch.Tensor) else list(timestamps)
                for t, rd, cd in zip(ts_cpu[:5], raw_dows[:5], cond_dows[:5]):
                    if len(dow_debug_samples) >= 5:
                        break
                    dow_debug_samples.append((t, rd, cd))
        
        all_x.append(x.cpu().numpy())
        for k in labels.keys():
            metadata[k].append(labels[k])
            
        all_meters.append(meter_id.cpu().numpy())
        

    all_x = np.concatenate(all_x, axis=0) # [N, F, 24] or [N, 24]
    if all_x.ndim == 3:
        all_x = all_x.sum(axis=1) # Sum over phases to get [N, 24]

    for k in metadata.keys():
        metadata[k] = np.concatenate(metadata[k], axis=0)
    all_meters = np.concatenate(all_meters)
    

    if len(months_in_timestamps) > 0 or len(months_in_condition) > 0:
        print("\n" + "=" * 50)
        print("MONTH CHECK (timestamps vs conditioner)")
        print("=" * 50)
        print(f"Months found in raw timestamps: {sorted(months_in_timestamps)}")
        print(f"Months found via month_of_year condition: {sorted(months_in_condition)}")
        if months_in_timestamps != months_in_condition:
            missing_in_condition = sorted(months_in_timestamps - months_in_condition)
            extra_in_condition = sorted(months_in_condition - months_in_timestamps)
            if missing_in_condition:
                print(f"WARNING: Months in timestamps but not in condition: {missing_in_condition}")
            if extra_in_condition:
                print(f"WARNING: Months in condition but not in timestamps: {extra_in_condition}")
            if month_debug_samples:
                print("Sample (timestamp_s -> raw_month vs condition_month):")
                for t, rm, cm in month_debug_samples[:5]:
                    try:
                        ts_str = str(pd.to_datetime(int(t), unit="s", utc=True))
                    except Exception:
                        ts_str = str(t)
                    print(f"  {ts_str} -> {rm} vs {cm}")

    if len(dows_in_timestamps) > 0 or len(dows_in_condition) > 0:
        print("\n" + "=" * 50)
        print("DAY-OF-WEEK CHECK (timestamps vs conditioner)")
        print("=" * 50)
        print(f"DOWs found in raw timestamps (0=Mon..6=Sun): {sorted(dows_in_timestamps)}")
        print(f"DOWs found via day_of_week condition: {sorted(dows_in_condition)}")
        if dows_in_timestamps != dows_in_condition:
            missing_in_condition = sorted(dows_in_timestamps - dows_in_condition)
            extra_in_condition = sorted(dows_in_condition - dows_in_timestamps)
            if missing_in_condition:
                print(f"WARNING: DOWs in timestamps but not in condition: {missing_in_condition}")
            if extra_in_condition:
                print(f"WARNING: DOWs in condition but not in timestamps: {extra_in_condition}")
            if dow_debug_samples:
                print("Sample (timestamp_s -> raw_dow vs condition_dow):")
                for t, rd, cd in dow_debug_samples[:5]:
                    try:
                        ts_str = str(pd.to_datetime(int(t), unit="s", utc=True))
                    except Exception:
                        ts_str = str(t)
                    print(f"  {ts_str} -> {rd} vs {cd}")
        
    return all_x, metadata, all_meters

def determine_optimal_clusters(features_pca, max_k=15):
    print("Determining optimal cluster count using Silhouette Score (subsampled)...")
    best_k = 5
    best_score = -1
    
    # Subsample if too large for silhouette
    if features_pca.shape[0] > 10000:
        idx = np.random.choice(features_pca.shape[0], 10000, replace=False)
        subset = features_pca[idx]
    else:
        subset = features_pca
        
    scores = []
    k_range = range(3, max_k + 1)
    
    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(subset)
        score = silhouette_score(subset, kmeans.labels_)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, scores, marker='o')
    plt.title("Silhouette Score vs. K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    os.makedirs("evaluation_comparison/art", exist_ok=True)
    plt.savefig("evaluation_comparison/art/cluster_silhouette.png")
    plt.close()
    
    print(f"Optimal K found: {best_k} (Score: {best_score:.3f})")
    return best_k

def run_pca_and_clustering(x_windows, metadata, k=None):
    print("Building features...")
    features = build_features(x_windows)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    features = features.reshape(features.shape[0], -1) # Flatten if needed
    scaled_features = scaler.fit_transform(features)
    
    print("Running PCA...")
    pca = PCA(n_components=5, random_state=42)
    features_pca = pca.fit_transform(scaled_features)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA top 5 components explain {explained_variance*100:.1f}% of variance.")
    
    if k is None:
        k = determine_optimal_clusters(features_pca)
        
    print(f"Running GMM clustering with {k} clusters...")
    # GMM often works slightly better for overlapping behavioral archetypes
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full', reg_covar=1e-4)
    cluster_labels = gmm.fit_predict(features_pca)
    
    return cluster_labels, features_pca, pca, gmm

def find_dominant_combinations(metadata, cluster_labels, min_support_ratio=0.05):
    """
    Cross-tabulate the metadata to find the dominant combo per cluster.
    """
    df = pd.DataFrame(metadata)
    
    # Translate raw indices into text labels
    if "month_of_year" in df.columns:
        df["season"] = df["month_of_year"].apply(season_from_month)
    if "day_of_week" in df.columns:
        df["day_type"] = df["day_of_week"].apply(day_type_from_day)
    if "ev_phase" in df.columns:
        df["ev"] = df["ev_phase"].apply(ev_from_idx)
    if "heatpump_phase" in df.columns:
        df["hp"] = df["heatpump_phase"].apply(hp_from_idx)
    
    # Target condition columns we care about combining
    condition_cols = ["season", "day_type", "ev", "hp"]
    existing_cols = [c for c in condition_cols if c in df.columns]
    
    print("\n" + "="*50)
    print("DATASET CONDITION DISTRIBUTION")
    print("="*50)
    total_samples = len(df)
    for col in existing_cols:
        counts = df[col].value_counts()
        print(f"\n{col.upper()} Distribution:")
        for val, count in counts.items():
            print(f"  {val}: {count} ({count/total_samples*100:.1f}%)")

    df["cluster"] = cluster_labels
    
    # Analyze each cluster
    k = df["cluster"].nunique()
    dominant_combos = {}
    
    print("\n" + "="*50)
    print("DOMINANT ARCHETYPES PER CLUSTER")
    print("="*50)
    
    for c_id in range(k):
        c_df = df[df["cluster"] == c_id]
        total_in_cluster = len(c_df)
        
        print(f"\nCluster {c_id} (N={total_in_cluster} | {total_in_cluster/len(df)*100:.1f}%)")
        
        if len(existing_cols) == 0:
            continue
            
        # Group by our condition combination
        combo_counts = c_df.groupby(existing_cols).size().reset_index(name='count')
        combo_counts['percentage'] = (combo_counts['count'] / total_in_cluster) * 100
        combo_counts = combo_counts.sort_values('percentage', ascending=False)
        
        # Display top 3 combos in this cluster
        for idx, row in combo_counts.head(3).iterrows():
            combo_str = " | ".join([f"{col}={row[col]}" for col in existing_cols])
            print(f"  - {row['percentage']:5.1f}% : {combo_str}")
            
        # Store the #1 dominant combo for actual evaluation targeting
        top_row = combo_counts.iloc[0]
        if (top_row['percentage'] / 100.0) >= min_support_ratio:
            dominant_combos[c_id] = {col: top_row[col] for col in existing_cols}
            
    return df, dominant_combos

def plot_archetypes(x_windows, cluster_labels, k, output_dir):
    """Plot the mean 24h shape of each cluster to see what we found."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    for c_id in range(k):
        c_idx = (cluster_labels == c_id)
        if not np.any(c_idx):
            continue
            
        c_data = x_windows[c_idx]
        mean_profile = c_data.mean(axis=0)
        std_profile = c_data.std(axis=0)
        
        plt.plot(mean_profile, label=f"Cluster {c_id} (N={np.sum(c_idx)})", linewidth=2)
        # plt.fill_between(range(24), mean_profile-std_profile*0.5, mean_profile+std_profile*0.5, alpha=0.1)
        
    plt.title("Mean 24h Consumption Profiles by Cluster Archetype")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Consumption (Sum of phases)")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "cluster_archetypes.png"))
    plt.close()

def plot_pca_scatters(features_pca, df, output_dir):
    """Plot the top 2 PCA components colored by various metadata labels."""
    os.makedirs(output_dir, exist_ok=True)
    
    # We will only use top 2 components
    pc1 = features_pca[:, 0]
    pc2 = features_pca[:, 1]
    
    # Subsample if too large to avoid massive overplotting
    max_pts = 10000
    if len(pc1) > max_pts:
        np.random.seed(42)
        idx = np.random.choice(len(pc1), max_pts, replace=False)
    else:
        idx = np.arange(len(pc1))
        
    pc1_sub = pc1[idx]
    pc2_sub = pc2[idx]
    df_sub = df.iloc[idx].copy()
    
    df_sub['day_season'] = df_sub['day_type'].astype(str) + " - " + df_sub['season'].astype(str)
    
    color_configs = [
        ("season", "Season"),
        ("day_type", "Day Type"),
        ("ev", "EV Presence"),
        ("hp", "Heatpump Presence"),
        ("day_season", "Day-Season Combinations")
    ]
    
    for col, title in color_configs:
        if col not in df_sub.columns:
            continue
            
        plt.figure(figsize=(10, 8))
        unique_vals = sorted(df_sub[col].unique())
        
        for val in unique_vals:
            mask = (df_sub[col] == val)
            plt.scatter(pc1_sub[mask], pc2_sub[mask], label=str(val), alpha=0.5, s=15)
            
        plt.title(f"PCA Top 2 Components - Colored by {title}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(alpha=0.3)
        
        filename = f"pca_scatter_{col}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Condition mixing clustering & analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--num_batches", type=int, default=200, help="Number of batches to sample from the stream (0 = no limit)")
    parser.add_argument("--k", type=int, default=6, help="Preset number of clusters. Pass 0 to auto-determine.")
    args = parser.parse_args()
    
    cfg = _load_train_config(args.config)
    
    # 1. Setup Dataloader
    # We enforce 24h windowing for day-by-day analysis!
    window_spec = WindowSpec(input_len=24, stride=24)
    # Re-use config files but override window for this explicit evaluation
    
    data_dir = cfg.data.data_dir
    
    sm_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    sm_files = [os.path.join(data_dir, f) for f in sm_files]

    if getattr(cfg.data, "data_files", None) is not None and len(cfg.data.data_files) > 0:
        sm_files = [os.path.join(data_dir, f) for f in cfg.data.data_files if f.endswith('.parquet')]
        
    print(f"Found {len(sm_files)} parquet files to process.")
    
    splits_dict = {
        "train_ratio": 1.0,  # Use 100% of data for full dataset analysis
        "val_ratio": 0.0,
        "test_ratio": 0.0,
    }
    
    # Init dataloaders
    print("Initializing dataloader...")
    dataloaders = PhasedReplayDataloaders(
        sm_files=sm_files,
        window_spec=window_spec,
        batch_size=cfg.data.batch_size * 2, # Load faster
        min_buffer_size=0,
        num_workers=cfg.data.num_workers,
        sample_random=True,
        pin_memory=False,
        hourly_aggregation=True,
        split_ratios=splits_dict,
        phased_mode=(cfg.data.mode == "phased"),
        split_seed=int(getattr(cfg.data.splits, 'random_seed', 42)),
        feature_cols=cfg.data.columns.value_cols,
        meter_col=cfg.data.columns.meter_id,
    )

    # Use the underlying streaming dataset directly to avoid caching large tensors in RAM.
    # (get_epoch_loaders materializes a cache; this script should stay streaming-friendly.)
    from torch.utils.data import DataLoader as TorchDataLoader
    train_loader = TorchDataLoader(
        dataloaders.train_dataset,
        batch_size=cfg.data.batch_size * 2,
        num_workers=0,
        pin_memory=False,
    )
    
    # 2. Setup standalone Conditioners to extract meta-labels
    # We want explicit features out.
    conditioners_dict = {
        "day_of_week": DayOfWeekConditioner(),
        "month_of_year": MonthOfYearConditioner(),
        "ev_phase": ElectricVehicleConditioner(),
        "heatpump_phase": HeatpumpPhaseConditioner(),
        # "solar_panel": SolarPanelConditioner() # Include if topology path is valid
    }
    
    # 3. Extract real test/val data
    x_windows, metadata, meters = extract_real_data(
        train_loader, 
        conditioners_dict, 
        num_batches=args.num_batches,
    )
    
    print(f"Extracted {x_windows.shape[0]} valid 24h windows.")
    
    # 4. PCA and Clustering
    k_target = args.k if args.k > 0 else None
    cluster_labels, features_pca, pca_model, gmm_model = run_pca_and_clustering(x_windows, metadata, k=k_target)
    
    # 5. Cross Tabulation
    df, dominant_combos = find_dominant_combinations(metadata, cluster_labels)
    
    # 6. Plot the discovered archetypes
    out_dir = os.path.join(ROOT_DIR, "evaluation_comparison", "archetypes")
    plot_archetypes(x_windows, cluster_labels, df["cluster"].nunique(), out_dir)
    
    # 7. Plot PCA scatters
    pca_dir = os.path.join(ROOT_DIR, "evaluation_comparison", "pca_scatters")
    plot_pca_scatters(features_pca, df, pca_dir)
    
    print(f"\nArchetypes saved to {out_dir}")
    print(f"PCA Scatters saved to {pca_dir}")
   
if __name__ == "__main__":
    main()
