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


def build_features(x_windows, conditions_in=False, metadata=None, conditioners_dict=None):
    """
    Returns [N, 27] feature matrix (24 normalised profile + 3 stats).
    When conditions_in=True and conditioners_dict is provided, appends one-hot
    condition features giving [N, 27+C] where C = sum of condition_dim values.
    """
    # 24-dim normalised profile: each window divided by its daily sum
    x_sum = x_windows.sum(axis=1, keepdims=True) + 1e-6
    x_norm = x_windows / x_sum  # [N, 24]

    # 3 scalar stats
    total_energy = x_windows.sum(axis=1, keepdims=True)          # [N, 1]
    peak = x_windows.max(axis=1, keepdims=True)                   # [N, 1]
    peak_minus_valley = (
        x_windows.max(axis=1) - x_windows.min(axis=1)
    ).reshape(-1, 1)                                              # [N, 1]

    parts = [x_norm, total_energy, peak, peak_minus_valley]       # → [N, 27]

    if conditions_in and metadata is not None and conditioners_dict is not None:
        for key, cond_obj in conditioners_dict.items():
            if key not in metadata:
                continue
            indices = np.asarray(metadata[key], dtype=int)
            n_classes = cond_obj.condition_dim
            one_hot = np.eye(n_classes, dtype=np.float32)[indices]  # [N, n_classes]
            parts.append(one_hot)

    return np.hstack(parts)


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


def extract_real_data_fast(sm_files, conditioners_dict, cfg, max_hour_value=None):
    """Read parquet files directly — orders of magnitude faster than the streaming dataloader.

    The parquet is expected to have a MultiIndex (meter_col, timestamp_col) and one or more
    value columns.  Windows are 24 consecutive hourly rows per meter (stride=24).

    Returns (x_windows [N,24], metadata dict, meter_ids [N]).
    """
    meter_col = cfg.data.columns.meter_id          # 'meter_number'
    value_cols = cfg.data.columns.value_cols        # e.g. ['hourly_energy_consumption']
    ts_candidates = ['timestamp_dst', 'timestamp_utc', 'timestamp']

    if max_hour_value is None and hasattr(cfg.data, 'max_hour_value'):
        max_hour_value = cfg.data.max_hour_value

    all_x_parts: list[np.ndarray] = []
    all_meters_parts: list[np.ndarray] = []
    # Collect month and dow directly from DatetimeIndex slices to avoid
    # the .asi8 unit ambiguity: pandas 2.x stores tz-aware datetimes as
    # datetime64[us] (microseconds), so .asi8 gives μs, not ns.
    all_months_parts: list[np.ndarray] = []
    all_dows_parts: list[np.ndarray] = []

    for fpath in tqdm(sm_files, desc="Reading parquets (fast)"):
        df = pd.read_parquet(fpath)

        # Detect value column(s) — handle both MultiIndex and flat
        if isinstance(df.index, pd.MultiIndex):
            meter_vals_flat = df.index.get_level_values(0).to_numpy()
            ts_idx = df.index.get_level_values(1)  # DatetimeIndex (may be us resolution)
        else:
            # Flat index: meter and timestamp must be columns
            ts_col = next((c for c in ts_candidates if c in df.columns), None)
            if ts_col is None:
                raise ValueError(f"No timestamp column found in {fpath}")
            df = df.sort_values([meter_col, ts_col])
            meter_vals_flat = df[meter_col].to_numpy()
            ts_idx = pd.to_datetime(df[ts_col], utc=True, errors='coerce')

        # Ensure timestamps are tz-aware UTC
        if ts_idx.tz is None:
            ts_idx = ts_idx.tz_localize('UTC')

        # Sum value columns to get single consumption series
        if len(value_cols) == 1 and value_cols[0] in df.columns:
            vals_flat = df[value_cols[0]].to_numpy(dtype=np.float32, na_value=0.0)
        else:
            present = [c for c in value_cols if c in df.columns]
            vals_flat = df[present].to_numpy(dtype=np.float32).sum(axis=1)

        # Find per-meter slices (data is sorted by meter_number in MultiIndex files)
        unique_meters, meter_starts = np.unique(meter_vals_flat, return_index=True)
        meter_ends = np.append(meter_starts[1:], len(vals_flat))

        for meter_id, start, end in zip(unique_meters, meter_starts, meter_ends):
            n_rows = end - start
            n_days = n_rows // 24
            if n_days == 0:
                continue
            T = n_days * 24

            vals_w = vals_flat[start: start + T].reshape(n_days, 24)  # [n_days, 24]
            ts_w = ts_idx[start: start + T: 24]                       # [n_days] DatetimeIndex

            if max_hour_value is not None:
                mask = vals_w.max(axis=1) <= float(max_hour_value)
                vals_w = vals_w[mask]
                ts_w = ts_w[mask]

            if len(vals_w) == 0:
                continue

            all_x_parts.append(vals_w)
            all_meters_parts.append(np.full(len(vals_w), meter_id, dtype=np.int64))
            # Extract directly from DatetimeIndex — no unit ambiguity
            all_months_parts.append((ts_w.month - 1).to_numpy(dtype=np.int64))
            all_dows_parts.append(ts_w.dayofweek.to_numpy(dtype=np.int64))

    all_x = np.vstack(all_x_parts)                     # [N, 24]
    all_meters = np.concatenate(all_meters_parts)       # [N]

    print(f"Extracted {len(all_x):,} windows from {len(np.unique(all_meters)):,} meters.")

    metadata: dict[str, np.ndarray] = {
        "day_of_week":   np.concatenate(all_dows_parts),       # 0=Mon
        "month_of_year": np.concatenate(all_months_parts),     # 0-based (0=Jan)
    }

    # EV / HP topology: call each conditioner once with all unique meters
    unique_m = np.unique(all_meters)
    for name, cond_obj in conditioners_dict.items():
        if name in ('day_of_week', 'month_of_year'):
            continue
        meter_t = torch.from_numpy(unique_m.astype(np.int64))
        cond = cond_obj.get_conditions(None, meter_t, None)
        if cond is None:
            continue
        if cond.shape[1] > 1:
            labels_u = cond.argmax(dim=1).cpu().numpy()
        else:
            labels_u = cond.squeeze().cpu().numpy().astype(np.int64)

        # Expand unique labels → per-window labels via searchsorted
        idx = np.searchsorted(unique_m, all_meters)
        metadata[name] = labels_u[idx].astype(np.int64)

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

def run_pca_and_clustering(x_windows, metadata, k=None, conditions_in=False, conditioners_dict=None):
    print("Building features...")
    features = build_features(x_windows, conditions_in=conditions_in, metadata=metadata, conditioners_dict=conditioners_dict)
    
    print("Standardizing features...")
    scaler = StandardScaler()
    features = features.reshape(features.shape[0], -1) # Flatten if needed
    scaled_features = scaler.fit_transform(features)
    
    print("Running PCA...")
    pca = PCA(n_components=5, random_state=42)
    features_pca = pca.fit_transform(scaled_features)
    
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"PCA top 2 components explain {explained_variance[1]*100:.1f}% of variance.")
    print(f"PCA top 3 components explain {explained_variance[2]*100:.1f}% of variance.")
    print(f"PCA top 4 components explain {explained_variance[3]*100:.1f}% of variance.")
    print(f"PCA top 5 components explain {explained_variance[-1]*100:.1f}% of variance.")
    
    if k is None:
        k = determine_optimal_clusters(features_pca)
        
    print(f"Running GMM clustering with {k} clusters...")
    # GMM often works slightly better for overlapping behavioral archetypes
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full', reg_covar=1e-4)
    cluster_labels = gmm.fit_predict(features_pca)
    
    return cluster_labels, features_pca, pca, gmm

def find_dominant_combinations(metadata, cluster_labels, min_support_ratio=0.05, save_dir=None):
    """
    Cross-tabulate the metadata to find the dominant combo per cluster.
    If save_dir is provided, saves the full cross-tabulation as cross_tabulation.csv.
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

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        condition_cols_present = [c for c in ["season", "day_type", "ev", "hp"] if c in df.columns]
        if condition_cols_present:
            all_combos = []
            for c_id in range(df["cluster"].nunique()):
                c_df = df[df["cluster"] == c_id]
                total_in_cluster = len(c_df)
                combo_counts = c_df.groupby(condition_cols_present).size().reset_index(name="count")
                combo_counts["percentage"] = combo_counts["count"] / total_in_cluster * 100
                combo_counts["cluster"] = c_id
                all_combos.append(combo_counts)
            cross_tab = pd.concat(all_combos, ignore_index=True)
            cross_tab = cross_tab[["cluster"] + condition_cols_present + ["count", "percentage"]]
            cross_tab.to_csv(os.path.join(save_dir, "cross_tabulation.csv"), index=False)

    return df, dominant_combos

def _archetype_label(cluster_id: int, n_windows: int, combo: dict | None) -> str:
    """Return a curve label for an archetype cluster."""
    if combo is None:
        return f"Cluster {cluster_id} (N={n_windows})"
    parts = " / ".join(str(v) for v in combo.values())
    return f"Cluster {cluster_id} — {parts}"


def plot_archetypes(x_windows, cluster_labels, k, output_dir, dominant_combos=None):
    """Plot the mean 24h shape of each cluster to see what we found."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    for c_id in range(k):
        c_idx = (cluster_labels == c_id)
        if not np.any(c_idx):
            continue

        c_data = x_windows[c_idx]
        mean_profile = c_data.mean(axis=0)

        combo = dominant_combos.get(c_id) if dominant_combos is not None else None
        label = _archetype_label(c_id, int(np.sum(c_idx)), combo)
        plt.plot(mean_profile, label=label, linewidth=2)

    plt.title("Mean 24h Consumption Profiles by Cluster Archetype")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Consumption (Sum of phases)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(output_dir, "cluster_archetypes.png"))
    plt.close()

def plot_pca_scatters(features_pca, df, output_dir):
    """Plot the top 3 PCA components colored by various metadata labels."""
    os.makedirs(output_dir, exist_ok=True)
    
    # We will use the top 3 components so the third dimension is visible too.
    pc1 = features_pca[:, 0]
    pc2 = features_pca[:, 1]
    pc3 = features_pca[:, 2]
    
    # Subsample if too large to avoid massive overplotting
    max_pts = 10000
    if len(pc1) > max_pts:
        np.random.seed(42)
        idx = np.random.choice(len(pc1), max_pts, replace=False)
    else:
        idx = np.arange(len(pc1))
        
    pc1_sub = pc1[idx]
    pc2_sub = pc2[idx]
    pc3_sub = pc3[idx]
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

        unique_vals = sorted(df_sub[col].unique())
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        for val in unique_vals:
            mask = (df_sub[col] == val)
            ax.scatter(pc1_sub[mask], pc2_sub[mask], pc3_sub[mask], label=str(val), alpha=0.5, s=15)
            
        # Place the 3D axes explicitly so the cube sits in the figure instead of hugging a corner.
        ax.set_position([0.06, 0.08, 0.72, 0.80])
        ax.set_box_aspect((1, 1, 0.85))

        # Use a figure-level suptitle with a small top margin.
        fig.suptitle(f"PCA Top 3 Components - Colored by {title}", y=0.96)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.view_init(elev=20, azim=35)
        ax.legend(loc='upper right', frameon=True)
        # Leave a little room around the whole plot, but do not shove the axes away.
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
        ax.grid(alpha=0.3)
        
        filename = f"pca_scatter_{col}.png"
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)

def plot_pca_loadings(pca, feature_names, output_path):
    """Bar chart of the top 3 PC loadings per input feature."""
    n_components = min(3, pca.n_components_)
    components = pca.components_[:n_components]  # [n_components, n_features]

    x = np.arange(len(feature_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.4), 5))

    for i, comp in enumerate(components):
        ax.bar(x + i * width, comp, width, label=f"PC{i+1}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_title("PCA Loadings — Top 3 Components")
    ax.set_ylabel("Loading")
    ax.legend()
    fig.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# UserProfiler analysis
# ---------------------------------------------------------------------------

def reconstruct_topic_curves(profiler) -> np.ndarray:
    """Returns [n_topics, 24] topic profiles as weighted averages of cluster centers."""
    lam = np.asarray(profiler.lambda_matrix)          # [n_topics, num_clusters]
    centers = np.asarray(profiler.cluster_centers_)   # [num_clusters, 24]
    weights = lam / lam.sum(axis=1, keepdims=True)    # [n_topics, num_clusters]
    return weights @ centers                           # [n_topics, 24]


def plot_topic_curves(topic_curves: np.ndarray, save_path: str) -> None:
    """Saves an 8×5 grid of 24h topic curve subplots."""
    nrows, ncols = 8, 5
    n_topics = topic_curves.shape[0]
    hours = np.arange(24)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2))
    axes_flat = axes.flatten()
    for k in range(n_topics):
        ax = axes_flat[k]
        ax.plot(hours, topic_curves[k])
        ax.set_title(f"Topic {k}", fontsize=8)
        ax.set_xticks([])
    for k in range(n_topics, len(axes_flat)):
        axes_flat[k].set_visible(False)
    fig.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def cluster_meters_by_gamma(profiler, k: int) -> dict:
    """Returns {meter_id -> cluster_id} for all meters in profiler.gamma_cache."""
    meter_ids = list(profiler.gamma_cache.keys())
    gamma_matrix = np.stack([profiler.gamma_cache[m] for m in meter_ids])
    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(gamma_matrix)
    return {m: int(labels[i]) for i, m in enumerate(meter_ids)}


def plot_meter_cluster_silhouette(profiler, k_range, save_path: str) -> int:
    """Plots silhouette scores for k in k_range, saves figure, returns best k."""
    meter_ids = list(profiler.gamma_cache.keys())
    gamma_matrix = np.stack([profiler.gamma_cache[m] for m in meter_ids])
    k_list = list(k_range)
    scores = []
    for k in k_list:
        labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(gamma_matrix)
        scores.append(silhouette_score(gamma_matrix, labels))
    best_k = k_list[int(np.argmax(scores))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_list, scores, marker="o")
    ax.axvline(best_k, linestyle="--", color="red", label=f"Best K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Meter cluster silhouette scores")
    ax.legend()
    fig.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return best_k


def cross_tabulate_meter_clusters(
    meter_cluster_map: dict,
    meter_ids: np.ndarray,
    metadata: dict,
    top_n: int = 3,
) -> dict:
    """Returns {cluster_id -> [(combo_dict, pct), ...]} with the top_n condition combinations."""
    meter_ids = np.asarray(meter_ids)
    n = len(meter_ids)
    window_clusters = np.array([meter_cluster_map.get(int(m), -1) for m in meter_ids])

    df = pd.DataFrame({
        "meter_cluster": window_clusters,
        "season": [season_from_month(metadata["month_of_year"][i]) for i in range(n)],
        "day_type": [day_type_from_day(metadata["day_of_week"][i]) for i in range(n)],
        "ev": [ev_from_idx(metadata["ev_phase"][i]) for i in range(n)],
        "hp": [hp_from_idx(metadata["heatpump_phase"][i]) for i in range(n)],
    })
    df = df[df["meter_cluster"] >= 0]

    result = {}
    for cluster_id, group in df.groupby("meter_cluster"):
        counts = group.groupby(["season", "day_type", "ev", "hp"]).size()
        total = len(group)
        top = counts.nlargest(top_n)
        result[int(cluster_id)] = [
            ({"season": s, "day_type": d, "ev": e, "hp": h}, cnt / total * 100)
            for (s, d, e, h), cnt in top.items()
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description="Condition mixing clustering & analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--k", type=int, default=6, help="Preset number of clusters. Pass 0 to auto-determine.")
    parser.add_argument("--no_scatter", action="store_true", help="Skip PCA scatter plots (saves time and disk).")
    parser.add_argument("--profiler_path", type=str, default=None, help="Path to a fitted UserProfiler pickle. Skipped if not provided.")
    parser.add_argument("--profiler_k", type=int, default=0, help="Meter cluster count for UserProfiler analysis. 0 = auto via silhouette K=3..12.")
    parser.add_argument("--mode", type=str, choices=["conditions_out", "conditions_in", "both"], default="both", help="Which mode(s) to run: conditions_out, conditions_in, or both")
    args = parser.parse_args()

    cfg = _load_train_config(args.config)

    data_dir = cfg.data.data_dir
    sm_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if getattr(cfg.data, "data_files", None):
        sm_files = [os.path.join(data_dir, f) for f in cfg.data.data_files if f.endswith('.parquet')]
    print(f"Found {len(sm_files)} parquet file(s).")

    conditioners_dict = {
        "day_of_week":    DayOfWeekConditioner(),
        "month_of_year":  MonthOfYearConditioner(),
        "ev_phase":       ElectricVehicleConditioner(),
        "heatpump_phase": HeatpumpPhaseConditioner(),
    }

    # Fast direct parquet extraction — bypasses the streaming dataloader
    x_windows, metadata, meters = extract_real_data_fast(sm_files, conditioners_dict, cfg)

    k_target = args.k if args.k > 0 else None
    base_dir = os.path.join(ROOT_DIR, "evaluation_comparison")

    # Build base feature names (same for both modes)
    shape_feature_names = [f"h{i:02d}" for i in range(24)] + ["total_energy", "peak", "peak_minus_valley"]

    for mode_label, do_conditions_in in [("conditions_out", False), ("conditions_in", True)]:
        # Respect --mode selection to run only one of the modes if requested
        if args.mode != "both" and args.mode != mode_label:
            print(f"Skipping mode {mode_label} (not selected).")
            continue
        print(f"\n{'='*60}")
        print(f"MODE: {mode_label}")
        print(f"{'='*60}")

        cluster_labels, features_pca, pca_model, _ = run_pca_and_clustering(
            x_windows, metadata, k=k_target,
            conditions_in=do_conditions_in,
            conditioners_dict=conditioners_dict,
        )

        # Cross tabulation
        xtab_dir = os.path.join(base_dir, "cross_tabulation", mode_label)
        df, dominant_combos = find_dominant_combinations(
            metadata, cluster_labels, save_dir=xtab_dir
        )

        # Archetypes
        archetype_dir = os.path.join(base_dir, "archetypes", mode_label)
        plot_archetypes(
            x_windows, cluster_labels, df["cluster"].nunique(), archetype_dir,
            dominant_combos=dominant_combos,
        )

        # PCA scatters (skip with --no_scatter to save time)
        if not args.no_scatter:
            scatter_dir = os.path.join(base_dir, "pca_scatters", mode_label)
            plot_pca_scatters(features_pca, df, scatter_dir)
            print(f"PCA scatters→ {scatter_dir}")

        # PCA loadings
        if do_conditions_in:
            cond_names = []
            for key, cond_obj in conditioners_dict.items():
                if key in metadata:
                    cond_names += [f"{key}_{j}" for j in range(cond_obj.condition_dim)]
            feature_names = shape_feature_names + cond_names
        else:
            feature_names = shape_feature_names

        loadings_path = os.path.join(base_dir, "pca_loadings", f"loadings_{mode_label}.png")
        plot_pca_loadings(pca_model, feature_names, loadings_path)

        print(f"Archetypes  → {archetype_dir}")
        print(f"Loadings    → {loadings_path}")

        expl = np.sum(pca_model.explained_variance_ratio_) * 100
        print(f"Explained variance (top {pca_model.n_components_} PCs): {expl:.1f}%")
    
    # 4. UserProfiler-specific analysis (if we have a profiler checkpoint)
    if args.profiler_path is not None:
        print(f"\n{'='*60}")
        print("UserProfiler analysis")
        print(f"{'='*60}")
        from src.data.conditioner import UserProfiler
        profiler = UserProfiler(path=None)  # type: ignore[arg-type]
        profiler.load(args.profiler_path)
        print(f"Loaded profiler: {profiler.n_topics} topics, {len(profiler.gamma_cache)} metered users")

        profiler_dir = os.path.join(base_dir, "user_profiler")

        # 4a. Topic curve grid
        topic_curves = reconstruct_topic_curves(profiler)
        topic_curves_path = os.path.join(profiler_dir, "topic_curves.png")
        plot_topic_curves(topic_curves, topic_curves_path)
        print(f"Topic curves → {topic_curves_path}")

        # 4b. Meter clustering: silhouette scan or fixed k
        sil_path = os.path.join(profiler_dir, "meter_silhouette.png")
        if args.profiler_k > 0:
            best_k = args.profiler_k
            print(f"Using fixed meter cluster count k={best_k}")
        else:
            best_k = plot_meter_cluster_silhouette(profiler, range(3, 13), sil_path)
            print(f"Silhouette best k={best_k} → {sil_path}")

        meter_cluster_map = cluster_meters_by_gamma(profiler, best_k)
        cluster_counts = {}
        for c in meter_cluster_map.values():
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        print(f"Meter clusters: { {k: cluster_counts[k] for k in sorted(cluster_counts)} }")

        # 4c. Window-level cross-tabulation against extracted conditions
        xtab = cross_tabulate_meter_clusters(meter_cluster_map, meters, metadata)
        print("\nTop-3 condition combinations per meter cluster:")
        for cluster_id in sorted(xtab):
            print(f"  Cluster {cluster_id}:")
            for combo, pct in xtab[cluster_id]:
                parts = [f"{v}" for v in combo.values()]
                print(f"    {' / '.join(parts):45s}  {pct:5.1f}%")

        xtab_path = os.path.join(profiler_dir, "meter_cluster_xtab.csv")
        rows = []
        for cluster_id, combos in xtab.items():
            for combo, pct in combos:
                rows.append({"meter_cluster": cluster_id, **combo, "percentage": round(pct, 2)})
        pd.DataFrame(rows).to_csv(xtab_path, index=False)
        print(f"Cross-tabulation → {xtab_path}")
    else:
        print("\nNo --profiler_path given; skipping UserProfiler analysis.")


if __name__ == "__main__":
    main()
