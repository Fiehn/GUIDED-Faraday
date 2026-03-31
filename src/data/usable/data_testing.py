import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- WORKER FUNCTION (Runs on multiple cores simultaneously) ---
def process_single_file(file_path: str, window_len: int = 24, stride: int = 1, 
                        variance_epsilon: float = 1e-5, zero_epsilon: float = 1e-5, 
                        hourly_aggregation: bool = True, aggregation_divisor: float = 1.0):
    try:
        feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        time_col_candidates = ['timestamp_dst', 'timestamp', 'timestamp_utc']
        
        # 1. Fast Parquet Read
        parquet_file = pq.ParquetFile(file_path)
        file_cols = parquet_file.schema.names
        
        cols_to_load = []
        rename_map = {}
        for base_feat in feature_cols:
            matched_col = next((c for c in file_cols if c.startswith(base_feat)), None)
            if matched_col:
                cols_to_load.append(matched_col)
                rename_map[matched_col] = base_feat
        
        time_col_name = next((c for c in file_cols if c.lower() in time_col_candidates), None)
        if time_col_name and time_col_name not in cols_to_load:
            cols_to_load.append(time_col_name)
            
        if not cols_to_load:
            return 0, 0, 0, 0, 0
            
        df = pq.read_table(file_path, columns=cols_to_load).to_pandas()
        df = df.rename(columns=rename_map)
        
        # 2. Time Index Prep
        time_values = df[time_col_name] if time_col_name in df.columns else df.index
        df.index = pd.to_datetime(time_values, errors='coerce', utc=True)
        df = df[~df.index.isna() & ~df.index.duplicated(keep='last')].sort_index()
        
        if df.empty:
            return 0, 0, 0, 0, 0

        # Calculate missing days natively
        expected_days = (df.index.max() - df.index.min()).days + 1
        actual_days = len(np.unique(df.index.normalize()))
        missing_days = max(0, expected_days - actual_days)
        
        # 3. Imputation (Your exact logic)
        base_frequency = '1h' if hourly_aggregation else '15min'
        if hourly_aggregation:
            df = df.resample(base_frequency).sum(min_count=1)
            if aggregation_divisor != 1.0:
                df = df / aggregation_divisor
        
        if df.empty:
            return 0, 0, 0, 0, missing_days
            
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=base_frequency, tz=df.index.tz)
        df = df.reindex(full_index)
        
        for feat in feature_cols:
            if feat not in df.columns:
                df[feat] = np.nan
        
        values = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        
        T = values.shape[0]
        if T < window_len:
            return 0, 0, 0, 0, missing_days
            
        # 4. INSTANT VECTORIZED SLIDING WINDOWS (No Python Loops!)
        # Creates a view of shape (T - window_len + 1, 3, window_len) instantly
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_len, axis=0)
        windows = np.transpose(windows, (0, 2, 1)) # Reshape to (Num_Windows, window_len, 3)
        
        if stride > 1:
            windows = windows[::stride]
            
        total_windows = windows.shape[0]
        if total_windows == 0:
            return 0, 0, 0, 0, missing_days
            
        # 5. INSTANT STATS CALCULATION (Using fast C-backend NumPy)
        seq_agg = windows.sum(axis=-1) # Shape: [Num_Windows, 24]
        
        window_sum = np.abs(seq_agg).sum(axis=-1) 
        is_zero = window_sum <= zero_epsilon
        
        window_var = np.var(seq_agg, axis=-1)
        is_flat = (window_var <= variance_epsilon) & (~is_zero)
        
        phase_sums = np.abs(windows).sum(axis=1) # Shape: [Num_Windows, 3]
        active_phases = (phase_sums > zero_epsilon).sum(axis=-1)
        is_partial = (active_phases > 0) & (active_phases < 3)
        
        return total_windows, int(is_zero.sum()), int(is_flat.sum()), int(is_partial.sum()), missing_days
        
    except Exception:
        # If a single file is completely corrupted, skip it so the whole run doesn't crash
        return 0, 0, 0, 0, 0

# --- MAIN ORCHESTRATOR ---
def run_fast_diagnostics(data_dir: str, num_workers: int = None):
    sm_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not sm_files:
        print(f"No parquet files found in {data_dir}.")
        return

    # Use all available CPU cores if not specified
    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 1))

    print(f"Launching FAST diagnostics across {len(sm_files)} files using {num_workers} CPU cores...")

    t_windows = t_zeros = t_flats = t_partials = t_missing = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the worker function to all files
        futures = {executor.submit(process_single_file, f): f for f in sm_files}
        
        with tqdm(total=len(sm_files), desc="Scanning", unit="file") as pbar:
            for future in as_completed(futures):
                w, z, f, p, m = future.result()
                t_windows += w
                t_zeros += z
                t_flats += f
                t_partials += p
                t_missing += m
                
                # Update progress bar stats natively
                pbar.set_postfix({
                    'Win': f"{t_windows/1000000:.1f}M", 
                    'Zero': t_zeros, 
                    'Flat': t_flats, 
                    'Part': t_partials
                })
                pbar.update(1)

    # Calculate final ratios
    if t_windows == 0:
        print("\nError: No valid data found.")
        return

    z_pct = (t_zeros / t_windows) * 100
    f_pct = (t_flats / t_windows) * 100
    p_pct = (t_partials / t_windows) * 100
    h_pct = 100.0 - z_pct - f_pct

    print("\n" + "="*70)
    print("PHASED PARQUET INTEGRITY REPORT (VECTORIZED)")
    print("="*70)
    print(f"Total Windows Scanned: {t_windows:,}")
    print(f"Total Missing Days:    {t_missing:,} (Meter completely offline)")
    print("-" * 70)
    print(f"Zero-Sum Windows:      {t_zeros:,} ({z_pct:.2f}%) - Fully disconnected")
    print(f"Flatline Windows:      {t_flats:,} ({f_pct:.2f}%) - Imputation artifacts")
    print(f"Partial Dead Phases:   {t_partials:,} ({p_pct:.2f}%) - 1 or 2 phases are 0.0")
    print("-" * 70)
    print(f"Fully Active Windows:  {t_windows - t_zeros - t_flats - t_partials:,} ({h_pct - p_pct:.2f}%)")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Point this to your parquet folder
    run_fast_diagnostics(data_dir="F:/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles")