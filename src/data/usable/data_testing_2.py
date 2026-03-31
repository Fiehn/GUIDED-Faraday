import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def debug_dataloader_pipeline(file_path: str):
    print(f"\n{'='*50}")
    print(f"DEBUGGING FILE: {os.path.basename(file_path)}")
    print(f"{'='*50}")

    feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
    
    # --- 1. RAW READ ---
    parquet_file = pq.ParquetFile(file_path)
    file_cols = parquet_file.schema.names
    
    cols_to_load = [c for c in file_cols if any(c.startswith(f) for f in feature_cols)]
    time_col = next((c for c in file_cols if c.lower() in ['timestamp_dst', 'timestamp', 'timestamp_utc']), None)
    if time_col: cols_to_load.append(time_col)
        
    df = pq.read_table(file_path, columns=cols_to_load).to_pandas()
    
    # Standardize column names for ease of debugging
    for col in df.columns:
        for base_feat in feature_cols:
            if col.startswith(base_feat):
                df = df.rename(columns={col: base_feat})
                break

    time_values = df[time_col] if time_col in df.columns else df.index
    df.index = pd.to_datetime(time_values, errors='coerce', utc=True)
    df = df[~df.index.isna() & ~df.index.duplicated(keep='last')].sort_index()

    print("\n--- STAGE 1: RAW DATA (From Parquet) ---")
    print(f"Total Rows: {len(df)}")
    raw_zeros = (df[feature_cols] == 0.0).all(axis=1).sum()
    raw_nans = df[feature_cols].isna().any(axis=1).sum()
    print(f"Rows with actual 0.0 on all phases: {raw_zeros}")
    print(f"Rows with NaN values:               {raw_nans}")

    # --- 2. RESAMPLING (Hourly Aggregation) ---
    df_resampled = df.resample('1h').sum(min_count=1)
    
    print("\n--- STAGE 2: AFTER RESAMPLING ('1h') ---")
    print(f"Total Rows: {len(df_resampled)}")
    res_zeros = (df_resampled[feature_cols] == 0.0).all(axis=1).sum()
    res_nans = df_resampled[feature_cols].isna().any(axis=1).sum()
    print(f"Rows with actual 0.0 on all phases: {res_zeros}")
    print(f"Rows with NaN values:               {res_nans}")

    # --- 3. REINDEXING (Forcing Continuous Timeline) ---
    start = df_resampled.index.min()
    end = df_resampled.index.max()
    full_index = pd.date_range(start=start, end=end, freq='1h', tz=df_resampled.index.tz)
    df_reindexed = df_resampled.reindex(full_index)

    print("\n--- STAGE 3: AFTER REINDEXING (Padding Gaps) ---")
    print(f"Total Rows: {len(df_reindexed)}")
    re_zeros = (df_reindexed[feature_cols] == 0.0).all(axis=1).sum()
    re_nans = df_reindexed[feature_cols].isna().any(axis=1).sum()
    print(f"Rows with actual 0.0 on all phases: {re_zeros}")
    print(f"Rows with NaN values:               {re_nans}  <-- Look at this number jump!")

    # --- 4. THE FILLNA BUG ---
    df_filled = df_reindexed.fillna(0.0)
    
    print("\n--- STAGE 4: AFTER FILLNA(0.0) ---")
    print(f"Total Rows: {len(df_filled)}")
    fill_zeros = (df_filled[feature_cols] == 0.0).all(axis=1).sum()
    print(f"Rows with 0.0 on all phases:        {fill_zeros}  <-- These become your dead windows")
    
    # Calculate the exact inflation ratio
    inflation = fill_zeros - raw_zeros
    print(f"\nCONCLUSION: fillna(0.0) artificially created {inflation} zero-hours.")
    
    # Show a visual slice of the crime scene
    if re_nans > 0:
        print("\n--- VISUAL SLICE OF IMPUTATION ---")
        # Find the first NaN row index
        first_nan_idx = np.where(df_reindexed[feature_cols].isna().any(axis=1))[0][0]
        # Show 2 hours before and 2 hours after
        start_idx = max(0, first_nan_idx - 2)
        end_idx = min(len(df_reindexed), first_nan_idx + 3)
        print("Before FillNA (Shows actual missing data):")
        print(df_reindexed.iloc[start_idx:end_idx][feature_cols])
        print("\nAfter FillNA (What your model sees):")
        print(df_filled.iloc[start_idx:end_idx][feature_cols])

if __name__ == "__main__":
    import glob
    # Grab just the first file in your directory
    data_dir = "F:/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles"
    test_file = glob.glob(os.path.join(data_dir, "*.parquet"))[0]
    debug_dataloader_pipeline(test_file)