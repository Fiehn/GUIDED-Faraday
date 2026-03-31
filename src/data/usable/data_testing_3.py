import pandas as pd
import pyarrow.parquet as pq

def analyze_nan_blocks(file_path: str):
    feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
    
    # Read raw parquet
    df = pq.read_table(file_path).to_pandas()
    
    # Standardize names
    for col in df.columns:
        for base_feat in feature_cols:
            if col.startswith(base_feat):
                df.rename(columns={col: base_feat}, inplace=True)
                break
                
    time_col = next((c for c in df.columns if c.lower() in ['timestamp_dst', 'timestamp', 'timestamp_utc']), df.index.name)
    df.index = pd.to_datetime(df[time_col] if time_col in df.columns else df.index, utc=True)
    df = df.sort_index()

    # Create a boolean mask of rows where ANY phase is NaN
    is_nan = df[feature_cols].isna().any(axis=1)
    
    # This clever pandas trick groups consecutive True values together
    # It assigns a unique ID to each continuous block of NaNs
    nan_blocks = is_nan != is_nan.shift()
    block_ids = nan_blocks.cumsum()
    
    # Filter to only look at the blocks that are actually NaNs
    nan_only_blocks = block_ids[is_nan]
    
    if nan_only_blocks.empty:
        print("No NaNs found in this file.")
        return

    # Count the size of each continuous NaN block
    block_sizes = nan_only_blocks.value_counts()
    
    print(f"\nNAN GAP ANALYSIS FOR: {file_path.split('/')[-1]}")
    print("="*50)
    print(f"Total NaN rows: {len(nan_only_blocks)}")
    print(f"Number of distinct missing periods: {len(block_sizes)}\n")
    
    print("Distribution of Gap Sizes (in 15-minute raw ticks):")
    
    # Categorize the gaps
    tiny_gaps = block_sizes[block_sizes <= 4].count()      # <= 1 hour
    medium_gaps = block_sizes[(block_sizes > 4) & (block_sizes <= 96)].count() # 1 hour to 1 day
    large_gaps = block_sizes[block_sizes > 96].count()     # > 1 day
    
    print(f"  Tiny gaps (<= 1 hour):     {tiny_gaps} occurrences")
    print(f"  Medium gaps (1h to 1 day): {medium_gaps} occurrences")
    print(f"  Massive gaps (> 1 day):    {large_gaps} occurrences\n")
    
    if large_gaps > 0:
        print("Details of the Massive Gaps (> 1 day):")
        # Get the actual start and end dates for the massive gaps
        for block_id in block_sizes[block_sizes > 96].index:
            block_data = df[block_ids == block_id]
            start_time = block_data.index.min()
            end_time = block_data.index.max()
            duration = end_time - start_time
            print(f"  -> Offline from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} (Duration: {duration})")

if __name__ == "__main__":
    # Point this at sm_100025.parquet
    analyze_nan_blocks("F:/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles/sm_100025.parquet")