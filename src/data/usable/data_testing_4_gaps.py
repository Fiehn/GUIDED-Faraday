import pandas as pd
import pyarrow.parquet as pq
import numpy as np

def verify_gap_consistency(csv_path: str, parquet_path: str, meter_id: int):
    print(f"{'='*60}\nGAP CONSISTENCY CHECK: METER {meter_id}\n{'='*60}")

    # ==========================================
    # 1. FIND THE GAP IN PARQUET
    # ==========================================
    print(f"1. Analyzing Parquet file: {parquet_path}")
    df_p = pd.read_parquet(parquet_path)
    
    # Time formatting
    tcol = next((c for c in df_p.columns if "time" in c.lower() or "date" in c.lower()), None)
    df_p.index = pd.to_datetime(df_p[tcol] if tcol else df_p.index, errors="coerce", utc=True)
    df_p = df_p.sort_index()

    # Isolate phase columns
    phase_cols = [c for c in df_p.columns if c.startswith('active_power_p14_l')]
    if not phase_cols:
        print("ERROR: Could not find active_power_p14 columns in Parquet.")
        return

    # Find rows where all phases are NaN
    is_missing = df_p[phase_cols].isna().all(axis=1)
    
    if not is_missing.any():
        print(f"Meter {meter_id} has NO missing data in the Parquet file. Pick a different meter.")
        return

    # Group consecutive NaNs to find the largest gap
    gap_blocks = is_missing != is_missing.shift()
    block_ids = gap_blocks.cumsum()
    missing_blocks = block_ids[is_missing]
    
    # Find the ID of the largest block
    largest_block_id = missing_blocks.value_counts().idxmax()
    largest_gap_data = df_p[block_ids == largest_block_id]
    
    gap_start = largest_gap_data.index.min()
    gap_end = largest_gap_data.index.max()
    gap_duration = gap_end - gap_start
    
    print(f"  -> Largest Parquet Gap Found: {gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"  -> Gap Duration: {gap_duration}")

    # ==========================================
    # 2. CHECK THE CSV FOR THAT EXACT GAP
    # ==========================================
    print(f"\n2. Scanning CSV to see if data exists during this gap... ({csv_path})")
    
    # Find columns heuristically
    sample = pd.read_csv(csv_path, nrows=5)
    meter_col = next(c for c in sample.columns if 'meter' in c.lower() or 'id' in c.lower())
    time_col = next(c for c in sample.columns if 'time' in c.lower() or 'date' in c.lower())
    energy_col = next(c for c in sample.columns if 'energy' in c.lower() or 'kwh' in c.lower())
    
    csv_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=False):
        chunk[meter_col] = pd.to_numeric(chunk[meter_col], errors="coerce")
        match = chunk[chunk[meter_col] == meter_id]
        
        if not match.empty:
            match["timestamp"] = pd.to_datetime(match[time_col], errors="coerce", utc=True)
            # Filter the CSV rows to ONLY those that fall inside the Parquet gap
            in_gap = match[(match["timestamp"] >= gap_start) & (match["timestamp"] <= gap_end)]
            if not in_gap.empty:
                csv_chunks.append(in_gap[["timestamp", energy_col]])

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if not csv_chunks:
        print("✅ DATA IS CONSISTENT.")
        print("The CSV also contains NO DATA for this time period.")
        print("Conclusion: The meter was physically offline or disconnected.")
    else:
        df_c = pd.concat(csv_chunks, ignore_index=True)
        df_c = df_c.dropna(subset=[energy_col])
        
        print("❌ CRITICAL DATA MISMATCH FOUND.")
        print(f"The Parquet file is completely missing data, but the CSV has {len(df_c)} valid readings during this gap!")
        print(f"Average CSV consumption during Parquet's 'dead' period: {df_c[energy_col].mean():.3f} kWh/h")
        print("\nConclusion: The pipeline that generated the Parquet files failed and dropped valid data.")

if __name__ == "__main__":
    METER_ID = 100025  # We know this one has an October gap from our earlier tests
    CSV_FILE = "data/hourly_measurements_2023_10.csv"
    PARQUET_FILE = f"/media/fiehn/LaCie/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles/sm_{METER_ID}.parquet"
    
    verify_gap_consistency(CSV_FILE, PARQUET_FILE, METER_ID)