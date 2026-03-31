import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pyarrow.parquet as pq

def investigate_single_meter(csv_path: str, parquet_path: str, meter_id: int, out_path: str):
    print(f"{'='*50}\nINVESTIGATING METER {meter_id}\n{'='*50}")

    # ==========================================
    # 1. LOAD PARQUET (The "Phased" Data)
    # ==========================================
    print(f"Loading Parquet: {parquet_path}")
    df_p = pd.read_parquet(parquet_path)
    
    # Time formatting
    tcol = next((c for c in df_p.columns if "time" in c.lower() or "date" in c.lower()), None)
    if tcol:
        df_p.index = pd.to_datetime(df_p[tcol], errors="coerce", utc=True)
    else:
        df_p.index = pd.to_datetime(df_p.index, errors="coerce", utc=True)
    
    # Identify phase columns and sum them
    phase_cols = [c for c in df_p.columns if 'active_power' in c.lower() or 'l1' in c.lower() or 'phase' in c.lower()]
    print(f"  -> Detected Phase Columns: {phase_cols}")
    
    phase_cols = [c for c in df_p.columns if c.startswith('active_power_p14_l')]
    
    phase_cols = [c for c in df_p.columns if c.startswith('active_power_p14_l')]
    
    if len(phase_cols) >= 1:
        df_phases = df_p[phase_cols].apply(pd.to_numeric, errors="coerce")
        # Attach the Aggregated column directly back to the main df_p dataframe!
        df_p["Aggregated"] = df_phases.sum(axis=1, min_count=1)
    else:
        df_p["Aggregated"] = np.nan # Failsafe if the columns don't exist
    
    # Drop NaNs to see what is ACTUALLY there
    df_p_valid = df_p.dropna(subset=["Aggregated"])
    print(f"  -> Raw Parquet Rows:   {len(df_p)}")
    print(f"  -> Valid Parquet Rows: {len(df_p_valid)} (Non-NaN)")
    if not df_p_valid.empty:
        print(f"  -> Parquet Date Range: {df_p_valid.index.min().date()} to {df_p_valid.index.max().date()}")

    # ==========================================
    # 2. LOAD CSV (The Baseline)
    # ==========================================
    print(f"\nScanning CSV for Meter {meter_id}... (This may take a minute)")
    csv_chunks = []
    
    # Find columns heuristically from first chunk
    sample = pd.read_csv(csv_path, nrows=5)
    meter_col = next(c for c in sample.columns if 'meter' in c.lower() or 'id' in c.lower())
    time_col = next(c for c in sample.columns if 'time' in c.lower() or 'date' in c.lower())
    energy_col = next(c for c in sample.columns if 'energy' in c.lower() or 'kwh' in c.lower())
    
    for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=False):
        chunk[meter_col] = pd.to_numeric(chunk[meter_col], errors="coerce")
        match = chunk[chunk[meter_col] == meter_id]
        if not match.empty:
            csv_chunks.append(match[[time_col, energy_col]])

    if not csv_chunks:
        print(f"\nERROR: Meter {meter_id} not found in CSV!")
        return

    df_c = pd.concat(csv_chunks, ignore_index=True)
    df_c.index = pd.to_datetime(df_c[time_col], errors="coerce", utc=True)
    df_c = df_c.dropna(subset=[energy_col]).sort_index()
    
    print(f"  -> Valid CSV Rows:     {len(df_c)}")
    print(f"  -> CSV Date Range:     {df_c.index.min().date()} to {df_c.index.max().date()}")

    # ==========================================
    # 3. ALIGN AND PLOT
    # ==========================================
    if df_p_valid.empty:
        print("\nERROR: Parquet data has 0 valid rows. The columns are entirely NaN or empty. Cannot plot.")
        return

    # Find the macro overlap
    start_date = max(df_p_valid.index.min(), df_c.index.min())
    end_date = min(df_p_valid.index.max(), df_c.index.max())
    
    if start_date >= end_date:
        print(f"\nERROR: Time ranges do not overlap!\nParquet: {df_p_valid.index.min().date()}->{df_p_valid.index.max().date()}\nCSV: {df_c.index.min().date()}->{df_c.index.max().date()}")
        return
        
    print(f"\nPlotting overlapping window: {start_date.date()} to {end_date.date()}")
    
    # Filter to overlap
    p_plot = df_p_valid[(df_p_valid.index >= start_date) & (df_p_valid.index <= end_date)]
    c_plot = df_c[(df_c.index >= start_date) & (df_c.index <= end_date)]
    
    # Resample Parquet to Hourly (using Mean to convert Power to Energy)
    p_hourly = p_plot.resample('1h').mean().dropna(subset=["Aggregated"])
    
    # Force Parquet to divide by 1000 (assuming it is in Wh or W)
    p_vals = p_hourly["Aggregated"].values / 1000.0 
    
    # Assume CSV is natively in kWh
    c_vals = c_plot[energy_col].values 
    
    print(f"  -> Max Parquet value (scaled): {p_vals.max():.4f} kWh")
    print(f"  -> Max CSV value: {c_vals.max():.4f} kWh")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(p_hourly.index, p_vals, label="Parquet 3-Phase (Hourly Avg)", linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.plot(c_plot.index, c_vals, label="CSV Hourly", linestyle='--', linewidth=2, marker='x', markersize=4, alpha=0.8)

    ax.set_xlim([start_date, end_date])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.title(f"Meter {meter_id} - Full Overlap Range")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    # YOU ONLY NEED TO CHANGE THESE THREE VARIABLES
    METER_ID = 393221
    CSV_FILE = "data/hourly_measurements_2023_11.csv"
    PARQUET_FILE = f"F:/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles/sm_{METER_ID}.parquet"
    
    investigate_single_meter(CSV_FILE, PARQUET_FILE, METER_ID, f"output/debug_meter_{METER_ID}.png")