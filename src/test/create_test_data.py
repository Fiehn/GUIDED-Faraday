"""
Create a small test dataset for fast validation.
Extracts 2 days of data for 10 meters.
"""
import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def create_test_dataset(
    input_file: str = "data/hourly_measurements_2023_10.csv",
    output_file: str = "data/test_data_small.csv",
    num_meters: int = 10,
    num_days: int = 2,
):
    """
    Create a small test dataset from the full data.
    
    Args:
        input_file: Path to full dataset
        output_file: Path to save test dataset
        num_meters: Number of meters to include
        num_days: Number of days to include per meter
    """
    print(f"Creating test dataset: {num_meters} meters × {num_days} days")
    print(f"Reading from: {input_file}")
    
    # Read full data
    df = pd.read_csv(input_file)
    
    # Get column names
    meter_col = "meter_number"
    timestamp_col = "timestamp_dst"
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Get unique meters
    all_meters = df[meter_col].unique()
    print(f"Total meters in dataset: {len(all_meters)}")
    
    # Select random subset of meters
    np.random.seed(42)  # For reproducibility
    selected_meters = np.random.choice(all_meters, size=min(num_meters, len(all_meters)), replace=False)
    print(f"Selected meters: {sorted(selected_meters)}")
    
    # Filter by selected meters
    df_subset = df[df[meter_col].isin(selected_meters)].copy()
    
    # Sort by timestamp to get earliest days
    df_subset = df_subset.sort_values([meter_col, timestamp_col])
    
    # For each meter, take only the first N days
    result_dfs = []
    for meter_id in selected_meters:
        meter_data = df_subset[df_subset[meter_col] == meter_id]
        
        # Get first N days worth of data (24 hours per day)
        hours_needed = num_days * 24
        meter_subset = meter_data.head(hours_needed)
        
        result_dfs.append(meter_subset)
        print(f"  Meter {meter_id}: {len(meter_subset)} hours")
    
    # Combine all subsets
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    # Sort by meter and timestamp
    final_df = final_df.sort_values([meter_col, timestamp_col])
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    
    print(f"\nTest dataset created:")
    print(f"  Output: {output_file}")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Meters: {final_df[meter_col].nunique()}")
    print(f"  Date range: {final_df[timestamp_col].min()} to {final_df[timestamp_col].max()}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create small test dataset for validation")
    parser.add_argument("--input", type=str, default="data/hourly_measurements_2023_10.csv",
                       help="Input CSV file")
    parser.add_argument("--output", type=str, default="data/test_data_small.csv",
                       help="Output CSV file")
    parser.add_argument("--meters", type=int, default=10,
                       help="Number of meters to include")
    parser.add_argument("--days", type=int, default=2,
                       help="Number of days per meter")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    create_test_dataset(
        input_file=args.input,
        output_file=args.output,
        num_meters=args.meters,
        num_days=args.days,
    )
