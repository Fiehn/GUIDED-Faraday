

def prepare_data_cache(data_dir: str, normalizer_path: str) -> None:
    """
    Prepares the Parquet cache for all CSV files in the specified data directory by loading each CSV file using SmartMeterCSV,
    converting it to Parquet format, and saving it in a "parquet" subdirectory within the data directory. 
    This allows for faster subsequent loading of the data, 
    as the Parquet format is more efficient for reading and can be directly loaded into memory without parsing CSV.

    Args:
        data_dir (str): The directory containing the CSV files to be cached. 
            The Parquet files will be saved in a "parquet" subdirectory within this directory.
        normalizer_path (str): Optional file path to save fitted normalizer stats.
            If None, defaults to <data_dir>/_parquet_cache/normalizer.pt.
            If a directory path is provided, the file will be saved as <dir>/normalizer.pt.
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from data.normalizer import Normalizer
    # Fit normalizer and conditioner across all files while preparing cache
    normalizer = Normalizer(method="log1p")
    if normalizer_path is None:
        normalizer_path = os.path.join(data_dir, "_parquet_cache", "normalizer.pt")
    else:
        normalizer_path = os.path.abspath(normalizer_path)
        if os.path.isdir(normalizer_path):
            normalizer_path = os.path.join(normalizer_path, "normalizer.pt")
    
    csv_path = os.path.join(data_dir, "csv")
    if os.path.exists(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
    else:
        csv_path = data_dir
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
    
    # Load all the files so they become parquet-cached, and fit normalizer and conditioner across them afterwards
    for filename in csv_files:
        print(f"Processing {filename} for cache preparation...")
        csv_loader = SmartMeterCSV(
            data_dir=data_dir,
            meter_col="meter_number",
            timestamp_dst_col="timestamp_dst",
            value_cols=["hourly_energy_consumption",],
            nan_as_zero=False,
            parquet_only=False,
        )
        csv_loader.load(filename)
        print(f"{csv_loader.df.isna().sum().sum()} NaN values in {filename} (will be cached as-is, but normalizer fitting will handle them according to its configuration)")
    
    # Then fit normalizer and conditioner across all files using the streaming multi-file fitting functions, which will read from the parquet cache we just prepared for efficiency
    fit_normalizer_on_files(
        normalizer=normalizer,
        data_dir=data_dir,
        filenames=csv_files,
        feature_cols=["hourly_energy_consumption",],
        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        train_frac=1.0,  # use all data for fitting
        seed=0,
        stats_path=normalizer_path,
        force_refit=True,  # force refit to ensure we use the newly prepared cache
    )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Parquet cache and fit normalizer stats for smart meter data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the CSV files (expects a 'csv' subdir).")
    parser.add_argument("--normalizer_path", type=str, default=None, help="Path to save the fitted normalizer stats (defaults to <data_dir>/parquet/normalizer.pt).")
    args = parser.parse_args()

    prepare_data_cache(args.data_dir, args.normalizer_path)



