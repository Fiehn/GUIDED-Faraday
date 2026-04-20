import os
import numpy as np
import pandas as pd

def load_csv_and_cache_as_parquet(
    filename: str,
    cache_dirs: list[str],
    parquet_name: str,
    data_dir: str,
    csv_path: str,
    meter_col: str,
    timestamp_dst_col: str,
    value_cols: list[str],
    timestamp_utc_col: str | None = None,
    max_failed_value_ratio: float | None = 0.01,
    max_bad_examples: int = 10,
) -> pd.DataFrame:
    """Loads the specified CSV file, performs data quality checks and cleaning, and caches the cleaned DataFrame as
    a Parquet file for faster future loading. The function reads the CSV in chunks to handle large files efficiently,
    and it collects statistics on data quality issues such as failed value conversions and control characters.
    Args:
        filename (str): The name of the CSV file to load (e.g., "data.csv").
        data_dir (str): The directory where the CSV file is located.
        cache_dirs (list[str]): A list of directories to check for existing Parquet cache files before loading from CSV.
        parquet_name (str): The name to use for the cached Parquet file (e.g., "data.parquet").
    Returns:
        pd.DataFrame: The loaded and cleaned DataFrame with a MultiIndex of (meter_col
, timestamp_dst_col) and sorted by this index.
    """
    if os.path.isabs(filename):
            path_candidates = [filename]
    else:
        path_candidates = [
            os.path.join(csv_path, filename),
            os.path.join(data_dir, filename),
        ]
    path = next((p for p in path_candidates if os.path.exists(p)), None)
        
    columns_to_load = [meter_col, timestamp_dst_col] + value_cols
    # Include UTC column only when explicitly requested
    if timestamp_utc_col is not None:
        columns_to_load.append(timestamp_utc_col)

    dtype_map = {
        meter_col: "int32",
    }

    chunks: list[pd.DataFrame] = []
    total_values = 0
    failed_values = 0
    control_char_values = 0
    coerced_nan_values = 0
    rows_seen = 0
    bad_examples: list[str] = []

    for chunk in pd.read_csv(
        path,
        usecols=columns_to_load,
        dtype=dtype_map,
        chunksize=200_000,
    ):
        chunk_start_row = rows_seen + 2  # 1-based file rows; row 1 is the header
        chunk[timestamp_dst_col] = pd.to_datetime(chunk[timestamp_dst_col], format='ISO8601')
        for c in value_cols:
            raw_values = chunk[c].astype("string")
            cleaned_values = raw_values.str.replace(r"[\x00-\x1F\x7F]", "", regex=True).str.strip()
            ctrl_mask = raw_values.str.contains(r"[\x00-\x1F\x7F]", regex=True, na=False)

            numeric_values = pd.to_numeric(cleaned_values, errors="coerce")
            failed_mask = cleaned_values.notna() & cleaned_values.ne("") & numeric_values.isna()
            problem_mask = ctrl_mask | failed_mask

            col_total = len(chunk)
            col_ctrl = int(ctrl_mask.sum())
            col_failed = int(problem_mask.sum())
            col_coerced = int(failed_mask.sum())

            total_values += col_total
            control_char_values += col_ctrl
            failed_values += col_failed
            coerced_nan_values += col_coerced

            if col_failed > 0 and len(bad_examples) < max_bad_examples:
                bad_positions = np.flatnonzero(problem_mask.to_numpy())
                needed = max_bad_examples - len(bad_examples)
                for pos in bad_positions[:needed]:
                    file_row = chunk_start_row + int(pos)
                    meter_val = chunk.iloc[int(pos)][meter_col]
                    ts_val = chunk.iloc[int(pos)][timestamp_dst_col]
                    raw_val = raw_values.iloc[int(pos)]
                    bad_examples.append(
                        f"row={file_row}, col={c}, meter={meter_val}, timestamp={ts_val}, raw={raw_val!r}"
                    )

            chunk[c] = numeric_values.astype(np.float32)
        chunks.append(chunk)
        rows_seen += len(chunk)

    failed_ratio = (failed_values / total_values) if total_values > 0 else 0.0
    last_quality_report = {
        "filename": filename,
        "total_values": int(total_values),
        "failed_values": int(failed_values),
        "failed_ratio": float(failed_ratio),
        "control_char_values": int(control_char_values),
        "coerced_nan_values": int(coerced_nan_values),
    }

    print(
        f"[DataQuality] {filename}: failed={failed_values}/{total_values} "
        f"({failed_ratio:.4%}), control_chars={control_char_values}, coerced_nan={coerced_nan_values}"
    )

    if bad_examples:
        print(f"[DataQuality] {filename}: sample problematic values:")
        for ex in bad_examples:
            print(f"  - {ex}")

    if max_failed_value_ratio is not None and failed_ratio > max_failed_value_ratio:
        raise ValueError(
            f"Too many failed values in {filename}: {failed_values}/{total_values} "
            f"({failed_ratio:.4%}) exceeds max_failed_value_ratio={max_failed_value_ratio:.2%}."
        )

    df = pd.concat(chunks, ignore_index=True)
    del chunks  # free intermediate memory before the MultiIndex sort

    df.set_index([meter_col, timestamp_dst_col], inplace=True)
    df.sort_index(inplace=True)

    # Persist as Parquet cache for future runs
    cache_dir = cache_dirs[0]
    parquet_path = os.path.join(cache_dir, parquet_name)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")
    return df

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
        csv_loader = load_csv_and_cache_as_parquet(
            filename=filename,
            cache_dirs=[os.path.join(data_dir, "parquet")],
            parquet_name=filename.replace(".csv", ".parquet"),
            data_dir=data_dir,
            csv_path=csv_path,
            meter_col="meter_number",
            timestamp_dst_col="timestamp_dst",
            value_cols=["hourly_energy_consumption"],
        )
        
        print(f"{csv_loader.isna().sum().sum()} NaN values in {filename} (will be cached as-is, but normalizer fitting will handle them according to its configuration)")
    
    # Then fit normalizer and conditioner across all files using the streaming multi-file fitting functions, which will read from the parquet cache we just prepared for efficiency
    normalizer.fit_normalizer_on_files(
        normalizer=normalizer,
        data_dir=data_dir,
        filenames=csv_files,
        feature_cols=["hourly_energy_consumption",],
        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        train_frac=1.0,  # use all data for fitting
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



