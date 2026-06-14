import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def create_subset(meters_file: str, output_file: str, data_dir: str):
    # The meters are in the csv file. The parquet datafiles are sorted by month,
    # so we load each month file and filter the data for the meters in the csv file, then save the filtered data to a new parquet file.
    meters_df = pd.read_csv(meters_file)

    # Normalize meter ids to numeric to avoid dtype mismatches during filtering.
    meter_ids = pd.to_numeric(meters_df["meter_nr"], errors="coerce").dropna().astype("int64")
    meter_id_set = set(meter_ids.tolist())

    if not meter_id_set:
        raise ValueError("No valid meter ids found in meters file.")

    print(f"Loaded {len(meter_id_set)} meter ids from {meters_file}.")

    # Get the list of parquet files in the data directory
    parquet_dir = os.path.join(data_dir, '_parquet_cache')
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}.")

    # Create an empty list to store the filtered dataframes
    filtered_dfs = []
    # Loop through each parquet file, filter the data for the meter ids, and append the
    # filtered dataframe to the list
    for parquet_file in parquet_files:
        print(f"Processing file: {parquet_file}")
        parquet_path = os.path.join(parquet_dir, parquet_file)
        df = pd.read_parquet(parquet_path)

        
        print("Found 'meter_number' in index levels.")
        meter_values = pd.to_numeric(
            df.index.get_level_values("meter_number"), errors="coerce"
        )
        print(f"Extracted meter values from index: {meter_values}")
        mask = meter_values.isin(meter_id_set)
        print(f"Mask for filtering: {mask}")
        filtered_df = df[mask]
        print(f"Filtered dataframe shape: {filtered_df.shape}")
        

        filtered_dfs.append(filtered_df)
        print(f"Filtered data for {parquet_file} has {len(filtered_df)} rows.")

    # Concatenate all the filtered dataframes into a single dataframe
    subset_df = pd.concat(filtered_dfs, ignore_index=False)

    # Save the subset dataframe to a new parquet file
    subset_table = pa.Table.from_pandas(subset_df, preserve_index=True)
    pq.write_table(subset_table, output_file)

    print(f"Subset created and saved to {output_file}")
    print(f"Total rows in subset: {len(subset_df)}")
    
if __name__ == "__main__":
    create_subset(meters_file='data/meters_from_bachelor.csv', output_file='data/hourly_subset.parquet', data_dir='data')
