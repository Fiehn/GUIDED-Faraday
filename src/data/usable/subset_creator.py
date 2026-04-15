import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def create_subset(meters_file: str, output_file: str, data_dir: str):
    # The meters are in the csv file. The parquet datafiles are sorted by month,
    # so we load each month file and filter the data for the meters in the csv file, then save the filtered data to a new parquet file.
    meters_df = pd.read_csv(meters_file)
    meter_ids = meters_df['meter_nr'].tolist()
    # Get the list of parquet files in the data directory
    parquet_dir = os.path.join(data_dir, '_parquet_cache')
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    # Create an empty list to store the filtered dataframes
    filtered_dfs = []
    # Loop through each parquet file, filter the data for the meter ids, and append the
    # filtered dataframe to the list
    for parquet_file in parquet_files:
        parquet_path = os.path.join(parquet_dir, parquet_file)
        df = pd.read_parquet(parquet_path)
        filtered_df = df[df['meter_nr'].isin(meter_ids)]
        filtered_dfs.append(filtered_df)
    # Concatenate all the filtered dataframes into a single dataframe
    subset_df = pd.concat(filtered_dfs, ignore_index=True)
    # Save the subset dataframe to a new parquet file
    subset_table = pa.Table.from_pandas(subset_df)
    pq.write_table(subset_table, output_file)

if __name__ == "__main__":
    create_subset(meters_file='data/meters_from_bachelor.csv', output_file='data/hourly_subset.parquet', data_dir='data')
