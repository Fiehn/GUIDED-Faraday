import math
import torch
import random
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import Optional
import numpy as np
import json
from dataclasses import dataclass
import os
import pandas as pd

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.normalizer import Normalizer

@dataclass(frozen=True)
class WindowSpec:
    input_len: int # length of input window (number of timesteps)
    stride: int = 1 # number of timesteps to move the window each step


class StreamingParquetBuffer(IterableDataset):
    """
    An IterableDataset that reads data from a Parquet file in a streaming manner and uses a replay buffer to store and sample data for training.
    """
    def __init__(self, 
                 sm_files, 
                 replay_buffer_capacity: int = 10000, 
                 min_buffer_size: int = 2000, 
                 window_spec: Optional[WindowSpec] = None,
                 normalizer: Optional[Normalizer] = None,
                 num_workers: int = 0,
                 ):
        self.sm_files = sm_files
        self.replay_buffer_capacity = replay_buffer_capacity
        self.min_buffer_size = min_buffer_size

        # The buffer exists per worker
        # if capacity is 500 and num_workers is 4, then the total effective buffer capacity is 2000
        self.buffer = [None] * replay_buffer_capacity
        self.ptr = 0
        self.current_size = 0

        self.window_spec = window_spec
        self.normalizer = normalizer
        self.num_workers = num_workers

        self.feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        self.time_col_candidates = ['timestamp_dst', 'timestamp', 'timestamp_utc']
        self.base_frequency = '15min'
    
    def _push_to_buffer(self, 
                        item: tuple):
        """
        Push a new item into the replay buffer, overwriting old items in a circular manner.
        """
        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.replay_buffer_capacity
        self.current_size = min(self.current_size + 1, self.replay_buffer_capacity)

    def _sample_from_buffer(self):
        """
        Sample a random item from the replay buffer.

        Returns: 
            A randomly sampled item from the buffer

        Raises:
            ValueError: If the buffer does not have enough items to sample from (i.e., current_size < min_buffer_size)
        """
        if self.current_size < self.min_buffer_size:
            raise ValueError(f"Buffer size {self.current_size} is less than the minimum required {self.min_buffer_size} for sampling.")
        idx = random.randrange(self.current_size)
        return self.buffer[idx]

    def _extract_meter_id(self, 
                          file_path: str) -> int:
        """
        Extract the meter ID from the file name. Assumes file names are in the format 'sm_{meter_id}.parquet'.
        
        Args:
            file_path: The full path to the Parquet file
        
        Returns:
            The extracted meter ID as an integer
        """
        basename = os.path.basename(file_path)
        return int(basename.replace('sm_', '').replace('.parquet', ''))

    def _select_columns(self, 
                        file_cols: list) -> tuple:
        """
        Select the columns to load from the Parquet file based on the expected feature columns and time column candidates.
        
        Args:
            file_cols: The list of column names available in the Parquet file
        
        Returns:            
            A tuple containing:
                - cols_to_load: The list of column names to load from the Parquet file
                - rename_map: A dictionary mapping the actual column names in the file to the expected feature column names
                - time_col_name: The name of the time column found in the file (or None if not found)
        """
        cols_to_load = []
        rename_map = {}

        for base_feat in self.feature_cols:
            matched_col = next((actual_col for actual_col in file_cols if actual_col.startswith(base_feat)), None)
            if matched_col is not None:
                cols_to_load.append(matched_col)
                rename_map[matched_col] = base_feat

        time_col_name = next((col for col in file_cols if col.lower() in self.time_col_candidates), None)
        if time_col_name is not None and time_col_name not in cols_to_load:
            cols_to_load.append(time_col_name)

        return cols_to_load, rename_map, time_col_name

    def _prepare_time_index(self, 
                            df: pd.DataFrame, 
                            time_col_name: Optional[str]) -> pd.DataFrame:
        """
        Prepare the DataFrame by setting a proper datetime index based on the time column. If no time column is found, use the existing index.
        
        Args: 
            df: The input DataFrame loaded from the Parquet file
            time_col_name: The name of the time column in the DataFrame (or None if not found)
        
        Returns:
            A DataFrame with a datetime index prepared for time series processing
        """
        if time_col_name and time_col_name in df.columns:
            time_values = df[time_col_name]
            df = df.drop(columns=[time_col_name])
        else:
            time_values = df.index

        time_index = pd.to_datetime(time_values, errors='coerce', utc=True)
        prepared = df.copy()
        prepared.index = time_index
        prepared = prepared[~prepared.index.isna()]
        prepared = prepared[~prepared.index.duplicated(keep='last')]
        prepared = prepared.sort_index()
        return prepared

    def _impute_to_base_frequency(self, 
                                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing timestamps in the DataFrame to ensure a continuous time index at the base frequency
        
        Args:
            df: The input DataFrame with a datetime index
        
        Returns:
            A DataFrame with missing timestamps imputed to create a continuous time index at the base frequency
        """
        if df.empty:
            return df

        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.base_frequency, tz=df.index.tz)
        return df.reindex(full_index)

    def _load_file_dataframe(self, 
                             file_path: str):
        """
        Load a Parquet file into a DataFrame, selecting and renaming columns as needed, and prepare the time index.
        
        Args:
            file_path: The path to the Parquet file to load
        
        Returns:
            A tuple containing:
                - A DataFrame with the selected features and a prepared time index, or None if loading failed
                - The extracted meter ID from the file name, or None if loading failed
        """
        parquet_file = pq.ParquetFile(file_path)
        file_cols = parquet_file.schema.names
        cols_to_load, rename_map, time_col_name = self._select_columns(file_cols)

        if not cols_to_load:
            return None, None

        df = pq.read_table(file_path, columns=cols_to_load).to_pandas()
        df = df.rename(columns=rename_map)
        meter_id = self._extract_meter_id(file_path)
        return self._prepare_time_index(df, time_col_name), meter_id
    
    def __iter__(self):
        """
        The main loop for the IterableDataset. 
        It controls the multiprocessing behavior, streams Parquet files, extracts sequences, and manages the replay buffer.
        
        Yields:
            (seq, cond) pairs where seq is the input sequence tensor and cond is the condition tensor containing meter ID and timestamps.
        """
        # === Multiprocessing traffic control ===
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process data loading
            files_for_this_worker = self.sm_files
        else:
            # Multi-process data loading
            per_worker = int(math.ceil(len(self.sm_files) / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(self.sm_files))

            files_for_this_worker = self.sm_files[start_idx:end_idx]

        # === Lazy parquet streaming ===
        for file in files_for_this_worker:
            file_df, meter_id = self._load_file_dataframe(file)
            if file_df is None or meter_id is None:
                continue

            sequences = self._extract_sequences(file_df, meter_id)

            for seq, cond in sequences:
                # Apply normalization if normalizer is provided
                if self.normalizer is not None:
                    # seq shape is [24, 3], broadcasting with [3,]
                    seq = self.normalizer.transform(seq)
                seq = seq.T

                # Push the (seq, cond) pair into the replay buffer
                self._push_to_buffer((seq, cond))

                # Yield only when the buffer is adequately mixed
                if self.current_size >= self.min_buffer_size:
                    # The yield pauses the loop right here until PyTorch needs more data
                    yield self._sample_from_buffer()
    
    def _extract_sequences(self, 
                           df: pd.DataFrame, 
                           meter_id: int) -> list:
        """
        Extract sequences of the specified input length and stride from the DataFrame for a given meter ID, and create corresponding condition tensors.
        
        Args:
            df: The input DataFrame containing the time series data for a single meter, with a datetime index and feature columns
            meter_id: The ID of the meter corresponding to the data in the DataFrame
        
        Returns:
            A list of tuples, where each tuple contains:
                - A tensor of shape [seq_len, num_features] representing the input sequence for the model
                - A tensor of shape [2, seq_len] representing the condition for the sequence, where the first row contains the meter ID and the second row contains the timestamps
        """
        sequences = []

        if self.window_spec is None:
            raise ValueError("window_spec must be provided for phased parquet streaming.")

        for feat in self.feature_cols:
            if feat not in df.columns:
                df[feat] = np.nan

        phase_df = df[self.feature_cols].apply(pd.to_numeric, errors='coerce')
        phase_df = self._impute_to_base_frequency(phase_df)

        if phase_df.empty:
            return sequences

        timestamps = (phase_df.index.view('int64') // 10**9).astype(np.int64)
        values = phase_df.fillna(0.0).astype(np.float32).to_numpy()

        for start in range(0, len(values) - self.window_spec.input_len + 1, self.window_spec.stride):
            end = start + self.window_spec.input_len
            
            # Shape is [seq_len, 3] -> (24, 3). We will transpose to [3, seq_len] in the end to be consistent with the rest of the code and how the model expects it
            window = torch.tensor(values[start:end], dtype=torch.float32)
            
            meter_ids = torch.full((self.window_spec.input_len,), meter_id, dtype=torch.long)
            window_timestamps = torch.tensor(timestamps[start:end], dtype=torch.long)
            condition = torch.stack([meter_ids, window_timestamps], dim=0)  # [2, seq_len]
            
            sequences.append((window, condition))
        
        return sequences


class SMPhaseData:
    """
    A data module for loading and preparing smart meter data in a phased manner using a streaming Parquet buffer. 
    This class is designed to work with PyTorch Lightning and provides DataLoaders for training, validation, and testing.
    
    Attributes:
        data_dir: The directory containing the Parquet files with smart meter data
        filename: Optional specific Parquet file to load (if not provided, all Parquet files in the data_dir will be used)
        window_spec: A WindowSpec object defining the input sequence length and stride for extracting sequences from the time series data
        batch_size: The batch size to use for the DataLoaders
        replay_buffer_capacity: The capacity of the replay buffer used in the StreamingParquetBuffer
        seed: An optional random seed for reproducibility of data shuffling and sampling
        train_frac: The fraction of the data to use for training (default: 0.8)
        val_frac: The fraction of the data to use for validation (default: 0.1)
        test_frac: The fraction of the data to use for testing (default: 0.1)
        normalizer: An optional Normalizer object to apply normalization to the input sequences
        num_workers: The number of worker processes to use for data loading (default: 0, meaning data loading will happen in the main process)
        train_loader: The DataLoader for the training set, initialized in the setup method
        val_loader: The DataLoader for the validation set, initialized in the setup method
        test_loader: The DataLoader for the test set, initialized in the setup method
    """
    def __init__(
            self,
            data_dir: str,
            filename: Optional[str] = None,
            window_spec: Optional[WindowSpec] = None,
            batch_size: int = 32,
            replay_buffer_capacity: int = 10000,
            seed: Optional[int] = None,
            train_frac: float = 0.8,
            val_frac: float = 0.1,
            test_frac: float = 0.1,
            normalizer: Optional[Normalizer] = None,
            num_workers: int = 0,
            time_series_info_json: Optional[str] = None,
            **kwargs
    ):
        if filename is not None:
            self.data_files = [filename]
        else:
            self.data_files = self._get_all_files(data_dir, time_series_info_json)
        
        self.window_spec = window_spec
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.normalizer = normalizer
        self.num_workers = num_workers
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self) -> None:
        """
        Set up the DataLoaders for training, validation, and testing by creating a StreamingParquetBuffer dataset and splitting it according to the specified fractions.
        This method should be called before accessing the DataLoaders.
        """
        # Create the streaming dataset
        dataset = StreamingParquetBuffer(
            sm_files=self.data_files,
            replay_buffer_capacity=self.replay_buffer_capacity,
            window_spec=self.window_spec,
            normalizer=self.normalizer,
            num_workers=self.num_workers
        )

        # Create the DataLoaders with the streaming dataset
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def _get_all_files(self, 
                       data_dir: str, 
                       time_series_info_json: Optional[str] = None) -> list:
        """
        Get a list of all Parquet files in the specified data directory. If a time_series_info_json file is provided, use it to determine which files to include based on the information in the JSON.
        
        Args: 
            data_dir: The directory to search for Parquet files
            time_series_info_json: An optional JSON file containing information about the time series data, which can be used to determine which files to include
        
        Returns:
            A list of file paths to the Parquet files to be used for loading data
        """
        if time_series_info_json is not None:
            with open(time_series_info_json, 'r') as f:
                info = json.load(f)
            # json does not have paths but names of the files, so we need to reconstruct the paths
            return [os.path.join(data_dir, filename) for filename in info['files']]
        # This function should return a list of all Parquet files in the data directory
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    

if __name__ == "__main__":
    normalizer = Normalizer(method='log1p_phase')
    normalizer.fit(X = None)  # This will compute and save the normalization stats to 'normalization_stats.json'
    # Example usage
    data_module = SMPhaseData(
        # data_dir='/Users/clarareginekold/Desktop/dtu/THESIS/rasmus-clara-synthetic-data/data/cleaned_profiles',
        data_dir='F:\Data\processed_data\second_batch\individual_sm_datasets\raw_profiles',
        # data_dir='F:\Data\processed_data\second_batch\individual_sm_datasets\cleaned_profiles',
        window_spec=WindowSpec(input_len=24, stride=1),
        batch_size=32,
        replay_buffer_capacity=10000,
        seed=4,
        num_workers=4,
        normalizer=normalizer
    )
    data_module.setup()
    for batch in data_module.train_loader:
        print(batch)
        break  # Just to test one batch
