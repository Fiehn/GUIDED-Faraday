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
from Data.normalizer import Normalizer

@dataclass(frozen=True)
class WindowSpec:
    input_len: int # length of input window - how many past hours go into the model as x
    stride: int = 1 # how many hours to move the window each step


class StreamingParquetBuffer(IterableDataset):
    """An IterableDataset that reads data from a Parquet file in a streaming manner and uses a replay buffer to store and sample data for training."""
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
    
    def _push_to_buffer(self, item):
        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.replay_buffer_capacity
        self.current_size = min(self.current_size + 1, self.replay_buffer_capacity)

    def _sample_from_buffer(self, batch_size):
        if self.current_size < self.min_buffer_size:
            raise ValueError(f"Buffer size {self.current_size} is less than the minimum required {self.min_buffer_size} for sampling.")
        idx = random.randrange(self.current_size)
        return self.buffer[idx]
    
    def __iter__(self):
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
            # Open parquet file for streaming (does not load entire file into memory)
            parquet_file = pq.ParquetFile(file)

            # strem the file in tiny batches of 1000 rows at a time
            for batch in parquet_file.iter_batches(batch_size=1000):

                # Convert to pandas dataframe for easier processing
                chunk_df = batch.to_pandas()

                # Timestep sequencing : sequences = [(tensor_seq_1, conditions_1), (tensor_seq_2, conditions_2), ...]
                sequences = self._extract_sequences(chunk_df)

                for seq, cond in sequences:
                    # Apply normalization if normalizer is provided
                    if self.normalizer is not None:
                        seq = self.normalizer.transform(seq)

                    # Push the (seq, cond) pair into the replay buffer
                    self._push_to_buffer((seq, cond))

                    # Yield only when the buffer is adequately mixed
                    if self.current_size >= self.min_buffer_size:
                        # The yield pauses the loop right here until PyTorch needs more data
                        yield self._sample_from_buffer(batch_size=1)
    
    def _extract_sequences(self, df):
        sequences = []
        # The desired columns are 'active_power_p14_l1_?meter_id?' and 'active_power_p14_l2_?meter_id?' and 'active_power_p14_l3_?meter_id?'
        # Along with the index extracted out as a separate column 'timestamp' and 'meter_id' as the condition

        # Extract meter_id from the original column names and add it as a separate column
        df['meter_id'] = int(df.columns[0].split('_')[-1])  # assuming all columns have the same meter_id
        # rename active_power_p14_l{n}_<meter_id> -> active_power_p14_l{n} (drop the trailing meter_id suffix)
        df = df.rename(columns={
            col: '_'.join(col.split('_')[:4])  # keep active_power_p14_l{n}
            for col in df.columns if col.startswith('active_power_p14')
        })
        columns = [col for col in df.columns if col.startswith('active_power_p14_l')]

        df['timestamp'] = df.index.astype(np.int64) // 10**9  # convert to unix timestamp in seconds
        
        df = df.reset_index(drop=True)  # reset index to have a clean range index

        # The meter_ids are constant so we dont need to group by them, we can just create windows across the entire dataframe and use the meter_id column as condition
        values = df[columns].apply(pd.to_numeric, errors='coerce').astype(np.float32).values  # shape: [n_rows, 3]
        for start in range(0, len(values) - self.window_spec.input_len + 1, self.window_spec.stride):
            end = start + self.window_spec.input_len
            # window shape: [3, seq_len]
            window = torch.tensor(values[start:end], dtype=torch.float32).T
            # Conditions stacked as [2, seq_len]: condition[0] = meter_id, condition[1] = timestamps
            meter_ids = torch.full((end - start,), df['meter_id'].iloc[0], dtype=torch.long)
            timestamps = torch.tensor(df['timestamp'].values[start:end], dtype=torch.long)
            condition = torch.stack([meter_ids, timestamps], dim=0)  # [2, seq_len]
            sequences.append((window, condition))
        
        return sequences

class SMPhaseData:
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

    def _get_all_files(self, data_dir, time_series_info_json=None):
        if time_series_info_json is not None:
            with open(time_series_info_json, 'r') as f:
                info = json.load(f)
            # json does not have paths but names of the files, so we need to reconstruct the paths
            return [os.path.join(data_dir, filename) for filename in info['files']]
        # This function should return a list of all Parquet files in the data directory
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    


if __name__ == "__main__":
    # Example usage
    data_module = SMPhaseData(
        data_dir='F:\Data\processed_data\second_batch\individual_sm_datasets\cleaned_profiles',
        window_spec=WindowSpec(input_len=24, stride=1),
        batch_size=32,
        replay_buffer_capacity=10000,
        seed=4,
        num_workers=4
    )
    data_module.setup()
    for batch in data_module.train_loader:
        print(batch)
        break  # Just to test one batch
