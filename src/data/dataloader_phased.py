import math
import torch
import random
import pyarrow.parquet as pq
import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, TensorDataset
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
    """An IterableDataset that reads data from a Parquet file in a streaming manner and uses a replay buffer to store and sample data for training."""
    def __init__(self, 
                 sm_files, 
                 replay_buffer_capacity: int = 10000, 
                 min_buffer_size: int = 2000, 
                 window_spec: Optional[WindowSpec] = None,
                 normalizer: Optional[Normalizer] = None,
                 num_workers: int = 0,
                 hourly_aggregation: bool = False,
                 aggregation_divisor: float = 1.0,
                 file_seen_map=None,
                 file_completed_map=None,
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
        self.aggregation = hourly_aggregation
        self.aggregation_divisor = float(aggregation_divisor) if float(aggregation_divisor) > 0 else 1.0
        self.feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        self.time_col_candidates = ['timestamp_dst', 'timestamp', 'timestamp_utc']
        if self.aggregation:
            self.base_frequency = '1h'
        else:
            self.base_frequency = '15min'

        # Shared (manager-backed) maps to track file coverage across worker processes.
        self.file_seen_map = file_seen_map
        self.file_completed_map = file_completed_map

    def _mark_file_seen(self, file_path: str):
        if self.file_seen_map is not None:
            try:
                self.file_seen_map[file_path] = 1
            except Exception:
                pass

    def _mark_file_completed(self, file_path: str):
        if self.file_completed_map is not None:
            try:
                self.file_completed_map[file_path] = 1
            except Exception:
                pass
    
    def _push_to_buffer(self, item):
        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.replay_buffer_capacity
        self.current_size = min(self.current_size + 1, self.replay_buffer_capacity)

    def _sample_from_buffer(self):
        if self.current_size < self.min_buffer_size:
            raise ValueError(f"Buffer size {self.current_size} is less than the minimum required {self.min_buffer_size} for sampling.")
        idx = random.randrange(self.current_size)
        return self.buffer[idx]

    def _extract_meter_id(self, file_path: str) -> int:
        basename = os.path.basename(file_path)
        return int(basename.replace('sm_', '').replace('.parquet', ''))

    def _select_columns(self, file_cols):
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

    def _prepare_time_index(self, df: pd.DataFrame, time_col_name: Optional[str]) -> pd.DataFrame:
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

    def _impute_to_base_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Aggregate first so index labels align with the target base frequency.
        # Building full_index before resampling can misalign (e.g., :45 vs :00)
        # and reindex would then wipe out all values as NaN.
        if self.aggregation:
            df = df.resample(self.base_frequency).mean() # TODO: REMEMBER THIS MOTHERFUCKER! we are using kw not kwh on the 15 min scale
            if self.aggregation_divisor != 1.0:
                df = df / self.aggregation_divisor

        if df.empty:
            return df

        start = df.index.min()
        end = df.index.max()
        full_index = pd.date_range(start=start, end=end, freq=self.base_frequency, tz=df.index.tz)

        return df.reindex(full_index)

    def _load_file_dataframe(self, file_path: str):
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
            self._mark_file_seen(file)
            try:
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
            finally:
                self._mark_file_completed(file)
    
    def _extract_sequences(self, df, meter_id):
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
        #values = phase_df.fillna(0.0).astype(np.float32).to_numpy()
        values = phase_df.dropna().astype(np.float32).to_numpy() # TODO: FUck sommerhuse! 

        for start in range(0, len(values) - self.window_spec.input_len + 1, self.window_spec.stride):
            end = start + self.window_spec.input_len
            
            # Shape is [seq_len, 3] -> (24, 3). We will transpose to [3, seq_len] in the end to be consistent with the rest of the code and how the model expects it
            window = torch.tensor(values[start:end], dtype=torch.float32)
            
            meter_ids = torch.full((self.window_spec.input_len,), meter_id, dtype=torch.long)
            window_timestamps = torch.tensor(timestamps[start:end], dtype=torch.long)
            condition = torch.stack([meter_ids, window_timestamps], dim=0)  # [2, seq_len]
            
            sequences.append((window, condition))
        
        return sequences


class PhasedReplayDataloaders:
    """Builds train/eval DataLoaders from a shared replay stream with optional GPU eval caching."""
    def __init__(
            self,
            *,
            sm_files,
            window_spec: Optional[WindowSpec],
            batch_size: int,
            replay_buffer_capacity: int = 10000,
            min_buffer_size: int = 2000,
            normalizer: Optional[Normalizer] = None,
            num_workers: int = 0,
            hourly_aggregation: bool = False,
            aggregation_divisor: float = 1.0,
            eval_cache_device: Optional[torch.device] = None,
            eval_cache_batches: Optional[int] = None,
            cache_scan_batches: Optional[int] = None,
            pin_memory: bool = False,
    ):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.eval_cache_device = eval_cache_device
        self.eval_cache_batches = eval_cache_batches
        self.cache_scan_batches = cache_scan_batches
        self.pin_memory = bool(pin_memory)
        self.total_files = len(set(sm_files))

        # Manager-backed maps are shared across DataLoader workers.
        self._stats_manager = mp.Manager()
        self._file_seen_map = self._stats_manager.dict()
        self._file_completed_map = self._stats_manager.dict()

        self.files_seen = 0
        self.files_completed = 0

        self.dataset = StreamingParquetBuffer(
            sm_files=sm_files,
            replay_buffer_capacity=replay_buffer_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=window_spec,
            normalizer=normalizer,
            num_workers=num_workers,
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
            file_seen_map=self._file_seen_map,
            file_completed_map=self._file_completed_map,
        )

        self.eval_loader = None
        self._stream_loader = None
        self._stream_iter = None

    def get_file_progress(self):
        self.files_seen = len(self._file_seen_map)
        self.files_completed = len(self._file_completed_map)
        return {
            "seen": int(self.files_seen),
            "completed": int(self.files_completed),
            "total": int(self.total_files),
        }

    def _make_stream(self):
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(self.dataset, **loader_kwargs)

    def _get_stream_loader(self):
        if self._stream_loader is None:
            self._stream_loader = self._make_stream()
        return self._stream_loader

    def _next_stream_batch(self):
        loader = self._get_stream_loader()
        if self._stream_iter is None:
            self._stream_iter = iter(loader)

        try:
            return next(self._stream_iter)
        except StopIteration:
            self._stream_iter = iter(loader)
            try:
                return next(self._stream_iter)
            except StopIteration:
                return None

    def _collect_cache(self, target_train: int, target_eval: int):
        train_x = None
        train_c = None
        eval_x = None
        eval_c = None
        train_samples = 0
        eval_samples = 0

        target_train = max(0, int(target_train))
        target_eval = max(0, int(target_eval))
        total_batches = max(1, (target_train + target_eval + self.batch_size - 1) // self.batch_size)
        max_scan_batches = self.cache_scan_batches or max(30, total_batches * 40)

        for i in range(max_scan_batches):
            batch = self._next_stream_batch()
            if batch is None:
                print("Warning: phased stream produced no data while collecting cache.")
                break

            x, cond = batch
            x = x.float()
            cond = cond.long()

            if train_x is None and target_train > 0:
                train_x = torch.empty((target_train, *x.shape[1:]), dtype=x.dtype)
                train_c = torch.empty((target_train, *cond.shape[1:]), dtype=cond.dtype)
            if eval_x is None and target_eval > 0:
                eval_x = torch.empty((target_eval, *x.shape[1:]), dtype=x.dtype)
                eval_c = torch.empty((target_eval, *cond.shape[1:]), dtype=cond.dtype)

            n = x.shape[0]
            remaining_train = max(0, target_train - train_samples)
            remaining_eval = max(0, target_eval - eval_samples)
            if remaining_train <= 0 and remaining_eval <= 0:
                break

            if remaining_eval > 0:
                denom = max(1, remaining_train + remaining_eval)
                p_eval = float(remaining_eval) / float(denom)
                eval_mask = torch.rand(n) < p_eval

                if eval_mask.sum().item() == 0 and remaining_eval > 0 and n > 1:
                    eval_mask[torch.randint(0, n, (1,)).item()] = True
                if (~eval_mask).sum().item() == 0 and remaining_train > 0 and n > 1:
                    eval_mask[torch.randint(0, n, (1,)).item()] = False
            else:
                eval_mask = torch.zeros(n, dtype=torch.bool)

            x_eval = x[eval_mask]
            c_eval = cond[eval_mask]
            x_train = x[~eval_mask]
            c_train = cond[~eval_mask]

            if remaining_train > 0 and x_train.shape[0] > 0:
                take_train = min(remaining_train, x_train.shape[0])
                train_x[train_samples:train_samples + take_train].copy_(x_train[:take_train])
                train_c[train_samples:train_samples + take_train].copy_(c_train[:take_train])
                train_samples += int(take_train)

            if remaining_eval > 0 and x_eval.shape[0] > 0:
                take_eval = min(remaining_eval, x_eval.shape[0])
                eval_x[eval_samples:eval_samples + take_eval].copy_(x_eval[:take_eval])
                eval_c[eval_samples:eval_samples + take_eval].copy_(c_eval[:take_eval])
                eval_samples += int(take_eval)

            if train_samples >= target_train and eval_samples >= target_eval:
                break

            if i == max_scan_batches - 1:
                print(f"Warning: reached scan limit ({max_scan_batches} batches) while collecting phased cache.")

        if train_samples < target_train and eval_x is not None and eval_samples > 0:
            need = target_train - train_samples
            take = min(need, eval_samples)
            if take > 0:
                train_x[train_samples:train_samples + take].copy_(eval_x[:take])
                train_c[train_samples:train_samples + take].copy_(eval_c[:take])
                train_samples += int(take)

        if train_x is not None:
            train_x = train_x[:train_samples]
        if train_c is not None:
            train_c = train_c[:train_samples]
        if eval_x is not None:
            eval_x = eval_x[:eval_samples]
        if eval_c is not None:
            eval_c = eval_c[:eval_samples]

        return train_x, train_c, eval_x, eval_c

    def _build_eval_loader(self, x_eval, c_eval):
        if x_eval is None or c_eval is None or x_eval.shape[0] == 0:
            self.eval_loader = None
            return

        if torch.is_tensor(c_eval) and c_eval.dim() == 3 and c_eval.shape[1] >= 2:
            meter_id = c_eval[:, 0, 0].long()
            timestamps = c_eval[:, 1, 0].long()
        elif torch.is_tensor(c_eval) and c_eval.dim() == 2 and c_eval.shape[1] >= 2:
            meter_id = c_eval[:, 0].long()
            timestamps = c_eval[:, 1].long()
        else:
            meter_id = torch.zeros(x_eval.shape[0], dtype=torch.long)
            timestamps = torch.zeros(x_eval.shape[0], dtype=torch.long)

        if self.eval_cache_device is not None:
            try:
                x_eval = x_eval.to(self.eval_cache_device)
                meter_id = meter_id.to(self.eval_cache_device)
                timestamps = timestamps.to(self.eval_cache_device)
            except RuntimeError as e:
                # Fall back to CPU cache if GPU preloading runs out of memory.
                if "out of memory" in str(e).lower():
                    print("Warning: eval cache GPU transfer OOM; falling back to CPU eval cache.")
                    self.eval_cache_device = None
                else:
                    raise

        eval_ds = TensorDataset(x_eval, meter_id, timestamps)
        eval_workers = 0
        pin_eval_memory = self.pin_memory and x_eval.device.type == "cpu"
        self.eval_loader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_eval_memory,
        )

    def get_epoch_loaders(self, steps_per_epoch: int, eval_batches: int, refresh_eval: bool = False):
        target_train = max(1, int(steps_per_epoch)) * self.batch_size
        need_eval = int(eval_batches) > 0
        target_eval = 0

        if need_eval:
            if self.eval_loader is None or refresh_eval:
                eval_cache_batches = self.eval_cache_batches if self.eval_cache_batches is not None else eval_batches
                target_eval = max(1, int(eval_cache_batches)) * self.batch_size

        train_x, train_c, eval_x, eval_c = self._collect_cache(target_train, target_eval)

        if train_x is None or train_c is None or train_x.shape[0] == 0:
            return None, self.eval_loader

        train_ds = TensorDataset(train_x, train_c)
        pin_train_memory = self.pin_memory and train_x.device.type == "cpu"
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_train_memory,
        )

        if target_eval > 0:
            self._build_eval_loader(eval_x, eval_c)

        return train_loader, self.eval_loader

    def refresh_eval_cache(self, eval_batches: int):
        target_eval = max(1, int(eval_batches)) * self.batch_size
        _, _, eval_x, eval_c = self._collect_cache(0, target_eval)
        self._build_eval_loader(eval_x, eval_c)

    def get_eval_loader(self):
        return self.eval_loader


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
            hourly_aggregation: bool = False,
                aggregation_divisor: float = 1.0,
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
        self.aggregation = hourly_aggregation
        self.aggregation_divisor = float(aggregation_divisor) if float(aggregation_divisor) > 0 else 1.0

    def setup(self) -> None:
        # Create the streaming dataset
        dataset = StreamingParquetBuffer(
            sm_files=self.data_files,
            replay_buffer_capacity=self.replay_buffer_capacity,
            window_spec=self.window_spec,
            normalizer=self.normalizer,
            num_workers=self.num_workers,
            hourly_aggregation=self.aggregation,
            aggregation_divisor=self.aggregation_divisor,
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
    