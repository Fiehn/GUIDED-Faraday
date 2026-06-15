import math
import hashlib
import json
import torch
import random
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, TensorDataset, Dataset, Subset
from typing import Optional
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
import gc
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading
import time

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.normalizer import Normalizer

@dataclass(frozen=True)
class WindowSpec:
    input_len: int # length of input window - how many past hours go into the model as x
    stride: int = 1 # how many hours to move the window each step


def _extract_meter_id_from_filename(fname: str) -> int:
    """Extract meter ID from a filename by collecting digit characters.

    Works for 'sm_1001.parquet', 'meter_1001.parquet', '1001.parquet'.
    Falls back to a stable hash when the filename contains no digits.
    """
    digits = "".join(c for c in fname if c.isdigit())
    return int(digits) if digits else hash(fname) & 0x7FFFFFFF


def _impute_to_base_frequency(
    df: pd.DataFrame,
    aggregation: bool = False,
    base_frequency: str = "1h",
    aggregation_divisor: float = 1.0,
) -> pd.DataFrame:
    """Resample df to base_frequency and apply divisor when aggregation=True, then fill gaps."""
    if df.empty:
        return df

    if aggregation:
        needs_resample = False
        if len(df) > 1:
            first_diff = df.index[1] - df.index[0]
            target_delta = pd.Timedelta(base_frequency)
            needs_resample = first_diff < target_delta * 0.9
        if needs_resample:
            df = df.resample(base_frequency).mean()
        if aggregation_divisor != 1.0:
            df = df / aggregation_divisor

    if df.empty:
        return df

    start = df.index.min()
    end = df.index.max()
    full_index = pd.date_range(start=start, end=end, freq=base_frequency, tz=df.index.tz)
    return df.reindex(full_index)


def _extract_sequences_for_meter(
    meter_df: pd.DataFrame,
    meter_id: int,
    window_len: int,
    stride: int,
    max_value: float | None,
    max_hour_value: float | None,
    nan_as_zero: bool,
    split_seed: int | None,
    split_ratios: dict,
    split: str | None,
) -> list:
    """Extract valid, midnight-aligned windows from a preprocessed (base-frequency) DataFrame.

    meter_df must already be at base frequency with the desired feature columns.
    Returns list of (window_t [window_len, num_phases], meter_id_t [], timestamp_t []).
    Only windows belonging to `split` are returned (all windows when split is None).
    """
    sequences = []

    if nan_as_zero:
        meter_df = meter_df.fillna(0.0)
    else:
        finite_mask = meter_df.notna().all(axis=1)
        meter_df = meter_df.loc[finite_mask]
        if meter_df.empty:
            return sequences

    try:
        dt_index = pd.DatetimeIndex(pd.to_datetime(meter_df.index, utc=True, errors="coerce"))
        if np.any(pd.isnull(dt_index)):
            return sequences
        naive = dt_index.tz_convert("UTC").tz_localize(None)
        timestamps = naive.to_numpy(dtype="datetime64[s]").view(np.int64)  # SHAPER: [num_timesteps]
    except Exception:
        return sequences

    values = meter_df.astype(np.float32).to_numpy()  # SHAPER: [num_timesteps, num_phases]
    if len(values) < window_len:
        return sequences

    abs_totals = np.abs(values).sum(axis=1)  # SHAPER: [num_timesteps]
    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(abs_totals, out=prefix[1:])

    if max_hour_value is not None:
        bad_hour = (abs_totals > max_hour_value).astype(np.int32)
        prefix_bad = np.empty(len(values) + 1, dtype=np.int64)
        prefix_bad[0] = 0
        np.cumsum(bad_hour, out=prefix_bad[1:])
    else:
        prefix_bad = None

    meter_id_t = torch.tensor(int(meter_id), dtype=torch.long)

    first_aligned_idx = 0
    for _tz_offset in (3600, 7200, 0):
        _hits = np.where((timestamps + _tz_offset) % 86400 == 0)[0]
        if len(_hits) > 0:
            first_aligned_idx = int(_hits[0])
            break

    all_starts = np.arange(first_aligned_idx, len(values) - window_len + 1, stride, dtype=np.intp)
    if len(all_starts) == 0:
        return sequences
    all_ends = all_starts + window_len

    window_sums = prefix[all_ends] - prefix[all_starts]
    valid = window_sums > 0.0
    if max_value is not None:
        valid &= window_sums <= max_value
    if prefix_bad is not None:
        valid &= prefix_bad[all_ends] == prefix_bad[all_starts]

    valid_starts = all_starts[valid]
    if len(valid_starts) == 0:
        return sequences

    indices = valid_starts[:, None] + np.arange(window_len, dtype=np.intp)
    windows_np = values[indices]  # SHAPER: [num_valid_windows, window_len, num_phases]
    windows_t = torch.from_numpy(windows_np)
    ts_t = torch.from_numpy(timestamps[valid_starts].copy())

    for i in range(len(valid_starts)):
        if split is not None:
            if csv_split_bucket(split_seed, split_ratios, int(meter_id), int(ts_t[i].item())) != split:
                continue
        sequences.append((windows_t[i].clone(), meter_id_t, ts_t[i].clone()))

    return sequences


def csv_split_bucket(
    seed: int | None,
    split_ratios: dict,
    meter_id: int,
    timestamp_s: int,
) -> str:
    """Assign a window to train/val/test via SHA1(seed|meter|timestamp).

    Single source of truth for split assignment; both StreamingParquetBuffer
    and InMemoryCSVDataset call this so train/val/test composition is identical.
    """
    train_ratio = float(split_ratios.get("train_ratio", 1.0))
    val_ratio   = float(split_ratios.get("val_ratio",   0.0))
    test_ratio  = float(split_ratios.get("test_ratio",  0.0))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return "train"
    train_ratio /= total
    val_ratio   /= total

    seed_str = "0" if seed is None else str(int(seed))
    key = f"{seed_str}|{int(meter_id)}|{int(timestamp_s)}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    u = int.from_bytes(digest[:8], "big") / float(2**64)

    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


class StreamingParquetBuffer(IterableDataset):
    """An IterableDataset that reads data from a Parquet file in a streaming manner and uses a replay buffer to store and sample data for training."""
    def __init__(self, 
                 sm_files, 
                 replay_buffer_capacity: int = 10000, 
                 min_buffer_size: int = 2000, 
                 window_spec = None,
                 normalizer: Optional[Normalizer] = None,
                 num_workers: int = 0,
                 hourly_aggregation: bool = False,
                 aggregation_divisor: float = 1.0,
                 file_seen_counter=None,
                 file_completed_counter=None,
                 file_skipped_counter=None,
                 skip_corrupt_files: bool = True,
                 max_skip_warnings: int = 20,
                 output_aggregate_stream: bool = False,
                 sample_random: bool = True,
                 nan_as_zero: bool = False,
                 phased_mode: bool = True,
                 meter_col: str = "meter_number",
                 feature_cols: list[str] | None = None,
                 seed: int | None = None,
                 split: str | None = None,
                 split_ratios: dict | None = None,
                 max_value: float | None = None,
                 max_hour_value: float | None = 20.0,
                 batch_size: int = 256,
                 ):
        self.sm_files = sm_files
        self.replay_buffer_capacity = replay_buffer_capacity
        self.min_buffer_size = min_buffer_size
        self.seed = None if seed is None else int(seed)
        self.split = None if split is None else str(split).lower()
        self.split_ratios = split_ratios or {}
        self.max_value = float(max_value) if max_value is not None else None
        self.max_hour_value = float(max_hour_value) if max_hour_value is not None else None
        self.batch_size = int(batch_size)

        # The buffer exists per worker
        # if capacity is 500 and num_workers is 4, then the total effective buffer capacity is 2000
        self.buffer = [None] * replay_buffer_capacity
        self.ptr = 0
        self.current_size = 0

        # Lock lives on the object so it survives across __iter__ calls
        # (persistent_workers=True calls __iter__ again without forking).
        self._buffer_lock = threading.Lock()
        # Monotonically-increasing epoch counter. Background threads capture
        # their generation at start; on mismatch they exit cleanly instead of
        # writing into a buffer that was reset for a new epoch.
        self._generation = 0

        self.window_spec = window_spec
        self.normalizer = normalizer
        self.num_workers = num_workers
        self.aggregation = hourly_aggregation
        self.aggregation_divisor = float(aggregation_divisor) if float(aggregation_divisor) > 0 else 1.0
        self.time_col_candidates = ['timestamp_dst', 'timestamp', 'timestamp_utc']
        self.meter_col = str(meter_col)
        if self.aggregation:
            self.base_frequency = '1h'
        else:
            self.base_frequency = '15min'
    
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
            matched_col = next((actual_col for actual_col in file_cols if actual_col == base_feat), None)
            if matched_col is None:
                matched_col = next((actual_col for actual_col in file_cols if actual_col.startswith(base_feat)), None)
            if matched_col is not None:
                cols_to_load.append(matched_col)
                rename_map[matched_col] = base_feat

        time_col_name = None
        lower_cols = [c.lower() for c in file_cols]
        for candidate in self.time_col_candidates:
            if candidate in lower_cols:
                time_col_name = next(c for c in file_cols if c.lower() == candidate)
                break

        if time_col_name is not None and time_col_name not in cols_to_load:
            cols_to_load.append(time_col_name)

        meter_col_name = next((col for col in file_cols if col.lower() == self.meter_col.lower()), None)
        if meter_col_name is not None and meter_col_name not in cols_to_load:
            cols_to_load.append(meter_col_name)

    def _prepare_time_index(self, df: pd.DataFrame, time_col_name: Optional[str]) -> pd.DataFrame:
        if time_col_name and time_col_name in df.columns:
            time_values = pd.to_datetime(df[time_col_name], errors='coerce', utc=True)
            df = df.drop(columns=[time_col_name])
        else:
            time_values = pd.to_datetime(df.index, errors='coerce', utc=True)

        prepared = df.copy()

        if self.phased_mode:
            prepared.index = time_values
            prepared = prepared[~prepared.index.isna()]
            prepared = prepared[~prepared.index.duplicated(keep='last')]
            prepared = prepared.sort_index()
            return prepared

    def _impute_to_base_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.base_frequency, tz=df.index.tz)
        # Aggregate to base frequency if needed
        if self.aggregation:
            df = df.resample(self.base_frequency).sum()

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
        cols_to_load, rename_map, time_col_name, meter_col_name = self._select_columns(file_cols)

        if not cols_to_load:
            return None, None, None

        if time_col_name is None:
            raise ValueError(
                f"No timestamp column found in {file_path}. "
                f"Expected one of {self.time_col_candidates}."
            )

        if not self.phased_mode and meter_col_name is None:
            raise ValueError(
                f"No meter column '{self.meter_col}' found in {file_path}."
            )

        table = pq.read_table(file_path, columns=cols_to_load)
        df = table.to_pandas(ignore_metadata=True)  
        
        df = df.rename(columns=rename_map)
        if meter_col_name not in df.columns and meter_col_name in list(getattr(df.index, "names", [])):
            df = df.reset_index()
            df = df.rename(columns={meter_col_name: self.meter_col})
            meter_col_name = self.meter_col

        prepared = self._prepare_time_index(df, time_col_name, meter_col_name)
        meter_id = self._extract_meter_id(file_path) if self.phased_mode else None
        return prepared, meter_id, meter_col_name

    def _iter_file_chunks(self, file_path: str, worker_id: int = 0, num_workers: int = 1):
        """Yield (prepared_df, meter_col_name) in row-bounded chunks.

        Only used in unphased mode. Partitions at the row-group level so that
        multiple DataLoader workers can operate on the same large file in parallel:
        worker i reads row groups i, i+num_workers, i+2*num_workers, …
        Within each row group, reads are chunked to _UNPHASED_CHUNK_ROWS to cap
        peak memory (each row group is ~1 M rows for this dataset).
        """
        parquet_file = pq.ParquetFile(file_path)
        file_cols = parquet_file.schema.names
        cols_to_load, rename_map, time_col_name, meter_col_name = self._select_columns(file_cols)

        if not cols_to_load or time_col_name is None or meter_col_name is None:
            return

        num_row_groups = parquet_file.metadata.num_row_groups
        for rg_idx in range(worker_id, num_row_groups, num_workers):
            try:
                for batch in parquet_file.iter_batches(
                    batch_size=self._UNPHASED_CHUNK_ROWS,
                    row_groups=[rg_idx],
                    columns=cols_to_load,
                ):
                    df = batch.to_pandas(ignore_metadata=True)
                    if df.empty:
                        continue
                    df = df.rename(columns=rename_map)
                    effective_meter_col = meter_col_name
                    if meter_col_name not in df.columns and meter_col_name in list(getattr(df.index, "names", [])):
                        df = df.reset_index()
                        df = df.rename(columns={meter_col_name: self.meter_col})
                        effective_meter_col = self.meter_col
                    prepared = self._prepare_time_index(df, time_col_name, effective_meter_col)
                    if prepared.empty:
                        continue
                    yield prepared, effective_meter_col
            except Exception as e:
                if self.skip_corrupt_files and self._is_recoverable_file_error(e):
                    print(f"  [data] Recovered: Skipped corrupted row group {rg_idx} in {os.path.basename(file_path)}", flush=True)
                    continue
                
                # If it's a completely unrecognized fatal error, raise it 
                # so the outer loop catches it and skips the file.
                raise

    def _meter_value_to_int(self, meter_value) -> int:
        try:
            return int(meter_value)
        except (TypeError, ValueError):
            # Only if everything has gone wrong with parsing
            digest = hashlib.sha1(str(meter_value).encode("utf-8")).hexdigest()
            return int(digest[:8], 16)
    
    def __iter__(self):
        # === Multiprocessing traffic control ===
        worker_info = get_worker_info()

        # Worker-local RNG so file order and buffer sampling are reproducible.
        base_seed = 0 if self.seed is None else int(self.seed)
        worker_id = 0 if worker_info is None else int(worker_info.id)
        self._rng = random.Random(base_seed + 1009 * worker_id)

        num_workers = 1 if worker_info is None else int(worker_info.num_workers)

        if worker_info is None:
            # Single-process data loading — worker owns all files.
            files_for_this_worker = list(self.sm_files)
        elif self.phased_mode:
            # Phased: many small files (one per meter) — partition by file so
            # each worker owns a disjoint subset and reads each file whole.
            per_worker = int(math.ceil(len(self.sm_files) / float(num_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.sm_files))
            files_for_this_worker = self.sm_files[start_idx:end_idx]
        else:
            # Unphased: few large files (all meters in one file) — every worker
            # gets every file, but _iter_file_chunks partitions within each file
            # at the row-group level so no two workers duplicate work.
            files_for_this_worker = list(self.sm_files)

        if self.sample_random:
            self._rng.shuffle(files_for_this_worker)

        if not files_for_this_worker:
            return
        
        # Reset buffer state while holding the shared lock so that any
        # background thread still alive from a previous epoch cannot write
        # a stale ptr value after we set ptr=0 (the root cause of the
        # "NoneType not subscriptable" crash on epoch 2+).
        with self._buffer_lock:
            self._generation += 1          # signal old threads to exit
            my_generation = self._generation

            self.buffer = [None] * self.replay_buffer_capacity
            self.ptr = 0
            self.current_size = 0

            self._push_count = 0
            self._yield_count = 0

        def background_file_reader():
            while my_generation == self._generation:
                for file in files_for_this_worker:
                    # Exit immediately if a newer epoch has started.
                    if my_generation != self._generation:
                        return
                    self._mark_file_seen(file)
                    try:
                        if self.phased_mode:
                            file_df, m_id, m_col = self._load_file_dataframe(file)
                            if file_df is None: continue
                            chunks = [(file_df, m_id, m_col)]
                        else:
                            chunks = (
                                (chunk_df, None, chunk_meter_col)
                                for chunk_df, chunk_meter_col in self._iter_file_chunks(file, worker_id, num_workers)
                            )

                        for chunk_df, chunk_m_id, chunk_m_col in chunks:
                            # If the GPU hasn't caught up, go to sleep and free the CPU/RAM.
                            while self._push_count - self._yield_count >= self.replay_buffer_capacity:
                                if my_generation != self._generation:
                                    return
                                time.sleep(0.05)

                            sequences = self._extract_sequences(chunk_df, meter_id=chunk_m_id, meter_col_name=chunk_m_col)

                            # Lock ONCE per chunk to minimize CPU contention.
                            # Re-check generation inside the lock: if __iter__ reset the
                            # buffer while we were outside the lock, bail out immediately
                            # rather than writing stale data at a mismatched ptr.
                            with self._buffer_lock:
                                if my_generation != self._generation:
                                    return
                                pushed = 0
                                for seq, meter_id_t, timestamp_t in sequences:
                                    if not self._keep_window(
                                        int(meter_id_t.item()), int(timestamp_t.item())
                                    ):
                                        continue
                                    if self.normalizer is not None:
                                        seq = self.normalizer.transform(seq)

                                    # Fast transpose
                                    seq = seq.T.contiguous()  # SHAPER: [input_len, num_phases] -> [num_phases, input_len]

                                    if self.output_aggregate_stream:
                                        agg = seq.sum(dim=0, keepdim=True)  # SHAPER: [num_phases, input_len] -> [1, input_len]
                                    else:
                                        agg = torch.empty((0, seq.shape[-1]), dtype=seq.dtype, device=seq.device)  # SHAPER: [0, input_len]

                                    # O(1) Pointer Overwrite. The OS Garbage Collector handles deletion silently.
                                    self.buffer[self.ptr] = (seq, meter_id_t, timestamp_t, agg)
                                    self.ptr = (self.ptr + 1) % self.replay_buffer_capacity
                                    self.current_size = min(self.current_size + 1, self.replay_buffer_capacity)
                                    pushed += 1
                                self._push_count += pushed

                    except Exception as e:
                        if self.skip_corrupt_files and self._is_recoverable_file_error(e):
                            self._warn_skipped_file(file, e)
                            continue
                        print(f"Background thread error on {file}: {e}", flush=True)
                    finally:
                        self._mark_file_completed(file)
                        
                if self.sample_random:
                    self._rng.shuffle(files_for_this_worker)

        # Start the background daemon thread
        io_thread = threading.Thread(target=background_file_reader, daemon=True)
        io_thread.start()

        # === Lazy parquet streaming (infinite loop so workers never exhaust) ===
        while True:
            # Wait for the buffer to warm up (always require at least 1 item
            # so min_buffer_size=0 callers still block until data arrives)
            while True:
                with self._buffer_lock:
                    if self.current_size >= max(self.min_buffer_size, 1):
                        break
                time.sleep(0.005)
            
            chunk_list = []
            
            with self._buffer_lock:
                take_count = min(self.batch_size, self.current_size)
                
                # THE MITIGATION: random.sample guarantees we NEVER pull the same index 
                # twice in a single batch. Intra-batch duplicates are mathematically eliminated.
                if self.sample_random:
                    indices = self._rng.sample(range(self.current_size), take_count)
                else:
                    # Deterministic fallback: just take the most recently written items
                    start_idx = (self.ptr - take_count) % self.replay_buffer_capacity
                    indices = [(start_idx + i) % self.replay_buffer_capacity for i in range(take_count)]
                
                for idx in indices:
                    chunk_list.append(self.buffer[idx])
            
            # Fast C++ Concatenation
            x_cat = torch.stack([c[0] for c in chunk_list], dim=0)  # SHAPER: [num_phases, input_len] * batch_size -> [batch_size, num_phases, input_len]
            m_cat = torch.stack([c[1] for c in chunk_list], dim=0)  # SHAPER: [] * batch_size -> [batch_size]
            t_cat = torch.stack([c[2] for c in chunk_list], dim=0)  # SHAPER: [] * batch_size -> [batch_size]
            a_cat = torch.stack([c[3] for c in chunk_list], dim=0)  # SHAPER: [1, input_len] * batch_size -> [batch_size, 1, input_len]

            self._yield_count += take_count
            
            # Yield massive blocks to prevent IPC queue flooding
            # .clone() severs the memory link so PyTorch doesn't leak memory
            yield (x_cat.clone(), m_cat.clone(), t_cat.clone(), a_cat.clone())
    
    def _extract_sequences(self, df, meter_id):
        sequences = []

        if self.window_spec is None:
            raise ValueError("window_spec must be provided for phased parquet streaming.")

        for feat in self.feature_cols:
            if feat not in meter_df.columns:
                meter_df[feat] = np.nan

        phase_df = meter_df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        phase_df = self._impute_to_base_frequency(phase_df)

        if phase_df.empty:
            return []

        return _extract_sequences_for_meter(
            phase_df, meter_id,
            window_len=self.window_spec.input_len,
            stride=self.window_spec.stride,
            max_value=self.max_value,
            max_hour_value=self.max_hour_value,
            nan_as_zero=self.nan_as_zero,
            split_seed=self.seed,
            split_ratios=self.split_ratios,
            split=None,  # split filtering done by _keep_window in __iter__
        )

    def _extract_sequences(self, df, meter_id=None, meter_col_name: str | None = None):
        if self.phased_mode:
            if meter_id is None:
                raise ValueError("phased_mode=True requires meter_id from filename parsing")
            return self._extract_sequences_for_meter(df, meter_id=int(meter_id))

        if meter_col_name is None or meter_col_name not in df.columns:
            raise ValueError(
                f"Unphased mode requires meter column '{self.meter_col}' in loaded data."
            )
        if "_timestamp" not in df.columns:
            raise ValueError("Unphased mode expected '_timestamp' column after timestamp parsing.")

        meter_series = df[meter_col_name]
        valid = meter_series.notna()
        if not valid.any():
            return []

        grouped = df.loc[valid].copy()

        # Vectorised meter-ID extraction: pd.to_numeric is 100-500x faster than
        # .apply() for the common case where IDs are numeric.  Non-numeric IDs
        # (SHA1 fallback) are handled row-by-row only for the rare mixed case.
        meter_numeric = pd.to_numeric(grouped[meter_col_name], errors='coerce')
        if meter_numeric.isna().any():
            grouped["_meter_id_int"] = grouped[meter_col_name].apply(self._meter_value_to_int).to_numpy(dtype=np.int64)
        else:
            grouped["_meter_id_int"] = meter_numeric.to_numpy(dtype=np.int64)

        sequences = []
        for meter_value, meter_df in grouped.groupby("_meter_id_int"):
            meter_frame = meter_df.set_index("_timestamp")
            meter_frame = meter_frame[~meter_frame.index.duplicated(keep='last')]
            sequences.extend(self._extract_sequences_for_meter(meter_frame, meter_id=int(meter_value)))
            
            del meter_frame
            del meter_df

        del grouped
        del meter_series
        del meter_numeric
        del df 
        
        # Force Python to tell PyArrow to release the C++ memory back to the OS
        gc.collect()
        return sequences


class SMPhaseData:
    def __init__(
            self,
            *,
            sm_files,
            window_spec,
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
            pin_memory: bool = True,
            output_aggregate_stream: bool = False,
            skip_corrupt_files: bool = True,
            max_skip_warnings: int = 20,
            sample_random: bool = True,
            nan_as_zero: bool = False,
            phased_mode: bool = True,
            meter_col: str = "meter_number",
            feature_cols: list[str] | None = None,
            split_seed: int | None = None,
            split_ratios: dict | None = None,
                max_value: float | None = None,
                max_hour_value: float | None = 20.0,
                holdout_feeder_id: int | None = None,
                topology_path: str | None = None,
                holdout_meter_ids: list[int] | None = None,
    ):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.eval_cache_device = eval_cache_device
        self.eval_cache_batches = eval_cache_batches
        self.cache_scan_batches = cache_scan_batches
        self.pin_memory = bool(pin_memory)
        self.output_aggregate_stream = bool(output_aggregate_stream)
        self.sample_random = bool(sample_random)
        self.phased_mode = bool(phased_mode)
        self.split_seed = None if split_seed is None else int(split_seed)
        self.split_ratios = split_ratios or {}
        self.max_hour_value = float(max_hour_value) if max_hour_value is not None else None
        self._holdout_feeder_id = holdout_feeder_id
        self._topology_path = topology_path
        self._holdout_meter_ids_direct: frozenset[int] = (
            frozenset(int(m) for m in holdout_meter_ids) if holdout_meter_ids else frozenset()
        )

        sm_files = list(sm_files)
        self.total_files = len(set(sm_files))

        if holdout_feeder_id is not None:
            if topology_path is None:
                raise ValueError("topology_path is required when holdout_feeder_id is set")
            from data.feeder_holdout import resolve_holdout_meter_ids
            _holdout_ids = resolve_holdout_meter_ids(topology_path, holdout_feeder_id)
            self._holdout_files = [
                f for f in sm_files
                if _extract_meter_id_from_filename(os.path.basename(f)) in _holdout_ids
            ]
            sm_files = [
                f for f in sm_files
                if _extract_meter_id_from_filename(os.path.basename(f)) not in _holdout_ids
            ]
        elif self._holdout_meter_ids_direct:
            # Phased-mode file-based holdout: exclude files whose meter IDs are in the set.
            self._holdout_files = [
                f for f in sm_files
                if _extract_meter_id_from_filename(os.path.basename(f)) in self._holdout_meter_ids_direct
            ]
            sm_files = [
                f for f in sm_files
                if _extract_meter_id_from_filename(os.path.basename(f)) not in self._holdout_meter_ids_direct
            ]
        else:
            self._holdout_files = []

        # mp.Value counters — shared memory scalars that survive DataLoader worker
        # pickling on Windows Python 3.13 (unlike Manager().dict() proxies).
        self._files_seen_ctr     = mp.Value('i', 0)
        self._files_completed_ctr = mp.Value('i', 0)
        self._files_skipped_ctr  = mp.Value('i', 0)

        self.files_seen = 0
        self.files_completed = 0

        self.train_dataset = StreamingParquetBuffer(
            sm_files=sm_files,
            replay_buffer_capacity=replay_buffer_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=window_spec,
            normalizer=normalizer,
            num_workers=num_workers,
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
            file_seen_counter=self._files_seen_ctr,
            file_completed_counter=self._files_completed_ctr,
            file_skipped_counter=self._files_skipped_ctr,
            skip_corrupt_files=skip_corrupt_files,
            max_skip_warnings=max_skip_warnings,
            output_aggregate_stream=self.output_aggregate_stream,
            sample_random=self.sample_random,
            nan_as_zero=nan_as_zero,
            phased_mode=self.phased_mode,
            meter_col=meter_col,
            feature_cols=feature_cols,
            seed=self.split_seed,
            split="train",
            split_ratios=self.split_ratios,
            max_value=max_value,
            max_hour_value=self.max_hour_value,
            batch_size=self.batch_size,
        )

        self.eval_loader = None
        self.test_loader = None
        self.aggregation = hourly_aggregation

    def setup(self) -> None:
        # Create the streaming dataset
        dataset = StreamingParquetBuffer(
            sm_files=self.data_files,
            replay_buffer_capacity=self.replay_buffer_capacity,
            window_spec=self.window_spec,
            normalizer=self.normalizer,
            num_workers=self.num_workers,
            hourly_aggregation=self.aggregation
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
