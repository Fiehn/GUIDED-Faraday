import math
import hashlib
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

    # Rows per chunk when reading large unphased files (one file = all meters).
    # Phased files are one-meter-per-file and are read whole.
    _UNPHASED_CHUNK_ROWS = 250_000

    def __init__(self,
                 sm_files, 
                 replay_buffer_capacity: int = 10000, 
                 min_buffer_size: int = 2000, 
                 window_spec = None,
                 normalizer: Optional[Normalizer] = None,
                 num_workers: int = 0,
                 hourly_aggregation: bool = False,
                 aggregation_divisor: float = 1.0,
                 file_seen_map=None,
                 file_completed_map=None,
                 file_skipped_map=None,
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

        if phased_mode:
            self.feature_cols = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        else:
            if not feature_cols:
                raise ValueError("feature_cols must be provided when phased_mode=False")
            self.feature_cols = list(feature_cols)

        self.nan_as_zero = bool(nan_as_zero)
        self.phased_mode = bool(phased_mode)

        # Shared (manager-backed) maps to track file coverage across worker processes.
        self.file_seen_map = file_seen_map
        self.file_completed_map = file_completed_map
        self.file_skipped_map = file_skipped_map
        self.skip_corrupt_files = bool(skip_corrupt_files)
        self.max_skip_warnings = max(0, int(max_skip_warnings))
        self._skip_warning_count = 0
        self.output_aggregate_stream = bool(output_aggregate_stream)
        self.sample_random = bool(sample_random)

    def __getstate__(self):
        state = self.__dict__.copy()
        # threading.Lock is not picklable on Windows; drop it and recreate in the worker
        del state['_buffer_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._buffer_lock = threading.Lock()

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

    def _mark_file_skipped(self, file_path: str, reason: str):
        if self.file_skipped_map is not None:
            try:
                self.file_skipped_map[file_path] = reason[:300]
            except Exception:
                pass

    def _split_bucket(self, meter_id: int, timestamp_s: int) -> str:
        return csv_split_bucket(self.seed, self.split_ratios, meter_id, timestamp_s)

    def _keep_window(self, meter_id: int, timestamp_s: int) -> bool:
        if self.split is None:
            return True
        return self._split_bucket(meter_id, timestamp_s) == self.split

    def _is_recoverable_file_error(self, error: Exception) -> bool:
        if isinstance(error, OSError):
            return True

        arrow_exception = getattr(pa, "ArrowException", None)
        if arrow_exception is not None and isinstance(error, arrow_exception):
            return True

        msg = str(error).lower()
        tokens = (
            "corrupt",
            "snappy",
            "parquet",
            "decompression",
            "invalid",
            "out-of-bounds",
        )
        return any(token in msg for token in tokens)

    def _warn_skipped_file(self, file_path: str, error: Exception):
        if self.max_skip_warnings == 0:
            return

        if self._skip_warning_count < self.max_skip_warnings:
            print(
                "Warning: skipping unreadable parquet file "
                f"{file_path} ({type(error).__name__}: {error})"
            )
            self._skip_warning_count += 1
            if self._skip_warning_count == self.max_skip_warnings:
                print(
                    f"Warning: reached skip warning limit ({self.max_skip_warnings}) "
                    "for this worker. Suppressing further skip logs."
                )
    
    def _extract_meter_id(self, file_path: str) -> int:
        if not self.phased_mode:
            raise ValueError("_extract_meter_id is only valid when phased_mode=True")
        return _extract_meter_id_from_filename(os.path.basename(file_path))

    def _select_columns(self, file_cols):
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

        return cols_to_load, rename_map, time_col_name, meter_col_name

    def _prepare_time_index(self, df: pd.DataFrame, time_col_name: str | None, meter_col_name: str | None) -> pd.DataFrame:
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

        prepared["_timestamp"] = time_values
        prepared = prepared[~prepared["_timestamp"].isna()]

        if meter_col_name is None or meter_col_name not in prepared.columns:
            raise ValueError(
                f"Unphased mode requires meter column '{self.meter_col}' in parquet data."
            )

        prepared = prepared[prepared[meter_col_name].notna()]
        prepared = prepared.sort_values([meter_col_name, "_timestamp"])
        return prepared

    def _impute_to_base_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: REMEMBER THIS 'guy'! we are using kw not kwh on the 15 min scale
        return _impute_to_base_frequency(df, self.aggregation, self.base_frequency, self.aggregation_divisor)

    def _load_file_dataframe(self, file_path: str):
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
        # If this split has a zero ratio, yield nothing. Without this guard,
        # the while True below loops forever without yielding, deadlocking any
        # caller that expects StopIteration (e.g. val stream when train_ratio=1.0).
        if self.split is not None:
            _ratio_key = f"{self.split}_ratio"
            if float(self.split_ratios.get(_ratio_key, 1.0)) <= 0.0:
                return

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
    
    def _extract_sequences_for_meter(self, meter_df: pd.DataFrame, meter_id: int):
        """
        SHAPER:
          meter_df : [num_timesteps × num_phases] time-indexed DataFrame
          ->       : list of ([input_len, num_phases], [] scalar meter_id, [] scalar timestamp)
        """
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

class PhasedReplayDataloaders:
    """Builds train/eval DataLoaders from a shared replay stream with optional GPU eval caching.

    Standardized output contract for both train and eval loaders:
        (x, meter_id, timestamps, agg)

    where:
        x:          [B, F, L]
        meter_id:   [B]
        timestamps: [B]
        agg:        [B, 1, L] when available, else [B, 0, L]
    """
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
        else:
            self._holdout_files = []

        # Regular dicts — per-process copies in DataLoader workers (logging only).
        # mp.Manager().dict() proxies fail to reconnect on Windows+Python 3.13 when
        # workers are spawned, causing silent worker crashes.
        self._stats_manager = None
        self._file_seen_map = {}
        self._file_completed_map = {}
        self._file_skipped_map = {}

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
            file_seen_map=self._file_seen_map,
            file_completed_map=self._file_completed_map,
            file_skipped_map=self._file_skipped_map,
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
        self._train_stream_loader = None
        self._train_stream_iter = None
        self._val_stream_loader = None
        self._val_stream_iter = None
        self._test_stream_loader = None
        self._test_stream_iter = None
        self._stream_epoch_count = 0

        # Validation stream (used during training): deterministic ordering and deterministic buffer sampling.
        self.val_dataset = StreamingParquetBuffer(
            sm_files=sm_files,
            replay_buffer_capacity=replay_buffer_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=window_spec,
            normalizer=normalizer,
            num_workers=0,  # keep eval deterministic / simple
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
            file_seen_map=None,
            file_completed_map=None,
            file_skipped_map=None,
            skip_corrupt_files=skip_corrupt_files,
            max_skip_warnings=max_skip_warnings,
            output_aggregate_stream=self.output_aggregate_stream,
            sample_random=False,
            nan_as_zero=nan_as_zero,
            phased_mode=self.phased_mode,
            meter_col=meter_col,
            feature_cols=feature_cols,
            seed=self.split_seed,
            split="val",
            split_ratios=self.split_ratios,
            max_value=max_value,
            max_hour_value=self.max_hour_value,
            batch_size=self.batch_size,
        )

        # True test stream (never used during training unless explicitly requested).
        self.test_dataset = StreamingParquetBuffer(
            sm_files=sm_files,
            replay_buffer_capacity=replay_buffer_capacity,
            min_buffer_size=min_buffer_size,
            window_spec=window_spec,
            normalizer=normalizer,
            num_workers=0,
            hourly_aggregation=hourly_aggregation,
            aggregation_divisor=aggregation_divisor,
            file_seen_map=None,
            file_completed_map=None,
            file_skipped_map=None,
            skip_corrupt_files=skip_corrupt_files,
            max_skip_warnings=max_skip_warnings,
            output_aggregate_stream=self.output_aggregate_stream,
            sample_random=False,
            nan_as_zero=nan_as_zero,
            phased_mode=self.phased_mode,
            meter_col=meter_col,
            feature_cols=feature_cols,
            seed=self.split_seed,
            split="test",
            split_ratios=self.split_ratios,
            max_value=max_value,
            max_hour_value=self.max_hour_value,
            batch_size=self.batch_size,
        )

    def get_holdout_loader(self):
        """Return a DataLoader over holdout-feeder windows, or None if not configured."""
        if not self._holdout_files:
            return None
        ds = self.train_dataset
        holdout_ds = InMemoryPhasedDataset(
            sm_files=self._holdout_files,
            split="holdout",
            feature_cols=ds.feature_cols,
            split_ratios=self.split_ratios,
            split_seed=self.split_seed,
            window_len=ds.window_spec.input_len,
            stride=ds.window_spec.stride,
            aggregation=ds.aggregation,
            aggregation_divisor=ds.aggregation_divisor,
            max_hour_value=self.max_hour_value,
            max_value=ds.max_value,
            nan_as_zero=ds.nan_as_zero,
            normalizer=ds.normalizer,
            output_aggregate_stream=self.output_aggregate_stream,
            holdout_feeder_id=self._holdout_feeder_id,
            topology_path=self._topology_path,
        )
        return DataLoader(holdout_ds, batch_size=self.batch_size, shuffle=False)

    def get_file_progress(self):
        self.files_seen = len(self._file_seen_map)
        self.files_completed = len(self._file_completed_map)
        return {
            "seen": int(self.files_seen),
            "completed": int(self.files_completed),
            "skipped": int(len(self._file_skipped_map)),
            "total": int(self.total_files),
        }

    def _can_pin_memory(self, dataset) -> bool:
        """pin_memory only works on CPU tensors; disable if the dataset's normalizer emits GPU tensors."""
        norm = getattr(dataset, "normalizer", None)
        if norm is not None and hasattr(norm, "device"):
            if torch.device(norm.device).type != "cpu":
                return False
        return self.pin_memory

    def _make_stream(self, dataset):
        loader_kwargs = {
            "batch_size": None,
            "num_workers": self.num_workers,
            "pin_memory": self._can_pin_memory(dataset),
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **loader_kwargs)

    def _reset_train_stream(self):
        """Reset the training stream iterator and optionally recycle workers.

        Called proactively every N epochs (OOM prevention) and reactively on
        worker crashes. With persistent workers, we must NOT drop the loader
        while workers may have buffered batches: GC cleanup of a DataLoader
        whose workers are blocked on a full output queue deadlocks on
        Windows + Python 3.13. Instead, keep the loader alive and let the
        next iter() call signal workers to start a new epoch cleanly.
        """
        self._train_stream_iter = None
        if self.num_workers == 0:
            # Single-process mode: no workers to deadlock; safe to recreate.
            self._train_stream_loader = None

    def _get_train_stream_loader(self):
        if self._train_stream_loader is None:
            self._train_stream_loader = self._make_stream(self.train_dataset)
        return self._train_stream_loader

    def _get_val_stream_loader(self):
        if self._val_stream_loader is None:
            loader_kwargs = {
                "batch_size": None,
                "num_workers": 0,
                "pin_memory": self._can_pin_memory(self.val_dataset),
            }
            self._val_stream_loader = DataLoader(self.val_dataset, **loader_kwargs)
        return self._val_stream_loader

    def _get_test_stream_loader(self):
        if self._test_stream_loader is None:
            loader_kwargs = {
                "batch_size": None,
                "num_workers": 0,
                "pin_memory": self._can_pin_memory(self.test_dataset),
            }
            self._test_stream_loader = DataLoader(self.test_dataset, **loader_kwargs)
        return self._test_stream_loader

    def _next_train_stream_batch(self):
        loader = self._get_train_stream_loader()
        if self._train_stream_iter is None:
            self._train_stream_iter = iter(loader)

        try:
            return next(self._train_stream_iter)
        except StopIteration:
            self._train_stream_iter = iter(loader)
            try:
                return next(self._train_stream_iter)
            except StopIteration:
                return None
        except RuntimeError as e:
            # A DataLoader worker process died (OOM or unhandled exception).
            # Drop the broken loader explicitly so _get_train_stream_loader
            # spawns a fresh one — safe here because the workers are already dead.
            print(
                f"[data] WARNING: train stream worker crashed ({type(e).__name__}: {e}); "
                "restarting workers and retrying.",
                flush=True,
            )
            self._train_stream_iter = None
            self._train_stream_loader = None
            self._train_stream_iter = iter(self._get_train_stream_loader())
            return next(self._train_stream_iter)

    def _next_val_stream_batch(self):
        loader = self._get_val_stream_loader()
        if self._val_stream_iter is None:
            self._val_stream_iter = iter(loader)

        try:
            return next(self._val_stream_iter)
        except StopIteration:
            self._val_stream_iter = iter(loader)
            try:
                return next(self._val_stream_iter)
            except StopIteration:
                return None

    def _next_test_stream_batch(self):
        loader = self._get_test_stream_loader()
        if self._test_stream_iter is None:
            self._test_stream_iter = iter(loader)

        try:
            return next(self._test_stream_iter)
        except StopIteration:
            self._test_stream_iter = iter(loader)
            try:
                return next(self._test_stream_iter)
            except StopIteration:
                return None

    def _collect_stream_cache(self, which: str, target: int):
        """
        SHAPER:
          ->: ([num_samples, num_phases, input_len], [num_samples], [num_samples], [num_samples, 1, input_len])
              last tensor is [num_samples, 0, input_len] when output_aggregate_stream=False
        """
        import time as _time
        target = max(0, int(target))
        if target == 0:
            return None, None, None, None

        x_out = None
        meter_out = None
        ts_out = None
        agg_out = None
        samples = 0

        total_batches = max(1, (target + self.batch_size - 1) // self.batch_size)
        max_scan_batches = self.cache_scan_batches or max(30, total_batches * 40)

        _t0 = _time.perf_counter()
        _batches_read = 0

        if which == "train":
            next_batch = self._next_train_stream_batch
        elif which == "val":
            next_batch = self._next_val_stream_batch
        elif which == "test":
            next_batch = self._next_test_stream_batch
        else:
            raise ValueError(f"Unknown stream selector: {which}")

        for _ in range(max_scan_batches):
            batch = next_batch()
            _batches_read += 1
            if batch is None:
                break
            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError(
                    "Streaming replay batch must be a 4-tuple: (x, meter_id, timestamps, agg)."
                )

            x, meter_batch, ts_batch, agg_batch = batch
            x = x.float()
            meter_batch = meter_batch.long()
            ts_batch = ts_batch.long()
            agg_batch = agg_batch.float()

            if x_out is None:
                x_out = torch.empty((target, *x.shape[1:]), dtype=x.dtype)  # SHAPER: [num_samples, num_phases, input_len]
                meter_out = torch.empty((target,), dtype=torch.long)  # SHAPER: [num_samples]
                ts_out = torch.empty((target,), dtype=torch.long)  # SHAPER: [num_samples]
                agg_out = torch.empty((target, *agg_batch.shape[1:]), dtype=agg_batch.dtype)  # SHAPER: [num_samples, 1, input_len]

            remaining = max(0, target - samples)
            if remaining <= 0:
                break

            take = min(remaining, x.shape[0])
            if take > 0:
                assert x_out is not None and meter_out is not None and ts_out is not None and agg_out is not None
                x_out[samples:samples + take].copy_(x[:take])
                meter_out[samples:samples + take].copy_(meter_batch[:take])
                ts_out[samples:samples + take].copy_(ts_batch[:take])
                agg_out[samples:samples + take].copy_(agg_batch[:take])
                samples += int(take)

            if samples >= target:
                break

        elapsed = _time.perf_counter() - _t0
        if x_out is None or samples == 0:
            print(f"  [data] WARNING: no {which} samples collected in {elapsed:.1f}s", flush=True)
            return None, None, None, None

        return x_out[:samples], meter_out[:samples], ts_out[:samples], agg_out[:samples]


    def _build_eval_loader(self, x_eval, meter_eval, timestamps_eval, agg_eval=None):
        """
        SHAPER:
          x_eval         : [num_samples, num_phases, input_len]
          meter_eval     : [num_samples]
          timestamps_eval: [num_samples]
          agg_eval       : [num_samples, 1, input_len] or None
        """
        if (
            x_eval is None
            or meter_eval is None
            or timestamps_eval is None
            or x_eval.shape[0] == 0
        ):
            self.eval_loader = None
            return

        meter_id = meter_eval.long()
        timestamps = timestamps_eval.long()

        if self.eval_cache_device is not None:
            try:
                x_eval = x_eval.to(self.eval_cache_device)
                meter_id = meter_id.to(self.eval_cache_device)
                timestamps = timestamps.to(self.eval_cache_device)
                if agg_eval is not None:
                    agg_eval = agg_eval.to(self.eval_cache_device)
            except RuntimeError as e:
                # Fall back to CPU cache if GPU preloading runs out of memory.
                if "out of memory" in str(e).lower():
                    print("Warning: eval cache GPU transfer OOM; falling back to CPU eval cache.")
                    self.eval_cache_device = None
                else:
                    raise

        if agg_eval is not None:
            eval_ds = TensorDataset(x_eval, meter_id, timestamps, agg_eval)
        else:
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

    def _build_test_loader(self, x_test, meter_test, timestamps_test, agg_test=None):
        """
        SHAPER:
          x_test         : [num_samples, num_phases, input_len]
          meter_test     : [num_samples]
          timestamps_test: [num_samples]
          agg_test       : [num_samples, 1, input_len] or None
        """
        if (
            x_test is None
            or meter_test is None
            or timestamps_test is None
            or x_test.shape[0] == 0
        ):
            self.test_loader = None
            return
    
        meter_id = meter_test.long()
        timestamps = timestamps_test.long()

        if self.eval_cache_device is not None:
            try:
                x_test = x_test.to(self.eval_cache_device)
                meter_id = meter_id.to(self.eval_cache_device)
                timestamps = timestamps.to(self.eval_cache_device)
                if agg_test is not None:
                    agg_test = agg_test.to(self.eval_cache_device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Warning: test cache GPU transfer OOM; falling back to CPU test cache.")
                else:
                    raise

        if agg_test is not None:
            test_ds = TensorDataset(x_test, meter_id, timestamps, agg_test)
        else:
            test_ds = TensorDataset(x_test, meter_id, timestamps)
        pin_test_memory = self.pin_memory and x_test.device.type == "cpu"
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_test_memory,
        )

    def get_train_epoch_loader(self, steps_per_epoch: int):
        """Builds ONLY the training cache and returns the loader.

        SHAPER: each batch yields ([batch_size, num_phases, input_len], [batch_size], [batch_size], [batch_size, 1, input_len])
        """
        self._stream_epoch_count += 1
        if self.num_workers > 0 and self._stream_epoch_count % 10 == 0:
            self._reset_train_stream()

        target_train = max(1, int(steps_per_epoch)) * self.batch_size
        
        # Only collect the training stream
        train_x, train_meter, train_timestamps, train_agg = self._collect_stream_cache("train", target_train)  # SHAPER: [num_samples, num_phases, input_len], [num_samples], [num_samples], [num_samples, 1, input_len]
        
        if train_x is None or train_meter is None or train_timestamps is None or train_x.shape[0] == 0:
            return None

        train_ds = TensorDataset(train_x, train_meter, train_timestamps, train_agg)
        pin_train_memory = self.pin_memory and train_x.device.type == "cpu"
        
        generator = None
        if self.split_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.split_seed))
            
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_train_memory,
            generator=generator,
        )
        return train_loader
    
    def get_val_epoch_loader(self, eval_batches: int):
        """Builds ONLY the validation cache and returns the loader.

        SHAPER: each batch yields ([batch_size, num_phases, input_len], [batch_size], [batch_size], [batch_size, 1, input_len])
        """
        target_eval = max(1, int(eval_batches)) * self.batch_size
        
        # Only collect the validation stream
        eval_x, eval_meter, eval_timestamps, eval_agg = self._collect_stream_cache("val", target_eval)  # SHAPER: [num_samples, num_phases, input_len], [num_samples], [num_samples], [num_samples, 1, input_len]
        
        self._build_eval_loader(eval_x, eval_meter, eval_timestamps, eval_agg)
        return self.eval_loader

    def refresh_test_cache(self, test_batches: int):
        target_test = max(1, int(test_batches)) * self.batch_size
        x_test, m_test, t_test, a_test = self._collect_stream_cache("test", target_test)
        self._build_test_loader(x_test, m_test, t_test, a_test)

    def get_test_loader(self):
        if self.test_loader is not None:
            return self.test_loader

        # Collect eval_cache_batches × 10 test batches.  The old default of 1000
        # was proportioned for a 100% pass-rate dataset; with a 10% test split
        # the stream scans 10× more data and hangs on small datasets or slow IO.
        test_batches = max(5, (self.eval_cache_batches or 5)) * 10
        self.refresh_test_cache(test_batches=test_batches)

        return self.test_loader

# =============================================================================
# In-memory CSV dataset and dataloaders
# =============================================================================

class InMemoryCSVDataset(Dataset):
    """Loads all CSV-mode parquet files into RAM as fixed tensors.

    Faster than StreamingParquetBuffer for a single aggregated parquet
    (e.g. 200 MB hourly dataset).  Split assignment uses csv_split_bucket
    (SHA1 on seed|meter|timestamp) so train/val/test composition is identical
    to the streaming path for the same seed.

    Opt-in: only used when data.in_memory=true in the config.
    """

    def __init__(
        self,
        sm_files: list[str],
        split: str,
        split_ratios: dict,
        split_seed: int | None,
        max_hour_value: float | None,
        max_value: float | None,
        nan_as_zero: bool,
        normalizer: Optional[Normalizer],
        output_aggregate_stream: bool,
        meter_col: str,
        value_cols: list[str],
        window_len: int = 24,
        stride: int = 24,
        holdout_feeder_id: int | list[int] | None = None,
        topology_path: str | None = None,
    ):
        split = str(split).lower()

        if split == "holdout" and holdout_feeder_id is None:
            raise ValueError("split='holdout' requires holdout_feeder_id to be configured")
        if holdout_feeder_id is not None and topology_path is None:
            raise ValueError("topology_path is required when holdout_feeder_id is set")

        _holdout_meter_ids: frozenset[int]
        if holdout_feeder_id is not None:
            from data.feeder_holdout import resolve_holdout_meter_ids
            _holdout_meter_ids = resolve_holdout_meter_ids(topology_path, holdout_feeder_id)
        else:
            _holdout_meter_ids = frozenset()

        ts_candidates = ["timestamp_dst", "timestamp_utc", "timestamp"]

        all_x:          list[np.ndarray] = []
        all_meters:     list[np.ndarray] = []
        all_timestamps: list[np.ndarray] = []

        for fpath in sm_files:
            df = pd.read_parquet(fpath)

            # ---- Detect index layout ----------------------------------------
            if isinstance(df.index, pd.MultiIndex):
                df.index = df.index.remove_unused_levels()  # fix stale level codes in parquet subsets
                idx_names = list(df.index.names)
                meter_level = idx_names.index(meter_col) if meter_col in idx_names else 0
                meter_vals_flat = df.index.get_level_values(meter_level).to_numpy()
                ts_level = 1 if meter_level == 0 else 0
                ts_idx = df.index.get_level_values(ts_level)
            else:
                ts_col = next((c for c in ts_candidates if c in df.columns), None)
                if ts_col is None:
                    raise ValueError(f"No timestamp column found in {fpath}")
                df = df.sort_values([meter_col, ts_col])
                meter_vals_flat = df[meter_col].to_numpy()
                ts_idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

            if getattr(ts_idx, "tz", None) is None:
                ts_idx = ts_idx.tz_localize("UTC")

            # Unix seconds: avoid .asi8 unit ambiguity (pandas 2.x uses us not ns)
            naive = ts_idx.tz_convert("UTC").tz_localize(None)
            ts_unix_all = naive.to_numpy(dtype="datetime64[s]").view(np.int64)  # [N]

            # ---- Value columns ----------------------------------------------
            present = [c for c in value_cols if c in df.columns]
            if not present:
                continue
            vals_flat = (
                df[present[0]].to_numpy(dtype=np.float32)
                if len(present) == 1
                else df[present].to_numpy(dtype=np.float32).sum(axis=1)
            )

            # ---- Per-meter processing ---------------------------------------
            unique_meters, meter_starts = np.unique(meter_vals_flat, return_index=True)
            meter_ends = np.append(meter_starts[1:], len(vals_flat))

            for meter_id_val, m_start, m_end in zip(unique_meters, meter_starts, meter_ends):
                m_int_pre = int(meter_id_val)
                if split in ("train", "val") and m_int_pre in _holdout_meter_ids:
                    continue
                if split == "holdout" and m_int_pre not in _holdout_meter_ids:
                    continue

                m_vals = vals_flat[m_start:m_end].copy()
                m_ts   = ts_unix_all[m_start:m_end]
                n_rows = len(m_vals)

                if nan_as_zero:
                    m_vals = np.nan_to_num(m_vals, nan=0.0)

                # Midnight alignment (matching streaming path)
                first_aligned_idx = 0
                for tz_offset in (3600, 7200, 0):
                    hits = np.where((m_ts + tz_offset) % 86400 == 0)[0]
                    if len(hits) > 0:
                        first_aligned_idx = int(hits[0])
                        break

                if first_aligned_idx + window_len > n_rows:
                    continue

                all_starts_arr = np.arange(
                    first_aligned_idx, n_rows - window_len + 1, stride, dtype=np.intp
                )
                if len(all_starts_arr) == 0:
                    continue

                # Extract all candidate windows at once
                idx_mat  = all_starts_arr[:, None] + np.arange(window_len, dtype=np.intp)
                windows  = m_vals[idx_mat].astype(np.float32)   # [n_win, window_len]
                ts_wins  = m_ts[all_starts_arr]                  # [n_win]

                # Validity filters (matching streaming path)
                abs_vals = np.abs(windows)
                valid    = abs_vals.sum(axis=1) > 0              # non-zero
                if not nan_as_zero:
                    valid &= ~np.isnan(windows).any(axis=1)
                if max_hour_value is not None:
                    valid &= abs_vals.max(axis=1) <= float(max_hour_value)
                if max_value is not None:
                    valid &= abs_vals.sum(axis=1) <= float(max_value)

                windows = windows[valid]
                ts_wins = ts_wins[valid]
                if len(windows) == 0:
                    continue

                # Split assignment via SHA1(seed|meter|timestamp).
                # Holdout split bypasses SHA1 — all windows from holdout meters are kept.
                m_int = int(meter_id_val)
                if split == "holdout":
                    keep = np.ones(len(ts_wins), dtype=bool)
                else:
                    keep = np.array([
                        csv_split_bucket(split_seed, split_ratios, m_int, int(t)) == split
                        for t in ts_wins
                    ])

                windows = windows[keep]
                ts_wins = ts_wins[keep]
                if len(windows) == 0:
                    continue

                all_x.append(windows)
                all_meters.append(np.full(len(windows), m_int, dtype=np.int64))
                all_timestamps.append(ts_wins)

        # ---- Assemble tensors -----------------------------------------------
        agg_dim = 1 if output_aggregate_stream else 0
        if not all_x:
            self.x          = torch.zeros((0, 1, window_len), dtype=torch.float32)
            self.meter_ids  = torch.zeros((0,), dtype=torch.long)
            self.timestamps = torch.zeros((0,), dtype=torch.long)
            self.agg        = torch.zeros((0, agg_dim, window_len), dtype=torch.float32)
            return

        x_np = np.vstack(all_x)              # [N, window_len]
        m_np = np.concatenate(all_meters)    # [N]
        t_np = np.concatenate(all_timestamps)  # [N]

        # Apply normalizer (same per-value semantics as streaming path)
        if normalizer is not None and normalizer.fitted:
            x_2d = torch.from_numpy(x_np.reshape(-1, 1)).float()   # [N*window_len, 1]
            x_2d = normalizer.transform(x_2d)
            x_t  = x_2d.reshape(len(x_np), window_len).unsqueeze(1)  # [N, 1, window_len]
        else:
            x_t = torch.from_numpy(x_np).float().unsqueeze(1)        # [N, 1, window_len]

        self.x          = x_t
        self.meter_ids  = torch.from_numpy(m_np)
        self.timestamps = torch.from_numpy(t_np)
        self.agg = (
            self.x.clone()
            if output_aggregate_stream
            else torch.zeros((len(self.x), 0, window_len), dtype=self.x.dtype)
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.meter_ids[idx], self.timestamps[idx], self.agg[idx]


class InMemoryCSVDataloaders:
    """Drop-in replacement for PhasedReplayDataloaders when data.in_memory=true.

    Loads the full CSV-mode dataset into RAM once at construction time,
    then serves finite shuffled loaders per epoch — much faster than the
    streaming replay-buffer path for datasets that fit in RAM (~1 GB).

    Public interface matches PhasedReplayDataloaders:
        get_train_epoch_loader(steps_per_epoch) -> DataLoader | None
        get_val_epoch_loader(eval_batches)       -> DataLoader | None
        get_test_loader()                        -> DataLoader | None
        get_file_progress()                      -> dict
    """

    def __init__(
        self,
        *,
        sm_files: list[str],
        window_len: int = 24,
        stride: int = 24,
        batch_size: int = 256,
        split_ratios: dict | None = None,
        split_seed: int | None = None,
        max_hour_value: float | None = 20.0,
        max_value: float | None = None,
        nan_as_zero: bool = False,
        normalizer: Optional[Normalizer] = None,
        output_aggregate_stream: bool = False,
        meter_col: str = "meter_number",
        value_cols: list[str] | None = None,
        pin_memory: bool = True,
        eval_cache_device: Optional[torch.device] = None,
        max_passes: int | None = None,
        holdout_feeder_id: int | None = None,
        topology_path: str | None = None,
    ):
        self.batch_size        = int(batch_size)
        self.pin_memory        = bool(pin_memory)
        self.eval_cache_device = eval_cache_device
        self.split_seed        = None if split_seed is None else int(split_seed)
        self.max_passes        = None if max_passes is None else int(max_passes)
        self._holdout_feeder_id = holdout_feeder_id
        self._topology_path     = topology_path

        split_ratios = split_ratios or {}
        value_cols   = value_cols or ["hourly_energy_consumption"]

        _common = dict(
            sm_files=list(sm_files),
            split_ratios=split_ratios,
            split_seed=split_seed,
            max_hour_value=max_hour_value,
            max_value=max_value,
            nan_as_zero=nan_as_zero,
            normalizer=normalizer,
            output_aggregate_stream=output_aggregate_stream,
            meter_col=meter_col,
            value_cols=value_cols,
            window_len=window_len,
            stride=stride,
            holdout_feeder_id=holdout_feeder_id,
            topology_path=topology_path,
        )

        self._common = _common  # stored for get_holdout_loader
        self.train_dataset = InMemoryCSVDataset(split="train", **_common)
        self.val_dataset   = InMemoryCSVDataset(split="val",   **_common)
        self.test_dataset  = InMemoryCSVDataset(split="test",  **_common)
        self._test_loader: Optional[DataLoader] = None

        self._base_seed: int = self.split_seed if self.split_seed is not None else 0
        self._train_perm: list[int] = self._new_perm(len(self.train_dataset), 0)
        self._train_ptr: int = 0
        self._passes_completed: int = 0
        self._total_samples_seen: int = 0

    def _new_perm(self, n: int, pass_idx: int) -> list[int]:
        if n == 0:
            return []
        g = torch.Generator()
        g.manual_seed(self._base_seed + pass_idx * 1009)
        return torch.randperm(n, generator=g).tolist()

    def get_train_epoch_loader(self, steps_per_epoch: int) -> Optional[DataLoader]:
        if len(self._train_perm) == 0:
            return None
        # Completed a full pass — check cap, then reshuffle and start the next one
        if self._train_ptr >= len(self._train_perm):
            self._passes_completed += 1
            if self.max_passes is not None and self._passes_completed >= self.max_passes:
                return None
            self._train_perm = self._new_perm(len(self.train_dataset), self._passes_completed)
            self._train_ptr = 0
        start = self._train_ptr
        end   = min(start + steps_per_epoch * self.batch_size, len(self._train_perm))
        indices = self._train_perm[start:end]
        self._train_ptr = end
        self._total_samples_seen += len(indices)
        subset = Subset(self.train_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
        )

    def get_val_epoch_loader(self, eval_batches: int) -> Optional[DataLoader]:
        if len(self.val_dataset) == 0:
            return None
        n = min(eval_batches * self.batch_size, len(self.val_dataset))
        indices = list(range(n))
        return DataLoader(
            Subset(self.val_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
        )

    def get_test_loader(self) -> Optional[DataLoader]:
        if self._test_loader is None and len(self.test_dataset) > 0:
            self._test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
            )
        return self._test_loader

    def get_holdout_loader(self) -> Optional[DataLoader]:
        """Return a DataLoader over holdout-feeder windows, or None if not configured."""
        if self._holdout_feeder_id is None:
            return None
        ds = InMemoryCSVDataset(split="holdout", **self._common)
        if len(ds) == 0:
            print(f"Warning: holdout_feeder_id={self._holdout_feeder_id} is configured but no windows were found in the data files. Check that the feeder ID matches the topology and that the parquet files contain meters from this feeder.")
            return None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=self.pin_memory)

    def refresh_test_cache(self, test_batches: int = 0):
        pass  # in-memory: test data already loaded; no streaming cache to fill

    def get_file_progress(self) -> dict:
        n = len(self.train_dataset)
        return {
            "seen": self._train_ptr,
            "completed": self._passes_completed,
            "skipped": 0,
            "total": n,
            "passes": self._passes_completed + (self._train_ptr / n if n > 0 else 0.0),
        }


# =============================================================================
# In-memory phased dataset / dataloaders
# =============================================================================

class InMemoryPhasedDataset(Dataset):
    """Loads one-meter-per-file phased parquets into RAM as fixed tensors.

    Files are processed one at a time — raw DataFrames are discarded after
    window extraction to keep peak memory bounded (see ADR-0005).
    """

    def __init__(
        self,
        sm_files: list[str],
        split: str,
        feature_cols: list[str],
        split_ratios: dict,
        split_seed: int | None,
        window_len: int = 24,
        stride: int = 24,
        aggregation: bool = True,
        base_frequency: str = "1h",
        aggregation_divisor: float = 1.0,
        max_hour_value: float | None = None,
        max_value: float | None = None,
        nan_as_zero: bool = False,
        normalizer: Optional[Normalizer] = None,
        output_aggregate_stream: bool = False,
        holdout_feeder_id: int | list[int] | None = None,
        topology_path: str | None = None,
    ):
        split = str(split).lower()

        if split == "holdout" and holdout_feeder_id is None:
            raise ValueError("split='holdout' requires holdout_feeder_id to be configured")
        if holdout_feeder_id is not None and topology_path is None:
            raise ValueError("topology_path is required when holdout_feeder_id is set")

        if holdout_feeder_id is not None:
            from data.feeder_holdout import resolve_holdout_meter_ids
            self._holdout_meter_ids: frozenset[int] = resolve_holdout_meter_ids(
                topology_path, holdout_feeder_id
            )
        else:
            self._holdout_meter_ids = frozenset()

        all_x:          list[np.ndarray] = []  # each: [window_len, num_phases]
        all_agg:        list[np.ndarray] = []  # each: [window_len] (sum of phases)
        all_meters:     list[int]        = []
        all_timestamps: list[int]        = []

        for fpath in sm_files:
            df = pd.read_parquet(fpath)

            # Ensure time index
            if not isinstance(df.index, pd.DatetimeIndex):
                ts_candidates = ["timestamp_dst", "timestamp_utc", "timestamp"]
                ts_col = next((c for c in ts_candidates if c in df.columns), None)
                if ts_col is None:
                    continue
                df = df.set_index(ts_col)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

            if getattr(df.index, "tz", None) is None:
                df.index = df.index.tz_localize("UTC")

            meter_id = _extract_meter_id_from_filename(os.path.basename(fpath))

            # Holdout filtering: skip holdout meters in train/val; skip non-holdout in holdout split
            if split in ("train", "val") and meter_id in self._holdout_meter_ids:
                continue
            if split == "holdout" and meter_id not in self._holdout_meter_ids:
                continue

            # Select and type-cast feature columns
            for feat in feature_cols:
                if feat not in df.columns:
                    df[feat] = np.nan
            phase_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

            # Resample to base frequency
            phase_df = _impute_to_base_frequency(phase_df, aggregation, base_frequency, aggregation_divisor)
            if phase_df.empty:
                continue

            seqs = _extract_sequences_for_meter(
                phase_df, meter_id,
                window_len=window_len,
                stride=stride,
                max_value=max_value,
                max_hour_value=max_hour_value,
                nan_as_zero=nan_as_zero,
                split_seed=split_seed,
                split_ratios=split_ratios,
                split=None if split == "holdout" else split,
            )

            for window_t, meter_id_t, ts_t in seqs:
                # window_t: [window_len, num_phases]
                w_np = window_t.numpy()  # [window_len, num_phases]
                all_x.append(w_np)
                all_agg.append(w_np.sum(axis=1))  # [window_len] — sum of phases, before normalizer
                all_meters.append(int(meter_id_t.item()))
                all_timestamps.append(int(ts_t.item()))

            del df, phase_df, seqs

        agg_dim = 1 if output_aggregate_stream else 0
        if not all_x:
            self.x          = torch.zeros((0, len(feature_cols), window_len), dtype=torch.float32)
            self.agg        = torch.zeros((0, agg_dim, window_len), dtype=torch.float32)
            self.meter_ids  = torch.zeros((0,), dtype=torch.long)
            self.timestamps = torch.zeros((0,), dtype=torch.long)
            return

        # Stack: [N, window_len, num_phases]
        x_np = np.stack(all_x, axis=0)  # [N, window_len, num_phases]
        # Transpose to [N, num_phases, window_len]
        x_np = x_np.transpose(0, 2, 1).astype(np.float32)

        if normalizer is not None and normalizer.fitted:
            # Flatten to [N * num_phases * window_len, 1], transform, reshape back
            n, p, l = x_np.shape
            x_2d = torch.from_numpy(x_np.reshape(-1, 1)).float()
            x_2d = normalizer.transform(x_2d)
            x_t = x_2d.reshape(n, p, l)
        else:
            x_t = torch.from_numpy(x_np)

        self.x          = x_t
        self.meter_ids  = torch.tensor(all_meters, dtype=torch.long)
        self.timestamps = torch.tensor(all_timestamps, dtype=torch.long)

        if output_aggregate_stream:
            agg_np = np.stack(all_agg, axis=0).astype(np.float32)  # [N, window_len]
            self.agg = torch.from_numpy(agg_np).unsqueeze(1)       # [N, 1, window_len]
        else:
            self.agg = torch.zeros((len(self.x), 0, window_len), dtype=self.x.dtype)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.meter_ids[idx], self.timestamps[idx], self.agg[idx]


class InMemoryPhasedDataloaders:
    """Drop-in replacement for PhasedReplayDataloaders when data.mode=phased and data.in_memory=true.

    Public interface matches InMemoryCSVDataloaders:
        get_train_epoch_loader(steps_per_epoch) -> DataLoader | None
        get_val_epoch_loader(eval_batches)       -> DataLoader | None
        get_test_loader()                        -> DataLoader | None
        get_file_progress()                      -> dict
    """

    def __init__(
        self,
        *,
        sm_files: list[str],
        feature_cols: list[str],
        window_len: int = 24,
        stride: int = 24,
        batch_size: int = 256,
        aggregation: bool = True,
        base_frequency: str = "1h",
        aggregation_divisor: float = 1.0,
        split_ratios: dict | None = None,
        split_seed: int | None = None,
        max_hour_value: float | None = 20.0,
        max_value: float | None = None,
        nan_as_zero: bool = False,
        normalizer: Optional[Normalizer] = None,
        output_aggregate_stream: bool = False,
        pin_memory: bool = True,
        eval_cache_device: Optional[torch.device] = None,
        holdout_feeder_id: Optional[int] = None,
        topology_path: Optional[str] = None,
    ):
        self.batch_size        = int(batch_size)
        self.pin_memory        = bool(pin_memory)
        self.eval_cache_device = eval_cache_device
        self.split_seed        = None if split_seed is None else int(split_seed)
        self._holdout_feeder_id = holdout_feeder_id

        split_ratios = split_ratios or {}

        self._common = dict(
            sm_files=list(sm_files),
            feature_cols=list(feature_cols),
            split_ratios=split_ratios,
            split_seed=split_seed,
            window_len=window_len,
            stride=stride,
            aggregation=aggregation,
            base_frequency=base_frequency,
            aggregation_divisor=aggregation_divisor,
            max_hour_value=max_hour_value,
            max_value=max_value,
            nan_as_zero=nan_as_zero,
            normalizer=normalizer,
            output_aggregate_stream=output_aggregate_stream,
            holdout_feeder_id=holdout_feeder_id,
            topology_path=topology_path,
        )

        self.train_dataset = InMemoryPhasedDataset(split="train", **self._common)
        self.val_dataset   = InMemoryPhasedDataset(split="val",   **self._common)
        self.test_dataset  = InMemoryPhasedDataset(split="test",  **self._common)
        self._test_loader: Optional[DataLoader] = None

        self._base_seed: int = self.split_seed if self.split_seed is not None else 0
        self._train_perm: list[int] = self._new_perm(len(self.train_dataset), 0)
        self._train_ptr: int = 0
        self._passes_completed: int = 0
        self._total_samples_seen: int = 0

    def _new_perm(self, n: int, pass_idx: int) -> list[int]:
        if n == 0:
            return []
        g = torch.Generator()
        g.manual_seed(self._base_seed + pass_idx * 1009)
        return torch.randperm(n, generator=g).tolist()

    def get_train_epoch_loader(self, steps_per_epoch: int) -> Optional[DataLoader]:
        if len(self._train_perm) == 0:
            return None
        if self._train_ptr >= len(self._train_perm):
            self._passes_completed += 1
            self._train_perm = self._new_perm(len(self.train_dataset), self._passes_completed)
            self._train_ptr = 0
        start = self._train_ptr
        end   = min(start + steps_per_epoch * self.batch_size, len(self._train_perm))
        indices = self._train_perm[start:end]
        self._train_ptr = end
        self._total_samples_seen += len(indices)
        subset = Subset(self.train_dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=self.pin_memory)

    def get_val_epoch_loader(self, eval_batches: int) -> Optional[DataLoader]:
        if len(self.val_dataset) == 0:
            return None
        n = min(eval_batches * self.batch_size, len(self.val_dataset))
        return DataLoader(Subset(self.val_dataset, list(range(n))),
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=self.pin_memory)

    def get_test_loader(self) -> Optional[DataLoader]:
        if self._test_loader is None and len(self.test_dataset) > 0:
            self._test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=0,
                                           pin_memory=self.pin_memory)
        return self._test_loader

    def get_holdout_loader(self) -> Optional[DataLoader]:
        if self._holdout_feeder_id is None:
            return None
        ds = InMemoryPhasedDataset(split="holdout", **self._common)
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, pin_memory=self.pin_memory)

    def refresh_test_cache(self, test_batches: int = 0):
        pass  # in-memory: test data already loaded; no streaming cache to fill

    def get_file_progress(self) -> dict:
        n = len(self.train_dataset)
        return {
            "seen": self._train_ptr,
            "completed": self._passes_completed,
            "skipped": 0,
            "total": n,
            "passes": self._passes_completed + (self._train_ptr / n if n > 0 else 0.0),
        }
