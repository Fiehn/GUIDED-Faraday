import math
import hashlib
import torch
import random
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, TensorDataset
from typing import Optional
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.normalizer import Normalizer

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
    
    def _push_to_buffer(self, item):
        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.replay_buffer_capacity
        self.current_size = min(self.current_size + 1, self.replay_buffer_capacity)

    def _sample_from_buffer(self):
        if self.current_size < self.min_buffer_size:
            raise ValueError(f"Buffer size {self.current_size} is less than the minimum required {self.min_buffer_size} for sampling.")
        if self.sample_random:
            idx = random.randrange(self.current_size)
            return self.buffer[idx]

        # Deterministic mode: always emit the newest inserted sample.
        idx = (self.ptr - 1) % self.replay_buffer_capacity
        return self.buffer[idx]

    def _extract_meter_id(self, file_path: str) -> int:
        if not self.phased_mode:
            raise ValueError("_extract_meter_id is only valid when phased_mode=True")
        basename = os.path.basename(file_path)
        stem = os.path.splitext(basename)[0].replace('sm_', '')
        try:
            return int(stem)
        except ValueError as e:
            raise ValueError(
                f"Could not parse phased meter id from filename '{basename}'. "
                "Expected pattern like 'sm_<id>.parquet' or '<id>.parquet'."
            ) from e

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

        time_col_name = next((col for col in file_cols if col.lower() in self.time_col_candidates), None)
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
            print(prepared.columns)
            raise ValueError(
                f"Unphased mode requires meter column '{self.meter_col}' in parquet data."
            )

        prepared = prepared[prepared[meter_col_name].notna()]
        prepared = prepared.sort_values([meter_col_name, "_timestamp"])
        return prepared

    def _impute_to_base_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Aggregate first so index labels align with the target base frequency.
        # Building full_index before resampling can misalign (e.g., :45 vs :00)
        # and reindex would then wipe out all values as NaN.
        if self.aggregation:
            df = df.resample(self.base_frequency).mean() # TODO: REMEMBER THIS 'guy'! we are using kw not kwh on the 15 min scale
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

        if worker_info is None:
            # Single-process data loading
            files_for_this_worker = list(self.sm_files) # copy the list
        else:
            # Multi-process data loading
            per_worker = int(math.ceil(len(self.sm_files) / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, len(self.sm_files))

            files_for_this_worker = self.sm_files[start_idx:end_idx]
        
        if self.sample_random:
            random.shuffle(files_for_this_worker)

        # === Lazy parquet streaming ===
        for file in files_for_this_worker:
            self._mark_file_seen(file)
            try:
                file_df, meter_id, meter_col_name = self._load_file_dataframe(file)
                if file_df is None:
                    continue

                sequences = self._extract_sequences(file_df, meter_id=meter_id, meter_col_name=meter_col_name)

                for seq, meter_id_t, timestamp_t in sequences:
                    # Apply normalization if normalizer is provided
                    if self.normalizer is not None:
                        # seq shape is [24, 3], broadcasting with [3,]
                        seq = self.normalizer.transform(seq)

                    # Keep model input as [F, L]
                    seq = seq.T

                    # Always emit aggregate tensor to keep tuple width fixed.
                    if self.output_aggregate_stream:
                        agg = seq.sum(dim=0, keepdim=True)
                    else:
                        agg = torch.empty((0, seq.shape[-1]), dtype=seq.dtype, device=seq.device)

                    # Push standardized sample tuple: (x, meter_id, timestamp, agg)
                    self._push_to_buffer((seq, meter_id_t, timestamp_t, agg))

                    # Yield only when the buffer is adequately mixed
                    if self.current_size >= self.min_buffer_size:
                        # The yield pauses the loop right here until PyTorch needs more data
                        yield self._sample_from_buffer()
            except Exception as e:
                if self.skip_corrupt_files and self._is_recoverable_file_error(e):
                    reason = f"{type(e).__name__}: {e}"
                    self._mark_file_skipped(file, reason)
                    self._warn_skipped_file(file, e)
                    continue
                raise
            finally:
                self._mark_file_completed(file)
    
    def _extract_sequences_for_meter(self, meter_df: pd.DataFrame, meter_id: int):
        sequences = []

        if self.window_spec is None:
            raise ValueError("window_spec must be provided for phased parquet streaming.")

        for feat in self.feature_cols:
            if feat not in meter_df.columns:
                meter_df[feat] = np.nan

        phase_df = meter_df[self.feature_cols].apply(pd.to_numeric, errors='coerce')
        phase_df = self._impute_to_base_frequency(phase_df)

        if phase_df.empty:
            return sequences

        if self.nan_as_zero:
            phase_df = phase_df.fillna(0.0)
        else:
            finite_mask = phase_df.notna().all(axis=1)
            phase_df = phase_df.loc[finite_mask]
            if phase_df.empty:
                return sequences

        timestamps = (phase_df.index.view('int64') // 10**9).astype(np.int64)
        values = phase_df.astype(np.float32).to_numpy()
        if len(values) < self.window_spec.input_len:
            return sequences

        for start in range(0, len(values) - self.window_spec.input_len + 1, self.window_spec.stride):
            end = start + self.window_spec.input_len
            
            # Window shape before transpose: [L, F]
            window = torch.tensor(values[start:end], dtype=torch.float32)

            # Per-window metadata (single values, not packed sequences)
            meter_id_t = torch.tensor(int(meter_id), dtype=torch.long)
            timestamp_t = torch.tensor(int(timestamps[start]), dtype=torch.long)

            sequences.append((window, meter_id_t, timestamp_t))
        
        return sequences

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
        grouped["_meter_id_int"] = grouped[meter_col_name].apply(self._meter_value_to_int).astype(np.int64)

        sequences = []
        for meter_value, meter_df in grouped.groupby("_meter_id_int", sort=False):
            meter_frame = meter_df.sort_values("_timestamp").set_index("_timestamp")
            meter_frame = meter_frame[~meter_frame.index.duplicated(keep='last')]
            sequences.extend(self._extract_sequences_for_meter(meter_frame, meter_id=int(meter_value)))
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
            pin_memory: bool = False,
            output_aggregate_stream: bool = False,
            skip_corrupt_files: bool = True,
            max_skip_warnings: int = 20,
            sample_random: bool = True,
            nan_as_zero: bool = False,
            phased_mode: bool = True,
            meter_col: str = "meter_number",
            feature_cols: list[str] | None = None,
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
        self.total_files = len(set(sm_files))

        # Manager-backed maps are shared across DataLoader workers.
        self._stats_manager = mp.Manager()
        self._file_seen_map = self._stats_manager.dict()
        self._file_completed_map = self._stats_manager.dict()
        self._file_skipped_map = self._stats_manager.dict()

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
            file_skipped_map=self._file_skipped_map,
            skip_corrupt_files=skip_corrupt_files,
            max_skip_warnings=max_skip_warnings,
            output_aggregate_stream=self.output_aggregate_stream,
            sample_random=self.sample_random,
            nan_as_zero=nan_as_zero,
            phased_mode=self.phased_mode,
            meter_col=meter_col,
            feature_cols=feature_cols,
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
            "skipped": int(len(self._file_skipped_map)),
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
        train_meter = None
        train_timestamps = None
        train_agg = None
        eval_x = None
        eval_meter = None
        eval_timestamps = None
        eval_agg = None
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

            if not isinstance(batch, (list, tuple)) or len(batch) != 4:
                raise ValueError(
                    "Streaming replay batch must be a 4-tuple: (x, meter_id, timestamps, agg)."
                )

            x, meter_batch, ts_batch, agg_batch = batch
            x = x.float()
            meter_batch = meter_batch.long()
            ts_batch = ts_batch.long()
            agg_batch = agg_batch.float()

            if train_x is None and target_train > 0:
                train_x = torch.empty((target_train, *x.shape[1:]), dtype=x.dtype)
                train_meter = torch.empty((target_train,), dtype=torch.long)
                train_timestamps = torch.empty((target_train,), dtype=torch.long)
                train_agg = torch.empty((target_train, *agg_batch.shape[1:]), dtype=agg_batch.dtype)
            if eval_x is None and target_eval > 0:
                eval_x = torch.empty((target_eval, *x.shape[1:]), dtype=x.dtype)
                eval_meter = torch.empty((target_eval,), dtype=torch.long)
                eval_timestamps = torch.empty((target_eval,), dtype=torch.long)
                eval_agg = torch.empty((target_eval, *agg_batch.shape[1:]), dtype=agg_batch.dtype)

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
            m_eval = meter_batch[eval_mask]
            t_eval = ts_batch[eval_mask]
            x_train = x[~eval_mask]
            m_train = meter_batch[~eval_mask]
            t_train = ts_batch[~eval_mask]
            agg_eval = agg_batch[eval_mask]
            agg_train = agg_batch[~eval_mask]

            if remaining_train > 0 and x_train.shape[0] > 0:
                take_train = min(remaining_train, x_train.shape[0])
                train_x[train_samples:train_samples + take_train].copy_(x_train[:take_train])
                train_meter[train_samples:train_samples + take_train].copy_(m_train[:take_train])
                train_timestamps[train_samples:train_samples + take_train].copy_(t_train[:take_train])
                train_agg[train_samples:train_samples + take_train].copy_(agg_train[:take_train])
                train_samples += int(take_train)

            if remaining_eval > 0 and x_eval.shape[0] > 0:
                take_eval = min(remaining_eval, x_eval.shape[0])
                eval_x[eval_samples:eval_samples + take_eval].copy_(x_eval[:take_eval])
                eval_meter[eval_samples:eval_samples + take_eval].copy_(m_eval[:take_eval])
                eval_timestamps[eval_samples:eval_samples + take_eval].copy_(t_eval[:take_eval])
                eval_agg[eval_samples:eval_samples + take_eval].copy_(agg_eval[:take_eval])
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
                train_meter[train_samples:train_samples + take].copy_(eval_meter[:take])
                train_timestamps[train_samples:train_samples + take].copy_(eval_timestamps[:take])
                train_agg[train_samples:train_samples + take].copy_(eval_agg[:take])
                train_samples += int(take)

        if train_x is not None:
            train_x = train_x[:train_samples]
        if train_meter is not None:
            train_meter = train_meter[:train_samples]
        if train_timestamps is not None:
            train_timestamps = train_timestamps[:train_samples]
        if train_agg is not None:
            train_agg = train_agg[:train_samples]
        if eval_x is not None:
            eval_x = eval_x[:eval_samples]
        if eval_meter is not None:
            eval_meter = eval_meter[:eval_samples]
        if eval_timestamps is not None:
            eval_timestamps = eval_timestamps[:eval_samples]
        if eval_agg is not None:
            eval_agg = eval_agg[:eval_samples]
        return train_x, train_meter, train_timestamps, train_agg, eval_x, eval_meter, eval_timestamps, eval_agg

    def _build_eval_loader(self, x_eval, meter_eval, timestamps_eval, agg_eval=None):
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

    def get_epoch_loaders(self, steps_per_epoch: int, eval_batches: int, refresh_eval: bool = False):
        target_train = max(1, int(steps_per_epoch)) * self.batch_size
        need_eval = int(eval_batches) > 0
        target_eval = 0

        if need_eval:
            if self.eval_loader is None or refresh_eval:
                eval_cache_batches = self.eval_cache_batches if self.eval_cache_batches is not None else eval_batches
                target_eval = max(1, int(eval_cache_batches)) * self.batch_size

        train_x, train_meter, train_timestamps, train_agg, eval_x, eval_meter, eval_timestamps, eval_agg = self._collect_cache(target_train, target_eval)
        if train_x is None or train_meter is None or train_timestamps is None or train_x.shape[0] == 0:
            return None, self.eval_loader

        train_ds = TensorDataset(train_x, train_meter, train_timestamps, train_agg)
        pin_train_memory = self.pin_memory and train_x.device.type == "cpu"
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_train_memory,
        )

        if target_eval > 0:
            self._build_eval_loader(eval_x, eval_meter, eval_timestamps, eval_agg)

        return train_loader, self.eval_loader

    def refresh_eval_cache(self, eval_batches: int):
        target_eval = max(1, int(eval_batches)) * self.batch_size
        _, _, _, _, eval_x, eval_meter, eval_timestamps, eval_agg = self._collect_cache(0, target_eval)
        self._build_eval_loader(eval_x, eval_meter, eval_timestamps, eval_agg)

    def get_eval_loader(self):
        return self.eval_loader

