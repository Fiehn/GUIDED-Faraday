from __future__ import annotations
import gc
import os
import json
from dataclasses import dataclass
import sys
from typing import Optional, Sequence, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.normalizer import Normalizer



def map_meter_ids_to_feeder_ids(
                                meter_ids: Sequence,
                                meter_to_feeder: dict[str, Optional[str]],
                                ) -> np.ndarray:
    """
    Map a sequence of meter ids to feeder ids using a pre-loaded mapping.
    
    Args: 
        meter_ids: A sequence of meter identifiers to be mapped.
        meter_to_feeder: A dictionary mapping meter identifiers (as strings) to feeder identifiers.
    
    Returns:
        A numpy array of feeder identifiers corresponding to the input meter_ids. If a meter_id does not have a mapping, None is used.
    """
    feeder_ids = []
    for meter in meter_ids:
        if isinstance(meter, np.generic):
            meter = meter.item()
        feeder_ids.append(meter_to_feeder.get(str(meter), None))
    return np.asarray(feeder_ids, dtype=object)

# read and preprocess 
class SmartMeterCSV:
    """
    Loads and preprocesses SM data from CSV into a MultiIndex DataFrame:
    
    Attributes: 
        index: (meter_number, timestamp_utc). 
        columns include consumption/production. 
    """

    def __init__(
        self,
        data_dir: str,                           # directory where CSV files are located
        *,
        meter_col: str,                          # name of the meter identifier column
        timestamp_dst_col: str,                  # name of the DST timestamp column
        timestamp_utc_col: Optional[str] = None, # UTC column – loaded but never used; omit to save I/O
        value_cols: Sequence[str],               # names of the value columns (e.g., consumption, production)
        nan_as_zero: bool = True,                # whether to replace NaN with 0
    ): 
        self.data_dir = data_dir
        self.meter_col = meter_col
        self.timestamp_dst_col = timestamp_dst_col
        self.timestamp_utc_col = timestamp_utc_col
        self.value_cols = list(value_cols)
        self.nan_as_zero = nan_as_zero

        self.df: Optional[pd.DataFrame] = None

    def load(self, 
             filename: str
             ) -> pd.DataFrame:
        """
        Load CSV (or a Parquet cache thereof) into a MultiIndex DataFrame.

        On the first call the 3 GB CSV is chunk-read and written to a
        _parquet_cache/<stem>.parquet file (zstd-compressed
        smaller than the CSV and faster to read).  Every subsequent
        call simply reads the Parquet file, skipping the expensive CSV
        parse entirely.  The cache is invalidated automatically whenever
        the source CSV is newer than the cached file.

        Args: 
            filename: The name of the CSV file to load (should be located in data_dir).
       
         Returns:
            A pandas DataFrame with a MultiIndex of (meter_id, timestamp_dst) and columns for the specified value_cols.
        """
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {filename} not found in directory {self.data_dir}")

        cache_dir    = os.path.join(self.data_dir, "_parquet_cache")
        parquet_path = os.path.join(cache_dir, os.path.splitext(filename)[0] + ".parquet")

        # --- fast path: read from cache ---
        if (
            os.path.exists(parquet_path)
            and os.path.getmtime(parquet_path) >= os.path.getmtime(path)
        ):
            self.df = pd.read_parquet(parquet_path, engine="pyarrow")
            return self.df

        # --- slow path: parse CSV in chunks, build Parquet cache ---
        columns_to_load = [self.meter_col, self.timestamp_dst_col] + self.value_cols
        # Include UTC column only when explicitly requested
        if self.timestamp_utc_col is not None:
            columns_to_load.append(self.timestamp_utc_col)

        dtype_map = {
            self.meter_col: "int32",
            **{col: "float32" for col in self.value_cols},
        }

        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            usecols=columns_to_load,
            dtype=dtype_map,
            chunksize=200_000,
        ):
            chunk[self.timestamp_dst_col] = pd.to_datetime(chunk[self.timestamp_dst_col], format='ISO8601')
            for c in self.value_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            if self.nan_as_zero:
                for c in self.value_cols:
                    chunk[c] = chunk[c].fillna(0.0)
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        del chunks  # free intermediate memory before the MultiIndex sort

        df.set_index([self.meter_col, self.timestamp_dst_col], inplace=True)
        df.sort_index(inplace=True)

        # Persist as Parquet cache for future runs
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")

        self.df = df
        return df

    # returns all unique meter IDs in the index
    def meter_ids(self) -> List[int]:
        """
        Get a list of all unique meter IDs present in the loaded DataFrame.
        
        Returns:
            A list of unique meter identifiers extracted from the DataFrame index.
        """
        self._require_loaded()
        return self.df.index.get_level_values(self.meter_col).unique().tolist()

    # returns DataFrame for a single meter ID
    def meter_frame(self,
                     meter_id: int) -> pd.DataFrame:
        """
        Get a DataFrame containing all rows for a specific meter ID, sorted by timestamp.
        
        Args: 
            meter_id: The identifier of the meter for which to retrieve the data.
        
        Returns:
            A pandas DataFrame containing all rows for the specified meter ID, indexed by timestamp and sorted in ascending order.
        """
        self._require_loaded()
        return self.df.xs(meter_id, level=self.meter_col).sort_index()

    # prints basic stats
    def examine(self) -> None:
        """
        Print basic information about the loaded DataFrame, including data types, summary statistics, and missing value counts.
        This method requires that the data has been loaded into the DataFrame. If the data is not loaded, it will raise a ValueError.
        """
        self._require_loaded()
        print("Data Info:")
        print(self.df.info())
        print("\nData Description:")
        print(self.df.describe())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def _require_loaded(self) -> None:
        """
        Utility method to check if the DataFrame has been loaded before performing operations that require data access.
        
        Raises:
            ValueError: If the DataFrame is not loaded (i.e., self.df is None).
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call .load(filename) first.")



# Dataset: sliding windows per meter
@dataclass(frozen=True)
class WindowSpec:
    """
    Specification for how to window the time series data for model input.
    """
    input_len: int # length of input window - how many past hours go into the model as x
    stride: int = 1 # how many hours to move the window each step

# need fixed window dataset per meter, an PyTorch Dataset needs samples of same shape
class SmartMeterWindowDataset(Dataset):
    """
    Convert a single meter's time series into x. 
    Assumes regular hourly sampling, so if timestamps are missing, it still windows by row order.
    
    Attributes:
        feature_cols: which columns to include in the x tensor (e.g. consumption, production)
        window: how to slice the time series into windows (input_len, stride)
        drop_nan_windows: whether to drop windows that contain any NaN values (default: True)
        dtype: the torch dtype for the x tensor (default: torch.float32)
    """

    def __init__(
        self,
        meter_df: pd.DataFrame,
        meter_id: int = 0,
        *,
        feature_cols: Sequence[str],
        window: WindowSpec,
        drop_nan_windows: bool = True,
        dtype: torch.dtype = torch.float32,
        normalizer = None,
    ):
        self.feature_cols = list(feature_cols) 
        self.window = window
        self.drop_nan_windows = drop_nan_windows
        self.dtype = dtype

        values_np = meter_df[self.feature_cols].to_numpy(dtype=np.float32)  # (T, F)

        if normalizer is not None and hasattr(normalizer, 'transform'):
            values_t = normalizer.transform(values_np)
            if isinstance(values_t, np.ndarray):
                values_t = torch.from_numpy(values_t)
            self.values = values_t.cpu().to(dtype)
        else:
            self.values = torch.from_numpy(values_np).to(dtype)   # (T, F)

        self.meter_id = torch.tensor(meter_id, dtype=torch.long)
        # Unix seconds fit in int32 for data up to year ~2038; halves memory vs int64
        self.times = torch.from_numpy(
            (meter_df.index.view(np.int64) // 10**9).astype(np.int32)
        )
        self.starts = self._compute_starts(drop_nan_windows)
        
    # determines which window start positions are allowed
    def _compute_starts(self, 
                        drop_nan: bool) -> np.ndarray:
        """
        Compute the valid starting indices for sliding windows based on the window specification and NaN handling.
        The method calculates the total number of time steps (T) in the meter's data and determines the maximum valid starting index for a window of the specified input length (L). It then generates potential starting indices based on the stride. If drop_nan is True, it filters out any starting indices where the corresponding window contains NaN values.

        Args:
            drop_nan: A boolean flag indicating whether to exclude windows that contain any NaN values.

        Returns:
            A numpy array of valid starting indices for the sliding windows. If drop_nan is True, only indices corresponding to windows without NaN values are included.
        """
        T = self.values.shape[0]
        L = self.window.input_len
        stride = self.window.stride

        max_start = T - L
        if max_start < 0:
            return torch.empty(0, dtype=torch.long)

        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        if not drop_nan:
            return starts

        good = [s for s in starts.tolist()
                if not self.values[s: s + L].isnan().any().item()]
        return torch.tensor(good, dtype=torch.long)

    # number of windows
    def __len__(self) -> int:
        return len(self.starts)

    # get x at index idx so x: [B, input_len, F]
    def __getitem__(self, 
                    idx: int) -> torch.Tensor:
        """
        Retrieve the input window (x), meter_id, and timestamp for a given index. The method calculates the starting index for the window based on the pre-computed valid starting positions. It then slices the values tensor to extract the window of data corresponding to the input length specified in the WindowSpec. The method also retrieves the meter_id and the timestamp corresponding to the start of the window.
        
        Args:
            idx: The index of the window to retrieve. This index corresponds to the position in the list of valid starting indices for the sliding windows.
        
        Returns:
            A tuple containing:
                - x: A tensor of shape [input_len, F] representing the input window of features for the specified index.
                - meter_id: A tensor containing the meter identifier associated with the window.
                - timestamp_unix: A tensor containing the Unix timestamp corresponding to the start of the window.
        """
        s = int(self.starts[idx])
        L = self.window.input_len
        x = self.values[s : s + L]
        timestamp_unix = self.times[s]
        
        return x, self.meter_id, timestamp_unix


# build DataLoaders for train/val/test splits (wraps full pipeline)
class SmartMeterDataModule:
    """
    Loads/preprocesses data, then splits meters into train/val/test and creates window datasets + dataloaders. 
    Split is done by meter_id.
    
    Attributes: 
        loader: SmartMeterCSV instance for loading/parsing CSV data into DataFrames
        filename: which CSV file to load (should be located in loader's data_dir)
        feature_cols: which columns to include in the x tensor (e.g. consumption, production)
        window: how to slice the time series into windows (input_len, stride)
        batch_size: how many windows per batch in the DataLoader
        num_workers: how many subprocesses to use for data loading in the DataLoader
        shuffle_train: whether to shuffle the training data in the DataLoader
        seed: random seed for reproducibility of train/val/test split
        train_frac: fraction of meters to use for training (default: 0.8)
        val_frac: fraction of meters to use for validation (default: 0.1)
        max_meters: optional limit on number of meters to load (for quick testing/debugging)
        normalizer: optional Normalizer instance to fit on training data and apply to all splits
        conditioner: optional Conditioner instance to fit on training data and apply to all splits
    """

    def __init__(
        self,
        data_dir: str,
        filename: str,
        *,
        meter_col: str,
        timestamp_dst_col: str,
        timestamp_utc_col: Optional[str] = None,  # not used in tensors; omit to save I/O
        feature_cols: Sequence[str] = (),
        window: WindowSpec = WindowSpec(input_len=24),
        batch_size: int = 64,
        num_workers: int = min(4, (os.cpu_count() or 1) // 2),  # auto-detect; override per run
        shuffle_train: bool = True,
        seed: int = 0,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        max_meters: Optional[int] = None,
        normalizer = None,
        conditioner = None,
    ):
        self.loader = SmartMeterCSV(
            data_dir=data_dir,
            meter_col=meter_col,
            timestamp_dst_col=timestamp_dst_col,
            timestamp_utc_col=timestamp_utc_col,
            value_cols=feature_cols,
        )
        self.filename = filename
        self.feature_cols = list(feature_cols)
        self.window = window
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.max_meters = max_meters

        self.normalizer = normalizer
        self.conditioner = conditioner

        self._train_ds = None 
        self._val_ds = None
        self._test_ds = None

    def setup(self) -> None:
        """
        Load and preprocess the data, then split meters into train/val/test and create window datasets for each split.
        """
        self.loader.load(self.filename)
        meter_ids = self.loader.meter_ids()

        # Limit number of meters if max_meters is specified
        if self.max_meters is not None:
            meter_ids = meter_ids[:self.max_meters]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(meter_ids)

        n = len(meter_ids)
        if n == 0:
            self.train_ids, self.val_ids, self.test_ids = [], [], []
        else:
            n_train = max(1, int(n * self.train_frac))
            n_val = max(0, int(n * self.val_frac))

            # Ensure we leave at least one meter for test when possible
            if n >= 2 and (n_train + n_val) >= n:
                # Reduce validation first, then training if needed
                n_val = max(0, n - n_train - 1)
                if n_train >= n:
                    n_train = n - 1

        self.train_ids  = meter_ids[:n_train]
        self.val_ids    = meter_ids[n_train : n_train + n_val]
        self.test_ids   = meter_ids[n_train + n_val :]

        # fit normalizer if provided
        
        self._fit_normalizer()
        self._fit_conditioner()

        self._train_ds  = self._make_dataset(self.train_ids)
        self._val_ds    = self._make_dataset(self.val_ids)
        self._test_ds   = self._make_dataset(self.test_ids)

        # Free the raw DataFrame now that all splits are converted to tensors.
        # This halves peak memory when the DataModule stays alive during training.
        self.loader.df = None
        gc.collect()

    def _yield_train_batches(self):
        """
        Yields batches of preprocessed (x, meter_id, timestamps) for training data.
        """
        L = self.window.input_len
        for mid in self.train_ids:
            meter_df = self.loader.meter_frame(mid)
            if meter_df.empty:
                continue

            values_np = meter_df[self.feature_cols].to_numpy(dtype=np.float32)
            times_np  = (meter_df.index.view(np.int64) // 10**9).astype(np.int32)

            if self.normalizer is not None and hasattr(self.normalizer, 'transform'):
                values_t = self.normalizer.transform(values_np)
                if isinstance(values_t, np.ndarray):
                    values_t = torch.from_numpy(values_t)
                values_t = values_t.cpu().to(torch.float32)
            else:
                values_t = torch.from_numpy(values_np)

            T = values_t.shape[0]
            if T < L:
                continue

            starts_np = np.arange(0, T - L + 1, self.window.stride, dtype=np.int64)
            valid     = [s for s in starts_np.tolist()
                         if not values_t[s: s + L].isnan().any().item()]
            if not valid:
                continue

            starts_t  = torch.tensor(valid, dtype=torch.long)
            idx       = starts_t.unsqueeze(1) + torch.arange(L, dtype=torch.long)
            x_windows = values_t[idx]

            meter_t = torch.full((len(valid),), mid, dtype=torch.long)
            times_t = torch.from_numpy(times_np)[starts_t]

            for i in range(0, len(x_windows), self.batch_size):
                yield x_windows[i:i+self.batch_size], meter_t[i:i+self.batch_size], times_t[i:i+self.batch_size]

    def _fit_conditioner(self):
        """
        Fit conditioner on training meters only, streamed one meter at a time.
        If the conditioner is already fitted (e.g. pre-fitted across multiple
        files) this method is a no-op, so the existing state is preserved.
        """
        if self.conditioner is None or "user_profile" not in self.conditioner.get_condition_tags():
            return  # nothing to fit
        
        user_profiler = self.conditioner.condition_items["user_profile"]
        
        if getattr(user_profiler, 'is_fitted', False):
            print("[Conditioner] UserProfiler already fitted. Performing incremental update...")
            for x_batch, m_batch, _ in self._yield_train_batches():
                user_profiler.update_model_incremental(x_batch, m_batch)
        else:
            print("[Conditioner] Fitting UserProfiler via multi-pass streaming...")
            class StreamLoader:
                def __init__(self, dm):
                    self.dm = dm
                def __iter__(self):
                    return self.dm._yield_train_batches()
                
            user_profiler.fit(StreamLoader(self))

    def _fit_normalizer(self):
        """
        Fit normalizer on training meters only, streamed one meter at a time.

        If the normalizer is already fitted (e.g. pre-fitted across multiple
        files via fit_normalizer_on_files()) this method is a no-op, so the
        existing stats are preserved.

        Values are streamed through fit_incremental() so peak RAM is O(one
        meter's data) regardless of training split size.
        """
        if self.normalizer is None or not hasattr(self.normalizer, 'fit'):
            return  # nothing to fit
        if getattr(self.normalizer, 'fitted', False):
            return  # already fitted externally — respect those stats

        if hasattr(self.normalizer, 'fit_incremental') and hasattr(self.normalizer, 'finalise_fit'):
            any_data = False
            for mid in self.train_ids:
                meter_df = self.loader.meter_frame(mid)
                values   = meter_df[self.feature_cols].to_numpy(dtype=np.float32)
                if values.size == 0:
                    continue
                self.normalizer.fit_incremental(values)
                any_data = True
            if not any_data:
                raise ValueError("No training data available to fit normalizer.")
            self.normalizer.finalise_fit()
        else:
            # Bulk fallback for normalizers without the streaming API
            all_values: list = []
            for mid in self.train_ids:
                meter_df = self.loader.meter_frame(mid)
                values   = meter_df[self.feature_cols].to_numpy(dtype=np.float32)
                if values.size == 0:
                    continue
                all_values.append(values)
            if not all_values:
                raise ValueError("No training data available to fit normalizer.")
            self.normalizer.fit(np.concatenate(all_values, axis=0))

    def _make_dataset(self, 
                      meter_ids: Sequence[int]) -> TensorDataset:
        """
        Pre-compute all windows for the given meters into a single TensorDataset.

        Pre-flattening means __getitem__ is an O(1) tensor index rather than
        an O(log N_meters) binary search through a ConcatDataset offset list.
        All windows are held in RAM, but their total size is small compared
        to the raw CSV (float32 windows only; no duplicate timestamp storage).

        Args: 
            meter_ids: A sequence of meter identifiers for which to create the dataset.
        
        Returns:
            A TensorDataset containing all valid windows for the specified meters, along with their corresponding meter IDs and timestamps. The dataset will have the following structure:
                - x: A tensor of shape [N, input_len, F] containing the input windows for all meters.
                - meter_id: A tensor of shape [N] containing the meter identifiers corresponding to each window.
                - timestamp_unix: A tensor of shape [N] containing the Unix timestamps corresponding to the start of each window.
                - conditions (optional): If a conditioner is provided, an additional tensor of shape [N, condition_dim] containing the conditions for each window.
        """
        L = self.window.input_len
        F = len(self.feature_cols)

        all_x:      list[torch.Tensor] = []
        all_cond:   list[torch.Tensor] = []
        all_meter:  list[torch.Tensor] = []
        all_times:  list[torch.Tensor] = []

        for mid in meter_ids:
            meter_df = self.loader.meter_frame(mid)
            if meter_df.empty:
                continue

            values_np = meter_df[self.feature_cols].to_numpy(dtype=np.float32)  # (T, F)
            times_np  = (meter_df.index.view(np.int64) // 10**9).astype(np.int32)

            # Normalise
            if self.normalizer is not None and hasattr(self.normalizer, 'transform'):
                values_t = self.normalizer.transform(values_np)
                if isinstance(values_t, np.ndarray):
                    values_t = torch.from_numpy(values_t)
                values_t = values_t.cpu().to(torch.float32)
            else:
                values_t = torch.from_numpy(values_np)  # (T, F)

            T = values_t.shape[0]
            if T < L:
                continue

            # Build valid window start indices (drop windows with NaN)
            starts_np = np.arange(0, T - L + 1, self.window.stride, dtype=np.int64)
            valid     = [s for s in starts_np.tolist()
                         if not values_t[s: s + L].isnan().any().item()]
            if not valid:
                continue

            starts_t  = torch.tensor(valid, dtype=torch.long)
            # Vectorised window extraction: (W, L, F)
            idx       = starts_t.unsqueeze(1) + torch.arange(L, dtype=torch.long)  # (W, L)
            x_windows = values_t[idx]                                                # (W, L, F)

            times_t = torch.from_numpy(times_np)

            # Conditioner
            meter_t = torch.full((len(valid),), mid, dtype=torch.long)
            time_windows = times_t[starts_t]
            if self.conditioner is not None:
                c_windows = self.conditioner.get_conditions(
                    x=x_windows, 
                    meter_id=meter_t, 
                    timestamps=time_windows
                )
                all_cond.append(c_windows)

            all_x.append(x_windows)
            all_meter.append(meter_t)
            all_times.append(time_windows)

        if not all_x:
            # Handle empty datasets safely by returning a TensorDataset with the correct column structure but zero rows.
            tensors = [
                torch.empty(0, L, F, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.int32)
            ]
            if self.conditioner is not None:
                tensors.insert(1, torch.empty(0, self.conditioner.get_condition_dim(), dtype=torch.float32))
            else:
                tensors.insert(1, torch.empty(0, 0, dtype=torch.float32))
            return TensorDataset(*tensors)

        tensors = [
            torch.cat(all_x,     dim=0),  # (N, L, F)
            torch.cat(all_meter, dim=0),  # (N,)
            torch.cat(all_times, dim=0),  # (N,)
        ]
        
        # Insert conditions as the second element if they exist, or empty tensor to maintain 4-tuple unpacking
        if self.conditioner is not None:
            tensors.insert(1, torch.cat(all_cond, dim=0)) # (N, Cond_Dim)
        else:
            tensors.insert(1, torch.empty((tensors[0].shape[0], 0), dtype=torch.float32))

        return TensorDataset(*tensors)

    def get_normalizer(self):
        """
        Get the fitted normalizer instance, if available. This can be used to access the normalizer's parameters or apply the same normalization to new data outside of the DataModule.
        
        Returns:
            The normalizer instance that was fitted on the training data, or None if no normalizer was provided or fitted. The normalizer should have a transform() method that can be applied to new data using the same parameters as were used for the training data.
        """
        return self.normalizer

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset with the specified batch size, shuffling, and number of workers. The DataLoader will yield batches of (x, meter_id, timestamp) for the training data.
        
        Returns:
            A DataLoader instance for the training dataset, configured with the specified batch size, shuffling, and number of workers. The DataLoader will yield batches of (x, meter_id, timestamp) for the training data.
        """
        return DataLoader(
            self._train_ds,
            batch_size         = self.batch_size,
            shuffle            = self.shuffle_train,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset with the specified batch size and number of workers. 
        Shuffling is disabled for validation data to ensure consistent evaluation.
        
        Return: 
            A DataLoader instance for the validation dataset, configured with the specified batch size and number of workers. 
        """
        return DataLoader(
            self._val_ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset with the specified batch size and number of workers.
        Shuffling is disabled for test data to ensure consistent evaluation.
        
        Returns: 
            A DataLoader instance for the test dataset, configured with the specified batch size and number of workers.
        """
        return DataLoader(
            self._test_ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )


def fit_normalizer_on_files(
    normalizer: Normalizer,
    data_dir: str,
    filenames: Sequence[str],
    feature_cols: Sequence[str],
    *,
    meter_col: str,
    timestamp_dst_col: str,
    train_frac: float = 0.8,
    seed: int = 0,
    stats_path: Optional[str] = None,
    force_refit: bool = False,
    ) -> None:
    """
    Fit a Normalizer across all data files using one streaming Welford pass.

    Reads from the Parquet cache when available (so the 3 GB CSV is only
    parsed once ever).  The fitted stats are saved next to the cache and
    reloaded automatically on subsequent calls, so training-run startup is
    O(seconds) after the first fit.

    Args: 
        normalizer: Normalizer instance to fit (modified in-place and saved).
        data_dir: directory that contains the CSV/Parquet files.
        filenames: CSV filenames to fit across (all passed to the same loader so they should share the same column schema).
        feature_cols: value columns to include in the fit.
        meter_col: meter identifier column name.
        timestamp_dst_col: DST timestamp column name.
        train_frac: fraction of meters per file used for fitting (mirrors the DataModule split for consistency).
        seed: RNG seed for the per-file meter shuffle.
        stats_path: explicit path for the cached stats file. If None, defaults to <data_dir>/_parquet_cache/normalizer_<method>.pt
        force_refit: if True, ignore any cached stats and refit from scratch.
    """
    if stats_path is None:
        stats_path = os.path.join(
            data_dir, "_parquet_cache",
            f"normalizer_{normalizer.method}.pt",
        )

    # --- try cache ---
    if not force_refit and os.path.exists(stats_path):
        try:
            saved = torch.load(stats_path, map_location="cpu", weights_only=False)
            manifest = saved.get("manifest", {})
            if (
                manifest.get("filenames")    == sorted(filenames)
                and manifest.get("train_frac") == train_frac
                and manifest.get("seed")       == seed
                and manifest.get("feature_cols") == list(feature_cols)
            ):
                # Invalidate if any source parquet (or CSV) is newer than stats
                stats_mtime = os.path.getmtime(stats_path)
                source_mtimes = []
                for f in filenames:
                    pq = os.path.join(data_dir, "_parquet_cache",
                                      os.path.splitext(f)[0] + ".parquet")
                    csv = os.path.join(data_dir, f)
                    if os.path.exists(pq):
                        source_mtimes.append(os.path.getmtime(pq))
                    elif os.path.exists(csv):
                        source_mtimes.append(os.path.getmtime(csv))
                if not source_mtimes or max(source_mtimes) <= stats_mtime:
                    # Restore stats into normalizer
                    ns = saved["normalizer_state"]
                    for k, v in ns.items():
                        setattr(normalizer, k, v)
                    normalizer.fitted = True
                    normalizer.to(normalizer.device)
                    print(f"[Normalizer] Loaded cached stats from {stats_path}")
                    return
        except Exception:
            pass  # corrupt cache — fall through to refit

    # --- fit from scratch ---
    csv_loader = SmartMeterCSV(
        data_dir=data_dir,
        meter_col=meter_col,
        timestamp_dst_col=timestamp_dst_col,
        value_cols=feature_cols,
    )

    feature_list = list(feature_cols)
    for filename in filenames:
        csv_loader.load(filename)
        meter_ids = csv_loader.meter_ids()
        rng = np.random.default_rng(seed)
        rng.shuffle(meter_ids)
        n_train = max(1, int(len(meter_ids) * train_frac))
        for mid in meter_ids[:n_train]:
            vals = csv_loader.meter_frame(mid)[feature_list].to_numpy(dtype=np.float32)
            if vals.size > 0:
                normalizer.fit_incremental(vals)

    normalizer.finalise_fit()

    # --- persist ---
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    ns = {
        "method":        normalizer.method,
        "epsilon":       normalizer.epsilon,
        "zero_id":       normalizer.zero_id,
        "shift":         normalizer.shift,
        "non_zero_mean": normalizer.non_zero_mean,
        "non_zero_std":  normalizer.non_zero_std,
        "mean": normalizer.mean.cpu() if isinstance(normalizer.mean, torch.Tensor) else normalizer.mean,
        "std":  normalizer.std.cpu()  if isinstance(normalizer.std,  torch.Tensor) else normalizer.std,
    }
    torch.save({
        "normalizer_state": ns,
        "manifest": {
            "filenames":    sorted(filenames),
            "train_frac":   train_frac,
            "seed":         seed,
            "feature_cols": list(feature_cols),
        },
    }, stats_path)
    print(f"[Normalizer] Stats fitted on {len(filenames)} file(s) and saved to {stats_path}")


