from __future__ import annotations
import gc
import os
from dataclasses import dataclass
from typing import Sequence, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def map_meter_ids_to_feeder_ids(
    meter_ids: Sequence,
    meter_to_feeder: dict[str, str | None],
) -> np.ndarray:
    """Map a sequence of meter ids to feeder ids using a pre-loaded mapping.
    If a meter id is not found in the mapping, None is used as the feeder id.
    
    Args:
        meter_ids (Sequence): Sequence of meter ids to map.
        meter_to_feeder (dict[str, str | None]): Mapping from meter ids to feeder ids.

    Returns:
        np.ndarray: Array of feeder ids.
    """
    feeder_ids = []
    for meter in meter_ids:
        if isinstance(meter, np.generic):
            meter = meter.item()
        feeder_ids.append(meter_to_feeder.get(str(meter), None))
    return np.asarray(feeder_ids, dtype=object)

# read and preprocess 
class SmartMeterCSV:
    """Loads and preprocesses SM data from CSV into a MultiIndex DataFrame 
    indexed by (meter_id, timestamp), with value columns for consumption/production/etc.

    The load() method implements a caching mechanism that converts the raw CSV
    into a smaller, faster-to-read Parquet file on the first run.  Subsequent
    runs read from the Parquet cache, skipping the expensive CSV parsing step.

    Args:
        data_dir (str): Directory where the CSV files are located.
        meter_col (str): Name of the meter identifier column in the CSV.
        timestamp_dst_col (str): Name of the DST timestamp column in the CSV.
        timestamp_utc_col (str | None): Name of the UTC timestamp column in the CSV.
            This column is loaded but not used in the DataFrame index, so it can be omitted to save I/O if not needed.
        value_cols (Sequence[str]): Names of the value columns to load (e.g., consumption, production).
        nan_as_zero (bool): Whether to replace NaN values in the value columns with 0 after loading.
            Parquet caches are always stored NaN-preserving regardless of this flag.
        parquet_only (bool): If True, bypass CSV parsing and read only parquet cache files.

    Attributes:
        df (pd.DataFrame | None): The loaded DataFrame after calling load(). Indexed by
            (meter_id, timestamp_dst_col) with value columns as specified. Initially None until load() is called.
    """

    def __init__(
        self,
        data_dir: str,                           # directory where CSV files are located
        *,
        meter_col: str,                          # name of the meter identifier column
        timestamp_dst_col: str,                  # name of the DST timestamp column
        timestamp_utc_col: str | None = None, # UTC column – loaded but never used; omit to save I/O
        value_cols: Sequence[str],               # names of the value columns (e.g., consumption, production)
        nan_as_zero: bool = True,                # apply NaN->0 policy after load (cache remains raw)
        parquet_only: bool = False,
        max_failed_value_ratio: float | None = 0.90,
        max_bad_examples: int = 5,
    ): 
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "csv")
        self.meter_col = meter_col
        self.timestamp_dst_col = timestamp_dst_col
        self.timestamp_utc_col = timestamp_utc_col
        self.value_cols = list(value_cols)
        self.nan_as_zero = nan_as_zero
        self.parquet_only = parquet_only
        self.max_failed_value_ratio = max_failed_value_ratio
        self.max_bad_examples = max_bad_examples

        self.df: pd.DataFrame | None = None
        self.last_quality_report: dict[str, int | float | str] | None = None

    def _apply_nan_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.nan_as_zero:
            return df
        cols = [c for c in self.value_cols if c in df.columns]
        if cols:
            df.loc[:, cols] = df.loc[:, cols].fillna(0.0)
        return df

    def load(self, filename: str) -> pd.DataFrame:
        """Load CSV (or a Parquet cache thereof) into a MultiIndex DataFrame.

        On the first call the 3 GB CSV is chunk-read and written to a
        _parquet_cache/<stem>.parquet file (zstd-compressed
        smaller than the CSV and faster to read).  Every subsequent
        call simply reads the Parquet file, skipping the expensive CSV
        parse entirely.  The cache is invalidated automatically whenever
        the source CSV is newer than the cached file.

        Args:
            filename (str): Name of the CSV file to load (must be located in data_dir
                and should have a corresponding Parquet cache file named <stem>.parquet in the _parquet_cache subdir).
        Returns:
            pd.DataFrame: The loaded DataFrame, indexed by (meter_id, timestamp_dst_col) with value columns as specified in the constructor.
        """
        base_stem = os.path.splitext(os.path.basename(filename))[0]
        parquet_name = f"{base_stem}.parquet"
        cache_dirs = [
            os.path.join(self.data_dir, "_parquet_cache"),
            os.path.join(self.data_dir, "parquet"),
        ]

        cache_candidates: list[str] = []
        if os.path.isabs(filename):
            cache_candidates.append(filename)
        else:
            cache_candidates.append(os.path.join(self.data_dir, filename))
            for cache_dir in cache_dirs:
                cache_candidates.append(os.path.join(cache_dir, filename))
        for cache_dir in cache_dirs:
            cache_candidates.append(os.path.join(cache_dir, parquet_name))

        # De-duplicate candidates while preserving order
        seen_cache_paths: set[str] = set()
        unique_cache_candidates: list[str] = []
        for p in cache_candidates:
            npth = os.path.normpath(p)
            if npth not in seen_cache_paths:
                seen_cache_paths.add(npth)
                unique_cache_candidates.append(p)
        cache_candidates = unique_cache_candidates
        parquet_candidates = [
            p for p in cache_candidates
            if os.path.splitext(p)[1].lower() == ".parquet"
        ]

        if self.parquet_only:
            unreadable_cache_errors: list[str] = []
            for parquet_path in parquet_candidates:
                if not os.path.exists(parquet_path):
                    continue
                try:
                    self.df = pd.read_parquet(parquet_path, engine="pyarrow")
                    self._apply_nan_policy(self.df)
                    return self.df
                except Exception as e:
                    unreadable_cache_errors.append(
                        f"{parquet_path} ({type(e).__name__}: {e})"
                    )

            details = (
                f"Unreadable parquet cache candidates: {unreadable_cache_errors}"
                if unreadable_cache_errors
                else f"Looked in: {parquet_candidates}"
            )
            raise FileNotFoundError(
                f"Parquet-only mode enabled and no cache file found for '{filename}'. "
                f"{details}"
            )

        if os.path.isabs(filename):
            path_candidates = [filename]
        else:
            path_candidates = [
                os.path.join(self.csv_path, filename),
                os.path.join(self.data_dir, filename),
            ]
        path = next((p for p in path_candidates if os.path.exists(p)), None)

        if path is None:
            unreadable_cache_errors: list[str] = []
            for parquet_fallback in parquet_candidates:
                if not os.path.exists(parquet_fallback):
                    continue
                try:
                    self.df = pd.read_parquet(parquet_fallback, engine="pyarrow")
                    self._apply_nan_policy(self.df)
                    return self.df
                except Exception as e:
                    unreadable_cache_errors.append(
                        f"{parquet_fallback} ({type(e).__name__}: {e})"
                    )

            details = (
                f" Unreadable parquet cache candidates: {unreadable_cache_errors}."
                if unreadable_cache_errors
                else ""
            )
            raise FileNotFoundError(
                f"File {filename} not found in {self.csv_path} or {self.data_dir}.{details}"
            )

        # --- fast path: read from cache ---
        for parquet_path in parquet_candidates:
            if (
                os.path.exists(parquet_path)
                and os.path.getmtime(parquet_path) >= os.path.getmtime(path)
            ):
                try:
                    self.df = pd.read_parquet(parquet_path, engine="pyarrow")
                    self._apply_nan_policy(self.df)
                    return self.df
                except Exception as e:
                    print(
                        "Warning: unreadable parquet cache "
                        f"{parquet_path}; rebuilding from CSV. "
                        f"({type(e).__name__}: {e})"
                    )

        # --- slow path: parse CSV in chunks, build Parquet cache ---
        columns_to_load = [self.meter_col, self.timestamp_dst_col] + self.value_cols
        # Include UTC column only when explicitly requested
        if self.timestamp_utc_col is not None:
            columns_to_load.append(self.timestamp_utc_col)

        dtype_map = {
            self.meter_col: "int32",
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
            chunk[self.timestamp_dst_col] = pd.to_datetime(chunk[self.timestamp_dst_col], format='ISO8601')
            for c in self.value_cols:
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

                if col_failed > 0 and len(bad_examples) < self.max_bad_examples:
                    bad_positions = np.flatnonzero(problem_mask.to_numpy())
                    needed = self.max_bad_examples - len(bad_examples)
                    for pos in bad_positions[:needed]:
                        file_row = chunk_start_row + int(pos)
                        meter_val = chunk.iloc[int(pos)][self.meter_col]
                        ts_val = chunk.iloc[int(pos)][self.timestamp_dst_col]
                        raw_val = raw_values.iloc[int(pos)]
                        bad_examples.append(
                            f"row={file_row}, col={c}, meter={meter_val}, timestamp={ts_val}, raw={raw_val!r}"
                        )

                chunk[c] = numeric_values.astype(np.float32)
            chunks.append(chunk)
            rows_seen += len(chunk)

        failed_ratio = (failed_values / total_values) if total_values > 0 else 0.0
        self.last_quality_report = {
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

        if self.max_failed_value_ratio is not None and failed_ratio > self.max_failed_value_ratio:
            raise ValueError(
                f"Too many failed values in {filename}: {failed_values}/{total_values} "
                f"({failed_ratio:.4%}) exceeds max_failed_value_ratio={self.max_failed_value_ratio:.2%}."
            )

        df = pd.concat(chunks, ignore_index=True)
        del chunks  # free intermediate memory before the MultiIndex sort

        df.set_index([self.meter_col, self.timestamp_dst_col], inplace=True)
        df.sort_index(inplace=True)

        # Persist as Parquet cache for future runs
        cache_dir = cache_dirs[0]
        parquet_path = os.path.join(cache_dir, parquet_name)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")

        self.df = df
        self._apply_nan_policy(self.df)
        return self.df

    # returns all unique meter IDs in the index
    def meter_ids(self) -> List[int]:
        """
        Returns all unique meter IDs in the index.

        Returns:
            List[int]: A list of unique meter IDs.
        """

        self._require_loaded()
        return self.df.index.get_level_values(self.meter_col).unique().tolist()

    # returns DataFrame for a single meter ID
    def meter_frame(self, meter_id: int) -> pd.DataFrame:
        """Returns a DataFrame for a single meter ID, indexed by timestamp.
         The returned DataFrame is sorted by timestamp and contains only the value columns specified in the constructor.
         Args:
            meter_id (int): The meter ID for which to retrieve the DataFrame.
        Returns:
            pd.DataFrame: A DataFrame for the specified meter ID, indexed by timestamp and containing"""
        self._require_loaded()
        return self.df.xs(meter_id, level=self.meter_col).sort_index()

    # prints basic stats
    def examine(self) -> None:
        self._require_loaded()
        print("Data Info:")
        print(self.df.info())
        print("\nData Description:")
        print(self.df.describe())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def _require_loaded(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Call .load(filename) first.")



# Dataset: sliding windows per meter
@dataclass(frozen=True)
class WindowSpec:
    input_len: int # length of input window - how many past hours go into the model as x
    stride: int = 1 # how many hours to move the window each step

# need fixed window dataset per meter, an PyTorch Dataset needs samples of same shape
class SmartMeterWindowDataset(Dataset):
    """ A PyTorch Dataset that provides sliding windows of smart meter data for a single meter ID.
        Each sample is a tuple (x, conditions, meter_id, timestamp) where:
        - x: Tensor of shape [input_len, num_features] containing the value columns for the window.
        - conditions: Tensor of shape [condition_dim] containing condition embeddings/features (or empty tensor).
        - meter_id: Tensor containing the meter ID (same for all samples in this dataset).
        - timestamp: Tensor containing the Unix timestamp (in seconds) of the start of the window.

        Args:
            meter_df (pd.DataFrame): DataFrame for a single meter ID, indexed by timestamp and containing the value columns.
            meter_id (int): The meter ID corresponding to the DataFrame.
            feature_cols (Sequence[str]): Names of the value columns to include in x.
            window (WindowSpec): Specification of the sliding window (input length and stride).
            drop_nan_windows (bool): Whether to drop windows that contain any NaN values. Defaults to True.
            dtype (torch.dtype): Data type for the x tensor. Defaults to torch.float32.
            normalizer: Optional normalizer object with a .transform() method to apply to the values before converting to tensors. If None, no normalization is applied.
            conditioner: Optional conditioner object with a .get_conditions() method to generate condition embeddings/features.

        Attributes:
            feature_cols (List[str]): List of value column names to include in x.
            window (WindowSpec): Specification of the sliding window.
            drop_nan_windows (bool): Whether to drop windows that contain any NaN values.
            dtype (torch.dtype): Data type for the x tensor.
            values (torch.Tensor): Tensor containing the value columns for the meter, of shape [T, num_features].
            meter_id (torch.Tensor): Tensor containing the meter ID.
            times (torch.Tensor): Tensor containing the Unix timestamps (in seconds) corresponding to each row in values.
            starts (torch.Tensor): Tensor containing the starting indices of valid windows in values.
            conditions (torch.Tensor): Tensor containing the condition embeddings/features for each window, of shape [num_windows, condition_dim] (or empty tensor if no conditioner).
            
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
        conditioner = None,
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
        
        if conditioner is not None:
            L = self.window.input_len
            idx = self.starts.unsqueeze(1) + torch.arange(L, dtype=torch.long)
            x_windows = self.values[idx]
            meter_t = torch.full((len(self.starts),), meter_id, dtype=torch.long)
            time_windows = self.times[self.starts]
            self.conditions = conditioner.get_conditions(
                x=x_windows, 
                meter_id=meter_t, 
                timestamps=time_windows
            )
        else:
            self.conditions = torch.empty((len(self.starts), 0), dtype=torch.float32)

    # determines which window start positions are allowed
    def _compute_starts(self, drop_nan: bool) -> np.ndarray:
        """Compute valid window start indices based on the window specification and NaN handling.
        Args:
            drop_nan (bool): Whether to drop windows that contain any NaN values.
        Returns:
            np.ndarray: Array of valid window start indices.
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
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        L = self.window.input_len
        x = self.values[s : s + L]
        timestamp_unix = self.times[s]
        c = self.conditions[idx]
        
        return x, c, self.meter_id, timestamp_unix


# build DataLoaders for train/val/test splits (wraps full pipeline)
class SmartMeterDataModule:
    """ A PyTorch DataModule that loads smart meter data from CSV, splits into train/val/test by meter ID, and provides DataLoaders that yield batches of sliding windows with optional conditioning.
        Each batch is a tuple (x, conditions, meter_id, timestamp) where:
        - x: Tensor of shape [batch_size, input_len, num_features] containing the value columns for the windows.
        - conditions: Tensor of shape [batch_size, condition_dim] containing condition embeddings/features (or empty tensor).
        - meter_id: Tensor of shape [batch_size] containing the meter IDs corresponding to each window.
        - timestamp: Tensor of shape [batch_size] containing the Unix timestamps (in seconds) of the start of each window.

        Args:
            data_dir (str): Directory where the CSV files are located.
            filename (str): Name of the CSV file to load (must be located in data_dir).
            meter_col (str): Name of the meter identifier column in the CSV.
            timestamp_dst_col (str): Name of the DST timestamp column in the CSV.
            timestamp_utc_col (str | None): Name of the UTC timestamp column in the CSV. This column is loaded but not used in the DataFrame index, so it can be omitted to save I/O if not needed.
            feature_cols (Sequence[str]): Names of the value columns to load (e.g., consumption, production).
            window (WindowSpec): Specification of the sliding window (input length and stride).
            batch_size (int): Batch size for the DataLoaders. Defaults to 64.
            num_workers (int): Number of worker processes for data loading. Defaults to min(4, half of CPU cores).
            shuffle_train (bool): Whether to shuffle the training data. Defaults to True.
            seed (int): Random seed for reproducibility of train/val/test splits. Defaults to 0.
            train_frac (float): Fraction of meters to use for training. Defaults to 0.8.
            val_frac (float): Fraction of meters to use for validation. Defaults to 0.1.
            max_meters (int | None): Optional maximum number of meters to include in the dataset (for debugging or memory constraints). If None, all meters are included.
            parquet_only (bool): If True, bypass CSV parsing and read only parquet cache files.
            normalizer: Optional normalizer object with .fit() and .transform() methods to apply normalization based on training data statistics.
            conditioner: Optional conditioner object with a .get_conditions() method to generate condition embeddings/features.

        Attributes:
            loader (SmartMeterCSV): The underlying data loader that reads the CSV and provides meter-specific DataFrames.
            filename (str): The name of the CSV file being loaded.
            feature_cols (List[str]): List of value column names to include in x.
            window (WindowSpec): Specification of the sliding window.
            batch_size (int): Batch size for the DataLoaders.
            num_workers (int): Number of worker processes for data loading.
            shuffle_train (bool): Whether to shuffle the training data.
            seed (int): Random seed for reproducibility of train/val/test splits.
            train_frac (float): Fraction of meters to use for training.
            val_frac (float): Fraction of meters to use for validation.
            max_meters (int | None): Optional maximum number of meters to include in the dataset.
            normalizer: Optional normalizer object for fitting and transforming the value columns.
            conditioner: Optional conditioner object for generating condition embeddings/features.
    """

    def __init__(
        self,
        data_dir: str,
        filename: str,
        *,
        meter_col: str,
        timestamp_dst_col: str,
        timestamp_utc_col: str | None = None,  # not used in tensors; omit to save I/O
        feature_cols: Sequence[str] = (),
        window: WindowSpec = WindowSpec(input_len=24),
        batch_size: int = 64,
        num_workers: int = min(4, (os.cpu_count() or 1) // 2),  # auto-detect; override per run
        shuffle_train: bool = True,
        seed: int = 0,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        max_meters: int | None = None,
        nan_as_zero: bool = True,
        parquet_only: bool = False,
        normalizer = None,
        conditioner = None,
    ):
        self.loader = SmartMeterCSV(
            data_dir=data_dir,
            meter_col=meter_col,
            timestamp_dst_col=timestamp_dst_col,
            timestamp_utc_col=timestamp_utc_col,
            value_cols=feature_cols,
            nan_as_zero=nan_as_zero,
            parquet_only=parquet_only,
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
        self.nan_as_zero = nan_as_zero

        self.normalizer = normalizer
        self.conditioner = conditioner

        self._train_ds = None 
        self._val_ds = None
        self._test_ds = None

    def setup(self) -> None:
        """Load data, split by meter ID, fit normalizer and conditioner on training data, and precompute window datasets for train/val/test splits.
        This method should be called before accessing the DataLoaders. It performs the following steps:
        1. Loads the data from the specified CSV file using the SmartMeterCSV loader.
        2. Retrieves the list of unique meter IDs and applies the max_meters limit if specified.
        3. Shuffles the meter IDs using the specified random seed for reproducibility.
        4. Splits the meter IDs into train/val/test sets based on the specified fractions, ensuring at least one meter in each split when possible.
        5. Fits the normalizer on the training data by streaming through the meters one at a time to compute necessary statistics without loading all data into memory at once.
        6. Fits the conditioner on the training data by streaming through the meters one at a time to compute necessary condition embeddings/features without loading all data into memory at once.
        7. Precomputes the sliding window datasets for the train/val/test splits and stores them as TensorDatasets for efficient access during training.
        """
        self.loader.load(self.filename)
        meter_ids = self.loader.meter_ids()

        # Limit number of meters if max_meters is specified
        if self.max_meters is not None:
            meter_ids = meter_ids[:self.max_meters]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(meter_ids)

        n = len(meter_ids)
        n_train = 0
        n_val = 0
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
        if self.normalizer is not None:
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
            This generator streams through the training meters one at a time, applying the same windowing and NaN handling logic as in SmartMeterWindowDataset, but without precomputing all windows into a single dataset. This allows fitting the conditioner on training data without loading everything into memory at once.
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
        """Fit conditioner on training meters only, streamed one meter at a time.
            If the conditioner is already fitted (e.g. pre-fitted across multiple files via fit_conditioner_on_files()) this method is a no-op, so the existing state is preserved.
            Values are streamed through get_conditions() so peak RAM is O(one meter's data) regardless of training split size.
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
                    for x, m, t in self.dm._yield_train_batches():
                        yield x, None, m, t

            user_profiler.fit(StreamLoader(self))

    def _fit_normalizer(self):
        """Fit normalizer on training meters only, streamed one meter at a time.

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
                if self.nan_as_zero:
                    values = np.nan_to_num(values, nan=0.0)
                else:
                    values = values[np.isfinite(values).all(axis=1)]
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
                if self.nan_as_zero:
                    values = np.nan_to_num(values, nan=0.0)
                else:
                    values = values[np.isfinite(values).all(axis=1)]
                    if values.size == 0:
                        continue
                all_values.append(values)
            if not all_values:
                raise ValueError("No training data available to fit normalizer.")
            self.normalizer.fit(np.concatenate(all_values, axis=0))

    def _make_dataset(self, meter_ids) -> TensorDataset:
        """Pre-compute all windows for the given meters into a single TensorDataset.
            This is more efficient for training than having a separate SmartMeterWindowDataset per meter, 
            because it allows larger batch sizes and shuffling across meters.  
            The same windowing and NaN handling logic as in SmartMeterWindowDataset is applied here, 
            but the results are concatenated into big tensors for all meters in the split.  
            If the conditioner is used, the condition embeddings/features are also pre-computed and included in the dataset.
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
        return self.normalizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size         = self.batch_size,
            shuffle            = self.shuffle_train,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = self.num_workers,
            pin_memory         = torch.cuda.is_available(),
            persistent_workers = self.num_workers > 0,
        )


def fit_normalizer_on_files(
    normalizer,
    data_dir: str,
    filenames: Sequence[str],
    feature_cols: Sequence[str],
    *,
    meter_col: str,
    timestamp_dst_col: str,
    train_frac: float = 0.8,
    seed: int = 0,
    stats_path: str | None = None,
    force_refit: bool = False,
    nan_as_zero: bool = True,
    parquet_only: bool = False,
) -> None:
    """Fit a Normalizer across all data files using one streaming Welford pass.

    Reads from the Parquet cache when available (so the 3 GB CSV is only
    parsed once ever).  The fitted stats are saved next to the cache and
    reloaded automatically on subsequent calls, so training-run startup is
    O(seconds) after the first fit.

    Args:
    normalizer: Normalizer object with .fit_incremental() and .finalise_fit() methods.
    data_dir (str): Directory where the CSV files are located.
    filenames (Sequence[str]): List of CSV filenames to include in the fit (must be located
        in data_dir and should have corresponding Parquet cache files named <stem>.parquet in the _parquet_cache subdir).
    feature_cols (Sequence[str]): Names of the value columns to include in the fit (e.g., consumption, production).
    meter_col (str): Name of the meter identifier column in the CSV.
    timestamp_dst_col (str): Name of the DST timestamp column in the CSV.
    train_frac (float): Fraction of meters to use for fitting the normalizer. Defaults to 0.8.
    seed (int): Random seed for reproducibility of the meter sampling. Defaults to 0.
    stats_path (str | None): Optional path to save/load the fitted normalizer stats. If None, defaults to <data_dir>/_parquet_cache/normalizer_<method>.pt.
    force_refit (bool): If True, forces refitting the normalizer from scratch even
        if cached stats are available. Defaults to False.
    parquet_only (bool): If True, load only parquet cache files while fitting.
    """
    if stats_path is None:
        cache_tag = "nan0" if nan_as_zero else "nanraw"
        stats_path = os.path.join(
            data_dir, "_parquet_cache",
            f"normalizer_{normalizer.method}_{cache_tag}.pt",
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
                and manifest.get("nan_as_zero", True) == nan_as_zero
            ):
                # Invalidate if any source parquet (or CSV) is newer than stats
                stats_mtime = os.path.getmtime(stats_path)
                source_mtimes = []
                for f in filenames:
                    source_candidates = []
                    if os.path.isabs(f):
                        source_candidates.append(f)
                    else:
                        source_candidates.append(os.path.join(data_dir, f))
                        source_candidates.append(os.path.join(data_dir, "csv", f))

                    base_stem = os.path.splitext(os.path.basename(f))[0]
                    parquet_name = f"{base_stem}.parquet"
                    source_candidates.append(os.path.join(data_dir, "_parquet_cache", parquet_name))
                    source_candidates.append(os.path.join(data_dir, "parquet", parquet_name))

                    existing_mtimes = [os.path.getmtime(p) for p in source_candidates if os.path.exists(p)]
                    if existing_mtimes:
                        source_mtimes.append(max(existing_mtimes))
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
        nan_as_zero=nan_as_zero,
        parquet_only=parquet_only,
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
            if vals.size == 0:
                continue

            if nan_as_zero:
                vals = np.nan_to_num(vals, nan=0.0)
            else:
                # Keep cache NaNs, but normalizer fitting requires finite values.
                finite_mask = np.isfinite(vals).all(axis=1)
                vals = vals[finite_mask]

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
            "nan_as_zero":  nan_as_zero,
        },
    }, stats_path)
    print(f"[Normalizer] Stats fitted on {len(filenames)} file(s) and saved to {stats_path}")

def fit_userprofiler_on_files(
    user_profiler,
    data_dir: str,
    filenames: Sequence[str],
    feature_cols: Sequence[str],
    *,
    window: WindowSpec,
    meter_col: str,
    timestamp_dst_col: str,
    timestamp_utc_col: str | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
    seed: int = 0,
    train_frac: float = 0.8,
    normalizer=None,
    force_refit: bool = False,
    nan_as_zero: bool = True,
    parquet_only: bool = False,
) -> None:
    
    if user_profiler is None:
        return
        
    stats_path = getattr(user_profiler, 'path', None)
    
    if stats_path is not None and not force_refit and os.path.exists(stats_path):
        print(f"[UserProfiler] Loaded cached profiler from {stats_path}")
        user_profiler.load(stats_path)
        return
        
    print(f"[UserProfiler] Fitting across {len(filenames)} file(s) from scratch...")

    class MultiFileLoader:
        def __init__(self):
            self.filenames = filenames
            
        def __iter__(self):
            for filename in self.filenames:
                dm = SmartMeterDataModule(
                    data_dir=data_dir,
                    filename=filename,
                    meter_col=meter_col,
                    timestamp_dst_col=timestamp_dst_col,
                    timestamp_utc_col=timestamp_utc_col,
                    feature_cols=feature_cols,
                    window=window,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle_train=False, # Must be false to properly build coherent time series words if sequential matters, actually UserProfiler just needs windows
                    seed=seed,
                    train_frac=train_frac,
                    nan_as_zero=nan_as_zero,
                    parquet_only=parquet_only,
                    normalizer=normalizer,
                    conditioner=None # none to prevent recursive fit!
                )

                # Initialize loader + meter splits for streaming without materializing datasets.
                dm.loader.load(filename)
                meter_ids = dm.loader.meter_ids()
                if dm.max_meters is not None:
                    meter_ids = meter_ids[:dm.max_meters]

                rng = np.random.default_rng(dm.seed)
                rng.shuffle(meter_ids)

                n = len(meter_ids)
                if n == 0:
                    dm.train_ids, dm.val_ids, dm.test_ids = [], [], []
                else:
                    n_train = max(1, int(n * dm.train_frac))
                    n_val = max(0, int(n * dm.val_frac))

                    if n >= 2 and (n_train + n_val) >= n:
                        n_val = max(0, n - n_train - 1)
                        if n_train >= n:
                            n_train = n - 1

                    dm.train_ids = meter_ids[:n_train]
                    dm.val_ids = meter_ids[n_train:n_train + n_val]
                    dm.test_ids = meter_ids[n_train + n_val:]

                print(f"  -> Profiling {filename}...")
                for x, m, t in dm._yield_train_batches():
                    # yield 4 items exactly like UserProfiler.fit expects: x, cond, meter, time
                    yield x, None, m, t

    loader = MultiFileLoader()
    user_profiler.fit(loader)


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
        csv_loader = SmartMeterCSV(
            data_dir=data_dir,
            meter_col="meter_number",
            timestamp_dst_col="timestamp_dst",
            value_cols=["hourly_energy_consumption",],
            nan_as_zero=False,
            parquet_only=False,
        )
        csv_loader.load(filename)
        print(f"{csv_loader.df.isna().sum().sum()} NaN values in {filename} (will be cached as-is, but normalizer fitting will handle them according to its configuration)")
    
    # Then fit normalizer and conditioner across all files using the streaming multi-file fitting functions, which will read from the parquet cache we just prepared for efficiency
    fit_normalizer_on_files(
        normalizer=normalizer,
        data_dir=data_dir,
        filenames=csv_files,
        feature_cols=["hourly_energy_consumption",],
        meter_col="meter_number",
        timestamp_dst_col="timestamp_dst",
        train_frac=1.0,  # use all data for fitting
        seed=0,
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



