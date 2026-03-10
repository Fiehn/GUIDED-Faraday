from __future__ import annotations
import gc
import os
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
from Data.normalizer import Normalizer

# read and preprocess 
class SmartMeterCSV:
    """
    Loads and preprocesses SM data from CSV into a MultiIndex DataFrame:

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

    def load(self, filename: str) -> pd.DataFrame:
        """Load CSV (or a Parquet cache thereof) into a MultiIndex DataFrame.

        On the first call the 3 GB CSV is chunk-read and written to a
        _parquet_cache/<stem>.parquet file (zstd-compressed
        smaller than the CSV and faster to read).  Every subsequent
        call simply reads the Parquet file, skipping the expensive CSV
        parse entirely.  The cache is invalidated automatically whenever
        the source CSV is newer than the cached file.
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
        self._require_loaded()
        return self.df.index.get_level_values(self.meter_col).unique().tolist()

    # returns DataFrame for a single meter ID
    def meter_frame(self, meter_id: int) -> pd.DataFrame:
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
    """
    Convert a single meter's time series into x. 
    Assumes regular hourly sampling, so if timestamps are missing, it still windows by row order.
    x: [input_len, num_features]
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
    def _compute_starts(self, drop_nan: bool) -> np.ndarray:
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
    def __getitem__(self, idx: int) -> torch.Tensor:
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

        self._train_ds = None 
        self._val_ds = None
        self._test_ds = None

    def setup(self) -> None:
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

        self._train_ds  = self._make_dataset(self.train_ids)
        self._val_ds    = self._make_dataset(self.val_ids)
        self._test_ds   = self._make_dataset(self.test_ids)

        # Free the raw DataFrame now that all splits are converted to tensors.
        # This halves peak memory when the DataModule stays alive during training.
        self.loader.df = None
        gc.collect()

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

    def _make_dataset(self, meter_ids) -> TensorDataset:
        """Pre-compute all windows for the given meters into a single TensorDataset.

        Pre-flattening means __getitem__ is an O(1) tensor index rather than
        an O(log N_meters) binary search through a ConcatDataset offset list.
        All windows are held in RAM, but their total size is small compared
        to the raw CSV (float32 windows only; no duplicate timestamp storage).
        """
        L = self.window.input_len
        F = len(self.feature_cols)

        all_x:      list[torch.Tensor] = []
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
            all_x.append(x_windows)
            all_meter.append(torch.full((len(valid),), mid, dtype=torch.long))
            all_times.append(times_t[starts_t])

        if not all_x:
            # Return empty dataset with correct shapes so DataLoaders don't crash
            return TensorDataset(
                torch.empty(0, L, F, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.int32),
            )

        return TensorDataset(
            torch.cat(all_x,      dim=0),  # (N, L, F)
            torch.cat(all_meter,  dim=0),  # (N,)
            torch.cat(all_times,  dim=0),  # (N,)
        )

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
    stats_path: Optional[str] = None,
    force_refit: bool = False,
) -> None:
    """Fit a Normalizer across all data files using one streaming Welford pass.

    Reads from the Parquet cache when available (so the 3 GB CSV is only
    parsed once ever).  The fitted stats are saved next to the cache and
    reloaded automatically on subsequent calls, so training-run startup is
    O(seconds) after the first fit.

    Parameters
    ----------
    normalizer       : Normalizer instance to fit (modified in-place and saved).
    data_dir         : directory that contains the CSV/Parquet files.
    filenames        : CSV filenames to fit across (all passed to the same
                       loader so they should share the same column schema).
    feature_cols     : value columns to include in the fit.
    meter_col        : meter identifier column name.
    timestamp_dst_col: DST timestamp column name.
    train_frac       : fraction of meters per file used for fitting
                       (mirrors the DataModule split for consistency).
    seed             : RNG seed for the per-file meter shuffle.
    stats_path       : explicit path for the cached stats file.
                       If None, defaults to
                       <data_dir>/_parquet_cache/normalizer_<method>.pt
    force_refit      : if True, ignore any cached stats and refit from scratch.
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


# usage
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR     = 'data'
    FILENAMES    = ['test_data_small.csv']  # replace with actual filenames in the data directory
    FEATURE_COLS = ('hourly_energy_consumption',)
    METER_COL    = 'meter_number'
    TS_COL       = 'timestamp_dst'

    norm = Normalizer(method='log1p')

    # Fit once across all available files (cached on disk; instant on subsequent runs)
    fit_normalizer_on_files(
        norm, DATA_DIR, FILENAMES, FEATURE_COLS,
        meter_col=METER_COL, timestamp_dst_col=TS_COL,
    )
    norm.to(device)

    # Build a data module for the most recent file
    dm = SmartMeterDataModule(
        data_dir         = DATA_DIR,
        filename         = FILENAMES[-1],
        meter_col        = METER_COL,
        timestamp_dst_col= TS_COL,
        feature_cols     = FEATURE_COLS,
        window           = WindowSpec(input_len=24, stride=24),
        batch_size       = 32,
        normalizer       = norm,
    )
    dm.setup()

    print("train meters:", len(dm.train_ids),
          "val meters:",   len(dm.val_ids),
          "test meters:",  len(dm.test_ids))

    train_loader = dm.train_dataloader()
    x, meter_id, times = next(iter(train_loader))
    x        = x.to(device)
    meter_id = meter_id.to(device)

    print("x shape:    ", x.shape)        # [B, 24, 2]
    print("meter_id:   ", meter_id.shape)
    print("times dtype:", times.dtype)    # int32

