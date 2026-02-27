from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
import torch

# read and preprocess 
class SmartMeterCSV:
    """
    Loads and preprocesses SM data from CSV into a MultiIndex DataFrame:

    index: (meter_number, timestamp_utc). 
    columns include consumption/production. 
    """

    def __init__(
        self,
        data_dir: str, # directory where CSV files are located
        *,
        meter_col: str, # name of the meter identifier column
        timestamp_dst_col: str, # name of the DST timestamp column
        timestamp_utc_col: str, # name of the UTC timestamp column
        value_cols: Sequence[str], # names of the value columns (e.g., consumption, production)
        nan_as_zero: bool = True, # whether to replace NaN with 0
    ):
        self.data_dir = data_dir
        self.meter_col = meter_col
        self.timestamp_dst_col = timestamp_dst_col
        self.timestamp_utc_col = timestamp_utc_col
        self.value_cols = list(value_cols)
        self.nan_as_zero = nan_as_zero

        self.df: Optional[pd.DataFrame] = None

    def load(self, filename: str) -> pd.DataFrame:

        # make sure file exists
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {filename} not found in directory {self.data_dir}")

        columns_to_load = [self.meter_col, self.timestamp_dst_col, self.timestamp_utc_col] + self.value_cols
        
        with open(path, 'r') as f:
            df = pd.read_csv(
                f,
                usecols=columns_to_load,
                dtype={
                    self.meter_col: 'int32',  # Optimize meter_number dtype
                    **{col: 'float32' for col in self.value_cols}  # Optimize value columns
                }
            )

        # Defensive copy to detach from original CSV data
        # df = df.copy() # Should genuinely not be a problem, but if any file is corrupted we will have to activate again. Will be a big slowdown, so let's hope not.

        # Parse timestamps
        df[self.timestamp_dst_col] = pd.to_datetime(df[self.timestamp_dst_col])

        # Set MultiIndex
        df.set_index([self.meter_col, self.timestamp_dst_col], inplace=True)
        df.sort_index(inplace=True)

        # Convert to numeric
        for c in self.value_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # replaces NaN with 0
        if self.nan_as_zero:
            for c in self.value_cols:
                df[c] = df[c].fillna(0)

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

        # per-row YYYYMM keys derived from the DatetimeIndex
        month_key_np = (meter_df.index.year * 100 + meter_df.index.month).to_numpy(dtype=np.int64)

        if normalizer is not None and hasattr(normalizer, 'transform'):
            is_online = getattr(normalizer, 'method', None) == 'online_log1p' \
                     or getattr(normalizer, 'type',   None) == 'online_log1p'
            if is_online:
                values_t = normalizer.transform(values_np, month_key=month_key_np)
            else:
                values_t = normalizer.transform(values_np)

            # ensure cpu float32 tensor – DataLoader will handle batching/pin_memory
            if isinstance(values_t, np.ndarray):
                values_t = torch.from_numpy(values_t)
            self.values = values_t.cpu().to(dtype)
        else:
            self.values = torch.from_numpy(values_np).to(dtype)   # (T, F)

        self.meter_id = torch.tensor(meter_id, dtype=torch.long)
        self.times = torch.from_numpy(
            meter_df.index.view(np.int64) // 10**9  # ns -> seconds
        )
        self.month_keys = torch.from_numpy(month_key_np)           # (T,)
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

        month_key = self.month_keys[s]
        
        return x, self.meter_id, timestamp_unix, month_key


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
        timestamp_utc_col: str,
        feature_cols: Sequence[str],
        window: WindowSpec,
        batch_size: int = 64,
        num_workers: int = 0,
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

    def _fit_normalizer(self):
        """ Fit normalizer on training data only """
        if self.normalizer is None or not hasattr(self.normalizer, 'fit'):
            return  # nothing to fit

        all_values = []
        all_month_keys = []
        is_online = getattr(self.normalizer, 'method', None) == 'online_log1p' \
                 or getattr(self.normalizer, 'type',   None) == 'online_log1p'
        
        for mid in self.train_ids:
            meter_df = self.loader.meter_frame(mid)
            values   = meter_df[self.feature_cols].to_numpy(dtype=np.float32)
            if values.size == 0:
                continue
            all_values.append(values)
            if is_online:
                mk = (meter_df.index.year * 100 + meter_df.index.month).to_numpy(dtype=np.int64)
                all_month_keys.append(mk)

        if not all_values:
            raise ValueError("No training data available to fit normalizer.")
        
        all_values_np = np.concatenate(all_values, axis=0)
        month_key     = np.concatenate(all_month_keys) if (is_online and all_month_keys) else None

        self.normalizer.fit(all_values_np, month_key)

    def _make_dataset(self, meter_ids):
        datasets = []
        for mid in meter_ids:
            meter_df = self.loader.meter_frame(mid)
            if meter_df.empty:
                continue
            ds = SmartMeterWindowDataset(
                meter_df,
                meter_id=mid,
                feature_cols=self.feature_cols,
                window=self.window,
                normalizer=self.normalizer,
            )
            if len(ds) > 0:
                datasets.append(ds)
        
        if not datasets:
            raise ValueError("No windows produced for the given meter list.")
        return ConcatDataset(datasets)

    def get_normalizer(self):
        return self.normalizer

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size  = self.batch_size,
            shuffle     = self.shuffle_train,
            num_workers = self.num_workers,
            pin_memory  = False,
            persistent_workers = self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = False,
            persistent_workers = self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
        )


# usage
if __name__ == '__main__':
    from Data.normalizer import Normalizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm = Normalizer(method='online_log1p', device=device)

    dm = SmartMeterDataModule(
        data_dir  = 'data',
        filename  = 'hourly_measurements_2023_12.csv',
        feature_cols = ('hourly_energy_consumption', 'hourly_energy_production'),
        window    = WindowSpec(input_len=24, stride=1),
        batch_size   = 32,
        num_workers  = 0,
        normalizer   = norm,
    )
    dm.setup()
    # move all normalizer tensors to GPU after fitting
    norm.to(device)

    print("train meters:", len(dm.train_ids),
          "val meters:",   len(dm.val_ids),
          "test meters:",  len(dm.test_ids))

    train_loader = dm.train_dataloader()
    x, meter_id, times, month_key = next(iter(train_loader))
    # move batch to GPU in the training loop
    x         = x.to(device)
    meter_id  = meter_id.to(device)
    month_key = month_key.to(device)

    print("x shape:",         x.shape)          # [B, 24, F]
    print("meter_id shape:",  meter_id.shape)
    print("month_key sample:", month_key[:5])    # e.g. [202312, 202312, ...]

