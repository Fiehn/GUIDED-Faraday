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
        
        df = pd.read_csv(
            path,
            usecols=columns_to_load,
            dtype={
                self.meter_col: 'int32',  # Optimize meter_number dtype
                **{col: 'float32' for col in self.value_cols}  # Optimize value columns
            }
        )

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
    ):
        self.feature_cols = list(feature_cols) 
        self.window = window
        self.drop_nan_windows = drop_nan_windows
        self.dtype = dtype
        self.values = meter_df[self.feature_cols].to_numpy(dtype=np.float32)
        self.meter_id = meter_id
        self.times = meter_df.index.to_numpy()
        self.starts = self._compute_starts(drop_nan_windows)
        

    # determines which window start positions are allowed
    def _compute_starts(self, drop_nan: bool) -> np.ndarray:
        T = self.values.shape[0]
        L = self.window.input_len
        stride = self.window.stride

        max_start = T - L
        if max_start < 0:
            return np.array([], dtype=np.int64)

        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        if not drop_nan:
            return starts

        good = []
        for s in starts:
            x = self.values[s : s + L]
            if not np.isnan(x).any():
                good.append(s)

        return np.array(good, dtype=np.int64)

    # number of windows
    def __len__(self) -> int:
        return len(self.starts)

    # get x at index idx so x: [B, input_len, F]
    def __getitem__(self, idx: int) -> torch.Tensor:
        s = int(self.starts[idx])
        L = self.window.input_len
        x = self.values[s : s + L]
        dow = pd.Timestamp(self.times[s]).dayofweek # day of week: monday=0 ... sunday=6  - this is for conditioning
        return (torch.tensor(x, dtype=self.dtype), torch.tensor(self.meter_id, dtype=torch.long), torch.tensor(dow, dtype=torch.long))


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
        n_train = max(1, int(n * self.train_frac)) # int(n * self.train_frac)
        n_val = max(1, int(n * self.val_frac)) # int(n * self.val_frac)
        if n_train + n_val >= n: # dont exceed n
            n_val = max(0, n-n_train - 1) # leave at least 1 for test if possible

        self.train_ids = meter_ids[:n_train]
        self.val_ids = meter_ids[n_train : n_train + n_val]
        self.test_ids = meter_ids[n_train + n_val :]

        self._train_ds = self._make_dataset(self.train_ids)
        self._val_ds = self._make_dataset(self.val_ids)
        self._test_ds = self._make_dataset(self.test_ids)

    def _make_dataset(self, meter_ids):

        datasets = []
        for mid in meter_ids:
            ds = SmartMeterWindowDataset(
                self.loader.meter_frame(mid),
                meter_id=mid,
                feature_cols=self.feature_cols,
                window=self.window,
            )
            if len(ds) > 0:
                datasets.append(ds)
            
        if len(datasets) == 0:
            x_empty = torch.empty((0, self.window.input_len, len(self.feature_cols)), dtype=torch.float32)
            meter_empty = torch.empty((0,), dtype=torch.long)
            dow_empty = torch.empty((0,), dtype=torch.long)
            return TensorDataset(x_empty, meter_empty, dow_empty)

        return ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# usage
if __name__ == "__main__":
    dm = SmartMeterDataModule(
        data_dir="data",
        filename="hourly_measurements_2023_12.csv",
        feature_cols=("hourly_energy_consumption", "hourly_energy_production"),
        window=WindowSpec(input_len=24, stride=1),
        batch_size=32,
        num_workers=0,
    )
    dm.setup()
    print("train meters:", len(dm.train_ids), "val meters:", len(dm.val_ids), "test meters:", len(dm.test_ids))
    print("train ds len:", len(dm._train_ds), "val ds len:", len(dm._val_ds), "test ds len:", len(dm._test_ds))

    train_loader = dm.train_dataloader()
    x, meter_id, dow = next(iter(train_loader))
    print("x shape:", x.shape) # [B, 24, F]
    print("meter_id shape:", meter_id.shape, "example:", meter_id[:5])
    print("dow shape:", dow.shape, "example:", dow[:10])



















# class RawData:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.data = None

#     def load_data(self, filename):
#         file_path = os.path.join(self.data_dir, filename)
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File {filename} not found in directory {self.data_dir}")
#         data = pd.read_csv(file_path)
#         return data
    
#     def preprocess_data(self, data):
#         # The columns are: meter_number, timestamp_dst, timestamp_utc, hourly_energy_consumption, hourly_energy_production
#         # The meter_number is a unique identifier for each meter, this will be used for indexing
#         data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'])
#         data.set_index(['meter_number', 'timestamp_utc'], inplace=True)
#         data.sort_index(inplace=True)

#         # 0 value imputation for consumption and production
#         data['hourly_energy_consumption'].replace(0, np.nan, inplace=True)
#         data['hourly_energy_production'].replace(0, np.nan, inplace=True)

#         return data
    
#     def get_data(self, filename):
#         raw_data = self.load_data(filename)
#         self.data = self.preprocess_data(raw_data)
#         return self.data
    
#     def get_all_meter_numbers(self):
#         if self.data is None:
#             raise ValueError("Data not loaded. Please load data first using get_data method.")
#         return self.data.index.get_level_values('meter_number').unique().tolist()
    
#     def get_data_by_meter(self, meter_number):
#         if self.data is None:
#             raise ValueError("Data not loaded. Please load data first using get_data method.")
#         return self.data.xs(meter_number, level='meter_number')

#     def examine_data(self):
#         if self.data is None:
#             raise ValueError("Data not loaded. Please load data first using get_data method.")
#         print("Data Info:")
#         print(self.data.info())
#         print("\nData Description:")
#         print(self.data.describe())
#         print("\nMissing Values:")
#         print(self.data.isnull().sum())

# if __name__ == "__main__":
#     data_dir = "data"
#     filename = "hourly_measurements_2023_12.csv"
#     raw_data_loader = RawData(data_dir)
#     data = raw_data_loader.get_data(filename)
#     print("Data loaded and preprocessed successfully.")
#     print(data.head())
#     raw_data_loader.examine_data()

#     # How many unique meter numbers are there?
#     unique_meters = raw_data_loader.get_all_meter_numbers()
#     print(f"Number of unique meter numbers: {len(unique_meters)}")

#     # Get data for a specific meter number
#     sample_meter_number = unique_meters[0]
#     meter_data = raw_data_loader.get_data_by_meter(sample_meter_number)
#     print(f"Data for meter number {sample_meter_number}:")
#     print(meter_data.head())
    
#     # Plot some consumption for a random 24 hours
#     sample_data = meter_data.head(24)
#     plt.figure(figsize=(10, 5))
#     plt.plot(sample_data.index, sample_data['hourly_energy_consumption'], marker='o')
#     plt.title(f'Hourly Energy Consumption for Meter {sample_meter_number} (First 24 Hours)')
#     plt.xlabel('Timestamp (UTC)')
#     plt.ylabel('Hourly Energy Consumption')
#     plt.grid()
#     plt.show()

