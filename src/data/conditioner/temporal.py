from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import BaseConditioner


class DayOfWeekConditioner(BaseConditioner):
    """Generates a one-hot encoded representation of the day of the week.

    Attributes:
        condition_dim (int): The dimension of the generated condition (7).
    """
    def __init__(self):
        self.condition_dim = 7

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Extracts day of the week from timestamps and returns a one-hot encoding.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps. If
                a tensor, assumes Unix timestamps (float).

        Returns:
            torch.Tensor: One-hot encoded days of shape `(batch_size, 7)`. Respects
                input tensor device if applicable.

        SHAPER:
          timestamps : [batch_size] (Unix seconds) or list[Timestamp]
          ->         : [batch_size, 7]
        """
        if timestamps is None:
            return None

        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            ts_cpu = timestamps.detach().to("cpu")
            if ts_cpu.is_floating_point():
                ts_cpu = torch.floor(ts_cpu)
            ts_cpu = ts_cpu.to(dtype=torch.long).numpy()
            dt = pd.to_datetime(ts_cpu, unit="s", utc=True, errors="coerce")
            dow = dt.dayofweek.astype("Int64").to_numpy()
            valid = ~pd.isna(dow)
            days_of_week = torch.from_numpy(dow[valid].astype(np.int64)).to(dtype=torch.long)
            if days_of_week.numel() != len(ts_cpu):
                padded = torch.zeros((len(ts_cpu),), dtype=torch.long)
                if valid.any():
                    padded[torch.from_numpy(np.nonzero(valid)[0]).long()] = days_of_week
                days_of_week = padded
        else:
            days_list = []
            for ts in timestamps:
                if isinstance(ts, (int, float, np.integer, np.floating)):
                    ts = pd.Timestamp(ts, unit="s", tz="UTC")
                days_list.append(ts.dayofweek)
            days_of_week = torch.tensor(days_list, dtype=torch.long)
            device = "cpu"

        dow_onehot = F.one_hot(days_of_week, num_classes=7).float().to(device)  # SHAPER: [batch_size, 7]
        return dow_onehot

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random day of week conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Random one-hot tensor of shape `(batch_size, 7)`.

        SHAPER:
          -> : [batch_size, 7]
        """
        random_days = torch.randint(0, 7, (batch_size,))  # SHAPER: [batch_size]
        dow_onehot = F.one_hot(random_days, num_classes=7).float()  # SHAPER: [batch_size, 7]
        return dow_onehot


class MonthOfYearConditioner(BaseConditioner):
    """Generates a one-hot encoded representation of the month of the year.

    Attributes:
        condition_dim (int): The dimension of the generated condition (12).
    """
    def __init__(self):
        self.condition_dim = 12

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Extracts month from timestamps and returns a one-hot encoding.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps.

        Returns:
            torch.Tensor: One-hot encoded months of shape `(batch_size, 12)`.

        SHAPER:
          timestamps : [batch_size] (Unix seconds) or list[Timestamp]
          ->         : [batch_size, 12]
        """
        if timestamps is None:
            return None

        if isinstance(timestamps, torch.Tensor):
            device = timestamps.device
            ts_cpu = timestamps.detach().to("cpu")
            if ts_cpu.is_floating_point():
                ts_cpu = torch.floor(ts_cpu)
            ts_cpu = ts_cpu.to(dtype=torch.long).numpy()
            dt = pd.to_datetime(ts_cpu, unit="s", utc=True, errors="coerce")
            month_idx = (dt.month.astype("Int64") - 1).to_numpy()
            valid = ~pd.isna(month_idx)
            months = torch.from_numpy(month_idx[valid].astype(np.int64)).to(dtype=torch.long)
            if months.numel() != len(ts_cpu):
                padded = torch.zeros((len(ts_cpu),), dtype=torch.long)
                if valid.any():
                    padded[torch.from_numpy(np.nonzero(valid)[0]).long()] = months
                months = padded
        else:
            months_list = []
            for ts in timestamps:
                if isinstance(ts, (int, float, np.integer, np.floating)):
                    ts = pd.Timestamp(ts, unit="s", tz="UTC")
                if isinstance(ts, pd.Timestamp):
                    months_list.append(ts.month - 1)
            months = torch.tensor(months_list, dtype=torch.long)
            device = "cpu"

        moy_onehot = F.one_hot(months, num_classes=12).float().to(device)  # SHAPER: [batch_size, 12]
        return moy_onehot

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random month conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Random one-hot tensor of shape `(batch_size, 12)`.

        SHAPER:
          -> : [batch_size, 12]
        """
        random_months = torch.randint(0, 12, (batch_size,))  # SHAPER: [batch_size]
        moy_onehot = F.one_hot(random_months, num_classes=12).float()  # SHAPER: [batch_size, 12]
        return moy_onehot


class HolidayConditioner(BaseConditioner):
    """Flags whether a given timestamp falls on a known holiday.
    As defined by: https://helligdage-dk.dk/helligdage-2024/

    Args:
        holiday_path (str): Path to JSON file containing holiday dates.
            Defaults to 'data/holidays/denmark_holidays.json'.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        holiday_dates (set): Set of loaded holiday dates.
    """
    def __init__(self, holiday_path='data/holidays/denmark_holidays.json'):
        import json
        self.condition_dim = 1
        with open(holiday_path, 'r') as f:
            holidays = json.load(f)

        parsed = pd.to_datetime(holidays.get('dates', []), errors='coerce', utc=True)
        parsed = parsed[~parsed.isna()]
        normalized_days = np.unique((parsed.normalize().asi8 // 10**9 // 86400).astype(np.int64))

        self.holiday_days_cpu = torch.from_numpy(normalized_days) if normalized_days.size > 0 else torch.empty(0, dtype=torch.long)
        self._holiday_days_by_device = {}

    def _get_holiday_days(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._holiday_days_by_device.get(key)
        if cached is not None and cached.device == device:
            return cached
        cached = self.holiday_days_cpu.to(device=device, dtype=torch.long)
        self._holiday_days_by_device[key] = cached
        return cached

    @staticmethod
    def _timestamps_to_unix_seconds(timestamps, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        if isinstance(timestamps, torch.Tensor):
            ts = timestamps
            if ts.is_floating_point():
                ts = torch.floor(ts)
            return ts.to(device=device, dtype=torch.long)

        if isinstance(timestamps, np.ndarray):
            if np.issubdtype(timestamps.dtype, np.datetime64):
                secs = timestamps.astype('datetime64[s]').astype(np.int64)
            else:
                secs = np.asarray(timestamps, dtype=np.int64)
            return torch.from_numpy(secs).to(device=device, dtype=torch.long)

        parsed = pd.to_datetime(timestamps, errors='coerce', utc=True)
        if isinstance(parsed, pd.Timestamp):
            parsed = pd.DatetimeIndex([parsed])

        secs = parsed.asi8 // 10**9
        return torch.from_numpy(secs.astype(np.int64)).to(device=device, dtype=torch.long)

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Checks if timestamps are holidays.

        Args:
            x (Any): Unused input data.
            meter_id (Any): Unused meter IDs.
            timestamps (torch.Tensor | list[int|float|pd.Timestamp]): Timestamps.

        Returns:
            torch.Tensor: Binary holiday indicators of shape `(batch_size, 1)`.

        SHAPER:
          timestamps : [batch_size] (Unix seconds) or list[Timestamp]
          ->         : [batch_size, 1]
        """
        device = timestamps.device if isinstance(timestamps, torch.Tensor) else torch.device('cpu')
        if timestamps is None:
            if isinstance(meter_id, torch.Tensor):
                return torch.zeros((meter_id.shape[0], 1), dtype=torch.float32, device=meter_id.device)
            if isinstance(meter_id, np.ndarray):
                return torch.zeros((meter_id.shape[0], 1), dtype=torch.float32, device=device)
            return torch.zeros((1, 1), dtype=torch.float32, device=device)

        unix_seconds = self._timestamps_to_unix_seconds(timestamps, device=device)  # SHAPER: [batch_size]
        day_numbers = torch.div(unix_seconds, 86400, rounding_mode='floor').to(dtype=torch.long, device=device)  # SHAPER: [batch_size]

        holidays = self._get_holiday_days(device=device)
        if holidays.numel() == 0:
            return torch.zeros((day_numbers.shape[0], 1), dtype=torch.float32, device=device)

        is_holiday = torch.isin(day_numbers, holidays).to(dtype=torch.float32, device=device)  # SHAPER: [batch_size]
        return is_holiday.unsqueeze(1)  # SHAPER: [batch_size, 1]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random holiday conditions for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Binary tensor of shape `(batch_size, 1)`.
        """
        random_holiday = torch.randint(0, 2, (batch_size, 1)).float()  # SHAPER: [batch_size, 1]
        return random_holiday