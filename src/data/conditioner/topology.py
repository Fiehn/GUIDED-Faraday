from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import BaseConditioner


class HeatpumpConditioner(BaseConditioner):
    """Flags whether a specific meter is associated with a heat pump.
    IN TOPOLOGY DATASET, NOT INFERRED FROM TIME SERIES. POSSIBLY NOT THAT GREAT

    Args:
        topology_path (str): Path to CSV file mapping meters to heat pump status.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        topology_df (pd.DataFrame): Loaded topology data.
    """
    def __init__(self, topology_path='/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        topology_df = pd.read_csv(topology_path)

        meter_col = pd.to_numeric(topology_df.get('meter_number'), errors='coerce')
        value_series = topology_df.get('has_heat_pump')
        if value_series is None:
            status_series = pd.Series(np.zeros(len(topology_df), dtype=np.int64))
        else:
            if pd.api.types.is_bool_dtype(value_series):
                status_series = value_series.fillna(False).astype(np.int64)
            else:
                status_series = value_series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'}).astype(np.int64)

        compact = pd.DataFrame({'meter_id': meter_col, 'status': status_series})
        compact = compact.dropna(subset=['meter_id']).drop_duplicates(subset=['meter_id'], keep='last')

        meter_ids_np = compact['meter_id'].astype(np.int64).to_numpy()
        status_np = compact['status'].astype(np.int64).to_numpy()
        order = np.argsort(meter_ids_np)

        self._meter_ids_cpu = torch.from_numpy(meter_ids_np[order]) if meter_ids_np.size > 0 else torch.empty(0, dtype=torch.long)
        self._status_cpu = torch.from_numpy(status_np[order]) if status_np.size > 0 else torch.empty(0, dtype=torch.long)
        self._lookup_by_device = {}

    def _get_lookup(self, device: torch.device):
        key = str(device)
        cached = self._lookup_by_device.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        cached = (
            self._meter_ids_cpu.to(device=device, dtype=torch.long),
            self._status_cpu.to(device=device, dtype=torch.long),
        )
        self._lookup_by_device[key] = cached
        return cached

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Retrieves heat pump status for given meter IDs.

        Args:
            x (Any): Unused input data.
            meter_id (torch.Tensor | np.ndarray): Meter IDs.
            timestamps (Any): Unused timestamps.

        Returns:
            torch.Tensor: Binary indicators of shape `(batch_size, 1)`. Defaults to
                0.0 if meter is not found.

        SHAPER:
          meter_id : [batch_size]
          ->       : [batch_size, 1]
        """
        if meter_id is None:
            if isinstance(timestamps, torch.Tensor):
                return torch.zeros((timestamps.shape[0], 1), dtype=torch.float32, device=timestamps.device)
            if isinstance(timestamps, np.ndarray):
                return torch.zeros((timestamps.shape[0], 1), dtype=torch.float32)
            return torch.zeros((1, 1), dtype=torch.float32)

        if isinstance(meter_id, torch.Tensor):
            meter_ids = meter_id.to(dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device
        else:
            meter_ids = torch.as_tensor(np.asarray(meter_id, dtype=np.int64), dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device

        lookup_ids, lookup_status = self._get_lookup(device=device)

        if lookup_ids.numel() == 0:
            return torch.zeros((meter_ids.shape[0], 1), dtype=torch.float32, device=device)

        meter_ids = meter_ids.contiguous()
        lookup_ids = lookup_ids.contiguous()
        idx = torch.searchsorted(lookup_ids, meter_ids)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids)

        status = torch.zeros(meter_ids.shape[0], dtype=torch.float32, device=device)  # SHAPER: [batch_size]
        if valid.any():
            status[valid] = lookup_status[idx_clamped[valid]].to(dtype=torch.float32)

        return status.unsqueeze(1)  # SHAPER: [batch_size, 1]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        random_status = torch.randint(0, 2, (batch_size, 1)).float()  # SHAPER: [batch_size, 1]
        return random_status


class SolarPanelConditioner(BaseConditioner):
    """Flags whether a specific meter is associated with solar panels.

    Args:
        topology_path (str): Path to CSV file mapping meters to solar panel status.

    Attributes:
        condition_dim (int): The dimension of the generated condition (1).
        topology_df (pd.DataFrame): Loaded topology data.
    """
    def __init__(self, topology_path='/media/fiehn/LaCie/Data/processed_data/second_batch/topology/cleaned_and_corrected/MeterAndCabinet.csv'):
        self.condition_dim = 1
        topology_df = pd.read_csv(topology_path)

        meter_col = pd.to_numeric(topology_df.get('meter_number'), errors='coerce')
        value_series = topology_df.get('has_solar_panel')
        if value_series is None:
            status_series = pd.Series(np.zeros(len(topology_df), dtype=np.int64))
        else:
            if pd.api.types.is_bool_dtype(value_series):
                status_series = value_series.fillna(False).astype(np.int64)
            else:
                status_series = value_series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'}).astype(np.int64)

        compact = pd.DataFrame({'meter_id': meter_col, 'status': status_series})
        compact = compact.dropna(subset=['meter_id']).drop_duplicates(subset=['meter_id'], keep='last')

        meter_ids_np = compact['meter_id'].astype(np.int64).to_numpy()
        status_np = compact['status'].astype(np.int64).to_numpy()
        order = np.argsort(meter_ids_np)

        self._meter_ids_cpu = torch.from_numpy(meter_ids_np[order]) if meter_ids_np.size > 0 else torch.empty(0, dtype=torch.long)
        self._status_cpu = torch.from_numpy(status_np[order]) if status_np.size > 0 else torch.empty(0, dtype=torch.long)
        self._lookup_by_device = {}

    def _get_lookup(self, device: torch.device):
        key = str(device)
        cached = self._lookup_by_device.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        cached = (
            self._meter_ids_cpu.to(device=device, dtype=torch.long),
            self._status_cpu.to(device=device, dtype=torch.long),
        )
        self._lookup_by_device[key] = cached
        return cached

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Retrieves solar panel status for given meter IDs.

        Args:
            x (Any): Unused input data.
            meter_id (torch.Tensor | np.ndarray): Meter IDs.
            timestamps (Any): Unused timestamps.

        Returns:
            torch.Tensor: Binary indicators of shape `(batch_size, 1)`. Defaults to
                0.0 if meter is not found.

        SHAPER:
          meter_id : [batch_size]
          ->       : [batch_size, 1]
        """
        if meter_id is None:
            if isinstance(timestamps, torch.Tensor):
                return torch.zeros((timestamps.shape[0], 1), dtype=torch.float32, device=timestamps.device)
            if isinstance(timestamps, np.ndarray):
                return torch.zeros((timestamps.shape[0], 1), dtype=torch.float32)
            return torch.zeros((1, 1), dtype=torch.float32)

        if isinstance(meter_id, torch.Tensor):
            meter_ids = meter_id.to(dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device
        else:
            meter_ids = torch.as_tensor(np.asarray(meter_id, dtype=np.int64), dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device

        lookup_ids, lookup_status = self._get_lookup(device=device)

        if lookup_ids.numel() == 0:
            return torch.zeros((meter_ids.shape[0], 1), dtype=torch.float32, device=device)

        meter_ids = meter_ids.contiguous()
        lookup_ids = lookup_ids.contiguous()
        idx = torch.searchsorted(lookup_ids, meter_ids)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids)

        status = torch.zeros(meter_ids.shape[0], dtype=torch.float32, device=device)  # SHAPER: [batch_size]
        if valid.any():
            status[valid] = lookup_status[idx_clamped[valid]].to(dtype=torch.float32)

        return status.unsqueeze(1)  # SHAPER: [batch_size, 1]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        random_status = torch.randint(0, 2, (batch_size, 1)).float()  # SHAPER: [batch_size, 1]
        return random_status


class ElectricVehicleConditioner(BaseConditioner):
    """
    Conditions the model on EV class one-hot encoded as:
    [No EV, 1-phase EV, 2-phase EV, 3-phase EV].

    Reads EV detector output from CSV (default: data/EV/EV_detection_reduced_60000_SM.csv)
    and maps base meter ids to an EV phase class.
    """
    def __init__(self, topology_path='data/EV/EV_detection_reduced_60000_SM.csv'):
        self.condition_dim = 4
        self._meter_ids_cpu, self._phase_class_cpu = self._load_topology_df(topology_path)
        self._lookup_by_device = {}

    def _load_topology_df(self, topology_path):
        topology_df = pd.read_csv(topology_path)

        meter_col = None
        if 'SM_ID' in topology_df.columns:
            meter_col = topology_df['SM_ID']
        elif 'meter_id' in topology_df.columns:
            meter_col = topology_df['meter_id']
        else:
            meter_col = topology_df.iloc[:, 0]

        meter_key = meter_col.astype(str).str.lower()
        base_meter_id = pd.to_numeric(meter_key.str.extract(r'(\d+)', expand=False), errors='coerce')

        ev_detected = topology_df.get('EV_detected', pd.Series(['no'] * len(topology_df))).astype(str).str.strip().str.lower()
        is_ev = ev_detected.isin({'yes', 'true', '1'})

        # Prefer explicit phase label from detector output (1ph/2ph/3ph).
        type_series = topology_df.get('Type', pd.Series(['none'] * len(topology_df))).astype(str).str.strip().str.lower()
        phase_from_type = pd.to_numeric(type_series.str.extract(r'([123])\s*ph', expand=False), errors='coerce')

        phase_class = np.zeros(len(topology_df), dtype=np.int64)
        known_type = phase_from_type.notna().to_numpy()
        phase_class[known_type] = phase_from_type[known_type].astype(np.int64)

        # Any detected EV with no phase evidence defaults to 1-phase.
        undecided_detected = is_ev.to_numpy() & (phase_class == 0)
        phase_class[undecided_detected] = 1

        # Non-EV stays class 0 regardless of noisy phase values.
        phase_class[~is_ev.to_numpy()] = 0
        phase_class = np.clip(phase_class, 0, 3)

        compact = pd.DataFrame(
            {'base_meter_id': base_meter_id, 'phase_class': phase_class}
        ).dropna(subset=['base_meter_id'])

        if compact.empty:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        compact['base_meter_id'] = compact['base_meter_id'].astype(np.int64)
        compact = compact.groupby('base_meter_id', as_index=False)['phase_class'].max()

        meter_ids_np = compact['base_meter_id'].to_numpy(dtype=np.int64)
        phase_classes_np = compact['phase_class'].to_numpy(dtype=np.int64)
        order = np.argsort(meter_ids_np)
        return (
            torch.from_numpy(meter_ids_np[order]),
            torch.from_numpy(phase_classes_np[order]),
        )

    def _get_lookup(self, device: torch.device):
        key = str(device)
        cached = self._lookup_by_device.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        cached = (
            self._meter_ids_cpu.to(device=device, dtype=torch.long),
            self._phase_class_cpu.to(device=device, dtype=torch.long),
        )
        self._lookup_by_device[key] = cached
        return cached

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """
        SHAPER:
          meter_id : [batch_size]
          ->       : [batch_size, 4]  (one-hot: no_ev, 1ph, 2ph, 3ph)
        """
        if meter_id is None:
            if isinstance(timestamps, torch.Tensor):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.long, device=timestamps.device)
            elif isinstance(timestamps, np.ndarray):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.long)
            else:
                zeros = torch.zeros(1, dtype=torch.long)
            return F.one_hot(zeros, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

        if isinstance(meter_id, torch.Tensor):
            meter_ids = meter_id.to(dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device
        else:
            meter_ids = torch.as_tensor(np.asarray(meter_id, dtype=np.int64), dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device

        lookup_ids, lookup_phase_class = self._get_lookup(device=device)
        if lookup_ids.numel() == 0:
            phase_class = torch.zeros_like(meter_ids, dtype=torch.long)  # SHAPER: [batch_size]
            return F.one_hot(phase_class, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

        meter_ids = meter_ids.contiguous()
        lookup_ids = lookup_ids.contiguous()
        idx = torch.searchsorted(lookup_ids, meter_ids)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids)

        phase_class = torch.zeros_like(meter_ids, dtype=torch.long)  # SHAPER: [batch_size]
        if valid.any():
            phase_class[valid] = lookup_phase_class[idx_clamped[valid]]

        return F.one_hot(phase_class, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        # No EV: 70%, 1-phase: 15%, 2-phase: 10%, 3-phase: 5%
        probs = torch.tensor([0.7, 0.15, 0.1, 0.05])
        random_status = torch.multinomial(probs, batch_size, replacement=True)  # SHAPER: [batch_size]
        return F.one_hot(random_status, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

    def get_conditions_no_data(self, meter_id, timestamps):
        # Backward-compatible helper used in some older call sites.
        return self.get_conditions(x=None, meter_id=meter_id, timestamps=timestamps)


class ElectricVehicleConditionerBool(BaseConditioner):
    """
    Conditions the model on EV presence one-hot encoded as:
    [No EV, Has EV].

    Reads EV detector output from CSV (default: data/EV/EV_detection_reduced_60000_SM.csv)
    and maps base meter ids to an EV presence class.
    """
    def __init__(self, topology_path='data/EV/EV_detection_reduced_60000_SM.csv'):
        self.condition_dim = 1
        self._meter_ids_cpu, self._phase_class_cpu = self._load_topology_df(topology_path)
        self._lookup_by_device = {}

    def _load_topology_df(self, topology_path):
        topology_df = pd.read_csv(topology_path)

        meter_col = None
        if 'SM_ID' in topology_df.columns:
            meter_col = topology_df['SM_ID']
        elif 'meter_id' in topology_df.columns:
            meter_col = topology_df['meter_id']
        else:
            meter_col = topology_df.iloc[:, 0]

        meter_key = meter_col.astype(str).str.lower()
        base_meter_id = pd.to_numeric(meter_key.str.extract(r'(\d+)', expand=False), errors='coerce')

        ev_detected = topology_df.get('EV_detected', pd.Series(['no'] * len(topology_df))).astype(str).str.strip().str.lower()
        is_ev = ev_detected.isin({'yes', 'true', '1'})
        ev_class = is_ev.astype(np.int64).to_numpy()

        compact = pd.DataFrame(
            {'base_meter_id': base_meter_id, 'phase_class': ev_class}
        ).dropna(subset=['base_meter_id'])

        if compact.empty:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        compact['base_meter_id'] = compact['base_meter_id'].astype(np.int64)
        compact = compact.groupby('base_meter_id', as_index=False)['phase_class'].max()

        meter_ids_np = compact['base_meter_id'].to_numpy(dtype=np.int64)
        phase_classes_np = compact['phase_class'].to_numpy(dtype=np.int64)
        order = np.argsort(meter_ids_np)
        return (
            torch.from_numpy(meter_ids_np[order]),
            torch.from_numpy(phase_classes_np[order]),
        )

    def _get_lookup(self, device: torch.device):
        key = str(device)
        cached = self._lookup_by_device.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        cached = (
            self._meter_ids_cpu.to(device=device, dtype=torch.long),
            self._phase_class_cpu.to(device=device, dtype=torch.long),
        )
        self._lookup_by_device[key] = cached
        return cached

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """
        SHAPER:
          meter_id : [batch_size]
          ->       : [batch_size, 1]  (binary: 0=no_ev, 1=has_ev)
        """
        if meter_id is None:
            if isinstance(timestamps, torch.Tensor):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.float32, device=timestamps.device)
            elif isinstance(timestamps, np.ndarray):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.float32)
            else:
                zeros = torch.zeros(1, dtype=torch.float32)
            return zeros.unsqueeze(1)  # SHAPER: [batch_size, 1]

        if isinstance(meter_id, torch.Tensor):
            meter_ids = meter_id.to(dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device
        else:
            meter_ids = torch.as_tensor(np.asarray(meter_id, dtype=np.int64), dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device

        lookup_ids, lookup_phase_class = self._get_lookup(device=device)
        if lookup_ids.numel() == 0:
            phase_class = torch.zeros_like(meter_ids, dtype=torch.float32)  # SHAPER: [batch_size]
            return phase_class.unsqueeze(1)  # SHAPER: [batch_size, 1]

        meter_ids = meter_ids.contiguous()
        lookup_ids = lookup_ids.contiguous()
        idx = torch.searchsorted(lookup_ids, meter_ids)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids)

        phase_class = torch.zeros_like(meter_ids, dtype=torch.float32)  # SHAPER: [batch_size]
        if valid.any():
            phase_class[valid] = lookup_phase_class[idx_clamped[valid]].to(dtype=torch.float32)

        return phase_class.unsqueeze(1)  # SHAPER: [batch_size, 1]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        # No EV: 86%, Has EV: 14%
        probs = torch.tensor([0.86, 0.14])
        random_status = torch.multinomial(probs, batch_size, replacement=True)  # SHAPER: [batch_size]
        return random_status.unsqueeze(1).float()  # SHAPER: [batch_size, 1]

    def get_conditions_no_data(self, meter_id, timestamps):
        # Backward-compatible helper used in some older call sites.
        return self.get_conditions(x=None, meter_id=meter_id, timestamps=timestamps)


class HeatpumpPhaseConditioner(BaseConditioner):
    """
    Conditions the model on what kind of heat pump it has one-hot encoded as:
    [No heat pump, 1-phase heat pump, 2-phase heat pump, 3-phase heat pump]

    Checks the label propagation from path with index like "<PHASE><METER_ID>" and columns predicted_hp : bool

    Then assigns dependent on the amount of active phases.
    """
    def __init__(self, topology_path='data/HP/label_propagation_predictions.csv'):
        self.condition_dim = 4
        self._meter_ids_cpu, self._phase_class_cpu = self._load_topology_df(topology_path)
        self._lookup_by_device = {}

    def _load_topology_df(self, topology_path):
        topology_df = pd.read_csv(topology_path)
        meter_col = None
        if 'Unnamed: 0' in topology_df.columns:
            meter_col = topology_df['Unnamed: 0']
        elif 'meter_id' in topology_df.columns:
            meter_col = topology_df['meter_id']
        else:
            meter_col = topology_df.iloc[:, 0]

        meter_key = meter_col.astype(str).str.lower()
        phase = meter_key.str.extract(r'^(l\d)_', expand=False)
        base_meter_id = pd.to_numeric(meter_key.str.extract(r'^l\d_(\d+)$', expand=False), errors='coerce')

        predicted_series = topology_df.get('predicted_hp', pd.Series(np.zeros(len(topology_df), dtype=np.int64)))
        if pd.api.types.is_bool_dtype(predicted_series):
            predicted_bool = predicted_series.fillna(False)
        else:
            predicted_bool = predicted_series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})

        compact = pd.DataFrame(
            {'base_meter_id': base_meter_id, 'phase': phase, 'predicted_hp': predicted_bool}
        ).dropna(subset=['base_meter_id', 'phase'])

        hp_counts = compact[compact['predicted_hp']].groupby('base_meter_id')['phase'].nunique()

        if hp_counts.empty:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        meter_ids_np = hp_counts.index.to_numpy(dtype=np.int64)
        phase_classes_np = np.clip(hp_counts.to_numpy(dtype=np.int64), 0, 3)
        order = np.argsort(meter_ids_np)
        return (
            torch.from_numpy(meter_ids_np[order]),
            torch.from_numpy(phase_classes_np[order]),
        )

    def _get_lookup(self, device: torch.device):
        key = str(device)
        cached = self._lookup_by_device.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        cached = (
            self._meter_ids_cpu.to(device=device, dtype=torch.long),
            self._phase_class_cpu.to(device=device, dtype=torch.long),
        )
        self._lookup_by_device[key] = cached
        return cached

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """
        SHAPER:
          meter_id : [batch_size]
          ->       : [batch_size, 4]  (one-hot: no_hp, 1ph, 2ph, 3ph)
        """
        if meter_id is None:
            if isinstance(timestamps, torch.Tensor):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.long, device=timestamps.device)
            elif isinstance(timestamps, np.ndarray):
                zeros = torch.zeros(timestamps.shape[0], dtype=torch.long)
            else:
                zeros = torch.zeros(1, dtype=torch.long)
            return F.one_hot(zeros, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

        if isinstance(meter_id, torch.Tensor):
            meter_ids = meter_id.to(dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device
        else:
            meter_ids = torch.as_tensor(np.asarray(meter_id, dtype=np.int64), dtype=torch.long)  # SHAPER: [batch_size]
            device = meter_ids.device

        lookup_ids, lookup_phase_class = self._get_lookup(device=device)
        if lookup_ids.numel() == 0:
            phase_class = torch.zeros_like(meter_ids, dtype=torch.long)  # SHAPER: [batch_size]
            return F.one_hot(phase_class, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

        meter_ids = meter_ids.contiguous()
        lookup_ids = lookup_ids.contiguous()
        idx = torch.searchsorted(lookup_ids, meter_ids)
        idx_clamped = idx.clamp(max=lookup_ids.shape[0] - 1)
        valid = (idx < lookup_ids.shape[0]) & (lookup_ids[idx_clamped] == meter_ids)

        phase_class = torch.zeros_like(meter_ids, dtype=torch.long)  # SHAPER: [batch_size]
        if valid.any():
            phase_class[valid] = lookup_phase_class[idx_clamped[valid]]

        return F.one_hot(phase_class, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        # No heat pump: 50%, 1-phase: 20%, 2-phase: 20%, 3-phase: 10%
        probs = torch.tensor([0.5, 0.2, 0.2, 0.1])
        random_status = torch.multinomial(probs, batch_size, replacement=True)  # SHAPER: [batch_size]
        return F.one_hot(random_status, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]