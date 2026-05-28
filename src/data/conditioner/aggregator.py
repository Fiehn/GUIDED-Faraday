from __future__ import annotations

import pickle

import numpy as np
import torch

from .base import BaseConditioner


class Conditioner:
    """Central manager that aggregates multiple condition modules.

    Holds objects that create and determine the type of conditioning used in models.
    Concatenates individual conditions into a single tensor.

    Args:
        tags (list[str], optional): List of active condition tags. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        tags (list[str]): Registered condition tags (e.g., "user_profiles", "days").
        condition_dim (int): Total concatenated dimension of all active conditions.
        condition_items (dict): Mapping of tags to their respective conditioner instances.
    """
    def __init__(self, tags: list[str] | None = None, **kwargs):
        self.tags = tags if tags is not None else []
        self.condition_dim = 0
        self.condition_items = {}

    def get_condition_tags(self) -> list[str]:
        """Retrieves list of active condition tags.

        Returns:
            list[str]: Tags registered in the Conditioner.
        """
        return self.tags

    def add_condition_item(self, tag: str, item):
        """Registers a new conditioner item to the Conditioner.

        Args:
            tag (str): Identifier for the condition (e.g., "holidays").
            item (Any): An instantiated conditioner object that implements
                `get_conditions` and has a `condition_dim` attribute. (e.g. HolidayConditioner())
        """
        self.tags.append(tag)
        self.condition_items[tag] = item
        self.condition_dim += item.condition_dim

    def to(self, device: str | torch.device):
        """Moves all internally registered Conditioners to the specified device."""
        for item in self.condition_items.values():
            if hasattr(item, "to") and callable(getattr(item, "to")):
                item.to(device)
        return self

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None = None,
        meter_id: torch.Tensor | np.ndarray | None = None,
        timestamps: torch.Tensor | np.ndarray | None = None
    ):
        """Gets and concatenates conditions from all registered conditioners.

        Args:
            x (torch.Tensor | np.ndarray): Input data. Used to infer target device.
            meter_id (torch.Tensor | np.ndarray, optional): Meter IDs.
            timestamps (torch.Tensor | list, optional): Timestamps.

        Returns:
            torch.Tensor | None: Concatenated condition tensor of shape
                `(batch_size, total_condition_dim)`. Returns None if no conditions
                are registered.

        SHAPER:
          x          : [batch_size, seq_len, num_features] or None
          meter_id   : [batch_size] or None
          timestamps : [batch_size] or None
          ->         : [batch_size, total_condition_dim]  (cat of each conditioner's output)
        """
        conditions = []
        target_device = None
        if isinstance(x, torch.Tensor):
            target_device = x.device
        elif isinstance(x, np.ndarray):
            target_device = torch.device('cpu')
        elif isinstance(meter_id, torch.Tensor):
            target_device = meter_id.device
        elif isinstance(meter_id, np.ndarray):
            target_device = torch.device('cpu')
        elif isinstance(timestamps, torch.Tensor):
            target_device = timestamps.device
        elif isinstance(timestamps, np.ndarray):
            target_device = torch.device('cpu')

        for tag in self.tags:
            cond = self.condition_items[tag].get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            if cond is None:
                continue
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond, dtype=torch.float32)
            if target_device is not None and cond.device != target_device:
                cond = cond.to(target_device, non_blocking=True)
            conditions.append(cond)

        if len(conditions) == 0:
            if target_device is not None:
                return torch.empty((x.shape[0] if x is not None else 1, 0), device=target_device)
            else:
                return torch.empty((x.shape[0] if x is not None else 1, 0))

        return torch.cat(conditions, dim=-1)  # SHAPER: [batch_size, total_condition_dim]

    def get_random_conditions(self, batch_size: int):
        """Generates random conditions concatenated from all active items for testing.

        Args:
            batch_size (int): Number of conditions to generate.

        Returns:
            torch.Tensor: Concatenated random conditions of shape
                `(batch_size, total_condition_dim)`.
        """
        conditions = []
        for tag in self.tags:
            item = self.condition_items[tag]
            conditions.append(item.get_random_conditions(batch_size))

        if len(conditions) == 0:
            return torch.empty((batch_size, 0))
        return torch.cat(conditions, dim=-1)  # SHAPER: [batch_size, total_condition_dim]

    def get_condition_dim(self):
        """Returns the total aggregated condition dimension.

        Returns:
            int: Size of condition vector
        """
        return int(self.condition_dim)

    def get_condition_indexes(self, tag: str):
        """Calculates the feature indices for a specific condition in the concatenated tensor.

        Args:
            tag (str): The condition tag to locate.

        Returns:
            tuple: Start and end indices `(start_idx, end_idx)`.

        Raises:
            ValueError: If the tag is not found in the registered tags.
        """
        if tag not in self.tags:
            raise ValueError(f"Condition tag '{tag}' not found in Conditioner.")
        idx = 0
        for t in self.tags:
            if t == tag:
                return (idx, idx + self.condition_items[t].condition_dim)
            idx += self.condition_items[t].condition_dim
        raise ValueError(f"Condition tag '{tag}' not found in Conditioner.")

    def save(self, path: str):
        """Saves the entire Conditioner state to a pickle file.

        Args:
            path (str): Filepath to save the model.
        """
        self.path = path
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Loads a Conditioner instance from a pickle file.

        Args:
            path (str): Filepath to load the model from.

        Returns:
            Conditioner: The unpickled Conditioner instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj