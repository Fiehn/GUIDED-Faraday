from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseConditioner(ABC):
    """Abstract base class for all conditioners.

    Any custom conditioner must inherit from this class and implement
    `get_conditions` and `get_condition_dim`.
    """

    def __init__(self):
        self.condition_dim: int = 0

    @abstractmethod
    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """Generates the conditioning tensor based on input"""
        pass

    @abstractmethod
    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        """Generates random conditions for testing and validation"""
        pass

    def to(self, device: str | torch.device):
        """Moves all internal torch tensors to the specified device."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
            elif isinstance(attr_value, dict):
                for k, v in attr_value.items():
                    if isinstance(v, torch.Tensor):
                        attr_value[k] = v.to(device)
            elif isinstance(attr_value, list):
                for i, v in enumerate(attr_value):
                    if isinstance(v, torch.Tensor):
                        attr_value[i] = v.to(device)
        return self