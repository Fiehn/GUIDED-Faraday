from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseConditioner


class ActivePhaseCountConditioner(BaseConditioner):
    """
    Conditions the model on HOW MANY phases are active in the current window.
    Outputs a one-hot vector of length 4 (representing 0, 1, 2, or 3 active phases).
    """
    def __init__(self, activity_threshold=1e-5):
        super().__init__()
        self.condition_dim = 4
        # Anything with an absolute sum below this threshold is considered "dead"
        self.activity_threshold = activity_threshold

    def get_conditions(
        self,
        x: torch.Tensor | np.ndarray | None,
        meter_id: torch.Tensor | np.ndarray | None,
        timestamps: torch.Tensor | list | None
    ) -> torch.Tensor | None:
        """
        SHAPER:
          x  : [batch_size, seq_len, num_phases] or [batch_size, num_phases, seq_len] or [batch_size, seq_len * num_phases]
          ->  : [batch_size, 4]  (one-hot: 0, 1, 2, or 3 active phases)
        """
        if x is None:
            if isinstance(meter_id, torch.Tensor):
                batch_size = int(meter_id.shape[0])
                device = meter_id.device
            elif isinstance(meter_id, np.ndarray):
                batch_size = int(meter_id.shape[0])
                device = torch.device("cpu")
            elif isinstance(timestamps, torch.Tensor):
                batch_size = int(timestamps.shape[0])
                device = timestamps.device
            elif isinstance(timestamps, np.ndarray):
                batch_size = int(timestamps.shape[0])
                device = torch.device("cpu")
            else:
                batch_size = 1
                device = torch.device("cpu")

            active_count = torch.full((batch_size,), 3, dtype=torch.long, device=device)  # SHAPER: [batch_size]
            count_onehot = F.one_hot(active_count, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]
            return count_onehot

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if not torch.is_tensor(x):
            raise ValueError("ActivePhaseCountConditioner expects a tensor or ndarray input.")

        feature_dim = self.condition_dim - 1  # num_phases = 3
        used_transpose = False

        if x.dim() == 2:
            if x.shape[1] % feature_dim != 0:
                raise ValueError(
                    f"Expected flattened input with feature_dim={feature_dim}, got shape {tuple(x.shape)}"
                )
            x_time_first = x.reshape(x.shape[0], -1, feature_dim)  # SHAPER: [batch_size, seq_len, num_phases]
        elif x.dim() == 3:
            if x.shape[2] == feature_dim:
                x_time_first = x  # SHAPER: [batch_size, seq_len, num_phases]
            elif x.shape[1] == feature_dim:
                x_time_first = x.transpose(1, 2)  # SHAPER: [batch_size, seq_len, num_phases]
                used_transpose = True
            else:
                x_time_first = x  # SHAPER: [batch_size, seq_len, num_phases] (best guess)
        else:
            raise ValueError(f"Expected 2D/3D input, got shape {tuple(x.shape)}")

        phase_activity = x_time_first.abs().sum(dim=1)  # SHAPER: [batch_size, num_phases]
        is_active = phase_activity > self.activity_threshold  # SHAPER: [batch_size, num_phases]
        active_count = is_active.sum(dim=1).long()  # SHAPER: [batch_size]

        if active_count.numel() > 0 and int(active_count.max().item()) >= self.condition_dim:
            if x.dim() == 3 and not used_transpose and x.shape[1] != x.shape[2]:
                x_time_first = x.transpose(1, 2)  # SHAPER: [batch_size, seq_len, num_phases]
                phase_activity = x_time_first.abs().sum(dim=1)  # SHAPER: [batch_size, num_phases]
                is_active = phase_activity > self.activity_threshold  # SHAPER: [batch_size, num_phases]
                active_count = is_active.sum(dim=1).long()  # SHAPER: [batch_size]

        if active_count.numel() > 0 and int(active_count.max().item()) >= self.condition_dim:
            print(
                f"Warning: active phase count exceeded expected max ({self.condition_dim - 1}); "
                "clamping to valid range."
            )
            active_count = active_count.clamp(max=self.condition_dim - 1)  # SHAPER: [batch_size]

        count_onehot = F.one_hot(active_count, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]
        return count_onehot.to(x_time_first.device)

    def get_random_conditions(self, batch_size: int) -> torch.Tensor | None:
        # 0 active: 0%, 1 active: 20%, 2 active: 30%, 3 active: 50%
        probs = torch.tensor([0.0, 0.2, 0.3, 0.5])
        random_counts = torch.multinomial(probs, batch_size, replacement=True)  # SHAPER: [batch_size]
        return F.one_hot(random_counts, num_classes=self.condition_dim).float()  # SHAPER: [batch_size, 4]