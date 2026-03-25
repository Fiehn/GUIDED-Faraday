import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Any

class SyntheticGenerator:
    """Called to generate synthetic samples from a trained model.
    This class provides methods to generate synthetic data using a trained generative model (like a CVAE).
    It supports:
    - Reconstructing input data with optional
        new conditions, allowing for manipulation of the reconstruction by changing the conditions.
    - Generating new samples for a single day based on specified conditions (meter ID, date, or pre-built conditions).
    - Generating continuous multi-day sequences by sampling multiple candidates for each day and selecting the one that best continues from the previous day, ensuring smoother transitions at day boundaries.

    Args:
        model: A trained generative model (e.g., CVAE) that has a .sample() method and optionally: [.conditioner, .encoder, .decoder, .hierarchy].
        conditioner: Optional Conditioner object to build conditions for generation. If None, it will try to use model.conditioner.
        normalizer: Optional Normalizer object to inverse-transform generated data back to original scale.
        device: Device to run generation on ('cpu' or 'cuda').

    Attributes:
        model: The generative model used for sampling.
        conditioner: The conditioner used to build conditions for generation.
        normalizer: The normalizer used to inverse-transform generated data.
        device: The device on which the model and data are placed for generation.
        _is_cvae: Boolean flag indicating if the model is a CVAE (i.e., has a conditioner).
    """

    def __init__(self,
                 model: Any,
                 conditioner: Any | None = None,
                 normalizer: Any | None = None,
                 device: str = 'cpu') -> None:
        """Initialize synthetic sample generator.

        Args:
            model: Trained generative model exposing `sample` and optionally conditioner hooks.
            conditioner: Optional explicit conditioner. Falls back to `model.conditioner`.
            normalizer: Optional normalizer used for inverse scaling.
            device: Device string for generation (`cpu` or `cuda`).
        """
        self.model = model
        self.conditioner = conditioner if conditioner is not None else getattr(model, 'conditioner', None)
        self.normalizer = normalizer
        self.device = device
        self._is_cvae = hasattr(model, 'conditioner') and model.conditioner is not None

    @torch.no_grad()
    def reconstruct(self, input_data: torch.Tensor,
                    num_reconstructions: int = 1,
                    meter_id: int | None = None, 
                    timestamp: datetime | torch.Tensor | None = None,
                    new_conditions: torch.Tensor | None = None,) -> torch.Tensor | None:
        """
        Reconstruct input data with optional new conditions. If new_conditions is provided,
        it will be used for the decoding and thus providing a way to manipulate the reconstruction by changing the conditions.

        Args:
            input_data (torch.Tensor): Tensor of original timeseries data to reconstruct of shape `(batch_size, 24, num_features)`.
            new_conditions (Optional[torch.Tensor]): Tensor of new conditions to use for reconstruction of shape `(batch_size, num_conditions)`.
                            If None, original conditions from the model will be used.
        Returns:
            Reconstructed data tensor shaped `(num_reconstructions, batch_size, 24, num_features)`.
        """
        self.model.eval()
        self.model.to(self.device)
        input_data = input_data.to(self.device)
        if self.normalizer is not None:
            input_data = self.normalizer.transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)
        if self._is_cvae and meter_id is not None and timestamp is not None:
            # Conditions for the encoding are always based on the original meter_id and timestamp, to get the correct latent representation.
            conditions = self.conditioner.get_conditions(input_data, meter_id, timestamp).to(self.device)
            
            # Ensure input_data is flattened appropriately
            input_data_flat = input_data.reshape(input_data.shape[0], -1).to(self.device)
            enc_in = torch.cat([input_data_flat, conditions], dim=-1)
            q_params = self.model.encoder(enc_in) # dict of [B, latent_dim] params

            # Sample latent variables using reparameterization trick - sample z: [S, B, latent_dim]
            z = self.model.encoder.reparameterize(q_params, num_samples=1)
    
            hierarchy_results = None
            if self.model.hierarchy is not None:
                S, B, D = z.shape
                z_flat = z.reshape(S * B, D)
                hierarchy_results = self.model.hierarchy.forward(z_flat)
                z_for_decoder = hierarchy_results["z"].reshape(S, B, -1)
            else:
                z_for_decoder = z
            self._last_hierarchy_results = hierarchy_results

            # Add new conditions in the decoder!
            if new_conditions is not None:
                conditions = new_conditions.to(self.device)

            c_expanded = conditions.unsqueeze(0).expand(1, -1, -1)
            
            dec_in = torch.cat([z_for_decoder, c_expanded], dim=-1)
            S, B, DC = dec_in.shape
            dec_in_flat = dec_in.reshape(S * B, DC)

            # Decode latent samples to get reconstruction parameters
            recon_params = self.model.decoder(dec_in_flat)   # dict of [S*B, D]
            
            # reshape recon params back to [S, B, D]
            for k in recon_params:
                recon_params[k] = recon_params[k].reshape(S, B, -1)

        else:
            recon_params, _ = self.model.reconstruct(input_data)

        # recon_params has shape [S, B, D]. We decode each sample s to get [num_reconstructions, B, D]
        S_dim = recon_params['mu'].shape[0]
        xs_all = []
        for s in range(S_dim):
            params_s = {k: v[s] for k, v in recon_params.items()}
            xs = self.model.decoder.sample(params_s, num_samples=num_reconstructions) # [num_recon, B, D]
            xs_all.append(xs)
            
        return torch.cat(xs_all, dim=0)  # [S*num_reconstructions, B, D]

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        meter_id: int | None = None,
        date: datetime | torch.Tensor | None = None,
        conditions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Generate samples from the model for a single day with given conditions.

        Args:
            num_samples (int): Number of samples to generate.
            meter_id (int | None): Meter ID for user-profile conditioning.
            date (datetime | torch.Tensor | None): Date for time-based conditioning.
            conditions (torch.Tensor | None): Pre-built conditioning tensor `(batch_size, num_conditions)`. 
                If provided, it will be used directly for generation. If None, conditions will be built using meter_id and date.

        Returns:
            (torch.Tensor | None, torch.Tensor | None): Tuple of (generated_samples, used_conditions).
            - generated_samples: Tensor of shape `(num_samples, 24, num_features)` in original scale.
            - used_conditions: The conditions tensor that was used for generation, of shape `(num_samples, num_conditions)`.
        """
        self.model.eval()
        self.model.to(self.device)

        if self._is_cvae and conditions is None:
            random = False
            if date is None:
                random = True
            elif isinstance(date, (int, float)):
                date = datetime.fromtimestamp(date)
            conditions = self._build_conditions(meter_id=meter_id, timestamp=date, random_condition=random)
        
        if self._is_cvae:
            samples = self.model.sample(
                conditions=conditions,
                num_prior_samples=num_samples,
                num_x_samples=1,
                device=self.device,
            )  # Usually returns [num_samples, 1, B, D] or [num_samples, B, D]
        else:
            samples = self.model.sample(num_samples_prior=num_samples, num_samples_posterior=1, device=self.device)
        
        flat = samples.reshape(-1, samples.shape[-1])
        N = flat.shape[0]
        F = flat.shape[-1] // 24
        output = flat.reshape(N, 24, F)  # [N, 24, F]

        if self.normalizer is not None:
            output = self.normalizer.inverse_transform(output.reshape(-1, F)).reshape(N, 24, F)

        return output, conditions
    
    @torch.no_grad()
    def generate_continous(
        self,
        num_days: int,
        meter_id: int,
        start_date: datetime,
        num_candidates: int = 32,
        continuity_weight: float = 1.0,
        conditions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Samples continuous multi-day sequences by sampling multiple candidates for each day and selecting the one that best continues from the previous day, ensuring smoother transitions at day boundaries.

        Args:
            num_days (int): Number of consecutive days to generate.
            meter_id (int): Meter ID for user-profile conditioning.
            start_date (datetime): Starting date for generation.
            num_candidates (int): Number of candidate windows to sample for each day after the first, to select the best continuation.
            continuity_weight (float): Weighting factor for the continuity score when selecting the best candidate. Higher values prioritize smoother transitions more strongly.

        Returns:
            (torch.Tensor | None, torch.Tensor | None): Tuple of (generated_samples, used_conditions).
            - generated_samples: Tensor of shape `(num_days * 24, num_features)` in original scale.
            - used_conditions: The conditions tensor that was used for generation, of shape `(num_days * 24, num_conditions)`.

        """
        self.model.eval()
        self.model.to(self.device)

        all_windows = []
        prev_window = None

        for day_idx in range(num_days):
            day_start = start_date + timedelta(days=day_idx)
            if conditions is None:
                conditions = self._build_conditions(meter_id=meter_id, timestamp=day_start)
            
            if prev_window is None or num_candidates <= 1:
                # First day or no selection: just sample one window
                window = self._sample_window(conditions)
            else:
                # Sample multiple candidates and pick the best one for continuity
                window = self._select_best_continuation(
                    conditions=conditions,
                    prev_window=prev_window,
                    continuity_weight=continuity_weight,
                    num_candidates=num_candidates,
                )
            all_windows.append(window.cpu())
            prev_window = window
        
        output = torch.cat(all_windows, dim=0)  # [num_days * 24, F]
        if self.normalizer is not None:
            output = self.normalizer.inverse_transform(output)
        
        return output, conditions

    def _sample_window(self, conditions: torch.Tensor) -> torch.Tensor:
        """Sample one 24-hour window for the provided conditions.

        Args:
            conditions: Condition tensor of shape `(1, num_conditions)`.

        Returns:
            Tensor shaped `(24, num_features)`.
        """
        # TODO: make sure this can be used by VAE as well
        samples = self.model.sample(
            conditions=conditions,
            num_prior_samples=1,
            num_x_samples=1,
            device=self.device,
        )
        flat = samples.reshape(-1)
        F = flat.shape[0] // 24
        return flat.reshape(24, F)
    
    def _select_best_continuation(
            self,
            conditions: torch.Tensor,
            prev_window: torch.Tensor,
            num_candidates: int,
            continuity_weight: float
        ) -> torch.Tensor:
        """ 
        Sample 'num_candidates' windows and return the one that
        connects most smoothly to 'prev_window' at the midnight boundary. 
        
        Scores by matching:
        - prev hour 23 candidate hour 0 (level continuity)
        - prev hour 22-23 slope vs candidate hour 0-1 slope (trend continuity)

        Args:
            conditions (torch.Tensor): Conditions to use for sampling candidates, shape `(1, num_conditions)`.
            prev_window (torch.Tensor): The previous day's window, shape `(24, num_features)`.
            num_candidates (int): Number of candidate windows to sample.
            continuity_weight (float): Weighting factor for the continuity score. Higher values prioritize smoother transitions more strongly.

        Returns:
            torch.Tensor: The selected window that best continues from the previous day, shape `(24, num_features)`.
        """
        samples = self.model.sample(
            conditions = conditions,
            num_prior_samples = num_candidates,
            num_x_samples = 1,
            device = self.device
        )
        candidates = samples.reshape(num_candidates, -1)
        F = candidates.shape[-1] // 24
        candidates = candidates.reshape(num_candidates, 24, F)  # [num_candidates, 24, F]

        # === Score each Candidate ===
        # level: |prev_23 - cand_0|²
        level_gap = (prev_window[-1] - candidates[:,0]) ** 2  # [num_candidates, F]
        level_cost = level_gap.sum(dim=-1) # [num_candidates]

        # trend: |(prev_23 - prev_22) - (cand_1 - cand_0)|²
        prev_slope = prev_window[-1] - prev_window[-2]
        cand_slope = candidates[:,1] - candidates[:,0]  # [num_candidates, F]
        slope_gap = (prev_slope - cand_slope) ** 2
        slope_cost = slope_gap.sum(dim=-1)  # [num_candidates]

        # combined cost
        total_cost = continuity_weight * level_cost + 0.5 * continuity_weight * slope_cost
        best_idx = torch.argmin(total_cost)
        return candidates[best_idx]
    
    def _build_conditions(self, meter_id: int | None, timestamp: datetime | None, random_condition: bool = False) -> torch.Tensor:
        """Build condition tensor for a single generation request.

        If `random_condition` is True, meter and timestamp are ignored and random
        condition vectors are produced from the conditioner.

        Args:
            meter_id: Meter ID for user-profile conditioning.
            timestamp: Timestamp for time-based conditioning.
            random_condition: Whether to sample random conditions instead.

        Returns:
            Condition tensor of shape `(1, num_conditions)`.
        """
        if random_condition:
            return self.conditioner.get_random_conditions(batch_size=1).to(self.device)
        meter_ids = torch.tensor([meter_id], device=self.device)
        unix_ts = int(timestamp.timestamp())
        timestamps = torch.tensor([unix_ts], device=self.device)
        conditions = self.conditioner.get_conditions(
            meter_id=meter_ids,
            timestamps=timestamps,
        )  # [B, C]
        return conditions.to(self.device)

    @staticmethod
    def _estimate_candidates_needed(test_loader: torch.utils.data.DataLoader,
                                    normalizer: Any | None = None,
                                    target_gap_kwh: float = 0.5) -> int:
        """
        Estimate how many candidate samples are needed to find a smooth
        midnight transition, based on the empirical variance at hour 23 and 0.
        
        Args:
            test_loader: (torch.utils.data.DataLoader): PyTorch DataLoader yielding 
                tuples of `(data, meter_id, timestamps)`. `data` must have shape 
                `(batch_size, seq_len, num_features)`.
            normalizer: Optional normalizer to inverse-transform the data.
            target_gap_kwh (float): The typical gap in kWh we want to be able to bridge.    
        Returns:
            int: number of candidates for a smooth transition
        """
        hour_0_vals = []
        hour_23_vals = []

        for batch in test_loader:
            windows = batch[0]
            if normalizer is not None:
                windows = normalizer.inverse_transform(windows)
            hour_0_vals.append(windows[:,0])  # [B, F]
            hour_23_vals.append(windows[:,-1])  # [B, F]

        hour_0_vals = torch.cat(hour_0_vals, dim=0)  # [N, F]
        hour_23_vals = torch.cat(hour_23_vals, dim=0)  #

        std_0 = hour_0_vals.std(dim=0).mean()
        std_23 = hour_23_vals.std(dim=0).mean()
        avg_std = (std_0 + std_23) / 2

        n_candidates = int(np.ceil((avg_std / target_gap_kwh) ** 2))
        n_candidates = max(8, min(n_candidates,256))  # reasonable bounds

        print(f"Estimated std at hour 0: {std_0:.2f} kWh, hour 23: {std_23:.2f} kWh"
              f" => avg std: {avg_std:.2f} kWh. To bridge a gap of {target_gap_kwh} kWh,"
              f" we estimate needing ~{n_candidates} candidates.")
        return n_candidates


if __name__ == "__main__":
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    sys.path.append(parent_dir)
    sys.path.append(root_dir)

    # Test the candidate generator
    from src.models.cvae import CVAE
    from src.data.conditioner import Conditioner
    from src.data.normalizer import Normalizer
    
    model, normalizer = CVAE.load_from_checkpoint("models/CVAE_DictionaryCoder_20260304_130859/CVAE_DictionaryCoder_20260304_130859_final.pt", device="cpu")

    if hasattr(model, 'conditioner'):
        for tag, item in model.conditioner.condition_items.items():
            if hasattr(item, 'path') and hasattr(item, 'load'):
                print(f"Loading fitted state for {tag} from {item.path}")
                item.load(item.path)

    generator = SyntheticGenerator(model=model, conditioner=model.conditioner, normalizer=normalizer, device="cpu")

    # plot the generated sequence to visually check continuity at day boundaries
    from matplotlib import pyplot as plt
    generated = generator.generate(
        num_days=7,
        meter_id=12345,
        start_date=datetime(2024, 1, 1),
        num_candidates=32,
        continuity_weight=1.0
    ).numpy()  # [num_days*24, F]

    plt.figure(figsize=(12,6))
    plt.plot(generated)
    plt.title("Generated Sequence (Check for Continuity at Day Boundaries)")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.grid()
    plt.savefig("generated_sequence.png")
