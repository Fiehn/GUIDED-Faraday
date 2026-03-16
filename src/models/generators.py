import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class SyntheticGenerator:
    """
    Generate multiple contiguous days by sampling one 24h window per day
    with correct day-level conditions.
    """

    def __init__(self, model, conditioner=None, normalizer = None, device = 'cpu'):
        self.model = model
        self.conditioner = conditioner if conditioner is not None else getattr(model, 'conditioner', None)
        self.normalizer = normalizer
        self.device = device
        self._is_cvae = hasattr(model, 'conditioner') and model.conditioner is not None

    @torch.no_grad()
    def reconstruct(self, input_data: torch.Tensor, num_reconstructions: int = 1,
                    meter_id: Optional[int] = None, timestamp: Optional[datetime] = None,
                    new_conditions: Optional[torch.Tensor] = None,):
        """
        Reconstruct input data with optional new conditions. If new_conditions is provided,
        it will be used for the decoding and thus providing a way to manipulate the reconstruction by changing the conditions.

        Args:
            input_data: [B, 24, F] tensor of original data to reconstruct.
            new_conditions: Optional [B, C] tensor of new conditions to use for reconstruction.
                            If None, original conditions from the model will be used.
        Returns:
            Reconstructed data tensor of shape [B, 24, F] in original scale.
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
            input_data_flat = input_data.reshape(input_data.shape[0], -1)
            enc_in = torch.cat([input_data_flat, conditions], dim=-1)
            q_params = self.model.encoder(enc_in) # dict of [B, latent_dim] params

            # Sample latent variables using reparameterization trick - sample z: [S, B, latent_dim]
            z = self.model.encoder.reparameterize(q_params, num_samples=1)
    
            hierarchy_results = None
            if self.model.hierarchy is not None:
                S, B, D = z.shape
                z_flat = z.reshape(S * B, D)
                hierarchy_results = self.model.hierarchy(z_flat)
                z_for_decoder = hierarchy_results["output"].reshape(S, B, -1)
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
        meter_id: Optional[int] = None,
        date = None,
        conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples for a single day with given conditions.
        Args:
            num_samples: Number of 24h windows to generate.
            meter_id: Meter ID for user-profile conditioning.
            date: Optional datetime or unix timestamp for time-based conditioning.
            conditions: Optional pre-built conditioning tensor [B, C]. If provided,
                        'meter_id' and 'date' are ignored.
        Returns:
            [num_samples, 24, F] tensor in original scale.
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
        conditions = None
    ) -> torch.Tensor:
        """
         For each day after the first, we sample `num_candidates` windows
        from the prior and pick the one whose first hours best continue
        the previous day's last hours — a simple rejection/selection
        approach that respects the model's positional structure.

        Args:
            num_days: Number of days to generate.
            meter_id: Meter ID for user-profile conditioning.
            start_date: Midnight of the first day.
            num_candidates: How many candidate windows to sample per day
                           for continuity selection (1 = no selection).
            continuity_weight: How strongly to penalise discontinuity
                              at the midnight boundary.

        Returns:
            [num_days * 24, F] tensor in original scale.

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
        
        return output

    def _sample_window(self, conditions: torch.Tensor) -> torch.Tensor:
        """ Sample a single 24h window returns [24, F]"""
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
        conncets most smoothly to 'prev_window' at the midnight boundary. 
        
        Scores by matching:
        - prev hour 23 candidate hour 0 (level continuity)
        - prev hour 22-23 slope vs candidate hour 0-1 slope (trend continuity)
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
    
    def _build_conditions(self, meter_id: int, timestamp: datetime, random_condition: bool = False) -> torch.Tensor:
        """ Build conditioning tensor for a single day. Returns [1, C] tensor. """
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

    def _estimate_candidates_needed(test_loader, normalizer=None, target_gap_kwh=0.5):
        """
        Estimate how many candidate samples are needed to find a smooth
        midnight transition, based on the empirical variance at hour 23 and 0.
        
        Args:
            test_loader: DataLoader for the test set, providing (windows, conditions).
            normalizer: Optional normalizer to inverse-transform the data.
            target_gap_kwh: The typical gap in kWh we want to be able to bridge.    
        Returns:
            Estimated number of candidates needed to have a good chance of finding
            a window with a gap less than 'target_gap_kwh'.
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
    from src.Models.CVAE import CVAE
    from src.Data.Conditioner import Conditioner
    from src.Data.normalizer import Normalizer
    
    model, normalizer = CVAE.load_from_checkpoint("models/CVAE_DictionaryCoder_20260304_130859/CVAE_DictionaryCoder_20260304_130859_final.pt", device="cpu")

    if hasattr(model, 'conditioner'):
        for tag, item in model.conditioner.condition_items.items():
            if hasattr(item, 'path') and hasattr(item, 'load'):
                print(f"Loading fitted state for {tag} from {item.path}")
                item.load(item.path)

    generator = CandidateGenerator(model=model, conditioner=model.conditioner, normalizer=normalizer, device="cpu")

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
