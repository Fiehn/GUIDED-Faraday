import torch
import numpy as np
from datetime import datetime, timedelta

class SyntheticGenerator:
    """Called to generate synthetic samples from a trained model.
    This class provides methods to generate synthetic data using a trained generative model (like a CVAE).
    It supports:
    - Reconstructing input data with optional
        new conditions, allowing for manipulation of the reconstruction by changing the conditions.
    - Generating new samples for a single day based on specified conditions (meter ID, date, or pre-built conditions).
    - Generating continuous multi-day sequences by sampling multiple candidates for each day and selecting the one that best continues from the previous day, ensuring smoother transitions at day boundaries.

    Includes built-in safeguards for log-space generation:
    - Temperature scaling: Shrink the variance of the generated noise.
    - Rejection sampling: Automatically discard and re-roll mathematically explosive samples.

    Args:
        model: A trained generative model (e.g., CVAE) that has a .sample() and .reconstruct() method and optionally: [.conditioner, .encoder, .decoder, .hierarchy].
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

    def __init__(self, model, conditioner=None, normalizer = None, secondary_model = None, device = 'cpu'):
        self.model = model
        self.conditioner = conditioner if conditioner is not None else getattr(model, 'conditioner', None)
        self.normalizer = normalizer
        self.secondary_model = secondary_model
        self.device = device
        self._is_cvae = hasattr(model, 'conditioner') and model.conditioner is not None
        self._is_diffusion = self._is_diffusion_model()

    def _is_diffusion_model(self) -> bool:
        return hasattr(self.model, 'noise_scheduler') and hasattr(self.model, 'reconstruct')

    def _is_phase_first_model(self) -> bool:
        return (
            hasattr(self.model, '_to_phase_first')
            and getattr(self.model, 'num_features', None) is not None
            and getattr(self.model, 'sequence_length', None) is not None
        )
    def _is_baseline_model(self) -> bool:
        return hasattr(self.model, 'tag') and 'baseline' in self.model.tag

    def _is_art_model(self) -> bool:
        """Best-effort capability check for ART-style token decoders."""
        decoder = getattr(self.model, 'decoder', None)
        return (
            hasattr(self.model, 'tokenizer')
            and decoder is not None
            and hasattr(decoder, 'token_embedding')
            and hasattr(decoder, 'sample')
        )

    def _normalizer_feature_dim(self) -> int | None:
        """Infer feature dimension expected by the normalizer stats."""
        if self.normalizer is None:
            return None
        stats = getattr(self.normalizer, 'mean', None)
        if stats is None:
            return None
        if isinstance(stats, torch.Tensor):
            if stats.numel() == 0:
                return None
            return int(stats.reshape(-1).shape[0])
        if isinstance(stats, np.ndarray):
            if stats.size == 0:
                return None
            return int(stats.reshape(-1).shape[0])
        try:
            return int(np.asarray(stats).reshape(-1).shape[0])
        except Exception:
            return None

    def _apply_normalizer(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply (inverse) normalization for 2D or 3D tensors with robust axis handling."""
        if self.normalizer is None:
            return x

        fn = self.normalizer.inverse_transform if inverse else self.normalizer.transform
        feature_dim = self._normalizer_feature_dim()

        if x.dim() == 2:
            return fn(x)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor for normalization, got shape {tuple(x.shape)}")

        if feature_dim is not None and x.shape[-1] == feature_dim:
            return fn(x.reshape(-1, feature_dim)).reshape_as(x)

        if feature_dim is not None and x.shape[1] == feature_dim:
            x_last = x.movedim(1, -1)
            y_last = fn(x_last.reshape(-1, feature_dim)).reshape_as(x_last)
            return y_last.movedim(-1, 1)

        # Fallback for unknown feature dim: preserve existing behavior.
        return fn(x.reshape(-1, x.shape[-1])).reshape_as(x)

    @torch.no_grad()
    def reconstruct(self, input_data: torch.Tensor,
                    num_reconstructions: int = 1,
                    meter_id: int | None = None, 
                    timestamp: datetime | torch.Tensor | None = None,
                    new_conditions: torch.Tensor | None = None,
                    encode_conditions: torch.Tensor | None = None,
                    is_denormalized: bool = False,
                    temperature: float = 1.0,
                    use_rejection: bool = False,
                    max_safe_z: float = 5.0,
                    max_retries: int = 10) -> torch.Tensor | None:
        """
        Reconstruct input data with optional new conditions. If new_conditions is provided,
        it will be used for the decoding and thus providing a way to manipulate the reconstruction by changing the conditions.

        Args:
            input_data (torch.Tensor): Tensor of original timeseries data to reconstruct of shape `(batch_size, 24, num_features)`.
            new_conditions (Optional[torch.Tensor]): Tensor of new conditions to use for reconstruction of shape `(batch_size, num_conditions)`.
                            If None, original conditions from the model will be used.
            encode_conditions (Optional[torch.Tensor]): Optional conditions to use for encoding only. If provided and
                            new_conditions is None, these conditions are also used for decoding.
            is_denormalized (bool): Whether the input_data is already normalized. If False, it will be normalized using self.normalizer before reconstruction and inverse-transformed back to original scale at the end.
            temperature (float): Temperature scaling factor for sampling. Values < 1.0 will produce less variance (safer samples), while values > 1.0 will produce more variance.
            use_rejection (bool): Whether to use rejection sampling to discard and re-roll samples with extreme latent values that could lead to mathematically explosive reconstructions.
            max_safe_z (float): The maximum absolute value for latent variables (z) considered safe. Samples with any latent dimension exceeding this value will be discarded and re-sampled if use_rejection is True.
            max_retries (int): Maximum number of retries for rejection sampling before giving up and returning the best attempt.
        Returns:
            Reconstructed data tensor of shape `(num_reconstructions, batch_size, 24, num_features)` in original scale.
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Robustly determine the model's dtype, handling models without parameters (like baselines)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            try:
                model_dtype = next(self.model.buffers()).dtype
            except StopIteration:
                model_dtype = torch.float32
                
        input_data = input_data.to(self.device, dtype=model_dtype)

        if self.normalizer is not None and not is_denormalized:
            input_data = self._apply_normalizer(input_data, inverse=False)
            # Normalizers may return float64 tensors; align with model dtype for linear layers.
            input_data = input_data.to(self.device, dtype=model_dtype)
        if self.secondary_model is not None:
            # For phase-splitter models we run the first model on aggregated data, therefore we sum over phases to get the correct input shape for the first model.
            phased_input_data = input_data.reshape(input_data.shape[0], 24, -1, self.model.num_features)  # [B, 24, num_phases, F]
            input_data = input_data.sum(dim=1)  # Sum over phase dimension to get [B, 24, F_agg]

        # Get conditions
        conditions = None
        if self._is_cvae and self.conditioner is not None:
            cond_input = self.model._build_condition_input(self.model._to_phase_first(input_data))
            conditions = self.conditioner.get_conditions(cond_input, meter_id, timestamp).to(self.device, dtype=model_dtype)
            enc_conditions = conditions
            if encode_conditions is not None:
                enc_conditions = encode_conditions.to(self.device, dtype=model_dtype)
                if new_conditions is None:
                    conditions = enc_conditions
        elif encode_conditions is not None or new_conditions is not None:
            # If conditions are provided but conditioner is None, use them directly
            if new_conditions is not None:
                conditions = new_conditions.to(self.device, dtype=model_dtype)
            elif encode_conditions is not None:
                conditions = encode_conditions.to(self.device, dtype=model_dtype)

        is_sequential_phase_vae = self.model.__class__.__name__ == 'SequentialPhaseVAE'
        
        if self._is_baseline_model():
            flat_output = self.model.reconstruct(input_data)
        elif self._is_diffusion:
            flat_output = self.model.reconstruct(
                input_data,
                meter_id=meter_id,
                timestamps=timestamp,
                c=new_conditions,
                num_reconstructions=num_reconstructions,
                temperature=temperature,
                device=self.device,
            )
            if flat_output is None:
                return None
            if flat_output.dim() == 2:
                flat_output = flat_output.unsqueeze(0)
            if flat_output.dim() != 3:
                raise ValueError(f"Unexpected diffusion reconstruction shape: {tuple(flat_output.shape)}")
        elif is_sequential_phase_vae:
            # For SequentialPhaseVAE, use its custom reconstruct method that handles phase concatenation
            # Convert meter_id and timestamp to tensors if needed
            B = input_data.shape[0]
            if meter_id is not None and not isinstance(meter_id, torch.Tensor):
                meter_ids_tensor = torch.full((B,), meter_id, dtype=torch.long, device=self.device)
            else:
                meter_ids_tensor = meter_id if isinstance(meter_id, torch.Tensor) else torch.zeros(B, dtype=torch.long, device=self.device)
            
            if timestamp is not None and isinstance(timestamp, torch.Tensor):
                timestamps_tensor = timestamp
            elif timestamp is not None:
                # For datetime or other types, create a dummy tensor
                timestamps_tensor = torch.zeros(B, dtype=torch.long, device=self.device)
            else:
                timestamps_tensor = torch.zeros(B, dtype=torch.long, device=self.device)
            
            phased_recon = self.model.reconstruct(input_data, meter_ids_tensor, timestamps_tensor)
            # Shape: [B, 3, 24] or [B, 24, 3] depending on phase_first. Flatten to [B, 72]
            flat_recon = phased_recon.reshape(phased_recon.shape[0], -1)
            # Add reconstruction count dimension to get [1, B, 72]
            flat_output = flat_recon.unsqueeze(0)
        # Conditions for the encoding are always based on the original meter_id and timestamp, to get the correct latent representation.
        elif self._is_cvae:
            # Ensure input_data is flattened appropriately
            if hasattr(self.model, '_set_encode'):
                x_enc = self.model._set_encode(self.model._to_phase_first(input_data))
                enc_in = torch.cat([x_enc, enc_conditions], dim=-1)
            else:
                input_data_flat = input_data.reshape(input_data.shape[0], -1).to(self.device)
                enc_in = torch.cat([input_data_flat, enc_conditions], dim=-1)
                
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
                conditions = new_conditions.to(self.device, dtype=model_dtype)

            c_expanded = conditions.unsqueeze(0).expand(1, -1, -1)

            if self._is_art_model():
                # ART decoder requires (z, c) and samples tokens autoregressively.
                S, B, D_z = z_for_decoder.shape
                D_c = c_expanded.shape[-1]
                z_flat = z_for_decoder.reshape(S * B, D_z)
                c_flat = c_expanded.reshape(S * B, D_c)

                art_samples = []
                for _ in range(max(1, int(num_reconstructions))):
                    sampled_tokens = self.model.decoder.sample(z_flat, c_flat, temperature=temperature)
                    sampled_tokens = sampled_tokens.reshape(S, B, -1)
                    art_samples.append(self.model.tokenizer.decode(sampled_tokens))

                flat_output = torch.cat(art_samples, dim=0)  # [S*num_recon, B, D]
            else:
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

        # Skip recon_params processing if we already have flat_output from diffusion/baseline/phased models
        if (not self._is_diffusion) and (not self._is_baseline_model()) and (not is_sequential_phase_vae) and not (self._is_cvae and meter_id is not None and timestamp is not None and self._is_art_model()):
            # Normalize decoder param tensors to [S, B, D] to avoid accidental shape expansion.
            # Some model.reconstruct() implementations return [B, D] instead of [S, B, D].
            for k, v in recon_params.items():
                if v.dim() == 2:
                    recon_params[k] = v.unsqueeze(0)
                elif v.dim() != 3:
                    raise ValueError(f"Unexpected decoder param shape for '{k}': {tuple(v.shape)}")

            # recon_params has shape [S, B, D]. We decode each sample s to get [num_reconstructions, B, D]
            S_dim = recon_params['mu'].shape[0]
            xs_all = []
            for s in range(S_dim):
                params_s = {k: v[s] for k, v in recon_params.items()}

                # --- Rejection Sampling Loop for Reconstruction ---
                valid_samples = []
                needed = num_reconstructions

                for attempt in range(max_retries):
                    xs = self.model.decoder.sample(params_s, num_samples=needed, temperature=temperature)

                    if not use_rejection:
                        valid_samples.append(xs)
                        needed = 0
                        break

                    # Check safe maximums across the [needed, B, D] tensor
                    max_vals = xs.abs().amax(dim=-1) # [needed, B]
                    good_mask = (max_vals < max_safe_z).all(dim=-1) # Only keep if entire batch is safe

                    good_xs = xs[good_mask]
                    if len(good_xs) > 0:
                        valid_samples.append(good_xs)
                        needed -= len(good_xs)

                    if needed <= 0:
                        break

                if needed > 0:
                    print(f"Warning: Rejection sampling failed to find safe reconstructions. Injecting {needed} flatline means.")
                    fallback = self.model.decoder.sample(params_s, num_samples=needed, temperature=0.0)
                    valid_samples.append(fallback)

                xs_all.append(torch.cat(valid_samples, dim=0)[:num_reconstructions])
            flat_output = torch.cat(xs_all, dim=0)  # [S*num_recon, B, D]

        N, B, D = flat_output.shape
        flat_output = flat_output.reshape(N * B, D)

        # 1. Un-flatten properly based on Phase-First vs Time-First
        L = getattr(self.model, 'sequence_length', 24)
        F = D // L
        is_phase_first = getattr(self.model, 'num_features', None) is not None

        if is_phase_first:
            output = flat_output.reshape(-1, F, L).transpose(1, 2)  # [N*B, L, F]
        else:
            output = flat_output.reshape(-1, L, F)

        if self.secondary_model is not None:
            # If this is a phase-splitter type model, we run reconstruct on the secondary model to get the phases
            # Pass normalizer=None so it returns normalized phase estimates, which we inverse transform afterward!
            output = self.secondary_model.reconstruct(x_agg_generated=output, conditions=conditions, normalizer=None, x_phased_real = phased_input_data)
            # Secondary model returns [B, 3, 24] (i.e. [N*B, F_split, L]) where F_split is 3 (for our dataset).
            # It replaces output, and we transpose it to [N*B, L, F_split] so apply_normalizer works (expects trailing feature dim).
            output = output.transpose(1, 2)
            
        # 2. Apply inverse transform so it's in actual kWh, not Z-scores
        if self.normalizer is not None and not is_denormalized and not self._is_art_model():
            output = self._apply_normalizer(output, inverse=True)

        # Return as 
        return output.reshape(N, B, -1)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        meter_id: int | None = None,
        date: datetime | torch.Tensor | None = None,
        conditions: torch.Tensor | None = None,
        temperature: float = 1.0,
        use_rejection: bool = False,
        max_safe_z: float = 4.5,
        max_retries: int = 10
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Generate samples from the model for a single day with given conditions.

        Args:
            num_samples (int): Number of samples to generate.
            meter_id (int | None): Meter ID for user-profile conditioning.
            date (datetime | torch.Tensor | None): Date for time-based conditioning.
            conditions (torch.Tensor | None): Pre-built conditioning tensor `(batch_size, num_conditions)`. 
                If provided, it will be used directly for generation. If None, conditions will be built using meter_id and date.
            temperature (float): Temperature scaling factor for sampling. Values < 1.0 will produce less variance (safer samples), while values > 1.0 will produce more variance.
            use_rejection (bool): Whether to use rejection sampling to discard and re-roll samples with extreme latent values that could lead to mathematically explosive generations.
            max_safe_z (float): The maximum absolute value for latent variables (z) considered safe. Samples with any latent dimension exceeding this value will be discarded and re-sampled if use_rejection is True.
            max_retries (int): Maximum number of retries for rejection sampling before giving up and returning the best attempt.

        Returns:
            (torch.Tensor | None, torch.Tensor | None): Tuple of (generated_samples, used_conditions).
            - generated_samples: Tensor of shape `(num_samples, 24, num_features)` in original scale.
            - used_conditions: The conditions tensor that was used for generation, of shape `(num_samples, num_conditions)`.
        """
        self.model.eval()
        self.model.to(self.device)

        valid_samples = []
        needed = num_samples

        if self._is_baseline_model():
            samples = self.model.sample(num_samples=num_samples)
            # Baseline returns [N, 1, 1, D], already correctly laid out (F-first if phased)
            final_samples = samples.reshape(num_samples, -1)
            valid_samples.append(final_samples)
            needed = 0
        elif self._is_art_model():
            # ART tokens are already bounded in output space; latent rejection on x-scale is not meaningful.
            use_rejection = False

        if conditions is None:
            random_cond = (date is None)
            if isinstance(date, (int, float)):
                date = datetime.fromtimestamp(date)
            conditions = self._build_conditions(meter_id=meter_id, timestamp=date, random_condition=random_cond)

        if conditions is not None:
            conditions = conditions.to(self.device)
            # Ensure we have exactly one condition row per requested sample.
            if conditions.shape[0] < num_samples:
                repeats = (num_samples // conditions.shape[0]) + 1
                conditions = conditions.repeat(repeats, 1)
            conditions = conditions[:num_samples]
        
        
        # --- Batched Rejection Sampling ---
        for attempt in range(max_retries):
            if self._is_cvae:
                # One prior sample per condition row to avoid quadratic SxB expansion.
                cond_batch = conditions[:needed] if conditions is not None else None
                samples = self.model.sample(
                    conditions=cond_batch,
                    num_prior_samples=1,
                    num_x_samples=1,
                    device=self.device,
                    temperature=temperature
                ) 
                # [1, 1, B, D] -> [B, D]
                samples = samples.squeeze(0).squeeze(0)
            else:
                samples = self.model.sample(num_samples_prior=needed, num_samples_posterior=1, device=self.device, temperature=temperature)
                if isinstance(samples, dict) and "samples" in samples:
                    samples = samples["samples"]
                # Collapse any extra sample dims into batch dim, keep last dim as feature dim.
                samples = samples.reshape(-1, samples.shape[-1])
            
            if not use_rejection:
                valid_samples.append(samples)
                needed = 0
                break
                
            # samples shape: [N, D]
            flat_samples = samples.reshape(samples.shape[0], -1)
            max_vals, _ = flat_samples.abs().max(dim=-1)
            
            good_mask = max_vals < max_safe_z
            good_samples = samples[good_mask]
            
            if len(good_samples) > 0:
                valid_samples.append(good_samples)
                needed -= len(good_samples)
                
            if needed <= 0:
                break
                
        if needed > 0:
            print(f"Warning: Rejection sampling yielded {needed} flatline means to avoid explosions.")
            if self._is_cvae:
                fallback = self.model.sample(
                    conditions=conditions[:needed],
                    num_prior_samples=1,
                    num_x_samples=1,
                    device=self.device,
                    temperature=0.0,
                ).squeeze(0).squeeze(0)
            else: 
                fallback = self.model.sample(num_samples_prior=needed, num_samples_posterior=1, device=self.device, temperature=0.0)
                if isinstance(fallback, dict) and "samples" in fallback:
                    fallback = fallback["samples"]
                fallback = fallback.reshape(-1, fallback.shape[-1])
            valid_samples.append(fallback)
        
        final_samples = torch.cat(valid_samples, dim=0)[:num_samples]  # [N, D]
        N, D = final_samples.shape
        flat = final_samples
        
        L = getattr(self.model, 'sequence_length', 24)
        F = D // L

        output = flat.reshape(N, L, F)

        if self.secondary_model is not None:
            # Run without normalizer so we get normalized phase proportions
            output = self.secondary_model.sample(x_agg_generated=output, conditions=conditions, normalizer=None)
            split_cond = conditions
            # Output is [N, F_split, L]. Transpose to [N, L, F_split]
            output = output.transpose(1, 2)
            F = output.shape[-1]
            conditions = split_cond

        if self.normalizer is not None and not self._is_art_model():
            output = self.normalizer.inverse_transform(output.reshape(-1, F)).reshape(N, L, F)

        # If conditions had a batch size > 1 and S > 1, the returned shape is (S * B, L, F).
        # To remain compatible with callers expecting (num_samples, L, F), 
        # if B == 1, we just return (S, L, F). 
        # If B > 1 and S == 1, it's (B, L, F).
        # We let the callers handle the S * B size.
        return output, conditions
    
    @torch.no_grad()
    def generate_continous(
        self,
        num_days: int,
        meter_id: int,
        start_date: datetime,
        num_candidates: int = 32,
        continuity_weight: float = 1.0,
        conditions = None,
        temperature: float = 1.0,
        use_rejection: bool = False,
        max_safe_z: float = 4.5
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Samples continuous multi-day sequences by sampling multiple candidates for each day and selecting the one that best continues from the previous day, ensuring smoother transitions at day boundaries.

        Args:
            num_days (int): Number of consecutive days to generate.
            meter_id (int): Meter ID for user-profile conditioning.
            start_date (datetime): Starting date for generation.
            num_candidates (int): Number of candidate windows to sample for each day after the first, to select the best continuation.
            continuity_weight (float): Weighting factor for the continuity score when selecting the best candidate. Higher values prioritize smoother transitions more strongly.
            conditions (torch.Tensor | None): Optional pre-built conditions tensor to use for all days. If None, conditions will be built for each day using meter_id and date.
            temperature (float): Temperature scaling factor for sampling. Values < 1.0 will produce less variance (safer samples), while values > 1.0 will produce more variance.
            use_rejection (bool): Whether to use rejection sampling to discard and re-roll samples with extreme latent values that could lead to mathematically explosive generations.
            max_safe_z (float): The maximum absolute value for latent variables (z) considered safe. Samples with any latent dimension exceeding this value will be discarded and re-sampled if use_rejection is True.

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
            current_conditions = self._build_conditions(meter_id=meter_id, timestamp=day_start) if conditions is None else conditions
            
            if prev_window is None or num_candidates <= 1:
                # Use generate() to get a single safe window
                window, _ = self.generate(
                    num_samples=1, 
                    conditions=current_conditions,
                    temperature=temperature,
                    use_rejection=use_rejection,
                    max_safe_z=max_safe_z
                )
                window = window.squeeze(0) # Remove batch dim
                
                # Inverse transform is handled inside generate, so we undo it here 
                # so the continuity tracker scores normalized data.
                if self.normalizer is not None:
                    window = self.normalizer.transform(window)
            else:
                window = self._select_best_continuation(
                    conditions=current_conditions,
                    prev_window=prev_window,
                    continuity_weight=continuity_weight,
                    num_candidates=num_candidates,
                    temperature=temperature,
                    use_rejection=use_rejection,
                    max_safe_z=max_safe_z
                )
                
            all_windows.append(window.cpu())
            prev_window = window
        
        output = torch.cat(all_windows, dim=0)  # [num_days * 24, F]
        if self.normalizer is not None:
            output = self.normalizer.inverse_transform(output)
        
        return output, conditions

    def _sample_window(self, conditions: torch.Tensor) -> torch.Tensor:
        samples = self.model.sample(
            conditions=conditions,
            num_prior_samples=1,
            num_x_samples=1,
            device=self.device,
        )
        flat = samples.reshape(-1)
        L = getattr(self.model, 'sequence_length', 24)
        F = flat.shape[0] // L
        
        # Safely un-flatten Phase-First data
        is_phase_first = self._is_phase_first_model()
        
        if is_phase_first:
            return flat.reshape(F, L).T  # [F, L] transposed to [L, F]
        else:
            return flat.reshape(L, F)
    
    def _select_best_continuation(
            self,
            conditions: torch.Tensor,
            prev_window: torch.Tensor,
            num_candidates: int,
            continuity_weight: float,
            temperature: float = 1.0,
            use_rejection: bool = False,
            max_safe_z: float = 4.5
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
            temperature (float): Temperature scaling factor for sampling. Values < 1.0 will produce less variance (safer samples), while values > 1.0 will produce more variance.
            use_rejection (bool): Whether to use rejection sampling to discard and re-roll samples with extreme latent values that could lead to mathematically explosive generations.
            max_safe_z (float): The maximum absolute value for latent variables (z) considered safe. Samples with any latent dimension exceeding this value will be discarded and re-sampled if use_rejection is True.

        Returns:
            torch.Tensor: The selected window that best continues from the previous day, shape `(24, num_features)`.
        """
        samples = self.model.sample(
            conditions = conditions,
            num_prior_samples = num_candidates,
            num_x_samples = 1,
            device = self.device,
            temperature = temperature
        )
        
        candidates = samples.reshape(num_candidates, -1)
        L = getattr(self.model, 'sequence_length', 24)
        F = candidates.shape[-1] // L
        
        # Pre-filter dangerous candidates before scoring
        if use_rejection and not self._is_art_model():
            max_vals, _ = candidates.abs().max(dim=-1)
            good_mask = max_vals < max_safe_z
            if good_mask.any():
                candidates = candidates[good_mask]
            else:
                print("Warning: All continuation candidates were explosive. Generating flatline mean.")
                safe_sample = self.model.sample(conditions=conditions, num_prior_samples=1, num_x_samples=1, device=self.device, temperature=0.0)
                candidates = safe_sample.reshape(1, -1)

        num_valid = candidates.shape[0]
        
        is_phase_first = self._is_phase_first_model()
        if is_phase_first:
            candidates = candidates.reshape(num_valid, F, L).transpose(1, 2)
        else:
            candidates = candidates.reshape(num_valid, L, F) 

        if self.normalizer is not None and self._is_art_model():
            # ART samples decode directly to original scale; continuity scoring is done in normalized space.
            candidates = self._apply_normalizer(candidates, inverse=False)

        # Score in Normalized Space
        level_gap = (prev_window[-1] - candidates[:,0]) ** 2  
        level_cost = level_gap.sum(dim=-1) 

        prev_slope = prev_window[-1] - prev_window[-2]
        cand_slope = candidates[:,1] - candidates[:,0]  
        slope_gap = (prev_slope - cand_slope) ** 2
        slope_cost = slope_gap.sum(dim=-1)  

        total_cost = continuity_weight * level_cost + 0.5 * continuity_weight * slope_cost
        best_idx = torch.argmin(total_cost)
        
        return candidates[best_idx]
    
    def _build_conditions(self, meter_id: int, timestamp: datetime, random_condition: bool = False) -> torch.Tensor:
        """Build conditions tensor for generation based on meter_id and timestamp. If random_condition is True, it will ignore meter_id and timestamp and return random conditions.
        Args:
            meter_id (int): Meter ID for user-profile conditioning.
            timestamp (datetime): Timestamp for time-based conditioning.
            random_condition (bool): If True, ignore meter_id and timestamp and return random conditions.
        Returns:
            torch.Tensor: Conditions tensor of shape `(1, num_conditions)`.
        """
        if self.conditioner is None:
            return None
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

    def _estimate_candidates_needed(test_loader, normalizer = None, target_gap_kwh: float = 0.5):
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

    