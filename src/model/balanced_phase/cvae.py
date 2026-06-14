from model.balanced_phase.vae import *

# ==============
# Define the Conditional VAE (CVAE) class
# ==============
class CVAE(VAE):
    def __init__(
        self,
        input_dim: int,
        *,
        conditioner: Conditioner,
        latent_dim: int = 32,
        distribution_dict: dict = None,
        prior_type: str = None,
        hierarchy_config: HierarchicalConfig = None,
        phased_mode: bool = False,
        sequence_length: int = 24,
        extra_condition_dim: int = 0,
        include_faraday_like: bool = False,
        encoder_average_max_std: float = 1.0,
        decoder_average_max_std: float = 1.0,
        decoder_mu_activation: str = "tanh",
        **kwargs,
    ):
        if distribution_dict is None:
            distribution_dict = {}
        distribution_dict = dict(distribution_dict)
        distribution_dict["latent_dim"] = latent_dim

        self.mu_activation = decoder_mu_activation if decoder_mu_activation is not None else "tanh"

        self.conditioner = conditioner
        self.condition_dim = conditioner.get_condition_dim() + extra_condition_dim
        self.latent_dim = int(latent_dim)

        self.include_faraday_like = include_faraday_like

        super().__init__(input_dim=input_dim, 
                         latent_dim=latent_dim, 
                         distribution_dict=distribution_dict, 
                         prior_type=prior_type, 
                         hierarchy_config=hierarchy_config, 
                         phased_mode=phased_mode,
                         sequence_length=sequence_length,
                        **kwargs)

        base_kwargs = dict(kwargs)
        for key in ("encoder_hidden_dims", "decoder_hidden_dims", "encoder_dropout", "decoder_dropout"):
            base_kwargs.pop(key, None)

        encoder_hidden_dims = kwargs.get("encoder_hidden_dims", None)
        decoder_hidden_dims = kwargs.get("decoder_hidden_dims", None)
        encoder_dropout = kwargs.get("encoder_dropout", None)
        decoder_dropout = kwargs.get("decoder_dropout", None)

        encoder_model_kwargs = {}
        if encoder_hidden_dims is not None:
            encoder_model_kwargs["hidden_dims"] = list(encoder_hidden_dims)
        if isinstance(encoder_dropout, bool):
            encoder_model_kwargs["dropout"] = encoder_dropout
        elif encoder_dropout is not None:
            try:
                encoder_dropout_rate = float(encoder_dropout)
                encoder_model_kwargs["dropout"] = encoder_dropout_rate > 0.0
                encoder_model_kwargs["dropout_rate"] = encoder_dropout_rate
            except (TypeError, ValueError):
                encoder_model_kwargs["dropout"] = bool(encoder_dropout)

        decoder_model_kwargs = {}
        if decoder_hidden_dims is not None:
            decoder_model_kwargs["hidden_dims"] = list(decoder_hidden_dims)
        if isinstance(decoder_dropout, bool):
            decoder_model_kwargs["dropout"] = decoder_dropout
        elif decoder_dropout is not None:
            try:
                decoder_dropout_rate = float(decoder_dropout)
                decoder_model_kwargs["dropout"] = decoder_dropout_rate > 0.0
                decoder_model_kwargs["dropout_rate"] = decoder_dropout_rate
            except (TypeError, ValueError):
                decoder_model_kwargs["dropout"] = bool(decoder_dropout)

        # Now override encoder and decoder with conditioned versions
        # Encoder takes [x, c] as input
        self.encoder = get_encoder_model(
            input_dim=input_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("encoder_type", "dictionary"),
            cond_dim=0,  # CNN encoders: full-input injection; cat([x_flat, c]) enters CNN trunk
            average_max_std=encoder_average_max_std,
            mu_activation="none",
            **encoder_model_kwargs,
            **base_kwargs,
        )

        # Decoder takes [z, c] as input
        self.decoder = get_decoder_model(
            latent_dim=self.decoder_input_dim + self.condition_dim,
            output_dim=input_dim,
            type=kwargs.get("decoder_type", "dictionary"),
            average_max_std=decoder_average_max_std,
            mu_activation=self.mu_activation,
            **decoder_model_kwargs,
            **base_kwargs,
        )

        # If VampPrior, re-point it to the new (conditioned) encoder and resize pseudo-inputs
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=input_dim + self.condition_dim)

        # Recalculate num_parameters after overriding encoder/decoder
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_samples: int = 1):
        """
        Forward pass with pre-computed conditions.
        
        Args:
            x: Input tensor [B, D] (already flattened)
            c: Condition tensor [B, C] (already computed)
            num_samples: Number of latent samples
        
        Returns:
            recon_params: dict with [S, B, D] tensors
            q_params: dict with [B, latent_dim] tensors
        """
        # Encode: [B, D+C] -> posterior params
        enc_in = torch.cat([x, c], dim=-1)
        q_params = self.encoder(enc_in) # dict of [B, latent_dim] params

        # Sample latent variables using reparameterization trick - sample z: [S, B, latent_dim]
        z = self.encoder.reparameterize(q_params, num_samples=num_samples)
 
        hierarchy_results = None
        if self.hierarchy is not None:
            S, B, D = z.shape
            z_flat = z.reshape(S * B, D)
            hierarchy_results = self.hierarchy.forward(z_flat)
            z_for_decoder = hierarchy_results["z"].reshape(S, B, -1)
        else:
            z_for_decoder = z
        self._last_hierarchy_results = hierarchy_results
        # expand c to match z: [S, B, C]
        c_expanded = c.unsqueeze(0).expand(num_samples, -1, -1)

        # Decode: [S, B, latent_dim + C] -> recon params
        dec_in = torch.cat([z_for_decoder, c_expanded], dim=-1)
        S, B, DC = dec_in.shape
        dec_in_flat = dec_in.reshape(S * B, DC)

        # Decode latent samples to get reconstruction parameters
        recon_params = self.decoder(dec_in_flat)   # dict of [S*B, D]
        
        # reshape recon params back to [S, B, D]
        for k in recon_params:
            recon_params[k] = recon_params[k].reshape(S, B, -1)

        return recon_params, q_params 

    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0, **_):
        """Compute ELBO loss."""
        # recon_params: dict of [S, B, D] -> use first sample for training
        if recon_params["mu"].dim() == 3:
            recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
        else:
            recon_params_squeezed = recon_params
        
        loss = super().loss(x, recon_params_squeezed, posterior_params, beta=beta, weighted_loss=weighted_loss)

        if self.include_faraday_like:
            from src.model.faraday import calculate_training_loss
            faraday_loss = calculate_training_loss(x_hat=recon_params_squeezed["mu"], x=x, z=posterior_params["mu"], 
                                                   mse_weight=1.0, upper_quantile_weight=1.0, lower_quantile_weight=1.0,
                                                   median_quantile_weight=1.0, mmd_weight=20.0)["loss"]
            loss["loss"] = loss["loss"] + faraday_loss
        
        return loss

    def train_core(self, batch, optim, beta=1.0,weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        self.train()
        x, meter_id, timestamps, agg = self._unpack_batch(batch)
                
        c = self._build_conditions(x_phase_first=self._to_phase_first(x), meter_id=meter_id, timestamps=timestamps)
        # Prep inputs: [B, L, F] -> [B, D] if needed
        x = self._prep_inputs(x)

        # Forward pass
        recon_params, q_params = self.forward(x, c, num_samples=1)

        # Backward pass
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        if gradient_clip is not None:
            gradient_clip_kwargs = gradient_clip_kwargs or {}
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        self.eval()
        metrics = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        for batch in valloader:            
            x, meter_id, timestamps, _agg = self._unpack_batch(batch)
                
            c = self._build_conditions(x_phase_first=self._to_phase_first(x), meter_id=meter_id, timestamps=timestamps)
            c = c.to(device)

            # Keep original 3D shape for conditioning
            x_3d = x.to(device)
            
            # Flatten for model
            x_flat = self._prep_inputs(x_3d)  # [B, D]

            # Forward pass
            recon_params, q_params = self.forward(x_flat, c, num_samples=1)
            loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta)

            for k in metrics:
                metrics[k] += float(loss_dict[k])
            num_batches += 1

        for k in metrics:
            metrics[k] /= max(num_batches, 1)
        return metrics

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu", **kwargs):
        self.eval()

        original_device = next(self.parameters()).device
        self.to(device)

        c = self.move_to_device(conditions, device=device)
        B = c.size(0)

        # Sample z from prior: handle different prior return shapes
        z = self.prior.sample(batch_size=B, num_samples=num_prior_samples)
        z = z.to(device)  # prior.device may differ from the requested device
        
        # Dictionary priors sample in dictionary space (dim=dictionary_size).
        # Map to latent space via the encoder's SigmaMapper so the decoder
        # receives the correct dimensionality (latent_dim).
        if hasattr(self.prior, 'dictionary_size') and z.shape[-1] != self.latent_dim:
            U = self.encoder.get_SigmaMapper()          # [latent_dim, dictionary_size]
            if U is not None:
                z = torch.einsum('sbd,do->sbo', z, U.T.to(z.device)) # [S, B, latent_dim]
                # Add prior mean (zeros for standard, learned for learnable)
                prior_params = self.prior.get_params()
                z = z + prior_params['mu'].to(z.device)
        
        # Ensure z has the correct shape [S, B, latent_dim]
        if z.dim() == 3:
            if z.shape[0] == B and z.shape[1] == num_prior_samples:
                # Shape is [B, S, latent_dim] - transpose to [S, B, latent_dim]
                z = z.transpose(0, 1)
            elif z.shape[0] == num_prior_samples and z.shape[1] == B:
                # Shape is already [S, B, latent_dim]
                pass
            else:
                # Unexpected shape - try to reshape
                print(f"Warning: Unexpected prior sample shape {z.shape}, attempting to reshape...")
                z = z.view(num_prior_samples, B, -1)
        elif z.dim() == 2:
            # Shape is [B, latent_dim] - add sample dimension
            z = z.unsqueeze(0).expand(num_prior_samples, B, -1)
        else:
            raise ValueError(f"Unexpected prior sample tensor dimensions: {z.shape}")

        if self.hierarchy is not None:
            S, B, D = z.shape
            z_flat = z.reshape(S * B, D)
            hierarchy_results = self.hierarchy.forward(z_flat)
            z = hierarchy_results["z"].reshape(S, B, -1)

        # Expand c to [S, B, C]
        c_expanded = c.unsqueeze(0).expand(num_prior_samples, B, -1)
        
        # Verify shapes before concatenation
        if z.shape[:2] != c_expanded.shape[:2]:
            raise ValueError(f"Shape mismatch: z.shape={z.shape}, c_expanded.shape={c_expanded.shape}")

        # decode 
        dec_in = torch.cat([z, c_expanded], dim=-1)  # [S, B, latent_dim + C]
        S, B, DC = dec_in.shape
        dec_in_flat = dec_in.reshape(S * B, DC)

        recon_params = self.decoder(dec_in_flat)  # dict [S*B, D]
        for k in recon_params:
            recon_params[k] = recon_params[k].reshape(S, B, -1)

        # sample x from decoder for each prior sample
        xs_all = []
        for s in range(S):
            params_s = {k: v[s] for k, v in recon_params.items()}         # [B, D]
            xs = self.decoder.sample(params_s, num_samples=num_x_samples, **kwargs) # [num_x_samples, B, D]
            xs_all.append(xs)

        return_val = torch.stack(xs_all, dim=0)  # [S, num_x_samples, B, D]
        self.to(original_device)
        return return_val
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, meter_ids: torch.Tensor, timestamps: torch.Tensor, num_samples: int = 1):
        self.eval()
        x = self._prep_inputs(x)
        c = self.conditioner.get_conditions(x, meter_id=meter_ids, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)
        return self.forward(x, c, num_samples=num_samples)
    
