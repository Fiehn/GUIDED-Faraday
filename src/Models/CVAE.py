from src.Models.VAE import *

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
        **kwargs,
    ):
        if distribution_dict is None:
            distribution_dict = {}
        distribution_dict = dict(distribution_dict)
        distribution_dict["latent_dim"] = latent_dim

        self.conditioner = conditioner
        self.condition_dim = conditioner.get_condition_dim()
        self.latent_dim = int(latent_dim)

        super().__init__(input_dim=input_dim, distribution_dict=distribution_dict, prior_type=prior_type, hierarchy_config=hierarchy_config, **kwargs)

        # Now override encoder and decoder with conditioned versions
        # Encoder takes [x, c] as input
        self.encoder = get_encoder_model(
            input_dim=input_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("encoder_type", "dictionary"),
            **kwargs,
        )

        # Decoder takes [z, c] as input
        self.decoder = get_decoder_model(
            latent_dim=self.decoder_input_dim + self.condition_dim,
            output_dim=input_dim,
            type=kwargs.get("decoder_type", "dictionary"),
            **kwargs,
        )

        # If VampPrior, re-point it to the new (conditioned) encoder and resize pseudo-inputs
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=input_dim + self.condition_dim)

        # Recalculate num_parameters after overriding encoder/decoder
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()

    def _prep_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """ Flatten [B, L, F] to [B, D] if needed """
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1)
        return x

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
            hierarchy_results = self.hierarchy(z_flat)
            z_for_decoder = hierarchy_results["output"].reshape(S, B, -1)
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

    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0,**_):
        """Compute ELBO loss."""
        # recon_params: dict of [S, B, D] -> use first sample for training
        if recon_params["mu"].dim() == 3:
            recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
        else:
            recon_params_squeezed = recon_params

        return super().loss(x, recon_params_squeezed, posterior_params, beta=beta, weighted_loss=weighted_loss)

    def train_core(self, batch, optim, beta=1.0,weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        self.train()
        if len(batch) == 4:
            x, meter_id, timestamps = batch
        elif len(batch) == 3:
            x, meter_id, timestamps = batch
        elif len(batch) == 2:
             x, meter_id = batch
        else:
            x = batch

        # Prep inputs: [B, L, F] -> [B, D] if needed
        x = self._prep_inputs(x)

        # Compute conditions
        c = self.conditioner.get_conditions(x, meter_id, timestamps)
        c = self.move_to_device(c, device=x.device)

        # Forward pass
        recon_params, q_params = self.forward(x, c, num_samples=1)

        # Backward pass
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        for x, meter_id, timestamps in valloader:
            # Keep original 3D shape for conditioning
            x_3d = x.to(device)
            
            # Flatten for model
            x_flat = self._prep_inputs(x_3d)  # [B, D]

            # Get conditions - PASS 3D tensor
            c = self.conditioner.get_conditions(x_3d, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device)

            # Forward pass
            recon_params, q_params = self.forward(x_flat, c, num_samples=1)
            loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta)

            for k in agg:
                agg[k] += float(loss_dict[k])
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu",):
        self.eval()

        c = self.move_to_device(conditions, device=device)
        B = c.size(0)

        # Sample z from prior: handle different prior return shapes
        z = self.prior.sample(batch_size=B, num_samples=num_prior_samples)
        
        # Dictionary priors sample in dictionary space (dim=dictionary_size).
        # Map to latent space via the encoder's SigmaMapper so the decoder
        # receives the correct dimensionality (latent_dim).
        if hasattr(self.prior, 'dictionary_size') and z.shape[-1] != self.latent_dim:
            U = self.encoder.get_SigmaMapper()          # [latent_dim, dictionary_size]
            if U is not None:
                z = torch.einsum('sbd,do->sbo', z, U.T) # [S, B, latent_dim]
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
            hierarchy_results = self.hierarchy(z_flat)
            z = hierarchy_results["output"].reshape(S, B, -1)

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
            xs = self.decoder.sample(params_s, num_samples=num_x_samples) # [num_x_samples, B, D]
            xs_all.append(xs)

        return torch.stack(xs_all, dim=0)  # [S, num_x_samples, B, D]
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, meter_ids: torch.Tensor, timestamps: torch.Tensor, num_samples: int = 1):
        self.eval()
        x = self._prep_inputs(x)
        c = self.conditioner.get_conditions(x, meter_id=meter_ids, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)
        return self.forward(x, c, num_samples=num_samples)
    
