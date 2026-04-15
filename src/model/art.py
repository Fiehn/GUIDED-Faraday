from src.model.cvae import *
import torch
from src.data.conditioner import Conditioner
from src.data.normalizer import Normalizer
from src.model.cvae import get_encoder_model, VampPrior
from src.model.vae import HierarchicalConfig

# A simple utility to convert floats to discrete tokens and back
class Tokenizer:
    def __init__(self, max_val=10.0, num_bins=256): # TODO: CHANGE
        self.num_bins = num_bins
        self.max_val = max_val
        # Create 256 boundaries
        self.bins = torch.linspace(0, max_val, num_bins)
        
    def encode(self, x_float):
        # Maps floats to integer tokens (0 to 255)
        # Anything above max_val gets clamped to the top bin
        idx = torch.bucketize(x_float.clamp(max=self.max_val), self.bins.to(x_float.device))
        return idx.clamp(max=self.num_bins - 1)
        
    def decode(self, x_int):
        # Maps integer tokens back to continuous floats
        return self.bins.to(x_int.device)[x_int]

class AutoRegressiveTransformerVAE(CVAE):
    def __init__(self,
                 input_dim: int,
                 conditioner: Conditioner,
                 latent_dim: int,
                 distribution_dict: dict = None,
                 prior_type: str = None,
                 hierarchy_config: HierarchicalConfig = None,
                 sequence_length: int = 24,
                 num_features: int = 3,
                 num_layers: int = 4,
                 nhead: int = 4,
                 d_model: int = 256,
                 dim_feedforward: int = 128,
                 bin_size: int = 256,
                 normalizer: Normalizer = None,
                    **kwargs):
        # Support config aliases used elsewhere in the codebase.
        num_layers = int(kwargs.pop("transformer_num_layers", num_layers))
        nhead = int(kwargs.pop("transformer_num_heads", nhead))
        bin_size = int(kwargs.pop("num_bins", bin_size))
        d_model = int(kwargs.pop("transformer_d_model", d_model))
        if nhead < 1:
            raise ValueError(f"transformer_num_heads must be >= 1, got {nhead}")

        super().__init__(
            input_dim=input_dim,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            **kwargs,
        )
        self.sequence_length = int(sequence_length)
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.tokenizer = Tokenizer(max_val=10.0, num_bins=bin_size)
        self.normalizer = normalizer

        # Encoder is handled by CVAE, but we need to override the decoder with an autoregressive transformer (or at least a transformer-based decoder).
        self.decoder = get_decoder_model(
            type="ART",
            latent_dim=self.decoder_input_dim,
            output_dim=self.sequence_length * self.num_features,
            num_bins=bin_size,
            seq_len=self.sequence_length * self.num_features,
            condition_dim=self.condition_dim,
            d_model=d_model,
            model_layers=num_layers,
            num_heads=nhead,
            hidden_dims=kwargs.get("decoder_hidden_dims", None)
        )

        # Keep decoder/prior from CVAE 
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=self.input_dim + self.condition_dim)

        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, c, num_samples=1, target_tokens=None):
        if target_tokens is None:
            raise ValueError("target_tokens must be provided for ART forward pass")

        # Get encoder output (latent distribution parameters)
        enc_in = torch.cat([x, c], dim=-1)
        q_params = self.encoder(enc_in) # posterior parameters (e.g. mean and logvar for Gaussian) based on pooled set representation + conditions
        z = self.encoder.reparameterize(q_params, num_samples=num_samples) # reparameterization trick

        # Hierarchy on latent space 
        hierarchy_results = None
        if self.hierarchy is not None:
            S, B, D = z.shape
            z_flat = z.reshape(S * B, D)
            hierarchy_results = self.hierarchy(z_flat)
            z_for_decoder = hierarchy_results["z"].reshape(S, B, -1)
        else:
            z_for_decoder = z
        self._last_hierarchy_results = hierarchy_results

        S, B, _ = z_for_decoder.shape
        c_expanded = c.unsqueeze(0).expand(S, B, -1)

        z_flat = z_for_decoder.reshape(S * B, -1)
        c_flat = c_expanded.reshape(S * B, -1)

        if target_tokens.dim() == 2:
            target_tokens = target_tokens.unsqueeze(0).expand(S, B, -1)
        target_tokens_flat = target_tokens.reshape(S * B, -1)

        # decoder
        logits = self.decoder(z_flat, c_flat, target_tokens=target_tokens_flat)["logits"]
        logits = logits.reshape(S, B, logits.shape[-2], logits.shape[-1])
        if S == 1:
            logits = logits.squeeze(0)

        return {"logits": logits}, q_params
    
    def train_core(self, batch,
                   optim, beta = 1.0,
                   weighted_loss = 0.0,
                   gradient_clip = None,
                   gradient_clip_kwargs = None,
                   condition_dropout = False, # new argument to enable condition dropout during training
                   **_):
        self.train()

        x, meter_id, timestamps = self._unpack_batch(batch)
        x_phase_first = self._to_phase_first(x).float()

        # ==========================
        # Random Phase Shuffling (Phase Invariance Augmentation)
        # During training, randomly shuffle the phase order of the input data to encourage the model to learn phase-invariant representations.
        # This is done by permuting the phase dimension of the input tensor.
        # ==========================
        if self.training:
            # Generate a random permutation of [0, 1, 2]
            perm = torch.randperm(x_phase_first.size(1), device=x_phase_first.device)
            x_phase_first = x_phase_first[:, perm, :]

        # Build conditions using the (potentially shuffled) input
        x_for_conditions = self._build_condition_input(x_phase_first)
        c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)

        if condition_dropout:
            # ==========================================
            # CONDITION DROPOUT
            # Randomly zero out conditions 30% of the time during training.
            # ==========================================
            if self.training and torch.rand(1).item() < 0.3:
                c = torch.zeros_like(c)

        # ==========================
        # Interleaved Flattening (Flatten the input while preserving phase-feature structure)
        # [B, 3, 24] -> [B, 72] but in an interleaved manner (phase1_feat1, phase1_feat2, phase1_feat3, phase2_feat1, ...)
        # This allows the autoregressive decoder to learn temporal dependencies across phases and features more effectively.
        # ==========================
        x_time_first = x_phase_first.transpose(1, 2)  # [B, L, F]
        x_interleaved = x_time_first.reshape(x_time_first.shape[0], -1)

        # 1. convert flat targets to integer tokens
        # Tokenization should happen to the unnormalized data.
        # This means we will unnormalize x_interleaved back to the original scale before tokenization, so that the tokens correspond to meaningful values.
        if self.normalizer is not None:
            x_unnormalized_time_first = self.normalizer.inverse_transform(x_time_first)
            x_unnormalized = x_unnormalized_time_first.reshape(x_interleaved.shape[0], -1).to(x_interleaved.device)
        else:
            x_unnormalized = x_interleaved
        x_flat_tokens = self.tokenizer.encode(x_unnormalized).long().to(x_interleaved.device) # [B, L*F] -> [B, L*F]

        # 2. Forward pass
        recon_params, q_params = self.forward(x_interleaved, c, num_samples=1, target_tokens=x_flat_tokens)

        # 3. Backward pass
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x_flat_tokens, recon_params, q_params, beta=beta, weighted_loss=weighted_loss, condition_dropout=condition_dropout)
        loss_dict["loss"].backward()

        if gradient_clip is not None:
            gradient_clip_kwargs = gradient_clip_kwargs or {}
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict


    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0, **_):
        """ Cross-entropy loss for autoregressive transformer decoder. 
            recon_params should contain 'logits' of shape [S, B, L*F, num_bins]
            x should be the original input of shape [B, L*F] (flattened) 
        """
        logits = recon_params["logits"] # [B, 72, 256]

        if logits.dim() == 4:
            logits = logits[0]
        
        # PyTorch CrossEntropy expects [Batch, Classes, Sequence]
        logits = logits.permute(0, 2, 1) # [B, 256, 72]
        
        # Standard Cross-Entropy
        ce_loss = F.cross_entropy(logits, x.long(), reduction='mean')
        
        kl_loss = self.encoder.kl_divergence(posterior_params, prior=self.prior).mean()
        
        total_loss = ce_loss + beta * kl_loss
        if self._last_hierarchy_results is not None:
            total_loss = total_loss + self._last_hierarchy_results["loss"]

        return {
            "loss": total_loss,
            "recon_loglikelihood": -ce_loss,
            "kl_divergence": kl_loss,
        }

    @torch.no_grad()
    def validate(self, valloader, num_samples=1, beta=1.0, device="cpu"):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0
        for batch in valloader:
            batch = self.move_to_device(batch, device=device)
            x, meter_id, timestamps = self._unpack_batch(batch)
            x_phase_first = self._to_phase_first(x).float()
            x_for_conditions = self._build_condition_input(x_phase_first)

            c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=x.device)

            x_time_first = x_phase_first.transpose(1, 2)  # [B, L, F]
            x_interleaved = x_time_first.reshape(x_time_first.shape[0], -1)
            if self.normalizer is not None:
                x_unnormalized_time_first = self.normalizer.inverse_transform(x_time_first)
                x_unnormalized = x_unnormalized_time_first.reshape(x_interleaved.shape[0], -1).to(x_interleaved.device)
            else:
                x_unnormalized = x_interleaved
            x_flat_tokens = self.tokenizer.encode(x_unnormalized).long().to(x_interleaved.device)

            recon_params, q_params = self.forward(
                x_interleaved,
                c,
                num_samples=num_samples,
                target_tokens=x_flat_tokens,
            )
            loss_dict = self.loss(x_flat_tokens, recon_params, q_params, beta=beta)

            for k in agg:
                agg[k] += float(loss_dict[k].detach())
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg
    
    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu", **kwargs):
        self.eval()

        original_device = next(self.parameters()).device
        self.to(device)

        c = self.move_to_device(conditions, device=device)
        B = c.size(0)

        # Sample from prior
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

        # Collapse the Sample (S) and Batch (B) dimensions for the Decoder
        S, B, D_z = z.shape
        D_c = c_expanded.shape[-1]
        z_flat = z.reshape(S * B, D_z)
        c_flat = c_expanded.reshape(S * B, D_c)

        temperature = float(kwargs.get("temperature", 0.75))

        # Generate num_x_samples decoded trajectories per prior sample.
        samples_per_prior = []
        for _ in range(max(1, int(num_x_samples))):
            sampled_tokens = self.decoder.sample(z_flat, c_flat, temperature=temperature)
            sampled_tokens = sampled_tokens.reshape(S, B, -1)
            samples_per_prior.append(self.tokenizer.decode(sampled_tokens))

        # [S, X, B, D] to match the CVAE sampling contract.
        samples = torch.stack(samples_per_prior, dim=1)
        self.to(original_device)
        return samples

    def _to_phase_first(self, x: torch.Tensor) -> torch.Tensor:
        """Return `x` as [B, F, L], supporting [B, D], [B, F, L], or [B, L, F]."""
        if x.dim() == 2:
            if x.shape[1] != self.sequence_length * self.num_features:
                raise ValueError(
                    f"Expected flattened input dim {self.sequence_length * self.num_features}, got {x.shape[1]}"
                )
            return x.reshape(x.shape[0], self.num_features, self.sequence_length)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D/3D input, got shape {tuple(x.shape)}")

        # [B, F, L]
        if x.shape[1] == self.num_features and x.shape[2] == self.sequence_length:
            return x

        # [B, L, F] -> [B, F, L]
        if x.shape[1] == self.sequence_length and x.shape[2] == self.num_features:
            return x.transpose(1, 2)

        raise ValueError(
            f"Could not infer phase/time axes for input shape {tuple(x.shape)} with "
            f"num_features={self.num_features}, sequence_length={self.sequence_length}"
        ) 
    def _build_condition_input(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Convert [B, F, L] -> [B, L, F] for conditioner compatibility."""
        return x_phase_first.transpose(1, 2)
    def _unpack_batch(self, batch):
        """ Unpack batch supporting (x, meter_id, timestamps), (x, cond), or (x, meter_id) formats. """
        if len(batch) == 4:
            x, second, third, fourth = batch
            # Regular CSV dataloader: (x, c, meter_id, timestamps)
            if torch.is_tensor(second) and second.dim() >= 2:
                return x, third, fourth
            # Eval/extended tuple: (x, meter_id, timestamps, agg)
            return x, second, third

        if len(batch) == 3:
            x, second, third = batch
            if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
                meter_id = second[:, 0, 0].long()
                timestamps = second[:, 1, 0].long()
                return x, meter_id, timestamps

            return x, second, third

        if len(batch) != 2:
            raise ValueError("ART expects batches as (x, meter_id, timestamps), (x, meter_id), or (x, cond)")

        x, second = batch
        meter_id = second
        timestamps = None

        if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
            # phased dataloader format: cond=[B, 2, L]
            meter_id = second[:, 0, 0].long()
            timestamps = second[:, 1, 0].long()
        elif torch.is_tensor(second) and second.dim() == 2 and second.shape[1] >= 2:
            # fallback packed format: [B, 2]
            meter_id = second[:, 0].long()
            timestamps = second[:, 1].long()

        return x, meter_id, timestamps

