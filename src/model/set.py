from src.model.cvae import *
import torch
import itertools
from src.data.conditioner import Conditioner
from src.model.cvae import get_encoder_model, VampPrior
from src.model.vae import HierarchicalConfig


# ==============
# SetGuideVAE: Set-encoder front-end on top of the GUIDE-VAE backbone
# ==============

# The model combines:
#  - the GUIDE-VAE backbone (PDCC dictionary encoder/decoder, user-profile + calendar conditioning)
#  - a permutation-invariant Set encoder (ISAB + ABL) that processes all three phases jointly
#    before feeding a pooled set representation into the posterior encoder


class SetGuideVAE(CVAE):
    """ 
    SetGuideVAE for phase-aware smart-meter data.

    GUIDE-VAE backbone (inherited via OnlineCVAE → CVAE → VAE):
    - conditioning (`Conditioner`) for user profiles / calendar / topology signals
    - PDCC dictionary encoder/decoder (PDCC-style covariance) through `encoder_type='dictionary'`

    Set-encoder front-end added here:
    - permutation-invariant set encoder (ISAB + ABL) before posterior parameterization,
      allowing the model to jointly reason over all three electrical phases.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        conditioner: Conditioner,
        latent_dim: int = 32,
        distribution_dict: dict = None,
        prior_type: str = None,
        hierarchy_config: HierarchicalConfig = None,
        sequence_length: int = 24,
        num_features: int = None,
        set_hidden_dim: int = 128,
        set_num_heads: int = 4,
        set_num_inds: int = 16,
        set_num_blocks: int = 2,
        is_isab: bool = True,
        use_abl_pool: bool = False,
        phased_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            phased_mode=phased_mode,
            sequence_length=sequence_length,
            **kwargs,
        )

        # Expected 3-phase setup defaults to [B, 3, 24] (phase-first) or [B, 24, 3].
        self.sequence_length = int(sequence_length) # 24 hours
        self.num_features = int(num_features) if num_features is not None else max(1, input_dim // self.sequence_length) # 3 phases
        self.element_dim = self.sequence_length # each phase is a sequence of 24 hours
        self.num_set_elements = self.num_features # 3 elements in our set (the 3 phases)

        # Encoder architecture
        self.set_hidden_dim = int(set_hidden_dim)
        # self.set_input_proj = nn.Linear(self.element_dim, self.set_hidden_dim) # projecting from 24 -> 128 dim for each set element
        
        # Using a small CNN to project each phase's 24-hour sequence into a higher-dimensional feature space before attention blocks. 
        # This allows the model to learn temporal patterns within each phase before attending across phases. The kernel size and number of layers can be tuned.
        self.set_input_proj = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # Now we flatten a temporally-aware feature map
        )
        self.set_element_proj = nn.Linear(32 * self.sequence_length, self.set_hidden_dim)
        self.use_abl_pool = bool(use_abl_pool)

        if is_isab:
            self.set_encoder_blocks = nn.ModuleList([
                ISAB(
                    dim_in=self.set_hidden_dim,
                    dim_out=self.set_hidden_dim,
                    num_heads=set_num_heads,
                    num_inds=set_num_inds,
                )
                for _ in range(max(1, int(set_num_blocks)))
            ])
        else:
            self.set_encoder_blocks = nn.ModuleList([
            SAB(
                dim_in=self.set_hidden_dim,
                dim_out=self.set_hidden_dim,
                num_heads=set_num_heads
            )
            for _ in range(max(1, int(set_num_blocks)))
        ])
        if self.use_abl_pool:
            self.set_pool = ABL(dim=self.set_hidden_dim, num_heads=set_num_heads, num_seed_vectors=1)
            self.set_repr_dim = self.set_hidden_dim
        else:
            self.set_pool = None
            self.set_repr_dim = self.set_hidden_dim * self.num_set_elements

        encoder_hidden_dims = kwargs.get("encoder_hidden_dims", None)
        encoder_dropout = kwargs.get("encoder_dropout", None)

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

        encoder_base_kwargs = dict(kwargs)
        for key in ("encoder_hidden_dims", "decoder_hidden_dims", "encoder_dropout", "decoder_dropout"):
            encoder_base_kwargs.pop(key, None)

        # Replace posterior encoder: it now consumes pooled set representation + conditions.
        self.encoder = get_encoder_model(
            input_dim=self.set_repr_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("encoder_type", "dictionary"),
            **encoder_model_kwargs,
            **encoder_base_kwargs,
        )

        # Keep decoder/prior from CVAE 
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=self.set_repr_dim + self.condition_dim)

        # Count trainable parameters for reporting
        self.num_parameters = (
            sum(p.numel() for p in self.set_input_proj.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.set_element_proj.parameters() if p.requires_grad)
            + sum(p.numel() for blk in self.set_encoder_blocks for p in blk.parameters() if p.requires_grad)
            + (sum(p.numel() for p in self.set_pool.parameters() if p.requires_grad) if self.set_pool is not None else 0)
            + self.encoder._num_parameters()
            + self.decoder._num_parameters()
        )

    # make sure input is handled correctly in all methods, supporting both phased and flattened formats
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

    # permuation-invariatn encoder 
    def _set_encode(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Encode set elements to a pooled representation [B, H]."""
        B, F, L = x_phase_first.shape
        if F != self.num_set_elements or L != self.sequence_length:
            raise ValueError(
                f"Expected set input [B, {self.num_set_elements}, {self.sequence_length}], got {tuple(x_phase_first.shape)}"
            )

        # Project each phase independently with the temporal CNN.
        x_elements = x_phase_first.reshape(B * F, 1, L)
        h = self.set_input_proj(x_elements)
        if h.dim() > 2:
            h = h.flatten(start_dim=1)
        h = self.set_element_proj(h)
        h = h.reshape(B, F, self.set_hidden_dim)

        for blk in self.set_encoder_blocks:
            h = blk(h) # passing through 2 ISAB blocks with attention between phases

        if self.set_pool is not None:
            pooled = self.set_pool(h).squeeze(1)  # [B, H]
        else:
            pooled = h.flatten(start_dim=1)  # [B, F*H] when ABL pooling is disabled
        return pooled

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_samples: int = 1):
        # Set encoder path
        x_phase_first = self._to_phase_first(x).float()
        set_repr = self._set_encode(x_phase_first)

        # Posterior q(z | X_set, c)
        enc_in = torch.cat([set_repr, c], dim=-1)
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

        # Decode p(x | z, c)
        c_expanded = c.unsqueeze(0).expand(num_samples, -1, -1)
        dec_in = torch.cat([z_for_decoder, c_expanded], dim=-1)
        S, B, DC = dec_in.shape
        dec_in_flat = dec_in.reshape(S * B, DC)
        recon_params = self.decoder(dec_in_flat)
        for k in recon_params:
            recon_params[k] = recon_params[k].reshape(S, B, -1)
        return recon_params, q_params

    def train_core(self, batch, optim, beta=1.0, weighted_loss=0.0, gradient_clip=None, gradient_clip_max_norm=None, condition_dropout=False, **_): # TODO: add condition_dropout argument
        self.train()

        x, meter_id, timestamps, agg = self._unpack_batch(batch)

        # prepare inputs: ensure phase-first format for encoder and conditioner
        x_phase_first = self._to_phase_first(x).float()
        x_for_conditions = self._build_condition_input(x_phase_first)
        x_flat = x_phase_first.reshape(x_phase_first.shape[0], -1)

        # get conditions
        c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)

        if condition_dropout:
            # ==========================================
            # CONDITION DROPOUT
            # Randomly zero out conditions 30% of the time during training.
            # ==========================================
            if self.training and torch.rand(1).item() < 0.3:
                c = torch.zeros_like(c)

        # forward pass
        recon_params, q_params = self.forward(x_phase_first, c, num_samples=1)

        # elbo loss 
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        # gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_max_norm)

        optim.step()
        return loss_dict

    # accumulate metrics across validation batches
    @torch.no_grad()
    def validate(self, valloader, num_samples=1, beta=1.0, device="cpu"):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        for batch in valloader:
            batch = self.move_to_device(batch, device)
            x, meter_id, timestamps, _agg = self._unpack_batch(batch)
            x_phase_first = self._to_phase_first(x).float()
            x_for_conditions = self._build_condition_input(x_phase_first)
            x_flat = x_phase_first.reshape(x_phase_first.shape[0], -1)

            c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device)

            recon_params, q_params = self.forward(x_phase_first, c, num_samples=max(1, int(num_samples)))
            loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta)

            for k in agg:
                agg[k] += float(loss_dict[k])
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg

    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0, **_):
        """
        Permutation-Invariant ELBO Loss (Memory Optimized).
        Finds the best permutation without tracking gradients, then calculates 
        the differentiable loss exactly once.

        Aggregation consistency
        """
        # 1. Get KL Divergence
        kl_divergences = self.encoder.kl_divergence(posterior_params, prior=self.prior) # [B]
        
        if recon_params["mu"].dim() == 3:
            recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
        else:
            recon_params_squeezed = recon_params

        B = x.shape[0]
        L = self.sequence_length
        F = self.num_features

        x_phases = x.reshape(B, F, L)

        # --- STEP 1/2: Find the best permutation and construct optimal target ---
        if F <= 1:
            x_optimal = x
        else:
            # Full permutation search is factorial in F. Keep exact behavior for small F.
            max_perm_features = 6
            if F > max_perm_features:
                permutations = [list(range(F))]
            else:
                permutations = [list(p) for p in itertools.permutations(range(F))]

            with torch.no_grad():
                ll_scores = []
                for perm in permutations:
                    x_permuted = x_phases[:, perm, :]
                    x_perm_flat = x_permuted.reshape(B, -1)

                    ll = self.decoder.loglikelihood(x_perm_flat, recon_params_squeezed)
                    ll_scores.append(ll.unsqueeze(1))

                all_ll = torch.cat(ll_scores, dim=1)
                best_perm_idx = all_ll.argmax(dim=1)

            x_optimal = torch.empty_like(x)
            for i in range(B):
                best_p = permutations[int(best_perm_idx[i].item())]
                x_optimal[i] = x_phases[i, best_p, :].reshape(-1)

        # --- STEP 3: Do the heavy math exactly ONCE with gradients ON ---
        best_ll = self.decoder.loglikelihood(x_optimal, recon_params_squeezed)

        if True: # TODO
            # --- STEP 4: Aggregate Total Phases --- (THIS IS A TEST, WILL MAYBE BE USED IN FUTURE)
            # Get the predicted means and reshape to [B, F, L]
            mu_predicted = recon_params_squeezed["mu"].reshape(B, F, L)
            
            # Sum across the phases (F dimension) -> [B, L]
            mu_sum = mu_predicted.sum(dim=1)
            x_sum = x_phases.sum(dim=1)
            
            # Calculate MSE of the Aggregated Total
            # Summing over L (24 hours) to get the error per sample [B]
            agg_mse = torch.nn.functional.mse_loss(mu_sum, x_sum, reduction='none').sum(dim=-1)

            # --- STEP 3: Final Combined ELBO ---
            # Maximize LL, Minimize KL, Minimize Agg_MSE
            agg_weight = 0.25 # The total counts slightly less than an individual phase to encourage learning both individual and aggregate patterns
            elbo = best_ll - (beta * kl_divergences) - (agg_weight * agg_mse)
        else:
            elbo = best_ll - (beta * kl_divergences)

        loss = -elbo.mean()

        return {
            "loss": loss,
            "recon_loglikelihood": best_ll.mean(),
            "kl_divergence": kl_divergences.mean()
        }
    # def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0, **_):
    #     """
    #     Permutation-Invariant MSE Loss (Diagnostic Test Version).
    #     Replaces Gaussian NLL with pure Mean Squared Error. This forces the model 
    #     to learn the shape of 'mu' without being able to cheat using 'sigma'.
    #     """
    #     # 1. Get KL Divergence
    #     kl_divergences = self.encoder.kl_divergence(posterior_params, prior=self.prior) # [B]
        
    #     if recon_params["mu"].dim() == 3:
    #         recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
    #     else:
    #         recon_params_squeezed = recon_params

    #     B = x.shape[0]
    #     L = self.sequence_length
    #     F = self.num_features

    #     x_phases = x.reshape(B, F, L)
        
    #     # Flatten mu for standard MSE calculation
    #     mu_predicted = recon_params_squeezed["mu"].reshape(B, F * L)

    #     permutations = [
    #         [0, 1, 2], [0, 2, 1], 
    #         [1, 0, 2], [1, 2, 0], 
    #         [2, 0, 1], [2, 1, 0]
    #     ]

    #     # --- STEP 1: Find the best permutation (Lowest MSE) ---
    #     with torch.no_grad():
    #         mse_scores = []
    #         for perm in permutations:
    #             x_permuted = x_phases[:, perm, :]
    #             x_perm_flat = x_permuted.reshape(B, -1)
                
    #             # Calculate MSE per sample (Summing over features and sequence length)
    #             mse = torch.nn.functional.mse_loss(x_perm_flat, mu_predicted, reduction='none').sum(dim=1) 
    #             mse_scores.append(mse.unsqueeze(1))
                
    #         all_mse = torch.cat(mse_scores, dim=1) # [B, 6]
    #         # Since it is MSE, we want the MINIMUM error, not maximum likelihood
    #         best_perm_idx = all_mse.argmin(dim=1) # [B]

    #     # --- STEP 2: Construct the single optimal target batch ---
    #     x_optimal = torch.empty_like(x)
    #     for i in range(B):
    #         best_p = permutations[best_perm_idx[i]]
    #         x_optimal[i] = x_phases[i, best_p, :].reshape(-1)

    #     # --- STEP 3: Do the heavy math exactly ONCE with gradients ON ---
    #     best_mse = torch.nn.functional.mse_loss(x_optimal, mu_predicted, reduction='none').sum(dim=1)

    #     if True: # TODO
    #         # --- STEP 4: Aggregate Total Phases --- 
    #         mu_predicted_phases = mu_predicted.reshape(B, F, L)
            
    #         # Sum across the phases (F dimension) -> [B, L]
    #         mu_sum = mu_predicted_phases.sum(dim=1)
    #         x_sum = x_phases.sum(dim=1)
            
    #         # Calculate MSE of the Aggregated Total
    #         agg_mse = torch.nn.functional.mse_loss(mu_sum, x_sum, reduction='none').sum(dim=-1)

    #         # --- FINAL COMBINED LOSS ---
    #         # We MINIMIZE the total MSE, the Aggregate MSE, and the KL Divergence
    #         agg_weight = 0.25 
    #         total_loss_per_sample = best_mse + (beta * kl_divergences) + (agg_weight * agg_mse)
    #     else:
    #         total_loss_per_sample = best_mse + (beta * kl_divergences)

    #     loss = total_loss_per_sample.mean()

    #     return {
    #         "loss": loss,
    #         # We return NEGATIVE MSE as the log-likelihood stand-in. 
    #         # This prevents your logger from crashing and correctly shows the error approaching 0.
    #         "recon_loglikelihood": -best_mse.mean(), 
    #         "kl_divergence": kl_divergences.mean()
    #     }
# test 1: MSE loss
# test 2: standard prior. Lowers the available variance and encourages learning with the mean

# ==============
# NN Blocks
# ==============

# === MAB (Multihead Attention Block) ===
class MAB(nn.Module):
    """
    Multihead Attention Block as used in Set Transformers.
    multi-head attention + layer norm + feedforward (like Transformer)
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads
        if dim_V % num_heads != 0:
            raise ValueError(f"dim_V ({dim_V}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(dim_Q, dim_V)
        self.k_proj = nn.Linear(dim_K, dim_V)
        self.v_proj = nn.Linear(dim_K, dim_V)
        self.out_proj = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.ff = nn.Sequential(
            nn.Linear(dim_V, dim_V),
            nn.GELU(),
            nn.Linear(dim_V, dim_V),
        )

    def forward(self, Q, K):
        # Q: [B, Nq, dQ], K: [B, Nk, dK]
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(K)

        B, Nq, D = q.shape
        Nk = k.shape[1]
        H = self.num_heads
        Dh = D // H

        qh = q.reshape(B, Nq, H, Dh).transpose(1, 2)  # [B, H, Nq, Dh]
        kh = k.reshape(B, Nk, H, Dh).transpose(1, 2)  # [B, H, Nk, Dh]
        vh = v.reshape(B, Nk, H, Dh).transpose(1, 2)  # [B, H, Nk, Dh]

        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) / (Dh ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, vh)  # [B, H, Nq, Dh]
        out = out.transpose(1, 2).reshape(B, Nq, D)

        out = self.ln0(out + q)
        out = self.ln1(out + self.ff(out))
        return self.out_proj(out)

# === ISAB (Induced Set Attention Block) ===
class ISAB(nn.Module):
    """Direct attention between all element
    's (phases) in the set, with inducing points for efficiency."""
    def __init__(self, dim_in, dim_out, num_heads=4, num_inds=16):
        super(ISAB, self).__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inds, dim_out) * 0.02)
        self.mab0 = MAB(dim_Q=dim_out, dim_K=dim_in, dim_V=dim_out, num_heads=num_heads)
        self.mab1 = MAB(dim_Q=dim_in, dim_K=dim_out, dim_V=dim_out, num_heads=num_heads)

    def forward(self, X):
        # X: [B, N, dim_in]
        B = X.shape[0]
        I = self.inducing.expand(B, -1, -1)
        H = self.mab0(I, X)
        return self.mab1(X, H)

# === SAB (Set Attention Block) ===
class SAB(nn.Module):
    """
    Standard Self-Attention for Sets. 
    Exact O(N^2) attention. Best for small sets (like 3 phases).
    """
    def __init__(self, dim_in, dim_out, num_heads=4):
        super(SAB, self).__init__()
        self.mab = MAB(dim_Q=dim_in, dim_K=dim_in, dim_V=dim_out, num_heads=num_heads)

    def forward(self, X):
        # X: [B, N, dim_in]
        # Queries, Keys, and Values are all the exact same set
        return self.mab(X, X)

# === ABL (Attentive Bottleneck Layer) ===
class ABL(nn.Module):
    """Pools a set into a fixed-size representation using attention to learn importance weights."""
    def __init__(self, dim, num_heads=4, num_seed_vectors=1):
        super(ABL, self).__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, dim) * 0.02)
        self.mab = MAB(dim_Q=dim, dim_K=dim, dim_V=dim, num_heads=num_heads)

    def forward(self, X):
        # X: [B, N, dim] -> [B, num_seed_vectors, dim]
        B = X.shape[0]
        S = self.seed_vectors.expand(B, -1, -1)
        return self.mab(S, X)

