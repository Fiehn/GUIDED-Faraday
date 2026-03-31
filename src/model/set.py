from src.model.cvae import *
import torch
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
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            conditioner=conditioner,
            latent_dim=latent_dim,
            distribution_dict=distribution_dict,
            prior_type=prior_type,
            hierarchy_config=hierarchy_config,
            **kwargs,
        )

        # Expected 3-phase setup defaults to [B, 3, 24] (phase-first) or [B, 24, 3].
        self.sequence_length = int(sequence_length) # 24 hours
        self.num_features = int(num_features) if num_features is not None else max(1, input_dim // self.sequence_length) # 3 phases
        self.element_dim = self.sequence_length # each phase is a sequence of 24 hours
        self.num_set_elements = self.num_features # 3 elements in our set (the 3 phases)

        # Encoder architecture
        self.set_hidden_dim = int(set_hidden_dim)
        self.set_input_proj = nn.Linear(self.element_dim, self.set_hidden_dim) # projecting from 24 -> 128 dim for each set element
        self.set_encoder_blocks = nn.ModuleList([
            ISAB(
                dim_in=self.set_hidden_dim,
                dim_out=self.set_hidden_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
            )
            for _ in range(max(1, int(set_num_blocks)))
        ])
        self.set_pool = ABL(dim=self.set_hidden_dim, num_heads=set_num_heads, num_seed_vectors=1) # pools 3 phases into 1 vector

        # Replace posterior encoder: it now consumes pooled set representation + conditions.
        self.encoder = get_encoder_model(
            input_dim=self.set_hidden_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("encoder_type", "dictionary"),
            **kwargs,
        )

        # Keep decoder/prior from CVAE 
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=self.set_hidden_dim + self.condition_dim)

        # Count trainable parameters for reporting
        self.num_parameters = (
            sum(p.numel() for p in self.set_input_proj.parameters() if p.requires_grad)
            + sum(p.numel() for blk in self.set_encoder_blocks for p in blk.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.set_pool.parameters() if p.requires_grad)
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

    # transpose input only for conditioner 
    def _build_condition_input(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Convert [B, F, L] -> [B, L, F] for conditioner compatibility."""
        return x_phase_first.transpose(1, 2)

    # permuation-invariatn encoder 
    def _set_encode(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Encode set elements to a pooled representation [B, H]."""
        h = self.set_input_proj(x_phase_first)  # [B, F, H] # projecting from 24 -> 128 dim for each phase
        for blk in self.set_encoder_blocks:
            h = blk(h) # passing through 2 ISAB blocks with attention between phases
        pooled = self.set_pool(h).squeeze(1)  # [B, H] # pooling 3 phases into 1 vector using attentive pooling
        return pooled

    def _unpack_batch(self, batch):
        """ Unpack batch supporting (x, meter_id, timestamps), (x, cond), or (x, meter_id) formats. """
        if len(batch) == 3:
            x, meter_id, timestamps = batch
            return x, meter_id, timestamps

        if len(batch) != 2:
            raise ValueError("SetGuideVAE expects batches as (x, meter_id, timestamps), (x, meter_id), or (x, cond)")

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

    def train_core(self, batch, optim, beta=1.0, weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        self.train()

        x, meter_id, timestamps = self._unpack_batch(batch)

        # prepare inputs: ensure phase-first format for encoder and conditioner
        x_phase_first = self._to_phase_first(x).float()
        x_for_conditions = self._build_condition_input(x_phase_first)
        x_flat = x_phase_first.reshape(x_phase_first.shape[0], -1)

        # get conditions
        c = self.conditioner.get_conditions(x_for_conditions, meter_id=meter_id, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)

        # forward pass
        recon_params, q_params = self.forward(x_phase_first, c, num_samples=1)

        # elbo loss 
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        # gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

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
            x, meter_id, timestamps = self._unpack_batch(batch)
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

        permutations = [
            [0, 1, 2], [0, 2, 1], 
            [1, 0, 2], [1, 2, 0], 
            [2, 0, 1], [2, 1, 0]
        ]

        # --- STEP 1: Find the best permutation ---
        with torch.no_grad():
            ll_scores = []
            for perm in permutations:
                x_permuted = x_phases[:, perm, :]
                x_perm_flat = x_permuted.reshape(B, -1)
                ll = self.decoder.loglikelihood(x_perm_flat, recon_params_squeezed) 
                ll_scores.append(ll.unsqueeze(1))
                
            all_ll = torch.cat(ll_scores, dim=1) # [B, 6]
            best_perm_idx = all_ll.argmax(dim=1) # [B]

        # --- STEP 2: Construct the single optimal target batch ---
        x_optimal = torch.empty_like(x)
        for i in range(B):
            best_p = permutations[best_perm_idx[i]]
            x_optimal[i] = x_phases[i, best_p, :].reshape(-1)

        # --- STEP 3: Do the heavy math exactly ONCE with gradients ON ---
        best_ll = self.decoder.loglikelihood(x_optimal, recon_params_squeezed)

        if False:
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

