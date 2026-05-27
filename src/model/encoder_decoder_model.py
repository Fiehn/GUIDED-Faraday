import torch
import torch.nn as nn
import torch.nn.functional as F
from model.priors import *

# === Gaussian encoder and decoder Model ===
class GaussianCoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim,  
                 learn_sigma=False, 
                 sigma_fixed=1.0,
                 num_hidden_layers=2,
                 num_neurons = 64,
                 hidden_dims=None,
                 dropout = True,
                 dropout_rate = 0.2,
                 batch_norm = False,
                 **_):
        super(GaussianCoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_sigma = learn_sigma
        self.sigma_fixed = sigma_fixed

        self.parameterizer = Parameterizer(
            input_dim=input_dim,
            output_dim=output_dim,
            dist_parameters=["mu", "sigma"],
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons,
            hidden_dims=hidden_dims,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["mu"] = torch.clamp(param_dict["mu"], min=0.0, max=30.0)
        if self.learn_sigma:
            param_dict["sigma"] = F.softplus(param_dict["sigma"], beta=15.0) + 1e-6 # Ensure positivity 
        else:
            param_dict["sigma"] = torch.ones_like(param_dict["mu"]) * self.sigma_fixed # Fixed sigma
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1, return_mean=False, **_):
        """Sample from the distribution. If return_mean is True, ignores noise."""
        if param_dict is None:
            raise ValueError("param_dict must be provided for sampling")
        
        if return_mean:
            # Safely return just the predicted mean without log-normal explosions
            return param_dict["mu"].unsqueeze(0).expand(num_samples, -1, -1)
            
        return self.reparameterize(param_dict, num_samples=num_samples)

    def loglikelihood(self, inputs, param_dict, **_):
        """ Compute log likelihood of inputs under the Gaussian defined by param_dict """
        mu = param_dict["mu"]
        sigma = param_dict["sigma"]
        var = sigma ** 2
        log_likelihood = -0.5 * torch.sum(((inputs - mu) ** 2) / var + torch.log(2 * torch.pi * var), dim=1)
        return log_likelihood

    def kl_divergence(self, param_dict_q, prior=None, **_):
        """
        Compute KL divergence KL(q||p).

        For priors that support analytical KL (diagonal Gaussians), uses the
        closed-form expression.  For non-analytical priors (VampPrior), uses
        single-sample Monte Carlo:  KL ≈ log q(z|x) - log p(z).
        
        Args:
            param_dict_q: Posterior parameters
            prior: BasePrior object (if None, uses standard normal)
        """
        # === Monte Carlo path (VampPrior, etc.) ===
        if prior is not None and not getattr(prior, 'supports_analytical_kl', True):
            z = self.reparameterize(param_dict_q, num_samples=1).squeeze(0)  # [B, D]
            log_q = self.loglikelihood(z, param_dict_q)  # [B]
            log_p = prior.log_prob(z)                     # [B]
            return log_q - log_p

        # === Analytical path (diagonal Gaussian priors) ===
        mu_q = param_dict_q["mu"]
        sigma_q = param_dict_q["sigma"]
        
        if prior is None:
            # KL divergence with standard normal N(0,I)
            mu_p = torch.zeros_like(mu_q)
            sigma_p = torch.ones_like(sigma_q)
        else:
            prior_params = prior.get_params()
            batch_size = mu_q.shape[0]
            prior_params = {k: v.expand(batch_size, -1).to(mu_q.device) for k, v in prior_params.items()}
            mu_p = prior_params["mu"]
            sigma_p = prior_params["sigma"]
        
        var_q = sigma_q ** 2
        var_p = sigma_p ** 2
        
        # Analytical KL: KL(q||p) = 0.5 * sum(var_p/var_q + (mu_q - mu_p)^2/var_p - 1 + log(var_q/var_p))
        kl = 0.5 * torch.sum(
            var_q / var_p + ((mu_q - mu_p) ** 2) / var_p - 1 + torch.log(var_p / var_q),
            dim=-1
        )
        return kl
    
    def create_covariance_matrix(self, param_dict):
        # Here in the Gaussian case, we create a diagonal covariance matrix from the sigma parameters
        return torch.eye(param_dict["mu"].shape[1]).to(param_dict["mu"].device) * (param_dict["sigma"] ** 2).unsqueeze(-1)
    
    def reparameterize(self, param_dict, num_samples=1, temperature: float = 1.0, **_):
        # Reparameterization trick: sample epsilon from N(0,1) and return mu + sigma * epsilon
        mu = param_dict["mu"]
        sigma = param_dict["sigma"]

        if temperature <= 0.0:
            # If temperature is 0 or negative, return the mean (deterministic output)
            return mu.unsqueeze(0).expand(num_samples, -1, -1)
        sigma = sigma * temperature

        batch_size, output_dim = mu.shape
        # Sample epsilon
        epsilon = torch.randn((num_samples, batch_size, output_dim)).to(mu.device)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        return samples
    
    def get_SigmaMapper(self):
        # In Gaussian case, there is no SigmaMapper
        return None

# =================
# Dictionary approach with PDCC matrix as the covariance (Guide VAE)
# =================
class DictionaryCoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dictionary_size=100,
                 sigma_lim=0.1,
                 marginal_std_lim=0.05,
                 average_max_std=1.0,
                 num_hidden_layers=2,
                 num_neurons=64,
                 hidden_dims=None,
                 dropout=True,
                 dropout_rate=0.2,
                 batch_norm=False,
                 layer_type='MLP',
                 learn_sigma=False,
                 mu_activation: str = "tanh", # Options: "softplus", "tanh", "none"
                 **kwargs):
        super(DictionaryCoder, self).__init__()
        if dictionary_size <= output_dim:
            raise ValueError("Dictionary size must be greater than output dimension")
        self.dictionary_size = dictionary_size

        self.marginal_std_lim = marginal_std_lim
        self.average_max_std = average_max_std
        scale_lim = (output_dim / dictionary_size) ** 0.5 # The variance is additive in the covariance, so we need to scale the sigma_lim by the sqrt of the ratio of output_dim to dictionary_size to get the right scale in the output space. This is because each dictionary element contributes to the variance in the output space, and we want to ensure that the total variance is in a reasonable range.
        self.sigma_lower_lim = sigma_lim * scale_lim
        self.sigma_upper_lim = (output_dim/self.dictionary_size)**0.5 * (average_max_std - self.marginal_std_lim) # TODO: UNderstand this step
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mu_activation = mu_activation

        layer_type_key = str(layer_type).lower()
        seq_len = int(kwargs.get("seq_len", kwargs.get("sequence_length", 24)))
        if seq_len <= 0:
            seq_len = 24

        if layer_type_key in {"conv1d", "cnn"}:
            num_features = kwargs.get("num_features", None)
            if num_features is None:
                if output_dim % seq_len == 0:
                    num_features = max(1, output_dim // seq_len)
                else:
                    num_features = 1
                    seq_len = int(output_dim)
            self.parameterizer = HybridCNNParameterizer(
                input_dim=input_dim,
                output_dim=output_dim,
                dictionary_size=dictionary_size,
                seq_len=seq_len,
                num_features=int(num_features),
                hidden_dim=hidden_dims,
            )
        elif layer_type_key in {"conv1d_encoder", "cnn_encoder"}:
            cond_dim = int(kwargs.get("cond_dim", 0))
            signal_dim = input_dim - cond_dim
            num_features = kwargs.get("num_features", None)
            if num_features is None:
                if signal_dim % seq_len == 0:
                    num_features = max(1, signal_dim // seq_len)
                else:
                    num_features = 1
                    seq_len = int(signal_dim)
            self.parameterizer = TemporalCNNEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                num_features=int(num_features),
                dist_parameters=["mu", "sigma"] if learn_sigma else ["mu"],
                param_output_dims={"mu": output_dim, "sigma": dictionary_size} if learn_sigma else None,
                hidden_dims=hidden_dims,
                cond_dim=cond_dim,
            )
        else:
            self.parameterizer = Parameterizer(
                input_dim=input_dim,
                output_dim={"sigma": dictionary_size, "mu": output_dim},
                dist_parameters=["mu", "sigma"],
                num_hidden_layers=num_hidden_layers,
                num_neurons=num_neurons,
                hidden_dims=hidden_dims,
                dropout=dropout,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                resnets=bool(kwargs.get("resnets", True)),
                block_type=layer_type,
            )
        
        # The dictionary SigmaMapper is a learnable parameter
        # It maps the dictionary codes to the output space
        # So it would be the U dictionary in the guide paper
        # Setting this to be a learnable parameter is equivalent to the dictionary being learned in training through the data
        self.SigmaMapper = torch.nn.Parameter(torch.randn(output_dim, dictionary_size), requires_grad=True)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def get_SigmaMapper(self):
        # The sigmaMapper is normalized to have unit norm
        return self.SigmaMapper / torch.norm(self.SigmaMapper, dim=0, keepdim=True)
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)

        if self.mu_activation == "softplus":
            # Non-negative outputs — correct for log1p-normalised or raw non-negative data
            param_dict["mu"] = F.softplus(param_dict["mu"], beta = 15.0)
        elif self.mu_activation == "tanh":
            # Symmetric [-5, 5] — use for z-score normalised data or latent encoder codes
            param_dict["mu"] = torch.tanh(param_dict["mu"]) * 5.0
        # else "none": unbounded, trust the optimizer

        
        max_sigma_range = self.sigma_upper_lim - self.sigma_lower_lim
        param_dict["sigma"] = self.sigma_lower_lim + torch.sigmoid(param_dict["sigma"]) * max_sigma_range

        return param_dict

    def reparameterize(self, param_dict, num_samples=1, temperature: float = 1.0, **_):
        """Reparameterization trick for sampling from the multivariate Gaussian defined by param_dict"""
        mu, sigma = param_dict["mu"], param_dict["sigma"]
        batch_size = mu.shape[0]

        if temperature <= 0.0:
            # If temperature is 0 or negative, return the mean (deterministic output)
            return mu.unsqueeze(0).expand(num_samples, -1, -1)
        sigma = sigma * temperature

        eps = torch.randn((num_samples, mu.shape[0], self.dictionary_size)).to(mu.device)

        # Scale by sigma in dictionary space: [num_samples, batch_size, dictionary_size]
        z_dict = eps * sigma.unsqueeze(0)

        # Map to output space using SigmaMapper: [output_dim, dictionary_size]
        U = self.get_SigmaMapper()  # [output_dim, dictionary_size]

        # Transform: [num_samples, batch_size, dictionary_size] @ [dictionary_size, output_dim]
        # -> [num_samples, batch_size, output_dim]
        z_output = torch.einsum('sbd,do->sbo', z_dict, U.T) # (done on GPU this way)

        # Add mean in output space
        z = mu.unsqueeze(0) + z_output

        return z
    
    def sample(self, param_dict=None, num_samples=1, temperature: float = 1.0, **_):
        """Sample from the distribution.

        Returns raw reparameterised samples in the model's output space (normalised
        when a Normalizer is in use).  Non-negativity in original (kWh) space is
        guaranteed by the Normalizer's own ``torch.clamp(..., min=0.0)`` in
        ``inverse_transform``; applying softplus here is wrong because it floors the
        *normalised* output at ~0, which after inverse-transform maps to
        ``expm1(normalizer_mean) > 0`` rather than 0.  See regression test
        ``test_dictionary_coder_sample_floor.py``.
        """
        if param_dict is None:
            raise ValueError("param_dict must be provided for sampling")

        return self.reparameterize(param_dict, num_samples=num_samples, temperature=temperature)
    
    def loglikelihood(self, inputs, param_dict,**_):
        """Compute log likelihood under multivariate Gaussian with PDCC covariance."""
        mu = param_dict['mu']
        Sigma = self.create_covariance_matrix(param_dict)  # [B, D, D]

        # Symmetrize first
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        D = Sigma.shape[-1]
        # Add jitter and use Cholesky — always PD, more numerically stable (It should have been stable but it seems to be an issue in practice, Maybe the heirarchy is causing some instability? I will need to check this later)
        jitter = 1e-5 * torch.eye(D, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
        Sigma_jittered = Sigma + jitter
        
        # Using scale_tril with Cholesky decomposition for better numerical stability (should be equivalent to covariance_matrix but more stable)
        L = torch.linalg.cholesky(Sigma_jittered)
        
        dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
        return dist.log_prob(inputs)
    
    def kl_divergence(self, param_dict_q, prior=None, **_):
        """
        Compute KL divergence KL(q||p).

        For priors that support analytical KL (e.g. StandardDictionaryPrior),
        uses the closed-form multivariate-Gaussian KL.

        For priors that do NOT (e.g. VampPrior), falls back to single-sample
        Monte Carlo:  KL ≈ E_q[log q(z|x) - log p(z)].
        
        Args:
            param_dict_q: Posterior parameters from the encoder
            prior: BasePrior object
        """
        # --- Monte Carlo path (VampPrior and other non-analytical priors) ---
        if prior is not None and not getattr(prior, 'supports_analytical_kl', True):
            z = self.reparameterize(param_dict_q, num_samples=1).squeeze(0)  # [B, D]
            log_q = self.loglikelihood(z, param_dict_q)  # [B]  — uses PDCC covariance
            log_p = prior.log_prob(z)                     # [B]
            return log_q - log_p

        # --- Analytical path (Gaussian priors) ---
        mu_q = param_dict_q["mu"]
        Sigma_q = self.create_covariance_matrix(param_dict_q)

        if prior is None:
            # Default to standard normal in output space
            mu_p = torch.zeros_like(mu_q)
            Sigma_p = torch.eye(self.output_dim, device=mu_q.device).unsqueeze(0).expand(mu_q.shape[0], -1, -1)
        else:
            # Get prior parameters and create covariance
            prior_params = prior.get_params()
            batch_size = mu_q.shape[0]
            prior_params = {k: v.expand(batch_size, -1).to(mu_q.device) for k, v in prior_params.items()}
            mu_p = prior_params["mu"]
            Sigma_p = self.create_covariance_matrix(prior_params)
        
        # Use analytical KL for multivariate Gaussians
        try:
            dist_q = torch.distributions.MultivariateNormal(loc=mu_q, covariance_matrix=Sigma_q)
            dist_p = torch.distributions.MultivariateNormal(loc=mu_p, covariance_matrix=Sigma_p)
            return torch.distributions.kl.kl_divergence(dist_q, dist_p)
        except Exception as e:
            print(f"Warning: Analytical KL failed, using Monte Carlo: {e}")
            z = self.reparameterize(param_dict_q, num_samples=1).squeeze(0)  # [B, D]
            log_q = self.loglikelihood(z, param_dict_q)
            if prior is None:
                log_p = -0.5 * torch.sum(z ** 2 + torch.log(torch.tensor(2 * torch.pi)), dim=-1)
            else:
                # Dictionary priors can parameterize sigma in dictionary space.
                # For MC fallback we evaluate p(z) in latent/output space using the
                # already-constructed Gaussian prior covariance.
                Sigma_p_sym = 0.5 * (Sigma_p + Sigma_p.transpose(-1, -2))
                D = Sigma_p_sym.shape[-1]
                jitter = 1e-5 * torch.eye(D, device=Sigma_p_sym.device, dtype=Sigma_p_sym.dtype).unsqueeze(0)
                L_p = torch.linalg.cholesky(Sigma_p_sym + jitter)
                dist_p = torch.distributions.MultivariateNormal(loc=mu_p, scale_tril=L_p)
                log_p = dist_p.log_prob(z)
            return log_q - log_p



    def create_covariance_matrix(self, param_dict):
        # THis is the spot that is interesting 
        # First get the U (sigma mapper)
        U = self.get_SigmaMapper()  # Output_dim x Dictionary_size
        # Then scale by the sigma parameters (learned per sample)
        # sigma has shape [batch_size, dictionary_size], we need [batch_size, dictionary_size, 1]
        # U has shape [output_dim, dictionary_size], we need it transposed to [dictionary_size, output_dim]
        S = param_dict["sigma"][..., None] * U.T  # Batch_size x Dictionary_size x Output_dim
        # Finally return the covariance matrix by S^T @ S + marginal_std_lim^2 * I
        # where S is the scaled U matrix
        # This ensures positive definiteness and adds marginal noise
        return (S.mT @ S) + torch.eye(self.output_dim, device=param_dict["mu"].device) * (self.marginal_std_lim ** 2)  # Batch_size x Output_dim x Output_dim

    def get_marginal_sigmas(self,param_dict):
        return torch.sqrt(torch.diagonal(self.create_covariance_matrix(param_dict), dim1=-2, dim2=-1))

# =================
# Autoregressive Transformer Approach
# =================

class AutoregressiveTransformerCoder(torch.nn.Module):
    def __init__(self, 
                 latent_dim, 
                 condition_dim, 
                 num_bins=256, 
                 seq_len=72, 
                 d_model=256, 
                 num_layers=4, 
                 nhead=8):
        super().__init__()
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. Embed the integer tokens (like LLMs)
        self.token_embedding = nn.Embedding(num_bins, d_model)

        # 2. Positional encoding for autoregressive structure (So the model knows what hour/phase it is generating)
        self.positional_embedding = nn.Embedding(seq_len, d_model)

        # 3. Project the "prompt" (latent vector + conditions) to the model dimension
        self.prompt_projection = nn.Linear(latent_dim + condition_dim, d_model)

        # 4. The Transformer (We use standard Encoder layers but apply a Causal Mask) TODO: replace with different method if working well
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # 5. Output Head (Predicts probabilities for each bin at each time step)
        self.to_logits = nn.Linear(d_model, num_bins)

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate_causal_mask(self, sz, device):
        # Creates a triangle of -inf. Prevents looking into the future. 
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, z, c, target_tokens):
        """
        TRAINING PASS: Uses "Teacher Forcing". 
        We pass the entire sequence at once, but the causal mask prevents cheating.
        """
        B = z.shape[0]
        device = z.device

        # 1. Create the prompt by concatenating z and c, then projecting
        prompt = torch.cat([z, c], dim=-1)  # [B, latent_dim + condition_dim]
        prompt_emb = self.prompt_projection(prompt).unsqueeze(1)  # [B, 1, d_model]

        # 2. Embed the target tokens and add positional encoding
        positions = torch.arange(0, self.seq_len, device=device).unsqueeze(0).expand(B, self.seq_len)  # [B, seq_len]
        token_emb = self.token_embedding(target_tokens) + self.positional_embedding(positions) # [B, seq_len, d_model]

        # 3. Shift targets right so we predict the NEXT token
        # We prepend the prompt as the very first token!
        # Input sequence: [Prompt, Token_0, Token_1, ..., Token_70]
        # Target sequence to predict: [Token_0, Token_1, Token_2, ..., Token_71]
        transformer_input = torch.cat([prompt_emb, token_emb[:, :-1, :]], dim=1)  # [B, seq_len, d_model]

        # 4. Apply cauysal mask
        causal_mask = self.generate_causal_mask(self.seq_len, device)  # [seq_len, seq_len]
        transformer_out = self.transformer(transformer_input, mask=causal_mask)  # [B, seq_len, d_model]

        # 5. Predict logits for each token
        logits = self.to_logits(transformer_out)  # [B, seq_len, num_bins]
        return {"logits": logits}

    @torch.no_grad()
    def sample(self, z, c, temperature=1.0):
        """
        GENERATION PASS: Autoregressive sampling.
        We must generate one token at a time, feeding the result back into the model.
        """
        B = z.shape[0]
        device = z.device
        
        prompt = torch.cat([z, c], dim=-1)
        prompt_emb = self.prompt_projection(prompt).unsqueeze(1) # [B, 1, d_model]
        
        # Start with an empty sequence
        generated_tokens = []

        for t in range(self.seq_len):
            if t == 0:
                transformer_input = prompt_emb  # First input is just the prompt
            else:
                # Embed the generated tokens so far and add positional encoding
                past_tokens = torch.stack(generated_tokens, dim=1)  # [B, t]
                positions = torch.arange(0, t, device=device).unsqueeze(0).expand(B, t)  # [B, t]
                token_emb = self.token_embedding(past_tokens) + self.positional_embedding(positions)  # [B, t, d_model]

                # Input to transformer is prompt + past tokens
                transformer_input = torch.cat([prompt_emb, token_emb], dim=1)  # [B, t+1, d_model]

            # run transformer with appropriate causal mask
            causal_mask = self.generate_causal_mask(transformer_input.size(1), device)
            transformer_out = self.transformer(transformer_input, mask=causal_mask)  # [B, t+1, d_model]

            # Last prediction
            last_timestep_out = transformer_out[:, -1, :]  # [B, d_model]
            next_token_logits = self.to_logits(last_timestep_out) / temperature  # [B, num_bins]

            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1)  # [B]
            else:
                probs = F.softmax(next_token_logits, dim=-1)  # [B, num_bins]
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            generated_tokens.append(next_token)
        
        # Stack generated tokens into a tensor
        final_sequence = torch.stack(generated_tokens, dim=1)  # [B, seq_len]
        return final_sequence
    


# === This is the actual network structure ===
class Parameterizer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dist_parameters,
                 num_neurons=128,
                 hidden_dims=None,
                 dropout=False,
                 dropout_rate=0.2,
                 batch_norm=False,
                 resnets=True,
                 block_type='MLP',
                 **_):
        super(Parameterizer, self).__init__()
        self.dist_parameters = dist_parameters
        self.num_neurons = num_neurons
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        
        
        # Shared block using NNBlock (outputs to num_neurons)
        self.shared_block = NNBlock(
            input_dim=input_dim,
            output_dim=num_neurons,
            num_neurons=num_neurons,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            hidden_dims=self.hidden_dims,
            resnets=resnets,
            type=block_type
        )

        # Build parameter-specific heads using NNBlock
        self.param_heads = nn.ModuleDict()
        
        if isinstance(output_dim, int):
            for param in dist_parameters:
                self.param_heads[param] = NNBlock(
                    input_dim=num_neurons,
                    output_dim=output_dim,
                    num_neurons=num_neurons,
                    dropout=dropout,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm,
                    num_hidden_layers=1,  # Shallow head (TODO: can experiment with deeper heads for more complex distributions)
                    resnet=resnets,
                    type=block_type
                )
        elif isinstance(output_dim, dict):
            for key in dist_parameters:
                if key in output_dim:
                    self.param_heads[key] = NNBlock(
                        input_dim=num_neurons,
                        output_dim=output_dim[key],
                        num_neurons=num_neurons,
                        dropout=dropout,
                        dropout_rate=dropout_rate,
                        batch_norm=batch_norm,
                        num_hidden_layers=1,  # Shallow head
                        resnet=resnets,
                        type=block_type
                    )

    def forward(self, x):
        h = self.shared_block(x)
        param_dict = {}
        for param in self.dist_parameters:
            param_dict[param] = self.param_heads[param](h)
        return param_dict
        
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class HybridCNNParameterizer(nn.Module):
    """
    Fuses a Temporal CNN (for the mean) with a flat Dictionary MLP (for the covariance).
    This allows temporal precision for spike timing while keeping GUIDE-VAE's PDCC matrix.
    """
    def __init__(self, input_dim, output_dim, 
                 dictionary_size, 
                 seq_len=24, 
                 num_features=3, 
                 hidden_dim = None,
                 **_):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.output_dim = output_dim # 72 (24 * 3)
        self.dictionary_size = dictionary_size
        
        # ==========================================
        # BRANCH 1: CNN for the Mean (mu)
        # ==========================================
        self.seed_length = seq_len // 4  # 6 hours
        self.seed_channels = 128
        self.project_seed = nn.Linear(input_dim, self.seed_channels * self.seed_length)
        
        if hidden_dim:
            cnn_hidden = [int(v) for v in hidden_dim if int(v) > 0]
            c1 = cnn_hidden[0] if len(cnn_hidden) > 0 else 64
            c2 = cnn_hidden[1] if len(cnn_hidden) > 1 else max(c1 // 2, 16)
            c3 = cnn_hidden[2] if len(cnn_hidden) > 2 else max(c2 // 2, self.num_features)
            self.mu_upsample = nn.Sequential(
                nn.ConvTranspose1d(self.seed_channels, c1, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(c1),
                nn.GELU(),
                nn.ConvTranspose1d(c1, c2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(c2),
                nn.GELU(),
                nn.Conv1d(c2, c3, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(c3, self.num_features, kernel_size=3, padding=1),
            )
        else:
            self.mu_upsample = nn.Sequential(
                nn.ConvTranspose1d(self.seed_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(16, self.num_features, kernel_size=3, padding=1),
            )

        # ==========================================
        # BRANCH 2: MLP for the Dictionary Variance (sigma)
        # ==========================================
        # Sigma expects a flat vector of size `dictionary_size`
        self.sigma_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.dictionary_size)
        )

    def forward(self, x):
        B = x.shape[0]
        param_dict = {}

        # --- 1. Generate mu using CNN ---
        h_cnn = self.project_seed(x)
        h_cnn = h_cnn.view(B, self.seed_channels, self.seed_length) 
        out_mu = self.mu_upsample(h_cnn) # [B, 3, 24]
        if out_mu.shape[-1] != self.seq_len:
            # Keep the temporal output size aligned with the configured sequence length.
            out_mu = F.interpolate(out_mu, size=self.seq_len, mode="linear", align_corners=False)
        param_dict["mu"] = out_mu.reshape(B, -1) # Flatten to [B, 72] for loss function compatibility
        
        # --- 2. Generate sigma dictionary weights using MLP ---
        param_dict["sigma"] = self.sigma_mlp(x) # [B, dictionary_size]

        return param_dict

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TemporalCNNEncoder(nn.Module):
    """
    A CNN-1D Encoder. 
    Reads a 24-hour time series, detects temporal patterns (spikes), 
    and compresses it down to a flat vector for the latent space bottleneck.
    """
    def __init__(self, input_dim,
                 output_dim,
                 seq_len=24,
                 num_features=3,
                 dist_parameters=["mu", "sigma"],
                 param_output_dims=None,
                 hidden_dims=None,
                 cond_dim=0,
                 **_):
        super().__init__()
        self.dist_parameters = dist_parameters
        self.param_output_dims = param_output_dims or {}
        self.cond_dim = int(cond_dim)
        # CNN trunk sees only the signal; conditions are injected at the bottleneck.
        self.num_features = int(num_features)
        self.seq_len = int(seq_len)

        self.downsample_blocks = nn.Sequential()

        if hidden_dims:
            in_channels = self.num_features
            for i, out_channels in enumerate([int(v) for v in hidden_dims]):
                self.downsample_blocks.add_module(f"Conv_{i}", nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
                self.downsample_blocks.add_module(f"BatchNorm_{i}", nn.BatchNorm1d(out_channels))
                self.downsample_blocks.add_module(f"GELU_{i}", nn.GELU())
                if (i+1) % 2 == 0:  # Downsample every 2 layers
                    self.downsample_blocks.add_module(f"Downsample_{i}", nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
                    self.downsample_blocks.add_module(f"BatchNorm_Down_{i}", nn.BatchNorm1d(out_channels))
                    self.downsample_blocks.add_module(f"GELU_Down_{i}", nn.GELU())
                in_channels = out_channels
        else:
            # Downsampling: [B, 3, 24] -> [B, 128, 6]
            self.downsample_blocks = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),

                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),

                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )

        # Infer flattened size from the actual downsampling stack.
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_features, self.seq_len)
            flattened_size = int(self.downsample_blocks(dummy).reshape(1, -1).shape[1])

        # Parameter heads receive [CNN bottleneck + conditions].
        head_input_size = flattened_size + self.cond_dim
        self.param_heads = nn.ModuleDict()
        for param in dist_parameters:
            head_output_dim = self.param_output_dims.get(param)
            if head_output_dim is None:
                head_output_dim = output_dim
            self.param_heads[param] = nn.Sequential(
                nn.Linear(head_input_size, 256),
                nn.GELU(),
                nn.Linear(256, int(head_output_dim))
            )

    def forward(self, x):
        B = x.shape[0]

        if self.cond_dim > 0:
            # Split signal and conditions; only signal enters the CNN trunk.
            c = x[:, -self.cond_dim:]
            x = x[:, :-self.cond_dim]
        else:
            c = None

        # If input is flat [B, signal_dim], reshape to [B, num_features, seq_len]
        if x.dim() == 2:
            if x.shape[1] % self.num_features != 0:
                raise ValueError(
                    f"TemporalCNNEncoder expected flattened signal dim divisible by num_features={self.num_features}, got {x.shape[1]}"
                )
            x = x.view(B, self.num_features, -1)
        elif x.dim() == 3 and x.shape[1] != self.num_features and x.shape[2] == self.num_features:
            x = x.transpose(1, 2)

        h = self.downsample_blocks(x)
        h = h.view(B, -1)

        # Inject conditions at the bottleneck (after CNN, before parameter heads).
        if c is not None:
            h = torch.cat([h, c], dim=-1)

        param_dict = {}
        for param in self.dist_parameters:
            param_dict[param] = self.param_heads[param](h)

        return param_dict

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=128, dropout=False, 
                 dropout_rate=0.2, batch_norm=False,num_hidden_layers=2, 
                 resnet=True, resnets=None, type='MLP', hidden_dims=None, **_):
        super(NNBlock, self).__init__()
        if resnets is not None:
            resnet = bool(resnets)
        self.resnet = resnet
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.num_hidden_layers = num_hidden_layers
        self.layer_type = type

        if hidden_dims is not None and len(hidden_dims) > 0:
            self.hidden_dims = [int(n) for n in hidden_dims]
        else:
            depth = max(0, int(self.num_hidden_layers))
            # Include one dimension for input projection and `depth` hidden transitions.
            self.hidden_dims = [int(self.num_neurons)] * (depth + 1)

        self.input_layer = self._build_input_layer()
        self.hidden_layer = self._build_hidden_layer()
        self.output_layer = self._build_output_layer()
        
        # Projection layer for residual connection if dimensions don't match
        if self.resnet and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hidden_layer(h)
        out = self.output_layer(h)
        
        # Add residual connection with proper dimension handling
        if self.resnet:
            if self.residual_proj is not None:
                # Use projection layer for dimension mismatch
                residual = self.residual_proj(x)
            elif x.shape[-1] == out.shape[-1]:
                # Direct residual if dimensions match
                residual = x
            else:
                # Slice residual if possible (backward compat)
                residual = x[:, :out.shape[-1]] if x.shape[-1] >= out.shape[-1] else x
            return out + residual
        else:
            return out
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _build_input_layer(self):
        start_n_neurons = self.hidden_dims[0]
        layers: list[nn.Module]
        if self.layer_type == 'MLP':
            layers = [nn.Linear(self.input_dim, start_n_neurons), nn.ReLU()]
        elif self.layer_type == 'Conv1D':
            layers = [nn.Conv1d(self.input_dim, start_n_neurons, kernel_size=3, padding=1), nn.ReLU()]
        elif self.layer_type == 'LSTM':
            layers = [nn.LSTM(self.input_dim, start_n_neurons, batch_first=True)]
        elif self.layer_type == 'Transformer':
            layers = [nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, dim_feedforward=start_n_neurons), nn.ReLU()]
        else:
            layers = [nn.Linear(self.input_dim, start_n_neurons), nn.ReLU()]

        if self.batch_norm:
            layers.append(nn.LayerNorm(start_n_neurons))
        if self.dropout:
            layers.append(nn.Dropout(self.dropout_rate * 0.5))  # Less dropout in the first layer
        return nn.Sequential(*layers)
    def _build_hidden_layer(self):
        layers = []
        for in_n, out_n in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            if self.layer_type == 'MLP':
                layers.append(nn.Linear(in_n, out_n))
                layers.append(nn.ReLU())
            elif self.layer_type == 'Conv1D':
                layers.append(nn.Conv1d(in_n, out_n, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            elif self.layer_type == 'LSTM':
                layers.append(nn.LSTM(in_n, out_n, batch_first=True))
            elif self.layer_type == 'Transformer':
                layers.append(nn.TransformerEncoderLayer(d_model=in_n, nhead=4, dim_feedforward=out_n))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_n, out_n))
                layers.append(nn.ReLU())

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(out_n))
            if self.dropout:
                layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)
    def _build_output_layer(self):
        end_n_neurons = self.hidden_dims[-1]
        if self.layer_type == 'MLP':
            layers = [nn.Linear(end_n_neurons, self.output_dim)] 
        elif self.layer_type == 'Conv1D':
            layers = [nn.Conv1d(end_n_neurons, self.output_dim, kernel_size=3, padding=1)]
        elif self.layer_type == 'LSTM':
            layers = [nn.LSTM(end_n_neurons, self.output_dim, batch_first=True)]
        elif self.layer_type == 'Transformer':
            layers = [nn.TransformerEncoderLayer(d_model=end_n_neurons, nhead=4, dim_feedforward=end_n_neurons), nn.ReLU(), nn.Linear(end_n_neurons, self.output_dim)]
        else:
            layers = [nn.Linear(end_n_neurons, self.output_dim)] 
        return nn.Sequential(*layers)

    
def get_encoder_model(input_dim, latent_dim, type, average_max_std=1.0, hidden_dims=None, **kwargs):
    # Encoder latent codes must be unconstrained — force mu_activation off
    # so softplus/tanh don't push the posterior away from a symmetric prior.
    kwargs.setdefault("mu_activation", "none")
    if type == "dictionary":
        return DictionaryCoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            average_max_std=average_max_std,
            **kwargs
        )
    elif type == "dictionary_cnn":
        return DictionaryCoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            layer_type='Conv1D_encoder',
            learn_sigma=True,
            hidden_dims=hidden_dims,
            average_max_std=average_max_std,
            **kwargs
        )
    else:
        return GaussianCoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            learn_sigma=True,
            hidden_dims=hidden_dims,
            **kwargs
        )
def get_decoder_model(latent_dim, output_dim, type, average_max_std=1.0, hidden_dims=None, **kwargs):
    if type == "dictionary":
        return DictionaryCoder(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            average_max_std=average_max_std,
            **kwargs
        )
    elif type == "dictionary_cnn":
        return DictionaryCoder(
            input_dim=latent_dim,
            output_dim=output_dim,
            layer_type='Conv1D',
            hidden_dims=hidden_dims,
            average_max_std=average_max_std,
            **kwargs
        )
    elif type == "ART":
        num_bins = kwargs.get("num_bins", 256)
        seq_len = kwargs.get("seq_len", 72)
        d_model = kwargs.get("d_model", 256)
        condition_dim = kwargs.get("condition_dim", 0)
        model_layers = kwargs.get("model_layers", 4)
        num_heads = kwargs.get("num_heads", 8)
        return AutoregressiveTransformerCoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_bins=num_bins,
            seq_len=seq_len,
            d_model=d_model,
            num_layers=model_layers,
            nhead=num_heads
        )
    else:
        return GaussianCoder(
            input_dim=latent_dim,
            output_dim=output_dim,
            learn_sigma=True,
            hidden_dims=hidden_dims,
            **kwargs
        )
