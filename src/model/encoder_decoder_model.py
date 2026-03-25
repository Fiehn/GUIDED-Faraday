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
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm)

    def _num_parameters(self):
        return self.parameterizer._num_parameters()
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["mu"] = torch.clamp(param_dict["mu"], min=0.0, max=12.0)
        if self.learn_sigma:
            param_dict["sigma"] = F.softplus(param_dict["sigma"]) + 1e-6 # Ensure positivity 
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
                 dropout=True,
                 dropout_rate=0.2,
                 batch_norm=False,
                 **_):
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

        self.parameterizer = Parameterizer(
            input_dim=input_dim,
            output_dim={"sigma": dictionary_size, "mu": output_dim},
            dist_parameters=["mu", "sigma"],
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm)
        
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
    
    def forward(self, inputs, test=True):
        if test:
            param_dict = self.parameterizer(inputs)
            
            # 1. Smoothly bound the Mean (mu) between -5.0 and +5.0
            # No dead gradients, no infinite predictions.
            param_dict["mu"] = torch.tanh(param_dict["mu"]) * 5.0 

            # 2. Smoothly bound the Variance (sigma) 
            # Sigmoid maps to [0, 1]. This maps sigma smoothly between lower_lim and ~2.0
            # No softplus exploding into infinity.
            max_sigma_range = 2.0 - self.sigma_lower_lim
            param_dict["sigma"] = self.sigma_lower_lim + (torch.sigmoid(param_dict["sigma"]) * max_sigma_range)
            
            return param_dict
        
        param_dict = self.parameterizer(inputs)
        #param_dict["sigma"] = torch.nn.functional.relu(param_dict["sigma"]).clamp(self.sigma_lower_lim, self.sigma_upper_lim)
        param_dict["mu"] = torch.clamp(param_dict["mu"], min=0.0, max=9.0)

        # The actual max allowed out of the clamp is (upper - lower)
        max_allowed_softplus = self.sigma_upper_lim - self.sigma_lower_lim
        # The upper limit is kinda not nessecary except as a safety precaution against collapse
        # The lower limit is implemented here by a constant added to the softplus output, which ensures that sigma is always above the lower limit (since softplus is always positive)
        param_dict["sigma"] = F.softplus(param_dict["sigma"]).clamp(max=max_allowed_softplus) + self.sigma_lower_lim
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
        """Sample from the distribution. If return_mean is True, ignores noise."""
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
        L = torch.linalg.cholesky(Sigma)
        
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
                log_p = prior.log_prob(z)
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
    
# === This is the actual network structure ===
class Parameterizer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dist_parameters,
                 num_hidden_layers=2,
                 num_neurons=128,
                 dropout=False,
                 dropout_rate=0.2,
                 batch_norm=False,
                 resnets=True,
                 block_type='MLP',
                 **_):
        super(Parameterizer, self).__init__()
        self.dist_parameters = dist_parameters
        self.num_neurons = num_neurons
        
        # Shared block using NNBlock (outputs to num_neurons)
        self.shared_block = NNBlock(
            input_dim=input_dim,
            output_dim=num_neurons,
            num_neurons=num_neurons,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            num_hidden_layers=num_hidden_layers,
            resnet=resnets,
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

class NNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=128, dropout=False, 
                 dropout_rate=0.2, batch_norm=False,num_hidden_layers=2, 
                 resnet=True, type='MLP', **_):
        super(NNBlock, self).__init__()
        self.resnet = resnet
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.num_hidden_layers = num_hidden_layers
        self.layer_type = type

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
        if self.layer_type == 'MLP':
            layers = [nn.Linear(self.input_dim, self.num_neurons), nn.ReLU()]
        elif self.layer_type == 'Conv1D':
            layers = [nn.Conv1d(self.input_dim, self.num_neurons, kernel_size=3, padding=1), nn.ReLU()]
        elif self.layer_type == 'LSTM':
            layers = [nn.LSTM(self.input_dim, self.num_neurons, batch_first=True)]
        elif self.layer_type == 'Transformer':
            layers = [nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4, dim_feedforward=self.num_neurons), nn.ReLU()]
        else:
            layers = [nn.Linear(self.input_dim, self.num_neurons), nn.ReLU()]

        if self.batch_norm:
            layers.append(nn.LayerNorm(self.num_neurons))
        if self.dropout:
            layers.append(nn.Dropout(self.dropout_rate * 0.5))  # Less dropout in the first layer
        return nn.Sequential(*layers)
    def _build_hidden_layer(self):
        layers = []
        for _ in range(self.num_hidden_layers):
            if self.layer_type == 'MLP':
                layers.append(nn.Linear(self.num_neurons, self.num_neurons))
                layers.append(nn.ReLU())
            elif self.layer_type == 'Conv1D':
                layers.append(nn.Conv1d(self.num_neurons, self.num_neurons, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            elif self.layer_type == 'LSTM':
                layers.append(nn.LSTM(self.num_neurons, self.num_neurons, batch_first=True))
            elif self.layer_type == 'Transformer':
                layers.append(nn.TransformerEncoderLayer(d_model=self.num_neurons, nhead=4, dim_feedforward=self.num_neurons))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.num_neurons, self.num_neurons))
                layers.append(nn.ReLU())

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.num_neurons))
            if self.dropout:
                layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)
    def _build_output_layer(self):
        if self.layer_type == 'MLP':
            layers = [nn.Linear(self.num_neurons, self.output_dim)] 
        elif self.layer_type == 'Conv1D':
            layers = [nn.Conv1d(self.num_neurons, self.output_dim, kernel_size=3, padding=1)]
        elif self.layer_type == 'LSTM':
            layers = [nn.LSTM(self.num_neurons, self.output_dim, batch_first=True)]
        elif self.layer_type == 'Transformer':
            layers = [nn.TransformerEncoderLayer(d_model=self.num_neurons, nhead=4, dim_feedforward=self.num_neurons), nn.ReLU(), nn.Linear(self.num_neurons, self.output_dim)]
        else:
            layers = [nn.Linear(self.num_neurons, self.output_dim)] 
        return nn.Sequential(*layers)
    
    
def get_encoder_model(input_dim, latent_dim, type, **kwargs):
    if type == "dictionary":
        return DictionaryCoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            **kwargs
        )
    else:
        return GaussianCoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            learn_sigma=True,
            **kwargs
        )
def get_decoder_model(latent_dim, output_dim, type, **kwargs):
    if type == "dictionary":
        return DictionaryCoder(
            input_dim=latent_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        return GaussianCoder(
            input_dim=latent_dim,
            output_dim=output_dim,
            learn_sigma=True,
            **kwargs
        )
