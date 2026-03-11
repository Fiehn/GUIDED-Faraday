from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
 
# =============
# Prior Dsitribution Classes
# =============
class BasePrior(nn.Module, ABC):
    """ Base class for prior distributions. 
    Parameters:
    - latent_dim: Dimension of the latent space
    - device: Device to run the prior on
    """
    def __init__(self, latent_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
    
    @abstractmethod
    def get_params(self):
        """Return dict of prior parameters."""
        ...

    @abstractmethod
    def sample(self,batch_size, num_samples=1):
        """Sample from the prior distribution."""
        ...
    
    @abstractmethod
    def log_prob(self, z):
        """Compute log probability of z under the prior."""
        ...
    
    @abstractmethod
    def forward(self, batch_size):
        """returns params expanded to batch_size."""
        ...

class StandardGaussianPrior(BasePrior):
    """ Standard Normal Prior N(0, I) - non-learnable """
    def __init__(self, latent_dim, device="cpu"):
        super().__init__(latent_dim, device)
        # We register parameters as buffers, this means that they are saved with the model and moved to device, but not updated during training
        self.register_buffer('mu', torch.zeros((1, latent_dim)))
        self.register_buffer('sigma', torch.ones((1, latent_dim)))

    def get_params(self):
        return {"mu": self.mu, "sigma": self.sigma}
    
    def sample(self, batch_size, num_samples=1):
        return torch.randn((num_samples, batch_size, self.latent_dim), device=self.device)
    
    def log_prob(self, z):
        # log prob of standard normal
        return torch.tensor(-0.5 * (z ** 2 + np.log(2 * np.pi)).mean(dim=-1), dtype=z.dtype, device=z.device)
    
    def forward(self, batch_size):
        return {"mu": self.mu.expand(batch_size, -1), "sigma": self.sigma.expand(batch_size, -1)}
    
class LearnableGaussianPrior(BasePrior):
    """ Learnable Gaussian Prior N(mu, sigma) """
    def __init__(self, latent_dim, device="cpu", init_mu = None, init_sigma = None):
        super().__init__(latent_dim, device)
        # Learnable parameters
        if init_mu is None:
            init_mu = torch.zeros((1, latent_dim))
        if init_sigma is None:
            init_sigma = torch.ones((1, latent_dim))

        self.mu = nn.Parameter(init_mu.to(device))
        self.log_sigma = nn.Parameter(torch.log(init_sigma.to(device)))  # We learn log(sigma) for stability

    def get_params(self):
        return {"mu": self.mu, "sigma": torch.exp(self.log_sigma)}
    
    def sample(self, batch_size, num_samples=1):
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn((num_samples, batch_size, self.latent_dim), device=self.device)
        return self.mu + eps * sigma
    
    def log_prob(self, z):
        params = self.get_params()
        mu = params["mu"]
        sigma = params["sigma"]
        return -0.5 * (((z - mu) / sigma) ** 2 + 2 * torch.log(sigma) + torch.log(2 * torch.pi)).sum(dim=-1) # TODO: verify correctness
    
    def forward(self, batch_size):
        sigma = torch.exp(self.log_sigma)
        return {"mu": self.mu.expand(batch_size, -1), "sigma": sigma.expand(batch_size, -1)}
    
class StandardDictionaryPrior(BasePrior):
    """ Standard Normal Prior N(0, I) in dictionary space - non-learnable """
    def __init__(self, latent_dim, dictionary_size, device="cpu"):
        super().__init__(latent_dim, device)
        self.dictionary_size = dictionary_size

        # Stores mu in output space and sigma in dictionary space
        self.register_buffer('mu', torch.zeros((1, latent_dim)))
        self.register_buffer('sigma', torch.ones((1, dictionary_size)))

    def get_params(self):
        return {"mu": self.mu, "sigma": self.sigma}
    
    def sample(self, batch_size, num_samples=1):
        eps = torch.randn(num_samples, batch_size, self.dictionary_size, device=self.device)
        return eps  # Will be mapped by SigmaMapper in encoder
    
    def log_prob(self, z):
        """
        Compute log prob.
        """
        
        # -0.5 * (z^2 + log(2π)) summed over latent dimensions
        log_prob_val = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_prob_val
        #covariance_matrix = torch.diag(self.sigma.squeeze(0) ** 2).to(self.device)
        #return torch.distributions.MultivariateNormal(self.mu.squeeze(0), covariance_matrix).log_prob(z)
    
    def forward(self, batch_size):
        return {"mu": self.mu.expand(batch_size, -1), "sigma": self.sigma.expand(batch_size, -1)}
    
class LearnableDictionaryPrior(BasePrior):
    """ Learnable Gaussian Prior N(mu, sigma) in dictionary space """
    def __init__(self, latent_dim, dictionary_size, device="cpu", init_mu = None, init_sigma = None):
        super().__init__(latent_dim, device)
        self.dictionary_size = dictionary_size

        if init_mu is None:
            init_mu = torch.zeros((1, latent_dim))
        if init_sigma is None:
            init_sigma = torch.ones((1, dictionary_size))

        self.mu = nn.Parameter(init_mu.to(device))
        self.log_sigma = nn.Parameter(torch.log(init_sigma.to(device)))  # We learn log(sigma) for stability

    def get_params(self):
        return {"mu": self.mu, "sigma": torch.exp(self.log_sigma)}
    def sample(self, batch_size, num_samples=1):
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn((num_samples, batch_size, self.dictionary_size), device=self.device)
        # Return samples in dictionary space only (mu in latent space is added after
        # the SigmaMapper transform in CVAE.sample, matching StandardDictionaryPrior behaviour)
        return eps * sigma
    def log_prob(self, z):
        params = self.get_params()
        mu = params["mu"]
        sigma = params["sigma"]
        return None # TODO: To get log prob in output space, we need to map z back to dictionary space. This requires the inverse of the SigmaMapper used in the encoder. Not implemented yet.
    def forward(self, batch_size):
        sigma = torch.exp(self.log_sigma)
        return {"mu": self.mu.expand(batch_size, -1), "sigma": sigma.expand(batch_size, -1)}

class VampPrior(BasePrior):
    """ 
    VampPrior: The prior is a mixture of posteriors generated by 'pseudo-inputs'.
    p(z) = (1/K) * sum_k q(z | u_k)
    where u_k are learnable pseudo-inputs passed through the encoder.

    The encoder reference is stored but NOT owned — it is shared with the parent
    VAE/CVAE model. This breaks the circular dependency: the encoder is created
    first, then VampPrior is created with a reference to it.

    Works with any encoder type (GaussianCoder, DictionaryCoder, etc.) because
    log_prob delegates to ``encoder.loglikelihood()`` which handles the
    encoder's own covariance structure (diagonal, PDCC, etc.).
    """
    supports_analytical_kl = False  # Must use Monte Carlo KL

    def __init__(self, latent_dim, input_shape, encoder=None, num_components=10, device="cpu", **_):
        super().__init__(latent_dim, device)
        self.num_components = num_components
        self.encoder = encoder  # Reference — assigned before use!

        # Pseudo-inputs: learnable "fake" data points u_k
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self._input_shape = input_shape

        # Initialize with small noise to break symmetry
        self.pseudo_inputs = nn.Parameter(torch.randn(num_components, *input_shape) * 0.01)

    def set_encoder(self, encoder, input_shape=None):
        """Re-point the prior to a (possibly new) encoder and optionally
        resize the pseudo-inputs to match a new input dimension."""
        self.encoder = encoder
        if input_shape is not None:
            if isinstance(input_shape, int):
                input_shape = (input_shape,)
            if input_shape != self._input_shape:
                self._input_shape = input_shape
                self.pseudo_inputs = nn.Parameter(
                    torch.randn(self.num_components, *input_shape,
                                device=self.pseudo_inputs.device) * 0.01
                )

    def get_params(self):
        """Return MEAN parameters across all components to work with existing expand() logic.
        This lets VampPrior work with the existing VAE structure."""
        component_params = self.encoder(self.pseudo_inputs)  # [K, ...]
        
        # Return the mean across components - this works with existing expand() logic
        return {k: v.mean(dim=0, keepdim=True) for k, v in component_params.items()}

    def sample(self, batch_size, num_samples=1):
        """Sample by randomly selecting components then using encoder's reparameterize."""
        component_params = self.encoder(self.pseudo_inputs)  # [K, ...]

        # Randomly pick components for each sample
        indices = torch.randint(0, self.num_components, (num_samples, batch_size),
                                device=self.pseudo_inputs.device)

        samples_list = []
        for s in range(num_samples):
            # Get params for selected components: [B, ...]
            params_selected = {k: v[indices[s]] for k, v in component_params.items()}
            # Sample using encoder's reparameterize: [1, B, D]
            z = self.encoder.reparameterize(params_selected, num_samples=1)
            samples_list.append(z)
        
        return torch.cat(samples_list, dim=0)  # [S, B, D]

    def log_prob(self, z):
        """Compute log p(z) = log( (1/K) sum_k q(z | u_k) )."""
        component_params = self.encoder(self.pseudo_inputs)  # [K, ...]

        K = self.num_components
        B, D = z.shape

        # Expand z to [B*K, D] — repeat each z for every component
        z_expanded = z.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

        # Expand component params to [B*K, ...]
        params_expanded = {}
        for key, val in component_params.items():
            rest = val.shape[1:]
            params_expanded[key] = (
                val.unsqueeze(0)               # [1, K, *rest]
                  .expand(B, *val.shape)        # [B, K, *rest]
                  .reshape(B * K, *rest)         # [B*K, *rest]
            )

        # log q(z | u_k) for every (batch, component) pair: [B*K]
        log_probs = self.encoder.loglikelihood(z_expanded, params_expanded)
        log_probs = log_probs.reshape(B, K)  # [B, K]

        # Mixture: log (1/K) sum_k exp(log_prob_k) = logsumexp - log(K)
        return torch.logsumexp(log_probs, dim=1) - np.log(K)
    
    def forward(self, batch_size):
        """Return mean parameters expanded to batch_size for compatibility."""
        params = self.get_params()
        return {k: v.expand(batch_size, -1) for k, v in params.items()}

def get_prior(prior_type, latent_dim, device="cpu", **kwargs):
    """ Function to get priors based on type string 
    Parameters:
    - prior_type: Type of prior ("standard_gaussian", "learnable_gaussian", "standard_dictionary", "learnable_dictionary", "vampprior")
    - latent_dim: Dimension of the latent space
    - device: Device to run the prior on
    - kwargs: Additional arguments for prior initialization
      For vampprior: input_shape, encoder, num_components
    """
    if prior_type == "standard_gaussian":
        return StandardGaussianPrior(latent_dim, device)
    elif prior_type == "learnable_gaussian":
        gaussian_kwargs = {k: kwargs[k] for k in ("init_mu", "init_sigma") if k in kwargs}
        return LearnableGaussianPrior(latent_dim, device, **gaussian_kwargs)
    elif prior_type == "standard_dictionary":
        dictionary_size = kwargs.get("dictionary_size", 100)
        return StandardDictionaryPrior(latent_dim, dictionary_size, device)
    elif prior_type == "learnable_dictionary":
        dictionary_size = kwargs.get("dictionary_size", 100)
        dict_kwargs = {k: kwargs[k] for k in ("init_mu", "init_sigma") if k in kwargs}
        return LearnableDictionaryPrior(latent_dim, dictionary_size, device, **dict_kwargs)
    elif prior_type == "vampprior":
        return VampPrior(
            latent_dim=latent_dim,
            input_shape=kwargs.get("input_shape", latent_dim),
            encoder=kwargs.get("encoder", None),
            num_components=kwargs.get("num_components", 20),
            device=device,
        )
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

