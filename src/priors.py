import torch
import torch.nn as nn
import numpy as np

# =============
# Prior Dsitribution Classes
# =============
class BasePrior(nn.Module):
    """ Base class for prior distributions. 
    Parameters:
    - latent_dim: Dimension of the latent space
    - device: Device to run the prior on
    """
    def __init__(self, latent_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
    
    def get_params(self):
        """Return dict of prior parameters."""
        raise NotImplementedError

    def sample(self,batch_size, num_samples=1):
        """Sample from the prior distribution."""
        raise NotImplementedError
    
    def log_prob(self, z):
        """Compute log probability of z under the prior."""
        raise NotImplementedError
    
    def forward(self, batch_size):
        """returns params expanded to batch_size."""
        raise NotImplementedError

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
        return torch.randn((batch_size, num_samples, self.latent_dim), device=self.device)
    
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
        eps = torch.randn((batch_size, num_samples, self.latent_dim), device=self.device)
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
        Compute log prob in OUTPUT space.
        """
        
        return torch.tensor(-0.5 * (z ** 2 + np.log(2 * np.pi)).mean(dim=-1), dtype=z.dtype, device=z.device)
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
        eps = torch.randn((batch_size, num_samples, self.dictionary_size), device=self.device)
        return self.mu + eps * sigma
    def log_prob(self, z):
        params = self.get_params()
        mu = params["mu"]
        sigma = params["sigma"]
        return None # TODO: To get log prob in output space, we need to map z back to dictionary space. This requires the inverse of the SigmaMapper used in the encoder. Not implemented yet.
    def forward(self, batch_size):
        sigma = torch.exp(self.log_sigma)
        return {"mu": self.mu.expand(batch_size, -1), "sigma": sigma.expand(batch_size, -1)}
    
def get_prior(prior_type, latent_dim, device="cpu", **kwargs):
    """ Function to get priors based on type string 
    Parameters:
    - prior_type: Type of prior ("standard_gaussian", "learnable_gaussian", "standard_dictionary", "learnable_dictionary")
    - latent_dim: Dimension of the latent space
    - device: Device to run the prior on
    - kwargs: Additional arguments for prior initialization
    """
    if prior_type == "standard_gaussian":
        return StandardGaussianPrior(latent_dim, device)
    elif prior_type == "learnable_gaussian":
        return LearnableGaussianPrior(latent_dim, device, **kwargs)
    elif prior_type == "standard_dictionary":
        dictionary_size = kwargs.get("dictionary_size", 100)
        return StandardDictionaryPrior(latent_dim, dictionary_size, device)
    elif prior_type == "learnable_dictionary":
        dictionary_size = kwargs.get("dictionary_size", 100)
        return LearnableDictionaryPrior(latent_dim, dictionary_size, device, **kwargs)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

