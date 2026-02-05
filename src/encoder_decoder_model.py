import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from src.priors import *

#zero preserving log transform!!! TODO: add this later

# Will be a standard Gaussian encoder/decoder model.
# GuideVAE style model will come later.
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
        param_dict["mu"] = param_dict["mu"] # clamp with upper and lower bounds?
        if self.learn_sigma:
            param_dict["sigma"] = F.softplus(param_dict["sigma"]) + 1e-6 # Ensure positivity 
        else:
            param_dict["sigma"] = torch.ones_like(param_dict["mu"]) * self.sigma_fixed # Fixed sigma
        return param_dict
    
    def sample(self, param_dict=None, num_samples=1,**_):
        """ Sampple from the Gaussian defined by param_dict using reparameterization trick """
        if param_dict is None:
            raise ValueError("param_dict must be provided for sampling")
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
        Compute KL divergence KL(q||p) using log probabilities.
        
        Args:
            param_dict_q: Posterior parameters
            prior: BasePrior object (if None, uses standard normal)
        """
        # KL(q||p) = 0.5 * sum(sigma_p^2/sigma_q^2 + (mu_q - mu_p)^2/sigma_q^2 - 1 + log(sigma_q^2/sigma_p^2)) TODO: Check back in later
        mu_q = param_dict_q["mu"]
        sigma_q = param_dict_q["sigma"]
        
        if prior is None:
            # KL divergence with standard normal N(0,I)
            mu_p = torch.zeros_like(mu_q)
            sigma_p = torch.ones_like(sigma_q)
        else:
            # Get prior parameters
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
    
    def reparameterize(self, param_dict, num_samples=1, **_):
        # Reparameterization trick: sample epsilon from N(0,1) and return mu + sigma * epsilon
        mu = param_dict["mu"]
        sigma = param_dict["sigma"]
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
                 marginal_std_lim=0.01,
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
        self.sigma_lower_lim = sigma_lim
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
    
    def forward(self, inputs):
        param_dict = self.parameterizer(inputs)
        param_dict["sigma"] = torch.nn.functional.relu(param_dict["sigma"]).clamp(self.sigma_lower_lim, self.sigma_upper_lim)
        # TODO: They clamp mu as well?? Check later
        return param_dict

    def reparameterize(self, param_dict, num_samples=1, **_):
        """Reparameterization trick for sampling from the multivariate Gaussian defined by param_dict"""
        mu, sigma = param_dict["mu"], param_dict["sigma"]
        batch_size = mu.shape[0]

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
    
    def sample(self, param_dict=None, num_samples=1, **_):
        """Sample from the dictionary-based distribution."""
        if param_dict is None:
            raise ValueError("param_dict must be provided for sampling")
        return self.reparameterize(param_dict, num_samples=num_samples)
    
    def loglikelihood(self, inputs, param_dict, **_):
        """Compute log likelihood under multivariate Gaussian with PDCC covariance."""
        mu = param_dict["mu"]
        Sigma = self.create_covariance_matrix(param_dict)
        # Use PyTorch's MultivariateNormal distribution
        try:
            dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
            return dist.log_prob(inputs)
        except Exception as e:
            # Fallback if covariance is not positive definite (IT SHOULD BE by the papers arguments)
            print(f"Warning: Covariance matrix issue in loglikelihood: {e}")
            # Add the marginal std lim to diagonal for stability again
            Sigma_stable = Sigma + torch.eye(self.output_dim, device=Sigma.device) * (self.marginal_std_lim ** 2)
            dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma_stable)
            return dist.log_prob(inputs)
    
    def kl_divergence(self, param_dict_q, prior=None, **_):
        """
        Compute KL divergence KL(q||p) using the prior's log_prob method.
        For multivariate Gaussians, we can use the analytical form or Monte Carlo.
        
        Args:
            param_dict_q: Posterior parameters
            prior: BasePrior object
        """
        mu_q = param_dict_q["mu"]
        Sigma_q = self.create_covariance_matrix(param_dict_q)
        # TODO: Check back in here, the prior assumption is a bit odd
        if prior is None:
            # Default to standard normal in dictionary space
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
            # Monte Carlo estimation: KL ≈ log q(z) - log p(z)
            z = self.reparameterize(param_dict_q, num_samples=1).squeeze(0)  # [B, D]
            
            # log q(z|x)
            dist_q = torch.distributions.MultivariateNormal(
                loc=mu_q, 
                covariance_matrix=Sigma_q + torch.eye(self.output_dim, device=Sigma_q.device) * (self.marginal_std_lim ** 2)
            )
            log_q = dist_q.log_prob(z)
            
            # log p(z) - use prior's log_prob method
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
                 num_neurons=64,
                 dropout=True,
                 dropout_rate=0.2,
                 batch_norm=False,
                 **_):
        super(Parameterizer, self).__init__()
        self.dist_parameters = dist_parameters
        
        # Build shared block - simplified to avoid deep nesting
        layers = []
        layers.append(nn.Linear(input_dim, num_neurons))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_neurons))
        if dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_neurons))
            if dropout:
                layers.append(nn.Dropout(dropout_rate))
        
        self.shared_block = nn.Sequential(*layers)
        
        # Build parameter-specific heads
        self.param_heads = nn.ModuleDict()
        
        if isinstance(output_dim, int):
            for param in dist_parameters:
                head_layers = []
                head_layers.append(nn.Linear(num_neurons, num_neurons))
                head_layers.append(nn.ReLU())
                if batch_norm:
                    head_layers.append(nn.BatchNorm1d(num_neurons))
                if dropout:
                    head_layers.append(nn.Dropout(dropout_rate))
                head_layers.append(nn.Linear(num_neurons, output_dim))
                self.param_heads[param] = nn.Sequential(*head_layers)
        elif isinstance(output_dim, dict):
            for key in dist_parameters:
                if key in output_dim:
                    head_layers = []
                    head_layers.append(nn.Linear(num_neurons, num_neurons))
                    head_layers.append(nn.ReLU())
                    if batch_norm:
                        head_layers.append(nn.BatchNorm1d(num_neurons))
                    if dropout:
                        head_layers.append(nn.Dropout(dropout_rate))
                    head_layers.append(nn.Linear(num_neurons, output_dim[key]))
                    self.param_heads[key] = nn.Sequential(*head_layers)
        
    def forward(self, x):
        h = self.shared_block(x)
        param_dict = {}
        for param in self.dist_parameters:
            param_dict[param] = self.param_heads[param](h)
        return param_dict
        
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    
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
