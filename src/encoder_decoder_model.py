import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

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

        # Do we learn the variance?
        if self.learn_sigma:
            # Output both mean and sigma
            dist_parameters = ["mu", "sigma"]
        else:
            # Output only mean
            dist_parameters = ["mu"]

        self.parameterizer = Parameterizer(
            input_dim=input_dim,
            output_dim=output_dim,
            dist_parameters=dist_parameters,
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
        # A helper function to sample from the Gaussian defined by param_dict using reparameterization trick
        # calls reparameterize function
        return self.reparameterize(param_dict, num_samples=num_samples)

    def loglikelihood(self, inputs, param_dict):
        # Compute log likelihood of inputs under the Gaussian defined by param_dict
        # TODO: I think this is where we might do multivariate Gaussian later??
        mu = param_dict["mu"]
        sigma = param_dict["sigma"]
        var = sigma ** 2
        log_likelihood = -0.5 * torch.sum(((inputs - mu) ** 2) / var + torch.log(2 * torch.pi * var), dim=1)
        return log_likelihood

    def kl_divergence(self, param_dict_q, param_dict_p=None, **_):
        # Compute KL divergence between two Gaussians
        return kl_divergence(param_dict_q, param_dict_p) # TODO: implement in utils.py
    
    def create_covariance_matrix(self, param_dict):
        # Here in the Gaussian case, we create a diagonal covariance matrix from the sigma parameters
        return torch.eye(param_dict["mu"].shape[1]).to(param_dict["mu"].device) * (param_dict["sigma"] ** 2).unsqueeze(-1)
    
    def reparameterize(self, param_dict, num_samples=1):
        # Reparameterization trick: sample epsilon from N(0,1) and return mu + sigma * epsilon
        mu = param_dict["mu"]
        sigma = param_dict["sigma"]
        batch_size, output_dim = mu.shape
        # Sample epsilon
        epsilon = torch.randn((num_samples, batch_size, output_dim)).to(mu.device)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon
        return samples
    
# === This is a basic NN block used in the Parameterizer ===
class NNBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_neurons=64,
                 num_hidden_layers=2,
                 dropout=True,
                dropout_rate=0.2,
                batch_norm=False,
                **_):
        super(NNBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        
        # === Here we build the layers of the block ===
        # The input layer will sequentially pass through the following: Linear, ReLU, (BatchNorm), (Dropout)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU()
        )
        if batch_norm:
            self.input_layer.add_module("batch_norm", nn.BatchNorm1d(num_neurons))
        if dropout:
            self.input_layer.add_module("dropout", nn.Dropout(dropout_rate))
        # Hidden layers is a module list of num_hidden_layers layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(num_neurons, num_neurons),
                nn.ReLU()
            )
            if batch_norm:
                layer.add_module("batch_norm", nn.BatchNorm1d(num_neurons))
            if dropout:
                layer.add_module("dropout", nn.Dropout(dropout_rate))
            self.hidden_layers.append(layer)
        # Output layer is a simple linear layer to output_dim
        self.output_layer = nn.Linear(num_neurons, output_dim)
    def forward(self, x):
        # Pass through the layers
        # view will flatten all dimensions except batch
        h = x.view(-1, self.input_dim)
        h = self.input_layer(h)
        for layer in self.hidden_layers:
            h = layer(h)
        h = self.output_layer(h)
        return h
    def _num_parameters(self):
        # honestly is ai, i dont know what numel is? TODO: look up later
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === This is the actual network structure ===
class Parameterizer(nn.Module):
    def __init__(self,
                 intput_dim,
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
        
        # We build the network from blocks of NN layers
        # We have a shared block followed by separate blocks for each distribution parameter
        # We dont want to cross the wires between parameters
        # So we use a ModuleDict to hold the blocks
        self.block_dict = nn.ModuleDict()
        self.block_dict["shared_block"] = NNBlock(
            input_dim=intput_dim,
            output_dim=num_neurons,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons,
            dropout=dropout,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
        for param in dist_parameters:
            self.block_dict[f"{param}_block"] = NNBlock(
                input_dim=num_neurons,
                output_dim=output_dim,
                num_hidden_layers=num_hidden_layers,
                num_neurons=num_neurons,
                dropout=dropout,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm
            )
    def forward(self, x):
        h = x.view(-1, self.block_dict["shared_block"].input_dim)
        h = self.block_dict["shared_block"](h)
        param_dict = {}
        for param in self.dist_parameters:
            param_dict[param] = self.block_dict[f"{param}_block"](h)
        return param_dict
    def _num_parameters(self):
        # TODO: look up what numel is?
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def get_encoder_model(input_dim, latent_dim, **kwargs):
    return GaussianCoder(
        input_dim=input_dim,
        output_dim=latent_dim,
        learn_sigma=True,
        **kwargs
    )
def get_decoder_model(latent_dim, output_dim, **kwargs):
    return GaussianCoder(
        input_dim=latent_dim,
        output_dim=output_dim,
        learn_sigma=True,
        **kwargs
    )
def get_prior_params(latent_dim, device, **kwargs):
    # Standard normal prior N(0, I)
    mu = torch.zeros((1, latent_dim)).to(device)
    sigma = torch.ones((1, latent_dim)).to(device)
    return {"mu": mu, "sigma": sigma}
