import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# https://arxiv.org/pdf/2411.03936
# GuideVAE model implementation
# Conditional VAE with conditioning on user profiles
# and a parameterized covariance matrix (PDCC)

# First a VAE model
class VAE(nn.Module):
    def __init__(self):
        # Initialize the GuideVAE model
        super(VAE, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Output mean and logvar for latent space
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(8 + 6, 32),  # Latent vector + user
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Reconstruct input
        )

    def encode(self, x):
        # Encode input to latent space
        h = self.encoder(x)
        mu, logvar = h[:, :8], h[:, 8:]  # Split into mean and logvar
        return mu, logvar
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, user):
        # HERE we condition on user profiles

        z_cond = torch.cat([z, user], dim=1)
        h = self.decoder(z_cond) # Decode to reconstruct input
        mu_x = h[:, :10] # Reconstructed mean (with input size 10)
        logvar_x = h[:, 10:] # Reconstructed logvar (with input size 10)
        return mu_x, logvar_x
    def forward(self, x, user):
        # Forward pass through the
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z, user)
        return recon_mu, recon_logvar, mu, logvar
    
    def loss_function(self, recon_mu, recon_logvar, x, mu, logvar):
        # HERE we implement the loss function as described in the GuideVAE paper

        # Explicit Gaussian likelihood: p(x|z,user) = N(x; recon_mu, diag(exp(recon_logvar)))
        recon_loss = 0.5 * torch.sum(
            recon_logvar + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
        )
        # KL divergence: KL[q(z|x) || p(z)] 
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
    

if __name__ == "__main__":
    # Test the VAE model
    model = VAE()
    x = torch.randn(4, 10)  # Batch of 4 samples, input size 10
    user = torch.randn(4, 6)  # Batch of 4 samples, user profile size 6
    recon_x, mu, logvar = model(x, user)
    loss = model.loss_function(recon_x, x, mu, logvar)
    print("Loss:", loss)