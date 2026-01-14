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
            nn.Linear(24, 64),
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
            nn.Linear(64, 48)  # Reconstruct input
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
        mu_x = h[:, :24] # Reconstructed mean (with input size 24)
        logvar_x = h[:, 24:] # Reconstructed logvar (with input size 24)
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
    from dataloader import RawData
    import matplotlib.pyplot as plt

    # load data
    data_dir = "data"
    filename = "hourly_measurements_2023_12.csv"
    raw_data_loader = RawData(data_dir)
    data = raw_data_loader.get_data(filename)
    print("Data loaded and preprocessed successfully.")
    
    # Get sample batches of 24 hours for a random meters
    unique_meters = raw_data_loader.get_all_meter_numbers()
    sample_meter_number = unique_meters[0]
    meter_data = raw_data_loader.get_data_by_meter(sample_meter_number)
    sample_data = meter_data.head(24)
    print(f"Data for meter number {sample_meter_number}:")

    consumption = sample_data['hourly_energy_consumption'].values

    # create sliding windows of 24 hours
    sequences = []
    for i in range(len(consumption) - 23):
        seq = consumption[i:i+24]
        sequences.append(seq)

    # convert to tensor
    sequences = torch.tensor(sequences, dtype=torch.float32)
    print(f"Sample sequences shape: {sequences.shape}")

    user_profiles = []
    for i in range(sequences.size(0)):
        user_profiles.append([0.5, 1.0, 0.0, 0.0, 1.0, 0.0]) # dummy user profiles
    user_profiles = torch.tensor(user_profiles, dtype=torch.float32)
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    num_epochs = 150
    batch_size = 32
    losses = []
    for epoch in range(num_epochs):
        perm = torch.randperm(sequences.size(0))
        epoch_loss = 0
        for i in range(0, sequences.size(0), batch_size):
            indices = perm[i:i+batch_size]
            batch_x = sequences[indices]
            batch_user = user_profiles[indices]

            optimizer.zero_grad()
            recon_mu, recon_logvar, mu, logvar = model(batch_x, batch_user)
            loss = model.loss_function(recon_mu, recon_logvar, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    
    # Plot training loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # test reconstruction
    model.eval()
    with torch.no_grad():
        test_idx = 0
        test_x = sequences[test_idx:test_idx+1]
        test_user = user_profiles[test_idx:test_idx+1]
        recon_mu, recon_logvar, _, _ = model(test_x, test_user)

    plt.plot(test_x.numpy().flatten(), label='Original')
    plt.plot(recon_mu.numpy().flatten(), label='Reconstructed')
    plt.fill_between(
        np.arange(recon_mu.shape[1]),
        (recon_mu - 1.96 * torch.exp(0.5 * recon_logvar)).numpy().flatten(),
        (recon_mu + 1.96 * torch.exp(0.5 * recon_logvar)).numpy().flatten(),
        color='gray', alpha=0.5, label='95% Confidence Interval'
    )
    plt.title("VAE Reconstruction with 95% Confidence Interval")
    plt.legend()
    plt.show()

