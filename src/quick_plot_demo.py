"""
Quick demo: Test all visualization functions from train_utils.py with a mock model.
This lets you test each visualization function quickly without loading the real checkpoint.

Tests ONLY the 8 visualization functions from train_utils:
1. _visualize_reconstructions
2. _visualize_synthetic_samples
3. _visualize_sigma_mapper
4. _visualize_latent_space
5. _visualize_synthetic_recreations
6. _visualize_user_dictionary
7. _save_model_summary
8. visualize_trained_model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data.normalizer import Normalizer
from Data.Conditioner import Conditioner, DayOfWeekConditioner, UserProfiler

from Visualization.Visualizer import Visualizer
from Visualization.Writer import Writer
from Visualization.PlotStyling import PlotStyle

# Create output directory
output_dir = "Output/Plots/quick_demo"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# MOCK MODEL SETUP
# ============================================================================
class MockEncoder(nn.Module):
    """Mock encoder that outputs latent distribution parameters."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc = nn.Linear(input_dim, latent_dim * 2)
    
    def forward(self, x):
        """Returns dict with 'mu' and 'sigma' keys."""
        out = self.fc(x)
        mu = out[:, :self.latent_dim]
        log_sigma = out[:, self.latent_dim:]
        return {"mu": mu, "log_sigma": log_sigma}
    
    def reparameterize(self, params, num_samples=1):
        """Reparameterization trick."""
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        sigma = torch.exp(log_sigma * 0.5)
        eps = torch.randn(num_samples, mu.shape[0], mu.shape[1], device=mu.device)
        z = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [num_samples, B, latent_dim]
        return z


class MockDecoder(nn.Module):
    """Mock decoder that outputs reconstruction parameters."""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(latent_dim, output_dim * 2)
    
    def forward(self, z):
        """z shape: [B, latent_dim] or [S, B, latent_dim]."""
        original_shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)
        out = self.fc(z_flat)
        mu = out[:, :self.output_dim]
        log_sigma = out[:, self.output_dim:]
        mu = mu.reshape(*original_shape[:-1], self.output_dim)
        log_sigma = log_sigma.reshape(*original_shape[:-1], self.output_dim)
        return {"mu": mu, "log_sigma": log_sigma}
    
    def sample(self, params, num_samples=1):
        """Sample from the distribution."""
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        sigma = torch.exp(log_sigma * 0.5)
        # mu shape: [B, output_dim], add num_samples dimension
        if mu.dim() == 2:
            mu = mu.unsqueeze(0).expand(num_samples, -1, -1)
            sigma = sigma.unsqueeze(0).expand(num_samples, -1, -1)
        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        return samples  # [num_samples, B, output_dim]


class MockPrior(nn.Module):
    """Mock prior (standard Gaussian)."""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
    
    def forward(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim, device=device)


class MockVAE(nn.Module):
    """Mock VAE for testing."""
    def __init__(self, input_dim=24, latent_dim=32, condition_dim=0, device="cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.device = device
        
        encoder_in = input_dim + condition_dim
        self.encoder = MockEncoder(encoder_in, latent_dim)
        decoder_in = latent_dim + condition_dim
        self.decoder = MockDecoder(decoder_in, input_dim)
        self.prior = MockPrior(latent_dim)
        
        # For CVAE: add conditioner
        if condition_dim > 0:
            self.conditioner = Conditioner()
            self.conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            
            # Create and pre-populate UserProfiler to avoid fitting errors
            user_profiler = UserProfiler(num_clusters=5, verbose=False)
            # Pre-populate cache with fake user profiles for wide range of meter IDs
            for meter_id in range(1, 101):  # 100 fake users to cover mock dataloader range
                # Random topic distribution
                profile = np.random.dirichlet(np.ones(5))  # 5 topics
                user_profiler.user_profiles_cache[meter_id] = profile
                # Also add gamma cache (shape and scale parameters for topic distribution)
                user_profiler.gamma_cache[meter_id] = (np.random.rand(5) + 1, np.random.rand(5) + 1)
            
            # Mark as fitted and set all required attributes to avoid disk loading
            user_profiler.is_fitted = True
            user_profiler.n_topics = 5
            user_profiler.cluster_centers_ = np.random.randn(5, 24)  # Fake cluster centers [n_topics, features]
            user_profiler.components_ = np.random.rand(5, 24) + 0.1  # NMF/LDA components (must be > 0)
            user_profiler.exp_dirichlet_component_ = np.random.rand(5, 24) + 1.0  # LDA exp(Dirichlet) (must be > 0)
            user_profiler.doc_topic_prior_ = 0.1  # LDA document-topic prior (alpha)
            user_profiler.topic_word_prior_ = 0.1  # LDA topic-word prior (beta)
            
            self.conditioner.add_condition_item("user_profile", user_profiler)
    
    def forward(self, x_flat, c=None, num_samples=1):
        """Forward pass."""
        if c is not None:
            enc_in = torch.cat([x_flat, c], dim=-1)
        else:
            enc_in = x_flat
        
        # Encode
        params = self.encoder(enc_in)
        z = self.encoder.reparameterize(params, num_samples=num_samples)  # [num_samples, B, latent_dim]
        
        # Decode
        if c is not None:
            c_exp = c.unsqueeze(0).expand(num_samples, -1, -1)
            z_with_c = torch.cat([z, c_exp], dim=-1)
        else:
            z_with_c = z
        
        recon_params = self.decoder(z_with_c)
        return recon_params, z
    
    def sample(self, conditions=None, num_prior_samples=None, num_x_samples=None, 
               num_samples_prior=None, num_samples_posterior=None, device=None):
        """Sample from model. Supports both parameter naming conventions."""
        # Handle both naming conventions
        if num_samples_prior is not None:
            num_prior_samples = num_samples_prior
        if num_samples_posterior is not None:
            num_x_samples = num_samples_posterior
        
        # Defaults
        num_prior_samples = num_prior_samples or 1
        num_x_samples = num_x_samples or 1
        
        if conditions is None:
            batch_size = num_prior_samples  # For VAE without conditions
        else:
            batch_size = conditions.shape[0]
        
        # Sample from prior multiple times
        all_samples = []
        for _ in range(num_prior_samples):
            z = self.prior(batch_size if conditions is not None else 1, device=self.device)
            
            # Decode with conditions
            if conditions is not None:
                z_with_c = torch.cat([z, conditions], dim=-1)
            else:
                z_with_c = z
            
            recon_params = self.decoder(z_with_c)
            samples = self.decoder.sample(recon_params, num_samples=num_x_samples)  # [num_x_samples, B, input_dim]
            all_samples.append(samples)
        
        # Stack all prior samples: [num_prior_samples, num_x_samples, B, input_dim]
        all_samples = torch.stack(all_samples, dim=0)
        
        # Return format depends on whether we have conditions
        if conditions is not None:
            return all_samples  # [num_prior_samples, num_x_samples, B, input_dim]
        else:
            # For VAE: return as dict with "samples" key
            # Shape should be [num_prior_samples, num_x_samples, num_samples, input_dim]
            return {"samples": all_samples}  # [num_prior_samples, num_x_samples, B, input_dim]
    
    def reconstruct(self, x_flat, meter_ids=None, timestamps=None, num_samples=1):
        """Reconstruct input."""
        if hasattr(self, 'conditioner'):
            c = self.conditioner.get_conditions(x_flat, meter_id=meter_ids, timestamps=timestamps)
            c = self.move_to_device(c, device=self.device)
        else:
            c = None
        
        return self.forward(x_flat, c=c, num_samples=num_samples)
    
    def move_to_device(self, obj, device):
        """Move object to device (helper for conditioning)."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_device(item, device) for item in obj]
        return obj
    
    def to(self, device):
        super().to(device)
        self.device = str(device)
        return self
    
    def get_SigmaMapper(self):
        """Optional method for decoders with sigma mapper. Returns fake matrix for testing."""
        # Return a fake sigma mapper matrix for testing purposes
        return torch.randn(self.input_dim, self.input_dim) * 0.1  # Small random matrix


class MockDataLoader:
    """Mock dataloader that yields batches."""
    def __init__(self, batch_size=4, num_batches=3, input_dim=24, device="cpu", with_timestamps=False):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_dim = input_dim
        self.device = device
        self.with_timestamps = with_timestamps
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate random batch
            x = torch.randn(self.batch_size, 24, 1, device=self.device) * 0.5 + 0.5
            meter_id = torch.randint(1, 100, (self.batch_size,), device=self.device)
            
            if self.with_timestamps:
                # Realistic timestamps (Unix timestamps)
                base_ts = 1700000000  # October 2023
                timestamps = torch.tensor(
                    [base_ts + i * 86400 for i in range(self.batch_size)],
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                timestamps = torch.zeros(self.batch_size, device=self.device)
            
            cond_idx = torch.zeros(self.batch_size, device=self.device)
            yield x, meter_id, timestamps, cond_idx


# Generate fake realistic energy data
np.random.seed(42)
B = 4  # batch size
L = 24  # window length (24 hours)
F = 1  # features (just consumption)

# Create realistic-looking consumption data (daily pattern)
consumption = []
for b in range(B):
    hour_pattern = np.sin(np.arange(L) * np.pi / 12) * 0.5 + 0.5  # Daily cycle
    noise = np.random.normal(0, 0.05, L)
    cons = np.maximum(hour_pattern + noise, 0) * (0.5 + np.random.rand())  # Scale 0-1
    consumption.append(cons)
x_real = np.array([np.column_stack([consumption[b]]) for b in range(B)])  # [B, L, 1]

# Normalize (fit first, then transform)
normalizer = Normalizer(method="log1p")
normalizer.fit(torch.tensor(x_real, dtype=torch.float32))

print("Generated fake data:")
print(f"  Real: {x_real.shape}")
print("\nInitializing mock model and dataloader...")

# Create mock model and dataloader
device = torch.device("cpu")
mock_model = MockVAE(input_dim=24, latent_dim=32, condition_dim=0, device=device)
mock_model.eval()
mock_dataloader = MockDataLoader(batch_size=B, num_batches=2, input_dim=24, device=device, with_timestamps=False)

# For CVAE tests
mock_cvae = MockVAE(input_dim=24, latent_dim=32, condition_dim=12, device=device)  # 7 (day-of-week) + 5 (user_profile)
mock_cvae.eval()
mock_dataloader_cvae = MockDataLoader(batch_size=B, num_batches=2, input_dim=24, device=device, with_timestamps=True)

# ============================================================================
# TEST EACH VISUALIZATION FUNCTION
# ============================================================================

# print("\n" + "="*70)
# print("TEST 1: _visualize_reconstructions")
# print("="*70)
# print("Input: mock_model, mock_dataloader, normalizer")
# try:
#     result = _visualize_reconstructions(
#         model=mock_model,
#         test_loader=mock_dataloader,
#         normalizer=normalizer,
#         device=device,
#         num_samples=2,
#         num_recon_per=2,
#         output_dir=output_dir,
#         model_tag="test_01_reconstructions",
#         is_cvae=False,
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")

# print("\n" + "="*70)
# print("TEST 2: _visualize_synthetic_samples")
# print("="*70)
# print("Input: mock_model, normalizer")
# try:
#     result = _visualize_synthetic_samples(
#         model=mock_model,
#         normalizer=normalizer,
#         device=device,
#         num_samples=3,
#         output_dir=output_dir,
#         model_tag="test_02_synthetic_samples",
#         is_cvae=False,
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")

# print("\n" + "="*70)
# print("TEST 3: _visualize_sigma_mapper")
# print("="*70)
# print("Input: mock_model, mock_dataloader (optional)")
# try:
#     result = _visualize_sigma_mapper(
#         model=mock_model,
#         test_loader=mock_dataloader,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_03_sigma_mapper",
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"SKIPPED: {e}")

# print("\n" + "="*70)
# print("TEST 4: _visualize_latent_space")
# print("="*70)
# print("Input: mock_model, mock_dataloader")
# try:
#     result = _visualize_latent_space(
#         model=mock_model,
#         test_loader=mock_dataloader,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_04_latent_space",
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")

# print("\n" + "="*70)
# print("TEST 5: _visualize_synthetic_recreations")
# print("="*70)
# print("Input: mock_cvae, mock_dataloader_cvae (requires CVAE with conditioner)")
# try:
#     result = _visualize_synthetic_recreations(
#         model=mock_cvae,
#         test_loader=mock_dataloader_cvae,
#         normalizer=normalizer,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_05_synthetic_recreations",
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")

# print("\n" + "="*70)
# print("TEST 6: _visualize_user_dictionary")
# print("="*70)
# print("Input: mock_cvae, mock_dataloader_cvae (requires CVAE with user profiler)")
# try:
#     result = _visualize_user_dictionary(
#         model=mock_cvae,
#         test_loader=mock_dataloader_cvae,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_06_user_dictionary",
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")

# print("\n" + "="*70)
# print("TEST 7: _save_model_summary")
# print("="*70)
# print("Input: mock_model")
# try:
#     result = _save_model_summary(
#         model=mock_model,
#         model_kwargs={"test": "mock"},
#         output_dir=output_dir,
#         model_tag="test_07_model_summary",
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")


# print("\n" + "="*70)
# print("TEST 9: _visualize_prior_vs_posterior")
# print("="*70)
# print("Input: mock_cvae, mock_dataloader_cvae")
# try:
#     result = _visualize_prior_vs_posterior(
#         model=mock_cvae,
#         test_loader=mock_dataloader_cvae,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_09_prior_posterior",
#         num_samples=200,
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")
#     import traceback
#     traceback.print_exc()

# print("\n" + "="*70)
# print("TEST 10: _visualize_user_clustering_quality")
# print("="*70)
# print("Input: mock_cvae (with UserProfiler)")
# try:
#     result = _visualize_user_clustering_quality(
#         model=mock_cvae,
#         test_loader=mock_dataloader_cvae,
#         device=device,
#         output_dir=output_dir,
#         model_tag="test_10_clustering",
#         method='tsne',
#     )
#     print(f"SUCCESS: {result}")
# except Exception as e:
#     print(f"FAILED: {e}")
#     import traceback
#     traceback.print_exc()



print("Input: mock_cvae (with real Conditioner), mock_dataloader_cvae, normalizer")
try:
    
    visualizer = Visualizer(plot_style=PlotStyle(), log_dir=os.path.join(output_dir, "Output/test_11_visualize_trained_model"), model=mock_cvae, writer=Writer(log_dir=os.path.join(output_dir, "Output/test_11_visualize_trained_model")), device=device, normalizer=normalizer)
    visualizer.visualize(test_loader=mock_dataloader_cvae, num_originals=3
                        , num_synthetic_per=30)
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("ALL TESTS COMPLETED")

