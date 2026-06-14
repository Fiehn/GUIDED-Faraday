import torch

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 
from model.base.encoder_decoder_model import *
from data.conditioner import *
from model.base.priors import *
from data.normalizer import *
 
from src.model.hierarchical.config import HierarchicalConfig
from src.model.hierarchical.core import HierarchicalLatentModule

from model.base.model import BaseModel


# https://arxiv.org/pdf/2411.03936
# GuideVAE model implementation
# Conditional VAE with conditioning on user profiles
# and a parameterized covariance matrix (PDCC)

# ==============
# Define a base VAE class that CVAE will inherit from
#   - should be the base that contains all core aspects
#   - encoder and decoder are modular and can be swapped out
# ==============
class VAE(BaseModel):
    """Base VAE class with modular encoder, decoder, prior and hierarchical latent space.
    """
    def __init__(self,
                 input_dim=24,
                 latent_dim=32,
                 dictionary_size=100,
                 vamp_num_components=20,
                 encoder_type = "dictionary",
                 decoder_type = "dictionary",
                 prior_type = None,
                 device="cpu",
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 encoder_dropout=None,
                 decoder_dropout=None,
                 hierarchy_config: HierarchicalConfig = None,
                 phased_mode: bool = False,
                 sequence_length: int = 24,
                 **_):
        super(VAE, self).__init__(phased_mode=phased_mode, sequence_length=sequence_length)

        # === Argument Handling ===
        # Capture all local variables in kwargs
        kwargs = dict(locals())

        # Remove unwanted keys from kwargs (all in the list)
        for key in ["self","__class__", "input_dim", "distribution_dict", "_"]: 
            kwargs.pop(key, None)  # Use pop with default to avoid KeyError
        
        self.model_kwargs = kwargs # Store remaining kwargs
        # Set each remaining kwarg as an attribute
        for key in kwargs: 
            setattr(self, key, kwargs[key])

        # === VAE Components ===
        self.input_dim = input_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type 
        self.prior_type = prior_type
        self.latent_dim = latent_dim

        self.device = device

        encoder_model_kwargs = {}
        if encoder_hidden_dims is not None:
            encoder_model_kwargs["hidden_dims"] = list(encoder_hidden_dims)
        if isinstance(encoder_dropout, bool):
            encoder_model_kwargs["dropout"] = encoder_dropout
        elif encoder_dropout is not None:
            try:
                encoder_dropout_rate = float(encoder_dropout)
                encoder_model_kwargs["dropout"] = encoder_dropout_rate > 0.0
                encoder_model_kwargs["dropout_rate"] = encoder_dropout_rate
            except (TypeError, ValueError):
                encoder_model_kwargs["dropout"] = bool(encoder_dropout)

        decoder_model_kwargs = {}
        if decoder_hidden_dims is not None:
            decoder_model_kwargs["hidden_dims"] = list(decoder_hidden_dims)
        if isinstance(decoder_dropout, bool):
            decoder_model_kwargs["dropout"] = decoder_dropout
        elif decoder_dropout is not None:
            try:
                decoder_dropout_rate = float(decoder_dropout)
                decoder_model_kwargs["dropout"] = decoder_dropout_rate > 0.0
                decoder_model_kwargs["dropout_rate"] = decoder_dropout_rate
            except (TypeError, ValueError):
                decoder_model_kwargs["dropout"] = bool(decoder_dropout)

        # 1. Create encoder and decoder and the latent module for the hierarchy if specified
        self.encoder = get_encoder_model(input_dim=input_dim, 
                                         latent_dim=latent_dim, 
                                         type=encoder_type, 
                                         dictionary_size=dictionary_size,
                                         **encoder_model_kwargs)

        # Hierarchical latent module (optional) - can be used to add multiple levels of latent variables with different strategies and connections between them
        self.hierarchy = None
        self.decoder_input_dim = latent_dim # By default, the decoder takes the latent variable as input, but if we have a hierarchy, it will take the output of the hierarchy instead
        if hierarchy_config is not None:
            self.hierarchy = HierarchicalLatentModule(
                config=hierarchy_config,
                encoder_latent_dim=latent_dim,
            )
            self.decoder_input_dim = self.hierarchy.output_dim
            
        self.decoder = get_decoder_model(latent_dim=self.decoder_input_dim, 
                                         output_dim=input_dim, 
                                         type=decoder_type, 
                                         dictionary_size=dictionary_size,
                                         **decoder_model_kwargs)

        # 2. Create prior after encoder (VampPrior needs a reference to the encoder)
        if prior_type is None:
            prior_type = "standard_dictionary" if encoder_type == "dictionary" else "standard_gaussian"
            self.prior_type = prior_type
        
        self.prior = get_prior(
            prior_type=prior_type,
            latent_dim=self.decoder_input_dim,
            device=self.device,
            dictionary_size=dictionary_size,
            input_shape=input_dim,
            encoder=self.encoder,
            num_components=vamp_num_components,
        )

        # Calculate and store the number of parameters
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()
    
    def forward(self, inputs, num_samples=1):
        # Encode inputs to get posterior parameters
        posterior_params_dict = self.encoder(inputs)
        # Sample latent variables using reparameterization trick 
        z = self.encoder.reparameterize(posterior_params_dict, num_samples=num_samples)
        
        # If we have a hierarchy, pass z through it
        hierarchy_results = None
        if self.hierarchy is not None:
            # z shape: [S, B, latent_dim]
            S,B,D = z.shape
            z_flat = z.reshape(S * B, D)  # Flatten for hierarchy: [S*B, latent_dim]
            hierarchy_results = self.hierarchy.forward(z_flat)  # Pass through hierarchy
            z_for_decoder = hierarchy_results["z"].reshape(S, B, -1)  # Reshape back to [S, B, output_dim]
        else:
            z_for_decoder = z
        
        # Decode latent samples to get reconstruction parameters
        recon_params_dict = self.decoder(z_for_decoder)

        self._last_hierarchy_results = hierarchy_results  # Store for later use in loss calculation or analysis

        return recon_params_dict, posterior_params_dict
    
    @torch.no_grad()
    def sample(self, num_samples_prior = 1, num_samples_posterior = 1, **kwargs):
        """Sample from prior and decode."""
        self.eval()

        # Get prior parameters and move to device
        prior_params = self.prior.get_params()
        prior_params = {k: v.to(self.device) for k, v in prior_params.items()}

        # For dictionary priors, we need to expand to a batch
        batch_size = 1
        prior_params_expanded = {
            k: v.expand(batch_size, -1) for k, v in prior_params.items()
        }
        # Sample from encoder using prior parameters
        z = self.encoder.sample(prior_params_expanded, num_samples=num_samples_prior)  # [S, B, latent_dim]

        # Flatten for decoder: [S*B, latent_dim]
        S, B, D = z.shape
        z_flat = z.reshape(S * B, D)

        # Hierarchy
        z_for_decoder = z_flat
        if self.hierarchy is not None:
            hierarchy_results = self.hierarchy.forward(z_flat)
            z_for_decoder = hierarchy_results["z"]

        # Decode
        recon_params_dict = self.decoder(z_for_decoder)

        # Reshape back: [S*B, output_dim] -> [S, B, output_dim]
        for k in recon_params_dict:
            recon_params_dict[k] = recon_params_dict[k].reshape(S, B, -1)

        # Sample from decoder
        samples_all = []
        for s in range(S):
            params_s = {k: v[s] for k, v in recon_params_dict.items()}
            samples = self.decoder.sample(params_s, num_samples=num_samples_posterior)  # [S2, B, D]
            samples_all.append(samples)
        
        samples_stacked = torch.stack(samples_all, dim=0)  # [S, S2, B, D]
        
        return {"samples": samples_stacked, "recon_params_dict": recon_params_dict}
    
    @torch.no_grad()
    def reconstruct(self, inputs, num_samples=1):
        # Run a single forward pass to get reconstruction parameters
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_samples)
        return recon_params_dict, posterior_params_dict
    
    def reconstruction_loglikelihood(self, inputs, recon_params_dict):
        # Compute reconstruction log-likelihood
        return self.decoder.loglikelihood(inputs, recon_params_dict)
    
    @torch.no_grad()
    def loglikelihood(self, inputs, num_mc_samples=1):
        """Compute log-likelihood"""
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_mc_samples)

        # Reconstruction log-likelihood
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)
        posterior_loglikelihood = self.encoder.loglikelihood(posterior_params_dict)
        prior_loglikelihood = self.prior.log_prob(posterior_params_dict['z'])
        
        return (recon_loglikelihood.sum(-1) + prior_loglikelihood.sum(-1) - posterior_loglikelihood.sum(-1)).mean(dim=0)
    
    def loss(self, inputs, recon_params_dict, posterior_params_dict, beta: float = 1.0, weighted_loss: float = 0.0, **_):
        # Compute reconstruction log-likelihood
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)

        # Compute KL divergence between posterior and prior
        kl_div = self.encoder.kl_divergence(posterior_params_dict, self.prior)
        kl_div = kl_div.mean()

        # Weighted loss (set weighted_loss > 0 to enable) - gives more weight to samples with higher absolute values, which should help with the sparsity issue in smart meter data
        weights = 1.0 + weighted_loss * torch.abs(inputs).mean(dim=-1)
        weights = weights / weights.mean() # Should not change the overall scale of the loss, just reweight samples within the batch (This will hopefully allow vamp to work!)
        recon_loglikelihood = (recon_loglikelihood * weights).mean()

        loss = -(recon_loglikelihood - beta * kl_div)

        if self._last_hierarchy_results is not None:
            hierarchy_loss = self._last_hierarchy_results["loss"]
            loss = loss + hierarchy_loss

        return {"loss": loss, "recon_loglikelihood": recon_loglikelihood, "kl_divergence": kl_div}

    def train_core(self, batch, optim, gradient_clip=None, gradient_clip_kwargs=None, beta=1.0, weighted_loss =0.0, **_):
        # batch can be x OR (x, meter_id, timeStamp)
        self.train()  # Set model to training mode

        x, meter_id, timestamps, agg = self._unpack_batch(batch)
        x_flat = self._prep_inputs(x.to(self.device))  # Flatten if needed and move to device

        # Forward pass
        recon_params_dict, posterior_params_dict = self.forward(x_flat, num_samples=1)

        optim.zero_grad(set_to_none=True)  # Zero gradients

        # Compute loss
        loss_dict = self.loss(x_flat, recon_params_dict, posterior_params_dict, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward() # Backpropagate
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))
        optim.step() # Update parameters
        return loss_dict
    
    def validate(self, valloader, num_samples = 1, beta=1, device = "cpu"):
        self.eval() # Set model to evaluation mode
        # Validation loop
        val_loss = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batch = 0
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over validation data
            for batch in valloader:
                x, meter_id, timestamps, agg = self._unpack_batch(batch)
                x_flat = self._prep_inputs(x.to(device))  # Flatten if needed and move to device
                # Forward pass
                recon_params_dict, posterior_params_dict = self.forward(x_flat, num_samples=num_samples)
                # Get prior params
                prior_params = self.prior.get_params()
                batch_size = x_flat.shape[0]
                prior_params = {k: v.expand(batch_size, -1).to(device) for k, v in prior_params.items()}
                # Get loss
                loss_dict = self.loss(x_flat, recon_params_dict, posterior_params_dict, beta=beta, prior_params=prior_params)
                # Accumulate losses
                for key in val_loss:
                    val_loss[key] += loss_dict[key].item()
                num_batch += 1
        # Average losses over dataset
        for key in val_loss:
            val_loss[key] /= num_batch

        return val_loss
    
    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        # Return Adam optimizer for model parameters
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_SigmaMapper(self):
        return self.decoder.get_SigmaMapper()
