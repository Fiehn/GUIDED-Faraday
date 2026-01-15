import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.encoder_decoder_model import *

# https://arxiv.org/pdf/2411.03936
# GuideVAE model implementation
# Conditional VAE with conditioning on user profiles
# and a parameterized covariance matrix (PDCC)

# ==============
# Define a base VAE class that CVAE will inherit from
#   - should be the base that contains all core aspects
#   - should we make this an abstract class? Probably not necessary in python
#   - encoder and decoder are modular and can be swapped out (which means they need to be implemented independently)
# ==============
class VAE(nn.Module):
    def __init__(self,
                 input_dim=24,
                 distribution_dict = {},
                 **_):
        super(VAE, self).__init__()

        # === Argument Handling === (Allows for flexible passing of arguments)
        # Capture all local variables in kwargs
        kwargs = dict(locals())

        # Remove unwanted keys from kwargs (all in the list)
        for key in ["self","__class__", "input_dim"]: kwargs.pop(key)
        distribution_dict = {}, self.model_kwargs = kwargs # Store remaining kwargs
        # Set each remaining kwarg as an attribute
        for key in kwargs: setattr(self, key, kwargs[key])

        # === VAE Components ===
        self.input_dim = input_dim

        # Define encoder and decoder networks
        self.encoder = get_encoder_model(input_dim=input_dim, latent_dim=distribution_dict.get("latent_dim", 10), **self.model_kwargs)
        self.decoder = get_decoder_model(latent_dim=distribution_dict.get("latent_dim", 10), output_dim=input_dim, **self.model_kwargs)

        # Prior parameters for latent spaces
        self.prior_params = get_prior_params(input_dim=input_dim, latent_dim=distribution_dict.get("latent_dim", 10), **self.model_kwargs)

        # Calculate and store the number of parameters
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()
    
    def forward(self, inputs, num_samples=1):
        # Encode inputs to get posterior parameters
        posterior_params_dict = self.encoder(inputs)
        # Sample latent variables using reparameterization trick 
        z = self.encoder.reparameterize(posterior_params_dict, num_samples, **self.model_kwargs) 
        # Decode latent samples to get reconstruction parameters
        recon_params_dict = self.decoder(z)
        # TODO: reshape issues are waiting here
        return recon_params_dict, posterior_params_dict
    
    @torch.no_grad()
    def sample(self, num_samples_prior = 1, num_samples_posterior = 1):
        # Sample from the encoder's prior giving latent space samples
        z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_samples_prior, **self.model_kwargs)
        # Decode the sampled latent variables to get reconstruction parameters
        recon_params_dict = self.decoder(z)
        # Generate samples from the decoder using the reconstruction parameters
        samples = self.decoder.sample(recon_params_dict, num_samples=num_samples_posterior, **self.model_kwargs)
        # Return both the samples and the reconstruction parameters
        return {"samples": samples, "recon_params_dict": recon_params_dict}
    
    @torch.no_grad()
    def reconstruct(self, inputs, num_samples=1):
        # Run a single forward pass to get reconstruction parameters
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_samples)
        return recon_params_dict, posterior_params_dict

    @torch.no_grad()
    def loglikelihood(self, inputs, num_samples=1):
        # TODO: double check this implementation
        
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_samples)
        # Compute log-likelihood of inputs given reconstruction parameters
        posterior_loglikelihood = self.encoder.loglikelihood() # TODO: implement this method
        recon_loglikelihood = self.decoder.loglikelihood() # TODO: implement this method
        prior_loglikelihood = self.encoder.loglikelihood() # TODO: implement this method

        # Combine log-likelihoods to compute ELBO ?? is it Elbo
        return (posterior_loglikelihood.sum(-1) + recon_loglikelihood.sum(-1) - prior_loglikelihood.sum(-1)).mean(0)
    
    def kl_divergence(self, posterior_params_dict):
        # Compute KL divergence between posterior and prior distributions
        kl_div = self.encoder.kl_divergence(posterior_params_dict, self.prior_params, **self.model_kwargs)
        return kl_div.mean(0)
    
    def reconstruction_loglikelihood(self, inputs, recon_params_dict, **_):
        # Compute reconstruction log-likelihood of inputs given reconstruction parameters
        # TODO: double check this implementation
        recon_loglikelihood = self.decoder.loglikelihood(inputs, recon_params_dict, **self.model_kwargs)
        return recon_loglikelihood.mean(0)

    def loss(self, inputs, recon_params_dict, posterior_params_dict, beta=1.0, prior_params=None, **_):
        # Compute reconstruction log-likelihood (we use reconstruction only because of VAE structure)
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict, **self.model_kwargs)
        # Compute KL divergence between posterior and prior
        if prior_params is None:
            prior_params = self.prior_params
        kl_div = self.encoder.kl_divergence(posterior_params_dict, prior_params, **self.model_kwargs)
        kl_div = kl_div.mean(0)
        loss = -(recon_loglikelihood - beta * kl_div)
        # TODO: MMD (mean maximum discrepancy) has been shown to be useful for smart meter data by one of the papers
        # TODO: take loglikelihood of samples too??
        return {"loss": loss, "recon_loglikelihood": recon_loglikelihood, "kl_divergence": kl_div}["loss"]

    def train_core(self, inputs, optim, **_):
        # Forward pass
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=1) # TODO: need to implement a input system for kwargs
        
        # Compute loss (negative ELBO)
        optim.zero_grad() # Zero gradients
        loss = self.loss(inputs, recon_params_dict, posterior_params_dict, beta=1.0, prior_params= self.prior_params, **self.model_kwargs) # TODO: need to implement a input system for kwargs
        loss.backward() # Backpropagate
        # if grad_clip is not None: (TODO: implement gradient clipping)
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optim.step() # Update parameters
        return loss
    
    def validate(self, valloader, num_samples = 1, device = "cpu"):
        self.eval() # Set model to evaluation mode
        # Validaition loop
        val_loss = {"elbo": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over validation data
            for batch in valloader:
                batch = batch.to(device)
                # Forward pass
                recon_params_dict, posterior_params_dict = self.forward(batch, num_samples=num_samples)
                loss_dict = self.loss(batch, recon_params_dict, posterior_params_dict, beta=1.0, prior_params=self.prior_params, **self.model_kwargs)
                # Accumulate losses
                for key in val_loss:
                    val_loss[key] += loss_dict[key].item() * batch.size(0)
        # Average losses over dataset
        for key in val_loss:
            val_loss[key] /= len(valloader.dataset)

        return val_loss
    
    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        # Return Adam optimizer for model parameters
        # TODO: Allow for different optimizers and parameter settings
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) # TODO: Parameters should be inputted
    
    def move_to_device(self, inputs, device):
        # Move data to specified device (CPU or GPU)
        return inputs.to(device)

    def save_model(self, save_path=None, model_name="trained_model",device="cpu"):
        if save_path is None: save_path = "./models/" # TODO: Should be determined by config file
        model_path = os.path.join(save_path, f"{model_name}.pt")
        self.eval() # Set model to evaluation mode
        self.to("cpu") # Move model to CPU for saving
        self.prior_params = {key: val.to("cpu") for key, val in self.prior_params.items()}
        torch.save({
            'model_state_dict': self.state_dict(),
            'prior_params': self.prior_params,
            'model_kwargs': self.model_kwargs
        }, model_path)
        self.to(device) # Move model back to original device
        self.prior_params = {key: val.to(device) for key, val in self.prior_params.items()}
        print(f"Model saved to {model_path}")

    def load_model(self, load_path, device="cpu"):
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.prior_params = {key: val.to(device) for key, val in checkpoint['prior_params'].items()}
        self.to(device)
        print(f"Model loaded from {load_path}")

    def fit(self,
            trainloader,
            valloader = None,
            lr = 1e-3,
            weight_decay = 1e-4,
            gradient_clip = None,
            gradient_clip_kwargs = {"max_norm": 1.0},
            lr_scheduler = False,
            lr_scheduler_kwargs = {"step_size": 10, "gamma": 0.9},
            beta = 1.0,
            num_samples = 1,
            epochs = 1000,
            validation_freq = 1,
            validation_num_samples = 10,
            device = "cpu",
            early_stopping = False,
            early_stopping_kwargs = {"patience": 10, "min_delta": 0.001},
            save_epoch_freq = 10,
            writer = None,
            **_):
        # === Take Kwarg Inputs ===
        self.model_kwargs.update(locals())
        for key in self.model_kwargs: setattr(self, key, self.model_kwargs[key])
        
        # TODO: FIGURE OUT logging SYSTEM (tensorboard, wandb, etc. ??)

        # === Setup Optimizer and Scheduler ===
        optim = self.get_optimizer(lr=lr, weight_decay=weight_decay)
        if lr._scheduler: # TODO: implement learning rate scheduler (should improve the model performance)
            scheduler = torch.optim.lr_scheduler.StepLR(optim, **lr_scheduler_kwargs)
        
        self.to(device)
        self.prior_params = {key: val.to(device) for key, val in self.prior_params.items()}

        if early_stopping:
            earlystopping = EarlyStopping(**early_stopping_kwargs) # TODO: implement EarlyStopping class
        
        # === Training Loop ===
        self.train()
        epochs = 0 # We store this for early stopping
        iterations = 0
        for _ in range(epochs):
            epochs += 1
            # LOG HERE TODO: implement logging system

            for batch in trainloader:
                iterations += 1
                batch = self.move_to_device(batch, device)
                loss = self.train_core(batch, optim, beta=beta, **self.model_kwargs)

            # Validation Step
            if valloader is not None and (epochs % validation_freq == 0):
                val_loss = self.validate(valloader, num_samples=validation_num_samples, device=device)
                # LOG HERE TODO: implement logging system

                if early_stopping:
                    earlystopping(val_loss["elbo"])
                    if earlystopping.early_stop:
                        print("Early stopping triggered.")
                        break
            # Learning Rate Scheduler Step
            if lr_scheduler:
                scheduler.step()
            # Save Model Checkpoint
            if epochs % save_epoch_freq == 0:
                self.save_model(save_path="./models/", model_name=f"model_epoch_{epochs}", device=device)
        
        self.to("cpu") # Move model back to CPU after training
        self.prior_params = {key: val.to("cpu") for key, val in self.prior_params.items()}
        self.eval() # Set model to evaluation mode
        print("Training complete.")
        return

class CVAE(VAE):
    def __init__(self,
                 input_dim=None,
                 condition_object = None,
                 latent_dim = 10,
                distribution_dict = {},
                 **_):
        # Kwarg Handling
        kwargs = dict(locals())
        for key in ["self","__class__", "input_dim", "condition_object"]: kwargs.pop(key)
        self.model_kwargs = kwargs
        super(CVAE, self).__init__(input_dim=input_dim, distribution_dict=distribution_dict, **self.model_kwargs)

        self.condition_dim = condition_object.condition_dim

        self.encoder = get_encoder_model() # TODO: implement this function
        self.decoder = get_decoder_model() # TODO: implement this function

        self.prior_params = get_prior_params() # TODO: implement this function
    
    def forward():
        pass # TODO: implement forward method

    def sample():
        pass # TODO: implement sample method

    def reconstruct():
        pass # TODO: implement reconstruct method

    def loglikelihood():
        pass # TODO: implement loglikelihood method

    def train_core():
        pass # TODO: implement train_core method

    def validate():
        pass # TODO: implement validate method

    def move_to_device(self, inputs, device):
        return super().move_to_device(inputs, device)