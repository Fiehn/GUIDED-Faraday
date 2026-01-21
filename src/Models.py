import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.encoder_decoder_model import *
from src.Conditioner import *

# https://arxiv.org/pdf/2411.03936
# GuideVAE model implementation
# Conditional VAE with conditioning on user profiles
# and a parameterized covariance matrix (PDCC)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.mode = mode

    def __call__(self, val_metric, epoch=None):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            # First time call
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

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
        distribution_dict = {}
        self.model_kwargs = kwargs # Store remaining kwargs
        # Set each remaining kwarg as an attribute
        for key in kwargs: setattr(self, key, kwargs[key])

        # === VAE Components ===
        self.input_dim = input_dim

        # Define encoder and decoder networks
        self.encoder = get_encoder_model(input_dim=input_dim, latent_dim=distribution_dict.get("latent_dim", 10), **self.model_kwargs)
        self.decoder = get_decoder_model(latent_dim=distribution_dict.get("latent_dim", 10), output_dim=input_dim, **self.model_kwargs)

        # Prior parameters for latent spaces
        self.prior_params = get_prior_params(input_dim=input_dim, latent_dim=distribution_dict.get("latent_dim", 10),device="cpu", **self.model_kwargs)

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
        # Sample from the encoder's prior giving latent space samples (ancestral sampling)
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
        posterior_loglikelihood = self.encoder.loglikelihood(inputs, posterior_params_dict, **self.model_kwargs)
        recon_loglikelihood = self.decoder.loglikelihood(inputs, recon_params_dict, **self.model_kwargs)
        prior_loglikelihood = self.encoder.loglikelihood(inputs, self.prior_params, **self.model_kwargs)

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
        kl_div = kl_div.mean()
        # Ensure recon_loglikelihood is also a scalar
        if recon_loglikelihood.dim() > 0:
            recon_loglikelihood = recon_loglikelihood.mean()
        loss = -(recon_loglikelihood - beta * kl_div)
        # TODO: MMD (mean maximum discrepancy) has been shown to be useful for smart meter data by one of the papers
        # TODO: take loglikelihood of samples too??
        return {"loss": loss, "recon_loglikelihood": recon_loglikelihood, "kl_divergence": kl_div}

    def train_core(self, inputs, optim, gradient_clip=None, gradient_clip_kwargs=None, **_):
        # Forward pass
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=1) # TODO: need to implement a input system for kwargs
        
        # Compute loss (negative ELBO)
        optim.zero_grad() # Zero gradients
        loss_dict = self.loss(inputs, recon_params_dict, posterior_params_dict, beta=1.0, prior_params= self.prior_params, **self.model_kwargs) # TODO: need to implement a input system for kwargs
        loss_dict["loss"].backward() # Backpropagate
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))
        optim.step() # Update parameters
        return loss_dict
    
    def validate(self, valloader, num_samples = 1, device = "cpu"):
        self.eval() # Set model to evaluation mode
        # Validaition loop
        val_loss = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
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
            early_stopping = True,
            early_stopping_kwargs = {"patience": 10, "min_delta": 0.001},
            save_epoch_freq = 10,
            writer = None,
            log_dir = './runs/VAE_training',
            **_):
        # === Take Kwarg Inputs ===
        self.model_kwargs.update(locals())
        for key in self.model_kwargs: setattr(self, key, self.model_kwargs[key])
        
        # Initialize TensorBoard writer
        if writer is None:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")

        # === Setup Optimizer and Scheduler ===
        optim = self.get_optimizer(lr=lr, weight_decay=weight_decay)
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, **lr_scheduler_kwargs)
        
        self.to(device)
        self.prior_params = {key: val.to(device) for key, val in self.prior_params.items()}

        if early_stopping:
            earlystopping = EarlyStopping(**early_stopping_kwargs)
        
        # Log model graph
        try:
            sample_batch = next(iter(trainloader))
            sample_batch = self.move_to_device(sample_batch, device)
            writer.add_graph(self, sample_batch)
        except:
            print("Could not log model graph to TensorBoard")
        
        # === Training Loop ===
        self.train()
        global_step = 0
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_recon = 0.0
            epoch_train_kl = 0.0
            num_batches = 0

            for batch in trainloader:
                global_step += 1
                batch = self.move_to_device(batch, device)
                loss_dict = self.train_core(batch, optim, beta=beta, gradient_clip=gradient_clip, gradient_clip_kwargs=gradient_clip_kwargs, **self.model_kwargs)
                
                # Accumulate epoch losses
                epoch_train_loss += loss_dict["loss"].item()
                epoch_train_recon += loss_dict["recon_loglikelihood"].item()
                epoch_train_kl += loss_dict["kl_divergence"].item()
                num_batches += 1
                
                # Log training metrics per iteration
                writer.add_scalar('Train/loss_step', loss_dict["loss"].item(), global_step)
                writer.add_scalar('Train/recon_loglikelihood_step', loss_dict["recon_loglikelihood"].item(), global_step)
                writer.add_scalar('Train/kl_divergence_step', loss_dict["kl_divergence"].item(), global_step)
                
                # Log gradients and parameters periodically
                if global_step % 100 == 0:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                        writer.add_histogram(f'Parameters/{name}', param, global_step)

            # Compute average epoch losses
            avg_train_loss = epoch_train_loss / num_batches
            avg_train_recon = epoch_train_recon / num_batches
            avg_train_kl = epoch_train_kl / num_batches
            
            # Log epoch-level training metrics
            writer.add_scalar('Train/loss_epoch', avg_train_loss, epoch)
            writer.add_scalar('Train/recon_loglikelihood_epoch', avg_train_recon, epoch)
            writer.add_scalar('Train/kl_divergence_epoch', avg_train_kl, epoch)
            
            # Log learning rate
            current_lr = optim.param_groups[0]['lr']
            writer.add_scalar('Train/learning_rate', current_lr, epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}")

            # Validation Step
            if valloader is not None and ((epoch + 1) % validation_freq == 0):
                val_loss = self.validate(valloader, num_samples=validation_num_samples, device=device)
                
                # Log validation metrics
                writer.add_scalar('Val/loss', val_loss["loss"], epoch)
                writer.add_scalar('Val/recon_loglikelihood', val_loss["recon_loglikelihood"], epoch)
                writer.add_scalar('Val/kl_divergence', val_loss["kl_divergence"], epoch)
                
                # Log train vs val comparison
                writer.add_scalars('Comparison/loss', {
                    'train': avg_train_loss,
                    'val': val_loss["loss"]
                }, epoch)
                
                print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss['loss']:.4f}, Recon: {val_loss['recon_loglikelihood']:.4f}, KL: {val_loss['kl_divergence']:.4f}")

                if early_stopping:
                    earlystopping(val_loss["loss"], epoch=epoch)
                    if earlystopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        writer.add_text('Training/early_stop', f'Stopped at epoch {epoch+1}', epoch)
                        break
            
            # Learning Rate Scheduler Step
            if lr_scheduler:
                scheduler.step()
            
            # Save Model Checkpoint
            if (epoch + 1) % save_epoch_freq == 0:
                self.save_model(save_path="./models/", model_name=f"model_epoch_{epoch+1}", device=device)
                writer.add_text('Training/checkpoint', f'Saved model at epoch {epoch+1}', epoch)
        
        self.to("cpu") # Move model back to CPU after training
        self.prior_params = {key: val.to("cpu") for key, val in self.prior_params.items()}
        self.eval() # Set model to evaluation mode
        print("Training complete.")
        
        # Close writer
        if writer is not None:
            writer.close()
            print(f"TensorBoard logs saved to: {log_dir}")
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

        self.encoder = get_encoder_model(input_dim=input_dim + self.condition_dim, latent_dim=latent_dim, **self.model_kwargs)
        self.decoder = get_decoder_model(input_dim=input_dim + self.condition_dim, latent_dim=latent_dim, **self.model_kwargs)

        self.prior_params = get_prior_params(latent_dim=latent_dim, device="cpu", **self.model_kwargs)
    
    def forward(self, inputs, conditions, num_samples=1):
        if conditions.ndims == 2:
            posterior_params_dict = self.encoder(torch.cat([inputs, conditions], dim=-1))
            z = self.encoder.reparameterize(posterior_params_dict, num_samples, **self.model_kwargs)
            recon_params_dict = self.decoder(torch.cat([z, conditions.unsqueeze(1).expand(-1, z.size(1), -1)], dim=-1))
            for key in recon_params_dict:
                recon_params_dict[key] = recon_params_dict[key].squeeze(1)
            return {"recon_params_dict": recon_params_dict, "posterior_params_dict": posterior_params_dict}, {"samples": z}
        elif conditions.ndims == 3:
            if conditions.size(1) != num_samples:
                raise ValueError("Conditions size does not match number of samples.")
            posterior_params_dict = self.encoder(torch.cat([inputs.unsqueeze(1).expand(-1,
                                                                    num_samples,
                                                                    -1), conditions], dim=-1))
            z = self.encoder.reparameterize(posterior_params_dict, num_samples, **self.model_kwargs)
            recon_params_dict = self.decoder(torch.cat([z, conditions], dim=-1))
            for key in recon_params_dict:
                recon_params_dict[key] = recon_params_dict[key].squeeze(1)
            return {"recon_params_dict": recon_params_dict, "posterior_params_dict": posterior_params_dict}, {"samples": z}
        else: # TODO: make this larger??
            raise ValueError("Conditions must be 2D or 3D tensor.") # TODO: double check this implementation

    @torch.no_grad()
    def sample(self, condition, num_prior_samples = 1, num_posterior_samples = 1):
        # TODO: double check this implementation
        if condition.ndims == 2:
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_prior_samples, **self.model_kwargs)
            recon_params_dict = self.decoder(torch.cat([z, condition.unsqueeze(1).expand(-1, z.size(1), -1)], dim=-1))
            samples = self.decoder.sample(recon_params_dict, num_samples=num_posterior_samples, **self.model_kwargs)
            return {"samples": samples, "recon_params_dict": recon_params_dict}
        elif condition.ndims == 3:
            if condition.size(1) != num_prior_samples:
                raise ValueError("Condition size does not match number of prior samples.")
            z = self.encoder.sample(param_dict=self.prior_params, num_samples=num_prior_samples, **self.model_kwargs)
            recon_params_dict = self.decoder(torch.cat([z, condition], dim=-1))
            samples = self.decoder.sample(recon_params_dict, num_samples=num_posterior_samples, **self.model_kwargs)
            return {"samples": samples, "recon_params_dict": recon_params_dict}
        else:
            raise ValueError("Condition must be 2D or 3D tensor.") # TODO: double check this implementation
        
    @torch.no_grad()
    def reconstruct(self, inputs, conditions, num_samples=1):
        outputs, _ = self.forward(inputs, conditions, num_samples=num_samples)
        return outputs["recon_params_dict"], outputs["posterior_params_dict"]

    @torch.no_grad()
    def loglikelihood(self, inputs, conditions, num_samples=1):
        
        outputs, _ = self.forward(inputs, conditions, num_samples=num_samples)
        posterior_params_dict = outputs["posterior_params_dict"]
        recon_params_dict = outputs["recon_params_dict"]
        posterior_loglikelihood = self.encoder.loglikelihood(inputs, posterior_params_dict, **self.model_kwargs)
        recon_loglikelihood = self.decoder.loglikelihood(inputs, recon_params_dict, **self.model_kwargs)
        prior_loglikelihood = self.encoder.loglikelihood(inputs, self.prior_params, **self.model_kwargs)
        return (posterior_loglikelihood.sum(-1) + recon_loglikelihood.sum(-1) - prior_loglikelihood.sum(-1)).mean(0)
        
    def train_core(self, inputs, optim, **_):
        inputs, conditions = inputs
        outputs, _ = self.forward(inputs, conditions, num_samples=1)
        optim.zero_grad(set_tp_none = True)
        loss_dict = self.loss(inputs, outputs["recon_params_dict"], outputs["posterior_params_dict"], beta=1.0, prior_params=self.prior_params, **self.model_kwargs)
        loss_dict["loss"].backward()
        optim.step()
        return loss_dict["loss"]

    @torch.no_grad()
    def validate(self, valloader, num_samples = 1, beta = 1.0, device = "cpu"):
        self.eval()
        val_loss = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        for batch in valloader:
            inputs, conditions = batch
            inputs = inputs.to(device)
            conditions = conditions.to(device)
            outputs, _ = self.forward(inputs, conditions, num_samples=num_samples)
            loss_dict = self.loss(inputs, outputs["recon_params_dict"], outputs["posterior_params_dict"], beta=beta, prior_params=self.prior_params, **self.model_kwargs)
            for key in val_loss:
                val_loss[key] += loss_dict[key].item() * inputs.size(0)
        for key in val_loss:
            val_loss[key] /= len(valloader.dataset)
        return val_loss

    def move_to_device(self, inputs, device):
        return super().move_to_device(inputs, device)