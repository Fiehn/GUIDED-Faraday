import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 
from src.encoder_decoder_model import *
from src.Conditioner import *
from src.priors import *

# https://arxiv.org/pdf/2411.03936
# GuideVAE model implementation
# Conditional VAE with conditioning on user profiles
# and a parameterized covariance matrix (PDCC)

# TODO:
# - Implement autocast with GradScaler for mixed precision training (Much faster training on GPUs)
# - Make sure everything works with CPU and GPU seamlessly
# - Fix the logging so that it is not too cluttered
# - Implement learning rate finder utility
# - Implement model checkpointing and resuming training
# - Implement unit tests for all components
# - Implement the user profile conditioning properly
# - Ensure PDCC works as intended
# - Declutter the code and add more comments/docstrings
# - Optimize data loading and preprocessing pipeline
# - Enhance the model saving/loading mechanism
# - Add more conditions
# - Hyperparameter tuning utilities (for optimizers, architectures, etc.)
# - Make sure the model works with variable length sequences (padding/masking)
# - IMPORTANT: Streaming algorithm for user profiles (online LDA) and normalization (running mean/std) and for the training (online/streaming VAE training)
# - Implement scheduler for beta (KL weight) during training (KL annealing)
# - Implement different priors (sparse, mixture of Gaussians, VampPrior, etc.)
# - Fix the conditioner to properly handle conditions
# - Test loading of models and make sure it is possible and reliable
# - Test different loss functions and double check the current implementation against the literature

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.mode = mode

    def __call__(self, train_metric, val_metric, epoch=None):
        score = -val_metric if self.mode == 'min' else val_metric

        if train_metric is not None and val_metric is not None:
            if abs(train_metric - val_metric) < self.min_delta * 10:
                # Overfitting detected - Stop training
                self.early_stop = True
                return False

        if self.best_score is None:
            # First time call
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False                
        else:
            self.best_score = score
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            return True
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

# ==============
# Define a base VAE class that CVAE will inherit from
#   - should be the base that contains all core aspects
#   - encoder and decoder are modular and can be swapped out (which means they need to be implemented independently)
# ==============
class VAE(nn.Module):
    def __init__(self,
                 input_dim=24,
                 distribution_dict = {},
                 coder_type = "dictionary",
                 prior_type = None,
                 **_):
        super(VAE, self).__init__()

        # === Argument Handling === (Allows for flexible passing of arguments)
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
        latent_dim = distribution_dict.get("latent_dim", 10)

        # Prior
        if prior_type is None:
            prior_type = "standard_dictionary" if coder_type == "dictionary" else "standard_gaussian"
        
        self.prior = get_prior(prior_type=prior_type, latent_dim=latent_dim, device="cpu", dictionary_size=distribution_dict.get("dictionary_size", 100))

        # Define encoder and decoder networks - don't pass model_kwargs to avoid circular references
        self.encoder = get_encoder_model(input_dim=input_dim, latent_dim=latent_dim, type=coder_type, prior=self.prior)
        self.decoder = get_decoder_model(latent_dim=latent_dim, output_dim=input_dim, type=coder_type, prior=self.prior)

        # Calculate and store the number of parameters
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()
    
    def forward(self, inputs, num_samples=1):
        # Encode inputs to get posterior parameters
        posterior_params_dict = self.encoder(inputs)
        # Sample latent variables using reparameterization trick 
        z = self.encoder.reparameterize(posterior_params_dict, num_samples=num_samples)
        # Decode latent samples to get reconstruction parameters
        recon_params_dict = self.decoder(z)
        return recon_params_dict, posterior_params_dict
    
    @torch.no_grad()
    def sample(self, num_samples_prior = 1, num_samples_posterior = 1):
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
        
        # Decode
        recon_params_dict = self.decoder(z_flat)
        
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
        return self.decoder.loglikelihood(inputs, recon_params_dict).mean(dim=0)
    @torch.no_grad()
    def loglikelihood(self, inputs, num_mc_samples=1):
        """Compute log-likelihood"""
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_mc_samples)

        # Reconstruction log-likelihood
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)
        posterior_loglikelihood = self.encoder.loglikelihood(posterior_params_dict)
        prior_loglikelihood = self.prior.log_prob(posterior_params_dict['z'])
        
        return (posterior_loglikelihood.sum(-1) + recon_loglikelihood.sum(-1) - prior_loglikelihood.sum(-1)).mean(dim=0)
    
    def loss(self, inputs, recon_params_dict, posterior_params_dict, beta=1.0, **_):
        # Compute reconstruction log-likelihood (we use reconstruction only because of VAE structure)
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)

        # Compute KL divergence between posterior and prior
        kl_div = self.encoder.kl_divergence(posterior_params_dict, self.prior)
        kl_div = kl_div.mean()

        # Ensure recon_loglikelihood is also a scalar
        if recon_loglikelihood.dim() > 0:
            recon_loglikelihood = recon_loglikelihood.mean()

        loss = -(recon_loglikelihood - beta * kl_div)
        # TODO: MMD (mean maximum discrepancy) has been shown to be useful for smart meter data by one of the papers
        return {"loss": loss, "recon_loglikelihood": recon_loglikelihood, "kl_divergence": kl_div}

    def train_core(self, x, optim, gradient_clip=None, gradient_clip_kwargs=None, beta=1.0, **_):
        # batch can be x OR (x, meter_id, timeStamp)
        self.train()  # Set model to training mode

        if isinstance(x, tuple) or isinstance(x, list):
            # Unpack if dataloader returns (x, meter_id, timestamps)
            if len(x) == 3:
                x_data, meter_id, timestamps = x
            elif len(x) == 2:
                x_data, meter_id = x
                timestamps = None
            else:
                x_data = x[0]
                meter_id = None
                timestamps = None
        else:
            x_data = x
            meter_id = None
            timestamps = None

       # Flatten x if needed
        if x_data.dim() == 3:  # [B, L, F]
            x_flat = x_data.reshape(x_data.shape[0], -1)
        else:
            x_flat = x_data

        x_flat = x_flat.to(self.device)
        
        # Forward pass
        recon_params_dict, posterior_params_dict = self.forward(x_flat, num_samples=1)

        optim.zero_grad(set_to_none=True)  # Zero gradients

        # Compute loss
        loss_dict = self.loss(x_flat, recon_params_dict, posterior_params_dict, beta=beta)
        loss_dict["loss"].backward() # Backpropagate
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))
        optim.step() # Update parameters
        return loss_dict
    
    def validate(self, valloader, num_samples = 1, device = "cpu"):
        self.eval() # Set model to evaluation mode
        # Validation loop
        val_loss = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over validation data
            for batch in valloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                if batch.dim() == 3:
                    batch = batch.reshape(batch.shape[0], -1)
                batch = batch.to(device)
                # Flatten if needed
                if batch.dim() == 3:
                    batch = batch.reshape(batch.shape[0], -1)

                # Forward pass
                recon_params_dict, posterior_params_dict = self.forward(batch, num_samples=num_samples)
                # Get prior params
                prior_params = self.prior.get_params()
                batch_size = batch.shape[0]
                prior_params = {k: v.expand(batch_size, -1).to(device) for k, v in prior_params.items()}
                # Get loss
                loss_dict = self.loss(batch, recon_params_dict, posterior_params_dict, beta=1.0, prior_params=prior_params)
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

    def move_to_device(self, batch, device): # Handles: Tensor, tuple/list of tensors, dict of tensors
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                b.to(device) if torch.is_tensor(b) else b
                for b in batch
            )
        if isinstance(batch, dict):
            return {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        return batch

    def save_model(self, save_path=None, model_name="trained_model",device="cpu"):
        """
        Save model with dynamic prior serialization.
        Works with any prior type without modification.
        """
        if save_path is None: save_path = "./models/" # TODO: Should be determined by config file
        model_path = os.path.join(save_path, f"{model_name}.pt")
        self.eval() # Set model to evaluation mode
        self.to("cpu") # Move model to CPU for saving
        self.prior.to("cpu")

        # Serialize prior
        prior_state = {
            'prior_type': self.prior.__class__.__name__,
            'module': self.prior.__class__.__module__,
            'state_dict': self.prior.state_dict(),
            'init_args': {},
        }
        # dynamically extract init args (latent_dim, dict_size, etc.)
        for attr_name in dir(self.prior):
            if not attr_name.startswith('_') and not callable(getattr(self.prior, attr_name)):
                attr = getattr(self.prior, attr_name)
                try:
                    if isinstance(attr,(int,float,str,bool,list,tuple,dict)):
                        prior_state['init_args'][attr_name] = attr
                    elif isinstance(attr, torch.dtype):
                        prior_state['init_args'][attr_name] = str(attr)
                except:
                    pass            

        torch.save({
            'model_state_dict': self.state_dict(),
            'prior_state': prior_state,
            'model_kwargs': self.model_kwargs
        }, model_path)
        self.to(device) # Move model back to original device
        self.prior.to(device)
        print(f"Model saved to {model_path}")

    def load_model(self, load_path, device="cpu"):
        """
        Load model with dynamic prior reconstruction.
        Works with any prior type without modification.
        TODO: should probably be handled in the BasePrior class
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=True)

        prior_state = checkpoint['prior_state']

        import importlib
        module = importlib.import_module(prior_state['module'])
        PriorClass = getattr(module, prior_state['prior_type'])

        init_args = prior_state.get('init_args', {})

        try:
            self.prior = PriorClass(**init_args)
        except Exception as e:
            print(f"Error initializing prior {prior_state['prior_type']} with args {init_args}: {e}")

        self.prior.load_state_dict(prior_state['state_dict'])
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        self.prior.to(device)
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
            device = "cpu",  # Add device parameter
            early_stopping = True,
            early_stopping_kwargs = {"patience": 10, "min_delta": 0.001},
            save_epoch_freq = 10,
            writer = None,
            log_dir = './runs/VAE_training',
            model_name = None, # Optionally specify model name for saving
            save_dir = './models/',
            **_):
        # === Generate a model tag ===
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_tag = f"{self.__class__.__name__}_{self.decoder.__class__.__name__}_{time_stamp}"
        else:
            model_tag = f"{model_name}_{time_stamp}"
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, model_tag)

        # === Take Kwarg Inputs ===
        fit_kwargs = dict(locals())
        for key in ["self", "trainloader", "valloader", "writer", "_"]:
            fit_kwargs.pop(key, None)
        self.model_kwargs.update(fit_kwargs)
        self.model_kwargs['model_tag'] = model_tag
        
        # === Save training configuration ===
        config_path = os.path.join(save_dir, "training_config.json")
        config = {
            'model_tag': model_tag,
            'timestamp': time_stamp,
            'hyperparameters': {
                'lr': lr,
                'weight_decay': weight_decay,
                'beta': beta,
                'epochs': epochs,
                'batch_size': trainloader.batch_size,
                'gradient_clip': gradient_clip,
                'lr_scheduler': lr_scheduler,
                'early_stopping': early_stopping,
            },
            'model_architecture': {
                'class_name': self.__class__.__name__,
                'input_dim': self.input_dim,
                'num_parameters': self.num_parameters,
                'encoder': self.encoder.__class__.__name__,
                'decoder': self.decoder.__class__.__name__,
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Training configuration saved to: {config_path}")

        # === Initialize TensorBoard writer ===
        if writer is None:
            log_dir_tagged = os.path.join(log_dir, model_tag)
            writer = SummaryWriter(log_dir=log_dir_tagged)
            print(f"TensorBoard logging to: {log_dir_tagged}")
            # Log configuration to TensorBoard
            writer.add_text('Config/model_tag', model_tag, 0)
            writer.add_text('Config/hyperparameters', json.dumps(config['hyperparameters'], indent=2), 0)

        # === Setup Optimizer and Scheduler ===
        optim = self.get_optimizer(lr=lr, weight_decay=weight_decay)
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, **lr_scheduler_kwargs)
        
        if next(self.parameters()).device != torch.device(device):
            self.to(device)
        self.prior.to(device)

        if early_stopping:
            earlystopping = EarlyStopping(**early_stopping_kwargs)

        # === Training Loop ===
        self.train()
        global_step = 0
        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_recon = 0.0
            epoch_train_kl = 0.0
            num_samples = 0

            for batch in trainloader:
                global_step += 1
                batch = self.move_to_device(batch, device)
                loss_dict = self.train_core(
                    batch, 
                    optim, 
                    beta=beta, 
                    gradient_clip=gradient_clip, 
                    gradient_clip_kwargs=gradient_clip_kwargs
                )
                if isinstance(batch, (list, tuple)):
                    bs = batch[0].size(0)
                else:
                    bs = batch.size(0)
                    
                # Accumulate epoch losses
                epoch_train_loss += loss_dict["loss"].item()
                epoch_train_recon += loss_dict["recon_loglikelihood"].item()
                epoch_train_kl += loss_dict["kl_divergence"].item()
                num_samples += bs
                
                # Log training metrics per iteration
                writer.add_scalar('Train/loss_step', loss_dict["loss"].item(), global_step)
                writer.add_scalar('Train/recon_loglikelihood_step', loss_dict["recon_loglikelihood"].item(), global_step)
                writer.add_scalar('Train/kl_divergence_step', loss_dict["kl_divergence"].item(), global_step)
                
                # Log gradients and parameters periodically
                if global_step % 100 == 0:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                            writer.add_scalar(f'Gradients_norm/{name}', param.grad.norm().item(), global_step)
                        writer.add_histogram(f'Parameters/{name}', param, global_step)

            # Compute average epoch losses
            avg_train_loss = epoch_train_loss / num_samples
            avg_train_recon = epoch_train_recon / num_samples
            avg_train_kl = epoch_train_kl / num_samples
            
            # Log epoch-level training metrics
            writer.add_scalar('Train/loss_epoch', avg_train_loss, epoch)
            writer.add_scalar('Train/recon_loglikelihood_epoch', avg_train_recon, epoch)
            writer.add_scalar('Train/kl_divergence_epoch', avg_train_kl, epoch)
            
            # Log learning rate
            current_lr = optim.param_groups[0]['lr']
            writer.add_scalar('Train/learning_rate', current_lr, epoch)

            # === Validation Phase ===
            val_loss_dict = None
            if valloader is not None and ((epoch + 1) % validation_freq == 0):
                val_loss_dict = self.validate(valloader, num_samples=validation_num_samples, device=device)
                
                # Log validation metrics
                writer.add_scalar('Val/Epoch/loss', val_loss_dict["loss"], epoch)
                writer.add_scalar('Val/Epoch/recon_loglikelihood', val_loss_dict["recon_loglikelihood"], epoch)
                writer.add_scalar('Val/Epoch/kl_divergence', val_loss_dict["kl_divergence"], epoch)
                
                # Log train vs val comparison
                writer.add_scalars('Comparison/loss', {
                    'train': avg_train_loss,
                    'val': val_loss_dict["loss"]
                }, epoch)
                writer.add_scalars('Comparison/recon', {
                    'train': avg_train_recon,
                    'val': val_loss_dict["recon_loglikelihood"]
                }, epoch)
                writer.add_scalars('Comparison/kl', {
                    'train': avg_train_kl,
                    'val': val_loss_dict["kl_divergence"]
                }, epoch)
                
                 # === Save Best Model ===
                if val_loss_dict["loss"] < best_val_loss:
                    best_val_loss = val_loss_dict["loss"]
                    best_epoch = epoch
                    best_model_path = os.path.join(save_dir, f"{model_tag}_best.pt")
                    self.save_model(save_path=save_dir, model_name=f"{model_tag}_best", device=device)
                    print(f"    New best model saved (Val Loss: {best_val_loss:.4f})")
                    writer.add_text('Training/best_model', f'New best at epoch {epoch+1}, Val Loss: {best_val_loss:.4f}', epoch)

                # Print progress
                print(f"Epoch [{epoch+1:4d}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss_dict['loss']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Best: {best_val_loss:.4f}@{best_epoch+1}")

                # Early stopping check
                if early_stopping:
                    is_best = earlystopping(val_metric=val_loss_dict["loss"],train_metric=avg_train_loss, epoch=epoch)
                    if earlystopping.early_stop:
                        print(f"\n{'='*60}")
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        print(f"Best model at epoch {best_epoch+1} with Val Loss: {best_val_loss:.4f}")
                        print(f"{'='*60}\n")
                        writer.add_text('Training/early_stop', 
                                      f'Stopped at epoch {epoch+1}, Best: {best_epoch+1}', epoch)
                        break
            else:
                # No validation - just print training progress
                print(f"Epoch [{epoch+1:4d}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Recon: {avg_train_recon:.4f} | "
                      f"KL: {avg_train_kl:.4f} | "
                      f"LR: {current_lr:.2e}")
                
            # === Learning Rate Scheduler Step ===
            if lr_scheduler:
                scheduler.step()
            
            # Save Model Checkpoint
            if (epoch + 1) % save_epoch_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"{model_tag}_epoch_{epoch+1}.pt")
                self.save_model(save_path=save_dir, model_name=f"{model_tag}_epoch_{epoch+1}", device=device)
                writer.add_text('Training/checkpoint', f'Checkpoint saved at epoch {epoch+1}', epoch)
                print(f"    Checkpoint saved: epoch_{epoch+1}")
        
        # === Training Complete ===
        print(f"\n{'='*60}")
        print(f"Training complete: {model_tag}")
        print(f"Best model: Epoch {best_epoch+1} with Val Loss: {best_val_loss:.4f}")
        print(f"Model saved to: {os.path.join(save_dir, model_tag)}")
        print(f"{'='*60}\n")

        final_model_path = os.path.join(save_dir, f"{model_tag}_final.pt")
        self.to("cpu")
        self.prior.to("cpu")
        self.save_model(save_path=save_dir, model_name=f"{model_tag}_final", device="cpu")

        # Save training summary
        summary = {
            'model_tag': model_tag,
            'total_epochs': epoch + 1,
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'early_stopped': early_stopping and earlystopping.early_stop,
        }
        summary_path = os.path.join(save_dir, f"{model_tag}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Log final summary to TensorBoard
        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
        writer.add_text('Training/summary', summary_text, epoch)
        
        self.eval()
        
        # Close writer
        if writer is not None:
            writer.close()
            print(f"TensorBoard logs saved to: {log_dir}")
        return model_tag
    
    def get_SigmaMapper(self):
        return self.decoder.get_SigmaMapper()


# ==============
# Define the Conditional VAE (CVAE) class
# ==============
class CVAE(VAE):
    def __init__(
        self,
        input_dim: int,
        *,
        conditioner: Conditioner,
        latent_dim: int = 10,
        distribution_dict: dict = None,
        prior_type: str = None,
        **kwargs,
    ):
        if distribution_dict is None:
            distribution_dict = {}
        distribution_dict = dict(distribution_dict)
        distribution_dict["latent_dim"] = latent_dim

        self.conditioner = conditioner
        self.condition_dim = conditioner.get_condition_dim()
        self.latent_dim = int(latent_dim)

        # Pass prior_type to parent so it creates the prior
        super().__init__(input_dim=input_dim, distribution_dict=distribution_dict, prior_type=prior_type, **kwargs)

        # NOW override encoder and decoder with conditioned versions
        # Encoder takes [x, c] as input
        self.encoder = get_encoder_model(
            input_dim=input_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("coder_type", "dictionary"),
            prior=self.prior,
            **kwargs,
        )

        # Decoder takes [z, c] as input
        self.decoder = get_decoder_model(
            latent_dim=latent_dim + self.condition_dim,
            output_dim=input_dim,
            type=kwargs.get("coder_type", "dictionary"),
            prior=self.prior,
            **kwargs,
        )

        # Recalculate num_parameters after overriding encoder/decoder
        self.num_parameters = self.encoder._num_parameters() + self.decoder._num_parameters()

    def _prep_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """ Flatten [B, L, F] to [B, D] if needed """
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1)
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_samples: int = 1):
        """
        Forward pass with pre-computed conditions.
        
        Args:
            x: Input tensor [B, D] (already flattened)
            c: Condition tensor [B, C] (already computed)
            num_samples: Number of latent samples
        
        Returns:
            recon_params: dict with [S, B, D] tensors
            q_params: dict with [B, latent_dim] tensors
        """
        # Encode: [B, D+C] -> posterior params
        enc_in = torch.cat([x, c], dim=-1)
        q_params = self.encoder(enc_in) # dict of [B, latent_dim] params

        # Sample latent variables using reparameterization trick - sample z: [S, B, latent_dim]
        z = self.encoder.reparameterize(q_params, num_samples=num_samples)

        # expand c to match z: [S, B, C]
        c_expanded = c.unsqueeze(0).expand(num_samples, -1, -1)

        # Decode: [S, B, latent_dim + C] -> recon params
        dec_in = torch.cat([z, c_expanded], dim=-1)
        S, B, DC = dec_in.shape
        dec_in_flat = dec_in.reshape(S * B, DC)

        # Decode latent samples to get reconstruction parameters
        recon_params = self.decoder(dec_in_flat)   # dict of [S*B, D]
        
        # reshape recon params back to [S, B, D]
        for k in recon_params:
            recon_params[k] = recon_params[k].reshape(S, B, -1)

        return recon_params, q_params 

    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, **_):
        """Compute ELBO loss."""
        # recon_params: dict of [S, B, D] -> use first sample for training
        if recon_params["mu"].dim() == 3:
            recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
        else:
            recon_params_squeezed = recon_params

        return super().loss(x, recon_params_squeezed, posterior_params, beta=beta)

    def train_core(self, batch, optim, beta=1.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
        self.train()
        x, meter_id, timestamps = batch

        # Prep inputs: [B, L, F] -> [B, D] if needed
        x = self._prep_inputs(x)

        # Compute conditions
        c = self.conditioner.get_conditions(x, meter_id, timestamps)
        c = self.move_to_device(c, device=x.device)

        # Forward pass
        recon_params, q_params = self.forward(x, c, num_samples=1)

        # Backward pass
        optim.zero_grad(set_to_none=True)
        loss_dict = self.loss(x, recon_params, q_params, beta=beta)
        loss_dict["loss"].backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        n = 0

        for x, meter_id, timestamps in valloader:
            # Keep original 3D shape for conditioning
            x_3d = x.to(device)
            
            # Flatten for model
            x_flat = self._prep_inputs(x_3d)  # [B, D]

            # Get conditions - PASS 3D tensor
            c = self.conditioner.get_conditions(x_3d, meter_id=meter_id, timestamps=timestamps)
            c = self.move_to_device(c, device=device)

            # Forward pass
            recon_params, q_params = self.forward(x_flat, c, num_samples=1)
            loss_dict = self.loss(x_flat, recon_params, q_params, beta=beta)

            bs = x.size(0)
            for k in agg:
                agg[k] += float(loss_dict[k]) * bs
            n += bs

        for k in agg:
            agg[k] /= max(n, 1)
        return agg

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu",):
        self.eval()

        c = self.move_to_device(conditions, device=device)
        B = c.size(0)

        # Get prior params and expand to batch
        prior_params = self.prior.get_params()
        prior_params = {k: v.expand(B, -1).to(device) for k, v in prior_params.items()}

        # Sample z: [S, B, latent_dim]
        z = self.encoder.sample(prior_params, num_samples=num_prior_samples)  # [S, B, latent_dim]

        # Expand c to [S, B, C]
        c_expanded = c.unsqueeze(0).expand(num_prior_samples, B, -1)

        # decode 
        dec_in = torch.cat([z, c_expanded], dim=-1)  # [S, B, latent_dim + C]
        S, B, DC = dec_in.shape
        dec_in_flat = dec_in.reshape(S * B, DC)

        recon_params = self.decoder(dec_in_flat)  # dict [S*B, D]
        for k in recon_params:
            recon_params[k] = recon_params[k].reshape(S, B, -1)

        # sample x from decoder for each prior sample
        xs_all = []
        for s in range(S):
            params_s = {k: v[s] for k, v in recon_params.items()}         # [B, D]
            xs = self.decoder.sample(params_s, num_samples=num_x_samples) # [num_x_samples, B, D]
            xs_all.append(xs)

        return torch.stack(xs_all, dim=0)  # [S, num_x_samples, B, D]
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, meter_ids: torch.Tensor, timestamps: torch.Tensor, num_samples: int = 1):
        self.eval()
        x = self._prep_inputs(x)
        c = self.conditioner.get_conditions(x, meter_id=meter_ids, timestamps=timestamps)
        c = self.move_to_device(c, device=x.device)
        return self.forward(x, c, num_samples=num_samples)