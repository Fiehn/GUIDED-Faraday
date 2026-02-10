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
# - Optimize data loading and preprocessing pipeline
# - Add more conditions
# - Hyperparameter tuning utilities (for optimizers, architectures, etc.)
# - Make sure the model works with variable length sequences (padding/masking)
# - IMPORTANT: Streaming algorithm for user profiles (online LDA) and normalization (running mean/std) and for the training (online/streaming VAE training)
# - Test different loss functions and double check the current implementation against the literature
# - Make sure loading and saving is robust


# ==============
# Cyclical Beta Annealing Scheduler
# Reference: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
# (Fu et al., 2019) https://arxiv.org/abs/1903.10145
# ==============
class CyclicalBetaScheduler:
    """Cyclical annealing schedule for the KL weight (beta) in VAE training.

    Each cycle consists of a linear warmup from 0 to *beta_max* followed by a
    constant phase at *beta_max*.  The proportion of each cycle spent warming
    up is controlled by *ratio* (default 0.5 = half warmup, half constant).

    Parameters
    ----------
    total_steps : int
        Total number of training steps (epochs * batches_per_epoch).
    n_cycles : int
        Number of full annealing cycles over training (default 4).
    ratio : float
        Fraction of each cycle devoted to the linear warmup (0-1, default 0.5).
    beta_max : float
        Maximum beta value reached at the top of each cycle (default 1.0).
    """

    def __init__(self, total_steps: int, n_cycles: int = 4, ratio: float = 0.5, beta_max: float = 1.0):
        assert total_steps > 0, "total_steps must be > 0"
        assert n_cycles >= 1, "n_cycles must be >= 1"
        assert 0.0 < ratio <= 1.0, "ratio must be in (0, 1]"
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.beta_max = beta_max
        self._step = 0

    def step(self) -> float:
        """Advance one step and return the current beta value."""
        beta = self._get_beta(self._step)
        self._step += 1
        return beta

    def _get_beta(self, step: int) -> float:
        """Compute beta for a given global step."""
        # How far through training are we (0..1)
        tau = step / self.total_steps
        # Which cycle are we in, and where inside that cycle (0..1)
        cycle_progress = (tau * self.n_cycles) % 1.0
        if cycle_progress < self.ratio:
            # Linear warmup phase
            return self.beta_max * (cycle_progress / self.ratio)
        else:
            # Constant phase
            return self.beta_max

    def state_dict(self):
        return {"step": self._step, "total_steps": self.total_steps,
                "n_cycles": self.n_cycles, "ratio": self.ratio, "beta_max": self.beta_max}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(d["total_steps"], d["n_cycles"], d["ratio"], d["beta_max"])
        obj._step = d["step"]
        return obj

    def __repr__(self):
        return (f"CyclicalBetaScheduler(total_steps={self.total_steps}, n_cycles={self.n_cycles}, "
                f"ratio={self.ratio}, beta_max={self.beta_max}, current_step={self._step})")


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
#   - encoder and decoder are modular and can be swapped out
# ==============
class VAE(nn.Module):
    def __init__(self,
                 input_dim=24,
                 distribution_dict = {},
                 encoder_type = "dictionary",
                 decoder_type = "dictionary",
                 prior_type = None,
                 device="cpu",
                 **_):
        super(VAE, self).__init__()

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
        
        latent_dim = distribution_dict.get("latent_dim", 32)
        self.latent_dim = latent_dim

        # 1. Create encoder and decoder
        self.encoder = get_encoder_model(input_dim=input_dim, latent_dim=latent_dim, type=encoder_type)
        self.decoder = get_decoder_model(latent_dim=latent_dim, output_dim=input_dim, type=decoder_type)

        # 2. Create prior after encoder (VampPrior needs a reference to the encoder)
        if prior_type is None:
            prior_type = "standard_dictionary" if encoder_type == "dictionary" else "standard_gaussian"
            self.prior_type = prior_type
        
        self.prior = get_prior(
            prior_type=prior_type,
            latent_dim=latent_dim,
            device=self.device,
            dictionary_size=distribution_dict.get("dictionary_size", 100),
            input_shape=input_dim,
            encoder=self.encoder,
            num_components=distribution_dict.get("vamp_num_components", 20),
        )

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
        return self.decoder.loglikelihood(inputs, recon_params_dict)
    
    @torch.no_grad()
    def loglikelihood(self, inputs, num_mc_samples=1):
        """Compute log-likelihood"""
        recon_params_dict, posterior_params_dict = self.forward(inputs, num_samples=num_mc_samples)

        # Reconstruction log-likelihood
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)
        posterior_loglikelihood = self.encoder.loglikelihood(posterior_params_dict)
        prior_loglikelihood = self.prior.log_prob(posterior_params_dict['z'])
        
        return (posterior_loglikelihood.sum(-1) + recon_loglikelihood.sum(-1) - prior_loglikelihood.sum(-1)).mean(dim=0)
    
    def loss(self, inputs, recon_params_dict, posterior_params_dict, beta: float = 1.0, weighted_loss: float = 0.0, **_):
        # Compute reconstruction log-likelihood
        recon_loglikelihood = self.reconstruction_loglikelihood(inputs, recon_params_dict)

        # Compute KL divergence between posterior and prior
        kl_div = self.encoder.kl_divergence(posterior_params_dict, self.prior)
        kl_div = kl_div.mean()

        # Weighted loss (set weighted_loss > 0 to enable) - gives more weight to samples with higher absolute values, which should help with the sparsity issue in smart meter data
        weights = 1.0 + weighted_loss * torch.abs(inputs).mean(dim=-1)
        recon_loglikelihood = (recon_loglikelihood * weights).mean() 

        loss = -(recon_loglikelihood - beta * kl_div)

        # TODO: MMD (mean maximum discrepancy) has been shown to be useful for smart meter data by one of the papers
        return {"loss": loss, "recon_loglikelihood": recon_loglikelihood, "kl_divergence": kl_div}

    def train_core(self, x, optim, gradient_clip=None, gradient_clip_kwargs=None, beta=1.0, weighted_loss =0.0, **_):
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
                loss_dict = self.loss(batch, recon_params_dict, posterior_params_dict, beta=beta, prior_params=prior_params)
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
        # TODO: Allow for different optimizers and parameter settings
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) # TODO: Parameters should be inputted

    def move_to_device(self, batch, device):
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

    def save_model(self, save_path=None, model_name="trained_model", device="cpu"):
        """
        Save model with complete reconstruction information.
        Stores all parameters needed to recreate the model from scratch.
        """
        if save_path is None: 
            save_path = "./models/"
        model_path = os.path.join(save_path, f"{model_name}.pt")
        self.eval()
        self.to("cpu")
        self.prior.to("cpu")

        # === Core Model Information ===
        model_info = {
            'model_class': self.__class__.__name__,
            'model_module': self.__class__.__module__,
            'input_dim': self.input_dim,
        }

        # === Constructor Arguments for VAE ===
        constructor_args = {
            'input_dim': self.input_dim,
            'encoder_type': getattr(self, 'encoder_type', 'dictionary'),
            'decoder_type': getattr(self, 'decoder_type', 'dictionary'), 
            'prior_type': getattr(self, 'prior_type', type(self.prior).__name__.lower().replace('prior', '')),
            'device': str(self.device),
        }

        # === Distribution Dictionary ===
        distribution_dict = {
            'latent_dim': getattr(self, 'latent_dim', getattr(self.encoder, 'latent_dim', 32)),
        }
        
        # Extract distribution parameters from model attributes and model_kwargs
        for key in ['dictionary_size', 'vamp_num_components', 'num_topics', 'sigma_min', 'sigma_max']:
            if hasattr(self, key):
                distribution_dict[key] = getattr(self, key)
            elif key in self.model_kwargs:
                distribution_dict[key] = self.model_kwargs[key]

        # Add latent_dim to distribution_dict
        constructor_args['distribution_dict'] = distribution_dict

        # === CVAE Specific Information ===
        if isinstance(self, CVAE):
            model_info['condition_dim'] = self.condition_dim
            constructor_args['latent_dim'] = self.latent_dim
            
            # Save conditioner information
            conditioner_state = {
                'condition_items': {},
                'condition_tags': self.conditioner.get_condition_tags(),
                'condition_dim': self.conditioner.get_condition_dim(),
            }
            
            # Save each condition item
            for tag, condition_item in self.conditioner.condition_items.items():
                item_state = {
                    'class_name': condition_item.__class__.__name__,
                    'module': condition_item.__class__.__module__,
                    'state_dict': condition_item.state_dict() if hasattr(condition_item, 'state_dict') else {},
                }
                
                # Save constructor arguments for condition items - use specific logic per class
                item_args = {}
                class_name = condition_item.__class__.__name__
                
                if class_name == 'DayOfWeekConditioner':
                    # DayOfWeekConditioner takes no constructor arguments except self
                    pass
                elif class_name == 'UserProfiler':
                    # UserProfiler constructor args
                    for attr in ['num_clusters', 'path', 'verbose']:
                        if hasattr(condition_item, attr):
                            item_args[attr] = getattr(condition_item, attr)
                else:
                    # Generic fallback for unknown condition types - be more selective
                    for attr_name in ['num_clusters', 'path', 'verbose', 'input_dim', 'output_dim']:
                        if hasattr(condition_item, attr_name):
                            try:
                                attr = getattr(condition_item, attr_name)
                                if isinstance(attr, (int, float, str, bool)):
                                    item_args[attr_name] = attr
                            except:
                                pass
                
                item_state['constructor_args'] = item_args
                conditioner_state['condition_items'][tag] = item_state
                
            model_info['conditioner_state'] = conditioner_state

        # === Prior Serialization ===
        prior_state = {
            'prior_type': self.prior.__class__.__name__,
            'module': self.prior.__class__.__module__,
            'state_dict': self.prior.state_dict(),
            'init_args': {},
        }
        
        # Extract prior initialization arguments - use specific logic per class
        prior_class_name = self.prior.__class__.__name__
        
        if prior_class_name == 'VampPrior':
            # VampPrior constructor: (latent_dim, input_shape, encoder=None, num_components=20, device='cpu')
            prior_state['init_args'] = {
                'latent_dim': getattr(self.prior, 'latent_dim', self.latent_dim),
                'input_shape': getattr(self.prior, 'input_shape', self.input_dim + getattr(self, 'condition_dim', 0)),
                'num_components': getattr(self.prior, 'num_components', 20),
            }
        elif prior_class_name in ['StandardGaussianPrior', 'LearnableGaussianPrior']:
            # Gaussian priors: (latent_dim, device)
            prior_state['init_args'] = {
                'latent_dim': getattr(self.prior, 'latent_dim', self.latent_dim),
            }
        elif prior_class_name in ['StandardDictionaryPrior', 'LearnableDictionaryPrior']:
            # Dictionary priors: (latent_dim, dictionary_size, device)
            prior_state['init_args'] = {
                'latent_dim': getattr(self.prior, 'latent_dim', self.latent_dim),
                'dictionary_size': getattr(self.prior, 'dictionary_size', 100),
            }
        else:
            # Generic fallback for unknown prior types
            for attr_name in ['latent_dim', 'dictionary_size', 'input_shape', 'num_components']:
                if hasattr(self.prior, attr_name):
                    try:
                        attr = getattr(self.prior, attr_name)
                        if isinstance(attr, (int, float, str, bool)):
                            prior_state['init_args'][attr_name] = attr
                        elif isinstance(attr, torch.dtype):
                            prior_state['init_args'][attr_name] = str(attr)
                    except:
                        pass

        # === Save Everything ===
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': model_info,
            'constructor_args': constructor_args,
            'prior_state': prior_state,
            'model_kwargs': self.model_kwargs,  # Keep for backward compatibility
        }

        torch.save(checkpoint, model_path)
        self.to(device)
        self.prior.to(device)
        print(f"Model saved to {model_path}")
        print(f"Model class: {model_info['model_class']}")
        if isinstance(self, CVAE):
            print(f"Conditioner tags: {conditioner_state['condition_tags']}")

    def load_model(self, load_path, device="cpu"):
        """
        Load model with complete reconstruction from saved information.
        Works with both old and new save formats.
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Handle new format with complete model info
        if 'model_info' in checkpoint:
            print(f"Loading model using new format: {checkpoint['model_info']['model_class']}")
            
            # Load prior
            prior_state = checkpoint['prior_state']
            import importlib
            module = importlib.import_module(prior_state['module'])
            PriorClass = getattr(module, prior_state['prior_type'])
            init_args = prior_state.get('init_args', {})
            
            try:
                self.prior = PriorClass(**init_args)
                self.prior.load_state_dict(prior_state['state_dict'])
                
                # Special handling for VampPrior - set encoder reference after loading
                if prior_state['prior_type'] == 'VampPrior':
                    self.prior.set_encoder(self.encoder, init_args.get('input_shape', self.input_dim))
                    
            except Exception as e:
                print(f"Error loading prior {prior_state['prior_type']}: {e}")

            # Load CVAE conditioner if present
            if isinstance(self, CVAE) and 'conditioner_state' in checkpoint['model_info']:
                conditioner_state = checkpoint['model_info']['conditioner_state']
                print(f"Reconstructing conditioner with tags: {conditioner_state['condition_tags']}")
                
                from src.Conditioner import Conditioner, DayOfWeekConditioner, UserProfiler
                self.conditioner = Conditioner()
                
                for tag, item_state in conditioner_state['condition_items'].items():
                    # Import the condition item class
                    item_module = importlib.import_module(item_state['module'])
                    ItemClass = getattr(item_module, item_state['class_name'])
                    
                    # Create instance with saved constructor args
                    constructor_args = item_state.get('constructor_args', {})
                    try:
                        condition_item = ItemClass(**constructor_args)
                        # Load state if available
                        if item_state['state_dict']:
                            condition_item.load_state_dict(item_state['state_dict'])
                        self.conditioner.add_condition_item(tag, condition_item)
                        print(f"  Reconstructed: {tag} ({item_state['class_name']})")
                    except Exception as e:
                        print(f"  Error reconstructing {tag}: {e}")
                
                self.condition_dim = conditioner_state['condition_dim']

        else:
            # Handle old format (backward compatibility)
            print("Loading model using legacy format")
            prior_state = checkpoint['prior_state']
            import importlib
            module = importlib.import_module(prior_state['module'])
            PriorClass = getattr(module, prior_state['prior_type'])
            init_args = prior_state.get('init_args', {})
            
            try:
                self.prior = PriorClass(**init_args)
                self.prior.load_state_dict(prior_state['state_dict'])
            except Exception as e:
                print(f"Error initializing prior {prior_state['prior_type']} with args {init_args}: {e}")

        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        self.prior.to(device)
        print(f"Model loaded from {load_path}")

    @classmethod
    def load_from_checkpoint(cls, load_path, device="cpu"):
        """
        Load a model instance completely from a saved checkpoint.
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        if 'model_info' not in checkpoint:
            raise ValueError("Checkpoint was saved with legacy format. Use load_model() on an existing instance instead.")
        
        model_info = checkpoint['model_info']
        constructor_args = checkpoint['constructor_args']
        
        print(f"Creating {model_info['model_class']} from checkpoint...")
        
        # Handle CVAE reconstruction
        if model_info['model_class'] == 'CVAE':
            # Reconstruct conditioner first
            conditioner_state = model_info['conditioner_state']
            
            import importlib
            from src.Conditioner import Conditioner
            conditioner = Conditioner()
            
            print(f"Reconstructing conditioner with {len(conditioner_state['condition_items'])} items...")
            for tag, item_state in conditioner_state['condition_items'].items():
                # Import and create condition item
                item_module = importlib.import_module(item_state['module'])
                ItemClass = getattr(item_module, item_state['class_name'])
                
                constructor_args_item = item_state.get('constructor_args', {})
                try:
                    condition_item = ItemClass(**constructor_args_item)
                    if item_state['state_dict']:
                        condition_item.load_state_dict(item_state['state_dict'])
                    conditioner.add_condition_item(tag, condition_item)
                    print(f"  {tag}: {item_state['class_name']}")
                except Exception as e:
                    print(f"  {tag}: {e}")
            
            # Create CVAE instance
            constructor_args['conditioner'] = conditioner
            model = cls(**constructor_args)
            
        else:
            # Create VAE instance
            model = cls(**constructor_args)
        
        # Load the saved state - need to handle VampPrior specially
        if 'prior_state' in checkpoint and checkpoint['prior_state']['prior_type'] == 'VampPrior':
            # For VampPrior, we need to recreate it with encoder reference after model is created
            prior_state = checkpoint['prior_state']
            init_args = prior_state.get('init_args', {})
            
            import importlib
            from src.priors import VampPrior
            
            # Create VampPrior with encoder reference 
            model.prior = VampPrior(
                latent_dim=init_args.get('latent_dim', model.latent_dim),
                input_shape=init_args.get('input_shape', model.input_dim + getattr(model, 'condition_dim', 0)),
                encoder=model.encoder,
                num_components=init_args.get('num_components', 20),
                device=device
            )
            
            # Load the state dict
            try:
                model.prior.load_state_dict(prior_state['state_dict'])
            except Exception as e:
                print(f"Warning: Could not load VampPrior state_dict: {e}")
            
            # Move to device
            model.to(device)
            model.prior.to(device)
            
            # Load the rest of the model (excluding the prior since we handled it above)
            model_state_without_prior = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                       if not k.startswith('prior.')}
            model.load_state_dict(model_state_without_prior, strict=False)
        else:
            # Normal loading for other prior types
            model.load_model(load_path, device)
            
        return model

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
            beta_annealing = False,  # Enable cyclical beta annealing to avoid posterior collapse
            beta_annealing_kwargs = {"n_cycles": 4, "ratio": 0.5},  # kwargs for CyclicalBetaScheduler
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
            model_name = None, # Optionally specify model name for saving
            save_dir = './models/',
            weighted_loss = 0.0, # Set > 0 to give more weight to samples with higher absolute values, which should help with the sparsity issue in smart meter data
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
                'beta_annealing': beta_annealing,
                'beta_annealing_kwargs': beta_annealing_kwargs if beta_annealing else None,
                'epochs': epochs,
                'batch_size': trainloader.batch_size,
                'gradient_clip': gradient_clip,
                'lr_scheduler': lr_scheduler,
                'early_stopping': early_stopping,
                'weighted_loss': weighted_loss,
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

        # === Beta Annealing Scheduler ===
        if beta_annealing:
            total_steps = epochs * len(trainloader)
            beta_scheduler = CyclicalBetaScheduler(
                total_steps=total_steps,
                beta_max=beta,  # use the provided beta as the ceiling
                **beta_annealing_kwargs,
            )
            print(f"Cyclical beta annealing enabled: {beta_scheduler}")

        # === Training Loop ===
        self.train()
        global_step = 0
        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_recon = 0.0
            epoch_train_kl = 0.0
            num_batches = 0

            for batch in trainloader:
                global_step += 1

                # Get current beta (annealed or fixed)
                current_beta = beta_scheduler.step() if beta_annealing else beta

                batch = self.move_to_device(batch, device)
                loss_dict = self.train_core(
                    batch, 
                    optim, 
                    beta=current_beta, 
                    gradient_clip=gradient_clip, 
                    gradient_clip_kwargs=gradient_clip_kwargs,
                    weighted_loss=weighted_loss
                )
                
                # Accumulate epoch losses
                epoch_train_loss += loss_dict["loss"].item()
                epoch_train_recon += loss_dict["recon_loglikelihood"].item()
                epoch_train_kl += loss_dict["kl_divergence"].item()
                num_batches += 1
                
                # Log training metrics per iteration
                writer.add_scalar('Train/loss_step', loss_dict["loss"].item(), global_step)
                writer.add_scalar('Train/recon_loglikelihood_step', loss_dict["recon_loglikelihood"].item(), global_step)
                writer.add_scalar('Train/kl_divergence_step', loss_dict["kl_divergence"].item(), global_step)
                writer.add_scalar('Train/beta', current_beta, global_step)

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
        latent_dim: int = 32,
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

        super().__init__(input_dim=input_dim, distribution_dict=distribution_dict, prior_type=prior_type, **kwargs)

        # Now override encoder and decoder with conditioned versions
        # Encoder takes [x, c] as input
        self.encoder = get_encoder_model(
            input_dim=input_dim + self.condition_dim,
            latent_dim=latent_dim,
            type=kwargs.get("encoder_type", "dictionary"),
            **kwargs,
        )

        # Decoder takes [z, c] as input
        self.decoder = get_decoder_model(
            latent_dim=latent_dim + self.condition_dim,
            output_dim=input_dim,
            type=kwargs.get("decoder_type", "dictionary"),
            **kwargs,
        )

        # If VampPrior, re-point it to the new (conditioned) encoder and resize pseudo-inputs
        if isinstance(self.prior, VampPrior):
            self.prior.set_encoder(self.encoder, input_shape=input_dim + self.condition_dim)

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

    def loss(self, x: torch.Tensor, recon_params, posterior_params, beta=1.0, weighted_loss=0.0,**_):
        """Compute ELBO loss."""
        # recon_params: dict of [S, B, D] -> use first sample for training
        if recon_params["mu"].dim() == 3:
            recon_params_squeezed = {k: v[0] for k, v in recon_params.items()}
        else:
            recon_params_squeezed = recon_params

        return super().loss(x, recon_params_squeezed, posterior_params, beta=beta, weighted_loss=weighted_loss)

    def train_core(self, batch, optim, beta=1.0,weighted_loss=0.0, gradient_clip=None, gradient_clip_kwargs=None, **_):
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
        loss_dict = self.loss(x, recon_params, q_params, beta=beta, weighted_loss=weighted_loss)
        loss_dict["loss"].backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

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

            for k in agg:
                agg[k] += float(loss_dict[k])
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, num_prior_samples: int = 1, num_x_samples: int = 1, device: str = "cpu",):
        self.eval()

        c = self.move_to_device(conditions, device=device)
        B = c.size(0)

        # Sample z from prior: handle different prior return shapes
        z = self.prior.sample(batch_size=B, num_samples=num_prior_samples)
        
        # Ensure z has the correct shape [S, B, latent_dim]
        if z.dim() == 3:
            if z.shape[0] == B and z.shape[1] == num_prior_samples:
                # Shape is [B, S, latent_dim] - transpose to [S, B, latent_dim]
                z = z.transpose(0, 1)
            elif z.shape[0] == num_prior_samples and z.shape[1] == B:
                # Shape is already [S, B, latent_dim]
                pass
            else:
                # Unexpected shape - try to reshape
                print(f"Warning: Unexpected prior sample shape {z.shape}, attempting to reshape...")
                z = z.view(num_prior_samples, B, -1)
        elif z.dim() == 2:
            # Shape is [B, latent_dim] - add sample dimension
            z = z.unsqueeze(0).expand(num_prior_samples, B, -1)
        else:
            raise ValueError(f"Unexpected prior sample tensor dimensions: {z.shape}")

        # Expand c to [S, B, C]
        c_expanded = c.unsqueeze(0).expand(num_prior_samples, B, -1)
        
        # Verify shapes before concatenation
        if z.shape[:2] != c_expanded.shape[:2]:
            raise ValueError(f"Shape mismatch: z.shape={z.shape}, c_expanded.shape={c_expanded.shape}")

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