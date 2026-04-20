import torch
import torch.nn as nn
from datetime import datetime
import json
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 
from model.encoder_decoder_model import *
from data.conditioner import *
from model.priors import *
from data.normalizer import *

from visualization.writer import Writer
 
from src.model.hierarchical.config import HierarchicalConfig
from src.model.hierarchical.core import HierarchicalLatentModule

from src.model.model_utils import CyclicalBetaScheduler, EarlyStopping

from model.model import BaseModel


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

        # TODO: MMD (mean maximum discrepancy) has been shown to be useful for smart meter data by one of the papers
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
        # TODO: Allow for different optimizers and parameter settings
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) # TODO: Parameters should be inputted

    def save_model(self, save_path=None, model_name="trained_model", device="cpu", normalizer=None):
        """
        Save model with complete reconstruction information and validation.
        Stores all parameters needed to recreate the model from scratch.
        Includes error recovery and integrity checking.
        """
        if save_path is None: 
            save_path = "./models/"
        
        model_path = os.path.join(save_path, f"{model_name}.pt")
        os.makedirs(save_path, exist_ok=True)
        
        # Move to CPU for serialization
        self.eval()
        self.to("cpu")
        if hasattr(self, 'prior'):
            self.prior.to("cpu")
        
        try:
            # === Core Model Information ===
            model_info = {
                'model_class': self.__class__.__name__,
                'model_module': self.__class__.__module__,
                'input_dim': self.input_dim,
                'save_format_version': 3,  # Track format for future compatibility (normalizer added in v3)
            }

            # === Constructor Arguments ===
            constructor_args = self._extract_constructor_args()
            
            # === Distribution Dictionary ===
            distribution_dict = self._extract_distribution_dict()
            constructor_args['distribution_dict'] = distribution_dict

            # === CVAE-Specific Information ===
            if hasattr(self, 'conditioner') and self.conditioner is not None:
                model_info = self._serialize_cvae_info(model_info)

            # === Prior Serialization ===
            prior_state = self._serialize_prior()

            # === Normalizer Serialization ===
            normalizer_state = None
            if normalizer is not None and hasattr(normalizer, 'fitted') and normalizer.fitted:
                normalizer_state = {
                    'method': normalizer.method,
                    'epsilon': normalizer.epsilon,
                    'zero_id': normalizer.zero_id,
                    'shift': normalizer.shift,
                    'non_zero_mean': normalizer.non_zero_mean,
                    'non_zero_std': normalizer.non_zero_std,
                    'mean': normalizer.mean,
                    'std': normalizer.std,
                }

            # === Build Checkpoint ===
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'model_info': model_info,
                'constructor_args': constructor_args,
                'prior_state': prior_state,
                'model_kwargs': self.model_kwargs,
                'normalizer_state': normalizer_state,
            }

            # === Validate Before Saving ===
            self._validate_checkpoint(checkpoint)

            # === Save to Temp File First (Atomic Write) ===
            temp_path = model_path + ".tmp"
            torch.save(checkpoint, temp_path)
            
            # === Verify Temp File Can Be Loaded ===
            try:
                test_load = torch.load(temp_path, map_location="cpu", weights_only=False)
                assert 'model_state_dict' in test_load, "Checkpoint missing model_state_dict"
            except Exception as e:
                os.remove(temp_path)
                raise RuntimeError(f"Checkpoint validation failed: {e}")

            # === Atomic Move (Replace Old File) ===
            if os.path.exists(model_path):
                backup_path = model_path + ".backup"
                os.replace(model_path, backup_path)
            os.replace(temp_path, model_path)

            print(f"✓ Model saved successfully: {model_path}")
            print(f"  - Model class: {model_info['model_class']}")
            print(f"  - Parameters: {self.num_parameters:,}")
            if self.__class__.__name__ == "CVAE":
                print(f"  - Condition dim: {self.condition_dim}")
                
        except Exception as e:
            # Cleanup temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"✗ Save failed: {e}", file=sys.stderr)
            raise
        finally:
            # Move back to original device
            self.to(device)
            if hasattr(self, 'prior'):
                self.prior.to(device)

    def _extract_constructor_args(self):
        """Safely extract and validate constructor arguments."""
        args =  {
            'input_dim': self.input_dim,
            'encoder_type': getattr(self, 'encoder_type', 'dictionary'),
            'decoder_type': getattr(self, 'decoder_type', 'dictionary'), 
            'prior_type': getattr(self, 'prior_type', self._infer_prior_type()),
            'device': str(self.device),
        }
        if self.hierarchy is not None:
            args['hierarchy_config'] = self.hierarchy.config.to_dict()
        else:
            args['hierarchy_config'] = None

        # Persist SetGuideVAE-specific architecture so checkpoints are fully reconstructable.
        if hasattr(self, 'sequence_length'):
            args['sequence_length'] = int(getattr(self, 'sequence_length'))
        if hasattr(self, 'num_features'):
            args['num_features'] = int(getattr(self, 'num_features'))
        if hasattr(self, 'set_hidden_dim'):
            args['set_hidden_dim'] = int(getattr(self, 'set_hidden_dim'))

        set_blocks = getattr(self, 'set_encoder_blocks', None)
        if set_blocks is not None:
            args['set_num_blocks'] = int(len(set_blocks))
            if len(set_blocks) > 0:
                first_block = set_blocks[0]
                if hasattr(first_block, 'mab0') and hasattr(first_block, 'mab1'):
                    args['is_isab'] = True
                    if hasattr(first_block.mab0, 'num_heads'):
                        args['set_num_heads'] = int(first_block.mab0.num_heads)
                    if hasattr(first_block, 'inducing') and torch.is_tensor(first_block.inducing):
                        args['set_num_inds'] = int(first_block.inducing.shape[1])
                elif hasattr(first_block, 'mab'):
                    args['is_isab'] = False
                    if hasattr(first_block.mab, 'num_heads'):
                        args['set_num_heads'] = int(first_block.mab.num_heads)

        set_pool = getattr(self, 'set_pool', None)
        args['use_abl_pool'] = bool(set_pool is not None)
        if set_pool is not None and hasattr(set_pool, 'mab') and hasattr(set_pool.mab, 'num_heads'):
            args['set_num_heads'] = int(set_pool.mab.num_heads)
        return args

    @staticmethod
    def _infer_setguide_constructor_args_from_state_dict(model_state_dict, input_dim=None):
        """Infer shape-critical SetGuideVAE args from checkpoint weights for backward compatibility."""
        inferred = {}
        if not isinstance(model_state_dict, dict) or len(model_state_dict) == 0:
            return inferred

        def _try_infer_num_features_from_input():
            input_dim_int = int(input_dim) if input_dim is not None else None
            seq_len = inferred.get('sequence_length', None)
            if input_dim_int is not None and seq_len is not None and seq_len > 0 and input_dim_int % seq_len == 0:
                inferred['num_features'] = int(input_dim_int // seq_len)

        # set_input_proj: [set_hidden_dim, sequence_length]
        set_input_proj_w = model_state_dict.get('set_input_proj.weight', None)
        if torch.is_tensor(set_input_proj_w) and set_input_proj_w.dim() == 2:
            inferred['set_hidden_dim'] = int(set_input_proj_w.shape[0])
            inferred['sequence_length'] = int(set_input_proj_w.shape[1])

        # New SetGuideVAE projection head: set_element_proj [set_hidden_dim, 32 * sequence_length]
        set_element_proj_w = model_state_dict.get('set_element_proj.weight', None)
        if torch.is_tensor(set_element_proj_w) and set_element_proj_w.dim() == 2:
            inferred['set_hidden_dim'] = int(set_element_proj_w.shape[0])
            in_dim = int(set_element_proj_w.shape[1])
            if in_dim % 32 == 0:
                inferred['sequence_length'] = int(in_dim // 32)

        try:
            _try_infer_num_features_from_input()
        except (TypeError, ValueError):
            pass

        # Number of set encoder blocks from key prefixes.
        block_ids = set()
        for key in model_state_dict.keys():
            if not key.startswith('set_encoder_blocks.'):
                continue
            parts = key.split('.')
            if len(parts) < 3:
                continue
            if parts[1].isdigit():
                block_ids.add(int(parts[1]))
        if block_ids:
            inferred['set_num_blocks'] = max(block_ids) + 1

        # Infer set block type from key patterns.
        has_isab_keys = any('.mab0.' in k or '.mab1.' in k or '.inducing' in k for k in model_state_dict.keys())
        has_sab_keys = any('.mab.' in k for k in model_state_dict.keys())
        if has_isab_keys and not has_sab_keys:
            inferred['is_isab'] = True
        elif has_sab_keys and not has_isab_keys:
            inferred['is_isab'] = False
        elif has_isab_keys:
            inferred['is_isab'] = True

        if inferred.get('is_isab', False):
            inducing = model_state_dict.get('set_encoder_blocks.0.inducing', None)
            if torch.is_tensor(inducing) and inducing.dim() >= 2:
                inferred['set_num_inds'] = int(inducing.shape[1])

        # Infer pooling mode: ABL checkpoints contain set_pool.* keys.
        inferred['use_abl_pool'] = any(k.startswith('set_pool.') for k in model_state_dict.keys())

        return inferred
        

    def _infer_prior_type(self):
        """Safely infer prior type from class name."""
        if not hasattr(self, 'prior'):
            return 'standard_gaussian'
        prior_class = self.prior.__class__.__name__
        return prior_class.lower().replace('prior', '').replace('standard_', '')

    def _extract_distribution_dict(self):
        """Extract distribution parameters with validation."""
        distribution_dict = {
            'latent_dim': getattr(self, 'latent_dim', 
                                getattr(self.encoder, 'latent_dim', 32)),
        }
        
        # Safe extraction of optional keys
        optional_keys = {
            'dictionary_size': int,
            'vamp_num_components': int,
            'num_topics': int,
            'sigma_min': float,
            'sigma_max': float,
        }
        
        for key, expected_type in optional_keys.items():
            value = None
            if hasattr(self, key):
                value = getattr(self, key)
            elif key in self.model_kwargs:
                value = self.model_kwargs[key]
            
            if value is not None:
                try:
                    distribution_dict[key] = expected_type(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {key}={value} to {expected_type.__name__}: {e}")
        
        return distribution_dict

    def _serialize_prior(self):
        """Serialize prior with fallback error handling."""
        if not hasattr(self, 'prior'):
            return {'prior_type': 'StandardGaussianPrior', 'init_args': {}, 'state_dict': {}}
        
        prior = self.prior
        prior_class_name = prior.__class__.__name__
        
        prior_state = {
            'prior_type': prior_class_name,
            'module': prior.__class__.__module__,
            'state_dict': prior.state_dict() if hasattr(prior, 'state_dict') else {},
            'init_args': {},
        }
        
        # Use isinstance checks instead of string matching
        try:
            if isinstance(prior, VampPrior):
                prior_state['init_args'] = {
                    'latent_dim': getattr(prior, 'latent_dim', self.latent_dim),
                    'input_shape': getattr(prior, 'input_shape', 
                                        self.input_dim + getattr(self, 'condition_dim', 0)),
                    'num_components': getattr(prior, 'num_components', 20),
                }
            elif isinstance(prior, (StandardGaussianPrior)):
                prior_state['init_args'] = {
                    'latent_dim': getattr(prior, 'latent_dim', self.latent_dim),
                }
            elif isinstance(prior, (StandardDictionaryPrior, LearnableDictionaryPrior)):
                prior_state['init_args'] = {
                    'latent_dim': getattr(prior, 'latent_dim', self.latent_dim),
                    'dictionary_size': getattr(prior, 'dictionary_size', 100),
                }
            else:
                # Generic fallback - extract any serializable attributes
                for attr_name in ['latent_dim', 'dictionary_size', 'input_shape', 'num_components']:
                    if hasattr(prior, attr_name):
                        val = getattr(prior, attr_name)
                        if isinstance(val, (int, float, str, bool)):
                            prior_state['init_args'][attr_name] = val
        except Exception as e:
            print(f"Warning: Error extracting prior args: {e}")
        
        return prior_state

    def _serialize_cvae_info(self, model_info):
        """Serialize CVAE-specific info with error handling."""
        model_info['condition_dim'] = self.condition_dim
        
        conditioner_state = {
            'condition_items': {},
            'condition_tags': self.conditioner.get_condition_tags(),
            'condition_dim': self.conditioner.get_condition_dim(),
        }
        
        for tag, condition_item in self.conditioner.condition_items.items():
            try:
                item_state = {
                    'class_name': condition_item.__class__.__name__,
                    'module': condition_item.__class__.__module__,
                    'state_dict': condition_item.state_dict() 
                                if hasattr(condition_item, 'state_dict') else {},
                    'constructor_args': self._extract_conditioner_args(condition_item),
                }
                conditioner_state['condition_items'][tag] = item_state
            except Exception as e:
                print(f"Warning: Could not serialize conditioner '{tag}': {e}")
                # Save minimal info for recovery
                conditioner_state['condition_items'][tag] = {
                    'class_name': condition_item.__class__.__name__,
                    'module': condition_item.__class__.__module__,
                    'error': str(e),
                }
        
        model_info['conditioner_state'] = conditioner_state
        return model_info

    def _extract_conditioner_args(self, condition_item):
        """Extract constructor args for a condition item."""
        item_args = {}
        class_name = condition_item.__class__.__name__
        
        # Use isinstance checks where possible
        if isinstance(condition_item, DayOfWeekConditioner):
            pass  # No constructor args needed
        elif isinstance(condition_item, UserProfiler):
            for attr in ['n_topics', 'num_clusters', 'random_state', 'path', 'verbose']:
                if hasattr(condition_item, attr):
                    item_args[attr] = getattr(condition_item, attr)
        else:
            # Generic extraction for unknown types
            safe_attrs = ['n_topics', 'num_clusters', 'path', 'verbose', 'input_dim', 'output_dim']
            for attr_name in safe_attrs:
                if hasattr(condition_item, attr_name):
                    try:
                        val = getattr(condition_item, attr_name)
                        if isinstance(val, (int, float, str, bool, type(None))):
                            item_args[attr_name] = val
                    except Exception:
                        pass
        
        return item_args

    def _validate_checkpoint(self, checkpoint):
        """Validate checkpoint structure and content."""
        required_keys = ['model_state_dict', 'model_info', 'constructor_args', 'prior_state']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")
        
        # Validate model_state_dict is non-empty
        if not checkpoint['model_state_dict']:
            raise ValueError("model_state_dict is empty")
        
        # Validate model_info
        info = checkpoint['model_info']
        if not isinstance(info.get('model_class'), str):
            raise ValueError("model_info['model_class'] must be a string")
        
        print(f"✓ Checkpoint validation passed")

    def load_model(self, load_path, device="cpu"):
        """
        Load model with complete reconstruction from saved information.
        Works with both old and new save formats.
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Handle new format with complete model info
        if 'model_info' in checkpoint:
            print(f"Loading model using new format: {checkpoint['model_info']['model_class']}")
            
            # Hierarchy config handling
            hier_raw = checkpoint.get('constructor_args', {}).get('hierarchy_config', None)
            if hier_raw is not None and isinstance(hier_raw, dict) and self.hierarchy is None:
                hier_cfg = HierarchicalConfig.from_dict(hier_raw)
                self.hierarchy = HierarchicalLatentModule(config=hier_cfg, encoder_latent_dim=self.latent_dim)
                self.decoder_input_dim = self.hierarchy.output_dim
            
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
            if self.__class__.__name__ == "CVAE" and 'conditioner_state' in checkpoint['model_info']:
                conditioner_state = checkpoint['model_info']['conditioner_state']
                print(f"Reconstructing conditioner with tags: {conditioner_state['condition_tags']}")
                
                from data.conditioner import Conditioner
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

        normalizer = self._load_normalizer_from_checkpoint(checkpoint=checkpoint, device="cpu")
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        self.prior.to(device)
        print(f"Model loaded from {load_path}")
        return normalizer
    
    @staticmethod
    def _load_normalizer_from_checkpoint(checkpoint, device="cpu"):
        """Load normalizer state from checkpoint if available."""
        normalizer_state = checkpoint.get('normalizer_state', None)
        if normalizer_state is not None:
            try:
                normalizer = Normalizer(method=normalizer_state['method'])
                if 'epsilon' in normalizer_state:
                    normalizer.epsilon = normalizer_state['epsilon']
                normalizer.zero_id = normalizer_state['zero_id']
                normalizer.shift = normalizer_state['shift']
                normalizer.non_zero_mean = normalizer_state['non_zero_mean']
                normalizer.non_zero_std = normalizer_state['non_zero_std']
                normalizer.mean = normalizer_state['mean']
                normalizer.std = normalizer_state['std']
                normalizer.fitted = True
                print(f"✓ Normalizer loaded from checkpoint")
                return normalizer
            except Exception as e:
                print(f"Warning: Could not load normalizer from checkpoint: {e}")
        return None

    @classmethod
    def load_from_checkpoint(cls, load_path, device="cpu", **kwargs):
        """
        Load a model instance completely from a saved checkpoint.
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        if 'model_info' not in checkpoint:
            raise ValueError("Checkpoint was saved with legacy format. Use load_model() on an existing instance instead.")
        
        model_info = checkpoint['model_info']
        constructor_args = dict(checkpoint.get('constructor_args', {}))
        model_class_name = model_info['model_class']

        if 'model_kwargs' in checkpoint:
            for k, v in checkpoint['model_kwargs'].items():
                if k not in constructor_args:
                    constructor_args[k] = v
                    
        for k, v in kwargs.items():
            if k not in ('device', 'conditioner') and k not in constructor_args:
                constructor_args[k] = v

        # Backward compatibility for older SetGuideVAE checkpoints that did not
        # persist set-encoder constructor arguments.
        if model_class_name == 'SetGuideVAE':
            inferred_set_args = cls._infer_setguide_constructor_args_from_state_dict(
                model_state_dict=checkpoint.get('model_state_dict', {}),
                input_dim=constructor_args.get('input_dim', None),
            )
            applied = {}
            for k, v in inferred_set_args.items():
                if k not in constructor_args or constructor_args[k] is None:
                    constructor_args[k] = v
                    applied[k] = v
            if applied:
                print(f"Inferred SetGuideVAE constructor args from checkpoint weights: {applied}")

        hier_raw = constructor_args.get('hierarchy_config', None)
        if hier_raw is not None and isinstance(hier_raw, dict):
            constructor_args['hierarchy_config'] = HierarchicalConfig.from_dict(hier_raw)
        else:
            constructor_args['hierarchy_config'] = None
        
        print(f"Creating {model_info['model_class']} from checkpoint...")
        
        # Handle CVAE reconstruction
        if model_info.get('model_class') in ("CVAE", "HierarchicalCVAE", "SetGuideVAE", "OnlineCVAE", "GuideVAE") or 'conditioner_state' in model_info:
            conditioner = kwargs.get('conditioner', None)
            
            if 'conditioner_state' in model_info:
                conditioner_state = model_info['conditioner_state']
                
                import importlib
                from data.conditioner import Conditioner
                if conditioner is None:
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
            
            if conditioner is None:
                print("Warning: Checkpoint is missing 'conditioner_state'. Must provide it via kwargs if the model requires it.")
            else:
                # Create CVAE instance
                constructor_args['conditioner'] = conditioner
                
            # CVAE expects latent_dim as a top-level kwarg, but it's also stored
            # inside distribution_dict.  Extract it so the constructor sees it.
            if 'latent_dim' not in constructor_args:
                dd = constructor_args.get('distribution_dict', {}) or {}
                if 'latent_dim' in dd:
                    constructor_args['latent_dim'] = dd['latent_dim']
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
            from model.priors import VampPrior

            encoder_input_dim = getattr(model.encoder, 'input_dim', None)
            if encoder_input_dim is not None:
                input_shape = int(encoder_input_dim)
            else:
                input_shape = init_args.get('input_shape', None)
                if input_shape is None:
                    if hasattr(model, 'set_hidden_dim') and hasattr(model, 'condition_dim'):
                        input_shape = int(model.set_hidden_dim) + int(model.condition_dim)
                    elif hasattr(model, 'condition_dim'):
                        input_shape = int(model.input_dim) + int(model.condition_dim)
                    else:
                        input_shape = int(model.input_dim)

            # Create VampPrior with encoder reference
            model.prior = VampPrior(
                latent_dim=init_args.get('latent_dim', model.latent_dim),
                input_shape=input_shape,
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
        
        normalizer = cls._load_normalizer_from_checkpoint(checkpoint=checkpoint, device=device)
        return model, normalizer
    
    def get_SigmaMapper(self):
        return self.decoder.get_SigmaMapper()




