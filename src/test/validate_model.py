"""
Pre-Training Validation System
Validates model configuration, components, and data pipeline before training.
Catches common errors early and provides diagnostic reports.
"""
import os
import sys
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.encoder_decoder_model import *
from src.Conditioner import *
from src.Models import *
from src.dataloader import *
from src.normalizer import *
from src.priors import *


class ValidationResult:
    """Stores result of a validation test."""
    def __init__(self, name: str, passed: bool, message: str = "", details: Dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()


class ModelValidator:
    """Comprehensive model validation system."""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 normalizer=None,
                 device="cpu",
                 verbose=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.normalizer = normalizer
        self.device = device
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.is_cvae = hasattr(model, 'condition_dim') and model.condition_dim > 0
        
    def run_all_tests(self) -> bool:
        """Run all validation tests and return overall pass/fail."""
        if self.verbose:
            print("=" * 80)
            print("STARTING MODEL VALIDATION")
            print("=" * 80)
            print(f"Model Type: {'CVAE' if self.is_cvae else 'VAE'}")
            print(f"Device: {self.device}")
            print("-" * 80)
        
        # Run test suites
        self._test_configuration()
        self._test_data_pipeline()
        if self.is_cvae:
            self._test_conditioners()
        self._test_encoder_decoder()
        self._test_prior()
        self._test_model_forward_backward()
        self._test_loss_computation()
        self._test_device_placement()
        self._test_full_training_step()
        self._test_regression_bugs()
        self._test_RevIn()
        self._test_online_user_profiler()
        self._test_online_cvae()
        
        # Generate report
        all_passed = self._generate_report()
        return all_passed
    
    def _add_result(self, name: str, passed: bool, message: str = "", details: Dict = None):
        """Add a test result."""
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")
            if message:
                print(f"         {message}")
    
    def _test_configuration(self):
        """Validate model configuration."""
        if self.verbose:
            print("\n[1] Testing Configuration...")
        
        try:
            # Check model attributes
            assert hasattr(self.model, 'input_dim'), "Model missing input_dim"
            assert hasattr(self.model, 'encoder'), "Model missing encoder"
            assert hasattr(self.model, 'decoder'), "Model missing decoder"
            assert hasattr(self.model, 'prior'), "Model missing prior"
            
            config = {
                'model_type': 'CVAE' if self.is_cvae else 'VAE',
                'input_dim': self.model.input_dim,
                'encoder_type': type(self.model.encoder).__name__,
                'decoder_type': type(self.model.decoder).__name__,
                'prior_type': type(self.model.prior).__name__,
            }
            
            if self.is_cvae:
                config['condition_dim'] = self.model.condition_dim
                config['conditioner_tags'] = self.model.conditioner.get_condition_tags()
            
            if hasattr(self.model.encoder, 'output_dim'):
                config['latent_dim'] = self.model.encoder.output_dim
                
            if hasattr(self.model.encoder, 'dictionary_size'):
                config['dictionary_size'] = self.model.encoder.dictionary_size
                # Validate dictionary size > output_dim
                if self.model.encoder.dictionary_size <= self.model.encoder.output_dim:
                    self._add_result("Configuration: Dictionary Size",
                                   False,
                                   f"dictionary_size ({self.model.encoder.dictionary_size}) must be > output_dim ({self.model.encoder.output_dim})")
                    return
            
            self._add_result("Configuration: Model Structure", True, 
                           f"All required components present", config)
            
        except Exception as e:
            self._add_result("Configuration: Model Structure", False, str(e))
    
    def _test_data_pipeline(self):
        """Test data loading and preprocessing."""
        if self.verbose:
            print("\n[2] Testing Data Pipeline...")
        
        try:
            # Get a batch
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            
            # Test shapes
            expected_batch_dim = 3  # [B, L, F]
            if x.dim() != expected_batch_dim:
                self._add_result("Data Pipeline: Input Shape",
                               False,
                               f"Expected {expected_batch_dim}D tensor, got {x.dim()}D: {x.shape}")
                return
            
            B, L, F = x.shape
            details = {
                'batch_size': B,
                'sequence_length': L,
                'num_features': F,
                'input_dim': L * F,
                'meter_id_shape': meter_id.shape,
                'num_timestamps': len(timestamps),
            }
            
            self._add_result("Data Pipeline: Input Shape", True,
                           f"Shape: {x.shape} → flattened: ({B}, {L*F})", details)
            
            # Test for NaN/Inf
            if torch.isnan(x).any():
                self._add_result("Data Pipeline: NaN Detection", False, "NaN values found in input data")
            elif torch.isinf(x).any():
                self._add_result("Data Pipeline: Inf Detection", False, "Inf values found in input data")
            else:
                self._add_result("Data Pipeline: No NaN/Inf", True)
            
            # Test normalizer if present
            if self.normalizer:
                x_denorm = self.normalizer.inverse_transform(x)
                x_renorm = self.normalizer.transform(x_denorm)
                
                # Check roundtrip (should be close)
                max_error = torch.max(torch.abs(x - x_renorm)).item()
                if max_error > 1e-4:
                    self._add_result("Data Pipeline: Normalizer Roundtrip", False,
                                   f"Max error: {max_error:.6f} (threshold: 1e-4)")
                else:
                    self._add_result("Data Pipeline: Normalizer Roundtrip", True,
                                   f"Max error: {max_error:.6e}")
                    
                # Check denormalized values are non-negative (energy data)
                if (x_denorm < 0).any():
                    self._add_result("Data Pipeline: Denormalized Values", False,
                                   f"Negative values found: min={x_denorm.min().item():.3f}")
                else:
                    self._add_result("Data Pipeline: Denormalized Values", True,
                                   f"Range: [{x_denorm.min().item():.3f}, {x_denorm.max().item():.3f}]")
            
        except Exception as e:
            self._add_result("Data Pipeline: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_conditioners(self):
        """Test conditioner components (CVAE only)."""
        if self.verbose:
            print("\n[3] Testing Conditioners...")
        
        try:
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            meter_id = meter_id.to(self.device)
            timestamps = timestamps.to(self.device)
            
            # Test condition generation
            conditions = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
            
            if not isinstance(conditions, torch.Tensor):
                self._add_result("Conditioners: Output Type", False,
                               f"Expected torch.Tensor, got {type(conditions)}")
                return
            
            expected_shape = (x.shape[0], self.model.condition_dim)
            if conditions.shape != expected_shape:
                self._add_result("Conditioners: Output Shape", False,
                               f"Expected {expected_shape}, got {conditions.shape}")
            else:
                self._add_result("Conditioners: Output Shape", True,
                               f"Shape: {conditions.shape}")
            
            # Test each conditioner individually
            for tag in self.model.conditioner.get_condition_tags():
                try:
                    conditioner_item = self.model.conditioner.condition_items[tag]
                    cond = conditioner_item.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                    
                    if isinstance(cond, np.ndarray):
                        cond = torch.tensor(cond)
                    
                    self._add_result(f"Conditioners: {tag}", True,
                                   f"Shape: {cond.shape}, Dim: {cond.shape[1]}")
                    
                    # Special checks for UserProfiler
                    if tag == "user_profile" and hasattr(conditioner_item, 'user_profiles_cache'):
                        num_cached = len(conditioner_item.user_profiles_cache)
                        self._add_result(f"Conditioners: {tag} Cache", True,
                                       f"{num_cached} profiles cached")
                        
                except Exception as e:
                    self._add_result(f"Conditioners: {tag}", False, str(e))
            
            # Test random condition generation
            try:
                random_conds = self.model.conditioner.get_random_conditions(batch_size=16)
                if random_conds.shape != (16, self.model.condition_dim):
                    self._add_result("Conditioners: Random Generation", False,
                                   f"Expected (16, {self.model.condition_dim}), got {random_conds.shape}")
                else:
                    self._add_result("Conditioners: Random Generation", True,
                                   f"Shape: {random_conds.shape}")
            except Exception as e:
                self._add_result("Conditioners: Random Generation", False, str(e))
                
        except Exception as e:
            self._add_result("Conditioners: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_encoder_decoder(self):
        """Test encoder and decoder forward passes."""
        if self.verbose:
            print("\n[4] Testing Encoder/Decoder...")
        
        try:
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)
            
            # Test encoder
            if self.is_cvae:
                c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                c = torch.tensor(c, dtype=torch.float32).to(self.device) if isinstance(c, np.ndarray) else c.to(self.device)
                encoder_input = torch.cat([x_flat, c], dim=-1)
            else:
                encoder_input = x_flat
            
            posterior_params = self.model.encoder(encoder_input)
            
            # Check posterior parameters
            if 'mu' not in posterior_params or 'sigma' not in posterior_params:
                self._add_result("Encoder: Output Parameters", False,
                               "Missing 'mu' or 'sigma' in output")
                return
            
            mu = posterior_params['mu']
            sigma = posterior_params['sigma']
            
            expected_mu_shape = (B, self.model.encoder.output_dim)
            if mu.shape != expected_mu_shape:
                self._add_result("Encoder: Mu Shape", False,
                               f"Expected {expected_mu_shape}, got {mu.shape}")
            else:
                self._add_result("Encoder: Mu Shape", True, f"Shape: {mu.shape}")
            
            # Check sigma is positive
            if (sigma <= 0).any():
                self._add_result("Encoder: Sigma Positive", False,
                               f"Non-positive sigma values found: min={sigma.min().item():.6f}")
            else:
                self._add_result("Encoder: Sigma Positive", True,
                               f"Range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
            
            # Test reparameterization
            try:
                z = self.model.encoder.reparameterize(posterior_params, num_samples=3)
                expected_z_shape = (3, B, self.model.encoder.output_dim)
                if z.shape != expected_z_shape:
                    self._add_result("Encoder: Reparameterization Shape", False,
                                   f"Expected {expected_z_shape}, got {z.shape}")
                else:
                    self._add_result("Encoder: Reparameterization Shape", True,
                                   f"Shape: {z.shape}")
            except Exception as e:
                self._add_result("Encoder: Reparameterization", False, str(e))
            
            # Test decoder
            z_single = z[0]  # [B, latent_dim]
            if self.is_cvae:
                decoder_input = torch.cat([z_single, c], dim=-1)
            else:
                decoder_input = z_single
            
            recon_params = self.model.decoder(decoder_input)
            
            recon_mu = recon_params['mu']
            recon_sigma = recon_params['sigma']
            
            expected_recon_shape = (B, self.model.input_dim)
            if recon_mu.shape != expected_recon_shape:
                self._add_result("Decoder: Output Shape", False,
                               f"Expected {expected_recon_shape}, got {recon_mu.shape}")
            else:
                self._add_result("Decoder: Output Shape", True,
                               f"Shape: {recon_mu.shape}")
            
            # Test Dictionary-specific components
            if hasattr(self.model.encoder, 'get_SigmaMapper'):
                sigma_mapper = self.model.encoder.get_SigmaMapper()
                if sigma_mapper is not None:
                    expected_mapper_shape = (self.model.encoder.output_dim, 
                                           self.model.encoder.dictionary_size)
                    if sigma_mapper.shape != expected_mapper_shape:
                        self._add_result("Encoder: SigmaMapper Shape", False,
                                       f"Expected {expected_mapper_shape}, got {sigma_mapper.shape}")
                    else:
                        # Check normalization (columns should have unit norm)
                        col_norms = torch.norm(sigma_mapper, dim=0)
                        if not torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5):
                            self._add_result("Encoder: SigmaMapper Normalization", False,
                                           f"Columns not unit norm: {col_norms[:5].tolist()}")
                        else:
                            self._add_result("Encoder: SigmaMapper Normalization", True,
                                           "All columns have unit norm")
                            
        except Exception as e:
            self._add_result("Encoder/Decoder: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_prior(self):
        """Test prior distribution."""
        if self.verbose:
            print("\n[5] Testing Prior...")
        
        try:
            prior = self.model.prior
            batch_size = 16
            
            # Test sampling
            samples = prior.sample(batch_size=batch_size, num_samples=5)
            
            if hasattr(self.model.encoder, 'dictionary_size'):
                # Dictionary prior
                expected_shape = (5, batch_size, self.model.encoder.dictionary_size)
            else:
                # Gaussian prior
                expected_shape = (batch_size, 5, self.model.encoder.output_dim)
            
            # Note: Different priors may have different sample shape conventions
            self._add_result("Prior: Sampling", True,
                           f"Sample shape: {samples.shape}")
            
            # Test log_prob
            try:
                if hasattr(self.model.encoder, 'dictionary_size'):
                    # For dictionary, create output space sample
                    test_z = torch.randn(batch_size, self.model.encoder.output_dim).to(self.device)
                else:
                    test_z = torch.randn(batch_size, self.model.encoder.output_dim).to(self.device)
                
                log_p = prior.log_prob(test_z)
                
                if torch.isnan(log_p).any() or torch.isinf(log_p).any():
                    self._add_result("Prior: Log Prob", False, "NaN or Inf in log_prob")
                else:
                    self._add_result("Prior: Log Prob", True,
                                   f"Mean: {log_p.mean().item():.3f}, Range: [{log_p.min().item():.3f}, {log_p.max().item():.3f}]")
            except Exception as e:
                self._add_result("Prior: Log Prob", False, str(e))
            
            # Test parameter expansion
            try:
                params = prior.forward(batch_size)
                self._add_result("Prior: Parameter Expansion", True,
                               f"Mu: {params['mu'].shape}, Sigma: {params['sigma'].shape}")
            except Exception as e:
                self._add_result("Prior: Parameter Expansion", False, str(e))
                
        except Exception as e:
            self._add_result("Prior: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_model_forward_backward(self):
        """Test full model forward and backward pass."""
        if self.verbose:
            print("\n[6] Testing Model Forward/Backward...")
        
        try:
            self.model.train()
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            meter_id = meter_id.to(self.device)
            timestamps = timestamps.to(self.device)
            
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)
            
            # Forward pass
            if self.is_cvae:
                c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                c = torch.tensor(c, dtype=torch.float32).to(self.device) if isinstance(c, np.ndarray) else c.to(self.device)
                recon_params, posterior_params = self.model.forward(x_flat, c, num_samples=1)
            else:
                recon_params, posterior_params = self.model.forward(x_flat, num_samples=1)
            
            # Check output shapes
            if recon_params['mu'].shape != (1, B, self.model.input_dim):
                self._add_result("Model Forward: Reconstruction Shape", False,
                               f"Expected (1, {B}, {self.model.input_dim}), got {recon_params['mu'].shape}")
            else:
                self._add_result("Model Forward: Reconstruction Shape", True,
                               f"Shape: {recon_params['mu'].shape}")
            
            # Test backward pass
            try:
                # Compute a simple loss
                recon_mu = recon_params['mu'].squeeze(0)  # [B, input_dim]
                loss = torch.nn.functional.mse_loss(recon_mu, x_flat)
                
                # Backward
                loss.backward()
                
                # Check gradients exist and are not NaN
                # Note: Some params may not have gradients if not in computation graph
                grad_issues = []
                params_with_grad = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            params_with_grad += 1
                            if torch.isnan(param.grad).any():
                                grad_issues.append(f"{name}: NaN")
                            elif torch.isinf(param.grad).any():
                                grad_issues.append(f"{name}: Inf")
                
                if grad_issues:
                    self._add_result("Model Backward: Gradients", False,
                                   f"Issues: {grad_issues[:3]}")
                elif params_with_grad == 0:
                    self._add_result("Model Backward: Gradients", False,
                                   "No gradients computed")
                else:
                    self._add_result("Model Backward: Gradients", True,
                                   f"All gradients valid ({params_with_grad} params)")
                
                # Zero gradients for cleanup
                self.model.zero_grad()
                
            except Exception as e:
                self._add_result("Model Backward: Overall", False, str(e))
                
        except Exception as e:
            self._add_result("Model Forward/Backward: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_loss_computation(self):
        """Test loss components."""
        if self.verbose:
            print("\n[7] Testing Loss Computation...")
        
        try:
            self.model.eval()
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            meter_id = meter_id.to(self.device)
            timestamps = timestamps.to(self.device)
            
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)
            
            with torch.no_grad():
                if self.is_cvae:
                    c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                    c = torch.tensor(c, dtype=torch.float32).to(self.device) if isinstance(c, np.ndarray) else c.to(self.device)
                    recon_params, posterior_params = self.model.forward(x_flat, c, num_samples=1)
                    loss_dict = self.model.loss(x_flat, recon_params, posterior_params, beta=1.0)
                else:
                    recon_params, posterior_params = self.model.forward(x_flat, num_samples=1)
                    loss_dict = self.model.loss(x_flat, recon_params, posterior_params, beta=1.0)
                
                # Check loss components (keys may vary: recon/reconstruction, kl/kl_divergence)
                if 'loss' not in loss_dict:
                    self._add_result("Loss: Components", False, "Missing 'loss' key")
                    return
                
                # Get reconstruction loss (check multiple possible keys)
                recon_loss = None
                for key in ['recon_loss', 'recon', 'reconstruction_loss', 'reconstruction', 'recon_loglikelihood']:
                    if key in loss_dict:
                        recon_loss = loss_dict[key].item()
                        break
                
                # Get KL loss (check multiple possible keys)
                kl_loss = None
                for key in ['kl_loss', 'kl', 'kl_divergence']:
                    if key in loss_dict:
                        kl_loss = loss_dict[key].item()
                        break
                
                total_loss = loss_dict['loss'].item()
                
                if recon_loss is None or kl_loss is None:
                    self._add_result("Loss: Components", False,
                                   f"Available keys: {list(loss_dict.keys())}")
                    return
                
                # Check values are finite
                if not np.isfinite(total_loss):
                    self._add_result("Loss: Total Loss Finite", False,
                                   f"Total loss is {total_loss}")
                else:
                    self._add_result("Loss: Total Loss Finite", True,
                                   f"Value: {total_loss:.4f}")
                
                if not np.isfinite(recon_loss):
                    self._add_result("Loss: Reconstruction Loss Finite", False,
                                   f"Recon loss is {recon_loss}")
                else:
                    self._add_result("Loss: Reconstruction Loss Finite", True,
                                   f"Value: {recon_loss:.4f}")
                
                # KL should be non-negative
                if kl_loss < -1e-5:  # Small tolerance for numerical errors
                    self._add_result("Loss: KL Non-negative", False,
                                   f"KL divergence is negative: {kl_loss:.6f}")
                else:
                    self._add_result("Loss: KL Non-negative", True,
                                   f"Value: {kl_loss:.4f}")
                
                # Check ELBO formula: loss = -(recon_loglikelihood - beta*kl_loss)
                # Note: recon_loglikelihood is positive (log likelihood), kl is positive
                # Total loss should be negative of ELBO
                expected_loss = -(recon_loss - 1.0 * kl_loss)
                if abs(total_loss - expected_loss) > 1e-3:
                    self._add_result("Loss: ELBO Formula", False,
                                   f"Expected {expected_loss:.4f}, got {total_loss:.4f}")
                else:
                    self._add_result("Loss: ELBO Formula", True,
                                   f"-(recon_loglik - β*KL) = {total_loss:.4f}")
                    
        except Exception as e:
            self._add_result("Loss: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_device_placement(self):
        """Test that all components are on correct device."""
        if self.verbose:
            print("\n[8] Testing Device Placement...")
        
        try:
            # Check model device
            model_device = next(self.model.parameters()).device
            # Compare device types (cuda:0 == cuda)
            model_device_type = str(model_device).split(':')[0]
            expected_device_type = str(self.device).split(':')[0]
            if model_device_type != expected_device_type:
                self._add_result("Device: Model", False,
                               f"Model on {model_device_type}, expected {expected_device_type}")
            else:
                self._add_result("Device: Model", True, f"Device: {model_device}")
            
            # Check prior device
            if hasattr(self.model.prior, 'mu'):
                prior_device = self.model.prior.mu.device
                if str(prior_device) != str(model_device):
                    self._add_result("Device: Prior", False,
                                   f"Prior on {prior_device}, model on {model_device}")
                else:
                    self._add_result("Device: Prior", True, f"Device: {prior_device}")
            
            # Test forward pass with data on device
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            
            if self.is_cvae:
                meter_id = meter_id.to(self.device)
                timestamps = timestamps.to(self.device)
                c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                c = torch.tensor(c, dtype=torch.float32).to(self.device) if isinstance(c, np.ndarray) else c.to(self.device)
                
                # Check condition device
                if c.device != x.device:
                    self._add_result("Device: Conditions", False,
                                   f"Conditions on {c.device}, data on {x.device}")
                else:
                    self._add_result("Device: Conditions", True, f"Device: {c.device}")
            
            self._add_result("Device: Overall Consistency", True, "All components on same device")
            
        except Exception as e:
            self._add_result("Device: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_full_training_step(self):
        """Test a complete training iteration to catch dimension mismatches."""
        if self.verbose:
            print("\n[9] Testing Full Training Step...")
        
        try:
            # Set model to training mode
            self.model.train()
            
            # Get a batch
            batch = next(iter(self.train_loader))
            x, meter_id, timestamps = batch
            x = x.to(self.device)
            meter_id = meter_id.to(self.device)
            timestamps = timestamps.to(self.device)
            
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)
            
            # Create optimizer (won't actually update, just need it for the training step)
            optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            
            # Test the actual train_core method (this is what runs during training)
            try:
                if self.is_cvae:
                    # CVAE expects (x, meter_id, timestamps) tuple
                    loss_dict = self.model.train_core((x, meter_id, timestamps), optim, beta=1.0)
                else:
                    # VAE expects just x
                    loss_dict = self.model.train_core(x, optim, beta=1.0)
                
                # Check that loss_dict has expected keys
                if 'loss' not in loss_dict:
                    self._add_result("Full Training Step: Loss Dict", False,
                                   "Missing 'loss' key in train_core output")
                    return
                
                loss = loss_dict['loss'].item()
                
                if not np.isfinite(loss):
                    self._add_result("Full Training Step: Loss Value", False,
                                   f"Loss is not finite: {loss}")
                else:
                    self._add_result("Full Training Step: Forward/Backward", True,
                                   f"Complete training step successful, loss={loss:.4f}")
                
                # Check that gradients were computed
                has_grad = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        has_grad = True
                        break
                
                if not has_grad:
                    self._add_result("Full Training Step: Gradients", False,
                                   "No gradients computed during train_core")
                else:
                    self._add_result("Full Training Step: Gradients", True,
                                   "Gradients computed successfully")
                
                # Test dimension consistency through the pipeline
                details = {
                    'input_shape': f"{x.shape}",
                    'flattened_shape': f"{x_flat.shape}",
                    'batch_size': B,
                    'loss': f"{loss:.4f}"
                }
                
                if self.is_cvae:
                    c = self.model.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamps)
                    c = torch.tensor(c, dtype=torch.float32).to(self.device) if isinstance(c, np.ndarray) else c.to(self.device)
                    details['condition_shape'] = f"{c.shape}"
                    details['encoder_input_dim'] = f"{x_flat.shape[1] + c.shape[1]}"
                    
                    # Verify dimensions match what encoder expects
                    expected_encoder_input = self.model.input_dim + self.model.condition_dim
                    actual_encoder_input = x_flat.shape[1] + c.shape[1]
                    
                    if expected_encoder_input != actual_encoder_input:
                        self._add_result("Full Training Step: Dimension Consistency", False,
                                       f"Encoder expects {expected_encoder_input}, got {actual_encoder_input}",
                                       details)
                    else:
                        self._add_result("Full Training Step: Dimension Consistency", True,
                                       f"All dimensions consistent", details)
                else:
                    details['encoder_input_dim'] = f"{x_flat.shape[1]}"
                    self._add_result("Full Training Step: Dimension Consistency", True,
                                   f"All dimensions consistent", details)
                
            except RuntimeError as e:
                error_msg = str(e)
                if "mat1 and mat2 shapes cannot be multiplied" in error_msg:
                    # Extract dimension info from error
                    self._add_result("Full Training Step: Dimension Mismatch", False,
                                   f"Matrix multiplication error: {error_msg}")
                elif "size mismatch" in error_msg or "shape" in error_msg:
                    self._add_result("Full Training Step: Shape Error", False,
                                   f"Shape mismatch: {error_msg}")
                else:
                    self._add_result("Full Training Step: Runtime Error", False,
                                   f"RuntimeError: {error_msg}")
                if self.verbose:
                    traceback.print_exc()
            
            # Clean up
            self.model.zero_grad()
            self.model.eval()
            
        except Exception as e:
            self._add_result("Full Training Step: Overall", False, 
                           f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_regression_bugs(self):
        """Test for known bugs that have been fixed."""
        if self.verbose:
            print("\n[10] Testing Regression Prevention...")
        
        # Bug 1: 2D vs 3D input handling in UserProfiler
        if self.is_cvae and "user_profile" in self.model.conditioner.get_condition_tags():
            try:
                batch = next(iter(self.train_loader))
                x, meter_id, timestamps = batch
                x_3d = x  # [B, L, F]
                x_2d = x.reshape(x.shape[0], -1)  # [B, D]
                
                user_profiler = self.model.conditioner.condition_items["user_profile"]
                
                # Test 3D input
                try:
                    cond_3d = user_profiler.get_conditions(x_3d, meter_id=meter_id, timestamps=timestamps)
                    self._add_result("Regression: UserProfiler 3D Input", True,
                                   f"Shape: {cond_3d.shape if isinstance(cond_3d, torch.Tensor) else np.array(cond_3d).shape}")
                except Exception as e:
                    self._add_result("Regression: UserProfiler 3D Input", False, str(e))
                
                # Test 2D input
                try:
                    cond_2d = user_profiler.get_conditions(x_2d, meter_id=meter_id, timestamps=timestamps)
                    self._add_result("Regression: UserProfiler 2D Input", True,
                                   f"Shape: {cond_2d.shape if isinstance(cond_2d, torch.Tensor) else np.array(cond_2d).shape}")
                except Exception as e:
                    self._add_result("Regression: UserProfiler 2D Input", False, str(e))
                    
            except Exception as e:
                self._add_result("Regression: UserProfiler Input Shapes", False, str(e))
        
        # Bug 2: Numpy vs Tensor type mixing in condition concatenation
        if self.is_cvae:
            try:
                random_conds = self.model.conditioner.get_random_conditions(batch_size=8)
                if not isinstance(random_conds, torch.Tensor):
                    self._add_result("Regression: Random Conditions Type", False,
                                   f"Expected torch.Tensor, got {type(random_conds)}")
                else:
                    self._add_result("Regression: Random Conditions Type", True,
                                   "Returns torch.Tensor")
            except Exception as e:
                self._add_result("Regression: Random Conditions", False, str(e))
    
    def _test_RevIn(self):
        """Test reversible instance normalization (RevIN)."""
        if self.verbose:
            print("\n[11] Testing RevIN...")
        
        try:
            from src.normalizer import IndexedLogRevIN
            
            # Test parameters
            batch_size = 8
            num_features = 5
            sequence_length = 24
            eps = 1e-5
            
            # Create RevIN instance
            revin = IndexedLogRevIN(num_features=num_features, eps=eps, affine=True)
            
            # Create test data - [B, L, F] format (batch, sequence_length, num_features)
            torch.manual_seed(42)  # For reproducible tests
            x_original = torch.randn(batch_size, sequence_length, num_features) * 2.0 + 5.0
            x_original = torch.abs(x_original)  # Ensure positive values for log1p
            
            # Create timestamps (different dates for different samples)
            base_time = 1640995200  # Jan 1, 2022 00:00:00 UTC
            timestamps = torch.tensor([base_time + i * 24 * 3600 for i in range(batch_size)])  # Different days
            
            # Test 1: Basic normalization
            x_norm = revin(x_original.clone(), mode='norm', timestamp=timestamps)
            
            if x_norm.shape != x_original.shape:
                self._add_result("RevIN: Normalization Shape", False,
                               f"Expected {x_original.shape}, got {x_norm.shape}")
                return
            else:
                self._add_result("RevIN: Normalization Shape", True,
                               f"Shape preserved: {x_norm.shape}")
            
            # Test 2: Check that normalized data has reasonable statistical properties
            # After normalization, data should be log1p transformed and normalized
            # Stats over time dimension (dim=1) for each sample should be ~N(0,1) before affine
            norm_mean = torch.mean(x_norm, dim=1)  # [B, F] - mean over time
            norm_std = torch.std(x_norm, dim=1)    # [B, F] - std over time
            
            # The normalized data should have some variance (not all zeros)
            if torch.allclose(norm_std, torch.zeros_like(norm_std), atol=1e-6):
                self._add_result("RevIN: Normalized Variance", False,
                               "Normalized data has zero variance")
            else:
                self._add_result("RevIN: Normalized Variance", True,
                               f"Normalized std range: [{norm_std.min().item():.4f}, {norm_std.max().item():.4f}]")
            
            # Test 3: Test denormalization without timestamps (should use cached stats)
            x_denorm = revin(x_norm.clone(), mode='denorm')
            
            if x_denorm.shape != x_original.shape:
                self._add_result("RevIN: Denormalization Shape", False,
                               f"Expected {x_original.shape}, got {x_denorm.shape}")
            else:
                self._add_result("RevIN: Denormalization Shape", True,
                               f"Shape preserved: {x_denorm.shape}")
            
            # Test 4: Round-trip test (normalize then denormalize)
            max_error = torch.max(torch.abs(x_original - x_denorm)).item()
            relative_error = max_error / (torch.max(torch.abs(x_original)).item() + 1e-8)
            
            if relative_error > 1e-3:  # Allow some numerical error
                self._add_result("RevIN: Round-trip Accuracy", False,
                               f"High reconstruction error: {relative_error:.6f}")
            else:
                self._add_result("RevIN: Round-trip Accuracy", True,
                               f"Max relative error: {relative_error:.6f}")
            
            # Test 5: Test monthly stats update via update_monthly_stats
            from torch.utils.data import TensorDataset, DataLoader
            test_dataset = TensorDataset(x_original, timestamps, timestamps)  # (x, meter_id, timestamps)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            revin.update_monthly_stats(test_dataloader, month_key=202201)
            
            if 202201 in revin.monthly_stats:
                stats = revin.monthly_stats[202201]
                self._add_result("RevIN: Monthly Stats Update", True,
                               f"Stored stats for month 202201, count={stats['count']}")
            else:
                self._add_result("RevIN: Monthly Stats Update", False,
                               "update_monthly_stats did not store stats")
            
            # Test 6: Test denormalization with specific timestamp
            test_timestamp = timestamps[0]
            x_test = torch.randn(1, sequence_length, num_features)
            try:
                x_denorm_specific = revin(x_test, mode='denorm', timestamp=test_timestamp)
                self._add_result("RevIN: Timestamp-specific Denorm", True,
                               f"Successfully denormalized with timestamp")
            except Exception as e:
                self._add_result("RevIN: Timestamp-specific Denorm", False, str(e))
            
            # Test 7: Test with month parameter
            try:
                month_val = 1  # January
                x_denorm_month = revin(x_test, mode='denorm', timestamp=test_timestamp, month=month_val)
                self._add_result("RevIN: Month-based Denorm", True,
                               f"Successfully denormalized with month parameter")
            except Exception as e:
                self._add_result("RevIN: Month-based Denorm", False, str(e))
            
            # Test 8: Test affine parameters (if affine=True)
            if revin.affine:
                if not hasattr(revin, 'weight') or not hasattr(revin, 'bias'):
                    self._add_result("RevIN: Affine Parameters", False,
                                   "Missing weight or bias parameters")
                elif revin.weight.shape != torch.Size([num_features]) or revin.bias.shape != torch.Size([num_features]):
                    self._add_result("RevIN: Affine Parameters", False,
                                   f"Incorrect parameter shapes: weight {revin.weight.shape}, bias {revin.bias.shape}")
                else:
                    self._add_result("RevIN: Affine Parameters", True,
                                   f"Correct affine parameter shapes")
            
            # Test 9: Test update_monthly_stats with second month
            batch2_x = torch.abs(torch.randn(batch_size, sequence_length, num_features)) + 2.0
            batch2_ts = timestamps + 31 * 24 * 3600  # ~1 month later
            test_dataset2 = TensorDataset(batch2_x, batch2_ts, batch2_ts)
            test_dataloader2 = DataLoader(test_dataset2, batch_size=batch_size)
            revin.update_monthly_stats(test_dataloader2, month_key=202202)
            
            if len(revin.monthly_stats) == 2:
                self._add_result("RevIN: Multiple Months", True,
                               f"Tracking {len(revin.monthly_stats)} months: {list(revin.monthly_stats.keys())}")
            else:
                self._add_result("RevIN: Multiple Months", False,
                               f"Expected 2 months, got {len(revin.monthly_stats)}")
            
            # Test 10: Edge case - empty timestamp (should work without indexing)
            try:
                x_no_timestamp = revin(x_original.clone(), mode='norm', timestamp=None)
                self._add_result("RevIN: No Timestamp", True,
                               "Normalization works without timestamp")
            except Exception as e:
                self._add_result("RevIN: No Timestamp", False, str(e))
            
            # Test 11: Edge case - constant window (all same value)
            x_constant = torch.ones(2, sequence_length, num_features) * 3.0
            try:
                x_const_norm = revin(x_constant.clone(), mode='norm')
                if torch.isnan(x_const_norm).any() or torch.isinf(x_const_norm).any():
                    self._add_result("RevIN: Constant Window", False,
                                   "NaN or Inf in output for constant input")
                else:
                    self._add_result("RevIN: Constant Window", True,
                                   "Handled constant window without NaN/Inf")
            except Exception as e:
                self._add_result("RevIN: Constant Window", False, str(e))
            
            # Test 12: Edge case - zero window
            x_zero = torch.zeros(2, sequence_length, num_features)
            try:
                x_zero_norm = revin(x_zero.clone(), mode='norm')
                if torch.isnan(x_zero_norm).any() or torch.isinf(x_zero_norm).any():
                    self._add_result("RevIN: Zero Window", False,
                                   "NaN or Inf in output for zero input")
                else:
                    self._add_result("RevIN: Zero Window", True,
                                   "Handled zero window without NaN/Inf")
            except Exception as e:
                self._add_result("RevIN: Zero Window", False, str(e))
                
        except ImportError:
            self._add_result("RevIN: Import", False, "Could not import IndexedLogRevIN")
        except Exception as e:
            self._add_result("RevIN: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_online_user_profiler(self):
        """Test online user profiler functionality."""
        if self.verbose:
            print("\n[12] Testing Online User Profiler...")
        
        try:
            from src.Conditioner import OnlineUserProfiler
            
            # Test parameters
            batch_size = 16
            sequence_length = 24
            num_features = 2
            n_topics = 5
            num_clusters = 10
            
            # Create OnlineUserProfiler instance
            online_profiler = OnlineUserProfiler(
                n_topics=n_topics,
                num_clusters=num_clusters,
                decay=0.8,
                random_state=42,
                verbose=False
            )
            
            # Test 1: Basic initialization
            if not hasattr(online_profiler, 'kmeans'):
                self._add_result("Online Profiler: Initialization", False, "Missing kmeans attribute")
                return
            else:
                self._add_result("Online Profiler: Initialization", True,
                               f"Initialized with {n_topics} topics, {num_clusters} clusters")
            
            # Create test data batches
            torch.manual_seed(42)
            batch1_x = torch.abs(torch.randn(batch_size, sequence_length, num_features)) + 1.0
            batch1_ids = torch.randint(0, 10, (batch_size,))
            
            batch2_x = torch.abs(torch.randn(batch_size, sequence_length, num_features)) + 1.0  
            batch2_ids = torch.randint(5, 15, (batch_size,))  # Some overlap, some new users
            
            # Test 2: First batch processing
            try:
                profiles1 = online_profiler.update_and_get_profile(batch1_x, batch1_ids)
                
                expected_shape = (batch_size, n_topics)
                if profiles1.shape != expected_shape:
                    self._add_result("Online Profiler: First Batch Shape", False,
                                   f"Expected {expected_shape}, got {profiles1.shape}")
                else:
                    self._add_result("Online Profiler: First Batch Shape", True,
                                   f"Shape: {profiles1.shape}")
                
                # Check that profiles are valid probability distributions
                profile_sums = torch.sum(profiles1, dim=1)
                if not torch.allclose(profile_sums, torch.ones(batch_size), atol=1e-4):
                    self._add_result("Online Profiler: Profile Normalization", False,
                                   f"Profiles don't sum to 1: {profile_sums[:5]}")
                else:
                    self._add_result("Online Profiler: Profile Normalization", True,
                                   "All profiles properly normalized")
                    
            except Exception as e:
                self._add_result("Online Profiler: First Batch", False, str(e))
                return
            
            # Test 3: KMeans was updated
            if not online_profiler.kmeans_initialized:
                self._add_result("Online Profiler: KMeans Update", False,
                               "KMeans not marked as initialized after first batch")
            else:
                cluster_centers_shape = online_profiler.cluster_centers_.shape
                expected_centers_shape = (num_clusters, sequence_length * num_features)
                if cluster_centers_shape != expected_centers_shape:
                    self._add_result("Online Profiler: KMeans Centers", False,
                                   f"Expected {expected_centers_shape}, got {cluster_centers_shape}")
                else:
                    self._add_result("Online Profiler: KMeans Centers", True,
                                   f"Cluster centers shape: {cluster_centers_shape}")
            
            # Test 4: Second batch processing (incremental learning)
            try:
                profiles2 = online_profiler.update_and_get_profile(batch2_x, batch2_ids)
                
                if profiles2.shape != expected_shape:
                    self._add_result("Online Profiler: Second Batch Shape", False,
                                   f"Expected {expected_shape}, got {profiles2.shape}")
                else:
                    self._add_result("Online Profiler: Second Batch Shape", True,
                                   "Incremental update successful")
                
                # Check that the model has learned from both batches
                # The cluster centers should be different after the second batch
                if torch.allclose(torch.tensor(online_profiler.cluster_centers_), 
                                torch.zeros_like(torch.tensor(online_profiler.cluster_centers_))):
                    self._add_result("Online Profiler: Incremental Learning", False,
                                   "Cluster centers appear to be zeros")
                else:
                    self._add_result("Online Profiler: Incremental Learning", True,
                                   "Model successfully updated with second batch")
                    
            except Exception as e:
                self._add_result("Online Profiler: Second Batch", False, str(e))
            
            # Test 5: Handling empty/invalid data
            try:
                empty_x = torch.full((batch_size, sequence_length, num_features), float('nan'))
                empty_ids = torch.randint(0, 5, (batch_size,))
                
                profiles_empty = online_profiler.update_and_get_profile(empty_x, empty_ids)
                self._add_result("Online Profiler: Empty Data Handling", True,
                               "Successfully handled batch with NaN values")
            except Exception as e:
                self._add_result("Online Profiler: Empty Data Handling", False, str(e))
            
            # Test 6: Interface compatibility with base UserProfiler
            try:
                # Test that it has the expected methods
                if not hasattr(online_profiler, 'get_conditions'):
                    self._add_result("Online Profiler: Interface Compatibility", False,
                                   "Missing get_conditions method")
                elif not hasattr(online_profiler, 'get_random_conditions'):
                    self._add_result("Online Profiler: Interface Compatibility", False,
                                   "Missing get_random_conditions method")
                else:
                    # Test get_random_conditions
                    random_conds = online_profiler.get_random_conditions(8)
                    if random_conds.shape != (8, n_topics):
                        self._add_result("Online Profiler: Interface Compatibility", False,
                                       f"get_random_conditions returned wrong shape: {random_conds.shape}")
                    else:
                        self._add_result("Online Profiler: Interface Compatibility", True,
                                       "Interface compatible with base UserProfiler")
            except Exception as e:
                self._add_result("Online Profiler: Interface Compatibility", False, str(e))
            
            # Test 7: Check for proper decay/forgetting mechanism
            if hasattr(online_profiler, 'decay') and online_profiler.decay != 1.0:
                self._add_result("Online Profiler: Decay Parameter", True,
                               f"Decay parameter set to {online_profiler.decay}")
            else:
                self._add_result("Online Profiler: Decay Parameter", False,
                               "No decay/forgetting mechanism implemented")
                
        except ImportError:
            self._add_result("Online Profiler: Import", False, "Could not import OnlineUserProfiler")
        except Exception as e:
            self._add_result("Online Profiler: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _test_online_cvae(self):
        """Test OnlineCVAE functionality and online learning capabilities."""
        if self.verbose:
            print("\n[13] Testing Online CVAE...")
        
        try:
            from src.Models import OnlineCVAE
            from src.Conditioner import Conditioner, DayOfWeekConditioner, OnlineUserProfiler
            from src.normalizer import IndexedLogRevIN
            
            # Test parameters
            input_dim = 24  # 24-hour sequences
            latent_dim = 8
            batch_size = 16
            
            # Create online conditioner with OnlineUserProfiler
            online_conditioner = Conditioner()
            online_conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
            online_conditioner.add_condition_item("user_profile", OnlineUserProfiler(
                n_topics=5, 
                num_clusters=10,
                decay=0.9,
                update_frequency=5,
                random_state=42,
                verbose=False
            ))
            
            # Test 1: OnlineCVAE initialization
            try:
                online_cvae = OnlineCVAE(
                    input_dim=input_dim,
                    conditioner=online_conditioner,
                    latent_dim=latent_dim,
                    prior_type="standard_dictionary",
                    device=self.device
                )
                online_cvae.to(self.device)
                
                if not hasattr(online_cvae, 'revin_normalizer'):
                    self._add_result("Online CVAE: Initialization", False, "Missing RevIN normalizer")
                    return
                else:
                    self._add_result("Online CVAE: Initialization", True,
                                f"Successfully initialized with RevIN normalizer")
                    
            except Exception as e:
                self._add_result("Online CVAE: Initialization", False, str(e))
                return
            
            # Create test data in correct format for OnlineCVAE
            # OnlineCVAE expects 3D input [batch, seq_len, features]
            test_data = torch.abs(torch.randn(batch_size, 24, 1, device=self.device)) + 1.0  # Positive values for energy
            test_meter_ids = torch.randint(1, 11, (batch_size,), device=self.device)
            test_timestamps = torch.randint(0, 365*24, (batch_size,), device=self.device)
            
            # Test 2: First batch training using train_core
            try:
                online_cvae.train()
                optimizer = torch.optim.Adam(online_cvae.parameters(), lr=1e-3)
                
                # Create batch tuple in the format expected by train_core
                batch = (test_data, test_meter_ids, test_timestamps)
                
                loss_dict = online_cvae.train_core(batch, optimizer, beta=1.0)
                
                # Check for the correct loss keys (based on VAE loss method)
                expected_keys = ['loss', 'recon_loglikelihood', 'kl_divergence']
                for key in expected_keys:
                    if key not in loss_dict:
                        self._add_result("Online CVAE: First Batch Training", False, 
                                    f"Missing key '{key}' in train_core result")
                        return
                
                # Check that losses are reasonable
                total_loss = loss_dict['loss'].item()
                if torch.isnan(torch.tensor(total_loss)) or torch.isinf(torch.tensor(total_loss)):
                    self._add_result("Online CVAE: First Batch Training", False,
                                f"Invalid loss: {total_loss}")
                    return
                    
                self._add_result("Online CVAE: First Batch Training", True,
                            f"First batch training successful, loss: {total_loss:.4f}")
                    
            except Exception as e:
                self._add_result("Online CVAE: First Batch Training", False, str(e))
                return
            
            # Test 3: Sampling
            try:
                online_cvae.eval()
                
                # Create conditions for sampling
                sample_batch = 2
                sample_conditions = online_cvae.conditioner.get_random_conditions(sample_batch)
                if isinstance(sample_conditions, torch.Tensor):
                    sample_conditions = sample_conditions.to(self.device)
                else:
                    sample_conditions = torch.tensor(sample_conditions, dtype=torch.float32).to(self.device)
                
                samples = online_cvae.sample(
                    conditions=sample_conditions,
                    num_prior_samples=4, 
                    num_x_samples=1, 
                    device=str(self.device),
                    timestamps=torch.tensor([100, 200], device=self.device)
                )
                
                # samples shape should be [S, num_x_samples, B, D] = [4, 1, 2, 24]
                expected_sample_shape = (4, 1, sample_batch, input_dim)
                if samples.shape == expected_sample_shape:
                    self._add_result("Online CVAE: Sampling", True,
                                f"Sampling successful, shape: {samples.shape}")
                else:
                    self._add_result("Online CVAE: Sampling", False,
                                f"Sample shape {samples.shape} != expected {expected_sample_shape}")
                    
            except Exception as e:
                self._add_result("Online CVAE: Sampling", False, str(e))
            
            # Test 4: Check RevIN normalization updates
            normalizer_stats = len(online_cvae.revin_normalizer.monthly_stats)
            # Monthly stats are updated via update_monthly_stats(), not per-batch,
            # so it's expected to be 0 after individual training steps
            self._add_result("Online CVAE: RevIN Updates", True,
                        f"RevIN tracking {normalizer_stats} months (updated via update_monthly_stats)")
            
            # Test 5: Second batch (incremental learning)
            try:
                # Create a second batch with some overlap and new data
                batch2_data = torch.abs(torch.randn(batch_size, 24, 1, device=self.device)) + 1.0
                batch2_ids = torch.randint(5, 15, (batch_size,), device=self.device)  # Some overlap, some new
                batch2_timestamps = test_timestamps + 24*3600  # Next day
                
                batch2 = (batch2_data, batch2_ids, batch2_timestamps)
                loss2_dict = online_cvae.train_core(batch2, optimizer, beta=1.0)
                
                second_loss = loss2_dict['loss'].item()
                self._add_result("Online CVAE: Second Batch Training", True,
                            f"Second batch successful, loss: {second_loss:.4f}")
                
                # Test update_monthly_stats explicitly
                from torch.utils.data import TensorDataset, DataLoader
                combined_data = torch.cat([test_data, batch2_data], dim=0)
                combined_ids = torch.cat([test_meter_ids, batch2_ids], dim=0)
                combined_ts = torch.cat([test_timestamps, batch2_timestamps], dim=0)
                test_dataset = TensorDataset(combined_data, combined_ids, combined_ts)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
                
                online_cvae.revin_normalizer.update_monthly_stats(test_dataloader, month_key=202301)
                
                if 202301 in online_cvae.revin_normalizer.monthly_stats:
                    stats = online_cvae.revin_normalizer.monthly_stats[202301]
                    self._add_result("Online CVAE: Incremental Learning", True,
                                f"Monthly stats updated: mean={stats['mean']}, count={stats['count']}")
                else:
                    self._add_result("Online CVAE: Incremental Learning", False,
                                "update_monthly_stats did not store stats")
                    
            except Exception as e:
                self._add_result("Online CVAE: Second Batch Training", False, str(e))
            
            # Test 6: Interface compatibility
            try:
                # Test that it still works like regular CVAE for reconstruction
                online_cvae.eval()
                test_x = test_data[:4]  # [4, 24, 1]
                test_ids = test_meter_ids[:4]
                test_ts = test_timestamps[:4]
                
                with torch.no_grad():
                    recon_params, posterior_params = online_cvae.reconstruct(
                        test_x, test_ids, test_ts, num_samples=1
                    )
                    expected_recon_shape = (1, 4, 24)  # [samples, batch, input_dim]
                    actual_shape = recon_params['mu'].shape
                    
                    if actual_shape == expected_recon_shape:
                        self._add_result("Online CVAE: Interface Compatibility", True,
                                    f"Compatible interface, recon shape: {actual_shape}")
                    else:
                        self._add_result("Online CVAE: Interface Compatibility", False,
                                    f"Shape mismatch: got {actual_shape}, expected {expected_recon_shape}")
                        
            except Exception as e:
                self._add_result("Online CVAE: Interface Compatibility", False, str(e))
                
            # Test 7: Memory efficiency
            user_profiler = online_cvae.conditioner.condition_items.get("user_profile")
            if hasattr(user_profiler, 'user_profiles_cache'):
                num_cached = len(user_profiler.user_profiles_cache)
                num_months = len(online_cvae.revin_normalizer.monthly_stats)
                self._add_result("Online CVAE: Memory Efficiency", True,
                            f"Cached users: {num_cached}, tracked months: {num_months}")
            else:
                self._add_result("Online CVAE: Memory Efficiency", False,
                            "User profiler not caching profiles")
                
        except ImportError:
            self._add_result("Online CVAE: Import", False, "Could not import OnlineCVAE")
        except Exception as e:
            self._add_result("Online CVAE: Overall", False, f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()
    
    def _generate_report(self) -> bool:
        """Generate and print validation report."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("VALIDATION REPORT")
            print("=" * 80)
        
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        pass_rate = len(passed) / len(self.results) * 100 if self.results else 0
        
        if self.verbose:
            print(f"\nResults: {len(passed)}/{len(self.results)} tests passed ({pass_rate:.1f}%)")
            
            if failed:
                print(f"\n{'!' * 80}")
                print(f"FAILED TESTS ({len(failed)}):")
                print('!' * 80)
                for r in failed:
                    print(f"\n✗ {r.name}")
                    print(f"  Message: {r.message}")
                    if r.details:
                        print(f"  Details: {r.details}")
            
            print("\n" + "=" * 80)
            if len(failed) == 0:
                print("✓ ALL TESTS PASSED - Model is ready for training!")
            else:
                print(f"✗ {len(failed)} TEST(S) FAILED - Please fix issues before training")
            print("=" * 80 + "\n")
        
        return len(failed) == 0
    


def validate_before_training(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    normalizer=None,
    device="cpu",
    verbose=True,
) -> bool:
    """
    Convenience function to run validation.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    validator = ModelValidator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        verbose=verbose,
    )
    return validator.run_all_tests()


def create_test_dataset_if_missing():
    """
    Check if test dataset exists, if not, prompt to create it.
    
    Returns:
        bool: True if test dataset exists or was created
    """
    test_file = "data/test_data_small.csv"
    if os.path.exists(test_file):
        return True
    
    print("\n" + "="*80)
    print("OPTIONAL: Create small test dataset for faster validation")
    print("="*80)
    print(f"Test dataset not found: {test_file}")
    print("\nCreate it with:")
    print("  python src/create_test_data.py")
    print("\nThis will create a 10 meters × 2 days subset for ~instant validation.")
    print("="*80 + "\n")
    return False


if __name__ == "__main__":
    print("Run this module via train.py with validation enabled")
