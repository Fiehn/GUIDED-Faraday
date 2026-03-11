"""
Pre-Training Validation System
===============================
Validates model configuration, components, and data pipeline before training.
Catches common errors early and provides diagnostic reports.

Each ``_test_*`` method is a self-contained validation suite that appends
``ValidationResult`` objects to ``self.results``.  They can be called
independently (handy for pytest wrappers) or all-at-once via
``run_all_tests()`` / ``validate_before_training()``.
"""

import os
import sys
import tempfile
import copy
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(root_dir)
 
from models.encoder_decoder_model import *
from data.conditioner import *
from src.models.vae import *
from src.models.cvae import *
from data.dataloader import *
from data.normalizer import *
from models.priors import *

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ValidationResult:
    """Stores result of a single validation check."""

    def __init__(self, name: str, passed: bool, message: str = "", details: Dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()


def _get_batch(loader, device="cpu"):
    """Grab one batch and move tensors to *device*."""
    x, meter_id, timestamps = next(iter(loader))
    return (
        x.to(device),
        meter_id.to(device),
        timestamps.to(device),
    )


def _to_tensor(val, device="cpu"):
    """Convert numpy/number to torch.Tensor on *device*."""
    if isinstance(val, np.ndarray):
        return torch.tensor(val, dtype=torch.float32).to(device)
    if isinstance(val, torch.Tensor):
        return val.to(device)
    return val


# ===========================================================================
# Main validator
# ===========================================================================

class ModelValidator:
    """Comprehensive model validation system.

    Parameters
    ----------
    model : VAE | CVAE | OnlineCVAE
        The model to validate.
    train_loader, val_loader, test_loader : DataLoader
        PyTorch data loaders for train / val / test splits.
    normalizer : Normalizer | None
        The normalizer used during data loading (for round-trip tests).
    device : str | torch.device
        Device the model lives on.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
        normalizer=None,
        device="cpu",
        verbose=True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.normalizer = normalizer
        self.device = device
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.is_cvae = hasattr(model, "condition_dim") and model.condition_dim > 0

    # ------------------------------------------------------------------
    # Result bookkeeping
    # ------------------------------------------------------------------

    def _add_result(self, name: str, passed: bool, message: str = "", details: Dict = None):
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")
            if message:
                print(f"         {message}")

    # ------------------------------------------------------------------
    # run_all_tests
    # ------------------------------------------------------------------

    def run_all_tests(self) -> bool:
        """Execute every validation suite and return *True* when all pass."""
        if self.verbose:
            print("=" * 80)
            print("STARTING MODEL VALIDATION")
            print("=" * 80)
            print(f"Model Type: {'CVAE' if self.is_cvae else 'VAE'}")
            print(f"Device: {self.device}")
            print("-" * 80)

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
        self._test_numerical_stability()
        self._test_weight_initialization()
        self._test_multi_step_convergence()
        self._test_save_load_roundtrip()
        self._test_validation_loop()
        self._test_sampling()
        if getattr(self.model, "hierarchy", None) is not None:
            self._test_hierarchy()
        self._test_regression_bugs()
        self._test_RevIn()
        self._test_online_user_profiler()
        self._test_online_cvae()

        return self._generate_report()

    # ==================================================================
    # 1. Configuration
    # ==================================================================

    def _test_configuration(self):
        """Validate model attributes and hyper-parameter sanity."""
        if self.verbose:
            print("\n[1] Testing Configuration...")

        try:
            for attr in ("input_dim", "encoder", "decoder", "prior"):
                if not hasattr(self.model, attr):
                    self._add_result("Config: Model Structure", False,
                                     f"Missing attribute '{attr}'")
                    return

            config = {
                "model_type": "CVAE" if self.is_cvae else "VAE",
                "input_dim": self.model.input_dim,
                "encoder": type(self.model.encoder).__name__,
                "decoder": type(self.model.decoder).__name__,
                "prior": type(self.model.prior).__name__,
            }

            if self.is_cvae:
                config["condition_dim"] = self.model.condition_dim
                config["conditioner_tags"] = self.model.conditioner.get_condition_tags()

            if hasattr(self.model.encoder, "output_dim"):
                config["latent_dim"] = self.model.encoder.output_dim

            if hasattr(self.model.encoder, "dictionary_size"):
                ds = self.model.encoder.dictionary_size
                od = self.model.encoder.output_dim
                config["dictionary_size"] = ds
                if ds <= od:
                    self._add_result(
                        "Config: Dictionary Size", False,
                        f"dictionary_size ({ds}) must be > output_dim ({od})",
                    )
                    return

            self._add_result("Config: Model Structure", True,
                             "All required components present", config)

            # Encoder / decoder input dim consistency
            if self.is_cvae:
                expected_enc_in = self.model.input_dim + self.model.condition_dim
            else:
                expected_enc_in = self.model.input_dim
            actual_enc_in = getattr(self.model.encoder, "input_dim", None)
            if actual_enc_in is not None and actual_enc_in != expected_enc_in:
                self._add_result(
                    "Config: Encoder Input Dim", False,
                    f"Encoder expects {actual_enc_in}, model provides {expected_enc_in}",
                )
            else:
                self._add_result("Config: Encoder Input Dim", True,
                                 f"Encoder input dim = {expected_enc_in}")

            # Parameter count
            n_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self._add_result("Config: Parameters", True,
                             f"Total: {n_params:,} | Trainable: {trainable:,}")

        except Exception as e:
            self._add_result("Config: Model Structure", False, str(e))

    # ==================================================================
    # 2. Data pipeline
    # ==================================================================

    def _test_data_pipeline(self):
        """Validate data shapes, NaN/Inf, and normalizer round-trip."""
        if self.verbose:
            print("\n[2] Testing Data Pipeline...")

        try:
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)

            if x.dim() != 3:
                self._add_result("Data: Input Shape", False,
                                 f"Expected 3-D [B,L,F], got {x.dim()}-D: {x.shape}")
                return

            B, L, F = x.shape
            self._add_result("Data: Input Shape", True,
                             f"[{B}, {L}, {F}] -> flat [{B}, {L*F}]")

            # NaN / Inf
            if torch.isnan(x).any():
                self._add_result("Data: NaN/Inf", False, "NaN values detected")
            elif torch.isinf(x).any():
                self._add_result("Data: NaN/Inf", False, "Inf values detected")
            else:
                self._add_result("Data: NaN/Inf", True, "Clean")

            # Normalizer round-trip
            if self.normalizer is not None:
                x_denorm = self.normalizer.inverse_transform(x).to(x.dtype)
                x_renorm = self.normalizer.transform(x_denorm).to(x.dtype).to(x.device)
                max_err = torch.max(torch.abs(x - x_renorm)).item()
                if max_err > 1e-4:
                    self._add_result("Data: Normalizer Roundtrip", False,
                                     f"Max error {max_err:.6f} > 1e-4")
                else:
                    self._add_result("Data: Normalizer Roundtrip", True,
                                     f"Max error: {max_err:.2e}")

                if (x_denorm < 0).any():
                    self._add_result("Data: Non-negative Energy", False,
                                     f"min = {x_denorm.min().item():.4f}")
                else:
                    self._add_result("Data: Non-negative Energy", True,
                                     f"[{x_denorm.min().item():.3f}, {x_denorm.max().item():.3f}]")

            # Check that multiple batches are iterable
            try:
                n_batches = sum(1 for _ in self.train_loader)
                self._add_result("Data: Loader Iteration", True,
                                 f"{n_batches} batches in train_loader")
            except Exception as e:
                self._add_result("Data: Loader Iteration", False, str(e))

        except Exception as e:
            self._add_result("Data: Overall", False, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 3. Conditioners (CVAE only)
    # ==================================================================

    def _test_conditioners(self):
        """Test condition generation, individual conditioners, random generation."""
        if self.verbose:
            print("\n[3] Testing Conditioners...")

        try:
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            B = x.shape[0]

            conditions = self.model.conditioner.get_conditions(
                x, meter_id=meter_id, timestamps=timestamps)
            conditions = _to_tensor(conditions, self.device)

            if not isinstance(conditions, torch.Tensor):
                self._add_result("Cond: Output Type", False, f"Got {type(conditions)}")
                return

            expected = (B, self.model.condition_dim)
            if conditions.shape != expected:
                self._add_result("Cond: Shape", False,
                                 f"Expected {expected}, got {conditions.shape}")
            else:
                self._add_result("Cond: Shape", True, f"{conditions.shape}")

            # Individual conditioners
            for tag in self.model.conditioner.get_condition_tags():
                try:
                    item = self.model.conditioner.condition_items[tag]
                    cond = item.get_conditions(
                        x, meter_id=meter_id, timestamps=timestamps)
                    cond = _to_tensor(cond, self.device)
                    self._add_result(f"Cond: {tag}", True, f"Shape {cond.shape}")
                    if tag == "user_profile" and hasattr(item, "user_profiles_cache"):
                        self._add_result(f"Cond: {tag} Cache", True,
                                         f"{len(item.user_profiles_cache)} profiles")
                except Exception as e:
                    self._add_result(f"Cond: {tag}", False, str(e))

            # Random conditions
            try:
                rc = self.model.conditioner.get_random_conditions(batch_size=16)
                if rc.shape != (16, self.model.condition_dim):
                    self._add_result("Cond: Random", False, f"Shape {rc.shape}")
                else:
                    self._add_result("Cond: Random", True, f"Shape {rc.shape}")
            except Exception as e:
                self._add_result("Cond: Random", False, str(e))

        except Exception as e:
            self._add_result("Cond: Overall", False, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 4. Encoder / Decoder
    # ==================================================================

    def _test_encoder_decoder(self):
        """Test encoder/decoder forward, shapes, sigma positivity, reparam."""
        if self.verbose:
            print("\n[4] Testing Encoder/Decoder...")

        try:
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)

            # Encoder
            if self.is_cvae:
                c = self.model.conditioner.get_conditions(
                    x, meter_id=meter_id, timestamps=timestamps)
                c = _to_tensor(c, self.device)
                enc_in = torch.cat([x_flat, c], dim=-1)
            else:
                enc_in = x_flat

            q = self.model.encoder(enc_in)
            for key in ("mu", "sigma"):
                if key not in q:
                    self._add_result(f"Encoder: Missing '{key}'", False,
                                     f"Keys: {list(q.keys())}")
                    return

            mu, sigma = q["mu"], q["sigma"]
            exp_shape = (B, self.model.encoder.output_dim)
            if mu.shape != exp_shape:
                self._add_result("Encoder: Mu Shape", False,
                                 f"Expected {exp_shape}, got {mu.shape}")
            else:
                self._add_result("Encoder: Mu Shape", True, f"{mu.shape}")

            if (sigma <= 0).any():
                self._add_result("Encoder: Sigma > 0", False,
                                 f"min = {sigma.min().item():.6f}")
            else:
                self._add_result("Encoder: Sigma > 0", True,
                                 f"[{sigma.min().item():.4f}, {sigma.max().item():.4f}]")

            # Reparameterization
            z = self.model.encoder.reparameterize(q, num_samples=3)
            exp_z = (3, B, self.model.encoder.output_dim)
            if z.shape != exp_z:
                self._add_result("Encoder: Reparam Shape", False,
                                 f"Expected {exp_z}, got {z.shape}")
            else:
                self._add_result("Encoder: Reparam Shape", True, f"{z.shape}")

            # Decoder
            z_single = z[0]
            if self.is_cvae:
                dec_in = torch.cat([z_single, c], dim=-1)
            else:
                dec_in = z_single
            rp = self.model.decoder(dec_in)

            if rp["mu"].shape != (B, self.model.input_dim):
                self._add_result("Decoder: Output Shape", False,
                                 f"Expected ({B}, {self.model.input_dim}), got {rp['mu'].shape}")
            else:
                self._add_result("Decoder: Output Shape", True, f"{rp['mu'].shape}")

            # Dictionary-specific: SigmaMapper normalization
            if hasattr(self.model.encoder, "get_SigmaMapper"):
                sm = self.model.encoder.get_SigmaMapper()
                if sm is not None:
                    norms = torch.norm(sm, dim=0)
                    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
                        self._add_result("Encoder: SigmaMapper Norms", False,
                                         f"Sample norms: {norms[:5].tolist()}")
                    else:
                        self._add_result("Encoder: SigmaMapper Norms", True,
                                         "Unit-norm columns")

        except Exception as e:
            self._add_result("Encoder/Decoder: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 5. Prior
    # ==================================================================

    def _test_prior(self):
        """Test prior sampling, log-prob, and parameter expansion."""
        if self.verbose:
            print("\n[5] Testing Prior...")

        try:
            prior = self.model.prior
            bs = 16

            samples = prior.sample(batch_size=bs, num_samples=5)
            self._add_result("Prior: Sampling", True, f"Shape: {samples.shape}")

            test_z = torch.randn(bs, self.model.encoder.output_dim, device=self.device)
            try:
                lp = prior.log_prob(test_z)
                if torch.isnan(lp).any() or torch.isinf(lp).any():
                    self._add_result("Prior: Log-prob", False, "NaN/Inf in log_prob")
                else:
                    self._add_result("Prior: Log-prob", True,
                                     f"mean={lp.mean().item():.3f}")
            except Exception as e:
                self._add_result("Prior: Log-prob", False, str(e))

            try:
                params = prior.forward(bs)
                self._add_result("Prior: Forward", True,
                                 f"mu {params['mu'].shape}, sigma {params['sigma'].shape}")
            except Exception as e:
                self._add_result("Prior: Forward", False, str(e))

        except Exception as e:
            self._add_result("Prior: Overall", False, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 6. Full forward + backward
    # ==================================================================

    def _test_model_forward_backward(self):
        """Full model forward/backward: shapes, gradient existence, NaN."""
        if self.verbose:
            print("\n[6] Testing Model Forward/Backward...")

        try:
            self.model.train()
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)

            if self.is_cvae:
                c = self.model.conditioner.get_conditions(
                    x, meter_id=meter_id, timestamps=timestamps)
                c = _to_tensor(c, self.device)
                recon, q = self.model.forward(x_flat, c, num_samples=1)
            else:
                recon, q = self.model.forward(x_flat, num_samples=1)

            exp = (1, B, self.model.input_dim)
            if recon["mu"].shape != exp:
                self._add_result("Forward: Recon Shape", False,
                                 f"Expected {exp}, got {recon['mu'].shape}")
            else:
                self._add_result("Forward: Recon Shape", True,
                                 f"{recon['mu'].shape}")

            # Backward via MSE
            loss = torch.nn.functional.mse_loss(recon["mu"].squeeze(0), x_flat)
            loss.backward()

            grad_issues, n_grad = [], 0
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    n_grad += 1
                    if torch.isnan(p.grad).any():
                        grad_issues.append(f"{name}: NaN")
                    elif torch.isinf(p.grad).any():
                        grad_issues.append(f"{name}: Inf")

            if grad_issues:
                self._add_result("Backward: Gradients", False,
                                 f"Issues: {grad_issues[:5]}")
            elif n_grad == 0:
                self._add_result("Backward: Gradients", False,
                                 "No gradients computed")
            else:
                self._add_result("Backward: Gradients", True,
                                 f"{n_grad} params with grad")

            self.model.zero_grad()

        except Exception as e:
            self._add_result("Forward/Backward: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 7. Loss computation (ELBO)
    # ==================================================================

    def _test_loss_computation(self):
        """Verify loss keys, finiteness, KL >= 0, ELBO formula."""
        if self.verbose:
            print("\n[7] Testing Loss Computation...")

        try:
            self.model.eval()
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            x_flat = x.reshape(x.shape[0], -1)

            with torch.no_grad():
                if self.is_cvae:
                    c = self.model.conditioner.get_conditions(
                        x, meter_id=meter_id, timestamps=timestamps)
                    c = _to_tensor(c, self.device)
                    rp, qp = self.model.forward(x_flat, c, num_samples=1)
                    ld = self.model.loss(x_flat, rp, qp, beta=1.0)
                else:
                    rp, qp = self.model.forward(x_flat, num_samples=1)
                    ld = self.model.loss(x_flat, rp, qp, beta=1.0)

            if "loss" not in ld:
                self._add_result("Loss: Keys", False,
                                 f"Missing 'loss'. Keys: {list(ld.keys())}")
                return

            _recon_keys = ("recon_loglikelihood", "recon_loss", "recon")
            _kl_keys = ("kl_divergence", "kl_loss", "kl")
            recon_val = next((ld[k].item() for k in _recon_keys if k in ld), None)
            kl_val = next((ld[k].item() for k in _kl_keys if k in ld), None)
            total = ld["loss"].item()

            if recon_val is None or kl_val is None:
                self._add_result("Loss: Components", False,
                                 f"Keys: {list(ld.keys())}")
                return

            if not np.isfinite(total):
                self._add_result("Loss: Finite", False, f"total = {total}")
            else:
                self._add_result("Loss: Finite", True, f"total = {total:.4f}")

            if kl_val < -1e-5:
                self._add_result("Loss: KL >= 0", False, f"KL = {kl_val:.6f}")
            else:
                self._add_result("Loss: KL >= 0", True, f"KL = {kl_val:.4f}")

            expected = -(recon_val - 1.0 * kl_val)
            if abs(total - expected) > 1e-3:
                self._add_result("Loss: ELBO Formula", False,
                                 f"Expected {expected:.4f}, got {total:.4f}")
            else:
                self._add_result("Loss: ELBO Formula", True,
                                 f"-(recon - beta*KL) = {total:.4f}")

            # Beta sensitivity
            with torch.no_grad():
                ld0 = self.model.loss(x_flat, rp, qp, beta=0.0)
            if abs(ld0["loss"].item() - total) < 1e-8 and kl_val > 1e-4:
                self._add_result("Loss: Beta Sensitivity", False,
                                 "Loss unchanged between beta=0 and beta=1")
            else:
                self._add_result("Loss: Beta Sensitivity", True,
                                 f"beta=0 -> {ld0['loss'].item():.4f}, beta=1 -> {total:.4f}")

        except Exception as e:
            self._add_result("Loss: Overall", False, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 8. Device placement
    # ==================================================================

    def _test_device_placement(self):
        """Ensure model, prior, and data all live on the expected device."""
        if self.verbose:
            print("\n[8] Testing Device Placement...")

        try:
            model_dev = str(next(self.model.parameters()).device).split(":")[0]
            expected_dev = str(self.device).split(":")[0]

            if model_dev != expected_dev:
                self._add_result("Device: Model", False,
                                 f"On {model_dev}, expected {expected_dev}")
            else:
                self._add_result("Device: Model", True, f"{model_dev}")

            if hasattr(self.model.prior, "mu"):
                prior_dev = str(self.model.prior.mu.device).split(":")[0]
                if prior_dev != model_dev:
                    self._add_result("Device: Prior", False,
                                     f"Prior on {prior_dev}, model on {model_dev}")
                else:
                    self._add_result("Device: Prior", True, f"{prior_dev}")

            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            if self.is_cvae:
                c = self.model.conditioner.get_conditions(
                    x, meter_id=meter_id, timestamps=timestamps)
                c = _to_tensor(c, self.device)
                if c.device != x.device:
                    self._add_result("Device: Conditions", False,
                                     f"Cond on {c.device}, data on {x.device}")
                else:
                    self._add_result("Device: Conditions", True, f"{c.device}")

            self._add_result("Device: Consistency", True, "All on same device")

        except Exception as e:
            self._add_result("Device: Overall", False, f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 9. Full training step (train_core)
    # ==================================================================

    def _test_full_training_step(self):
        """Run one ``train_core`` call and check loss / gradients."""
        if self.verbose:
            print("\n[9] Testing Full Training Step...")

        try:
            self.model.train()
            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)
            B, L, F = x.shape
            x_flat = x.reshape(B, -1)
            optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

            if self.is_cvae:
                ld = self.model.train_core(
                    (x, meter_id, timestamps), optim, beta=1.0)
            else:
                ld = self.model.train_core(x, optim, beta=1.0)

            if "loss" not in ld:
                self._add_result("TrainStep: Loss Key", False, "Missing 'loss'")
                return

            loss_v = ld["loss"].item()
            if not np.isfinite(loss_v):
                self._add_result("TrainStep: Loss Finite", False,
                                 f"loss = {loss_v}")
            else:
                self._add_result("TrainStep: Forward/Backward", True,
                                 f"loss = {loss_v:.4f}")

            has_grad = any(p.grad is not None for p in self.model.parameters())
            self._add_result("TrainStep: Gradients", has_grad,
                             "Computed" if has_grad else "None computed")

            # Dimension consistency
            if self.is_cvae:
                c = self.model.conditioner.get_conditions(
                    x, meter_id=meter_id, timestamps=timestamps)
                c = _to_tensor(c, self.device)
                expected = self.model.input_dim + self.model.condition_dim
                actual = x_flat.shape[1] + c.shape[1]
                if expected != actual:
                    self._add_result("TrainStep: Dim Consistency", False,
                                     f"Encoder expects {expected}, got {actual}")
                else:
                    self._add_result("TrainStep: Dim Consistency", True,
                                     f"Encoder input = {expected}")
            else:
                self._add_result("TrainStep: Dim Consistency", True,
                                 f"Encoder input = {x_flat.shape[1]}")

            self.model.zero_grad()
            self.model.eval()

        except RuntimeError as e:
            msg = str(e)
            if "mat1 and mat2" in msg or "size mismatch" in msg:
                self._add_result("TrainStep: Shape Error", False, msg)
            else:
                self._add_result("TrainStep: Runtime Error", False, msg)
            if self.verbose:
                traceback.print_exc()
        except Exception as e:
            self._add_result("TrainStep: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 10. Numerical stability
    # ==================================================================

    def _test_numerical_stability(self):
        """Feed extreme inputs (zeros, large, small) and check for NaN/Inf."""
        if self.verbose:
            print("\n[10] Testing Numerical Stability...")

        try:
            self.model.eval()
            B = 4
            D = self.model.input_dim

            cases = {
                "zeros": torch.zeros(B, D, device=self.device),
                "ones": torch.ones(B, D, device=self.device),
                "large": torch.ones(B, D, device=self.device) * 100.0,
                "small_pos": torch.ones(B, D, device=self.device) * 1e-6,
            }

            for tag, x_test in cases.items():
                try:
                    with torch.no_grad():
                        if self.is_cvae:
                            c = self.model.conditioner.get_random_conditions(
                                batch_size=B)
                            c = _to_tensor(c, self.device)
                            rp, qp = self.model.forward(x_test, c, num_samples=1)
                        else:
                            rp, qp = self.model.forward(x_test, num_samples=1)

                    has_nan = any(torch.isnan(v).any() for v in rp.values())
                    has_inf = any(torch.isinf(v).any() for v in rp.values())
                    if has_nan or has_inf:
                        self._add_result(f"Stability: {tag}", False,
                                         "NaN/Inf in output")
                    else:
                        self._add_result(f"Stability: {tag}", True, "Clean")
                except Exception as e:
                    self._add_result(f"Stability: {tag}", False, str(e))

        except Exception as e:
            self._add_result("Stability: Overall", False,
                             f"{type(e).__name__}: {e}")

    # ==================================================================
    # 11. Weight initialization
    # ==================================================================

    def _test_weight_initialization(self):
        """Check for dead layers (all-zero) and exploding weights."""
        if self.verbose:
            print("\n[11] Testing Weight Initialization...")

        try:
            dead, exploding = [], []
            for name, p in self.model.named_parameters():
                if p.numel() == 0:
                    continue
                if p.abs().max().item() == 0.0:
                    dead.append(name)
                if p.abs().max().item() > 1e6:
                    exploding.append((name, p.abs().max().item()))

            if dead:
                self._add_result("Weights: Dead Layers", False,
                                 f"{len(dead)} all-zero: {dead[:3]}")
            else:
                self._add_result("Weights: Dead Layers", True, "None detected")

            if exploding:
                self._add_result("Weights: Exploding", False,
                                 f"{len(exploding)} > 1e6: {exploding[:3]}")
            else:
                self._add_result("Weights: Exploding", True, "All within bounds")

            # Check encoder/decoder independence
            enc_p = list(self.model.encoder.parameters())
            dec_p = list(self.model.decoder.parameters())
            if len(enc_p) > 0 and len(dec_p) > 0:
                e0 = enc_p[0].data.flatten()[:10]
                d0 = dec_p[0].data.flatten()[:10]
                if e0.shape == d0.shape and torch.allclose(e0, d0):
                    self._add_result("Weights: Enc/Dec Independence", False,
                                     "First tensors appear identical")
                else:
                    self._add_result("Weights: Enc/Dec Independence", True,
                                     "Distinct weights")

        except Exception as e:
            self._add_result("Weights: Overall", False,
                             f"{type(e).__name__}: {e}")

    # ==================================================================
    # 12. Multi-step convergence
    # ==================================================================

    def _test_multi_step_convergence(self):
        """Run a few optimization steps; loss should trend downward."""
        if self.verbose:
            print("\n[12] Testing Multi-Step Convergence...")

        try:
            model_copy = copy.deepcopy(self.model)
            model_copy.train()
            opt = torch.optim.Adam(model_copy.parameters(), lr=1e-3)

            x, meter_id, timestamps = _get_batch(self.train_loader, self.device)

            losses = []
            n_steps = 5
            for _ in range(n_steps):
                if self.is_cvae:
                    ld = model_copy.train_core(
                        (x, meter_id, timestamps), opt, beta=1.0)
                else:
                    ld = model_copy.train_core(x, opt, beta=1.0)
                losses.append(ld["loss"].item())

            if not all(np.isfinite(l) for l in losses):
                self._add_result("Convergence: Finite", False,
                                 f"Losses: {losses}")
                return

            improved = losses[-1] < losses[0]
            suffix = " (improved)" if improved else " (may be okay for 1 batch)"
            self._add_result(
                "Convergence: Loss Decreased", improved,
                f"Step 1: {losses[0]:.4f} -> Step {n_steps}: {losses[-1]:.4f}" + suffix,
            )

            self._add_result("Convergence: Stable", True,
                             f"All {n_steps} losses finite: "
                             f"{[f'{l:.2f}' for l in losses]}")

            del model_copy

        except Exception as e:
            self._add_result("Convergence: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 13. Save / Load round-trip
    # ==================================================================

    def _test_save_load_roundtrip(self):
        """Save model to temp dir, reload, and compare parameters."""
        if self.verbose:
            print("\n[13] Testing Save/Load Round-trip...")

        try:
            self.model.eval()
            original_sd = {k: v.clone() for k, v in self.model.state_dict().items()}

            with tempfile.TemporaryDirectory() as tmpdir:
                self.model.save_model(
                    save_path=tmpdir, model_name="val_test", device="cpu")

                pt_path = os.path.join(tmpdir, "val_test.pt")
                if not os.path.exists(pt_path):
                    self._add_result("Save/Load: File Created", False,
                                     "Checkpoint not found")
                    return
                self._add_result("Save/Load: File Created", True,
                                 f"{os.path.getsize(pt_path) / 1024:.1f} KB")

                loaded, normalizer = type(self.model).load_from_checkpoint(
                    pt_path, device="cpu")
                loaded_sd = loaded.state_dict()

                mismatched = []
                for k in original_sd:
                    if k not in loaded_sd:
                        mismatched.append(f"missing: {k}")
                    elif not torch.allclose(
                        original_sd[k].float().cpu(), loaded_sd[k].float(), atol=1e-5
                    ):
                        mismatched.append(f"differs: {k}")

                if mismatched:
                    self._add_result("Save/Load: Params Match", False,
                                     f"{len(mismatched)} issues: {mismatched[:5]}")
                else:
                    self._add_result("Save/Load: Params Match", True,
                                     f"All {len(original_sd)} params match")

            # Restore device
            self.model.to(self.device)
            if hasattr(self.model, "prior"):
                self.model.prior.to(self.device)

        except Exception as e:
            self._add_result("Save/Load: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()
            try:
                self.model.to(self.device)
            except Exception:
                pass

    # ==================================================================
    # 14. Validation loop
    # ==================================================================

    def _test_validation_loop(self):
        """Run model.validate() on val_loader and check finiteness."""
        if self.verbose:
            print("\n[14] Testing Validation Loop...")

        if self.val_loader is None:
            self._add_result("ValLoop: Skipped", True, "No val_loader")
            return

        try:
            self.model.eval()
            if self.is_cvae:
                agg = self.model.validate(
                    self.val_loader, num_samples=1, beta=1.0,
                    device=str(self.device))
            else:
                agg = self.model.validate(
                    self.val_loader, num_samples=1, beta=1.0,
                    device=str(self.device))

            for k, v in agg.items():
                if not np.isfinite(v):
                    self._add_result(f"ValLoop: {k}", False, f"Non-finite: {v}")
                else:
                    self._add_result(f"ValLoop: {k}", True, f"{v:.4f}")

        except Exception as e:
            self._add_result("ValLoop: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 15. Sampling
    # ==================================================================

    def _test_sampling(self):
        """Sample from prior and check shapes, NaN, diversity."""
        if self.verbose:
            print("\n[15] Testing Sampling...")

        try:
            self.model.eval()

            if self.is_cvae:
                rc = self.model.conditioner.get_random_conditions(batch_size=4)
                rc = _to_tensor(rc, self.device)
                samples = self.model.sample(
                    conditions=rc, num_prior_samples=3, num_x_samples=1,
                    device=str(self.device),
                )
                if samples.dim() != 4:
                    self._add_result("Sampling: Shape", False,
                                     f"Expected 4-D, got {samples.shape}")
                else:
                    self._add_result("Sampling: Shape", True,
                                     f"{samples.shape}")
            else:
                result = self.model.sample(num_samples_prior=3,
                                           num_samples_posterior=1)
                samples = result["samples"]
                self._add_result("Sampling: Shape", True, f"{samples.shape}")

            if torch.isnan(samples).any() or torch.isinf(samples).any():
                self._add_result("Sampling: Clean", False, "NaN/Inf in samples")
            else:
                self._add_result("Sampling: Clean", True, "No NaN/Inf")

            flat = samples.reshape(-1, samples.shape[-1])
            var = flat.var(dim=0).mean().item()
            if var < 1e-8:
                self._add_result("Sampling: Diversity", False,
                                 f"Near-zero var ({var:.2e}) - mode collapse?")
            else:
                self._add_result("Sampling: Diversity", True,
                                 f"Mean var = {var:.4f}")

        except Exception as e:
            self._add_result("Sampling: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 16. Hierarchical module
    # ==================================================================

    def _test_hierarchy(self):
        """Validate the optional HierarchicalLatentModule."""
        if self.verbose:
            print("\n[16] Testing Hierarchy...")

        try:
            h = self.model.hierarchy
            if h is None:
                self._add_result("Hierarchy: Present", True, "None - skipped")
                return

            self._add_result("Hierarchy: Present", True,
                             f"{len(h.levels)} levels, "
                             f"agg={type(h.aggregation).__name__}")

            z = torch.randn(8, h.config.levels[0].input_dim, device=self.device)
            out = h(z)

            for key in ("z", "loss", "level_outputs"):
                if key not in out:
                    self._add_result(f"Hierarchy: Key '{key}'", False, "Missing")
                else:
                    self._add_result(f"Hierarchy: Key '{key}'", True, "Present")

            if out["z"].shape != (8, h.output_dim):
                self._add_result("Hierarchy: Output Shape", False,
                                 f"Got {out['z'].shape}, expected (8, {h.output_dim})")
            else:
                self._add_result("Hierarchy: Output Shape", True,
                                 f"{out['z'].shape}")

            z_g = torch.randn(4, h.config.levels[0].input_dim,
                              device=self.device, requires_grad=True)
            out_g = h(z_g)
            out_g["z"].sum().backward()
            if z_g.grad is None or z_g.grad.abs().max() == 0:
                self._add_result("Hierarchy: Gradient Flow", False,
                                 "No gradients")
            else:
                self._add_result("Hierarchy: Gradient Flow", True, "OK")

        except Exception as e:
            self._add_result("Hierarchy: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 17. Regression bugs
    # ==================================================================

    def _test_regression_bugs(self):
        """Known-bug regression tests."""
        if self.verbose:
            print("\n[17] Testing Regression Bugs...")

        if self.is_cvae and "user_profile" in self.model.conditioner.get_condition_tags():
            try:
                x, meter_id, timestamps = _get_batch(
                    self.train_loader, self.device)
                up = self.model.conditioner.condition_items["user_profile"]
                x_2d = x.reshape(x.shape[0], -1)

                for tag, inp in [("3D", x), ("2D", x_2d)]:
                    try:
                        c = up.get_conditions(
                            inp, meter_id=meter_id, timestamps=timestamps)
                        c = _to_tensor(c, self.device)
                        self._add_result(
                            f"Regression: UserProfiler {tag}", True,
                            f"Shape {c.shape}")
                    except Exception as e:
                        self._add_result(
                            f"Regression: UserProfiler {tag}", False, str(e))
            except Exception as e:
                self._add_result("Regression: UserProfiler", False, str(e))

        if self.is_cvae:
            try:
                rc = self.model.conditioner.get_random_conditions(batch_size=8)
                if not isinstance(rc, torch.Tensor):
                    self._add_result("Regression: Random Cond Type", False,
                                     f"Got {type(rc)}")
                else:
                    self._add_result("Regression: Random Cond Type", True,
                                     "torch.Tensor")
            except Exception as e:
                self._add_result("Regression: Random Cond", False, str(e))

    # ==================================================================
    # 18. RevIN (IndexedLogRevIN)
    # ==================================================================

    def _test_RevIn(self):
        """Reversible instance normalization -- synthetic test."""
        if self.verbose:
            print("\n[18] Testing RevIN...")

        try:
            from Data.normalizer import IndexedLogRevIN

            B, L, F, eps = 8, 24, 5, 1e-5
            revin = IndexedLogRevIN(num_features=F, eps=eps, affine=True)

            torch.manual_seed(42)
            x = torch.abs(torch.randn(B, L, F) * 2.0 + 5.0)
            ts = torch.tensor([1640995200 + i * 86400 for i in range(B)])

            xn = revin(x.clone(), mode="norm", timestamp=ts)
            if xn.shape != x.shape:
                self._add_result("RevIN: Norm Shape", False, f"{xn.shape}")
                return
            self._add_result("RevIN: Norm Shape", True, f"{xn.shape}")

            std = torch.std(xn, dim=1)
            if torch.allclose(std, torch.zeros_like(std), atol=1e-6):
                self._add_result("RevIN: Variance", False, "Zero after norm")
            else:
                self._add_result("RevIN: Variance", True,
                                 f"std [{std.min().item():.4f}, {std.max().item():.4f}]")

            xd = revin(xn.clone(), mode="denorm")
            rel_err = (x - xd).abs().max().item() / (x.abs().max().item() + 1e-8)
            if rel_err > 1e-3:
                self._add_result("RevIN: Round-trip", False,
                                 f"rel err = {rel_err:.6f}")
            else:
                self._add_result("RevIN: Round-trip", True,
                                 f"rel err = {rel_err:.6f}")

            from torch.utils.data import TensorDataset, DataLoader as DL
            dl = DL(TensorDataset(x, ts, ts), batch_size=B)
            revin.update_monthly_stats(dl, month_key=202201)
            if 202201 in revin.monthly_stats:
                self._add_result("RevIN: Monthly Stats", True, "Stored")
            else:
                self._add_result("RevIN: Monthly Stats", False, "Not stored")

            for tag, xt in [("constant", torch.ones(2, L, F) * 3.0),
                             ("zeros", torch.zeros(2, L, F))]:
                try:
                    out = revin(xt.clone(), mode="norm")
                    ok = not (torch.isnan(out).any() or torch.isinf(out).any())
                    self._add_result(f"RevIN: {tag}", ok,
                                     "Clean" if ok else "NaN/Inf")
                except Exception as e:
                    self._add_result(f"RevIN: {tag}", False, str(e))

            if revin.affine:
                ok = (hasattr(revin, "weight") and hasattr(revin, "bias")
                      and revin.weight.shape == torch.Size([F])
                      and revin.bias.shape == torch.Size([F]))
                self._add_result("RevIN: Affine Params", ok,
                                 "Correct shapes" if ok else "Bad shapes")

        except ImportError:
            self._add_result("RevIN: Import", False,
                             "Could not import IndexedLogRevIN")
        except Exception as e:
            self._add_result("RevIN: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 19. Online User Profiler
    # ==================================================================

    def _test_online_user_profiler(self):
        """OnlineUserProfiler -- synthetic incremental-update test."""
        if self.verbose:
            print("\n[19] Testing Online User Profiler...")

        try:
            from Data.Conditioner import OnlineUserProfiler

            B, L, F, K, T = 16, 24, 2, 10, 5
            op = OnlineUserProfiler(
                n_topics=T, num_clusters=K, decay=0.8,
                random_state=42, verbose=False)

            if not hasattr(op, "kmeans"):
                self._add_result("OnlineProf: Init", False, "Missing kmeans")
                return
            self._add_result("OnlineProf: Init", True,
                             f"topics={T}, clusters={K}")

            torch.manual_seed(42)
            x1 = torch.abs(torch.randn(B, L, F)) + 1.0
            ids1 = torch.randint(0, 10, (B,))
            p1 = op.update_and_get_profile(x1, ids1)

            if p1.shape != (B, T):
                self._add_result("OnlineProf: Batch1", False,
                                 f"Shape {p1.shape}")
            else:
                self._add_result("OnlineProf: Batch1", True,
                                 f"Shape {p1.shape}")
                sums = torch.sum(p1, dim=1)
                ok = torch.allclose(sums, torch.ones(B), atol=1e-4)
                self._add_result("OnlineProf: Normalized", ok,
                                 "Sum ~ 1" if ok else f"Sums: {sums[:5]}")

            if not op.kmeans_initialized:
                self._add_result("OnlineProf: KMeans", False, "Not init")
            else:
                self._add_result("OnlineProf: KMeans", True,
                                 f"Centers {op.cluster_centers_.shape}")

            x2 = torch.abs(torch.randn(B, L, F)) + 1.0
            ids2 = torch.randint(5, 15, (B,))
            p2 = op.update_and_get_profile(x2, ids2)
            self._add_result("OnlineProf: Incremental",
                             p2.shape == (B, T), f"Shape {p2.shape}")

            try:
                rc = op.get_random_conditions(8)
                self._add_result("OnlineProf: Interface",
                                 rc.shape == (8, T), f"Shape {rc.shape}")
            except Exception as e:
                self._add_result("OnlineProf: Interface", False, str(e))

            if hasattr(op, "decay") and op.decay != 1.0:
                self._add_result("OnlineProf: Decay", True,
                                 f"decay = {op.decay}")
            else:
                self._add_result("OnlineProf: Decay", False, "No decay")

        except ImportError:
            self._add_result("OnlineProf: Import", False, "Could not import")
        except Exception as e:
            self._add_result("OnlineProf: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # 20. Online CVAE
    # ==================================================================

    def _test_online_cvae(self):
        """OnlineCVAE lifecycle: init, train, sample, reconstruct."""
        if self.verbose:
            print("\n[20] Testing Online CVAE...")

        try:
            from src.Models.Online_CVAE import OnlineCVAE
            from Data.Conditioner import (Conditioner, DayOfWeekConditioner,
                                         OnlineUserProfiler)

            input_dim, latent, B = 24, 8, 16

            cond = Conditioner()
            cond.add_condition_item("day_of_week", DayOfWeekConditioner())
            cond.add_condition_item("user_profile", OnlineUserProfiler(
                n_topics=5, num_clusters=10, decay=0.9,
                update_frequency=5, random_state=42, verbose=False,
            ))

            norm = Normalizer("log1p")
            try:
                ocvae = OnlineCVAE(
                    input_dim=input_dim, conditioner=cond,
                    latent_dim=latent, prior_type="standard_dictionary",
                    device=self.device, normalizer=norm,
                )
                ocvae.to(self.device)
                self._add_result("OnlineCVAE: Init", True, "OK")
            except Exception as e:
                self._add_result("OnlineCVAE: Init", False, str(e))
                return

            td = torch.abs(torch.randn(B, 24, 1, device=self.device)) + 1.0
            ids = torch.randint(1, 11, (B,), device=self.device)
            ts = torch.randint(0, 365 * 24, (B,), device=self.device)
            norm.fit(td)

            # Train
            try:
                ocvae.train()
                opt = torch.optim.Adam(ocvae.parameters(), lr=1e-3)
                ld1 = ocvae.train_core((td, ids, ts), opt, beta=1.0)
                loss1 = ld1["loss"].item()
                if not np.isfinite(loss1):
                    self._add_result("OnlineCVAE: Train", False,
                                     f"loss = {loss1}")
                    return
                self._add_result("OnlineCVAE: Train", True,
                                 f"loss = {loss1:.4f}")
            except Exception as e:
                self._add_result("OnlineCVAE: Train", False, str(e))
                return

            # Sample
            try:
                ocvae.eval()
                sc = ocvae.conditioner.get_random_conditions(2)
                sc = _to_tensor(sc, self.device)
                samples = ocvae.sample(
                    conditions=sc, num_prior_samples=4, num_x_samples=1,
                    device=str(self.device)
                )
                exp = (4, 1, 2, input_dim)
                self._add_result(
                    "OnlineCVAE: Sample",
                    samples.shape == exp,
                    f"Shape {samples.shape}"
                    + ("" if samples.shape == exp else f" != {exp}"))
            except Exception as e:
                self._add_result("OnlineCVAE: Sample", False, str(e))

            # Second batch
            try:
                ocvae.train()
                td2 = torch.abs(
                    torch.randn(B, 24, 1, device=self.device)) + 1.0
                ids2 = torch.randint(5, 15, (B,), device=self.device)
                ld2 = ocvae.train_core((td2, ids2, ts + 86400), opt, beta=1.0)
                self._add_result("OnlineCVAE: Incremental", True,
                                 f"loss = {ld2['loss'].item():.4f}")
            except Exception as e:
                self._add_result("OnlineCVAE: Incremental", False, str(e))

            # Reconstruct
            try:
                ocvae.eval()
                with torch.no_grad():
                    rp, qp = ocvae.reconstruct(
                        td[:4], ids[:4], ts[:4], num_samples=1)
                exp_r = (1, 4, input_dim)
                self._add_result("OnlineCVAE: Reconstruct",
                                 rp["mu"].shape == exp_r,
                                 f"Shape {rp['mu'].shape}")
            except Exception as e:
                self._add_result("OnlineCVAE: Reconstruct", False, str(e))

            # Memory
            up = ocvae.conditioner.condition_items.get("user_profile")
            if hasattr(up, "user_profiles_cache"):
                self._add_result(
                    "OnlineCVAE: Memory", True,
                    f"Cached: {len(up.user_profiles_cache)} users, ")
            else:
                self._add_result("OnlineCVAE: Memory", False,
                                 "No profile cache")

        except ImportError:
            self._add_result("OnlineCVAE: Import", False,
                             "Could not import OnlineCVAE")
        except Exception as e:
            self._add_result("OnlineCVAE: Overall", False,
                             f"{type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

    # ==================================================================
    # Report
    # ==================================================================

    def _generate_report(self) -> bool:
        if self.verbose:
            print("\n" + "=" * 80)
            print("VALIDATION REPORT")
            print("=" * 80)

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        rate = len(passed) / len(self.results) * 100 if self.results else 0

        if self.verbose:
            print(f"\nResults: {len(passed)}/{len(self.results)} "
                  f"passed ({rate:.1f}%)")
            if failed:
                print(f"\n{'!' * 80}")
                print(f"FAILED TESTS ({len(failed)}):")
                print("!" * 80)
                for r in failed:
                    print(f"\n  x {r.name}")
                    print(f"    {r.message}")
                    if r.details:
                        print(f"    {r.details}")
            print("\n" + "=" * 80)
            if not failed:
                print("ALL TESTS PASSED")
            else:
                print(f"{len(failed)} FAILED")
            print("=" * 80 + "\n")

        return len(failed) == 0


# ===========================================================================
# Convenience entry points
# ===========================================================================

def validate_before_training(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    normalizer=None,
    device="cpu",
    verbose=True,
) -> bool:
    """Run the full validation suite.  Returns True when everything passes."""
    v = ModelValidator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        verbose=verbose,
    )
    return v.run_all_tests()


def create_test_dataset_if_missing():
    test_file = "data/test_data_small.csv"
    if os.path.exists(test_file):
        return True
    print(f"\nTest dataset not found: {test_file}")
    print("Create it with:  python src/test/create_test_data.py\n")
    return False


if __name__ == "__main__":
    print("Run this module via train.py with validation enabled,")
    print("or via pytest:  uv run pytest src/test/test_validation.py -v")
