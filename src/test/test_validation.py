"""
Pytest wrapper for the ModelValidator validation system.
=========================================================

Each ``_test_*`` method from ``ModelValidator`` is exposed as an individual
pytest test for both VAE and CVAE models.

The 20 test suites are:
  1. configuration         10. numerical_stability
  2. data_pipeline          11. weight_initialization
  3. conditioners (CVAE)    12. multi_step_convergence
  4. encoder_decoder        13. save_load_roundtrip
  5. prior                  14. validation_loop
  6. model_forward_backward 15. sampling
  7. loss_computation       16. hierarchy
  8. device_placement       17. regression_bugs
  9. full_training_step     18. RevIn    19. online_user_profiler    20. online_cvae

Data comes from ``data/test_data_small.csv`` (via conftest.py fixtures).
"""
import os, sys, pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
 
from src.test.validate_model import ModelValidator
from src.test.conftest import skip_no_data


# ==================================================================
# Helper
# ==================================================================

def _run_validator_suite(
    model, train_loader, val_loader, test_loader,
    normalizer, device, suite_name,
):
    """Create a ModelValidator, run *suite_name*, and assert all pass."""
    v = ModelValidator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalizer=normalizer,
        device=device,
        verbose=True,
    )
    getattr(v, suite_name)()
    failures = [r for r in v.results if not r.passed]
    if failures:
        msgs = "\n".join(f"  FAIL: {r.name} — {r.message}" for r in failures)
        pytest.fail(f"{suite_name}: {len(failures)} failure(s):\n{msgs}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Macro to reduce boilerplate -- each class is 4 lines
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
_COMMON_SUITES = [
    ("Configuration",       "_test_configuration"),
    ("DataPipeline",        "_test_data_pipeline"),
    ("EncoderDecoder",      "_test_encoder_decoder"),
    ("Prior",               "_test_prior"),
    ("ForwardBackward",     "_test_model_forward_backward"),
    ("LossComputation",     "_test_loss_computation"),
    ("DevicePlacement",     "_test_device_placement"),
    ("FullTrainingStep",    "_test_full_training_step"),
    ("NumericalStability",  "_test_numerical_stability"),
    ("WeightInitialization","_test_weight_initialization"),
    ("MultiStepConvergence","_test_multi_step_convergence"),
    ("SaveLoadRoundtrip",   "_test_save_load_roundtrip"),
    ("ValidationLoop",      "_test_validation_loop"),
    ("Sampling",            "_test_sampling"),
    ("RegressionBugs",      "_test_regression_bugs"),
]

_CVAE_ONLY_SUITES = [
    ("Conditioners",        "_test_conditioners"),
]

_STANDALONE_SUITES = [
    ("RevIN",               "_test_RevIn"),
    ("OnlineUserProfiler",  "_test_online_user_profiler"),
    ("OnlineCVAE",          "_test_online_cvae"),
]


# ==================================================================
# CVAE Tests
# ==================================================================

@skip_no_data
class TestCVAEConfiguration:
    def test_configuration(self, cvae_model, train_loader, val_loader,
                           test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_configuration")


@skip_no_data
class TestCVAEDataPipeline:
    def test_data_pipeline(self, cvae_model, train_loader, val_loader,
                           test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_data_pipeline")


@skip_no_data
class TestCVAEConditioners:
    def test_conditioners(self, cvae_model, train_loader, val_loader,
                          test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_conditioners")


@skip_no_data
class TestCVAEEncoderDecoder:
    def test_encoder_decoder(self, cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_encoder_decoder")


@skip_no_data
class TestCVAEPrior:
    def test_prior(self, cvae_model, train_loader, val_loader,
                   test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_prior")


@skip_no_data
class TestCVAEForwardBackward:
    def test_forward_backward(self, cvae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_model_forward_backward")


@skip_no_data
class TestCVAELossComputation:
    def test_loss_computation(self, cvae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_loss_computation")


@skip_no_data
class TestCVAEDevicePlacement:
    def test_device_placement(self, cvae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_device_placement")


@skip_no_data
class TestCVAEFullTrainingStep:
    def test_full_training_step(self, cvae_model, train_loader, val_loader,
                                test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_full_training_step")


@skip_no_data
class TestCVAENumericalStability:
    def test_numerical_stability(self, cvae_model, train_loader, val_loader,
                                 test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_numerical_stability")


@skip_no_data
class TestCVAEWeightInitialization:
    def test_weight_initialization(self, cvae_model, train_loader, val_loader,
                                   test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_weight_initialization")


@skip_no_data
class TestCVAEMultiStepConvergence:
    def test_multi_step_convergence(self, cvae_model, train_loader, val_loader,
                                    test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_multi_step_convergence")


@skip_no_data
class TestCVAESaveLoadRoundtrip:
    def test_save_load_roundtrip(self, cvae_model, train_loader, val_loader,
                                 test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_save_load_roundtrip")


@skip_no_data
class TestCVAEValidationLoop:
    def test_validation_loop(self, cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_validation_loop")


@skip_no_data
class TestCVAESampling:
    def test_sampling(self, cvae_model, train_loader, val_loader,
                      test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_sampling")


@skip_no_data
class TestCVAERegressionBugs:
    def test_regression_bugs(self, cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_regression_bugs")


# ==================================================================
# VAE Tests
# ==================================================================

@skip_no_data
class TestVAEConfiguration:
    def test_configuration(self, vae_model, train_loader, val_loader,
                           test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_configuration")


@skip_no_data
class TestVAEDataPipeline:
    def test_data_pipeline(self, vae_model, train_loader, val_loader,
                           test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_data_pipeline")


@skip_no_data
class TestVAEEncoderDecoder:
    def test_encoder_decoder(self, vae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_encoder_decoder")


@skip_no_data
class TestVAEPrior:
    def test_prior(self, vae_model, train_loader, val_loader,
                   test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_prior")


@skip_no_data
class TestVAEForwardBackward:
    def test_forward_backward(self, vae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_model_forward_backward")


@skip_no_data
class TestVAELossComputation:
    def test_loss_computation(self, vae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_loss_computation")


@skip_no_data
class TestVAEDevicePlacement:
    def test_device_placement(self, vae_model, train_loader, val_loader,
                              test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_device_placement")


@skip_no_data
class TestVAEFullTrainingStep:
    def test_full_training_step(self, vae_model, train_loader, val_loader,
                                test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_full_training_step")


@skip_no_data
class TestVAENumericalStability:
    def test_numerical_stability(self, vae_model, train_loader, val_loader,
                                 test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_numerical_stability")


@skip_no_data
class TestVAEWeightInitialization:
    def test_weight_initialization(self, vae_model, train_loader, val_loader,
                                   test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_weight_initialization")


@skip_no_data
class TestVAEMultiStepConvergence:
    def test_multi_step_convergence(self, vae_model, train_loader, val_loader,
                                    test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_multi_step_convergence")


@skip_no_data
class TestVAESaveLoadRoundtrip:
    def test_save_load_roundtrip(self, vae_model, train_loader, val_loader,
                                 test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_save_load_roundtrip")


@skip_no_data
class TestVAEValidationLoop:
    def test_validation_loop(self, vae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_validation_loop")


@skip_no_data
class TestVAESampling:
    def test_sampling(self, vae_model, train_loader, val_loader,
                      test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_sampling")


@skip_no_data
class TestVAERegressionBugs:
    def test_regression_bugs(self, vae_model, train_loader, val_loader,
                             test_loader, normalizer, device):
        _run_validator_suite(vae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_regression_bugs")


# ==================================================================
# Standalone tests (not tied to a specific model fixture)
# ==================================================================

@skip_no_data
class TestRevIN:
    """RevIN tests use synthetic data inside the validator."""
    def test_revin(self, cvae_model, train_loader, val_loader,
                   test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_RevIn")


@skip_no_data
class TestOnlineUserProfiler:
    """Online user profiler tests (synthetic data)."""
    def test_online_user_profiler(self, cvae_model, train_loader, val_loader,
                                  test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_online_user_profiler")


@skip_no_data
class TestOnlineCVAE:
    """OnlineCVAE lifecycle tests (synthetic data)."""
    def test_online_cvae(self, cvae_model, train_loader, val_loader,
                         test_loader, normalizer, device):
        _run_validator_suite(cvae_model, train_loader, val_loader,
                             test_loader, normalizer, device,
                             "_test_online_cvae")


# ==================================================================
# Full integration test via run_validation()
# ==================================================================

@skip_no_data
class TestRunValidationIntegration:
    """End-to-end: calls run_validation() and asserts success."""

    def test_run_validation_cvae(self):
        from src.test.run_validation import run_validation
        assert run_validation(model_type="cvae", use_test_data=True, verbose=True)

    def test_run_validation_vae(self):
        from src.test.run_validation import run_validation
        assert run_validation(model_type="vae", use_test_data=True, verbose=True)
