import os
import tempfile
import pytest
import torch

from src.model.vae import VAE
from src.model.cvae import CVAE
from src.model.diffusion import DiffusionModel, NoiseScheduler, ConditionalUNet1D
from src.model.phase_splitter import HierarchicalSplitter
from src.data.conditioner import Conditioner, DayOfWeekConditioner
from src.data.normalizer import Normalizer

@pytest.fixture
def dummy_conditioner():
    cond = Conditioner()
    cond.add_condition_item("day", DayOfWeekConditioner())
    return cond

@pytest.fixture
def dummy_normalizer():
    norm = Normalizer(method="log1p")
    norm.fitted = True
    return norm

def verify_save_load_roundtrip(model, dummy_normalizer, tmpdir):
    """
    Helper function that saves the model to a temporary directory, 
    verifies the checkpoint format, completely duplicates the model,
    loads the checkpoint, and asserts state_dicts are identical.
    """
    # Save model
    model.save_model(save_path=tmpdir, model_name="test_chkpt", device="cpu", normalizer=dummy_normalizer)
    
    ckpt_path = os.path.join(tmpdir, "test_chkpt.pt")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created"
    
    # Verify checkpoint structure
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "format_version" in ckpt
    assert ckpt["format_version"] == 2
    assert "model_class" in ckpt
    assert ckpt["model_class"] == model.__class__.__name__
    assert "model_state_dict" in ckpt
    
    # Check normalizer and conditioner states
    assert "normalizer_state" in ckpt
    assert "conditioner_state" in ckpt
    assert ckpt["conditioner_state"] is not None, "conditioner_state is None inside checkpoint!"
    
    return ckpt_path

def test_vae_serialization():
    model = VAE(
        input_dim=24,
        distribution_dict={"latent_dim": 16, "dictionary_size": 50},
        encoder_type="dictionary",
        decoder_type="dictionary",
        device="cpu"
    )
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_model(save_path=tmpdir, model_name="vae", device="cpu")
        ckpt_path = os.path.join(tmpdir, "vae.pt")
        
        loaded_model = VAE(
            input_dim=24,
            distribution_dict={"latent_dim": 16, "dictionary_size": 50},
            encoder_type="dictionary",
            decoder_type="dictionary",
            device="cpu"
        )
        loaded_model.load_model(ckpt_path, device="cpu")
        
        orig_keys = set(model.state_dict().keys())
        loaded_keys = set(loaded_model.state_dict().keys())
        assert orig_keys == loaded_keys
        
        for k, v in model.state_dict().items():
            assert torch.allclose(v, loaded_model.state_dict()[k])

def test_cvae_with_conditioner_serialization(dummy_conditioner, dummy_normalizer):
    model = CVAE(
        input_dim=24,
        latent_dim=16,
        conditioner=dummy_conditioner,
        encoder_type="dictionary",
        decoder_type="dictionary",
        prior_type="standard_dictionary"
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = verify_save_load_roundtrip(model, dummy_normalizer, tmpdir)
        
        loaded_model = CVAE(
            input_dim=24,
            latent_dim=16,
            conditioner=dummy_conditioner,
            encoder_type="dictionary",
            decoder_type="dictionary",
            prior_type="standard_dictionary"
        )
        
        loaded_normalizer, loaded_conditioner = loaded_model.load_model(ckpt_path, device="cpu")
    
    # Verify conditioner roundtripped
    assert loaded_conditioner is not None
    assert "day" in loaded_conditioner.condition_items
    
    # Verify normalizer roundtripped
    assert loaded_normalizer is not None
    assert loaded_normalizer.method == dummy_normalizer.method

def test_diffusion_model_serialization(dummy_conditioner):
    scheduler = NoiseScheduler(
        num_timesteps=10, 
        beta_start=1e-4, 
        beta_end=0.02, 
        schedule_type="linear", 
        device="cpu"
    )
    denoiser = ConditionalUNet1D(
        channels=3,
        condition_dim=dummy_conditioner.get_condition_dim(),
        time_dim=16,
        base_channels=16,
        num_blocks=2
    )
    
    model = DiffusionModel(
        noise_scheduler=scheduler,
        model=denoiser,
        conditioner=dummy_conditioner,
        sequence_length=8,
        num_features=3
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = verify_save_load_roundtrip(model, None, tmpdir)
        
        # Load back
        loaded_scheduler = NoiseScheduler(
            num_timesteps=10, 
            beta_start=1e-4, 
            beta_end=0.02, 
            schedule_type="linear", 
            device="cpu"
        )
        loaded_denoiser = ConditionalUNet1D(
            channels=3,
            condition_dim=dummy_conditioner.get_condition_dim(),
            time_dim=16,
            base_channels=16,
            num_blocks=2
        )
        loaded_model = DiffusionModel(
            noise_scheduler=loaded_scheduler,
            model=loaded_denoiser,
            conditioner=dummy_conditioner,
            sequence_length=8,
            num_features=3
        )
        
        loaded_model.load_model(ckpt_path, device="cpu")
    
    # Specifically ensure the scheduler buffers were matched
    assert torch.allclose(model.noise_scheduler.betas, loaded_model.noise_scheduler.betas)
    assert torch.allclose(model.noise_scheduler.alphas_cumprod, loaded_model.noise_scheduler.alphas_cumprod)

def test_hierarchical_splitter_serialization(dummy_conditioner):
    model = HierarchicalSplitter(
        conditioner=dummy_conditioner,
        sequence_length=24,
        condition_dim=dummy_conditioner.get_condition_dim(),
        base_channels=16
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = verify_save_load_roundtrip(model, None, tmpdir)
        
        loaded_model = HierarchicalSplitter(
            conditioner=dummy_conditioner,
            sequence_length=24,
            condition_dim=dummy_conditioner.get_condition_dim(),
            base_channels=16
        )
        loaded_model.load_model(ckpt_path, device="cpu")
    for k, v in model.state_dict().items():
        assert torch.allclose(v, loaded_model.state_dict()[k])
