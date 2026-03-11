import argparse
import os
import sys
from pathlib import Path
import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config
from data.dataloader import SmartMeterDataModule, WindowSpec
from data.normalizer import Normalizer
from src.models import VAE, CVAE
from data.conditioner import Conditioner, UserProfiler, DayOfWeekConditioner
import torch


def main():
    """
    Main training script using configuration system.
    
    Usage:
        python main.py                                          # Uses config
        python main.py --config config.yaml --device cuda       # Override with CLI args
    """
    
    # ========== Parse Command Line Arguments ==========
    parser = argparse.ArgumentParser(
        description='Train VAE/CVAE on smart meter data using configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" Examples:
  python main.py                                                  # Use config
  python main.py --config config.yaml --device cuda --epochs 100  # Override settings
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON). Default: config/config.yaml'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device setting (cpu or cuda)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=None,
        help='Override latent dimension'
    )
    
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='Disable early stopping'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Print configuration summary and exit'
    )
    
    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    print("Loading configuration...")
    config = load_config(config_path=args.config)
    
    # Override config with CLI arguments
    if args.device:
        config.loader.config['model']['device'] = args.device
    if args.epochs:
        config.loader.config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config.loader.config['data']['dataloader']['batch_size'] = args.batch_size
    if args.lr:
        config.loader.config['training']['learning_rate'] = args.lr
    if args.latent_dim:
        config.loader.config['model']['latent_dim'] = args.latent_dim
    if args.no_early_stopping:
        config.loader.config['training']['early_stopping']['enabled'] = False
    
    # ========== Print Configuration ==========
    config.print_summary()
    
    if args.show_config:
        print("\nFull configuration:")
        print(yaml.dump(config.loader.config, default_flow_style=False))
        return
    
    # ========== Setup Device ==========
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # ========== Load Data ==========
    print("\nLoading data...")
    data_kwargs = config.to_data_kwargs()
    data_module = SmartMeterDataModule(**data_kwargs)
    data_module.setup()  # Initialize datasets
    
    # Get dataloaders
    trainloader = data_module.train_dataloader()
    valloader = data_module.val_dataloader()
    
    # Auto-detect input_dim if not specified
    if config.input_dim is None:
        # Infer from first batch
        for x, _, _ in trainloader:
            if x.dim() == 3:
                B, L, F = x.shape
                input_dim = L * F
            else:
                input_dim = x.shape[-1]
            break
        print(f"Auto-detected input_dim: {input_dim}")
    else:
        input_dim = config.input_dim
    
    # ========== Setup Normalization ==========
    print("Setting up normalization...")
    normalizer = Normalizer(method=config.loader.get('normalization.method', 'log1p'))
    
    # ========== Create Model ==========
    print(f"\nCreating {config.model_type.upper()} model...")
    
    distribution_dict = {
        'latent_dim': config.latent_dim,
        'dictionary_size': config.loader.get('model.prior.dictionary_size', 100),
        'vamp_num_components': config.loader.get('model.prior.vamp_num_components', 20),
    }
    
    if config.model_type == 'vae':
        model = VAE(
            input_dim=input_dim,
            distribution_dict=distribution_dict,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            prior_type=config.prior_type,
            device=str(device),
        )
    
    elif config.model_type == 'cvae':
        if not config.conditioning_enabled:
            print("Warning: CVAE selected but conditioning disabled. Enabling conditioning...")
            config.loader.config['conditioning']['enabled'] = True
        
        # Setup conditioner
        conditioner = Conditioner()
        
        # Add user profiler
        if config.loader.get('conditioning.user_profile.enabled', True):
            user_profiler = UserProfiler(
                n_topics=config.n_topics,
                num_clusters=config.num_clusters,
                random_state=config.loader.get('advanced.seed', 42),
                path=config.loader.get('conditioning.user_profile.cache_path'),
                verbose=config.loader.get('conditioning.user_profile.verbose', False),
            )
            conditioner.add_condition_item('user_profile', user_profiler)
        
        # Add day-of-week conditioner
        if config.loader.get('conditioning.day_of_week.enabled', True):
            dow_conditioner = DayOfWeekConditioner()
            conditioner.add_condition_item('day_of_week', dow_conditioner)
        
        model = CVAE(
            input_dim=input_dim,
            conditioner=conditioner,
            distribution_dict=distribution_dict,
            encoder_type=config.encoder_type,
            decoder_type=config.decoder_type,
            prior_type=config.prior_type,
            device=str(device),
        )
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    print(f"Model created with {model.num_parameters:,} parameters")
    print(f"Model device: {next(model.parameters()).device}")
    
    # ========== Training ==========
    print("\nStarting training...")
    training_kwargs = config.to_training_kwargs()
    
    model_tag = model.fit(
        trainloader=trainloader,
        valloader=valloader,
        **training_kwargs,
    )
    
    print(f"\n✓ Training complete: {model_tag}")
    
    # ========== Save Configuration ==========
    if config.loader.get('logging.save_config', True):
        config_save_path = os.path.join(config.save_dir, f"{model_tag}_config.yaml")
        config.loader.save(config_save_path, format='yaml')
        print(f"✓ Configuration saved: {config_save_path}")


if __name__ == "__main__":
    main()
