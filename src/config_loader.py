"""
Configuration loader for VAE/CVAE training and inference.
Handles loading, validation, and merging of YAML/JSON configuration files.
"""

import os
import json
import yaml
from typing import Any, Dict, Optional, List
import torch
import warnings
 

class AttrDict:
    """Wraps a dict for recursive attribute-style access."""
    def __init__(self, d):
        object.__setattr__(self, '_d', d)

    def __getattr__(self, key):
        try:
            val = object.__getattribute__(self, '_d')[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        return AttrDict(val) if isinstance(val, dict) else val

    def __repr__(self):
        return f"AttrDict({object.__getattribute__(self, '_d')})"


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigLoader:
    """
    Load and manage configuration from YAML or JSON files.
    """
    
    def __init__(self, config_path: Optional[str] = None, defaults_path: Optional[str] = None):
        """
        Initialize config loader.
        Args:
            config_path: Path to user config file (YAML or JSON)
        """
        self.config_path = config_path
        self.defaults_path = defaults_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files."""
        # Load defaults first
        if self.defaults_path:
            if not os.path.exists(self.defaults_path):
                raise ConfigError(f"Default config file not found: {self.defaults_path}")
            self.config = self._load_file(self.defaults_path)
        
        # Merge with user config if provided
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise ConfigError(f"Config file not found: {self.config_path}")
            user_config = self._load_file(self.config_path)
            self.config = self._deep_merge(self.config, user_config)
    
    def _load_file(self, path: str) -> Dict:
        """Load YAML or JSON file."""
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ConfigError(f"Unsupported file format: {path}. Use .yaml, .yml, or .json")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation (e.g., 'data.batch_size').
        
        Args:
            key: Configuration key with dot notation for nested access
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of validation warning messages
        """
        warnings_list = []
        
        # Data validation
        if not os.path.exists(self.get('data.data_dir')):
            warnings_list.append(f"Data directory not found: {self.get('data.data_dir')}")
        
        # Device validation
        device = self.get('model.device')
        if device == 'cuda' and not torch.cuda.is_available():
            warnings_list.append(f"CUDA requested but not available. Falling back to CPU.")
        
        # Model validation
        if self.get('model.latent_dim') <= 0:
            warnings_list.append(f"latent_dim should be > 0, got {self.get('model.latent_dim')}")
        
        # Training validation
        if self.get('training.learning_rate') <= 0:
            warnings_list.append(f"learning_rate should be > 0")
        
        if self.get('training.num_epochs') <= 0:
            warnings_list.append(f"num_epochs should be > 0")
        
        # Data split validation
        splits = self.get_section('data')['splits']
        total = splits['train_ratio'] + splits['val_ratio'] + splits['test_ratio']
        if abs(total - 1.0) > 1e-6:
            warnings_list.append(f"Data splits don't sum to 1.0: {total}")
        
        return warnings_list
    
    def to_dict(self) -> Dict:
        """Return config as dictionary."""
        return self.config.copy()
    
    def save(self, path: str, format: str = 'yaml'):
        """
        Save current config to file.
        
        Args:
            path: Path to save file
            format: 'yaml' or 'json'
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        if format == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ConfigError(f"Unsupported format: {format}")
    
    def __getattr__(self, name):
        # Only called when normal attribute lookup fails (i.e. not config/config_path/defaults_path)
        try:
            val = self.config[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        return AttrDict(val) if isinstance(val, dict) else val

    def __repr__(self):
        return f"ConfigLoader(config_path={self.config_path}, sections={list(self.config.keys())})"


class ExperimentConfig:
    """
    Accessing configuration with type hints and validation.
    """
    
    def __init__(self, loader: ConfigLoader):
        """Initialize with a ConfigLoader instance."""
        self.loader = loader
    
    # ========== DATA PROPERTIES ==========
    @property
    def data_dir(self) -> str:
        return self.loader.get('data.data_dir', './data')
    
    @property
    def data_filename(self) -> str:
        return self.loader.get('data.filename', 'data.csv')
    
    @property
    def batch_size(self) -> int:
        return int(self.loader.get('data.dataloader.batch_size', 64))
    
    @property
    def window_size(self) -> int:
        return int(self.loader.get('data.window.input_len', 24))
    
    @property
    def window_stride(self) -> int:
        return int(self.loader.get('data.window.stride', 24))
    
    @property
    def train_val_test_split(self) -> tuple:
        splits = self.loader.get_section('data')['splits']
        return (splits['train_ratio'], splits['val_ratio'], splits['test_ratio'])
    
    # ========== MODEL PROPERTIES ==========
    @property
    def model_type(self) -> str:
        """Return 'vae' or 'cvae'."""
        return self.loader.get('model.type', 'vae').lower()
    
    @property
    def input_dim(self) -> Optional[int]:
        """Return input_dim (None means auto-detect)."""
        val = self.loader.get('model.input_dim')
        return int(val) if val is not None else None
    
    @property
    def latent_dim(self) -> int:
        return int(self.loader.get('model.latent_dim', 32))
    
    @property
    def encoder_type(self) -> str:
        return self.loader.get('model.encoder.type', 'dictionary')
    
    @property
    def decoder_type(self) -> str:
        return self.loader.get('model.decoder.type', 'dictionary')
    
    @property
    def prior_type(self) -> str:
        return self.loader.get('model.prior.type', 'standard_dictionary')
    
    @property
    def device(self) -> str:
        device = self.loader.get('model.device', 'cpu').lower()
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available. Falling back to CPU.")
            return 'cpu'
        return device
    
    # ========== CONDITIONING PROPERTIES ==========
    @property
    def conditioning_enabled(self) -> bool:
        return self.loader.get('conditioning.enabled', False)
    
    @property
    def n_topics(self) -> int:
        """Number of topics for user profiling."""
        return int(self.loader.get('conditioning.user_profile.n_topics', 10))
    
    @property
    def num_clusters(self) -> int:
        """Number of clusters for user profiling."""
        return int(self.loader.get('conditioning.user_profile.num_clusters', 100))
    
    # ========== TRAINING PROPERTIES ==========
    @property
    def learning_rate(self) -> float:
        return float(self.loader.get('training.learning_rate', 1e-3))
    
    @property
    def weight_decay(self) -> float:
        return float(self.loader.get('training.weight_decay', 1e-4))
    
    @property
    def num_epochs(self) -> int:
        return int(self.loader.get('training.num_epochs', 1000))
    
    @property
    def beta(self) -> float:
        return float(self.loader.get('training.beta', 1.0))
    
    @property
    def beta_annealing_enabled(self) -> bool:
        return self.loader.get('training.beta_annealing.enabled', False)
    
    @property
    def early_stopping_enabled(self) -> bool:
        return self.loader.get('training.early_stopping.enabled', True)
    
    @property
    def early_stopping_patience(self) -> int:
        return int(self.loader.get('training.early_stopping.patience', 15))
    
    @property
    def gradient_clip(self) -> bool:
        return self.loader.get('training.gradient_clip', True)
    
    @property
    def gradient_clip_max_norm(self) -> float:
        return float(self.loader.get('training.gradient_clip_max_norm', 1.0))
    
    @property
    def weighted_loss(self) -> float:
        return float(self.loader.get('training.weighted_loss', 0.0))
    
    @property
    def validation_freq(self) -> int:
        return int(self.loader.get('training.validation.validation_freq', 1))
    
    @property
    def save_dir(self) -> str:
        return self.loader.get('training.checkpointing.save_dir', './models')
    
    # ========== UTILITY METHODS ==========
    def to_training_kwargs(self) -> Dict:
        """Convert config to kwargs for model.fit()."""
        # Get beta annealing config and remove 'enabled' key (it's just a flag)
        beta_anneal_config = self.loader.get('training.beta_annealing', {}).copy()
        beta_anneal_config.pop('enabled', None)
        
        # Get early stopping config
        early_stop_config = self.loader.get('training.early_stopping', {}).copy()
        early_stop_config.pop('enabled', None)
        
        return {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.num_epochs,
            'beta': self.beta,
            'beta_annealing': self.beta_annealing_enabled,
            'beta_annealing_kwargs': beta_anneal_config,
            'early_stopping': self.early_stopping_enabled,
            'early_stopping_kwargs': {
                'patience': self.early_stopping_patience,
                'min_delta': early_stop_config.get('min_delta', 0.001),
            },
            'gradient_clip': self.gradient_clip,
            'gradient_clip_kwargs': {'max_norm': self.gradient_clip_max_norm},
            'validation_freq': self.validation_freq,
            'device': self.device,
            'save_dir': self.save_dir,
            'save_epoch_freq': self.loader.get('training.checkpointing.save_freq', 10),
            'weighted_loss': self.weighted_loss,
        }
    
    def to_dataloader_kwargs(self) -> Dict:
        """Convert config to kwargs for DataLoader creation."""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.loader.get('data.dataloader.num_workers', 0),
        }
    
    def to_data_kwargs(self) -> Dict:
        """Convert config to kwargs for data loading."""
        from data.dataloader import WindowSpec
        
        data_config = self.loader.get_section('data')
        return {
            'data_dir': self.data_dir,
            'filename': self.data_filename,
            'meter_col': data_config['columns']['meter_id'],
            'timestamp_dst_col': data_config['columns']['timestamp_dst'],
            'timestamp_utc_col': data_config['columns']['timestamp_utc'],
            'feature_cols': data_config['columns']['value_cols'],
            'window': WindowSpec(
                input_len=self.window_size,
                stride=self.window_stride,
            ),
            'train_frac': self.train_val_test_split[0],
            'val_frac': self.train_val_test_split[1],
            'batch_size': self.batch_size,
            'seed': self.loader.get('advanced.seed', 42),
        }
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"\nData:")
        print(f"  Directory: {self.data_dir}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Window: {self.window_size}h with stride {self.window_stride}h")
        print(f"  Splits: {self.train_val_test_split}")
        
        print(f"\nModel:")
        print(f"  Type: {self.model_type.upper()}")
        print(f"  Latent Dim: {self.latent_dim}")
        print(f"  Encoder: {self.encoder_type}")
        print(f"  Decoder: {self.decoder_type}")
        print(f"  Prior: {self.prior_type}")
        print(f"  Device: {self.device}")
        
        if self.conditioning_enabled:
            print(f"\nConditioning:")
            print(f"  User Topics: {self.n_topics}")
            print(f"  User Clusters: {self.num_clusters}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Beta: {self.beta}")
        print(f"  Beta Annealing: {self.beta_annealing_enabled}")
        print(f"  Early Stopping: {self.early_stopping_enabled}")
        print(f"  Weighted Loss: {self.weighted_loss}")
        
        print("\n" + "="*70 + "\n")


def load_config(config_path: Optional[str] = None, 
                defaults_path: Optional[str] = None) -> ExperimentConfig:
    """
    Convenient function to load configuration.
    
    Args:
        config_path: User config file path (YAML or JSON)
        defaults_path: Defaults config file path (YAML or JSON)
    
    Returns:
        ExperimentConfig instance
    """
    # Use default path if not specified
    if defaults_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        defaults_path = os.path.join(os.path.dirname(script_dir), 'config', 'config.yaml')
    
    loader = ConfigLoader(config_path=config_path, defaults_path=defaults_path)
    
    # Validate and warn
    warnings_list = loader.validate()
    for warning in warnings_list:
        warnings.warn(warning)
    
    return ExperimentConfig(loader)
