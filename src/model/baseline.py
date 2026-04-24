from model.model import BaseModel
import torch

class BaselineModel(BaseModel):
    def __init__(self, phased_mode: bool = False, 
                 sequence_length: int = 24, 
                 method: str = "quantile",
                 **kwargs):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.method = method

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        return super().get_optimizer(lr, weight_decay, *args, **kwargs)
    
    def get_tag(self):
        return "baseline_" + self.method

    def train_core(self, optim=None,
                    gradient_clip=0.0,
                    gradient_clip_max_norm=0.0,
                    beta=0.0,
                    weighted_loss=0.0):
        pass

    def validate(self, valloader, num_samples = 1, beta=1, device = "cpu", **kwargs):
        pass

    def sample(self, num_samples_prior = 1, num_samples_posterior = 1, **kwargs):
        pass

    def reconstruct(self, inputs, num_samples=1, **kwargs):
        pass


 
def _build_quantile_profiles_per_feature(train_flat: np.ndarray, window_size: int, n_profiles: int = 100) -> np.ndarray:
    """Build quantile profiles per feature, returns [n_profiles, window_size, n_features]."""
    if n_profiles <= 0:
        raise ValueError('n_profiles must be positive.')
 
    windows = _reshape_flat_to_windows(train_flat, window_size=window_size)
    n_features = windows.shape[2]
    qs = np.linspace(1.0 / n_profiles, 1.0, n_profiles)
    out = np.zeros((n_profiles, window_size, n_features), dtype=np.float64)
 
    for f in range(n_features):
        for t in range(window_size):
            col = windows[:, t, f]
            col = col[np.isfinite(col)]
            non_zero = col[col != 0.0]
            base = non_zero if non_zero.size > 0 else col
            if base.size == 0:
                out[:, t, f] = 0.0
            else:
                out[:, t, f] = np.quantile(base, qs, method='linear')
 
    return _sanitize_numeric_array(out)