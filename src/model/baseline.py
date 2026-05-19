from model.model import BaseModel
import torch

class QuantileBaselineModel(BaseModel):
    """A simple baseline model that maintains a global reservoir of observed sequences and computes quantiles from it.
    This model does not learn any parameters and serves as a non-parametric baseline for comparison.
    The reservoir is updated using reservoir sampling to ensure a representative sample of the data stream.
    We cannot use exact quantiles of the entire dataset since we are in an online setting, so we maintain a fixed-size reservoir that is updated probabilistically as new data comes in.
    """
    def __init__(self, phased_mode: bool = False, 
                 sequence_length: int = 24, 
                    reservoir_size: int = 5000,
                    device: str = "cpu",
                 **kwargs):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.reservoir_size = reservoir_size
        self.device = device
        self.tag = "baseline_quantile" + str("_phased" if phased_mode else "")
        
        # Initialize an empty reservoir
        self.register_buffer("n_seen", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer(
            "reservoir", 
            torch.empty((reservoir_size, sequence_length, self.num_features), dtype=torch.float32, device=device)
        )
    
    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        """Processes a new batch and probabilistically updates the global reservoir."""
        x, meter_id, timestamp, agg = self._unpack_batch(batch)
        x = x.transpose(1, 2).to(device=self.reservoir.device, dtype=self.reservoir.dtype) # [B, F, L] -> [B, L, F]
            
        batch_size = x.shape[0]
        seen_so_far = int(self.n_seen.item())

        # Phase 1: Fill the reservoir until it's full
        if seen_so_far < self.reservoir_size:
            fill_amount = min(batch_size, self.reservoir_size - seen_so_far)
            self.reservoir[seen_so_far : seen_so_far + fill_amount] = x[:fill_amount]
            self.n_seen += fill_amount
            
            # If the batch had leftovers, process them as a stream
            remaining = batch_size - fill_amount
            if remaining > 0:
                self._stream_update(x[fill_amount:])
        else:
            # Phase 2: Reservoir is full, probabilistically replace items
            self._stream_update(x)
        
    @torch.no_grad()
    def _stream_update(self, batch: torch.Tensor):
        batch = batch.to(device=self.reservoir.device, dtype=self.reservoir.dtype)
        batch_size = batch.shape[0]
        stream_start = int(self.n_seen.item()) + 1
        stream_end = stream_start + batch_size
        
        # For each incoming item i, the probability of keeping it is (reservoir_size / i)
        # We calculate the "i" index for each item in the incoming batch
        stream_indices = torch.arange(stream_start, stream_end, device=self.reservoir.device)
        # Roll a random number between 0 and the current stream index
        random_rolls = torch.rand(batch_size, device=self.reservoir.device) * stream_indices
        # Keep items where the roll falls within the reservoir size
        keep_mask = random_rolls < self.reservoir_size
        if keep_mask.any():
            kept_items = batch[keep_mask]
            num_kept = kept_items.shape[0]
            
            # Choose random locations in the reservoir to overwrite
            replace_indices = torch.randint(0, self.reservoir_size, (num_kept,), device=self.reservoir.device)
            self.reservoir[replace_indices] = kept_items

        self.n_seen += batch_size
    
    @torch.no_grad()
    def get_global_quantiles(self, n_profiles: int = 100) -> torch.Tensor:
        """Computes the quantiles on the globally representative reservoir."""
        if int(self.n_seen.item()) == 0:
            # Return zeros if evaluated before training starts to avoid crashes
            return torch.zeros((n_profiles, self.sequence_length, self.num_features), device=self.reservoir.device)
            
        qs = torch.linspace(1.0 / n_profiles, 1.0, n_profiles, device=self.reservoir.device)
        valid_reservoir = self.reservoir[:min(self.n_seen.item(), self.reservoir_size)]
        
        masked_res = valid_reservoir.clone()
        invalid_mask = ~torch.isfinite(masked_res) | (masked_res == 0.0)
        masked_res[invalid_mask] = torch.nan
        
        valid_counts = (~invalid_mask).sum(dim=0)
        all_nan_mask = (valid_counts == 0)
        masked_res[:, all_nan_mask] = 0.0
        
        out = torch.nanquantile(masked_res, qs, dim=0)
        out[:, all_nan_mask] = 0.0
        
        return out

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        # REturn a dummy optimizer since this model has no learnable parameters
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=lr)
    
    def train_core(self, batch,
                   optim=None,
                    gradient_clip=0.0,
                    gradient_clip_max_norm=0.0,
                    beta=0.0,
                    weighted_loss=0.0):
        self.update(batch)
        return {
            "loss": torch.tensor(0.0, device=self.reservoir.device),
            "recon_loglikelihood": torch.tensor(0.0, device=self.reservoir.device),
            "kl_divergence": torch.tensor(0.0, device=self.reservoir.device),
        }

    def validate(self, valloader, num_samples = 1, beta=1, device = "cpu", **kwargs):
        return {"loss": torch.tensor(0.0, device=device), "recon_loglikelihood": torch.tensor(0.0, device=device), "kl_divergence": torch.tensor(0.0, device=device)}

    @torch.no_grad()
    def sample(self, num_samples_prior=100, num_samples_posterior=1, **kwargs):
        num_samples = kwargs.get('num_samples', num_samples_prior)
        profiles = self.get_global_quantiles(n_profiles=100)
        
        # randomly sample from the quantiles to create a diverse set of samples
        indices = torch.randint(0, profiles.shape[0], (num_samples,), device=profiles.device)
        samples = profiles[indices] # [N, L, F]
        
        if self.phased_mode:
            samples = samples.transpose(1, 2) # [N, F, L]
            
        # FIX: Explicitly define the flattened dimension to prevent the 0-element reshape crash
        flattened_dim = self.sequence_length * self.num_features
        
        # Returns shape: [num_samples, 1, 1, flattened_dim]
        return samples.reshape(num_samples, 1, 1, flattened_dim)

    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        """Baseline reconstruction: predict the median profile for everything."""
        profiles = self.get_global_quantiles(n_profiles=100)
        median_profile = profiles[50] # 50th percentile is the median
        
        if self.phased_mode:
            median_profile = median_profile.transpose(0, 1) # [F, L]
            
        batch_size = x.shape[0]
        # Return shape [S, B, D]
        reconstruction = median_profile.reshape(1, 1, -1).expand(num_samples, batch_size, -1)
        return reconstruction

class AverageBaselineModel(BaseModel):
    def __init__(self, phased_mode: bool = False,
                    sequence_length: int = 24,
                        device: str = "cpu",
                    **kwargs):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.device = device
        self.normalizer = None  # injected at inference time by SyntheticGenerator
        self.tag = "baseline_average" + str("_phased" if phased_mode else "")


        self.register_buffer(
            "running_sum",
            torch.zeros((sequence_length, self.num_features), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "running_count",
            torch.zeros((sequence_length, self.num_features), dtype=torch.long, device=device)
        )

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        """Processes a new batch and updates the global sum and count."""
        x, meter_id, timestamp, agg = self._unpack_batch(batch)
        x = x.transpose(1, 2).to(device=self.running_sum.device, dtype=self.running_sum.dtype)  # (B, F, T) -> (B, T, F)
        
        batch_sum = x.sum(dim=0)
        batch_count = x.shape[0] * torch.ones_like(batch_sum, dtype=self.running_count.dtype, device=self.running_count.device)
        
        # 4. Add to the global tracking buffers
        self.running_sum += batch_sum
        self.running_count += batch_count
    
    @torch.no_grad()
    def get_global_mean(self) -> torch.Tensor:
        """Computes the exact global average across the dataset."""
        # Prevent division by zero by clamping the denominator to 1
        safe_counts = torch.clamp(self.running_count, min=1)
        mean = self.running_sum / safe_counts
        
        # If a feature/timestep was entirely unseen across all data, set to 0.0
        mean[self.running_count == 0] = 0.0
        
        return mean.to(torch.float32)
    
    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        # Dummy optimizer to prevent the PyTorch optimizer crash in fit()
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=lr)
    
    def train_core(self, batch,
                   optim=None,
                   gradient_clip=0.0,
                   gradient_clip_max_norm=0.0,
                   beta=0.0,
                   weighted_loss=0.0):
        # Update the running global mean in a streaming fashion
        self.update(batch)
        return {
            "loss": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "recon_loglikelihood": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "kl_divergence": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
        }

    def validate(self, valloader, num_samples=1, beta=1, device="cpu", **kwargs):
        buffer_device = self.running_sum.device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=False), "recon_loglikelihood": torch.tensor(0.0, device=buffer_device, requires_grad=False), "kl_divergence": torch.tensor(0.0, device=buffer_device, requires_grad=False)}

    def _log1p_bias_correction(self, shape_lf: torch.Size) -> torch.Tensor:
        """Return per-feature Jensen correction (+σ/2 in z-space) for log1p normalization.

        For log1p normalization inverse_transform(0) = expm1(μ) = geometric mean, not
        the arithmetic mean.  Adding σ/2 in z-space gives
        expm1(σ/2 * σ + μ) = expm1(μ + σ²/2) ≈ E[x], recovering the true mean.
        Returns a [L, F] zero tensor when no correction is needed.
        """
        L, F = shape_lf
        zero = torch.zeros(L, F, dtype=torch.float32)
        if self.normalizer is None:
            return zero
        log_methods = getattr(self.normalizer, 'LOG_METHODS', set())
        if getattr(self.normalizer, 'method', None) not in log_methods:
            return zero
        std = getattr(self.normalizer, 'std', None)
        if std is None:
            return zero
        # std is [1, F]; correction is σ/2 per feature, broadcast to [L, F]
        correction_f = (std.reshape(-1).float() / 2.0).cpu()  # [F]
        return correction_f.unsqueeze(0).expand(L, F)  # [L, F]

    @torch.no_grad()
    def sample(self, num_samples_prior=1, num_samples_posterior=1, **kwargs):
        # Sampling simply returns the mean profile expanded to the requested size
        num_samples = kwargs.get('num_samples', num_samples_prior)
        global_mean = self.get_global_mean()  # [L, F]

        # Correct Jensen's inequality bias introduced by log1p normalization
        correction = self._log1p_bias_correction(global_mean.shape).to(global_mean.device)
        global_mean = global_mean + correction

        # Consistent with CVAE: return [S, num_x_samples, B, D]
        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]

        return global_mean.reshape(1, 1, 1, -1).expand(num_samples, -1, -1, -1)

    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        # Reconstructions ignore the input values and return the dataset-wide mean
        global_mean = self.get_global_mean()  # [L, F]

        correction = self._log1p_bias_correction(global_mean.shape).to(global_mean.device)
        global_mean = global_mean + correction

        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]

        batch_size = x.shape[0]
        # Return shape [S, B, D]
        reconstruction = global_mean.reshape(1, 1, -1).expand(num_samples, batch_size, -1)
        return reconstruction

class FlatMeanBaselineModel(BaseModel):
    """
    Predicts the mean of the dataset (at all timesteps at once. Just a pure mean over features) as a constant for every input. Uses train_core to update the mean.
    """
    def __init__(self, phased_mode: bool = False,
                    sequence_length: int = 24,
                    device: str = "cpu",
                    **kwargs):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.device = device
        self.normalizer = None  # injected at inference time by SyntheticGenerator
        self.tag = "flat_mean_baseline" + str("_phased" if phased_mode else "")

        self.register_buffer(
            "running_sum",
            torch.zeros((1, self.num_features), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "running_count",
            torch.zeros((1, self.num_features), dtype=torch.long, device=device)
        )
    
    def train_core(self, batch,
                   optim=None,
                   gradient_clip=0.0,
                   gradient_clip_max_norm=0.0,
                   beta=0.0,
                   weighted_loss=0.0):
        # Update the running global mean in a streaming fashion
        self.update(batch)
        return {
            "loss": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "recon_loglikelihood": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "kl_divergence": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
        }

    def validate(self, valloader, num_samples=1, beta=1, device="cpu", **kwargs):
        buffer_device = self.running_sum.device
        return {"loss": torch.tensor(0.0, device=device, requires_grad=False), "recon_loglikelihood": torch.tensor(0.0, device=buffer_device, requires_grad=False), "kl_divergence": torch.tensor(0.0, device=buffer_device, requires_grad=False)}
    
    def _log1p_bias_correction(self, shape_lf: torch.Size) -> torch.Tensor:
        """See AverageBaselineModel._log1p_bias_correction for details."""
        L, F = shape_lf
        zero = torch.zeros(L, F, dtype=torch.float32)
        if self.normalizer is None:
            return zero
        log_methods = getattr(self.normalizer, 'LOG_METHODS', set())
        if getattr(self.normalizer, 'method', None) not in log_methods:
            return zero
        std = getattr(self.normalizer, 'std', None)
        if std is None:
            return zero
        correction_f = (std.reshape(-1).float() / 2.0).cpu()
        return correction_f.unsqueeze(0).expand(L, F)

    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        # Reconstructions ignore the input values and return the dataset-wide mean
        global_mean = self.get_global_mean()  # [L, F]

        correction = self._log1p_bias_correction(global_mean.shape).to(global_mean.device)
        global_mean = global_mean + correction

        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]

        batch_size = x.shape[0]
        # Return shape [S, B, D]
        reconstruction = global_mean.reshape(1, 1, -1).expand(num_samples, batch_size, -1)
        return reconstruction

    def get_global_mean(self) -> torch.Tensor:
        """Computes the exact global average across the dataset."""
        # Prevent division by zero by clamping the denominator to 1
        safe_counts = torch.clamp(self.running_count, min=1)
        mean = self.running_sum / safe_counts
        
        # If a feature/timestep was entirely unseen across all data, set to 0.0
        mean[self.running_count == 0] = 0.0

        # expand the mean to the sequence length
        mean = mean.repeat(self.sequence_length, 1)
        
        return mean.to(torch.float32)
    
    def update(self, batch: torch.Tensor):
        """Processes a new batch and updates the global sum and count."""
        x, meter_id, timestamp, agg = self._unpack_batch(batch)
        x = x.to(device=self.running_sum.device, dtype=self.running_sum.dtype)
        # Sum over batch and sequence length to get total sum per feature
        # x is [B, F, L], we want [1, F]
        batch_sum = x.sum(dim=(0, 2)).unsqueeze(0)
        batch_count = (x.shape[0] * x.shape[2]) * torch.ones_like(batch_sum, dtype=self.running_count.dtype, device=self.running_count.device)
        
        self.running_sum += batch_sum
        self.running_count += batch_count
    
    def sample(self, num_samples_prior=1, num_samples_posterior=1, **kwargs):
        # Sampling simply returns the mean profile expanded to the requested size
        num_samples = kwargs.get('num_samples', num_samples_prior)
        global_mean = self.get_global_mean()  # [L, F]

        correction = self._log1p_bias_correction(global_mean.shape).to(global_mean.device)
        global_mean = global_mean + correction

        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]

        return global_mean.reshape(1, 1, 1, -1).expand(num_samples, -1, -1, -1)

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        # Dummy optimizer to prevent the PyTorch optimizer crash in fit()
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=lr)