from model.model import BaseModel
import torch


class RealDataBaselineModel(BaseModel):
    """Stores 500k real training windows (reservoir sampling, bucketed by conditions).
    sample() draws randomly from the matching condition bucket — no generation involved.
    reconstruct() is identity, giving RMSE=0 by construction.
    Use this as an evaluation oracle: its metric scores represent what a perfect
    generative model would achieve if it reproduced the training distribution exactly.
    """

    def __init__(
        self,
        phased_mode: bool = False,
        sequence_length: int = 24,
        device: str = "cpu",
        conditioner=None,
        reservoir_size: int = 500_000,
        min_bucket_samples: int = 30,
        **kwargs,
    ):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.device = device
        self.conditioner = conditioner
        self.reservoir_size = reservoir_size
        self.min_bucket_samples = min_bucket_samples
        self.normalizer = None  # injected at inference time by SyntheticGenerator
        self.tag = "baseline_real_data" + ("_phased" if phased_mode else "")

        if conditioner is not None:
            self._bucket_tags, self._bucket_slices, self._bucket_cardinalities = \
                AverageBaselineModel._compute_bucket_metadata(conditioner)
            self._num_buckets = 1
            for c in self._bucket_cardinalities:
                self._num_buckets *= c
            self._num_buckets = max(self._num_buckets, 1)
        else:
            self._bucket_tags = []
            self._bucket_slices = []
            self._bucket_cardinalities = []
            self._num_buckets = 0

        self.register_buffer("n_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "reservoir",
            torch.empty((reservoir_size, sequence_length, self.num_features), dtype=torch.float32),
        )
        self.register_buffer(
            "reservoir_bucket_ids",
            torch.zeros(reservoir_size, dtype=torch.long),
        )

    def _conditions_to_bucket_indices(self, conditions: torch.Tensor) -> torch.Tensor:
        """Convert [B, D] condition vector to [B] mixed-radix bucket index."""
        B = conditions.shape[0]
        device = conditions.device
        indices = torch.zeros(B, dtype=torch.long, device=device)
        n = len(self._bucket_slices)
        for i, ((start, end), cardinality) in enumerate(zip(self._bucket_slices, self._bucket_cardinalities)):
            sliced = conditions[:, start:end]
            disc = sliced.argmax(dim=-1) if (end - start) > 1 else (sliced[:, 0] >= 0.5).long()
            stride = 1
            for j in range(i + 1, n):
                stride *= self._bucket_cardinalities[j]
            indices = indices + disc * stride
        return indices.clamp(0, max(self._num_buckets - 1, 0))

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=lr)

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        x, meter_id, timestamp, agg = self._unpack_batch(batch)
        x = x.transpose(1, 2).to(device=self.reservoir.device, dtype=self.reservoir.dtype)  # [B, F, L] -> [B, L, F]

        if self.conditioner is not None and self._num_buckets > 0:
            conditions = self.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamp)
            conditions = self.move_to_device(conditions, device=self.reservoir.device).float()
            bucket_ids = self._conditions_to_bucket_indices(conditions)
        else:
            bucket_ids = torch.zeros(x.shape[0], dtype=torch.long, device=self.reservoir.device)

        batch_size = x.shape[0]
        seen_so_far = int(self.n_seen.item())

        if seen_so_far < self.reservoir_size:
            fill_amount = min(batch_size, self.reservoir_size - seen_so_far)
            self.reservoir[seen_so_far: seen_so_far + fill_amount] = x[:fill_amount]
            self.reservoir_bucket_ids[seen_so_far: seen_so_far + fill_amount] = bucket_ids[:fill_amount]
            self.n_seen += fill_amount
            remaining = batch_size - fill_amount
            if remaining > 0:
                self._stream_update(x[fill_amount:], bucket_ids[fill_amount:])
        else:
            self._stream_update(x, bucket_ids)

    @torch.no_grad()
    def _stream_update(self, batch: torch.Tensor, bucket_ids: torch.Tensor):
        batch = batch.to(device=self.reservoir.device, dtype=self.reservoir.dtype)
        batch_size = batch.shape[0]
        stream_start = int(self.n_seen.item()) + 1
        stream_end = stream_start + batch_size

        stream_indices = torch.arange(stream_start, stream_end, device=self.reservoir.device)
        random_rolls = torch.rand(batch_size, device=self.reservoir.device) * stream_indices
        keep_mask = random_rolls < self.reservoir_size
        if keep_mask.any():
            kept_items = batch[keep_mask]
            kept_bucket_ids = bucket_ids[keep_mask]
            num_kept = kept_items.shape[0]
            replace_indices = torch.randint(0, self.reservoir_size, (num_kept,), device=self.reservoir.device)
            self.reservoir[replace_indices] = kept_items
            self.reservoir_bucket_ids[replace_indices] = kept_bucket_ids
        self.n_seen += batch_size

    @torch.no_grad()
    def _sample_by_bucket(self, bucket_indices: torch.Tensor) -> torch.Tensor:
        """Sample one window per entry in bucket_indices from the reservoir."""
        n_filled = min(int(self.n_seen.item()), self.reservoir_size)
        n_filled = max(n_filled, 1)
        filled_bucket_ids = self.reservoir_bucket_ids[:n_filled]
        B = bucket_indices.shape[0]
        result = torch.zeros(
            B, self.sequence_length, self.num_features,
            device=self.reservoir.device, dtype=torch.float32,
        )
        for b in bucket_indices.unique():
            req_mask = bucket_indices == b
            n_req = int(req_mask.sum().item())
            avail_mask = filled_bucket_ids == b
            n_avail = int(avail_mask.sum().item())
            if n_avail >= self.min_bucket_samples:
                avail_idx = avail_mask.nonzero(as_tuple=True)[0]
                chosen = avail_idx[torch.randint(0, n_avail, (n_req,), device=self.reservoir.device)]
            else:
                chosen = torch.randint(0, n_filled, (n_req,), device=self.reservoir.device)
            result[req_mask] = self.reservoir[chosen]
        return result

    def train_core(self, batch, optim=None, gradient_clip=0.0, gradient_clip_max_norm=0.0,
                   beta=0.0, weighted_loss=0.0):
        self.update(batch)
        return {
            "loss": torch.tensor(0.0, device=self.reservoir.device),
            "recon_loglikelihood": torch.tensor(0.0, device=self.reservoir.device),
            "kl_divergence": torch.tensor(0.0, device=self.reservoir.device),
        }

    def validate(self, valloader, num_samples=1, beta=1, device="cpu", **kwargs):
        return {
            "loss": torch.tensor(0.0, device=device),
            "recon_loglikelihood": torch.tensor(0.0, device=device),
            "kl_divergence": torch.tensor(0.0, device=device),
        }

    @torch.no_grad()
    def sample(self, conditions=None, num_prior_samples=1, num_samples_prior=1,
               num_x_samples=1, device="cpu", **kwargs):
        n_filled = min(int(self.n_seen.item()), self.reservoir_size)
        n_filled = max(n_filled, 1)

        if self.conditioner is not None and conditions is not None:
            conditions = self.move_to_device(conditions, device=self.reservoir.device).float()
            bucket_indices = self._conditions_to_bucket_indices(conditions)
            samples = self._sample_by_bucket(bucket_indices)  # [B, L, F]
            if self.phased_mode:
                samples = samples.transpose(1, 2)  # [B, F, L]
            B, d1, d2 = samples.shape
            return samples.reshape(1, 1, B, d1 * d2)

        num_samples = kwargs.get("num_samples", max(num_prior_samples, num_samples_prior))
        idx = torch.randint(0, n_filled, (num_samples,), device=self.reservoir.device)
        samples = self.reservoir[idx]  # [N, L, F]
        if self.phased_mode:
            samples = samples.transpose(1, 2)  # [N, F, L]
        return samples.reshape(num_samples, 1, 1, self.sequence_length * self.num_features)

    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        """Identity reconstruction — returns the input unchanged (RMSE = 0 by construction)."""
        flat = x.reshape(x.shape[0], -1)  # [B, F*L]
        return flat.unsqueeze(0).expand(num_samples, -1, -1)  # [S, B, F*L]

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
                 conditioner=None,
                 min_bucket_samples: int = 120,
                 force_global_mean: bool = False,
                 **kwargs):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length, **kwargs)
        self.device = device
        self.normalizer = None  # injected at inference time by SyntheticGenerator
        self.conditioner = conditioner
        self.min_bucket_samples = min_bucket_samples
        self.force_global_mean = force_global_mean

        if conditioner is not None:
            self._bucket_tags, self._bucket_slices, self._bucket_cardinalities = \
                self._compute_bucket_metadata(conditioner)
            self._num_buckets = 1
            for c in self._bucket_cardinalities:
                self._num_buckets *= c
            self._num_buckets = max(self._num_buckets, 1)
            self.tag = "baseline_average_conditional" + ("_phased" if phased_mode else "")
        else:
            self._bucket_tags = []
            self._bucket_slices = []
            self._bucket_cardinalities = []
            self._num_buckets = 0
            self.tag = "baseline_average" + ("_phased" if phased_mode else "")

        self.register_buffer(
            "running_sum",
            torch.zeros((sequence_length, self.num_features), dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "running_count",
            torch.zeros((sequence_length, self.num_features), dtype=torch.long, device=device)
        )

        if self._num_buckets > 0:
            self.register_buffer(
                "bucket_sum",
                torch.zeros((self._num_buckets, sequence_length, self.num_features), dtype=torch.float64, device=device)
            )
            self.register_buffer(
                "bucket_count",
                torch.zeros((self._num_buckets, sequence_length, self.num_features), dtype=torch.long, device=device)
            )

    @staticmethod
    def _compute_bucket_metadata(conditioner):
        """Extract (tags, slices, cardinalities) for all non-user-profile conditions."""
        tags = [t for t in conditioner.get_condition_tags() if 'user_profile' not in t.lower()]
        slices, cardinalities = [], []
        for tag in tags:
            start, end = conditioner.get_condition_indexes(tag)
            dim = end - start
            # one-hot → cardinality = dim; binary scalar → cardinality = 2
            cardinalities.append(dim if dim > 1 else 2)
            slices.append((start, end))
        return tags, slices, cardinalities

    def _conditions_to_bucket_indices(self, conditions: torch.Tensor) -> torch.Tensor:
        """Convert [B, D] conditions tensor to [B] mixed-radix bucket indices."""
        B = conditions.shape[0]
        device = conditions.device
        indices = torch.zeros(B, dtype=torch.long, device=device)
        n = len(self._bucket_slices)
        for i, ((start, end), cardinality) in enumerate(zip(self._bucket_slices, self._bucket_cardinalities)):
            sliced = conditions[:, start:end]
            disc = sliced.argmax(dim=-1) if (end - start) > 1 else (sliced[:, 0] >= 0.5).long()
            stride = 1
            for j in range(i + 1, n):
                stride *= self._bucket_cardinalities[j]
            indices = indices + disc * stride
        return indices.clamp(0, self._num_buckets - 1)

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        """Processes a new batch and updates global and per-bucket statistics."""
        x, meter_id, timestamp, agg = self._unpack_batch(batch)
        x = x.transpose(1, 2).to(device=self.running_sum.device, dtype=self.running_sum.dtype)  # (B, F, T) -> (B, T, F)

        # Global stats (always kept for fallback)
        self.running_sum += x.sum(dim=0)
        self.running_count += x.shape[0] * torch.ones_like(
            self.running_count, dtype=self.running_count.dtype)

        # Per-bucket stats
        if self.conditioner is not None and self._num_buckets > 0:
            conditions = self.conditioner.get_conditions(x, meter_id=meter_id, timestamps=timestamp)
            conditions = self.move_to_device(conditions, device=self.running_sum.device).float()
            bucket_indices = self._conditions_to_bucket_indices(conditions)  # [B]

            B = x.shape[0]
            idx = bucket_indices.view(B, 1, 1).expand(B, self.sequence_length, self.num_features)
            self.bucket_sum.scatter_add_(0, idx, x)
            ones = torch.ones(B, self.sequence_length, self.num_features,
                              dtype=self.bucket_count.dtype, device=self.bucket_count.device)
            self.bucket_count.scatter_add_(0, idx, ones)

    @torch.no_grad()
    def get_global_mean(self) -> torch.Tensor:
        """Computes the exact global average across the dataset."""
        safe_counts = torch.clamp(self.running_count, min=1)
        mean = self.running_sum / safe_counts
        mean[self.running_count == 0] = 0.0
        return mean.to(torch.float32)

    @torch.no_grad()
    def get_bucket_mean(self, bucket_indices: torch.Tensor) -> torch.Tensor:
        """Returns per-bucket mean [B, L, F], falling back to global mean if count < min_bucket_samples or force_global_mean is set."""
        global_mean = self.get_global_mean()  # [L, F]
        result = global_mean.unsqueeze(0).expand(len(bucket_indices), -1, -1).clone()
        if self.force_global_mean:
            return result
        # Vectorised: gather bucket stats for all requested indices at once
        _bc: torch.Tensor = self.bucket_count  # type: ignore[assignment]
        _bs: torch.Tensor = self.bucket_sum    # type: ignore[assignment]
        bucket_counts = _bc[bucket_indices]                        # [B, L, F]
        bucket_sums   = _bs[bucket_indices]                        # [B, L, F]
        has_enough    = bucket_counts[:, 0, 0] >= self.min_bucket_samples  # [B]
        safe_counts   = bucket_counts.clamp(min=1).float()
        bucket_means  = (bucket_sums / safe_counts).float()        # [B, L, F]
        result[has_enough] = bucket_means[has_enough]
        return result

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
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
            "loss": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "recon_loglikelihood": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
            "kl_divergence": torch.tensor(0.0, device=self.running_sum.device, requires_grad=False),
        }

    def validate(self, valloader, num_samples=1, beta=1, device="cpu", **kwargs):
        buffer_device = self.running_sum.device
        return {
            "loss": torch.tensor(0.0, device=device, requires_grad=False),
            "recon_loglikelihood": torch.tensor(0.0, device=buffer_device, requires_grad=False),
            "kl_divergence": torch.tensor(0.0, device=buffer_device, requires_grad=False),
        }

    @torch.no_grad()
    def sample(self, conditions=None, num_prior_samples=1, num_samples_prior=1,
               num_x_samples=1, device='cpu', **kwargs):
        if self.conditioner is not None and conditions is not None:
            # Conditioned path — returns [1, 1, B, D] to match CVAE contract
            conditions = self.move_to_device(conditions, device=self.running_sum.device).float()
            bucket_indices = self._conditions_to_bucket_indices(conditions)
            means = self.get_bucket_mean(bucket_indices)  # [B, L, F]
            if self.phased_mode:
                means = means.transpose(1, 2)  # [B, F, L]
            B, d1, d2 = means.shape
            return means.reshape(1, 1, B, d1 * d2)

        # Unconditional path — returns [N, 1, 1, D]
        num_samples = kwargs.get('num_samples', max(num_prior_samples, num_samples_prior))
        global_mean = self.get_global_mean()  # [L, F]
        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]
        return global_mean.reshape(1, 1, 1, -1).expand(num_samples, -1, -1, -1)

    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        if self.conditioner is not None:
            x_phase_first = self._to_phase_first(x)
            cond_input = self._build_condition_input(x_phase_first)  # [B, L, F]
            conditions = self.conditioner.get_conditions(cond_input, meter_id=meter_ids, timestamps=timestamps)
            conditions = self.move_to_device(conditions, device=self.running_sum.device).float()
            bucket_indices = self._conditions_to_bucket_indices(conditions)
            means = self.get_bucket_mean(bucket_indices)  # [B, L, F]
            if self.phased_mode:
                means = means.transpose(1, 2)  # [B, F, L]
            B, d1, d2 = means.shape
            return means.reshape(1, B, d1 * d2).expand(num_samples, -1, -1)

        # Unconditional path
        global_mean = self.get_global_mean()  # [L, F]
        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]
        batch_size = x.shape[0]
        return global_mean.reshape(1, 1, -1).expand(num_samples, batch_size, -1)

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
    
    @torch.no_grad()
    def reconstruct(self, x, meter_ids=None, timestamps=None, num_samples=1, **kwargs):
        # Reconstructions ignore the input values and return the dataset-wide mean
        global_mean = self.get_global_mean()  # [L, F]

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

        if self.phased_mode:
            global_mean = global_mean.transpose(0, 1)  # [F, L]

        return global_mean.reshape(1, 1, 1, -1).expand(num_samples, -1, -1, -1)

    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        # Dummy optimizer to prevent the PyTorch optimizer crash in fit()
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.Adam([dummy_param], lr=lr)