import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import *

from model.model import BaseModel

class NoiseScheduler(nn.Module):
    """Noise schedule for diffusion models.

    Inherits from nn.Module so that all schedule tensors are automatically
    captured by the parent model's state_dict() via register_buffer().
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        cosine_s=0.008,
        max_beta=0.999,
        device="cpu",
    ):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        if self.num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {self.num_timesteps}")
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.schedule_type = str(schedule_type).lower()
        if self.schedule_type not in {"linear", "cosine"}:
            raise ValueError(f"Unsupported noise schedule '{self.schedule_type}'. Use 'linear' or 'cosine'.")
        self.cosine_s = float(cosine_s)
        self.max_beta = float(max_beta)
        self._init_buffers(torch.device(device))

    def _build_betas(self, device: torch.device) -> torch.Tensor:
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=device)

        steps = self.num_timesteps
        t = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float32)
        alpha_bar = torch.cos(((t / steps) + self.cosine_s) / (1.0 + self.cosine_s) * (math.pi * 0.5)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0].clamp(min=1e-12)
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-12))
        return betas.clamp(min=1e-8, max=self.max_beta)

    def _init_buffers(self, device: torch.device) -> None:
        betas = self._build_betas(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]],
            dim=0,
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)

        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev)) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Forward noising q(x_t | x_0). Inputs are expected in [B, F, L]."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_noise = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noise, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, time):
        time = time.float()
        device = time.device
        half_dim = self.dim // 2
        if half_dim < 1:
            raise ValueError(f"Sinusoidal embedding dimension must be >= 2, got {self.dim}")

        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        embeddings = time[:, None] * frequencies[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, time_channels, dilation):
        super().__init__()
        # Dilated convolution increases the receptive field exponentially
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_channels, channels)
        
    def forward(self, x, t_emb):
        # 1. Add time & condition embedding
        h = x + self.time_mlp(t_emb).unsqueeze(-1)
        # 2. First Conv (Dilated)
        h = F.gelu(self.conv1(h))
        # 3. Second Conv
        h = self.conv2(h)
        # 4. Residual Connection
        return x + h

class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        channels=3,
        condition_dim=32,
        time_dim=128,
        base_channels=128,
        num_blocks=4,
        self_attention_heads=4,
        use_condition_cross_attention=False,
        condition_attention_heads=4,
        condition_tokens=4,
    ):
        super().__init__()
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        self.time_dim = int(time_dim)
        self.base_channels = int(base_channels)
        self.self_attention_heads = int(self_attention_heads)
        self.use_condition_cross_attention = bool(use_condition_cross_attention)
        self.condition_attention_heads = int(condition_attention_heads)
        self.condition_tokens = int(condition_tokens)

        if self.base_channels % self.self_attention_heads != 0:
            raise ValueError(
                f"base_channels ({self.base_channels}) must be divisible by self_attention_heads ({self.self_attention_heads})."
            )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.cond_proj = nn.Linear(self.condition_dim, self.time_dim)

        # Multi head self attention
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.base_channels,
            num_heads=self.self_attention_heads,
            batch_first=True,
        )
        self.mhsa_norm = nn.LayerNorm(self.base_channels)

        if self.use_condition_cross_attention:
            if self.condition_tokens < 1:
                raise ValueError(f"condition_tokens must be >= 1, got {self.condition_tokens}")
            if self.base_channels % self.condition_attention_heads != 0:
                raise ValueError(
                    f"base_channels ({self.base_channels}) must be divisible by "
                    f"condition_attention_heads ({self.condition_attention_heads})."
                )
            self.cond_token_proj = nn.Linear(self.condition_dim, self.base_channels * self.condition_tokens)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.base_channels,
                num_heads=self.condition_attention_heads,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(self.base_channels)
        else:
            self.cond_token_proj = None
            self.cross_attn = None
            self.cross_attn_norm = None

        self.init_conv = nn.Conv1d(self.channels, self.base_channels, kernel_size=3, padding=1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.time_dim * 2, self.base_channels),
            nn.GELU(),
            nn.Linear(self.base_channels, self.base_channels),
        )

        # WaveNet-style Dilated Residual Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Dilation grows: 1, 2, 4, 8... allowing the network to see the whole day
            dilation = 2 ** i 
            self.blocks.append(ResidualBlock1D(self.base_channels, self.base_channels, dilation))

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.base_channels, self.base_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.base_channels // 2, self.channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, c):
        # 1. Embed Time and Conditions
        time_emb = self.time_mlp(t)
        cond_emb = self.cond_proj(c)
        
        # Combine [B, time_dim * 2] -> [B, time_dim]
        emb = torch.cat([time_emb, cond_emb], dim=-1)
        emb = self.shared_mlp(emb)

        # 2. Process Sequence
        h = self.init_conv(x)
        for block in self.blocks:
            h = block(h, emb)
        
        h = h.transpose(1, 2)  # [B, C, L] -> [B, L, C] for MHSA
        h_res = h
        h_norm = self.mhsa_norm(h)
        h, _ = self.mhsa(h_norm, h_norm, h_norm)
        h = h + h_res

        if self.use_condition_cross_attention:
            cond_tokens = self.cond_token_proj(c).view(c.shape[0], self.condition_tokens, self.base_channels)
            h_res = h
            h_norm = self.cross_attn_norm(h)
            h, _ = self.cross_attn(h_norm, cond_tokens, cond_tokens)
            h = h + h_res

        h = h.transpose(1, 2)  # Back to [B, C, L]
            
        # 3. Output Noise Prediction
        return self.out_conv(h)


class DiffusionModel(BaseModel):
    def __init__(
        self,
        noise_scheduler,
        model,
        conditioner,
        sequence_length=24,
        num_features=3,
        condition_dropout_prob=0.0,
        reconstruction_noise_fraction=0.3,
        deterministic_reconstruction=True,
        phased_mode=False,
    ):
        super().__init__(phased_mode=phased_mode, sequence_length=sequence_length)
        self.noise_scheduler = noise_scheduler
        self.model = model
        self.conditioner = conditioner
        self.sequence_length = int(sequence_length)
        self.num_features = int(num_features)
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.reconstruction_noise_fraction = float(reconstruction_noise_fraction)
        self.deterministic_reconstruction = bool(deterministic_reconstruction)
        self.model_kwargs = {}
        self.prior = None

        if self.conditioner is None or not hasattr(self.conditioner, "get_conditions"):
            raise ValueError("DiffusionModel requires a conditioner implementing get_conditions().")

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, (str, torch.device)):
                device = arg0
            elif torch.is_tensor(arg0):
                device = arg0.device
        if device is not None:
            self.noise_scheduler.to(device)
        return module

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _prepare_provided_conditions(self, provided_conditions, batch_size: int, device: torch.device):
        """Return valid precomputed conditions [B, C] or None.

        Phased loaders emit 4-tuples as (x, meter_id, timestamps, agg). The
        final element is aggregate stream data (often [B, 0, L] or [B, 1, L]),
        not conditioner features. We therefore accept provided conditions only
        when shape matches the model's expected condition width.
        """
        if provided_conditions is None:
            return None

        if torch.is_tensor(provided_conditions):
            c = provided_conditions.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            try:
                c = torch.as_tensor(provided_conditions, dtype=torch.float32, device=device)
            except Exception:
                return None

        if c.dim() == 1:
            c = c.unsqueeze(0)

        expected_dim = getattr(getattr(self.model, "cond_proj", None), "in_features", None)

        if c.dim() != 2:
            return None
        if int(c.shape[0]) != int(batch_size):
            return None
        if expected_dim is not None and int(c.shape[1]) != int(expected_dim):
            return None

        return c

    def _p_denoise(self, x, c, start_t, temperature=1.0, stochastic=True):
        for t_idx in reversed(range(start_t + 1)):
            t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            predicted_noise = self.model(x, t, c)

            alpha_t_bar = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1)
            
            # 1. Predict the clean signal (x_0) from the noise prediction
            x_0_pred = (x - torch.sqrt(1.0 - alpha_t_bar) * predicted_noise) / torch.sqrt(alpha_t_bar)
            
            # 2. clamp x_0 to a reasonable range to prevent numerical issues in mean calculation
            x_0_pred = torch.clamp(x_0_pred, min=-5.0, max=5.0)
            
            # 3. Calculate the stable mean using the clamped x_0
            coef1 = self.noise_scheduler.posterior_mean_coef1[t].view(-1, 1, 1)
            coef2 = self.noise_scheduler.posterior_mean_coef2[t].view(-1, 1, 1)
            
            mean = coef1 * x_0_pred + coef2 * x

            if t_idx > 0 and stochastic and temperature > 0.0:
                posterior_var = self.noise_scheduler.posterior_variance[t].view(-1, 1, 1)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * noise * float(temperature)
            else:
                x = mean
                
        return x

    def _sample_from_conditions(self, conditions, num_prior_samples=1, temperature=1.0):
        c = conditions
        if c.dim() == 1:
            c = c.unsqueeze(0)

        b_size = c.shape[0]
        c_rep = c.unsqueeze(0).expand(num_prior_samples, b_size, -1).reshape(num_prior_samples * b_size, -1)
        self._validate_condition_dim(c_rep)

        x = torch.randn(
            (num_prior_samples * b_size, self.num_features, self.sequence_length),
            device=c.device,
        )
        if temperature != 1.0:
            x = x * float(temperature)

        x = self._p_denoise(
            x,
            c_rep,
            start_t=self.noise_scheduler.num_timesteps - 1,
            temperature=temperature,
            stochastic=True,
        )
        return x.reshape(num_prior_samples, b_size, -1)

    def train_core(
        self,
        batch,
        optim,
        condition_dropout=False,
        gradient_clip=None,
        gradient_clip_kwargs=None,
        **_,
    ):
        self.train()

        x_raw, meter_id, timestamps, provided_conditions = self._unpack_batch(batch)
        x_phase_first = self._to_phase_first(x_raw).float()

        if self.training:
            perm = torch.randperm(x_phase_first.size(1), device=x_phase_first.device)
            x_phase_first = x_phase_first[:, perm, :]

        b_size = x_phase_first.size(0)
        c = self._prepare_provided_conditions(
            provided_conditions,
            batch_size=b_size,
            device=x_phase_first.device,
        )

        if c is None:
            c = self._build_conditions(
                x_phase_first,
                meter_id=meter_id,
                timestamps=timestamps,
                device=x_phase_first.device,
            )

        dropout_prob = self.condition_dropout_prob
        if bool(condition_dropout) and dropout_prob <= 0.0:
            dropout_prob = 0.1
        if dropout_prob > 0.0 and self.training and torch.rand(1).item() < dropout_prob:
            c = torch.zeros_like(c)

        t = torch.randint(0, self.noise_scheduler.num_timesteps, (b_size,), device=x_phase_first.device)
        noise = torch.randn_like(x_phase_first)
        x_noisy, noise = self.noise_scheduler.add_noise(x_phase_first, t, noise)

        predicted_noise = self.model(x_noisy, t, c)
        loss_dict = self.loss(predicted_noise, noise)

        optim.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()

        if gradient_clip:
            gradient_clip_kwargs = gradient_clip_kwargs or {}
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_kwargs.get("max_norm", 1.0))

        optim.step()
        return loss_dict

    @torch.no_grad()
    def validate(self, valloader, **_):
        self.eval()
        agg = {"loss": 0.0, "recon_loglikelihood": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        device = next(self.parameters()).device
        for batch in valloader:
            x_raw, meter_id, timestamps, provided_conditions = self._unpack_batch(batch)
            x_raw = self.move_to_device(x_raw, device=device).float()
            x_phase_first = self._to_phase_first(x_raw)

            b_size = x_phase_first.size(0)
            c = self._prepare_provided_conditions(
                provided_conditions,
                batch_size=b_size,
                device=device,
            )

            if c is None:
                c = self._build_conditions(
                    x_phase_first,
                    meter_id=meter_id,
                    timestamps=timestamps,
                    device=device,
                )

            t = torch.randint(0, self.noise_scheduler.num_timesteps, (b_size,), device=device)
            noise = torch.randn_like(x_phase_first)

            x_noisy, noise = self.noise_scheduler.add_noise(x_phase_first, t, noise)
            predicted_noise = self.model(x_noisy, t, c)
            loss_dict = self.loss(predicted_noise, noise)

            for k in agg:
                agg[k] += float(loss_dict[k].detach())
            num_batches += 1

        for k in agg:
            agg[k] /= max(num_batches, 1)
        return agg

    def loss(self, predicted_noise, target_noise):
        loss = F.mse_loss(predicted_noise, target_noise)
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        return {
            "loss": loss,
            "recon_loglikelihood": loss,
            "kl_divergence": zero,
        }

    @torch.no_grad()
    def sample(
        self,
        conditions=None,
        num_prior_samples=1,
        num_x_samples=1,
        device="cpu",
        temperature=1.0,
        num_samples=None,
        c=None,
        **_,
    ):
        legacy_mode = num_samples is not None
        if legacy_mode:
            num_prior_samples = int(num_samples)
            num_x_samples = 1

        if conditions is None:
            conditions = c
        if conditions is None:
            raise ValueError("Diffusion sampling requires 'conditions' or 'c'.")

        self.eval()
        original_device = next(self.parameters()).device
        self.to(device)

        c_in = self.move_to_device(conditions, device=device).float()
        if c_in.dim() == 1:
            c_in = c_in.unsqueeze(0)
        self._validate_condition_dim(c_in)

        samples_per_x = []
        for _idx in range(max(1, int(num_x_samples))):
            samples_per_x.append(
                self._sample_from_conditions(
                    conditions=c_in,
                    num_prior_samples=max(1, int(num_prior_samples)),
                    temperature=float(temperature),
                )
            )

        samples = torch.stack(samples_per_x, dim=1)
        self.to(original_device)
        if legacy_mode:
            return samples[:, 0]
        return samples

    @torch.no_grad()
    def reconstruct(
        self,
        x,
        meter_id=None,
        timestamps=None,
        num_reconstructions=1,
        reconstruction_timestep=None,
        reconstruction_noise_fraction=None,
        deterministic=None,
        temperature=1.0,
        device=None,
        c=None,
        **_,
    ):
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        original_device = next(self.parameters()).device
        self.to(device)

        x_in = self.move_to_device(x, device=device).float()
        x_phase_first = self._to_phase_first(x_in)

        if c is None:
            c = self._build_conditions(
                x_phase_first,
                meter_id=meter_id,
                timestamps=timestamps,
                device=device,
            )
        else:
            c = self.move_to_device(c, device=device).float()
            if c.dim() == 1:
                c = c.unsqueeze(0)
            self._validate_condition_dim(c)

        if reconstruction_timestep is None:
            frac = self.reconstruction_noise_fraction
            if reconstruction_noise_fraction is not None:
                frac = float(reconstruction_noise_fraction)
            frac = min(max(frac, 0.0), 1.0)
            reconstruction_timestep = int(round(frac * (self.noise_scheduler.num_timesteps - 1)))

        reconstruction_timestep = max(1, min(int(reconstruction_timestep), self.noise_scheduler.num_timesteps - 1))
        deterministic = self.deterministic_reconstruction if deterministic is None else bool(deterministic)

        t = torch.full((x_phase_first.size(0),), reconstruction_timestep, device=device, dtype=torch.long)
        reconstructions = []
        for _idx in range(max(1, int(num_reconstructions))):
            q_noise = torch.randn_like(x_phase_first) * float(temperature)
            x_t, _ = self.noise_scheduler.add_noise(x_phase_first, t, noise=q_noise)
            x_recon = self._p_denoise(
                x_t,
                c,
                start_t=reconstruction_timestep,
                temperature=float(temperature),
                stochastic=not deterministic,
            )
            reconstructions.append(x_recon.reshape(x_recon.shape[0], -1))

        out = torch.stack(reconstructions, dim=0)
        self.to(original_device)
        return out

