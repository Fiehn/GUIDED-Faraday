import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
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
        self._init_tensors(torch.device(device))

    def _build_betas(self, device: torch.device) -> torch.Tensor:
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=device)

        steps = self.num_timesteps
        t = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float32)
        alpha_bar = torch.cos(((t / steps) + self.cosine_s) / (1.0 + self.cosine_s) * (math.pi * 0.5)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0].clamp(min=1e-12)
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1].clamp(min=1e-12))
        return betas.clamp(min=1e-8, max=self.max_beta)

    def _init_tensors(self, device: torch.device) -> None:
        self.betas = self._build_betas(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]],
            dim=0,
        )

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_recip_alphas",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def state_dict(self):
        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "schedule_type": self.schedule_type,
            "cosine_s": self.cosine_s,
            "max_beta": self.max_beta,
            "betas": self.betas.detach().cpu(),
            "alphas": self.alphas.detach().cpu(),
            "alphas_cumprod": self.alphas_cumprod.detach().cpu(),
            "sqrt_recip_alphas": self.sqrt_recip_alphas.detach().cpu(),
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod.detach().cpu(),
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod.detach().cpu(),
            "posterior_variance": self.posterior_variance.detach().cpu(),
        }

    def load_state_dict(self, state_dict):
        self.num_timesteps = int(state_dict["num_timesteps"])
        self.beta_start = float(state_dict["beta_start"])
        self.beta_end = float(state_dict["beta_end"])
        self.schedule_type = str(state_dict.get("schedule_type", "linear")).lower()
        self.cosine_s = float(state_dict.get("cosine_s", 0.008))
        self.max_beta = float(state_dict.get("max_beta", 0.999))

        if self.schedule_type not in {"linear", "cosine"}:
            self.schedule_type = "linear"

        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_recip_alphas",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            if name in state_dict:
                setattr(self, name, state_dict[name])

        # Fallback for partial/legacy scheduler states.
        if not all(hasattr(self, name) for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_recip_alphas",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        )):
            self._init_tensors(torch.device("cpu"))

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


class DiffusionModel(nn.Module):
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
    ):
        super().__init__()
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
        self._condition_dim_checked = False

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

    def get_tag(self):
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.__class__.__name__}_{self.model.__class__.__name__}_{time_stamp}"

    def _validate_condition_dim(self, c: torch.Tensor) -> None:
        if self._condition_dim_checked:
            return
        expected_dim = getattr(getattr(self.model, "cond_proj", None), "in_features", None)
        if expected_dim is not None and c.shape[-1] != expected_dim:
            raise ValueError(f"Condition dim mismatch: model expects {expected_dim}, got {c.shape[-1]}")
        self._condition_dim_checked = True

    def _build_conditions(self, x_phase_first, meter_id=None, timestamps=None, device=None):
        cond_input = self._build_condition_input(x_phase_first)
        if meter_id is None and timestamps is None:
            c = self.conditioner.get_conditions(cond_input)
        else:
            c = self.conditioner.get_conditions(cond_input, meter_id=meter_id, timestamps=timestamps)

        c = self.move_to_device(c, device=device)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        c = c.float()
        self._validate_condition_dim(c)
        return c

    def _p_denoise(self, x, c, start_t, temperature=1.0, stochastic=True):
        for t_idx in reversed(range(start_t + 1)):
            t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            predicted_noise = self.model(x, t, c)

            alpha_t = self.noise_scheduler.alphas[t].view(-1, 1, 1)
            alpha_t_bar = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1)
            beta_t = self.noise_scheduler.betas[t].view(-1, 1, 1)

            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_t_bar)) * predicted_noise) / torch.sqrt(alpha_t)

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

        if provided_conditions is not None:
            c = self.move_to_device(provided_conditions, device=x_phase_first.device).float()
            if c.dim() == 1:
                c = c.unsqueeze(0)
            self._validate_condition_dim(c)
        else:
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

        b_size = x_phase_first.size(0)
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

            if provided_conditions is not None:
                c = self.move_to_device(provided_conditions, device=device).float()
                if c.dim() == 1:
                    c = c.unsqueeze(0)
                self._validate_condition_dim(c)
            else:
                c = self._build_conditions(
                    x_phase_first,
                    meter_id=meter_id,
                    timestamps=timestamps,
                    device=device,
                )

            b_size = x_phase_first.size(0)
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

    def save_model(self, save_path=None, model_name="trained_model", device="cpu", normalizer=None):
        if save_path is None:
            save_path = "./models/"

        model_path = os.path.join(save_path, f"{model_name}.pt")
        os.makedirs(save_path, exist_ok=True)

        original_device = next(self.parameters()).device
        self.eval()
        self.to("cpu")

        normalizer_state = None
        if normalizer is not None and hasattr(normalizer, "fitted") and normalizer.fitted:
            normalizer_state = {
                "method": normalizer.method,
                "epsilon": normalizer.epsilon,
                "zero_id": normalizer.zero_id,
                "shift": normalizer.shift,
                "non_zero_mean": normalizer.non_zero_mean,
                "non_zero_std": normalizer.non_zero_std,
                "mean": normalizer.mean,
                "std": normalizer.std,
            }

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "constructor_args": {
                "sequence_length": self.sequence_length,
                "num_features": self.num_features,
                "condition_dropout_prob": self.condition_dropout_prob,
                "reconstruction_noise_fraction": self.reconstruction_noise_fraction,
                "deterministic_reconstruction": self.deterministic_reconstruction,
                "condition_dim": getattr(self.model.cond_proj, "in_features", None),
                "time_dim": int(getattr(self.model, "time_dim", 128)),
                "base_channels": int(getattr(self.model, "base_channels", 64)),
                "self_attention_heads": int(getattr(self.model, "self_attention_heads", 4)),
                "use_condition_cross_attention": bool(getattr(self.model, "use_condition_cross_attention", False)),
                "condition_attention_heads": int(getattr(self.model, "condition_attention_heads", 4)),
                "condition_tokens": int(getattr(self.model, "condition_tokens", 4)),
            },
            "noise_scheduler_state": self.noise_scheduler.state_dict(),
            "model_kwargs": self.model_kwargs,
            "normalizer_state": normalizer_state,
            "model_info": {
                "model_class": self.__class__.__name__,
                "save_format_version": 1,
            },
        }

        temp_path = f"{model_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, model_path)

        self.to(device)
        self.to(original_device)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model with {num_params} parameters saved to {model_path}")

    def load_model(self, load_path, device="cpu"):
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "noise_scheduler_state" in checkpoint:
            self.noise_scheduler.load_state_dict(checkpoint["noise_scheduler_state"])
            self.noise_scheduler.to(device)
        self.to(device)
        return self._load_normalizer_from_checkpoint(checkpoint)

    @staticmethod
    def _load_normalizer_from_checkpoint(checkpoint):
        normalizer_state = checkpoint.get("normalizer_state", None)
        if normalizer_state is None:
            return None
        try:
            from data.normalizer import Normalizer

            normalizer = Normalizer(method=normalizer_state["method"])
            if "epsilon" in normalizer_state:
                normalizer.epsilon = normalizer_state["epsilon"]
            normalizer.zero_id = normalizer_state["zero_id"]
            normalizer.shift = normalizer_state["shift"]
            normalizer.non_zero_mean = normalizer_state["non_zero_mean"]
            normalizer.non_zero_std = normalizer_state["non_zero_std"]
            normalizer.mean = normalizer_state["mean"]
            normalizer.std = normalizer_state["std"]
            normalizer.fitted = True
            return normalizer
        except Exception:
            return None

    def _to_phase_first(self, x: torch.Tensor) -> torch.Tensor:
        """Return x as [B, F, L], supporting [B, D], [B, F, L], or [B, L, F]."""
        if x.dim() == 2:
            if x.shape[1] != self.sequence_length * self.num_features:
                raise ValueError(
                    f"Expected flattened input dim {self.sequence_length * self.num_features}, got {x.shape[1]}"
                )
            return x.reshape(x.shape[0], self.num_features, self.sequence_length)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D/3D input, got shape {tuple(x.shape)}")

        if x.shape[1] == self.num_features and x.shape[2] == self.sequence_length:
            return x

        if x.shape[1] == self.sequence_length and x.shape[2] == self.num_features:
            return x.transpose(1, 2)

        raise ValueError(
            f"Could not infer phase/time axes for input shape {tuple(x.shape)} with "
            f"num_features={self.num_features}, sequence_length={self.sequence_length}"
        )

    def _build_condition_input(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """Convert [B, F, L] -> [B, L, F] for conditioner compatibility."""
        return x_phase_first.transpose(1, 2)

    def _unpack_batch(self, batch):
        """Support (x, meter_id, timestamps), (x, cond), (x, meter_id), and (x, c, meter_id, timestamps)."""
        provided_conditions = None
        expected_cond_dim = getattr(getattr(self.model, "cond_proj", None), "in_features", None)

        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            x, second, third, fourth = batch
            # Regular CSV explicit tuple: (x, c, meter_id, timestamps)
            if (
                torch.is_tensor(second)
                and second.dim() >= 2
                and expected_cond_dim is not None
                and int(expected_cond_dim) > 0
                and second.shape[-1] == int(expected_cond_dim)
            ):
                return x, third, fourth, second
            # Extended eval tuple: (x, meter_id, timestamps, agg)
            return x, second, third, None

        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, second, third = batch
            if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
                meter_id = second[:, 0, 0].long()
                timestamps = second[:, 1, 0].long()
                return x, meter_id, timestamps, None
            return x, second, third, None

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "DiffusionModel expects batches as (x, meter_id, timestamps), (x, meter_id), (x, cond), or (x, c, meter_id, timestamps)."
            )

        x, second = batch
        meter_id = second
        timestamps = None

        if torch.is_tensor(second) and second.dim() == 3 and second.shape[1] >= 2:
            meter_id = second[:, 0, 0].long()
            timestamps = second[:, 1, 0].long()
        elif torch.is_tensor(second) and second.dim() == 2:
            if second.shape[1] >= 2 and not torch.is_floating_point(second):
                meter_id = second[:, 0].long()
                timestamps = second[:, 1].long()
            elif expected_cond_dim is not None and second.shape[1] == expected_cond_dim:
                provided_conditions = second

        return x, meter_id, timestamps, provided_conditions

    def move_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                b.to(device, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch
            )
        if isinstance(batch, dict):
            return {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        return batch