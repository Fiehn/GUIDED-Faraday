import os
import sys
import json
import torch
import torch.nn as nn
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Data.Conditioner import Conditioner
from Visualization.Writer import Writer
class GAN(nn.Module):
    """
    A WGAN-GP implementation for time series data.
    The generator takes noise and optional conditions to produce synthetic time series data.
    The discriminator evaluates the authenticity of the generated data, optionally conditioned on the same information.
    
    Based on Gulrajani et al., "Improved Training of Wasserstein GANs" (https://arxiv.org/abs/1704.00028)"""
    def __init__(self,
                 input_dim: int = 24,
                 *,
                 noise_dim: int = 32,
                 conditioner: Conditioner = None,
                 generator_hidden: list = None,
                 discriminator_hidden: list = None,
                 n_critic: int = 5,
                 gp_lambda: float = 10.0,
                 dropout: float = 0.1,
                 use_spectral_norm: bool = True,
                 device: str = "cpu",
                 **_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.device = torch.device(device)

        # Conditioner
        self.conditioner = conditioner
        self.condition_dim = conditioner.get_condition_dim() if conditioner else 0

        # Build Networks.
        self.generator = Generator(
            noise_dim=noise_dim,
            condition_dim=self.condition_dim,
            hidden_dims=generator_hidden,
            output_dim=input_dim,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            input_dim=input_dim,
            condition_dim=self.condition_dim,
            hidden_dims=discriminator_hidden,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
        )

        self.num_parameters = (self.generator._num_parameters() + self.discriminator._num_parameters())

        self.model_kwargs = {
            "input_dim": input_dim,
            "noise_dim": noise_dim,
            "conditioner": conditioner,
            "generator_hidden": generator_hidden,
            "discriminator_hidden": discriminator_hidden,
            "n_critic": n_critic,
            "gp_lambda": gp_lambda,
            "dropout": dropout,
            "use_spectral_norm": use_spectral_norm,
        }

    def _prep_inpiuts(self, x: torch.Tensor) -> torch.Tensor:
        """ Flatten [B, L, F] -> [B, D]"""
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        return x
    
    def move_to_device(self,batch, device = None):
        device = device or self.device
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(b.to(device) if torch.is_tensor(b) else b for b in batch)
        if isinstance(batch, dict):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        return batch
    
    def _sample_noise(self, bathc_size: int, device = None) -> torch.Tensor:
        """ Sample noise from a standard normal distribution."""
        device = device or self.device
        return torch.randn(bathc_size, self.noise_dim, device=device)
    
    def _get_conditions(self, x, meter_id, timestamps):
        if self.conditioner is None:
            return None
        return self.move_to_device(self.conditioner.get_conditions(meter_id, timestamps))
    
    def _gradient_penalty(self, x_real: torch.Tensor, x_fake: torch.Tensor,
                          conditions: torch.Tensor = None) -> torch.Tensor:
        """ Compute the gradient penalty for WGAN-GP."""
        B = x_real.size(0)
        alpha = torch.rand(B, 1, device=x_real.device)
        x_interp = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
        d_interp = self.discriminator(x_interp, conditions)

        grad_outputs = torch.ones_like(d_interp, device=x_real.device)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=x_interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp
    
    def train_core(self, batch, optim_g, optim_d, step: int = 0, **_):
        
        self.train()

        # uncpack batches
        if isinstance(batch, (list, tuple)):
            x_real, meter_id, timestamps = batch
        elif isinstance(batch, dict):
            x_real, meter_id, timestamps = batch['x'], batch['meter_id'], batch['timestamp']
        else:
            x_real = batch
            meter_id, timestamps = None, None

        x_3d = x_real
        x = self._prep_inpiuts(x_3d).to(self.device)
        B = x.size(0)
        conditions = self._get_conditions(x_3d, meter_id, timestamps)

        # Train Discriminator
        for _ in range(self.n_critic):
            z = self._sample_noise(B)
            with torch.no_grad():
                x_fake = self.generator(z, conditions)
            
            d_real = self.discriminator(x, conditions)
            d_fake = self.discriminator(x_fake, conditions)

            # Wasserstein loss
            loss_d = d_fake.mean() - d_real.mean()

            # Gradient penalty
            gp = self._gradient_penalty(x, x_fake, conditions)
            loss_d_total = loss_d + self.gp_lambda * gp

            optim_d.zero_grad(set_to_none=True)
            loss_d_total.backward()
            optim_d.step()
        
        # Train Generator
        z = self._sample_noise(B)
        x_fake = self.generator(z, conditions)
        d_fake = self.discriminator(x_fake, conditions)
        
        loss_g = -d_fake.mean()

        optim_g.zero_grad(set_to_none=True)
        loss_g.backward()
        optim_g.step()

        wasserstein_d = d_real.mean().item() - d_fake.mean().item()

        return {
            "loss": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "wasserstein_d": wasserstein_d,
            "gp": gp.item(),
        }
    
    @torch.no_grad()
    def validate(self, valloader, num_samples: int = 1, beta: float = 1.0, device: str = "cpu"):
        """ Wasserstein distance between real and generated distributions on the validation set."""
        self.eval()
        total_wd = 0.0
        total_loss_d = 0.0
        num_batches = 0
        
        for batch in valloader:
            if isinstance(batch, (list, tuple)):
                x_real, meter_id, timestamps = batch
            elif isinstance(batch, dict):
                x_real, meter_id, timestamps = batch['x'], batch['meter_id'], batch['timestamp']
            else:
                x_real = batch
                meter_id, timestamps = None, None
            
            x_3d = x_real
            x = self._prep_inpiuts(x_3d).to(device)
            B = x.size(0)
            conditions = self._get_conditions(x_3d, meter_id, timestamps)

            z = self._sample_noise(B, device=device)
            x_fake = self.generator(z, conditions)

            d_real = self.discriminator(x, conditions)
            d_fake = self.discriminator(x_fake, conditions)

            wd = d_real.mean().item() - d_fake.mean().item()
            total_wd += wd
            total_loss_d += (d_fake.mean() - d_real.mean()).item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss_d / n,
            "recon_loglikelihood": 0.0, # not applicable for GANs
            "kl_divergence": 0.0, # not applicable for GANs
            "wasserstein_distance": total_wd / n,
        }
    
    @torch.no_grad()
    def sample(self, conditions: torch.Tensor = None,
               num_prior_samples: int = 1, num_x_samples: int = 1,
               batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Generate synthetic load profiles.
        
        Args:
            conditions: [B, C] pre-computed condition tensor (same as CVAE.sample).
                        If None, generates unconditionally.
            num_prior_samples: Number of independent noise draws per condition.
            num_x_samples: Kept for API compatibility (GAN is deterministic given z).
            batch_size: Used only when conditions is None.
            device: Target device.
            
        Returns:
            samples: [S, 1, B, D] to match CVAE's [S, num_x_samples, B, D] shape.
        """
        self.eval()

        if conditions is not None:
            c = conditions.to(device)
            B = c.size(0)
        else:
            B = batch_size
            c = None
        
        all_samples = []
        for _ in range(num_prior_samples):
            z = self._sample_noise(B, device=device)
            x_fake = self.generator(z, c)
            all_samples.append(x_fake)  # [1, B, D]
        
        samples = torch.stack(all_samples, dim=0)
        return samples.unsqueeze(1) # [S, 1, B, D]
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, meter_ids: torch.Tensor = None, timestamps: torch.Tensor = None,
                    num_samples: int = 1):
        """ GANs don't reconstruct approximate by finding closest generation.
        For evaluation compatibility, we encode via the discriminator features
        and regenerate. This is a rough proxy. Mostly for full comparisons to work
        """
        self.eval()
        x_flat = self._prep_inputs(x)
        B = x_flat.size(0)

        c = None
        if self.conditioner is not None and meter_ids is not None:
            c = self._get_conditions(x, meter_ids, timestamps)

        # Best-of-K approximation: generate K samples, pick closest per instance
        K = max(num_samples * 10, 50)
        best_recon = None
        best_dist = None

        for _ in range(K // B + 1):
            z = self._sample_noise(B, device=x_flat.device)
            x_fake = self.generator(z, c)
            dist = ((x_fake - x_flat) ** 2).sum(dim=-1)  # [B]

            if best_recon is None:
                best_recon = x_fake
                best_dist = dist
            else:
                improved = dist < best_dist
                best_recon[improved] = x_fake[improved]
                best_dist[improved] = dist[improved]

        recon_params = {"mu": best_recon.unsqueeze(0)}  # [1, B, D]
        return recon_params, {}
    
    def get_optimizers(self, lr_g: float = 1e-4, lr_d: float = 1e-4,
                       weight_decay: float = 1e-4, betas: tuple = (0.0, 0.9)):
        """ Return separate optimizers for generator and discriminator.
        WGAN-GP typically use ADAM"""
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas, weight_decay=weight_decay)
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas, weight_decay=weight_decay)    
        return optim_g, optim_d
    
    def fit(
            self,
            trainloader,
            valloader = None,
            lr=1e-4,
            lr_g: float = None,
            lr_d: float = None,
            weight_decay: float = 1e-4,
            gradient_clip=None,
            gradient_clip_kwargs=None,
            lr_scheduler=False,
            lr_scheduler_kwargs=None,
            num_epochs: int = 100,
            validation_freq=1,
            device: str = "cpu",
            early_stopping = True,
            early_stopping_kwargs = None,
            save_epoch_freq = 10,
            writer=None,
            log_dir='./models/',
            n_critic=None,
            **_,
    ):
        from src.Models.model_utils import EarlyStopping

        if n_critic is not None:
            self.n_critic = n_critic
        if lr_g is None:
            lr_g = lr
        if lr_d is None:
            lr_d = lr
        if early_stopping_kwargs is None:
            early_stopping_kwargs = {"patience": 10, "min_delta": 0.001}
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {"step_size": 30, "gamma": 0.5}
        
        # Model tag
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = f"GAN_{time_stamp}"
        if not model_tag.endswith(time_stamp):
            model_tag += f"_{time_stamp}"
        log_dir_tagged = os.path.join(log_dir, model_tag)
        os.makedirs(log_dir_tagged, exist_ok=True)

        # Writer
        if writer is None:
            writer = Writer(log_dir=log_dir_tagged)
            print(f"Writer logging to {log_dir_tagged}")

        # Save config
        config = {
            'model_tag': model_tag,
            'hyperparameters': {
                'lr_g': lr_g, 'lr_d': lr_d,
                'weight_decay': weight_decay,
                'n_critic': self.n_critic,
                'gp_lambda': self.gp_lambda,
                'epochs': num_epochs,
                'noise_dim': self.noise_dim,
                'condition_dim': self.condition_dim,
            },
            'model_architecture': {
                'class_name': 'GAN (WGAN-GP)',
                'input_dim': self.input_dim,
                'num_parameters': self.num_parameters,
                'generator_params': self.generator._num_parameters(),
                'discriminator_params': self.discriminator._num_parameters(),
            },
        }
        with open(os.path.join(log_dir_tagged, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Optimizers and schedulers
        optim_g, optim_d = self.get_optimizers(lr_g=lr_g,
                                                  lr_d=lr_d,
                                                  weight_decay=weight_decay)
        if lr_scheduler:
            scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, **lr_scheduler_kwargs)
            scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, **lr_scheduler_kwargs)

        self.to(device)
        self.device = torch.device(device)

        if early_stopping:
            earlystoping = EarlyStopping(**early_stopping_kwargs)

        # Training loop
        global_step = 0
        best_val_wd = float('-inf')
        best_epoch = 0

        for epoch in range(num_epochs):
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            epoch_gp = 0.0
            epoch_wd = 0.0
            num_batches = 0

            self.train()
            for batch in trainloader:
                global_step += 1
                batch = self.move_to_device(batch, device=self.device)

                loss_dict = self.train_core(batch, optim_g, optim_d, step=global_step)
                epoch_loss_g += loss_dict["loss_g"]
                epoch_loss_d += loss_dict["loss_d"]
                epoch_gp += loss_dict["gp"]
                epoch_wd += loss_dict["wasserstein_d"]
                num_batches += 1

                writer.add_scalar("train_loss_g", loss_dict["loss_g"], global_step)
                writer.add_scalar("train_loss_d", loss_dict["loss_d"], global_step)
                writer.add_scalar("train_wasserstein_distance", loss_dict["wasserstein_d"], global_step)
                writer.add_scalar("train_gradient_penalty", loss_dict["gp"], global_step)
            # Log training metrics
            n = max(num_batches, 1)
            avg_loss_g = epoch_loss_g / n
            avg_loss_d = epoch_loss_d / n
            avg_gp = epoch_gp / n
            avg_wd = epoch_wd / n

            writer.add_scalar("train_loss_g_epoch", avg_loss_g, epoch)
            writer.add_scalar("train_loss_d_epoch", avg_loss_d, epoch)
            writer.add_scalar("train_wasserstein_distance_epoch", avg_wd, epoch)
            writer.add_scalar("train_gradient_penalty_epoch", avg_gp, epoch)

            # Validation
            val_dict = None
            if valloader is not None and (epoch + 1) % validation_freq == 0:
                val_dict = self.validate(valloader, device=device)
                writer.add_scalar("val_loss", val_dict["loss"], epoch)
                writer.add_scalar("val_wasserstein_distance", val_dict["wasserstein_distance"], epoch)

                # Track best by Wasserstein distance (higher = better separation)
                if val_dict["wasserstein_distance"] > best_val_wd:
                    best_val_wd = val_dict["wasserstein_distance"]
                    best_epoch = epoch
                    self.save_model(save_path=log_dir_tagged, model_name=f"{model_tag}_best", device=device)

                print(f"Epoch [{epoch+1:4d}/{num_epochs}] "
                      f"G: {avg_loss_g:.4f} | D: {avg_loss_d:.4f} | "
                      f"WD: {avg_wd:.4f} | GP: {avg_gp:.4f} | "
                      f"Val WD: {val_dict['wasserstein_distance']:.4f}")

                if early_stopping:
                    # Use negative WD as "loss" (lower = better for early stopping)
                    earlystoping(val_metric=-val_dict["wasserstein_distance"],
                                  train_metric=avg_loss_g, epoch=epoch)
                    if earlystoping.early_stop:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch [{epoch+1:4d}/{num_epochs}] "
                      f"G: {avg_loss_g:.4f} | D: {avg_loss_d:.4f} | "
                      f"WD: {avg_wd:.4f} | GP: {avg_gp:.4f}")
            
            if lr_scheduler:
                scheduler_g.step()
                scheduler_d.step()
            
            if save_epoch_freq and (epoch + 1) % save_epoch_freq == 0:
                self.save_model(save_path=log_dir_tagged, model_name=f"{model_tag}_epoch{epoch+1}", device=device)
            
        print(f"\nTraining complete. Best validation WD: {best_val_wd:.4f} at epoch {best_epoch+1}")
        self.save_model(save_path=log_dir_tagged, model_name=f"{model_tag}_final", device=device)

        summary = {
            "model_tag": model_tag,
            "best_epoch": best_epoch,
            "best_val_wd": best_val_wd,
            "final_train_loss_g": avg_loss_g,
            "final_train_loss_d": avg_loss_d,
            "final_train_wd": avg_wd,
        }
        with open(os.path.join(log_dir_tagged, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        if writer is not None:
            writer.close()

        return model_tag, writer

    def save_model(self, save_path="./models/", model_name="gan_model", device="cpu"):
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f"{model_name}.pt")

        self.eval()
        original_device = self.device
        self.to("cpu")

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_kwargs': self.model_kwargs,
            'model_info': {
                'model_class': 'GAN',
                'input_dim': self.input_dim,
                'noise_dim': self.noise_dim,
                'condition_dim': self.condition_dim,
                'num_parameters': self.num_parameters,
            },
        }

        # Serialize conditioner
        if self.conditioner is not None:
            conditioner_state = {
                'condition_items': {},
                'condition_tags': self.conditioner.get_condition_tags(),
                'condition_dim': self.conditioner.get_condition_dim(),
            }
            for tag, item in self.conditioner.condition_items.items():
                conditioner_state['condition_items'][tag] = {
                    'class_name': item.__class__.__name__,
                    'module': item.__class__.__module__,
                    'state_dict': item.state_dict() if hasattr(item, 'state_dict') else {},
                }
            checkpoint['conditioner_state'] = conditioner_state

        torch.save(checkpoint, model_path)
        self.to(original_device)

    def load_model(self, load_path, device="cpu"):
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        self.device = torch.device(device)

class Generator(nn.Module):
    """
    Maps [z, c] -> x_fake.
    Architecture mirrors the VAE decoder: MLP with residual connections.
    """
    def __init__(self, noise_dim: int, condition_dim: int, output_dim: int,
                 hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        layers = []
        input_dim = noise_dim + condition_dim
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # No final activation — output is in normalised space (matching VAE decoder mu)
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, conditions: torch.Tensor = None) -> torch.Tensor:
        """ Args:
            z: [B, noise_dim]
            conditions: [B, condition_dim] or None
            """
        inp = torch.cat([z, conditions], dim=-1) if conditions is not None else z
        return self.net(inp)
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class Discriminator(nn.Module):
    """
    Maps [x, c] -> real/fake score.
    """
    def __init__(self, input_dim: int, condition_dim: int, 
                 hidden_dims: list = None, dropout: float = 0.1,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        total_input = input_dim + condition_dim
        def _maybe_sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        layers = []
        prev_dim = total_input
        for h_dim in hidden_dims:
            layers.extend([
                _maybe_sn(nn.Linear(prev_dim, h_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(_maybe_sn(nn.Linear(prev_dim, 1)))  # Output a single score
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor = None) -> torch.Tensor:
        """ Args:
            x: [B, input_dim]
            conditions: [B, condition_dim] or None
            """
        inp = torch.cat([x, conditions], dim=-1) if conditions is not None else x
        return self.net(inp)
    
    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)