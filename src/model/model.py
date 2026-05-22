import math
import os
import sys
import pickle
from abc import ABC, abstractmethod
import time
from datetime import datetime

import torch
from torch import nn

def _init_vamp_prior_from_batch(model, batch, device, noise_scale=0.01):
    if batch is None:
        print("No batch available to initialize VampPrior. Skipping initialization.")
        return
    from src.model.priors import VampPrior

    x, meter_id, timestamps, agg = model._unpack_batch(batch)
    x = model.move_to_device(x, device)
    x_phase_first = model._to_phase_first(x)  # SHAPER: [batch_size, num_features, sequence_length]
    c = model._build_conditions(x_phase_first, meter_id=meter_id, timestamps=timestamps, device=device)  # SHAPER: [batch_size, condition_dim]

    with torch.no_grad():
        z = model.encoder(x_phase_first, c)  # SHAPER: [batch_size, latent_dim]
        if isinstance(model.prior, VampPrior):
            model.prior.initialize_pseudo_inputs(z, noise_scale=noise_scale)

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, phased_mode: bool = False, sequence_length: int = None, **kwargs):
        super(BaseModel, self).__init__()
        self.tag = self.get_tag()
        self.phased_mode = phased_mode
        self.num_features = 3 if phased_mode else 1
        self.sequence_length = sequence_length
        self._condition_dim_checked = False

    @abstractmethod
    def train_core(self, *args, **kwargs):
        pass

    @abstractmethod
    @torch.no_grad()
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def reconstruct(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_optimizer(self, lr, weight_decay, *args, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Unified save / load
    # ------------------------------------------------------------------

    def save_model(self, save_path=None, model_name="trained_model", device="cpu", normalizer=None):
        """Save model checkpoint with atomic write.

        Checkpoint format (v2)::

            {
                "format_version": 2,
                "model_class":    "<ClassName>",
                "model_state_dict": self.state_dict(),
                "normalizer_state": {...} | None,
                "conditioner_state": <pickle bytes> | None,
            }

        The conditioner (if present) is pickled because it is a plain
        Python object (not an nn.Module).
        state_dict() via registered buffers / submodules (nn.Module).
        """
        if save_path is None:
            save_path = "./models/"

        model_path = os.path.join(save_path, f"{model_name}.pt")
        os.makedirs(save_path, exist_ok=True)

        # Remember the original device so we can move back after saving.
        # Some baseline models are parameterless and keep all state in buffers.
        first_param = next(self.parameters(), None)
        if first_param is not None:
            original_device = first_param.device
        else:
            first_buffer = next(self.buffers(), None)
            original_device = first_buffer.device if first_buffer is not None else torch.device("cpu")
            
        self.eval()
        self.to("cpu")

        # --- normalizer ---
        normalizer_to_save = normalizer
        if normalizer_to_save is None:
            normalizer_to_save = getattr(self, "normalizer", None)
            
        normalizer_state = None
        if normalizer_to_save is not None and getattr(normalizer_to_save, "fitted", False):
            normalizer_state = {
                "method": normalizer_to_save.method,
                "epsilon": normalizer_to_save.epsilon,
                "zero_id": normalizer_to_save.zero_id,
                "shift": normalizer_to_save.shift,
                "non_zero_mean": normalizer_to_save.non_zero_mean,
                "non_zero_std": normalizer_to_save.non_zero_std,
                "mean": normalizer_to_save.mean.cpu() if isinstance(normalizer_to_save.mean, torch.Tensor) else normalizer_to_save.mean,
                "std": normalizer_to_save.std.cpu() if isinstance(normalizer_to_save.std, torch.Tensor) else normalizer_to_save.std,
            }

        # --- conditioner (pickle) ---
        conditioner_state = None
        conditioner = getattr(self, "conditioner", None)
        if conditioner is not None:
            if hasattr(conditioner, "to") and callable(getattr(conditioner, "to")):
                conditioner.to("cpu")
            conditioner_state = pickle.dumps(conditioner)

        # --- build checkpoint ---
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        checkpoint = {
            "format_version": 2,
            "model_class": self.__class__.__name__,
            "model_state_dict": self.state_dict(),
            "normalizer_state": normalizer_state,
            "conditioner_state": conditioner_state,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }

        # --- atomic write ---
        temp_path = model_path + ".tmp"
        try:
            torch.save(checkpoint, temp_path)

            # Quick integrity check
            test_load = torch.load(temp_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in test_load, "Checkpoint missing model_state_dict"

            if os.path.exists(model_path):
                backup_path = model_path + ".backup"
                os.replace(model_path, backup_path)
            os.replace(temp_path, model_path)

            num_params = sum(p.numel() for p in self.parameters())
            print(f"Model saved: {model_path}  ({num_params:,} params)")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Save failed: {e}", file=sys.stderr)
            raise
        finally:
            self.to(original_device)

    def load_model(self, load_path, device="cpu"):
        """Load a checkpoint saved by save_model().

        Returns:
            tuple: ``(normalizer_or_None, conditioner_or_None)``
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        # --- model weights ---
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)

        # --- normalizer ---
        normalizer = None
        ns = checkpoint.get("normalizer_state")
        if ns is not None:
            normalizer = self._restore_normalizer(ns, device=device)
            self.normalizer = normalizer

        # --- conditioner ---
        conditioner = None
        cs = checkpoint.get("conditioner_state")
        if cs is not None:
            # Custom Unpickler to correctly map GPU tensors in pickle to the target device
            import io
            class DeviceUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name in ('_load_from_bytes', '_legacy_load'):
                        return lambda b: torch.load(io.BytesIO(b) if isinstance(b, bytes) else b, map_location=device, weights_only=False)
                    return super().find_class(module, name)

            try:
                conditioner = DeviceUnpickler(io.BytesIO(cs)).load()
            except Exception:
                conditioner = pickle.loads(cs)
            
            self.conditioner = conditioner

        return normalizer, conditioner

    @staticmethod
    def _restore_normalizer(state: dict, device="cpu"):
        """Reconstruct a Normalizer from its serialized state dict."""
        from data.normalizer import Normalizer
        norm = Normalizer(method=state["method"])
        norm.epsilon = state.get("epsilon", 1e-8)
        norm.zero_id = state.get("zero_id", -3)
        norm.shift = state.get("shift", 1.0)
        norm.non_zero_mean = state.get("non_zero_mean")
        norm.non_zero_std = state.get("non_zero_std")
        norm.mean = state.get("mean")
        norm.std = state.get("std")
        norm.fitted = True
        norm.to(device)
        return norm

    def fit(
        self,
        data_loader,
        device,
        logging_dir,
        writer,
        data_cfg,
        checkpoint_cfg,
        train_cfg,
        prior_cfg,
    ):
        num_epochs = int(getattr(train_cfg, "num_epochs", 50))
        weight_decay = float(getattr(train_cfg, "weight_decay", 1.0e-4))
        learning_rate = float(getattr(train_cfg, "learning_rate", 1.0e-3))
        # === Checkpointing Setup ===
        optimizer = self.get_optimizer(lr=learning_rate, weight_decay=weight_decay)
        save_best_model = bool(getattr(checkpoint_cfg, "save_best_model", True))
        save_best_after_fraction = float(getattr(checkpoint_cfg, "save_best_after_fraction", 0.25))
        save_best_after_fraction = max(0.0, min(1.0, save_best_after_fraction))
        save_best_after_epoch = max(1, int(math.ceil(num_epochs * save_best_after_fraction)))
        if save_best_model:
            print(
                f"Best-model checkpointing starts at epoch {save_best_after_epoch} "
                f"({save_best_after_fraction:.0%} of training)."
            )
        else:
            print("Best-model checkpointing is disabled by config.training.checkpointing.save_best_model.")
        
        # === Early Stopping Setup ===
        use_early_stopping = bool(getattr(train_cfg.early_stopping, "enabled", False))
        stop_on_all_files_used = bool(getattr(train_cfg.early_stopping, "stop_on_all_files_used", True))
        min_epochs_before_full_data_stop = int(getattr(train_cfg.early_stopping, "min_epochs_before_full_data_stop", 1))
        early_stopper = None
        if use_early_stopping:
            from src.model.model_utils import EarlyStopping
            early_stopper = EarlyStopping(
                patience=getattr(train_cfg.early_stopping, "patience", 5),
                mode=getattr(train_cfg.early_stopping, "mode", "min"),
                min_delta=getattr(train_cfg.early_stopping, "min_delta", 0.0),
            )
        
        # === Prior Setup ===
        self.to(device)
        if hasattr(self, 'prior') and self.prior is not None and hasattr(self.prior, "to"):
            self.prior.to(device)
        if hasattr(self, 'normalizer') and self.normalizer is not None and hasattr(self.normalizer, "to"):
            self.normalizer.to(device)

        if not bool(getattr(prior_cfg, "init_from_data", True)) and hasattr(self, "prior") and isinstance(self.prior, VampPrior):
            init_loader = data_loader.get_train_epoch_loader(
                steps_per_epoch=1,
            )
            init_batch = next(iter(init_loader)) if init_loader is not None else None
            noise_scale = float(getattr(prior_cfg, "init_noise_scale", 0.01))
            _init_vamp_prior_from_batch(self, init_batch, device, noise_scale=noise_scale)

        # === Learning Rate Scheduler Setup ===
        if getattr(train_cfg.lr_scheduler, "enabled", False):
            scheduler_type = getattr(train_cfg.lr_scheduler, "scheduler_type", "step").lower()
            if scheduler_type == "step":
                step_size = int(getattr(train_cfg.lr_scheduler, "step_size", 10))
                gamma = float(getattr(train_cfg.lr_scheduler, "gamma", 0.1))
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                print(f"Using StepLR with step_size={step_size}, gamma={gamma}.")
            elif scheduler_type == "exponential":
                gamma = float(getattr(train_cfg.lr_scheduler, "gamma", 0.95))
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
                print(f"Using ExponentialLR with gamma={gamma}.")
            elif scheduler_type == "cosine":
                T_max = int(getattr(train_cfg.lr_scheduler, "T_max", num_epochs))
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
                print(f"Using CosineAnnealingLR with T_max={T_max}.")
            elif scheduler_type == "seq":
                warmup_epochs = 5
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, 
                    start_factor=0.01, 
                    end_factor=1.0, 
                    total_iters=warmup_epochs
                )
                decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=(num_epochs - warmup_epochs), 
                    eta_min=1e-6
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, decay_scheduler], 
                    milestones=[warmup_epochs]
                )
            else:
                raise ValueError(f"Unsupported lr_scheduler.type: {scheduler_type}")
        else:
            lr_scheduler = None

        # === Beta Scheduler Setup ===
        _steps_per_epoch_for_beta = int(getattr(data_cfg, "steps_per_epoch", 100))
        beta_scheduler = None
        if getattr(train_cfg.beta_annealing, "enabled", False):
            from src.model.model_utils import CyclicalBetaScheduler
            total_steps = num_epochs * _steps_per_epoch_for_beta
            beta_scheduler = CyclicalBetaScheduler(
                total_steps=total_steps,
                beta_max= float(getattr(train_cfg.beta, "beta", 1.0)),
                n_cycles= int(getattr(train_cfg.beta_annealing, "n_cycles", 1)),
                ratio= float(getattr(train_cfg.beta_annealing, "cycle_ratio", 0.5)),
            )
        
        # === Main Training Loop ===
        eval_batches = int(getattr(data_cfg, "eval_batches", 5))
        eval_every_n_epochs = int(getattr(train_cfg, "eval_every_n_epochs", 1))
        eval_cache_refresh = bool(getattr(data_cfg, "eval_cache_refresh", True))
        steps_per_epoch = int(getattr(data_cfg, "steps_per_epoch", 100))

        gradient_clip = float(getattr(train_cfg, "gradient_clip", True))
        gradient_clip_max_norm = float(getattr(train_cfg, "gradient_clip_max_norm", 1.0))

        best_eval_loss = float("inf")
        best_eval_loss_for_saving = float("inf")
        for epoch in range(num_epochs):
            epoch_start_t = time.perf_counter()
            should_eval_epoch = eval_batches > 0
            should_eval_epoch = eval_batches > 0 and ((epoch + 1) % max(1, eval_every_n_epochs) == 0)

            print(f"\nEpoch [{epoch+1}/{num_epochs}] — loading data ...", flush=True)
            cache_start_t = time.perf_counter()
            train_loader = data_loader.get_train_epoch_loader(
                steps_per_epoch = steps_per_epoch,
            )
            cache_time_s = time.perf_counter() - cache_start_t
            print(f"Epoch [{epoch+1}/{num_epochs}] — data ready in {cache_time_s:.1f}s, training ...", flush=True)
            file_progress = data_loader.get_file_progress()
            if "passes" in file_progress:
                pct_in_pass = (file_progress['seen'] / file_progress['total'] * 100) if file_progress['total'] > 0 else 0.0
                print(f"Data: pass {file_progress['completed'] + 1}, {pct_in_pass:.1f}% through ({file_progress['passes']:.2f}x total)")
            else:
                pct_of_data_seen = (file_progress['seen'] / file_progress['total'] * 100) if file_progress['total'] > 0 else 0.0
                print(f"Files seen so far: {file_progress['seen']} / {file_progress['total']} ({pct_of_data_seen:.1f}%)")
            if train_loader is None:
                print(f"[train] All training windows exhausted after {epoch} epoch(s). Stopping.")
                break
            
            self.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n = 0
            eval_time_s = 0.0
            save_time_s = 0.0

            train_start_t = time.perf_counter()
            for step, batch in enumerate(train_loader, start = 1):
                batch = self.move_to_device(batch, device)

                current_beta = beta_scheduler.step() if beta_scheduler is not None else train_cfg.beta

                loss_dict = self.train_core(
                    batch=batch,
                    optim=optimizer,
                    gradient_clip=gradient_clip,
                    gradient_clip_max_norm=gradient_clip_max_norm,
                    beta=current_beta,
                    weighted_loss=train_cfg.weighted_loss,
                    # TODO
                )
                epoch_loss += float(loss_dict["loss"].detach())
                epoch_recon += float(loss_dict["recon_loglikelihood"].detach())
                epoch_kl += float(loss_dict["kl_divergence"].detach())
                n += 1
            train_time_s = time.perf_counter() - train_start_t

            if n == 0:
                raise RuntimeError("No training batches were processed in this epoch. Cannot compute epoch metrics.")
            
            avg_loss = epoch_loss / n
            avg_recon = epoch_recon / n
            avg_kl = epoch_kl / n
            file_progress = data_loader.get_file_progress()
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
                f"data {cache_time_s:.1f}s  train {train_time_s:.1f}s"
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            writer.add_scalar('train_loss_epoch', avg_loss, epoch)
            writer.add_scalar('train_recon_loglikelihood_epoch', avg_recon, epoch)
            writer.add_scalar('train_kl_divergence_epoch', avg_kl, epoch)
            writer.add_scalar('stream_files_seen', file_progress['seen'], epoch)
            writer.add_scalar('stream_files_completed', file_progress['completed'], epoch)
            writer.add_scalar('stream_files_skipped', file_progress.get('skipped', 0), epoch)

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            epoch_eval_loader = data_loader.get_val_epoch_loader(
                eval_batches = eval_batches if should_eval_epoch else 0,
            )

            if should_eval_epoch and epoch_eval_loader is not None:
                eval_start_t = time.perf_counter()
                try:
                    eval_metrics = self.validate(
                        valloader=epoch_eval_loader,
                        num_samples=1,
                        beta=train_cfg.beta,
                        device=device,
                    )
                    print(
                        f"Epoch [{epoch+1:4d}/{num_epochs}] "
                        f"Eval Loss: {eval_metrics['loss']:.4f} | "
                        f"Recon: {eval_metrics['recon_loglikelihood']:.4f} | "
                        f"KL: {eval_metrics['kl_divergence']:.4f}"
                    )
                    writer.add_scalar('eval_loss_epoch', eval_metrics['loss'], epoch)
                    writer.add_scalar('eval_recon_loglikelihood_epoch', eval_metrics['recon_loglikelihood'], epoch)
                    writer.add_scalar('eval_kl_divergence_epoch', eval_metrics['kl_divergence'], epoch)
                    # Alias eval metrics to val_* tags for shared visualization.
                    writer.add_scalar('val_loss_epoch', eval_metrics['loss'], epoch)
                    writer.add_scalar('val_recon_loglikelihood_epoch', eval_metrics['recon_loglikelihood'], epoch)
                    writer.add_scalar('val_kl_divergence_epoch', eval_metrics['kl_divergence'], epoch)

                    if eval_metrics['loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['loss']
                    
                    if (
                        save_best_model
                        and (epoch + 1) >= save_best_after_epoch
                        and eval_metrics['loss'] < best_eval_loss_for_saving
                    ):
                        best_eval_loss_for_saving = eval_metrics['loss']
                        save_start_t = time.perf_counter()
                        best_model_name = f"best_model_{self.tag}_{epoch+1}"
                        self.save_model(
                            save_path=logging_dir,
                            model_name=best_model_name,
                            device=device,
                            normalizer=getattr(self, "normalizer", None)
                        )
                        self._best_checkpoint_path = os.path.join(logging_dir, f"{best_model_name}.pt")
                        save_time_s = time.perf_counter() - save_start_t

                    if use_early_stopping and early_stopper is not None:
                        early_stopper(val_metric=eval_metrics['loss'], train_metric=avg_loss, epoch=epoch)
                        if early_stopper.early_stop:
                            break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error during evaluation at epoch {epoch+1}: {e}")
                    raise e
                finally:
                    eval_time_s = time.perf_counter() - eval_start_t
            if use_early_stopping and early_stopper is not None and early_stopper.early_stop:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best eval loss: {best_eval_loss:.4f} at epoch {epoch+1 - early_stopper.patience}")
                print(f"{'='*60}\n")
                break
            
            if (epoch + 1) % checkpoint_cfg.save_freq == 0:
                save_start_t = time.perf_counter()
                self.save_model(
                    save_path=logging_dir,
                    model_name=f"{self.tag}_epoch_{epoch+1}",
                    device=device,
                    normalizer=getattr(self, "normalizer", None)
                )

            total_epoch_time_s = time.perf_counter() - epoch_start_t
            print(
                f"Epoch [{epoch+1}/{num_epochs}] completed in {total_epoch_time_s:.2f}s "
                f"(Cache: {cache_time_s:.2f}s, Train: {train_time_s:.2f}s, Eval: {eval_time_s:.2f}s, Save: {save_time_s:.2f}s)"
            )

            if (
                stop_on_all_files_used
                and file_progress['total'] > 0
                and file_progress['completed'] >= file_progress['total']
                and (epoch + 1) >= max(1, min_epochs_before_full_data_stop)
            ):
                print(
                    f"Early stop (all data used): completed {file_progress['completed']} "
                    f"of {file_progress['total']} files at epoch {epoch+1}."
                )
                break
        
        self.save_model(
            save_path=logging_dir,
            model_name=f"{self.tag}_final",
            device=device,
        )

    def get_tag(self):
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.__class__.__name__}_{time_stamp}"
    def _prep_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        SHAPER:
          x  : [batch_size, sequence_length, num_features] | [batch_size, sequence_length * num_features]
          ->  : [batch_size, sequence_length * num_features]
        """
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1)  # SHAPER: [batch_size, sequence_length, num_features] -> [batch_size, sequence_length * num_features]
        return x
    def _to_phase_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        SHAPER:
          x  : [batch_size, sequence_length * num_features]
             | [batch_size, num_features, sequence_length]
             | [batch_size, sequence_length, num_features]
          ->  : [batch_size, num_features, sequence_length]
        """
        if x.dim() == 2:
            if x.shape[1] != self.sequence_length * self.num_features:
                raise ValueError(
                    f"Expected flattened input dim {self.sequence_length * self.num_features}, got {x.shape[1]}"
                )
            return x.reshape(x.shape[0], self.num_features, self.sequence_length)  # SHAPER: [batch_size, sequence_length * num_features] -> [batch_size, num_features, sequence_length]

        if x.dim() != 3:
            raise ValueError(f"Expected 2D/3D input, got shape {tuple(x.shape)}")

        if x.shape[1] == self.num_features and x.shape[2] == self.sequence_length:
            return x  # SHAPER: [batch_size, num_features, sequence_length] already phase-first

        if x.shape[1] == self.sequence_length and x.shape[2] == self.num_features:
            return x.transpose(1, 2)  # SHAPER: [batch_size, sequence_length, num_features] -> [batch_size, num_features, sequence_length]

        raise ValueError(
            f"Could not infer phase/time axes for input shape {tuple(x.shape)} with "
            f"num_features={self.num_features}, sequence_length={self.sequence_length}"
        )

    def _build_condition_input(self, x_phase_first: torch.Tensor) -> torch.Tensor:
        """
        SHAPER:
          x_phase_first : [batch_size, num_features, sequence_length]
          ->             : [batch_size, sequence_length, num_features]
        """
        return x_phase_first.transpose(1, 2)  # SHAPER: [batch_size, num_features, sequence_length] -> [batch_size, sequence_length, num_features]

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

    def _unpack_batch(self, batch):
        """Strict unpack contract: (x, meter_id, timestamps, agg)."""
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            raise ValueError(
                "Expected batch as a 4-tuple: (x, meter_id, timestamps, agg)."
            )

        x, meter_id, timestamps, agg = batch

        if torch.is_tensor(meter_id):
            meter_id = meter_id.long()
        if torch.is_tensor(timestamps):
            timestamps = timestamps.long()

        return x, meter_id, timestamps, agg
    def _validate_condition_dim(self, c: torch.Tensor) -> None:
        if self._condition_dim_checked:
            return
        expected_dim = getattr(getattr(self, "cond_proj", None), "in_features", None)
        if expected_dim is not None and c.shape[-1] != expected_dim:
            raise ValueError(f"Condition dim mismatch: model expects {expected_dim}, got {c.shape[-1]}")
        self._condition_dim_checked = True

    def _build_conditions(self, x_phase_first, meter_id=None, timestamps=None, device=None):
        """
        SHAPER:
          x_phase_first : [batch_size, num_features, sequence_length]
          ->             : [batch_size, condition_dim]
        """
        cond_input = self._build_condition_input(x_phase_first)  # SHAPER: [batch_size, sequence_length, num_features]
        if meter_id is None and timestamps is None:
            c = self.conditioner.get_conditions(cond_input)
        else:
            c = self.conditioner.get_conditions(cond_input, meter_id=meter_id, timestamps=timestamps)
        # SHAPER: c = [batch_size, condition_dim]

        c = self.move_to_device(c, device=device)
        if c.dim() == 1:
            c = c.unsqueeze(0)  # SHAPER: [condition_dim] -> [1, condition_dim]
        c = c.float()
        self._validate_condition_dim(c)
        return c