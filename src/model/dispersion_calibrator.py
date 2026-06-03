"""DispersionCalibrator: post-hoc temperature calibration for Jensen bias correction.

Neural generative models trained with log1p normalisation produce under-dispersed
samples (sigma_z < 1).  Because expm1 is convex, this shrinks the inverse-transformed
mean by 12–33%.  Temperature T = 1/sigma_z restores unit variance in z-space and
recovers the correct conditional mean without retraining.
"""
from __future__ import annotations

import torch


class DispersionCalibrator:
    """Measures sigma_z for a trained model and returns the corrective temperature."""

    @staticmethod
    def calibrate(generator, num_windows: int = 1000, device: str = "cpu") -> float | None:
        """Generate num_windows samples in z-space, compute sigma_z, return T = 1/sigma_z.

        Args:
            generator: SyntheticGenerator wrapping the model to calibrate.
            num_windows: Number of calibration samples to draw. Default 1000.
            device: Device to run on. Default "cpu" so no GPU is required.

        Returns:
            Calibrated temperature T = 1/sigma_z, or None if the model is a baseline
            (baselines do not generate z-values through sampling).
        """
        if generator._is_baseline_model():
            return None

        model = generator.model
        model.eval()
        model.to(device)

        is_cvae = getattr(generator, "_is_cvae", False)

        with torch.no_grad():
            if is_cvae:
                conditioner = generator.conditioner
                cond = conditioner.get_random_conditions(batch_size=num_windows).to(device)
                raw = model.sample(
                    conditions=cond,
                    num_prior_samples=1,
                    num_x_samples=1,
                    device=device,
                    temperature=1.0,
                )
                # [1, 1, B, D] -> flat
                flat = raw.reshape(-1).float()
            else:
                raw = model.sample(
                    num_samples_prior=num_windows,
                    num_samples_posterior=1,
                    device=device,
                    temperature=1.0,
                )
                if isinstance(raw, dict) and "samples" in raw:
                    raw = raw["samples"]
                flat = raw.reshape(-1).float()

        sigma_z = float(flat.std().item())
        if sigma_z < 1e-6:
            return 1.0
        return 1.0 / sigma_z

    @staticmethod
    def save_to_checkpoint(checkpoint_path: str, temperature: float) -> None:
        """Write calibrated_temperature into an existing .pt checkpoint file."""
        with open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        ckpt["calibrated_temperature"] = float(temperature)
        with open(checkpoint_path, "wb") as f:
            torch.save(ckpt, f)

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> float | None:
        """Read calibrated_temperature from a .pt checkpoint, or None if absent."""
        with open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        return ckpt.get("calibrated_temperature", None)