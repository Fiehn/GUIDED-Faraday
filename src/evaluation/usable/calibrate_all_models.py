"""Standalone script: calibrate dispersion temperature for all saved model checkpoints.

Usage:
    python calibrate_all_models.py <glob_pattern> [--num-windows N] [--device cpu]

Example:
    python calibrate_all_models.py "models/*" --num-windows 2000

For each checkpoint directory that matches the glob pattern the script:
  1. Loads the model via the existing load_model() entry-point.
  2. Runs DispersionCalibrator.calibrate() to measure sigma_z.
  3. Writes calibrated_temperature back into the checkpoint file.
  4. Prints a summary table.

QuantileBaseline and other non-sampling baselines are skipped silently.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.load_model import load_model
from model.dispersion_calibrator import DispersionCalibrator
from model.generators import SyntheticGenerator


def _find_checkpoint_file(model_dir: str) -> str | None:
    """Return the .pt checkpoint path used by load_model(), or None."""
    import os
    import re
    pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    max_epoch, best = -1, None
    for f in pt_files:
        m = re.match(r"best_model_.+_(\d+)\.pt$", f)
        if m:
            epoch = int(m.group(1))
            if epoch > max_epoch:
                max_epoch, best = epoch, os.path.join(model_dir, f)
    if best is None and pt_files:
        pt_full = [os.path.join(model_dir, f) for f in pt_files]
        best = max(pt_full, key=os.path.getmtime)
    return best


def calibrate_directory(model_dir: str, num_windows: int, device: str) -> dict | None:
    """Calibrate one model directory. Returns result dict or None if skipped."""
    model_name = Path(model_dir).name
    try:
        model, normalizer, conditioner = load_model(model_dir, device=device)
    except Exception as e:
        print(f"  [SKIP] {model_name}: failed to load — {e}")
        return None

    gen = SyntheticGenerator(model, conditioner=conditioner, normalizer=normalizer, device=device)

    temperature = DispersionCalibrator.calibrate(gen, num_windows=num_windows, device=device)

    if temperature is None:
        tag = getattr(model, "tag", "unknown")
        print(f"  [SKIP] {model_name} ({tag}): baseline model, no z-sampling")
        return None

    sigma_z = 1.0 / temperature

    checkpoint_path = _find_checkpoint_file(model_dir)
    if checkpoint_path is None:
        print(f"  [WARN] {model_name}: no checkpoint file found, cannot save")
    else:
        DispersionCalibrator.save_to_checkpoint(checkpoint_path, temperature)

    return {"model": model_name, "sigma_z": sigma_z, "temperature": temperature}


def main():
    parser = argparse.ArgumentParser(description="Batch-calibrate dispersion temperature for saved checkpoints")
    parser.add_argument("glob_pattern", help="Glob pattern over model checkpoint directories")
    parser.add_argument("--num-windows", type=int, default=1000,
                        help="Calibration sample count (default: 1000)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    dirs = sorted(glob.glob(args.glob_pattern))
    if not dirs:
        print(f"No directories matched: {args.glob_pattern}")
        sys.exit(1)

    print(f"Calibrating {len(dirs)} model(s) with {args.num_windows} windows on {args.device}...\n")

    results = []
    for model_dir in dirs:
        if not Path(model_dir).is_dir():
            continue
        print(f"→ {Path(model_dir).name}")
        result = calibrate_directory(model_dir, args.num_windows, args.device)
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo models were calibrated.")
        return

    # Print summary table
    col_w = max(len(r["model"]) for r in results) + 2
    print(f"\n{'Model':<{col_w}} {'sigma_z':>10} {'temperature':>12}")
    print("-" * (col_w + 24))
    for r in results:
        print(f"{r['model']:<{col_w}} {r['sigma_z']:>10.4f} {r['temperature']:>12.4f}")
    print(f"\nCalibrated temperature written to checkpoint for {len(results)} model(s).")


if __name__ == "__main__":
    main()