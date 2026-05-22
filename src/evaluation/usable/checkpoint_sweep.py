"""
Compare all best_model_* checkpoints in a model folder to detect overfitting
and identify the best-performing epoch on evaluation metrics.

Outputs are written to <model_folder>/checkpoint_sweep/:
  epoch_<N>/eval_results.json   — scalar metrics for that checkpoint
  metrics_reconstruction.png    — line chart: reconstruction metrics vs epoch
  metrics_synthetic.png         — line chart: synthetic metrics vs epoch
  synthetic_score_summary.png   — MinMax synthetic score vs epoch
  comparison_table.txt          — table with epochs as rows, metrics as columns
  best_checkpoint.txt           — best epoch ranked by MinMax synthetic score

Checkpoints are automatically skipped if their eval_results.json already exists
(safe to resume interrupted sweeps).

Usage:
    python src/evaluation/usable/checkpoint_sweep.py <model_folder_path> [options]

Options:
    --all              Evaluate every checkpoint (default: evenly spaced subset)
    --n-checkpoints N  Number of checkpoints in subset (default: 10)
    --n-samples N      Synthetic samples per checkpoint eval (default: 1000)
    --device DEVICE    Torch device (default: cuda if available, else cpu)
"""

import argparse
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model.factory import _load_train_config, create_model_from_config
from src.evaluation.load_and_eval_model import setup_data_loader, _load_config_for_model
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite


# ---------------------------------------------------------------------------
# Fixed-batch loader: snapshot test data once so every checkpoint eval
# iterates the exact same windows regardless of streaming randomness.
# ---------------------------------------------------------------------------

class _FixedLoader:
    """Minimal DataLoader-like wrapper over a pre-collected list of batches.

    The streaming path (StreamingParquetBuffer) fills a circular buffer from a
    background thread, so two calls to iter(test_loader) can yield different
    windows due to thread scheduling.  Snapshotting all batches into this class
    once guarantees every checkpoint evaluation — and any later cross-model
    comparison — sees the same windows.
    """

    def __init__(self, batches: list):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def snapshot_test_loader(test_loader) -> _FixedLoader:
    """Eagerly consume test_loader and return a _FixedLoader over those batches."""
    print("Snapshotting test batches into memory for deterministic evaluation...")
    batches = list(test_loader)
    if not batches:
        raise ValueError("Test loader yielded zero batches — cannot snapshot.")
    total_windows = sum(b[0].shape[0] for b in batches)
    print(f"  Snapshot: {len(batches)} batches, {total_windows} windows total.")
    return _FixedLoader(batches)


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

_SYNTHETIC_METRIC_KEYS = {
    "aggregatedprofilesflexibilitydistance",
    "aggregatedprofiles",
    "tstrtrainonsynthetictestonreal",
    "trainonsynthetictestonrealtstr",
    "tstr",
    "maximummeandiscrepancymmd",
    "maximummeandiscrepancy",
    "setmmdpermutationinvariant",
    "mmd",
    "acfdistributionmatchks",
    "acfdistribution",
    "hourlyconsumptionbymonth",
    "hourlyconsumptionbyday",
    "peakanalysis",
    "hourlyprofilesbydayofweek",
    "hourlyprofilesbymonth",
}

_RECONSTRUCTION_METRIC_KEYS = {
    "flexibilitydistance",
    "reconstructionflexibilitydistancefd",
    "reconstructionloglikelihood",
    "reconstructionrmse",
}

_GOAL_MAP = {
    "flexibilitydistance": "lower",
    "reconstructionflexibilitydistancefd": "lower",
    "reconstructionloglikelihood": "higher",
    "reconstructionrmse": "lower",
    "aggregatedprofilesflexibilitydistance": "lower",
    "aggregatedprofiles": "lower",
    "tstrtrainonsynthetictestonreal": "target_1",
    "trainonsynthetictestonrealtstr": "target_1",
    "tstr": "target_1",
    "maximummeandiscrepancymmd": "target_0",
    "maximummeandiscrepancy": "target_0",
    "setmmdpermutationinvariant": "target_0",
    "mmd": "target_0",
    "acfdistributionmatchks": "lower",
    "acfdistribution": "lower",
    "hourlyconsumptionbymonth": "lower",
    "hourlyconsumptionbyday": "lower",
    "peakanalysis": "lower",
    "hourlyprofilesbydayofweek": "lower",
    "hourlyprofilesbymonth": "lower",
    "classicalstatistics": "lower",
}


def _canonical(metric: str) -> str:
    return (
        str(metric).lower()
        .replace("-", "").replace("_", "").replace(" ", "")
        .replace("(", "").replace(")", "")
    )


def _metric_group(metric: str) -> str | None:
    key = _canonical(metric)
    if key in _RECONSTRUCTION_METRIC_KEYS:
        return "reconstruction"
    if key in _SYNTHETIC_METRIC_KEYS:
        return "synthetic"
    return None


def _metric_goal(metric: str) -> str:
    return _GOAL_MAP.get(_canonical(metric), "lower")


def _goal_arrow(metric: str) -> str:
    goal = _metric_goal(metric)
    return {"higher": "↑", "lower": "↓", "target_0": "→0", "target_1": "→1"}.get(goal, "↓")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_model_checkpoints(model_path: str) -> list[tuple[int, str]]:
    """Return sorted (epoch, abs_path) pairs for all best_model_*.pt files."""
    checkpoints = []
    for fname in os.listdir(model_path):
        m = re.match(r"best_model_.+_(\d+)\.pt$", fname)
        if m:
            checkpoints.append((int(m.group(1)), os.path.join(model_path, fname)))
    return sorted(checkpoints)


def select_evenly_spaced(checkpoints: list, n: int) -> list:
    """Pick n evenly spaced items (by list index) from a sorted checkpoint list."""
    if n >= len(checkpoints):
        return checkpoints
    indices = np.linspace(0, len(checkpoints) - 1, n, dtype=int)
    seen: set[int] = set()
    result = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            result.append(checkpoints[i])
    return result


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def _extract_scalar(value) -> float | None:
    """Extract a float from a metric value that may be a scalar or a nested dict."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, dict):
        raw = value.get("value")
        return _extract_scalar(raw)
    return None


def eval_one_checkpoint(
    model,
    checkpoint_path: str,
    test_loader,
    config,
    device: str,
    n_synthetic_samples: int,
) -> dict:
    """Load checkpoint weights, run eval in a temp dir, return flat scalar dict.

    PDFs and other artifacts from individual metrics land in the temp dir and
    are discarded automatically — only the JSON scalar values are returned.
    """
    normalizer, conditioner = model.load_model(checkpoint_path, device)
    model.eval()

    eval_norm = copy.deepcopy(normalizer) if normalizer is not None else None
    if eval_norm is not None and hasattr(eval_norm, "to"):
        eval_norm = eval_norm.to(device)

    data_mode = str(getattr(config.data, "mode", "csv")).lower()
    phased_mode = data_mode in {"phased", "phased_parquet"}

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "eval_results.json")

        if phased_mode:
            evaluator = PhasedEvaluationSuite(
                model, test_loader,
                normalizer=eval_norm,
                conditioner=conditioner,
                device=device,
                window_size=int(config.data.window.input_len),
                n_synthetic_samples=n_synthetic_samples,
                output_path=tmp,
            )
            evaluator.evaluate()
            # Use aggregated_phase as the canonical per-checkpoint result.
            agg_json = os.path.join(tmp, "aggregated_phase_eval", "eval_results.json")
            raw = {}
            if os.path.exists(agg_json):
                with open(agg_json) as f:
                    raw = json.load(f)
        else:
            evaluator = EvalFramework(
                model, test_loader,
                eval_norm,
                conditioner=conditioner,
                device=device,
                n_synthetic_samples=n_synthetic_samples,
                # Passing a .json path: EvalFramework resolves the parent dir
                # for metric plot artifacts and save_results writes the JSON here.
                output_path=json_path,
            )
            evaluator.register_all()
            evaluator.evaluate()
            evaluator.save_results(json_path)
            with open(json_path) as f:
                raw = json.load(f)

    scalars: dict[str, float] = {}
    for k, v in raw.items():
        s = _extract_scalar(v)
        if s is not None:
            scalars[k] = s
    return scalars


# ---------------------------------------------------------------------------
# MinMax synthetic scoring
# ---------------------------------------------------------------------------

def compute_minmax_synthetic_scores(all_results: dict[int, dict]) -> dict[int, float]:
    """Compute normalized MinMax synthetic score (1.0=best, 0.0=worst) per epoch."""
    epochs = sorted(all_results.keys())
    all_keys = set().union(*[r.keys() for r in all_results.values()])
    synth_metrics = [m for m in all_keys if _metric_group(m) == "synthetic"]

    if not synth_metrics:
        return {e: 0.0 for e in epochs}

    per_epoch_scores: dict[int, list[float]] = {e: [] for e in epochs}

    for metric in synth_metrics:
        goal = _metric_goal(metric)
        available = {e: all_results[e][metric] for e in epochs if metric in all_results[e]}
        if not available:
            continue

        # Map so that lower-mapped value = worse
        mapped: dict[int, float] = {}
        for e, v in available.items():
            if goal == "higher":
                mapped[e] = -v
            elif goal == "target_1":
                mapped[e] = abs(v - 1.0)
            elif goal == "target_0":
                mapped[e] = abs(v)
            else:
                mapped[e] = v

        vmin = min(mapped.values())
        vmax = max(mapped.values())
        for e, val in mapped.items():
            if vmax == vmin:
                per_epoch_scores[e].append(1.0)
            else:
                per_epoch_scores[e].append(1.0 - (val - vmin) / (vmax - vmin))

    return {
        e: float(np.mean(per_epoch_scores[e])) if per_epoch_scores[e] else 0.0
        for e in epochs
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_metrics_over_epochs(
    all_results: dict[int, dict],
    synth_scores: dict[int, float],
    output_dir: str,
) -> None:
    """Save one PNG per metric group plus a summary synthetic score chart."""
    epochs = sorted(all_results.keys())
    all_metric_names = sorted(set().union(*[r.keys() for r in all_results.values()]))

    for group in ("reconstruction", "synthetic"):
        group_metrics = [m for m in all_metric_names if _metric_group(m) == group]
        if not group_metrics:
            continue

        n = len(group_metrics)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        title = "Reconstruction Metrics" if group == "reconstruction" else "Synthetic Metrics"
        fig.suptitle(f"{title} vs Training Epoch", fontsize=14, fontweight="bold")

        for idx, metric in enumerate(group_metrics):
            ax = axes_flat[idx]
            ys = [all_results[e].get(metric) for e in epochs]
            valid_pairs = [(e, y) for e, y in zip(epochs, ys) if y is not None]
            if not valid_pairs:
                ax.set_title(f"{metric}\n(no data)", fontsize=9)
                continue

            ve, vy = zip(*valid_pairs)
            ax.plot(ve, vy, marker="o", markersize=4, linewidth=1.5)

            goal = _metric_goal(metric)
            if goal == "higher":
                best_i = int(np.argmax(vy))
            elif goal == "target_1":
                best_i = int(np.argmin([abs(v - 1.0) for v in vy]))
            elif goal == "target_0":
                best_i = int(np.argmin([abs(v) for v in vy]))
            else:
                best_i = int(np.argmin(vy))

            ax.axvline(ve[best_i], color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.scatter([ve[best_i]], [vy[best_i]], color="red", zorder=5, s=60)
            ax.set_title(f"{metric} ({_goal_arrow(metric)})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.5)

        for i in range(n, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"metrics_{group}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Overall synthetic score summary
    if synth_scores:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = sorted(synth_scores.keys())
        ys = [synth_scores[e] for e in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        best_epoch = max(synth_scores, key=synth_scores.get)
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
        ax.scatter(
            [best_epoch], [synth_scores[best_epoch]],
            color="red", zorder=5, s=80,
            label=f"Best: epoch {best_epoch} ({synth_scores[best_epoch]:.3f})",
        )
        ax.set_title("MinMax Synthetic Score vs Epoch (↑ better)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg MinMax Synthetic Score")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "synthetic_score_summary.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def save_comparison_table(
    all_results: dict[int, dict],
    synth_scores: dict[int, float],
    output_dir: str,
) -> None:
    """Write a text table: rows=epochs, columns=metrics + MinMax synthetic score."""
    epochs = sorted(all_results.keys())
    all_metrics = sorted(set().union(*[r.keys() for r in all_results.values()]))

    header = ["Epoch"] + all_metrics + ["MinMax Synth Score"]
    col_w = [max(len(h), 8) for h in header]

    def fmt(v) -> str:
        return "N/A" if v is None else f"{v:.4f}"

    rows = []
    for e in epochs:
        row = [str(e)]
        for m in all_metrics:
            row.append(fmt(all_results[e].get(m)))
        row.append(fmt(synth_scores.get(e)))
        rows.append(row)

    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def fmt_row(cells: list[str]) -> str:
        return "|" + "|".join(f" {c:<{col_w[i]}} " for i, c in enumerate(cells)) + "|"

    lines = [sep, fmt_row(header), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)

    out_path = os.path.join(output_dir, "comparison_table.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Best checkpoint report
# ---------------------------------------------------------------------------

def save_best_checkpoint_report(
    synth_scores: dict[int, float],
    model_path: str,
    output_dir: str,
) -> None:
    """Write best_checkpoint.txt ranked by MinMax synthetic score."""
    if not synth_scores:
        return

    ranked = sorted(synth_scores.items(), key=lambda x: x[1], reverse=True)
    best_epoch, best_score = ranked[0]

    best_ckpt = None
    for fname in os.listdir(model_path):
        m = re.match(r"best_model_.+_(\d+)\.pt$", fname)
        if m and int(m.group(1)) == best_epoch:
            best_ckpt = os.path.join(model_path, fname)
            break

    lines = [
        "CHECKPOINT SWEEP — BEST EPOCH RANKING",
        "=" * 50,
        "Ranked by: MinMax Synthetic Score (↑ higher = better)",
        "",
        f"BEST CHECKPOINT: epoch {best_epoch}",
        f"  MinMax Synthetic Score : {best_score:.4f}",
    ]
    if best_ckpt:
        lines.append(f"  Checkpoint path        : {best_ckpt}")
    lines += [
        "",
        "Full ranking:",
        f"  {'Epoch':<8} {'MinMax Synthetic Score':>22}",
        "  " + "-" * 32,
    ]
    for epoch, score in ranked:
        lines.append(f"  {epoch:<8} {score:>22.4f}")

    out_path = os.path.join(output_dir, "best_checkpoint.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")
    print(f"\n*** BEST EPOCH: {best_epoch}  (MinMax Synthetic Score: {best_score:.4f}) ***")
    if best_ckpt:
        print(f"    Checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep best_model_* checkpoints in a model folder to compare eval metrics."
    )
    parser.add_argument("model_path", type=str, help="Path to trained model folder.")
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate every checkpoint (default: evenly spaced subset).",
    )
    parser.add_argument(
        "--n-checkpoints", type=int, default=10,
        help="Number of checkpoints to evaluate when not using --all (default: 10).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Synthetic samples per checkpoint evaluation (default: 1000).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    model_path = str(args.model_path).rstrip("/\\")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    print(f"\n{'='*70}")
    print("CHECKPOINT SWEEP")
    print(f"{'='*70}")
    print(f"Model path  : {model_path}")
    print(f"Device      : {device}")
    print(f"N samples   : {args.n_samples}")

    # 1. Find and select checkpoints
    checkpoints = find_best_model_checkpoints(model_path)
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pt files found in {model_path}")

    if args.all:
        selected = checkpoints
    else:
        selected = select_evenly_spaced(checkpoints, args.n_checkpoints)

    epochs_selected = [e for e, _ in selected]
    print(f"Checkpoints : {len(checkpoints)} total → {len(selected)} selected")
    print(f"Epochs      : {epochs_selected}\n")

    # 2. Load config and build model skeleton
    config = _load_config_for_model(model_path)
    model = create_model_from_config(config, device)

    # 3. Load the first selected checkpoint to get the normalizer for the data loader
    first_normalizer, _ = model.load_model(selected[0][1], device)

    # 4. Build test loader and immediately snapshot it so every checkpoint
    #    evaluation — and any later cross-model comparison — sees identical windows.
    print("Building test data loader...")
    test_loader = setup_data_loader(config, first_normalizer, device=device, model_path=model_path)
    test_loader = snapshot_test_loader(test_loader)

    # 5. Create output directory
    sweep_dir = os.path.join(model_path, "checkpoint_sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"Output dir  : {sweep_dir}\n")

    # 6. Evaluate each selected checkpoint
    all_results: dict[int, dict] = {}

    for epoch, ckpt_path in selected:
        epoch_dir = os.path.join(sweep_dir, f"epoch_{epoch}")
        json_path = os.path.join(epoch_dir, "eval_results.json")

        if os.path.exists(json_path):
            print(f"[epoch {epoch:>4}] Skipping (cached)")
            with open(json_path) as f:
                all_results[epoch] = json.load(f)
            continue

        print(f"[epoch {epoch:>4}] Evaluating {os.path.basename(ckpt_path)} ...")
        try:
            results = eval_one_checkpoint(
                model=model,
                checkpoint_path=ckpt_path,
                test_loader=test_loader,
                config=config,
                device=device,
                n_synthetic_samples=args.n_samples,
            )
            os.makedirs(epoch_dir, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            all_results[epoch] = results
            synth_keys = [k for k in results if _metric_group(k) == "synthetic"]
            recon_keys = [k for k in results if _metric_group(k) == "reconstruction"]
            print(
                f"[epoch {epoch:>4}] Done. "
                f"recon={len(recon_keys)} synthetic={len(synth_keys)} metrics"
            )
        except Exception as e:
            import traceback
            print(f"[epoch {epoch:>4}] ERROR: {e}")
            traceback.print_exc()

    if not all_results:
        print("No results collected. Exiting.")
        return

    # 7. Compute MinMax synthetic scores
    synth_scores = compute_minmax_synthetic_scores(all_results)

    # 8. Generate outputs
    print()
    plot_metrics_over_epochs(all_results, synth_scores, sweep_dir)
    save_comparison_table(all_results, synth_scores, sweep_dir)
    save_best_checkpoint_report(synth_scores, model_path, sweep_dir)

    print(f"\n{'='*70}")
    print(f"Sweep complete. Results in: {sweep_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()