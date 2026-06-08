"""
24-hour real vs. synthetic comparison for the 7 phased models.

Outputs (all in evaluation_comparison/3-phase-generated/):
  grid_mean.pdf          — 2×4 landscape grid, mean of 12 draws
  grid_single.pdf        — 2×4 landscape grid, single draw
  appendix_mean.pdf      — multi-page, one page per model, real+mean side-by-side
  appendix_single.pdf    — multi-page, one page per model, real+single side-by-side
  standalone/
    {name}_mean.pdf      — real+mean side-by-side  (×7)
    {name}_single.pdf    — real+single side-by-side (×7)
  generation_timing.txt  — wall-clock time per model for mean-of-12 and single draw

Phase ordering: all panels sorted ascending by per-panel mean energy.
Phase labels: L_min / L_mid / L_max + Aggregated.
Y-axis: shared across all 8 panels in grid; shared between real/synthetic per page.

Usage:
    uv run src/evaluation/usable/plot_model_comparison_24h.py \
        --data-path "data/3p/sm_101015.parquet" \
        [--ev-path "data/EV/EV_detection_reduced_60000_SM.csv"]
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent

for _p in [str(ROOT_DIR), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.axes
import matplotlib.figure
from src.evaluation.load_model import load_model
from src.model.factory import _load_train_config
from src.model.generators import SyntheticGenerator
from src.visualization.plot_styling import PlotStyle


OUTPUT_DIR = os.path.join("evaluation_comparison", "3-phase-generated")
STANDALONE_DIR = os.path.join(OUTPUT_DIR, "standalone")
N_MEAN_DRAWS = 12
PHASE_LABELS = ["L_min", "L_mid", "L_max"]

MODELS = [
    "E:/models/phased2/SetGuideVAE_20260607_124614",
    "E:/models/phased2/SetGuideVAE_20260607_132508_CNN",
    "E:/models/phased2/DiffusionModel_20260607_080656",
    "E:/models/phased2/SequentialPhaseVAE_20260607_082110",
    "E:/models/new4/GuidedFaradayModel_20260605_105946",
    "E:/models/phased2/QuantileBaselineModel_20260607_112946",
    "E:/models/phased2/AverageBaselineModel_20260607_113136",
]
SECONDARY_MODELS = [
    None,
    None,
    None,
    None,
    "E:/models/phased/HierarchicalSplitter_20260604_171559",
    None,
    None,
]
NAMES = [
    "Set-Guide-VAE",
    "Set-Guide-VAE-CNN",
    "Diffusion (Phased)",
    "Sequential",
    "Guide-CNN + Hierarchical Splitter",
    "Quantile Baseline",
    "Average Baseline",
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_config(model_path: str) -> Any:
    preferred = os.path.join(model_path, "training_config_used.yaml")
    if os.path.exists(preferred):
        return _load_train_config(preferred)
    for fname in os.listdir(model_path):
        if fname.endswith(".yaml"):
            return _load_train_config(os.path.join(model_path, fname))
    raise FileNotFoundError(f"No .yaml config found in {model_path}")


def _get_aggregation_divisor(model_path: str) -> float:
    try:
        cfg = _resolve_config(model_path)
        data_cfg = getattr(cfg, "data", SimpleNamespace())
        unit = str(getattr(data_cfg, "aggregation_energy_unit", "wh")).lower()
        return 1000.0 if unit == "kwh" else 1.0
    except Exception:
        traceback.print_exc()
        print(f"  WARNING: could not read aggregation_divisor from {model_path}, defaulting to 1.0")
        return 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_real_window(
    data_path: str,
    seq_len: int,
    aggregation_divisor: float,
) -> tuple[np.ndarray, int, int]:
    """Load a random 24h window from a phased parquet file using InMemoryPhasedDataloaders.

    Returns (values [seq_len, num_phases], meter_id, unix_timestamp).
    """
    from src.data.dataloader_phased import InMemoryPhasedDataloaders

    _PHASED_FEATURE_COLS = ["active_power_p14_l1", "active_power_p14_l2", "active_power_p14_l3"]

    dl = InMemoryPhasedDataloaders(
        sm_files=[data_path],
        feature_cols=_PHASED_FEATURE_COLS,
        window_len=seq_len,
        stride=seq_len,
        batch_size=256,
        aggregation=aggregation_divisor > 1.0,
        aggregation_divisor=aggregation_divisor,
        split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
        normalizer=None,
        num_workers=0,
    )

    ds = dl.train_dataset
    if len(ds) == 0:
        raise ValueError(f"No windows loaded from {data_path} — check the file and feature columns.")

    idx = int(torch.randint(len(ds), (1,)).item())
    # ds.x shape: [N, num_phases, seq_len] — take one window and transpose to [seq_len, num_phases]
    values = ds.x[idx].cpu().numpy().T
    meter_id = int(ds.meter_ids[idx].item())
    timestamp = int(ds.timestamps[idx].item())
    return values, meter_id, timestamp


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _sort_phases(series: np.ndarray) -> np.ndarray:
    """Sort columns ascending by mean energy; no-op for single-phase output."""
    if series.ndim < 2 or series.shape[1] < 2:
        return series
    return series[:, np.argsort(series.mean(axis=0))]


def _compute_ylim(series_list: list[np.ndarray | None], pad: float = 0.05) -> tuple[float, float]:
    """Shared y-limits covering all individual phases and their aggregated sums."""
    all_vals: list[float] = []
    for s in series_list:
        if s is None:
            continue
        all_vals.extend(s.flatten().tolist())
        if s.ndim > 1 and s.shape[1] > 1:
            all_vals.extend(s.sum(axis=1).tolist())
    if not all_vals:
        return 0.0, 1.0
    y_min = min(min(all_vals), 0.0)
    y_max = max(all_vals)
    rng = max(y_max - y_min, 1e-6)
    return y_min - rng * pad, y_max + rng * pad


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_panel(ax: matplotlib.axes.Axes, series: np.ndarray, title: str, style: PlotStyle) -> None:
    """Draw L_min/L_mid/L_max phases + Aggregated on ax. Caller sets ylim."""
    hours = np.arange(series.shape[0])
    n = series.shape[1] if series.ndim > 1 else 1

    if n > 1:
        agg = series.sum(axis=1)
        ax.plot(hours, agg, label="Aggregated", color=style.color_original, linewidth=2.0, zorder=3)
        for i, lbl in enumerate(PHASE_LABELS[:n]):
            ax.plot(hours, series[:, i], label=lbl, color=style.get_color(i),
                    alpha=0.85, linewidth=1.5, marker="o", markersize=2)
    else:
        vals = series[:, 0] if series.ndim > 1 else series
        ax.plot(hours, vals, label=PHASE_LABELS[0], color=style.get_color(0), linewidth=1.8)

    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Hour", fontsize=8)
    ax.set_ylabel("Load (kWh)", fontsize=8)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)


def _make_side_by_side(
    real: np.ndarray,
    gen: np.ndarray | None,
    model_name: str,
    draw_key: str,
    style: PlotStyle,
    meter_id: int,
    date_str: str,
) -> matplotlib.figure.Figure:
    """Create a real (left) + synthetic (right) figure with shared y-axis."""
    y_min, y_max = _compute_ylim([real, gen])
    draw_label = f"Mean of {N_MEAN_DRAWS} Draws" if draw_key == "mean" else "Single Draw"

    fig, (ax_real, ax_syn) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    _draw_panel(ax_real, real, f"Real — {date_str}", style)
    ax_real.set_ylim(y_min, y_max)
    ax_real.set_facecolor("#f5f5f5")

    if gen is not None:
        _draw_panel(ax_syn, gen, f"{model_name}\n({draw_label})", style)
    else:
        ax_syn.text(0.5, 0.5, f"Generation failed\n{model_name}",
                    ha="center", va="center", transform=ax_syn.transAxes,
                    color="red", fontsize=11)
        ax_syn.set_title(model_name, fontsize=10)
        ax_syn.axis("off")
    ax_syn.set_ylim(y_min, y_max)

    return fig


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------

def _save_grid(
    real: np.ndarray,
    model_results: list[dict],
    draw_key: str,
    style: PlotStyle,
    meter_id: int,
    date_str: str,
    output_path: str,
) -> None:
    """2×4 landscape grid with shared y-axis across all 8 panels."""
    all_series: list[np.ndarray | None] = [real, *[r[draw_key] for r in model_results]]
    y_min, y_max = _compute_ylim(all_series)

    fig, axes = plt.subplots(2, 4, figsize=(24, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    _draw_panel(axes_flat[0], real, f"Real — {date_str}", style)
    axes_flat[0].set_ylim(y_min, y_max)
    axes_flat[0].set_facecolor("#f5f5f5")

    for idx, result in enumerate(model_results):
        ax = axes_flat[idx + 1]
        if result[draw_key] is not None:
            _draw_panel(ax, result[draw_key], result["name"], style)
        else:
            ax.text(0.5, 0.5, f"Error\n{result['name']}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=9)
            ax.set_title(result["name"], fontsize=10)
            ax.axis("off")
        ax.set_ylim(y_min, y_max)

    draw_label = f"Mean of {N_MEAN_DRAWS} Draws" if draw_key == "mean" else "Single Draw"
    fig.suptitle(
        f"3-Phase Synthetic Comparison — {draw_label} — {date_str}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def _save_appendix(
    real: np.ndarray,
    model_results: list[dict],
    draw_key: str,
    style: PlotStyle,
    meter_id: int,
    date_str: str,
    output_path: str,
) -> None:
    """Multi-page PDF: one page per model, real+synthetic side-by-side."""
    with PdfPages(output_path) as pdf:
        for result in model_results:
            fig = _make_side_by_side(real, result[draw_key], result["name"],
                                     draw_key, style, meter_id, date_str)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"  Saved {output_path}")


def _save_standalones(
    real: np.ndarray,
    model_results: list[dict],
    draw_key: str,
    style: PlotStyle,
    meter_id: int,
    date_str: str,
) -> None:
    """One PDF file per model."""
    for result in model_results:
        safe_name = result["name"].replace(" ", "_").replace("/", "-").replace("+", "plus")
        out = os.path.join(STANDALONE_DIR, f"{safe_name}_{draw_key}.pdf")
        fig = _make_side_by_side(real, result[draw_key], result["name"],
                                  draw_key, style, meter_id, date_str)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="24h real vs. synthetic comparison for 7 phased models."
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to a phased parquet file (one meter per file).")
    parser.add_argument("--ev-path", default=None,
                        help="EV CSV path for conditioners that require it.")
    parser.add_argument("--seq-len", type=int, default=24)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    style = PlotStyle()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STANDALONE_DIR, exist_ok=True)

    # Verify aggregation divisors are consistent across all models
    divisors = {path: _get_aggregation_divisor(path) for path in MODELS}
    unique = set(divisors.values())
    if len(unique) > 1:
        print(f"WARNING: inconsistent aggregation_divisors across models: {divisors}")
        print("Using divisor from first model for real data loading.")
    else:
        print(f"aggregation_divisor={next(iter(unique))} (consistent across all models)")
    aggregation_divisor = divisors[MODELS[0]]

    print(f"device={device}\n")

    # Load a random real window
    print(f"Loading random real window from: {args.data_path}")
    real_raw, meter_id, timestamp_unix = _load_real_window(
        args.data_path, args.seq_len, aggregation_divisor
    )
    real_sorted = _sort_phases(real_raw)
    timestamp_dt = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
    date_str = str(pd.to_datetime(timestamp_unix, unit="s").date())
    print(f"  date={date_str}  shape={real_sorted.shape}\n")

    # Generate for all models
    model_results: list[dict] = []
    for model_path, sec_path, name in zip(MODELS, SECONDARY_MODELS, NAMES):
        print(f"[{name}]")
        result: dict = {"name": name, "mean": None, "single": None,
                        "t_mean_s": None, "t_single_s": None}
        try:
            model, normalizer, conditioner = load_model(
                model_path, device=device, ev_path=args.ev_path
            )
            sec_model = None
            if sec_path is not None:
                print(f"  loading secondary model: {sec_path}")
                sec_model_obj, *_ = load_model(sec_path, device=device)
                sec_model = sec_model_obj

            generator = SyntheticGenerator(
                model=model,
                conditioner=conditioner,
                normalizer=normalizer,
                secondary_model=sec_model,
                device=device,
            )

            # Mean of N_MEAN_DRAWS
            _t0 = time.perf_counter()
            samples_mean, _ = generator.generate(
                num_samples=N_MEAN_DRAWS,
                meter_id=meter_id,
                date=timestamp_dt,
                meter_ids=np.array([meter_id] * N_MEAN_DRAWS),
                timestamps=np.array([timestamp_unix] * N_MEAN_DRAWS),
            )
            result["t_mean_s"] = time.perf_counter() - _t0
            if samples_mean is not None:
                arr = samples_mean.detach().cpu().numpy().mean(axis=0)
                result["mean"] = _sort_phases(arr[:, None] if arr.ndim == 1 else arr)
                print(f"  mean-of-{N_MEAN_DRAWS}: shape={result['mean'].shape}  ({result['t_mean_s']:.2f}s)")
            else:
                print(f"  WARNING: generate() returned None for mean draws.")

            # Single draw
            _t1 = time.perf_counter()
            samples_single, _ = generator.generate(
                num_samples=1,
                meter_id=meter_id,
                date=timestamp_dt,
                meter_ids=np.array([meter_id]),
                timestamps=np.array([timestamp_unix]),
            )
            result["t_single_s"] = time.perf_counter() - _t1
            if samples_single is not None:
                arr = samples_single[0].detach().cpu().numpy()
                result["single"] = _sort_phases(arr[:, None] if arr.ndim == 1 else arr)
                print(f"  single draw:          shape={result['single'].shape}  ({result['t_single_s']:.2f}s)")
            else:
                print(f"  WARNING: generate() returned None for single draw.")

        except Exception:
            traceback.print_exc()
            print(f"  ERROR: generation failed for {name} — see traceback above.")

        model_results.append(result)
        print()

    # Save timing report
    timing_path = os.path.join(OUTPUT_DIR, "generation_timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Generation timing — date={date_str}\n")
        f.write(f"N_MEAN_DRAWS={N_MEAN_DRAWS}  device={device}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<45} {'Mean 12 draws':>14} {'Single draw':>12}\n")
        f.write("-" * 60 + "\n")
        for r in model_results:
            t_mean  = f"{r['t_mean_s']:.3f}s"  if r["t_mean_s"]  is not None else "ERROR"
            t_single = f"{r['t_single_s']:.3f}s" if r["t_single_s"] is not None else "ERROR"
            f.write(f"{r['name']:<45} {t_mean:>14} {t_single:>12}\n")
        f.write("-" * 60 + "\n")
        valid_mean   = [r["t_mean_s"]   for r in model_results if r["t_mean_s"]   is not None]
        valid_single = [r["t_single_s"] for r in model_results if r["t_single_s"] is not None]
        if valid_mean:
            f.write(f"{'Per-draw (mean / N_MEAN_DRAWS)':<45} "
                    f"{sum(valid_mean)/len(valid_mean)/N_MEAN_DRAWS*1000:>11.1f}ms {'':>12}\n")
        if valid_single:
            f.write(f"{'Per-draw (single)':<45} {'':>14} "
                    f"{sum(valid_single)/len(valid_single)*1000:>9.1f}ms\n")
    print(f"  Saved {timing_path}")

    # Save all figures
    print("Saving grid figures...")
    for draw_key in ("mean", "single"):
        _save_grid(
            real=real_sorted, model_results=model_results, draw_key=draw_key,
            style=style, meter_id=meter_id, date_str=date_str,
            output_path=os.path.join(OUTPUT_DIR, f"grid_{draw_key}.pdf"),
        )

    print("\nSaving appendix PDFs...")
    for draw_key in ("mean", "single"):
        _save_appendix(
            real=real_sorted, model_results=model_results, draw_key=draw_key,
            style=style, meter_id=meter_id, date_str=date_str,
            output_path=os.path.join(OUTPUT_DIR, f"appendix_{draw_key}.pdf"),
        )

    print("\nSaving standalone PDFs...")
    for draw_key in ("mean", "single"):
        _save_standalones(
            real=real_sorted, model_results=model_results, draw_key=draw_key,
            style=style, meter_id=meter_id, date_str=date_str,
        )

    print(f"\nDone. All figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()