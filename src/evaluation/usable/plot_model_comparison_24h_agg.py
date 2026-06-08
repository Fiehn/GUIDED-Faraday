"""
24-hour real vs. synthetic comparison for the 9 aggregated (single-phase) new4 models.

Real data comes from the RealDataBaselineModel reservoir — no data file required.

Outputs (all in evaluation_comparison/1-generated/):
  grid_mean.pdf          — 2×5 landscape grid, mean of N_MEAN_DRAWS draws
  grid_single.pdf        — 2×5 landscape grid, single draw
  appendix_mean.pdf      — multi-page, one page per model, real+mean side-by-side
  appendix_single.pdf    — multi-page, one page per model, real+single side-by-side
  standalone/
    {name}_mean.pdf      — real+mean side-by-side  (×9)
    {name}_single.pdf    — real+single side-by-side (×9)
  generation_timing.txt  — wall-clock time per model

Usage:
    uv run src/evaluation/usable/plot_model_comparison_24h_agg.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
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
from src.model.generators import SyntheticGenerator
from src.visualization.plot_styling import PlotStyle


OUTPUT_DIR = os.path.join("evaluation_comparison", "1-generated")
STANDALONE_DIR = os.path.join(OUTPUT_DIR, "standalone")
N_MEAN_DRAWS = 12

REAL_DATA_MODEL = "E:/models/new4/RealDataBaselineModel_20260608_143634"

MODELS = [
    "E:/models/new4/GuidedFaradayModel_20260605_105946",
    "E:/models/new4/GuidedFaradayModel_20260605_201145",
    "E:/models/new4/CVAE_20260531_185015_guide_cnn_h_broken",
    "E:/models/new4/DiffusionModel_20260601_002842",
    "E:/models/new4/FaradayModel_20260601_060319",
    "E:/models/new4/CVAE_20260601_022452_guide",
    "E:/models/new4/CVAE_20260531_180003_cvae",
    "E:/models/new4/QuantileBaselineModel_20260601_005326",
    "E:/models/new4/AverageBaselineModel_20260531_173051",
]
NAMES = [
    "Guided-Faraday-CNN",
    "Guided-Faraday-CNN-UserProfiles",
    "H-Guide-CNN",
    "Diffusion",
    "Faraday",
    "Guide",
    "CVAE",
    "Baseline-Quantile",
    "Baseline-Average",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_real_windows(device: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw n random windows from the RealDataBaselineModel reservoir.

    Returns (windows [n, seq_len, 1], mean_window [seq_len, 1]).
    """
    print(f"Loading real windows from RealDataBaselineModel reservoir...")
    model, normalizer, conditioner = load_model(REAL_DATA_MODEL, device=device)
    generator = SyntheticGenerator(
        model=model,
        conditioner=conditioner,
        normalizer=normalizer,
        device=device,
    )
    # Pass random conditions so the reservoir bucket lookup is valid;
    # RealDataBaselineModel.sample() ignores conditions when num_buckets=0,
    # but this is safe either way.
    conds = conditioner.get_random_conditions(n).to(device) if conditioner else None
    samples, _ = generator.generate(
        num_samples=n,
        conditions=conds,
    )
    if samples is None:
        raise RuntimeError("RealDataBaselineModel.generate() returned None — reservoir may be empty.")
    windows = samples.detach().cpu().numpy()           # [n, seq_len, 1]
    mean_window = windows.mean(axis=0)                 # [seq_len, 1]
    print(f"  loaded {n} real windows  shape={windows.shape}")
    return windows, mean_window


# ---------------------------------------------------------------------------
# Y-axis helpers
# ---------------------------------------------------------------------------

def _compute_ylim(series_list: list[np.ndarray | None], pad: float = 0.05) -> tuple[float, float]:
    all_vals: list[float] = []
    for s in series_list:
        if s is None:
            continue
        all_vals.extend(s.flatten().tolist())
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
    """Draw a single aggregated 24h profile. Caller sets ylim."""
    hours = np.arange(series.shape[0])
    vals = series[:, 0] if series.ndim > 1 else series
    ax.plot(hours, vals, color=style.color_original, linewidth=1.8)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Hour", fontsize=8)
    ax.set_ylabel("Load (kWh)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)


def _make_side_by_side(
    real: np.ndarray,
    gen: np.ndarray | None,
    model_name: str,
    draw_key: str,
    style: PlotStyle,
) -> matplotlib.figure.Figure:
    """Create a real (left) + synthetic (right) figure with shared y-axis."""
    y_min, y_max = _compute_ylim([real, gen])
    draw_label = f"Mean of {N_MEAN_DRAWS} Draws" if draw_key == "mean" else "Single Draw"

    fig, (ax_real, ax_syn) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    real_title = f"Real (mean of {N_MEAN_DRAWS} draws)" if draw_key == "mean" else "Real (single draw)"
    _draw_panel(ax_real, real, real_title, style)
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
    output_path: str,
) -> None:
    """2×5 landscape grid with shared y-axis across all 10 panels."""
    all_series: list[np.ndarray | None] = [real, *[r[draw_key] for r in model_results]]
    y_min, y_max = _compute_ylim(all_series)

    fig, axes = plt.subplots(2, 5, figsize=(30, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    real_title = f"Real (mean of {N_MEAN_DRAWS} draws)" if draw_key == "mean" else "Real (single draw)"
    _draw_panel(axes_flat[0], real, real_title, style)
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
    fig.suptitle(f"Aggregated Synthetic Comparison — {draw_label}", fontsize=13)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def _save_appendix(
    real: np.ndarray,
    model_results: list[dict],
    draw_key: str,
    style: PlotStyle,
    output_path: str,
) -> None:
    """Multi-page PDF: one page per model, real+synthetic side-by-side."""
    with PdfPages(output_path) as pdf:
        for result in model_results:
            fig = _make_side_by_side(real, result[draw_key], result["name"], draw_key, style)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"  Saved {output_path}")


def _save_standalones(
    real: np.ndarray,
    model_results: list[dict],
    draw_key: str,
    style: PlotStyle,
) -> None:
    """One PDF file per model."""
    for result in model_results:
        safe_name = result["name"].replace(" ", "_").replace("/", "-").replace("+", "plus")
        out = os.path.join(STANDALONE_DIR, f"{safe_name}_{draw_key}.pdf")
        fig = _make_side_by_side(real, result[draw_key], result["name"], draw_key, style)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style = PlotStyle()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STANDALONE_DIR, exist_ok=True)

    print(f"device={device}\n")

    # Real data from RealDataBaselineModel reservoir
    real_windows, real_mean_raw = _load_real_windows(device, n=N_MEAN_DRAWS)
    real_single = real_windows[0]   # [seq_len, 1]
    real_mean   = real_mean_raw     # [seq_len, 1]
    print()

    # Generate for all 9 models
    model_results: list[dict] = []
    for model_path, name in zip(MODELS, NAMES):
        print(f"[{name}]")
        result: dict = {"name": name, "mean": None, "single": None,
                        "t_mean_s": None, "t_single_s": None}
        try:
            model, normalizer, conditioner = load_model(model_path, device=device)
            generator = SyntheticGenerator(
                model=model,
                conditioner=conditioner,
                normalizer=normalizer,
                device=device,
            )

            # Pre-build random conditions so each sample gets an independent draw
            # (passing conditions=None would tile a single random condition across all samples)
            conds_mean   = conditioner.get_random_conditions(N_MEAN_DRAWS).to(device) if conditioner else None
            conds_single = conditioner.get_random_conditions(1).to(device)            if conditioner else None

            # Mean of N_MEAN_DRAWS
            _t0 = time.perf_counter()
            samples_mean, _ = generator.generate(
                num_samples=N_MEAN_DRAWS,
                conditions=conds_mean,
            )
            result["t_mean_s"] = time.perf_counter() - _t0
            if samples_mean is not None:
                arr = samples_mean.detach().cpu().numpy().mean(axis=0)  # [seq_len, 1]
                result["mean"] = arr[:, None] if arr.ndim == 1 else arr
                print(f"  mean-of-{N_MEAN_DRAWS}: shape={result['mean'].shape}  ({result['t_mean_s']:.2f}s)")
            else:
                print(f"  WARNING: generate() returned None for mean draws.")

            # Single draw
            _t1 = time.perf_counter()
            samples_single, _ = generator.generate(
                num_samples=1,
                conditions=conds_single,
            )
            result["t_single_s"] = time.perf_counter() - _t1
            if samples_single is not None:
                arr = samples_single[0].detach().cpu().numpy()  # [seq_len, 1]
                result["single"] = arr[:, None] if arr.ndim == 1 else arr
                print(f"  single draw:          shape={result['single'].shape}  ({result['t_single_s']:.2f}s)")
            else:
                print(f"  WARNING: generate() returned None for single draw.")

        except Exception:
            traceback.print_exc()
            print(f"  ERROR: generation failed for {name} — see traceback above.")

        model_results.append(result)
        print()

    # Timing report
    timing_path = os.path.join(OUTPUT_DIR, "generation_timing.txt")
    with open(timing_path, "w") as f:
        f.write(f"Generation timing  device={device}  N_MEAN_DRAWS={N_MEAN_DRAWS}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<45} {'Mean 12 draws':>14} {'Single draw':>12}\n")
        f.write("-" * 60 + "\n")
        for r in model_results:
            t_mean   = f"{r['t_mean_s']:.3f}s"  if r["t_mean_s"]   is not None else "ERROR"
            t_single = f"{r['t_single_s']:.3f}s" if r["t_single_s"] is not None else "ERROR"
            f.write(f"{r['name']:<45} {t_mean:>14} {t_single:>12}\n")
        f.write("-" * 60 + "\n")
        valid_mean   = [r["t_mean_s"]   for r in model_results if r["t_mean_s"]   is not None]
        valid_single = [r["t_single_s"] for r in model_results if r["t_single_s"] is not None]
        if valid_mean:
            f.write(f"{'Per-draw (mean / N_MEAN_DRAWS)':<45} "
                    f"{sum(valid_mean)/len(valid_mean)/N_MEAN_DRAWS*1000:>11.1f}ms\n")
        if valid_single:
            f.write(f"{'Per-draw (single)':<45} {'':>14} "
                    f"{sum(valid_single)/len(valid_single)*1000:>9.1f}ms\n")
    print(f"  Saved {timing_path}")

    real_by_key = {"mean": real_mean, "single": real_single}

    print("\nSaving grid figures...")
    for draw_key in ("mean", "single"):
        _save_grid(
            real=real_by_key[draw_key], model_results=model_results, draw_key=draw_key,
            style=style,
            output_path=os.path.join(OUTPUT_DIR, f"grid_{draw_key}.pdf"),
        )

    print("\nSaving appendix PDFs...")
    for draw_key in ("mean", "single"):
        _save_appendix(
            real=real_by_key[draw_key], model_results=model_results, draw_key=draw_key,
            style=style,
            output_path=os.path.join(OUTPUT_DIR, f"appendix_{draw_key}.pdf"),
        )

    print("\nSaving standalone PDFs...")
    for draw_key in ("mean", "single"):
        _save_standalones(
            real=real_by_key[draw_key], model_results=model_results, draw_key=draw_key,
            style=style,
        )

    print(f"\nDone. All figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()