"""Standalone diagnostic: Diffusion advantage vs feeder size and EV penetration.

For each holdout feeder, computes:
    advantage = competitor_metric - diffusion_metric
(lower-is-better metrics; positive = Diffusion wins). Produces two PDF figures:
  feeder_size_vs_advantage.pdf  — advantage vs n_meters
  feeder_ev_vs_advantage.pdf    — advantage vs EV penetration (%)
Both figures have one subplot per metric and one regression line per competitor.
A companion JSON is written alongside the first PDF.

Run from repo root:
    python src/evaluation/plot_feeder_size_advantage.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

EVAL_DIR      = ROOT / "evaluation_comparison"
TOPOLOGY_PATH = ROOT / "data/topology/cleaned/Cleaned_SM_Topology_mapping.json"
EV_CSV_PATH   = ROOT / "data/EV/EV_detection_reduced_60000_SM.csv"
OUTPUT_SIZE   = EVAL_DIR / "feeder_size_vs_advantage.pdf"
OUTPUT_EV     = EVAL_DIR / "feeder_ev_vs_advantage.pdf"
OUTPUT_JSON   = EVAL_DIR / "feeder_size_vs_advantage.json"

DIFFUSION_NAME = "Diffusion"
COMPETITORS = [
    "Guided-Faraday-CNN",
    "Guided-Faraday-CNN-UserProfiles",
    "H-Guide-CNN",
    "Faraday",
    "Guide",
    "CVAE",
    "Baseline-Quantile",
    "Baseline-Average",
]

# (metric_key, short_label, advantage_mode)
# mode "lower": advantage = comp - diffusion  (positive = diffusion wins)
# mode "target1": advantage = |comp-1| - |diffusion-1|  (positive = diffusion closer to 1)
METRICS: list[tuple[str, str, str]] = [
    ("Aggregated Profiles (Flexibility Distance)", "Agg. Profiles (FD)",       "lower"),
    ("Peak Magnitude",                             "Peak Magnitude",            "lower"),
    ("Maximum Mean Discrepancy (MMD)",             "MMD",                       "lower"),
    ("Classical Statistics",                       "Classical Stats",           "lower"),
    ("ACF Distribution Match (KS)",                "ACF KS",                   "lower"),
    ("CF_area_independent",                        "CF area (indep.)",          "lower"),
    ("Flexibility Distance",                       "Flexibility Distance",      "lower"),
    ("TSTR (Train on Synthetic, Test on Real)",    "TSTR (target=1)",           "target1"),
]


def load_summary(model_name: str) -> dict:
    path = EVAL_DIR / model_name / "eval_results_feeder_holdout_summary.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def per_feeder_values(summary: dict, metric_key: str) -> dict[str, float]:
    entry = summary.get(metric_key)
    if entry is None:
        raise KeyError(f"Metric '{metric_key}' missing in summary")
    return {str(k): float(v) for k, v in entry["per_feeder"].items()}


def n_meters_per_feeder(feeder_ids: list[str]) -> dict[str, int]:
    with open(TOPOLOGY_PATH, encoding="utf-8") as fh:
        topo = json.load(fh)
    counts: dict[str, int] = {fid: 0 for fid in feeder_ids}
    for entry in topo.values():
        fid = str(entry.get("Feeder ID", ""))
        if fid in counts:
            counts[fid] += 1
    missing = [fid for fid, n in counts.items() if n == 0]
    if missing:
        print(f"  WARNING: feeder IDs not found in topology: {missing}")
    return counts


def ev_pct_per_feeder(feeder_ids: list[str]) -> dict[str, float]:
    """Return EV penetration fraction per feeder using EV detection CSV + topology."""
    with open(TOPOLOGY_PATH, encoding="utf-8") as fh:
        topo = json.load(fh)

    df_ev = pd.read_csv(EV_CSV_PATH)
    ev_flags: dict[str, bool] = {}
    for _, row in df_ev.iterrows():
        meter_id = re.sub(r"[^\d]", "", str(row["SM_ID"]))
        ev_flags[meter_id] = str(row["EV_detected"]).strip().lower() == "yes"

    ev_count: dict[str, int] = {fid: 0 for fid in feeder_ids}
    total:    dict[str, int] = {fid: 0 for fid in feeder_ids}
    for meter_id, entry in topo.items():
        fid = str(entry.get("Feeder ID", ""))
        if fid not in ev_count:
            continue
        total[fid] += 1
        if ev_flags.get(meter_id, False):
            ev_count[fid] += 1

    result: dict[str, float] = {}
    for fid in feeder_ids:
        n = total[fid]
        result[fid] = ev_count[fid] / n if n > 0 else float("nan")
        if n == 0:
            print(f"  WARNING: feeder {fid} has no meters in topology for EV lookup.")
    return result


def regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, pearson_r)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    xd, yd = x - xm, y - ym
    denom_x = float(np.dot(xd, xd))
    denom_y = float(np.dot(yd, yd))
    if denom_x == 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.dot(xd, yd) / denom_x)
    intercept = float(ym - slope * xm)
    r = float(np.dot(xd, yd) / np.sqrt(denom_x * denom_y)) if denom_y > 0 else float("nan")
    return slope, intercept, r


def _advantage(diffusion_val: float, comp_val: float, mode: str) -> float:
    if mode == "lower":
        return comp_val - diffusion_val
    if mode == "target1":
        return abs(comp_val - 1.0) - abs(diffusion_val - 1.0)
    raise ValueError(f"Unknown advantage mode: {mode!r}")


def _draw_advantage_figure(
    feeder_ids: list[str],
    x_vals: dict[str, float],
    x_label: str,
    title: str,
    diff_vals_by_metric: dict,
    comp_summaries: dict,
    colors: list,
    output_path: Path,
    collect_results: dict | None = None,
) -> None:
    n_metrics = len(METRICS)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes_flat = axes.flatten()

    for ax_idx, (metric_key, short_label, mode) in enumerate(METRICS):
        ax = axes_flat[ax_idx]
        diff_vals = diff_vals_by_metric.get(metric_key)
        if diff_vals is None:
            ax.set_visible(False)
            continue

        if collect_results is not None and metric_key not in collect_results:
            collect_results[metric_key] = {"mode": mode, "competitors": {}}

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        for color, name in zip(colors, COMPETITORS):
            summary = comp_summaries.get(name)
            if summary is None:
                continue
            try:
                comp_vals = per_feeder_values(summary, metric_key)
            except KeyError as exc:
                print(f"  SKIP {name}/{short_label}: {exc}")
                continue

            common = [fid for fid in feeder_ids if fid in comp_vals and fid in x_vals]
            if len(common) < 3:
                continue

            x_sub = np.array([x_vals[fid] for fid in common], dtype=float)
            y = np.array([_advantage(diff_vals[fid], comp_vals[fid], mode) for fid in common], dtype=float)

            slope, intercept, r = regression(x_sub, y)
            x_line = np.array([x_sub.min(), x_sub.max()])
            y_line = slope * x_line + intercept

            if collect_results is not None and name not in collect_results[metric_key]["competitors"]:
                collect_results[metric_key]["competitors"][name] = {
                    "pearson_r": round(r, 4),
                    "slope": round(slope, 6),
                    "n_feeders": len(common),
                    "advantage_mean": round(float(y.mean()), 6),
                    "advantage_positive_frac": round(float((y > 0).mean()), 4),
                    "per_feeder": {fid: round(float(adv), 6) for fid, adv in zip(common, y.tolist())},
                }

            n_label = f",n={len(common)}" if len(common) < len(feeder_ids) else ""
            label = f"{name} (r={r:+.2f}{n_label})"
            ax.scatter(x_sub, y, color=color, s=25, alpha=0.6, zorder=3)
            ax.plot(x_line, y_line, color=color, linewidth=1.2, label=label)

        ax.set_title(short_label, fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Comp − Diffusion\n(↑ = Diffusion wins)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best", framealpha=0.85)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    diff_summary = load_summary(DIFFUSION_NAME)
    diff_vals_by_metric: dict = {}
    for metric_key, _, _ in METRICS:
        try:
            diff_vals_by_metric[metric_key] = per_feeder_values(diff_summary, metric_key)
        except KeyError as exc:
            print(f"  WARNING Diffusion: {exc}")

    feeder_ids = sorted(next(iter(diff_vals_by_metric.values())).keys())

    print("Computing feeder sizes...")
    n_meters = n_meters_per_feeder(feeder_ids)
    print("Computing EV penetration...")
    ev_pct = ev_pct_per_feeder(feeder_ids)

    comp_summaries: dict[str, dict] = {}
    for name in COMPETITORS:
        try:
            comp_summaries[name] = load_summary(name)
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}")

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i / len(COMPETITORS)) for i in range(len(COMPETITORS))]

    results: dict = {}

    _draw_advantage_figure(
        feeder_ids=feeder_ids,
        x_vals=n_meters,
        x_label="n meters",
        title="Diffusion advantage vs feeder size — 20 holdout feeders, per metric",
        diff_vals_by_metric=diff_vals_by_metric,
        comp_summaries=comp_summaries,
        colors=colors,
        output_path=OUTPUT_SIZE,
        collect_results=results,
    )

    ev_pct_pct = {fid: v * 100 for fid, v in ev_pct.items()}
    _draw_advantage_figure(
        feeder_ids=feeder_ids,
        x_vals=ev_pct_pct,
        x_label="EV penetration (%)",
        title="Diffusion advantage vs EV penetration — 20 holdout feeders, per metric",
        diff_vals_by_metric=diff_vals_by_metric,
        comp_summaries=comp_summaries,
        colors=colors,
        output_path=OUTPUT_EV,
        collect_results=None,
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()