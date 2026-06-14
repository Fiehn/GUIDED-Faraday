"""
Compare 3-phase aggregated test windows against hourly parquet ground truth.

For each sampled test window, plots three signals:
  1. Raw (loader)             — sum of phases directly from the test loader (may be normalised)
  2. Inv-transform (framework)— after _inverse_transform_for_metrics with the model normalizer
  3. Hourly parquet (GT)      — hourly_energy_consumption from data/_parquet_cache

This lets you diagnose whether the "consistently higher real aggregated data" symptom
is caused by a broken inverse-transform path or a unit mismatch in the raw data.

Usage:
    python src/evaluation/scripts/compare_phased_vs_hourly.py <model_path>
    python src/evaluation/scripts/compare_phased_vs_hourly.py <model_path> --n-windows 200
    python src/evaluation/scripts/compare_phased_vs_hourly.py <model_path> --data-dir data/_parquet_cache
"""

from pathlib import Path
import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
ROOT_DIR = SRC_DIR.parent

for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluation.load_model import load_model
from src.evaluation.load_and_eval_model import _load_config_for_model, setup_data_loader


# ---------------------------------------------------------------------------
# Inverse-transform helper (replicates EvalFramework._inverse_transform_for_metrics
# for a single [F, L] window tensor using the model's own normalizer)
# ---------------------------------------------------------------------------

def _inv_transform(x: torch.Tensor, normalizer) -> torch.Tensor:
    """Apply normalizer.inverse_transform to a [F, L] window, matching the
    axis-detection logic used by EvalFramework._inverse_transform_for_metrics."""
    if normalizer is None or not hasattr(normalizer, "inverse_transform"):
        return x

    inv = normalizer.inverse_transform
    stats = getattr(normalizer, "mean", None)
    feat_dim = None
    if stats is not None:
        try:
            feat_dim = int(np.asarray(stats).reshape(-1).shape[0])
        except Exception:
            pass

    x_b = x.unsqueeze(0)  # [1, F, L]

    if feat_dim is not None and x_b.shape[1] == feat_dim:
        # [B, F, L] — features in dim-1 (phase-first layout)
        x_last = x_b.movedim(1, -1).contiguous()           # [1, L, F]
        y_last = inv(x_last.reshape(-1, feat_dim)).reshape_as(x_last)
        return y_last.movedim(-1, 1).contiguous().squeeze(0)

    if feat_dim is not None and x_b.shape[-1] == feat_dim:
        # [B, L, F] — features in last dim
        return inv(x_b.reshape(-1, feat_dim)).reshape_as(x_b).squeeze(0)

    # Fallback — let the normalizer attempt the transform as-is
    try:
        return inv(x_b).squeeze(0)
    except Exception:
        return x


# ---------------------------------------------------------------------------
# Hourly parquet helpers
# ---------------------------------------------------------------------------

def _load_hourly_parquet(data_dir: str, meter_ids: set) -> pd.DataFrame:
    """Load all monthly hourly parquet files filtered to the given meter IDs."""
    parquet_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
        and not f.startswith("test")
        and not f.startswith("normalizer")
    ])
    if not parquet_files:
        raise FileNotFoundError(f"No hourly parquet files found in {data_dir}")

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, filters=[("meter_number", "in", list(meter_ids))])
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {pf}: {e}")

    if not dfs:
        return pd.DataFrame(
            columns=["hourly_energy_consumption"],
            index=pd.MultiIndex.from_tuples([], names=["meter_number", "timestamp_dst"]),
        )

    return pd.concat(dfs).sort_index()


def _lookup_window(hourly_df: pd.DataFrame, meter_id: int, start_ts_unix: int, window_len: int):
    """Return hourly_energy_consumption array for the window, or None if unavailable.

    start_ts_unix is a real-UTC Unix epoch (seconds).

    The GT parquet's labeling convention differs by DST period (empirically verified):
      - CEST months (UTC+2): GT uses end-of-hour labeling.  GT[H] = energy (H-1h, H] UTC.
        Correct lookup requires ts_range shifted +1h: (Copenhagen UTC offset - 1) = 1h.
      - CET months  (UTC+1): GT uses start-of-hour labeling.  GT[H] = energy [H, H+1) UTC.
        Correct lookup requires ts_range shifted +0h: (Copenhagen UTC offset - 1) = 0h.

    The formula `gt_offset = Copenhagen_UTC_offset_hours - 1` unifies both cases.
    """
    ts_utc = pd.Timestamp(start_ts_unix, unit="s", tz="UTC")
    _utcoffset = ts_utc.tz_convert("Europe/Copenhagen").utcoffset()
    cph_offset_h = int(_utcoffset.total_seconds() // 3600) if _utcoffset is not None else 1
    gt_offset_h = cph_offset_h - 1  # 1 for CEST, 0 for CET
    ts_start = ts_utc.tz_localize(None) + pd.Timedelta(hours=gt_offset_h)
    ts_range = pd.date_range(start=ts_start, periods=window_len, freq="1h")

    meter_ids_in_index = hourly_df.index.get_level_values("meter_number")
    if meter_id not in meter_ids_in_index:
        return None

    meter_series = pd.Series(hourly_df.loc[meter_id, "hourly_energy_consumption"])

    # Strip the UTC timezone label so the tz-naive ts_range can reindex it.
    # Both parquets use real UTC timestamps — tz_localize(None) just drops the label.
    if getattr(meter_series.index, "tz", None) is not None:
        meter_series = meter_series.copy()
        meter_series.index = pd.DatetimeIndex(meter_series.index).tz_localize(None)

    # Deduplicate: DST fall-back can produce duplicate local timestamps.
    meter_series = meter_series[~meter_series.index.duplicated(keep="last")]

    values = meter_series.reindex(ts_range).values.astype(np.float32)

    if np.isnan(values).mean() > 0.5:
        return None  # more than half the window is missing
    return values


# ---------------------------------------------------------------------------
# Window collection
# ---------------------------------------------------------------------------

def collect_windows(test_loader, normalizer, n_windows: int, device: str) -> list:
    """Collect up to n_windows from the test loader.

    Each entry is a dict:
        meter_id         int
        start_ts         int      Unix seconds (UTC) for the first hour of the window
        window_len       int      number of hourly steps
        normalized_sum   ndarray [L]  — sum of phases directly from loader (normalised scale)
        denormalized_sum ndarray [L]  — sum of phases after inv_transform (physical kWh)
        parquet_values   ndarray [L] | None
    """
    windows = []
    with torch.no_grad():
        for batch in test_loader:
            if len(windows) >= n_windows:
                break

            x, meter_ids, timestamps, _agg = batch
            x = x.to(device)

            for i in range(x.shape[0]):
                if len(windows) >= n_windows:
                    break

                xi = x[i].cpu()  # [F, L]
                meter_id = int(meter_ids[i].item() if torch.is_tensor(meter_ids) else meter_ids[i])
                start_ts = int(timestamps[i].item() if torch.is_tensor(timestamps) else timestamps[i])

                normalized_sum = xi.sum(dim=0).numpy()
                denormalized_sum = _inv_transform(xi, normalizer).sum(dim=0).cpu().numpy()

                windows.append(
                    dict(
                        meter_id=meter_id,
                        start_ts=start_ts,
                        window_len=int(xi.shape[-1]),
                        normalized_sum=normalized_sum,
                        denormalized_sum=denormalized_sum,
                        parquet_values=None,
                    )
                )

    return windows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_N_INDIVIDUAL = 10   # individual window plots
_COLS = 2


def plot_individual(windows: list, output_dir: str) -> None:
    """Plot up to _N_INDIVIDUAL windows, preferring those with parquet matches."""
    os.makedirs(output_dir, exist_ok=True)

    matched = [w for w in windows if w["parquet_values"] is not None]
    unmatched = [w for w in windows if w["parquet_values"] is None]
    candidates = matched + unmatched

    rng = np.random.default_rng(0)
    if len(candidates) > _N_INDIVIDUAL:
        idx = rng.choice(len(candidates), _N_INDIVIDUAL, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]

    n = len(candidates)
    n_rows = (n + _COLS - 1) // _COLS
    fig, axes = plt.subplots(n_rows, _COLS, figsize=(14, 3.5 * n_rows), squeeze=False)

    for idx, w in enumerate(candidates):
        ax = axes[idx // _COLS][idx % _COLS]
        hours = np.arange(w["window_len"])

        ax.plot(hours, w["normalized_sum"], label="Normalised (loader)",
                color="steelblue", alpha=0.8, linewidth=1.5)
        ax.plot(hours, w["denormalized_sum"], label="Denormalised (kWh)",
                color="tomato", alpha=0.8, linewidth=1.5)
        if w["parquet_values"] is not None:
            pv = w["parquet_values"].copy().astype(float)
            pv[np.isnan(pv)] = np.nan
            ax.plot(hours, pv, label="Parquet GT (kWh)",
                    color="seagreen", linewidth=2.0, linestyle="--", alpha=0.9)

        ts_str = pd.Timestamp(w["start_ts"], unit="s", tz="UTC").strftime("%Y-%m-%d")
        ax.set_title(f"Meter {w['meter_id']}  |  {ts_str}", fontsize=9)
        ax.set_xlabel("Hour", fontsize=8)
        ax.set_ylabel("kWh", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, n_rows * _COLS):
        axes[idx // _COLS][idx % _COLS].set_visible(False)

    fig.suptitle("Individual windows — 3-phase sum vs hourly parquet", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "individual_windows.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_aggregated(windows: list, output_dir: str) -> None:
    """Plot mean ± 10th/90th percentile across all matched windows."""
    os.makedirs(output_dir, exist_ok=True)

    matched = [w for w in windows if w["parquet_values"] is not None]
    if not matched:
        print("  [WARN] No matched windows — skipping aggregated plot.")
        return

    L = matched[0]["window_len"]
    norm_arr  = np.stack([w["normalized_sum"]  for w in matched])   # [N, L]
    denorm_arr = np.stack([w["denormalized_sum"] for w in matched])  # [N, L]
    parquet_arr = np.stack([
        np.where(np.isnan(w["parquet_values"]), np.nan, w["parquet_values"])
        for w in matched
    ])  # [N, L]

    hours = np.arange(L)
    _, ax = plt.subplots(figsize=(12, 5))

    def _plot_band(arr, color, label):
        mean = np.nanmean(arr, axis=0)
        p10  = np.nanpercentile(arr, 10, axis=0)
        p90  = np.nanpercentile(arr, 90, axis=0)
        ax.plot(hours, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(hours, p10, p90, color=color, alpha=0.15)

    _plot_band(norm_arr,   "steelblue", f"Normalised (loader)  n={len(matched)}")
    _plot_band(denorm_arr, "tomato",    f"Denormalised (kWh)   n={len(matched)}")
    _plot_band(parquet_arr,"seagreen",  f"Parquet GT (kWh)     n={len(matched)}")

    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("kWh / normalised units", fontsize=11)
    ax.set_title("Aggregated mean ± p10/p90  —  3-phase sum vs hourly parquet", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "aggregated_comparison.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(windows: list) -> None:
    # Accumulate all values across matched windows for a global (aggregate) ratio.
    # Per-window ratios (mean(denorm_i) / mean(gt_i)) are highly sensitive to windows
    # where GT ≈ 0 (meter offline, all zeros), causing the mean of those ratios to blow
    # up by many orders of magnitude.  The global ratio is numerically stable.
    all_gt, all_denorm = [], []
    per_window_denorm_mae = []
    per_window_denorm_ratio = []  # for median — robust to outlier windows

    for w in windows:
        pv = w["parquet_values"]
        if pv is None:
            continue
        mask = ~np.isnan(pv)
        if mask.sum() == 0:
            continue

        gt = pv[mask]
        denorm = w["denormalized_sum"][mask]

        all_gt.append(gt)
        all_denorm.append(denorm)
        per_window_denorm_mae.append(float(np.mean(np.abs(denorm - gt))))

        gt_win_mean = float(np.mean(np.abs(gt)))
        if gt_win_mean > 1e-6:  # skip near-zero GT windows for per-window ratio
            per_window_denorm_ratio.append(float(np.mean(denorm)) / gt_win_mean)

    n = len(per_window_denorm_mae)
    print(f"\nStatistics over {n} windows with parquet ground truth:")
    if n == 0:
        print("  No matched windows — check that meter IDs and dates overlap with the parquet cache.")
        return

    gt_concat   = np.concatenate(all_gt)
    denorm_concat = np.concatenate(all_denorm)

    global_gt_mean    = float(np.mean(np.abs(gt_concat))) + 1e-9
    global_denorm_mean = float(np.mean(denorm_concat))
    global_ratio = global_denorm_mean / global_gt_mean

    median_ratio = float(np.median(per_window_denorm_ratio)) if per_window_denorm_ratio else float("nan")

    print(f"  MAE  denormalised vs parquet:      {np.mean(per_window_denorm_mae):.4f} kWh/h")
    print(f"  Global scale  denorm / GT:         {global_ratio:.3f}x  (mean denorm / mean |GT|)")
    print(f"  Median per-window  denorm / GT:    {median_ratio:.3f}x  (robust, excludes ~0 GT windows)")
    print(f"  Global mean denorm:                {global_denorm_mean:.4f} kWh/h")
    print(f"  Global mean |GT|:                  {global_gt_mean:.4f} kWh/h")
    print()
    if abs(global_ratio - 1.0) < 0.3:
        print("  Diagnosis: denormalised values match parquet scale. Framework path is correct.")
    elif global_ratio > 1.3:
        print(f"  Diagnosis: denormalised values are {global_ratio:.1f}x LARGER than parquet GT.")
        if global_ratio > 5.0:
            print("  => Likely cause: normalizer was NOT applied during loading but inv_transform was")
            print("     applied anyway (e.g. normalizer=None in setup_data_loader, but normalizer")
            print("     passed to EvalFramework._inverse_transform_for_metrics).")
        else:
            print("  => Moderate over-scale: check aggregation_divisor / aggregation_energy_unit in config.")
    elif global_ratio < 0.7:
        print(f"  Diagnosis: denormalised values are {1/global_ratio:.1f}x SMALLER than parquet GT.")
        print("  => The raw phase columns may not cover 100% of household consumption (normal")
        print("     if measuring individual circuit phases rather than the whole-house meter).")


def print_mismatch_summary(windows: list, max_shift: int = 3, tail_len: int = 3,
                            extreme_sigma: float = 4.0, n_examples: int = 5) -> None:
    """Analyse two known anomaly patterns and print a summary.

    1. Shift pattern — for each window, compute MAE at shifts -max_shift..+max_shift and
       report which offset gives the best alignment. A systematic +1 or -1 mode points to
       a timestamp alignment bug.

    2. Tail extremes — flag windows whose last `tail_len` values of denormalized_sum deviate
       more than `extreme_sigma` standard deviations from the window body mean/std.
    """
    matched = [w for w in windows if w["parquet_values"] is not None]
    if not matched:
        print("\nNo matched windows for mismatch analysis.")
        return

    # ── 1. Shift analysis ────────────────────────────────────────────────────
    shift_counts: dict[int, int] = {}
    shift_mae_improvement: list[float] = []  # improvement % when using best shift

    for w in matched:
        gt = w["parquet_values"].astype(float)
        d  = w["denormalized_sum"].astype(float)
        L  = len(d)

        # MAE at shift=0 (no shift)
        mask0 = ~np.isnan(gt)
        if mask0.sum() < L // 2:
            continue
        mae0 = np.mean(np.abs(d[mask0] - gt[mask0]))

        best_shift, best_mae = 0, mae0
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            # Shift d only (keep gt fixed): test whether d[i] matches gt[i+s].
            # Positive s: d is s steps ahead of gt (d[0..L-s-1] vs gt[s..L-1]).
            # Negative s: d is |s| steps behind gt (d[|s|..L-1] vs gt[0..L-|s|-1]).
            if s > 0:
                d_part  = d[:L - s]
                gt_part = gt[s:]
            else:
                d_part  = d[-s:]
                gt_part = gt[:L + s]
            mask_s = ~(np.isnan(d_part) | np.isnan(gt_part))
            if mask_s.sum() < L // 2:
                continue
            mae_s = np.mean(np.abs(d_part[mask_s] - gt_part[mask_s]))
            if mae_s < best_mae:
                best_mae, best_shift = mae_s, s

        shift_counts[best_shift] = shift_counts.get(best_shift, 0) + 1
        if mae0 > 0:
            shift_mae_improvement.append(100.0 * (mae0 - best_mae) / mae0)

    print(f"\n-- Shift analysis (n={len(matched)} windows) --")
    print(f"  Best-shift distribution (shift -> count):")
    for s in sorted(shift_counts):
        bar = "|" * shift_counts[s]
        pct = 100.0 * shift_counts[s] / len(matched)
        marker = " ← check timestamp alignment" if abs(s) > 0 and pct > 30 else ""
        print(f"    shift={s:+d}: {shift_counts[s]:4d}  ({pct:5.1f}%)  {bar}{marker}")
    if shift_mae_improvement:
        print(f"  Median MAE improvement with best shift: {np.median(shift_mae_improvement):.1f}%")

    # ── 2. large deviation analysis ─────────────────────────────────────────────
    extreme_windows = []
    L = len(matched[0]["denormalized_sum"])
    for w in matched:
        # we find largest deviation in hours between GT and denorm
        gt = w["parquet_values"].astype(float)
        d  = w["denormalized_sum"].astype(float)
        mask = ~np.isnan(gt)
        if mask.sum() < L // 2:
            continue
        gt_masked = gt[mask]
        d_masked = d[mask]
        # WE dont care about the tail we are purely looking for large differences between the two signals
        deviations = np.abs(d_masked - gt_masked)
        max_dev = np.max(deviations)
        max_dev_idx = np.argmax(deviations)
        if max_dev > 1:  # Large deviation threshold
            extreme_windows.append({
                'meter_id': w['meter_id'],
                'max_deviation': max_dev,
                'gt_value': gt_masked[max_dev_idx],
                'd_value': d_masked[max_dev_idx],
                'index': max_dev_idx
            })
    
    if extreme_windows:
        print(f"\n-- Large deviation analysis (threshold > 1000) --")
        print(f"  Found {len(extreme_windows)} windows with extreme deviations:")
        for ew in sorted(extreme_windows, key=lambda x: x['max_deviation'], reverse=True)[:10]:
            print(f"    Meter {ew['meter_id']}: max_dev={ew['max_deviation']:.1f}, GT={ew['gt_value']:.1f}, D={ew['d_value']:.1f}")
        print("  ⚠ SERIOUS DATA ISSUE: Large deviations detected between GT and denormalized values!")
    else:
        print(f"\n-- Large deviation analysis (threshold > 1000) --")
        print(f"  No extreme deviations found (all max deviations ≤ 1000). Data looks consistent.")
    


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare 3-phase test windows (summed across phases) against hourly parquet ground truth."
    )
    parser.add_argument("model_path", type=str,
                        help="Path to trained model folder (must use phased data config)")
    parser.add_argument("--n-windows", type=int, default=200,
                        help="Number of test windows to compare (default: 200)")
    parser.add_argument("--data-dir", type=str,
                        default=str(ROOT_DIR / "data" / "_parquet_cache"),
                        help="Directory containing hourly parquet files (default: data/_parquet_cache)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_path = str(args.model_path).rstrip("/\\")

    print(f"\n{'='*70}")
    print("3-PHASE SUM vs HOURLY PARQUET  —  diagnostic")
    print(f"{'='*70}")
    print(f"Model path : {model_path}")
    print(f"Data dir   : {args.data_dir}")
    print(f"N windows  : {args.n_windows}")
    print(f"Device     : {args.device}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    # 1. Load config + model
    print("\nStep 1: Loading config and model...")
    config = _load_config_for_model(model_path)
    model, normalizer, conditioner = load_model(model_path, device=args.device)
    print(f"  Normalizer: {type(normalizer).__name__ if normalizer is not None else 'None'}")
    if normalizer is not None:
        stats = getattr(normalizer, "mean", None)
        if stats is not None:
            try:
                feat_dim = int(np.asarray(stats).reshape(-1).shape[0])
                print(f"  Normalizer feature dim: {feat_dim}")
            except Exception:
                pass

    # 2. Build test loader
    print("\nStep 2: Building test loader...")
    test_loader = setup_data_loader(config, normalizer, device=args.device, model_path=model_path)
    if test_loader is None:
        raise ValueError("setup_data_loader returned None — no test data available.")

    # 3. Collect windows
    print(f"\nStep 3: Collecting up to {args.n_windows} test windows...")
    windows = collect_windows(test_loader, normalizer, args.n_windows, args.device)
    print(f"  Collected {len(windows)} windows from "
          f"{len({w['meter_id'] for w in windows})} distinct meters")

    if not windows:
        raise ValueError("No windows collected.")

    # 4. Load matching hourly parquet data
    meter_ids = {w["meter_id"] for w in windows}
    print(f"\nStep 4: Loading hourly parquet for {len(meter_ids)} meters from {args.data_dir}...")
    hourly_df = _load_hourly_parquet(args.data_dir, meter_ids)
    print(f"  Loaded {len(hourly_df):,} rows")

    matched = 0
    for w in windows:
        gt = _lookup_window(hourly_df, w["meter_id"], w["start_ts"], w["window_len"])
        w["parquet_values"] = gt
        if gt is not None:
            matched += 1
    print(f"  Matched {matched}/{len(windows)} windows to parquet ground truth")

    # 5. Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(model_path, f"phased_vs_hourly_{timestamp}")

    print(f"\nStep 5: Generating plots...")
    plot_individual(windows, output_dir)
    plot_aggregated(windows, output_dir)

    print_statistics(windows)
    print_mismatch_summary(windows)

    print(f"\n{'='*70}")
    print(f"✓ Done — plots saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()