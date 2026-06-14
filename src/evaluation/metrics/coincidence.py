"""Coincidence Factor (CF(N)) — cross-customer dependence evaluation.

Public API:
  compute_coincidence_factor            for real or shared-condition synthetic windows
  compute_coincidence_factor_independent  for independent synthetic windows
  coincidence_factor_scalars            CF(N_max) error + area between curves
  plot_coincidence_factor_curves        three-curve PDF plot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_data_matrix(
    windows: np.ndarray,
    meter_ids: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Reshape flat window array into [n_meters, n_days, window_size].

    Returns (data_matrix, valid_mask, unique_meters, unique_dates).
    data_matrix[i, j] is the window for meter unique_meters[i] on day
    unique_dates[j]; NaN where no window exists for that pair.
    valid_mask[i, j] is True where the window is present.
    """
    windows = np.asarray(windows, dtype=np.float64)
    window_size = windows.shape[1]
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .normalize()
        .astype(str)
        .tolist()
    )

    unique_meters: list = sorted(set(meter_ids.tolist()))
    unique_dates: list = sorted(set(dates))
    meter_to_row = {m: i for i, m in enumerate(unique_meters)}
    date_to_col = {d: j for j, d in enumerate(unique_dates)}

    n_meters = len(unique_meters)
    n_days = len(unique_dates)
    data_matrix = np.full((n_meters, n_days, window_size), np.nan, dtype=np.float64)
    for idx, (meter, date) in enumerate(zip(meter_ids.tolist(), dates)):
        data_matrix[meter_to_row[meter], date_to_col[date]] = windows[idx]

    valid_mask = ~np.isnan(data_matrix[:, :, 0])  # [n_meters, n_days]
    return data_matrix, valid_mask, unique_meters, unique_dates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coincidence_factor(
    windows: np.ndarray,
    meter_ids: np.ndarray,
    timestamps: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.9,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute CF(N) for real or shared-condition synthetic windows.

    CF(N) = E[peak of aggregate of N meters] / (N × E[individual peak]).
    N meters are bootstrapped with replacement from the meter pool.
    For each draw, the aggregate window is summed per calendar date and the
    peak is taken; the mean over valid dates is one CF observation.
    Bootstrap bands reflect meter-selection uncertainty.

    Args:
        windows:     [N_windows, window_size] — one window per row.
        meter_ids:   [N_windows] int — meter ID for each window.
        timestamps:  [N_windows] int64 Unix seconds.
        n_bootstrap: bootstrap draws per N value.
        ci:          confidence interval width (0.9 → 5th–95th percentile bands).
        rng:         NumPy Generator; seeded with 42 if None.

    Returns dict with:
        cf_mean, cf_lower, cf_upper  [n_max] — mean and CI for CF(N).
        n_values                     [n_max] = [1, 2, …, n_max].
        mean_individual_peak         float — E[max(window)] across all windows.
        n_max                        int — number of unique meters.
        daily_peaks_at_n_max         [n_valid_days] — true aggregate daily peaks
                                     using all n_max meters simultaneously; used as
                                     the real aggregate-peak distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    windows = np.asarray(windows, dtype=np.float64)
    meter_ids = np.asarray(meter_ids)
    timestamps = np.asarray(timestamps, dtype=np.int64)

    data_matrix, valid_mask, unique_meters, _ = _build_data_matrix(
        windows, meter_ids, timestamps
    )
    n_meters = len(unique_meters)

    mean_individual_peak = float(np.nanmean(np.nanmax(data_matrix, axis=2)))
    if mean_individual_peak == 0.0:
        raise ValueError("mean_individual_peak is zero — all windows appear to be zero.")

    n_max = n_meters
    alpha = (1.0 - ci) / 2.0

    cf_mean = np.full(n_max, np.nan)
    cf_lower = np.full(n_max, np.nan)
    cf_upper = np.full(n_max, np.nan)

    for n in range(1, n_max + 1):
        cf_draws = np.full(n_bootstrap, np.nan)
        for b in range(n_bootstrap):
            row_idx = rng.integers(0, n_meters, size=n)           # [n]
            aggregate = data_matrix[row_idx].sum(axis=0)          # [n_days, window_size]
            day_valid = np.all(valid_mask[row_idx], axis=0)       # [n_days]
            daily_peaks = aggregate.max(axis=1)                   # [n_days]
            valid_peaks = daily_peaks[day_valid]
            if len(valid_peaks) > 0:
                cf_draws[b] = float(np.mean(valid_peaks)) / (n * mean_individual_peak)

        ok = ~np.isnan(cf_draws)
        if ok.any():
            cf_mean[n - 1] = float(np.mean(cf_draws[ok]))
            cf_lower[n - 1] = float(np.percentile(cf_draws[ok], 100.0 * alpha))
            cf_upper[n - 1] = float(np.percentile(cf_draws[ok], 100.0 * (1.0 - alpha)))

    # True aggregate daily peaks for all n_max meters (no bootstrap).
    all_valid_days = np.all(valid_mask, axis=0)          # [n_days]
    full_aggregate = data_matrix.sum(axis=0)             # [n_days, window_size]
    daily_peaks_at_n_max = full_aggregate.max(axis=1)[all_valid_days]

    return {
        "cf_mean": cf_mean,
        "cf_lower": cf_lower,
        "cf_upper": cf_upper,
        "n_values": np.arange(1, n_max + 1),
        "mean_individual_peak": mean_individual_peak,
        "n_max": n_max,
        "daily_peaks_at_n_max": daily_peaks_at_n_max,
    }


def compute_coincidence_factor_independent(
    windows: np.ndarray,
    n_max: int,
    n_bootstrap: int = 500,
    ci: float = 0.9,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute CF(N) for independent synthetic draws (no day structure).

    For each N, n_bootstrap draws of N windows are sampled uniformly at random,
    summed, and the peak is taken. No same-day grouping.

    Args:
        windows:     [n_synth, window_size] — independent synthetic windows.
        n_max:       maximum N (should match the real feeder's meter count).
        n_bootstrap: bootstrap draws per N.
        ci:          confidence interval width.
        rng:         NumPy Generator; seeded with 42 if None.

    Returns same dict structure as compute_coincidence_factor, plus:
        bootstrap_peaks_at_n_max  [n_bootstrap] — aggregate peak distribution at
                                  n_max; used for the aggregate-peak comparison.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    windows = np.asarray(windows, dtype=np.float64)
    n_synth = len(windows)

    mean_individual_peak = float(np.mean(windows.max(axis=1)))
    if mean_individual_peak == 0.0:
        raise ValueError("mean_individual_peak is zero — all windows appear to be zero.")

    alpha = (1.0 - ci) / 2.0
    cf_mean = np.full(n_max, np.nan)
    cf_lower = np.full(n_max, np.nan)
    cf_upper = np.full(n_max, np.nan)

    for n in range(1, n_max + 1):
        idx = rng.integers(0, n_synth, size=(n_bootstrap, n))     # [n_bootstrap, n]
        aggregate = windows[idx].sum(axis=1)                       # [n_bootstrap, window_size]
        agg_peaks = aggregate.max(axis=1)                          # [n_bootstrap]
        cf_vals = agg_peaks / (n * mean_individual_peak)
        cf_mean[n - 1] = float(np.mean(cf_vals))
        cf_lower[n - 1] = float(np.percentile(cf_vals, 100.0 * alpha))
        cf_upper[n - 1] = float(np.percentile(cf_vals, 100.0 * (1.0 - alpha)))

    # Bootstrap peak distribution at n_max for aggregate-peak comparison.
    idx_max = rng.integers(0, n_synth, size=(n_bootstrap, n_max))
    bootstrap_peaks_at_n_max = windows[idx_max].sum(axis=1).max(axis=1)  # [n_bootstrap]

    return {
        "cf_mean": cf_mean,
        "cf_lower": cf_lower,
        "cf_upper": cf_upper,
        "n_values": np.arange(1, n_max + 1),
        "mean_individual_peak": mean_individual_peak,
        "n_max": n_max,
        "bootstrap_peaks_at_n_max": bootstrap_peaks_at_n_max,
    }


def coincidence_factor_scalars(real_cf: dict, synth_cf: dict) -> dict:
    """Compute table scalars from real and synthetic CF(N) result dicts.

    Returns:
        cf_error_at_n_max    |CF_real(N_max) - CF_synth(N_max)|
        area_between_curves  mean(|CF_real(N) - CF_synth(N)|) for N=1..min(n_max)
    """
    n = min(real_cf["n_max"], synth_cf["n_max"])
    real = real_cf["cf_mean"][:n]
    synth = synth_cf["cf_mean"][:n]
    valid = ~(np.isnan(real) | np.isnan(synth))

    cf_error = (
        float(abs(real[-1] - synth[-1]))
        if not (np.isnan(real[-1]) or np.isnan(synth[-1]))
        else float("nan")
    )
    area = float(np.mean(np.abs(real[valid] - synth[valid]))) if valid.any() else float("nan")

    return {"cf_error_at_n_max": cf_error, "area_between_curves": area}


def plot_coincidence_factor_curves(
    real_cf: dict,
    synth_indep_cf: dict,
    synth_shared_cf: dict | None,
    output_path: str,
) -> None:
    """Save a three-curve CF(N) comparison plot as PDF.

    Curves: real (blue), synthetic-independent (orange),
    synthetic-shared-condition (green, optional).
    Shaded bands show the CI from bootstrap meter-selection draws.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    def _curve(result: dict, color: str, label: str) -> None:
        n = result["n_values"]
        ax.plot(n, result["cf_mean"], color=color, label=label, linewidth=2)
        ax.fill_between(n, result["cf_lower"], result["cf_upper"], color=color, alpha=0.15)

    _curve(real_cf, "steelblue", "Real")
    _curve(synth_indep_cf, "darkorange", "Synthetic (independent draws)")
    if synth_shared_cf is not None:
        _curve(synth_shared_cf, "forestgreen", "Synthetic (shared condition)")

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of meters N")
    ax.set_ylabel("Coincidence Factor CF(N)")
    ax.set_title("Coincidence Factor vs. Portfolio Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)