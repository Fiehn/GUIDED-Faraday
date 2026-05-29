"""Feeder holdout utilities: meter-ID resolution and aggregated profile computation."""
from __future__ import annotations

import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_holdout_meter_ids(topology_path: str, feeder_id: int) -> frozenset[int]:
    """Return the set of meter IDs that belong to *feeder_id* in the topology JSON.

    The topology JSON maps meter ID strings to dicts that contain a "Feeder ID" key.
    Raises ValueError if feeder_id is not found (zero matching meters).
    """
    with open(topology_path, encoding="utf-8") as fh:
        topology: dict = json.load(fh)

    feeder_str = str(feeder_id)
    meter_ids = frozenset(
        int(meter_str)
        for meter_str, entry in topology.items()
        if str(entry.get("Feeder ID", "")) == feeder_str
    )
    if not meter_ids:
        raise ValueError(
            f"Feeder ID {feeder_id} not found in topology file '{topology_path}'. "
            "Check that holdout_feeder_id is a valid numeric Feeder ID present in the file."
        )
    return meter_ids


def _compute_daily_matrix(windows: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Return [n_dates, window_size] daily feeder aggregate matrix.

    For each calendar date, sums all meter windows for that date into one
    aggregate curve and stacks the results.
    """
    windows = np.asarray(windows, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    dates = pd.to_datetime(timestamps, unit="s", utc=True).normalize().astype(str)
    unique_dates = np.unique(dates)
    return np.stack([windows[dates == d].sum(axis=0) for d in unique_dates], axis=0)


def compute_feeder_aggregated_profile(
    windows: np.ndarray,
    timestamps: np.ndarray,
    meter_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 5th/95th-percentile daily feeder-level load profiles.

    For each calendar date, sums *windows* across all meters present that day to
    produce one aggregated curve. The mean and percentile bands are then computed
    across all dates.

    Args:
        windows:    [N, window_size] float array — one row per window.
        timestamps: [N] int64 array — Unix seconds marking the start of each window.
        meter_ids:  [N] int array — meter ID for each window (used for grouping only).

    Returns:
        (mean, p05, p95) — three arrays each of shape [window_size].
    """
    daily_matrix = _compute_daily_matrix(windows, timestamps)
    mean = daily_matrix.mean(axis=0)
    p05 = np.percentile(daily_matrix, 5, axis=0)
    p95 = np.percentile(daily_matrix, 95, axis=0)
    return mean.astype(np.float32), p05.astype(np.float32), p95.astype(np.float32)


def plot_feeder_aggregated_profile(
    real_windows: np.ndarray,
    real_timestamps: np.ndarray,
    real_meter_ids: np.ndarray,
    synth_windows: np.ndarray,
    synth_timestamps: np.ndarray,
    synth_meter_ids: np.ndarray,
    output_path: str,
    title_suffix: str = "",
    synth_extra_draws: list[np.ndarray] | None = None,
) -> None:
    """Save a two-curve shaded-band PDF comparing real vs synthetic feeder profiles.

    synth_extra_draws: additional draw arrays (each shaped like synth_windows). When
    provided, daily aggregate matrices are pooled across all K draws before computing
    synthetic percentile bands, giving inter-day AND inter-draw variance instead of
    just inter-day variance from a single averaged draw.
    """
    mean_r, p05_r, p95_r = compute_feeder_aggregated_profile(
        real_windows, real_timestamps, real_meter_ids
    )

    synth_daily = _compute_daily_matrix(synth_windows, synth_timestamps)
    if synth_extra_draws:
        extra = [_compute_daily_matrix(d, synth_timestamps) for d in synth_extra_draws]
        synth_daily = np.concatenate([synth_daily] + extra, axis=0)
    mean_s = synth_daily.mean(axis=0).astype(np.float32)
    p05_s = np.percentile(synth_daily, 5, axis=0).astype(np.float32)
    p95_s = np.percentile(synth_daily, 95, axis=0).astype(np.float32)

    n_meters = len(np.unique(real_meter_ids))
    title = f"Feeder Holdout Aggregated Profile — {n_meters} meters"
    if title_suffix:
        title = f"{title} — {title_suffix}"

    hours = np.arange(len(mean_r))
    fig, ax = plt.subplots()
    ax.plot(hours, mean_r, color="steelblue", label="Real (mean)")
    ax.fill_between(hours, p05_r, p95_r, color="steelblue", alpha=0.25, label="Real (5–95%)")
    ax.plot(hours, mean_s, color="darkorange", label="Synthetic (mean)")
    ax.fill_between(hours, p05_s, p95_s, color="darkorange", alpha=0.25, label="Synthetic (5–95%)")
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Aggregated consumption [kWh]")
    ax.legend()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _get_ev_hp_flags_per_window(
    meter_ids: np.ndarray,
    conditioner,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ev_flags, hp_flags) bool arrays [N] keyed by each window's meter_id.

    Uses the conditioner's public get_conditions() API so the lookup is identical
    to what the model saw during training. Returns all-False arrays (with a warning)
    if the conditioner lacks EV/HP items or the call fails.

    Interpretation:
      ElectricVehicleConditionerBool  → [N, 1] float: 0=no EV, 1=has EV
      ElectricVehicleConditioner      → [N, 4] one-hot: index 0 = no EV
      HeatpumpPhaseConditioner        → [N, 4] one-hot: index 0 = no HP
    In all cases: index/value 0 means absence, non-zero means presence.
    """
    import torch

    n = len(meter_ids)
    ev_flags = np.zeros(n, dtype=bool)
    hp_flags = np.zeros(n, dtype=bool)

    if conditioner is None:
        return ev_flags, hp_flags

    meter_tensor = torch.as_tensor(np.asarray(meter_ids, dtype=np.int64), dtype=torch.long)

    for flag_arr, keys in [
        (ev_flags, ("ev_phase", "ev")),
        (hp_flags, ("heatpump_phase", "hp", "heatpump")),
    ]:
        cond_item = None
        for key in keys:
            cond_item = conditioner.condition_items.get(key)
            if cond_item is not None:
                break
        if cond_item is None:
            continue

        try:
            # Use the public API — identical code path to training, handles all edge cases.
            result = cond_item.get_conditions(x=None, meter_id=meter_tensor, timestamps=None)
            # result: [N, dim] float tensor
            # dim=1 (Bool): 0.0 = absent, 1.0 = present
            # dim=4 (Phase): one-hot where index 0 = absent, indices 1-3 = present
            if result is None:
                print(f"  [holdout] Warning: get_conditions returned None for '{key}'; flags will be all-False.")
                continue
            result_np = result.detach().cpu().numpy()
            if result_np.shape[1] == 1:
                flag_arr[:] = result_np[:, 0] > 0.5
            else:
                # one-hot: presence when argmax != 0
                flag_arr[:] = result_np[:, 0] < 0.5
        except Exception as e:
            key_used = keys[0]
            print(f"  [holdout] Warning: EV/HP flag lookup failed for '{key_used}': {e}")
            traceback.print_exc()

    n_ev = ev_flags.sum()
    n_hp = hp_flags.sum()
    n_unique = len(np.unique(meter_ids))
    print(f"  [holdout] Condition flags over {n_unique} unique meters: "
          f"EV={n_ev} ({100*n_ev/n_unique:.1f}%),  HP={n_hp} ({100*n_hp/n_unique:.1f}%)")

    return ev_flags, hp_flags


def plot_feeder_holdout_analysis(
    real_windows: np.ndarray,
    real_timestamps: np.ndarray,
    real_meter_ids: np.ndarray,
    synth_windows: np.ndarray,
    synth_timestamps: np.ndarray,
    synth_meter_ids: np.ndarray,
    output_dir: str,
    conditioner=None,
    synth_extra_draws: list[np.ndarray] | None = None,
) -> None:
    """Generate all feeder holdout profile plots and save them to output_dir.

    Produces:
    - feeder_holdout_profile_all.pdf       — full dataset
    - feeder_holdout_profile_dow_*.pdf     — one per day of week (Mon–Sun)
    - feeder_holdout_profile_month_*.pdf   — one per month present in the data
    - feeder_holdout_profile_ev.pdf / no_ev.pdf   — EV vs no-EV (if conditioner)
    - feeder_holdout_profile_hp.pdf / no_hp.pdf   — HP vs no-HP (if conditioner)
    """
    os.makedirs(output_dir, exist_ok=True)

    real_windows    = np.asarray(real_windows,    dtype=np.float32)
    synth_windows   = np.asarray(synth_windows,   dtype=np.float32)
    real_timestamps  = np.asarray(real_timestamps,  dtype=np.int64)
    synth_timestamps = np.asarray(synth_timestamps, dtype=np.int64)
    real_meter_ids   = np.asarray(real_meter_ids,   dtype=np.int64)
    synth_meter_ids  = np.asarray(synth_meter_ids,  dtype=np.int64)

    def _safe_plot(r_win, r_ts, r_mid, s_win, s_ts, s_mid, fname, suffix, s_extra_draws=None):
        if len(r_win) == 0:
            print(f"  [holdout analysis] Skipping '{suffix}' — no real windows.")
            return
        if len(s_win) == 0:
            print(f"  [holdout analysis] Skipping '{suffix}' — no synthetic windows.")
            return
        n_m = len(np.unique(r_mid))
        try:
            plot_feeder_aggregated_profile(
                real_windows=r_win, real_timestamps=r_ts, real_meter_ids=r_mid,
                synth_windows=s_win, synth_timestamps=s_ts, synth_meter_ids=s_mid,
                output_path=os.path.join(output_dir, fname),
                title_suffix=suffix,
                synth_extra_draws=s_extra_draws,
            )
            print(f"  [holdout analysis] Saved {fname}  ({n_m} meters)")
        except Exception as e:
            print(f"  [holdout analysis] Failed '{fname}': {e}")
            traceback.print_exc()

    # --- All windows ---
    _safe_plot(real_windows, real_timestamps, real_meter_ids,
               synth_windows, synth_timestamps, synth_meter_ids,
               "feeder_holdout_profile_all.pdf", "",
               s_extra_draws=synth_extra_draws)

    # --- Day of week ---
    real_dts  = pd.to_datetime(real_timestamps,  unit="s", utc=True)
    synth_dts = pd.to_datetime(synth_timestamps, unit="s", utc=True)
    real_dow  = real_dts.day_of_week.to_numpy()
    synth_dow = synth_dts.day_of_week.to_numpy()

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for dow in range(7):
        r_mask = real_dow == dow
        s_mask = synth_dow == dow
        if r_mask.sum() == 0:
            print(f"  [holdout analysis] No real windows for {dow_names[dow]}; skipping.")
            continue
        _s_extra = [d[s_mask] for d in synth_extra_draws] if synth_extra_draws else None
        _safe_plot(
            real_windows[r_mask],  real_timestamps[r_mask],  real_meter_ids[r_mask],
            synth_windows[s_mask], synth_timestamps[s_mask], synth_meter_ids[s_mask],
            f"feeder_holdout_profile_dow_{dow}_{dow_names[dow].lower()}.pdf",
            dow_names[dow],
            s_extra_draws=_s_extra,
        )

    # --- Month of year ---
    real_month  = real_dts.month.to_numpy()
    synth_month = synth_dts.month.to_numpy()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for month in sorted(np.unique(real_month)):
        r_mask = real_month == month
        s_mask = synth_month == month
        name = month_names[int(month) - 1]
        _s_extra = [d[s_mask] for d in synth_extra_draws] if synth_extra_draws else None
        _safe_plot(
            real_windows[r_mask],  real_timestamps[r_mask],  real_meter_ids[r_mask],
            synth_windows[s_mask], synth_timestamps[s_mask], synth_meter_ids[s_mask],
            f"feeder_holdout_profile_month_{int(month):02d}_{name.lower()}.pdf",
            name,
            s_extra_draws=_s_extra,
        )

    # --- EV / HP ---
    if conditioner is None:
        print("  [holdout analysis] No conditioner provided — skipping EV/HP plots.")
        return

    ev_flags_r, hp_flags_r = _get_ev_hp_flags_per_window(real_meter_ids, conditioner)
    # synth_meter_ids == real_meter_ids when preserve_condition_order=True;
    # derive separately for correctness in case they differ.
    ev_flags_s, hp_flags_s = _get_ev_hp_flags_per_window(synth_meter_ids, conditioner)

    for r_flag, s_flag, label_pos, label_neg, fname_pos, fname_neg in [
        (ev_flags_r, ev_flags_s, "EV",    "No EV", "ev",    "no_ev"),
        (hp_flags_r, hp_flags_s, "HP",    "No HP", "hp",    "no_hp"),
    ]:
        n_pos = len(np.unique(real_meter_ids[r_flag]))  if r_flag.any()  else 0
        n_neg = len(np.unique(real_meter_ids[~r_flag])) if (~r_flag).any() else 0
        _s_extra_pos = [d[s_flag]  for d in synth_extra_draws] if synth_extra_draws else None
        _s_extra_neg = [d[~s_flag] for d in synth_extra_draws] if synth_extra_draws else None
        _safe_plot(
            real_windows[r_flag],   real_timestamps[r_flag],   real_meter_ids[r_flag],
            synth_windows[s_flag],  synth_timestamps[s_flag],  synth_meter_ids[s_flag],
            f"feeder_holdout_profile_{fname_pos}.pdf",
            f"{label_pos} (N={n_pos} meters)",
            s_extra_draws=_s_extra_pos,
        )
        _safe_plot(
            real_windows[~r_flag],  real_timestamps[~r_flag],  real_meter_ids[~r_flag],
            synth_windows[~s_flag], synth_timestamps[~s_flag], synth_meter_ids[~s_flag],
            f"feeder_holdout_profile_{fname_neg}.pdf",
            f"{label_neg} (N={n_neg} meters)",
            s_extra_draws=_s_extra_neg,
        )


def run_coincidence_factor_analysis(
    real_windows: np.ndarray,
    real_meter_ids: np.ndarray,
    real_timestamps: np.ndarray,
    synth_windows: np.ndarray,
    synth_meter_ids: np.ndarray,
    synth_timestamps: np.ndarray,
    output_dir: str,
    n_bootstrap: int = 200,
) -> dict:
    """Compute CF(N) three-curve analysis and save the plot to output_dir.

    Three curves:
      (1) Real — same-day grouping on holdout feeder windows.
      (2) Synthetic independent — random draws from synth_windows, no day structure.
      (3) Synthetic shared-condition — same-day grouping on synth_windows using the
          real meter IDs and timestamps (valid when preserve_condition_order=True).

    Returns a dict of scalars suitable for merging into eval_results_feeder_holdout.json:
      CF_error_N_max_independent   |CF_real(N_max) - CF_synth_indep(N_max)|
      CF_area_independent          mean |CF_real(N) - CF_synth_indep(N)| over N
      CF_error_N_max_shared_cond   |CF_real(N_max) - CF_synth_shared(N_max)|
      CF_area_shared_cond          mean |CF_real(N) - CF_synth_shared(N)| over N
    """
    try:
        from evaluation.coincidence import (
            compute_coincidence_factor,
            compute_coincidence_factor_independent,
            coincidence_factor_scalars,
            plot_coincidence_factor_curves,
        )
    except ImportError:
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
            from src.evaluation.coincidence import (
                compute_coincidence_factor,
                compute_coincidence_factor_independent,
                coincidence_factor_scalars,
                plot_coincidence_factor_curves,
            )
        except ImportError as exc:
            print(f"  [CF] Skipping coincidence factor analysis — import failed: {exc}")
            return {}

    os.makedirs(output_dir, exist_ok=True)
    real_windows   = np.asarray(real_windows,   dtype=np.float64)
    synth_windows  = np.asarray(synth_windows,  dtype=np.float64)
    real_meter_ids  = np.asarray(real_meter_ids,  dtype=np.int64)
    synth_meter_ids = np.asarray(synth_meter_ids, dtype=np.int64)
    real_timestamps  = np.asarray(real_timestamps,  dtype=np.int64)
    synth_timestamps = np.asarray(synth_timestamps, dtype=np.int64)

    n_max = len(np.unique(real_meter_ids))
    rng = np.random.default_rng(42)

    try:
        print(f"  [CF] Computing real CF(N) — {n_max} meters, {n_bootstrap} bootstrap draws...", flush=True)
        real_cf = compute_coincidence_factor(
            real_windows, real_meter_ids, real_timestamps,
            n_bootstrap=n_bootstrap, rng=rng,
        )
    except Exception as e:
        print(f"  [CF] Real CF(N) failed: {e}")
        traceback.print_exc()
        return {}

    try:
        print(f"  [CF] Computing synthetic-independent CF(N)...", flush=True)
        synth_indep_cf = compute_coincidence_factor_independent(
            synth_windows, n_max=n_max,
            n_bootstrap=n_bootstrap, rng=np.random.default_rng(42),
        )
    except Exception as e:
        print(f"  [CF] Independent synthetic CF(N) failed: {e}")
        traceback.print_exc()
        synth_indep_cf = None

    synth_shared_cf = None
    if len(np.unique(synth_meter_ids)) >= n_max:
        try:
            print(f"  [CF] Computing synthetic-shared-condition CF(N)...", flush=True)
            synth_shared_cf = compute_coincidence_factor(
                synth_windows, synth_meter_ids, synth_timestamps,
                n_bootstrap=n_bootstrap, rng=np.random.default_rng(42),
            )
        except Exception as e:
            print(f"  [CF] Shared-condition synthetic CF(N) failed: {e}")
            traceback.print_exc()
    else:
        print(f"  [CF] Skipping shared-condition CF(N) — synthetic meter IDs cover "
              f"{len(np.unique(synth_meter_ids))} unique meters (need {n_max}).")

    try:
        plot_coincidence_factor_curves(
            real_cf,
            synth_indep_cf if synth_indep_cf is not None else real_cf,
            synth_shared_cf,
            os.path.join(output_dir, "coincidence_factor.pdf"),
        )
        print(f"  [CF] Plot saved to {output_dir}/coincidence_factor.pdf")
    except Exception as e:
        print(f"  [CF] Plot failed: {e}")
        traceback.print_exc()

    scalars: dict = {}
    if synth_indep_cf is not None:
        s = coincidence_factor_scalars(real_cf, synth_indep_cf)
        scalars["CF_error_N_max_independent"] = s["cf_error_at_n_max"]
        scalars["CF_area_independent"] = s["area_between_curves"]
    if synth_shared_cf is not None:
        s = coincidence_factor_scalars(real_cf, synth_shared_cf)
        scalars["CF_error_N_max_shared_cond"] = s["cf_error_at_n_max"]
        scalars["CF_area_shared_cond"] = s["area_between_curves"]

    if scalars:
        n_max_val = real_cf["n_max"]
        print(
            f"  [CF] N_max={n_max_val}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in scalars.items())
        )
    return scalars