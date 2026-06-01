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


def resolve_holdout_meter_ids(
    topology_path: str, feeder_id: int | list[int]
) -> frozenset[int]:
    """Return the set of meter IDs that belong to *feeder_id* in the topology JSON.

    feeder_id may be a single integer or a list of integers; the returned set is the
    union of all meters across all specified feeders.
    The topology JSON maps meter ID strings to dicts that contain a "Feeder ID" key.
    Raises ValueError if any feeder_id is not found (zero matching meters).
    """
    with open(topology_path, encoding="utf-8") as fh:
        topology: dict = json.load(fh)

    feeder_ids = [feeder_id] if isinstance(feeder_id, int) else list(feeder_id)
    meter_ids: frozenset[int] = frozenset()
    for fid in feeder_ids:
        feeder_str = str(fid)
        matched = frozenset(
            int(meter_str)
            for meter_str, entry in topology.items()
            if str(entry.get("Feeder ID", "")) == feeder_str
        )
        if not matched:
            raise ValueError(
                f"Feeder ID {fid} not found in topology file '{topology_path}'. "
                "Check that holdout_feeder_id is a valid numeric Feeder ID present in the file."
            )
        meter_ids |= matched
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
        windows:    [N, window_size] float array â€” one row per window.
        timestamps: [N] int64 array â€” Unix seconds marking the start of each window.
        meter_ids:  [N] int array â€” meter ID for each window (used for grouping only).

    Returns:
        (mean, p05, p95) â€” three arrays each of shape [window_size].
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
    title = f"Feeder Holdout Aggregated Profile â€” {n_meters} meters"
    if title_suffix:
        title = f"{title} â€” {title_suffix}"

    hours = np.arange(len(mean_r))
    fig, ax = plt.subplots()
    ax.plot(hours, mean_r, color="steelblue", label="Real (mean)")
    ax.fill_between(hours, p05_r, p95_r, color="steelblue", alpha=0.25, label="Real (5â€“95%)")
    ax.plot(hours, mean_s, color="darkorange", label="Synthetic (mean)")
    ax.fill_between(hours, p05_s, p95_s, color="darkorange", alpha=0.25, label="Synthetic (5â€“95%)")
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
      ElectricVehicleConditionerBool  â†’ [N, 1] float: 0=no EV, 1=has EV
      ElectricVehicleConditioner      â†’ [N, 4] one-hot: index 0 = no EV
      HeatpumpPhaseConditioner        â†’ [N, 4] one-hot: index 0 = no HP
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
            # Use the public API â€” identical code path to training, handles all edge cases.
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
    - feeder_holdout_profile_all.pdf       â€” full dataset
    - feeder_holdout_profile_dow_*.pdf     â€” one per day of week (Monâ€“Sun)
    - feeder_holdout_profile_month_*.pdf   â€” one per month present in the data
    - feeder_holdout_profile_ev.pdf / no_ev.pdf   â€” EV vs no-EV (if conditioner)
    - feeder_holdout_profile_hp.pdf / no_hp.pdf   â€” HP vs no-HP (if conditioner)
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
            print(f"  [holdout analysis] Skipping '{suffix}' â€” no real windows.")
            return
        if len(s_win) == 0:
            print(f"  [holdout analysis] Skipping '{suffix}' â€” no synthetic windows.")
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
        print("  [holdout analysis] No conditioner provided â€” skipping EV/HP plots.")
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


def extract_power_flow_scalars(
    net,
    meter_to_load_index: dict,
    load_array: np.ndarray,
    feeder_id: str | int = "",
    day_index: int = 0,
) -> tuple[float, float]:
    """Return (voltage_nadir, transformer_utilisation) for one calendar day.

    load_array: [n_meters Ã— 24] kW. Row order must match meter_to_load_index key order.
    Non-converging timesteps are logged and skipped; returns (nan, nan) if all fail.
    """
    try:
        import pandapower as pp
    except ImportError as exc:
        raise ImportError(
            "pandapower is required for power flow analysis. Install with: pip install pandapower"
        ) from exc

    load_array = np.asarray(load_array, dtype=np.float64)
    meter_keys = list(meter_to_load_index.keys())

    nadirs: list[float] = []
    utils: list[float] = []

    for h in range(24):
        for row_idx, meter_key in enumerate(meter_keys):
            load_idx = meter_to_load_index[meter_key]
            net.load.at[load_idx, "p_mw"] = load_array[row_idx, h] / 1000.0

        try:
            pp.runpp(net, algorithm="bfsw", numba=True, verbose=False)
        except Exception as exc:
            print(
                f"  [power_flow] WARNING: feeder={feeder_id}, day={day_index}, "
                f"step={h}: {exc}"
            )
            continue

        nadirs.append(float(net.res_bus.vm_pu.min()))
        if len(net.res_trafo) > 0:
            utils.append(float(net.res_trafo.loading_percent.max()) / 100.0)

    if not nadirs:
        return float("nan"), float("nan")

    return float(min(nadirs)), float(max(utils)) if utils else float("nan")


def compute_power_flow_daily_scalars(
    net,
    meter_to_load_index: dict,
    windows: np.ndarray,
    timestamps: np.ndarray,
    meter_ids: np.ndarray,
    feeder_id: str | int = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (voltage_nadirs, transformer_utilisations) â€” one scalar per calendar day.

    Groups windows by calendar date, builds a [meters Ã— 24] kW load matrix for each
    date (meters absent on a date get 0 kW), and calls extract_power_flow_scalars.
    """
    windows = np.asarray(windows, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    meter_ids = np.asarray(meter_ids, dtype=np.int64)

    dates = pd.to_datetime(timestamps, unit="s", utc=True).normalize().astype(str)
    unique_dates = np.unique(dates)

    meter_keys = list(meter_to_load_index.keys())
    n_meters = len(meter_keys)

    nadirs_out: list[float] = []
    utils_out: list[float] = []

    for day_idx, date in enumerate(unique_dates):
        mask = dates == date
        day_windows = windows[mask]
        day_meter_ids = meter_ids[mask]

        load_array = np.zeros((n_meters, 24), dtype=np.float64)
        for row_idx, meter_key in enumerate(meter_keys):
            try:
                mid = int(meter_key)
            except (ValueError, TypeError):
                mid = meter_key
            meter_mask = day_meter_ids == mid
            if meter_mask.any():
                load_array[row_idx, :] = day_windows[meter_mask][0]

        nadir, util = extract_power_flow_scalars(net, meter_to_load_index, load_array, feeder_id, day_idx)
        nadirs_out.append(nadir)
        utils_out.append(util)

    return np.array(nadirs_out, dtype=np.float64), np.array(utils_out, dtype=np.float64)


def run_power_flow_analysis(
    real_nadirs: np.ndarray,
    real_utils: np.ndarray,
    synth_nadirs_per_draw: list[np.ndarray],
    synth_utils_per_draw: list[np.ndarray],
) -> dict:
    """Compute RMSE and inter-draw stats for voltage_nadir and transformer_utilisation.

    Compares sorted real distribution against mean sorted synthetic distribution
    (same method as PeakMagnitude). Returns dict with voltage_nadir and
    transformer_utilisation keys, each containing rmse_vs_real, mean_across_draws,
    std_across_draws (plus mean/std aliases for the summary writer).
    """

    def _sorted_rmse(real: np.ndarray, synth_draws: list[np.ndarray]) -> tuple[float, float, float, list[float]]:
        real_sorted = np.sort(real)
        per_draw_rmse: list[float] = []
        synth_sorted_draws: list[np.ndarray] = []

        for draw in synth_draws:
            draw_sorted = np.sort(draw)
            n_r, n_s = len(real_sorted), len(draw_sorted)
            if n_r == n_s:
                diff = real_sorted - draw_sorted
            else:
                n_out = min(n_r, n_s)
                q = np.linspace(0, 1, n_out)
                r_rs = np.interp(q, np.linspace(0, 1, n_r), real_sorted)
                s_rs = np.interp(q, np.linspace(0, 1, n_s), draw_sorted)
                diff = r_rs - s_rs
            per_draw_rmse.append(float(np.sqrt(np.mean(np.square(diff)))))
            synth_sorted_draws.append(draw_sorted)

        mean_synth = np.mean(synth_sorted_draws, axis=0)
        n_r, n_s = len(real_sorted), len(mean_synth)
        if n_r == n_s:
            diff_mean = real_sorted - mean_synth
        else:
            n_out = min(n_r, n_s)
            q = np.linspace(0, 1, n_out)
            diff_mean = (
                np.interp(q, np.linspace(0, 1, n_r), real_sorted)
                - np.interp(q, np.linspace(0, 1, n_s), mean_synth)
            )
        rmse = float(np.sqrt(np.mean(np.square(diff_mean))))
        return rmse, float(np.mean(per_draw_rmse)), float(np.std(per_draw_rmse)) if len(per_draw_rmse) > 1 else 0.0, per_draw_rmse

    nadir_rmse, nadir_mean, nadir_std, nadir_values = _sorted_rmse(real_nadirs, synth_nadirs_per_draw)
    util_rmse, util_mean, util_std, util_values = _sorted_rmse(real_utils, synth_utils_per_draw)

    return {
        "voltage_nadir": {
            "rmse_vs_real": nadir_rmse,
            "mean_across_draws": nadir_mean,
            "std_across_draws": nadir_std,
            "mean": nadir_rmse,
            "std": nadir_std,
            "values": nadir_values,
        },
        "transformer_utilisation": {
            "rmse_vs_real": util_rmse,
            "mean_across_draws": util_mean,
            "std_across_draws": util_std,
            "mean": util_rmse,
            "std": util_std,
            "values": util_values,
        },
    }


def plot_power_flow_cdf(
    real_scalars: np.ndarray,
    synth_scalars_per_draw: list[np.ndarray],
    output_path: str,
    x_label: str,
    reference_x: float,
    reference_label: str,
    _ax=None,
) -> None:
    """Save a CDF plot comparing real vs synthetic daily power-flow scalars.

    X-axis: scalar value (e.g. voltage in pu, utilisation ratio).
    Y-axis: cumulative fraction of days (empirical CDF).
    Synthetic mean CDF is plotted with a Â±1 std shaded band across draws.
    A vertical dashed reference line is drawn at *reference_x*.

    output_path must end in .png; its parent directory is created if absent.
    Pass *_ax* to inject an existing Axes (used by tests to inspect the plot).
    """
    real_clean = np.sort(real_scalars[~np.isnan(real_scalars)])
    synth_clean = [np.sort(d[~np.isnan(d)]) for d in synth_scalars_per_draw]

    n_quantiles = max(len(real_clean), max((len(d) for d in synth_clean), default=0))
    q = np.linspace(0, 1, n_quantiles)

    real_cdf_x = np.interp(q, np.linspace(0, 1, len(real_clean)), real_clean)

    synth_cdf_xs = [
        np.interp(q, np.linspace(0, 1, len(d)), d) for d in synth_clean if len(d) > 0
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    own_fig = _ax is None
    if own_fig:
        fig, ax = plt.subplots()
    else:
        ax = _ax

    ax.plot(real_cdf_x, q, color="steelblue", label="Real", linewidth=2)

    if synth_cdf_xs:
        synth_arr = np.stack(synth_cdf_xs, axis=0)
        synth_mean = synth_arr.mean(axis=0)
        synth_std = synth_arr.std(axis=0)
        ax.plot(synth_mean, q, color="darkorange", label="Synthetic (mean)", linewidth=2)
        if len(synth_cdf_xs) > 1:
            ax.fill_betweenx(
                q,
                synth_mean - synth_std,
                synth_mean + synth_std,
                color="darkorange",
                alpha=0.25,
                label="Synthetic (Â±1 std)",
            )

    ax.axvline(x=reference_x, color="black", linestyle="--", linewidth=1.2, label=reference_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative fraction")
    ax.legend()
    ax.grid(True)

    if own_fig:
        fig.savefig(output_path, format="png")
        plt.close(fig)


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
      (1) Real â€” same-day grouping on holdout feeder windows.
      (2) Synthetic independent â€” random draws from synth_windows, no day structure.
      (3) Synthetic shared-condition â€” same-day grouping on synth_windows using the
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
            print(f"  [CF] Skipping coincidence factor analysis â€” import failed: {exc}")
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
        print(f"  [CF] Computing real CF(N) â€” {n_max} meters, {n_bootstrap} bootstrap draws...", flush=True)
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
        print(f"  [CF] Skipping shared-condition CF(N) â€” synthetic meter IDs cover "
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