"""Candidate-discovery script for holdout feeder selection.

Usage:
    python scripts/find_holdout_feeder.py \\
        --topology data/topology/cleaned/Cleaned_SM_Topology_mapping.json \\
        --ev       data/EV/EV_detection_reduced_60000_SM.csv \\
        --hp       data/HP/label_propagation_predictions.csv \\
        [--parquet data/_parquet_cache/hourly.parquet]  # single multi-index file OR a directory of sm_*.parquet \\
        [--ev-thresh 0.20] \\
        [--hp-thresh 0.10]

Prints a table of LV Feeders that meet the EV% and HP% thresholds.
The researcher then manually copies the chosen Feeder ID into the config.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeederStats:
    feeder_id: str
    n_meters: int
    ev_pct: float
    hp_pct: float
    mean_daily_kwh: float = float("nan")
    coverage_pct: float = float("nan")  # fraction of 365 days covered, averaged across feeder meters


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_feeder_meter_map(topology_path) -> dict[str, set[str]]:
    """Return {feeder_id → set of meter_ids} for meters where Has data=True."""
    with open(topology_path) as f:
        topology = json.load(f)
    result: dict[str, set[str]] = {}
    for meter_id, info in topology.items():
        if not info.get("Has data", False):
            continue
        feeder_id = info["Feeder ID"]
        result.setdefault(feeder_id, set()).add(meter_id)
    return result


def load_ev_flags(ev_path) -> dict[str, bool]:
    """Return {meter_id → has_ev} from EV detection CSV.

    SM_ID column contains values like 'sm_100025.parquet'; we strip the prefix/suffix.
    """
    df = pd.read_csv(ev_path)
    flags: dict[str, bool] = {}
    for _, row in df.iterrows():
        sm_id = str(row["SM_ID"])
        meter_id = re.sub(r"[^\d]", "", sm_id)
        flags[meter_id] = str(row["EV_detected"]).strip().lower() == "yes"
    return flags


def load_hp_flags(hp_path) -> dict[str, bool]:
    """Return {meter_id → has_hp} from HP prediction CSV.

    Index contains phase-prefixed IDs like 'l1_100025'. A meter is flagged True
    if any phase has predicted_hp=True.
    """
    df = pd.read_csv(hp_path, index_col=0)
    flags: dict[str, bool] = {}
    for idx, row in df.iterrows():
        # strip phase prefix: 'l1_100025' → '100025'
        meter_id = re.sub(r"^l\d+_", "", str(idx))
        has_hp = bool(row["predicted_hp"])
        flags[meter_id] = flags.get(meter_id, False) or has_hp
    return flags


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_feeder_stats(
    feeder_map: dict[str, set[str]],
    ev_flags: dict[str, bool],
    hp_flags: dict[str, bool],
    parquet_dir=None,
) -> list[FeederStats]:
    """Compute per-feeder statistics."""
    stats: list[FeederStats] = []
    for feeder_id, meters in feeder_map.items():
        n = len(meters)
        ev_count = sum(ev_flags.get(m, False) for m in meters)
        hp_count = sum(hp_flags.get(m, False) for m in meters)
        if parquet_dir:
            mean_kwh, coverage = _compute_parquet_stats(meters, parquet_dir)
        else:
            mean_kwh, coverage = float("nan"), float("nan")
        stats.append(FeederStats(
            feeder_id=feeder_id,
            n_meters=n,
            ev_pct=ev_count / n if n > 0 else 0.0,
            hp_pct=hp_count / n if n > 0 else 0.0,
            mean_daily_kwh=mean_kwh,
            coverage_pct=coverage,
        ))
    return stats


def _compute_parquet_stats(meters: set[str], parquet_path) -> tuple[float, float]:
    """Dispatch to file or directory loader. Returns (mean_daily_kwh, coverage_pct)."""
    p = Path(parquet_path)
    if p.is_dir():
        return _parquet_stats_from_dir(meters, p)
    elif p.is_file():
        return _parquet_stats_from_file(meters, p)
    return float("nan"), float("nan")


def _parquet_stats_from_dir(meters: set[str], parquet_dir: Path) -> tuple[float, float]:
    """Per-feeder stats from a directory of sm_{id}.parquet files."""
    kwh_per_meter: list[float] = []
    cov_per_meter: list[float] = []
    for meter_id in meters:
        p = parquet_dir / f"sm_{meter_id}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["hourly_energy_consumption", "timestamp_dst"])
        df["date"] = pd.to_datetime(df["timestamp_dst"]).dt.date
        daily = df.groupby("date")["hourly_energy_consumption"].sum()
        if len(daily) > 0:
            kwh_per_meter.append(daily.mean())
            span = (pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days + 1
            cov_per_meter.append(len(daily) / span)
    kwh = float(np.mean(kwh_per_meter)) if kwh_per_meter else float("nan")
    cov = float(np.mean(cov_per_meter)) if cov_per_meter else float("nan")
    return kwh, cov


def _parquet_stats_from_file(meters: set[str], parquet_file: Path) -> tuple[float, float]:
    """Per-feeder stats from a single multi-index parquet file.

    Expects index (meter_number, timestamp_dst) and column hourly_energy_consumption.
    Uses pyarrow filtering to avoid scanning all rows.
    """
    import pyarrow.parquet as pq

    int_meter_ids = [int(m) for m in meters if m.lstrip("-").isdigit()]
    if not int_meter_ids:
        return float("nan"), float("nan")

    table = pq.read_table(parquet_file, filters=[("meter_number", "in", int_meter_ids)])
    df = table.to_pandas()
    if df.empty:
        return float("nan"), float("nan")

    if "meter_number" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["timestamp_dst"]).dt.date
    kwh_per_meter: list[float] = []
    cov_per_meter: list[float] = []
    for _, grp in df.groupby("meter_number"):
        daily = grp.groupby("date")["hourly_energy_consumption"].sum()
        if len(daily) > 0:
            kwh_per_meter.append(daily.mean())
            span = (pd.Timestamp(daily.index.max()) - pd.Timestamp(daily.index.min())).days + 1
            cov_per_meter.append(len(daily) / span)
    kwh = float(np.mean(kwh_per_meter)) if kwh_per_meter else float("nan")
    cov = float(np.mean(cov_per_meter)) if cov_per_meter else float("nan")
    return kwh, cov


# ---------------------------------------------------------------------------
# Filtering and display
# ---------------------------------------------------------------------------

def filter_and_sort(
    stats: list[FeederStats],
    ev_thresh: float = 0.20,
    hp_thresh: float = 0.10,
) -> list[FeederStats]:
    """Return feeders meeting both thresholds, sorted by EV% descending."""
    candidates = [s for s in stats if s.ev_pct > ev_thresh and s.hp_pct > hp_thresh]
    return sorted(candidates, key=lambda s: s.ev_pct, reverse=True)


def print_table(candidates: list[FeederStats]) -> None:
    if not candidates:
        print("No feeders meet the specified thresholds.")
        return
    header = f"{'Feeder ID':>15}  {'n_meters':>8}  {'EV%':>7}  {'HP%':>7}  {'Mean daily kWh':>15}  {'Coverage%':>10}"
    print(header)
    print("-" * len(header))
    for s in candidates:
        kwh = f"{s.mean_daily_kwh:>15.2f}" if not math.isnan(s.mean_daily_kwh) else f"{'N/A':>15}"
        cov = f"{s.coverage_pct:>9.1%}" if not math.isnan(s.coverage_pct) else f"{'N/A':>9}"
        print(f"{s.feeder_id:>15}  {s.n_meters:>8d}  {s.ev_pct:>6.1%}  {s.hp_pct:>6.1%}  {kwh}  {cov}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", required=True, help="Path to Cleaned_SM_Topology_mapping.json")
    p.add_argument("--ev", required=True, help="Path to EV detection CSV")
    p.add_argument("--hp", required=True, help="Path to HP prediction CSV")
    p.add_argument("--parquet", "--parquet-dir", dest="parquet", default=None,
                   help="Hourly data: a single multi-index parquet file OR a directory of sm_*.parquet files (optional)")
    p.add_argument("--ev-thresh", type=float, default=0.20, help="Minimum EV%% (default 0.20)")
    p.add_argument("--hp-thresh", type=float, default=0.10, help="Minimum HP%% (default 0.10)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    feeder_map = load_feeder_meter_map(args.topology)
    ev_flags = load_ev_flags(args.ev)
    hp_flags = load_hp_flags(args.hp)
    stats = compute_feeder_stats(feeder_map, ev_flags, hp_flags, parquet_dir=args.parquet)
    candidates = filter_and_sort(stats, ev_thresh=args.ev_thresh, hp_thresh=args.hp_thresh)
    print_table(candidates)


if __name__ == "__main__":
    main()