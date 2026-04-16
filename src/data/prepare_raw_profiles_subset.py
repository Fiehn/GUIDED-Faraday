
#!/usr/bin/env python3
"""Prepare a raw_profiles parquet subset from a meter list CSV.

Usage example:
  python src/data/prepare_raw_profiles_subset.py \
    --meters-csv data/meters_from_bachelor.csv \
    --source-dir data/raw_profiles \
    --output-dir data/raw_profiles_bachelor6000 \
    --mode symlink \
    --clear-output
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path


def _read_meter_ids(csv_path: Path) -> list[int]:
    meter_ids: list[int] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        # Prefer known column name from the project, fallback to first column.
        if "meter_nr" in reader.fieldnames:
            meter_col = "meter_nr"
        elif "meter_number" in reader.fieldnames:
            meter_col = "meter_number"
        else:
            meter_col = reader.fieldnames[0]

        for row in reader:
            raw = (row.get(meter_col) or "").strip()
            if not raw:
                continue
            try:
                meter_ids.append(int(raw))
            except ValueError:
                continue

    # Keep deterministic unique ordering.
    return sorted(set(meter_ids))


def _index_parquet_files(source_dir: Path) -> dict[int, Path]:
    file_by_meter: dict[int, Path] = {}
    pattern = re.compile(r"^sm_(\d+)\.parquet$")

    for p in source_dir.rglob("*.parquet"):
        match = pattern.match(p.name)
        if not match:
            continue
        meter_id = int(match.group(1))
        # If duplicates exist, first one wins deterministically by sort order.
        if meter_id not in file_by_meter:
            file_by_meter[meter_id] = p

    return file_by_meter


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw_profiles parquet subset for selected meter IDs")
    parser.add_argument("--meters-csv", required=True, help="Path to CSV with meter IDs (e.g. data/meters_from_bachelor.csv)")
    parser.add_argument("--source-dir", required=True, help="Source raw_profiles directory with sm_<id>.parquet files")
    parser.add_argument("--output-dir", required=True, help="Output subset directory")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="How to materialize subset files")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory before writing")
    args = parser.parse_args()

    meters_csv = Path(args.meters_csv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not meters_csv.exists():
        raise FileNotFoundError(f"Meter CSV not found: {meters_csv}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    meter_ids = _read_meter_ids(meters_csv)
    if not meter_ids:
        raise ValueError("No meter IDs found in CSV")

    file_by_meter = _index_parquet_files(source_dir)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    missing: list[int] = []

    for meter_id in meter_ids:
        src = file_by_meter.get(meter_id)
        if src is None:
            missing.append(meter_id)
            continue
        dst = output_dir / src.name
        _link_or_copy(src, dst, args.mode)
        found += 1

    manifest_path = output_dir / "subset_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(f"meters_requested={len(meter_ids)}\n")
        f.write(f"files_found={found}\n")
        f.write(f"files_missing={len(missing)}\n")
        if missing:
            f.write("missing_meter_ids=\n")
            for m in missing:
                f.write(f"{m}\n")

    print(f"Requested meters: {len(meter_ids)}")
    print(f"Subset files created: {found}")
    print(f"Missing meters: {len(missing)}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

