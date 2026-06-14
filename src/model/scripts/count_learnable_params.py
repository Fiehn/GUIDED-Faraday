"""Count model parameters for a given YAML configuration.

This script instantiates the architecture described by a config file and prints
the number of total, trainable, and frozen parameters. Model construction is
delegated entirely to the production factory so the reported count always
matches what gets trained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

for _p in (SRC_DIR, ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(config_path: str):
    """Load a YAML config (with base: override composition) and add factory aliases."""
    from pathlib import Path as _Path
    path = _Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "base" in raw:
        base_ref = raw.pop("base")
        base_path = _Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_ref
        with open(base_path, "r", encoding="utf-8") as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(base_raw, raw)

    cfg = _to_namespace(raw)
    cfg._config_path = os.path.abspath(config_path)

    if not hasattr(cfg, "data"):
        raise ValueError("Config must contain a top-level 'data' section")
    if not hasattr(cfg, "model"):
        raise ValueError("Config must contain a top-level 'model' section")
    if not hasattr(cfg.data, "window"):
        raise ValueError("Config must contain 'data.window' section")

    # Aliases expected by create_model_from_config
    cfg.window = cfg.data.window
    cfg.columns = getattr(cfg.data, "columns", SimpleNamespace(value_cols=[]))
    cfg.model_type = str(getattr(cfg.model, "type", "vae")).lower()

    return cfg


def _default_config_path() -> str:
    return os.path.join(ROOT_DIR, "config", "config.yaml")


# ---------------------------------------------------------------------------
# Core counting helpers
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module):
    """Return (total, trainable, frozen) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def module_trainable_breakdown(model: torch.nn.Module):
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        if total == 0:
            continue
        rows.append({"module": name, "total": total, "trainable": trainable})
    rows.sort(key=lambda r: r["trainable"], reverse=True)
    return rows


def count_from_config(config) -> tuple[int, int, int]:
    """Instantiate model via production factory and return (total, trainable, frozen).

    ``config`` must be a namespace accepted by ``create_model_from_config``
    (i.e. it already has top-level ``window`` and ``columns`` aliases set).
    """
    from model.base.factory import create_model_from_config  # noqa: PLC0415
    model = create_model_from_config(config, device="cpu")
    return count_parameters(model)


# ---------------------------------------------------------------------------
# Reference budget helpers
# ---------------------------------------------------------------------------

def load_reference_budget(checkpoint_path: str) -> int:
    """Return trainable parameter count stored in a checkpoint.

    Falls back to summing all state-dict tensors if the checkpoint predates the
    stored-count feature; this proxy is slightly inflated (includes buffers) and
    emits a warning.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "trainable_params" in ckpt:
        return int(ckpt["trainable_params"])

    print(
        "Warning: checkpoint predates stored-count feature; "
        "falling back to state_dict tensor sum (includes buffers, slightly inflated).",
        file=sys.stderr,
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    return int(sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)))


def parity_check(new_count: int, ref_count: int, tolerance: float = 0.10) -> dict:
    """Return a verdict dict for whether new_count is within ±tolerance of ref_count."""
    abs_diff = abs(new_count - ref_count)
    pct_diff = abs_diff / ref_count if ref_count > 0 else float("inf")
    return {
        "pass": pct_diff <= tolerance,
        "new": int(new_count),
        "ref": int(ref_count),
        "abs_diff": int(abs_diff),
        "pct_diff": float(pct_diff),
    }


def check_parity(config, reference_path: str, tolerance: float = 0.10) -> dict:
    """Count new model from config and compare against a saved reference checkpoint."""
    _, new_trainable, _ = count_from_config(config)
    ref_budget = load_reference_budget(reference_path)
    return parity_check(new_trainable, ref_budget, tolerance)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _fmt_int(value: int) -> str:
    return f"{int(value):,}"


def _resolve_primary(folder: str, primary: str | None) -> str:
    """Resolve --primary to an absolute path.

    If *primary* is a bare filename (no directory component) it is looked up
    inside *folder*.  Otherwise it is used as-is.
    """
    if primary is None:
        raise ValueError("--primary is required when --folder is used")
    from pathlib import Path as _Path
    p = _Path(primary)
    if not p.is_absolute() and p.parent == _Path("."):
        p = _Path(folder) / p
    resolved = p.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Primary config not found: {resolved}")
    return str(resolved)


def _scan_folder(folder: str, exclude_path: str) -> list[str]:
    """Return sorted list of *.yaml paths in *folder*, excluding base.yaml and *exclude_path*."""
    from pathlib import Path as _Path
    exclude_abs = os.path.abspath(exclude_path)
    paths = []
    for p in sorted(_Path(folder).glob("*.yaml")):
        if p.name.lower() == "base.yaml":
            continue
        if os.path.abspath(str(p)) == exclude_abs:
            continue
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# Folder comparison mode
# ---------------------------------------------------------------------------

def _run_folder_comparison(folder: str, primary_path: str, tolerance: float, as_json: bool) -> None:
    """Count params for every config in *folder* and compare against *primary_path*."""
    from pathlib import Path as _Path

    primary_config = _load_config(primary_path)
    _, primary_trainable, _ = count_from_config(primary_config)
    primary_name = _Path(primary_path).name

    candidate_paths = _scan_folder(folder, primary_path)

    rows = []
    for cfg_path in candidate_paths:
        name = _Path(cfg_path).name
        try:
            cfg = _load_config(cfg_path)
            _, trainable, _ = count_from_config(cfg)
            verdict = parity_check(trainable, primary_trainable, tolerance)
            rows.append({
                "config": name,
                "trainable": trainable,
                "pass": verdict["pass"],
                "abs_diff": verdict["abs_diff"],
                "pct_diff": verdict["pct_diff"],
                "error": None,
            })
        except Exception as exc:
            rows.append({
                "config": name,
                "trainable": None,
                "pass": None,
                "abs_diff": None,
                "pct_diff": None,
                "error": str(exc),
            })

    if as_json:
        output = {
            "folder": os.path.abspath(folder),
            "primary": {"config": primary_name, "trainable_params": primary_trainable},
            "tolerance": tolerance,
            "results": rows,
        }
        print(json.dumps(output, indent=2))
    else:
        _print_folder_table(folder, primary_name, primary_trainable, tolerance, rows)

    n_fail  = sum(1 for r in rows if r["pass"] is False)
    n_error = sum(1 for r in rows if r["pass"] is None)

    if n_fail > 0 or n_error > 0:
        sys.exit(1)


def _print_folder_table(
    folder: str,
    primary_name: str,
    primary_trainable: int,
    tolerance: float,
    rows: list[dict],
) -> None:
    COL_NAME = 32
    COL_PARAMS = 14
    COL_DIFF = 13
    COL_PCT = 9
    COL_VERDICT = 7

    header_name = "CONFIG"
    header = (
        f"  {header_name:<{COL_NAME}}"
        f"{'TRAINABLE':>{COL_PARAMS}}"
        f"{'DIFF':>{COL_DIFF}}"
        f"{'%DIFF':>{COL_PCT}}"
        f"  {'VERDICT':<{COL_VERDICT}}"
    )
    separator = "  " + "-" * (COL_NAME + COL_PARAMS + COL_DIFF + COL_PCT + COL_VERDICT + 4)

    print("=== Parameter Budget Comparison ===")
    print(f"Folder:    {os.path.abspath(folder)}")
    print(f"Primary:   {primary_name}  ->  {_fmt_int(primary_trainable)} trainable params")
    print(f"Tolerance: +/-{tolerance:.0%}")
    print("")
    print(header)
    print(separator)

    n_pass = n_fail = n_error = 0
    for r in rows:
        if r["error"] is not None:
            n_error += 1
            print(f"  {'[ERROR] ' + r['config']:<{COL_NAME}}  (skipped: {r['error'][:50]})")
            continue

        diff_int = r["trainable"] - primary_trainable
        diff_str = f"{diff_int:+,}"
        signed_pct = diff_int / primary_trainable if primary_trainable > 0 else float("inf")
        pct_str = f"{signed_pct:+.1%}"
        verdict_str = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1

        print(
            f"  {r['config']:<{COL_NAME}}"
            f"{_fmt_int(r['trainable']):>{COL_PARAMS}}"
            f"{diff_str:>{COL_DIFF}}"
            f"{pct_str:>{COL_PCT}}"
            f"  {verdict_str:<{COL_VERDICT}}"
        )

    print(separator)
    parts = []
    if n_pass:  parts.append(f"{n_pass} PASS")
    if n_fail:  parts.append(f"{n_fail} FAIL")
    if n_error: parts.append(f"{n_error} ERROR")
    print("  " + "  ".join(parts) if parts else "  (no configs found)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Count model parameters from YAML config, with optional parity check."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a single YAML config file (single-config mode)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder containing YAML configs to compare (folder mode)",
    )
    parser.add_argument(
        "--primary",
        type=str,
        default=None,
        help="Primary/reference config filename (required with --folder); "
             "bare filename is resolved relative to --folder",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a reference .pt checkpoint; prints a parity verdict against it",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Parity tolerance as a fraction (default 0.10 = +/-10%%)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help="Print per-module trainable parameter breakdown (single-config mode only)",
    )
    parser.add_argument(
        "--top-modules",
        type=int,
        default=25,
        help="Number of modules to show in --detailed output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print output as JSON",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ folder mode
    if args.folder:
        primary_path = _resolve_primary(args.folder, args.primary)
        _run_folder_comparison(
            folder=args.folder,
            primary_path=primary_path,
            tolerance=args.tolerance,
            as_json=args.json,
        )
        return

    # ------------------------------------------------------------------ single-config mode
    config_path = os.path.abspath(args.config or _default_config_path())
    config = _load_config(config_path)
    total, trainable, frozen = count_from_config(config)

    from model.base.factory import create_model_from_config  # noqa: PLC0415
    model = create_model_from_config(config, device="cpu")

    payload = {
        "config_path": config_path,
        "model_class": model.__class__.__name__,
        "metadata": {"model_type": getattr(config, "model_type", "unknown")},
        "parameters": {"total": int(total), "trainable": int(trainable), "frozen": int(frozen)},
    }

    if args.detailed:
        payload["module_breakdown"] = module_trainable_breakdown(model)

    if args.reference:
        payload["parity"] = check_parity(config, args.reference, tolerance=args.tolerance)

    if args.json:
        print(json.dumps(payload, indent=2))
        if args.reference and not payload["parity"]["pass"]:
            sys.exit(1)
        return

    print("=== Model Parameter Count ===")
    print(f"Config:         {config_path}")
    print(f"Model class:    {model.__class__.__name__}")
    print(f"Model type:     {config.model_type}")
    print("")
    print(f"Total params:      {_fmt_int(total)}")
    print(f"Trainable params:  {_fmt_int(trainable)}")
    print(f"Frozen params:     {_fmt_int(frozen)}")

    if args.detailed:
        rows = payload.get("module_breakdown", [])[: max(1, int(args.top_modules))]
        print("")
        print(f"Top {len(rows)} modules by trainable parameters")
        for row in rows:
            print(
                f"  - {row['module']}: trainable={_fmt_int(row['trainable'])}, "
                f"total={_fmt_int(row['total'])}"
            )

    if args.reference:
        v = payload["parity"]
        verdict_str = "PASS" if v["pass"] else "FAIL"
        print("")
        print("=== Parameter Budget Parity ===")
        print(f"Reference:         {args.reference}")
        print(f"New model:         {_fmt_int(v['new'])} trainable params")
        print(f"Reference budget:  {_fmt_int(v['ref'])} trainable params")
        print(f"Absolute diff:     {_fmt_int(v['abs_diff'])}")
        print(f"Percentage diff:   {v['pct_diff']:.1%}")
        print(f"Tolerance:         +/-{args.tolerance:.0%}")
        print(f"Verdict:           {verdict_str}")
        if not v["pass"]:
            sys.exit(1)


if __name__ == "__main__":
    main()