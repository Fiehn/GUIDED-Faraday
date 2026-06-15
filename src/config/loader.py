from __future__ import annotations

from pathlib import Path

import yaml

from config.schema import ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    if "base" in raw:
        base_ref = raw.pop("base")
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        with open(base_path, encoding="utf-8") as f:
            base_raw: dict = yaml.safe_load(f) or {}
        raw = _deep_merge(base_raw, raw)

    return ExperimentConfig.model_validate(raw)