from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataloader_phased import StreamingParquetBuffer, WindowSpec
from data.normalizer import Normalizer
from torch.utils.data import DataLoader


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_numeric_stem(path: Path) -> bool:
    stem = path.stem.replace("sm_", "")
    return stem.isdigit()


def _collect_parquet_files(root: Path, max_files: int | None) -> list[str]:
    if not root.exists():
        return []

    files: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".parquet"):
                continue
            path = Path(dirpath) / filename
            if not _is_numeric_stem(path):
                continue
            files.append(str(path))
            if max_files is not None and len(files) >= max_files:
                return files
    return files


def _load_yaml_value(path: Path, keys: list[str]) -> object | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception:
        return None
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _resolve_phased_config_path() -> Path | None:
    config_env = os.getenv("PHASED_CONFIG_PATH")
    if config_env:
        return Path(config_env)
    default_path = ROOT_DIR / "config" / "tests" / "phased" / "set_guide.yaml"
    return default_path if default_path.exists() else None


def _get_config_max_hour_value() -> tuple[float | None, Path | None]:
    config_path = _resolve_phased_config_path()
    if config_path is None:
        return None, None
    value = _coerce_float(_load_yaml_value(config_path, ["data", "max_hour_value"]))
    return value, config_path


def _percentile_stats(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    if values.size == 0:
        return {"min": float("nan"), **{f"p{p}": float("nan") for p in percentiles}}
    return {
        "min": float(np.min(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in percentiles},
    }


def _summarize_windows(x_phase_first: torch.Tensor) -> dict[str, object]:
    x_np = x_phase_first.detach().cpu().numpy()
    if x_np.ndim != 3:
        raise ValueError(f"Expected [N, F, L] windows, got shape {x_np.shape}")

    percentiles = [1, 5, 50]
    phase_stats = []
    for phase_idx in range(x_np.shape[1]):
        values = x_np[:, phase_idx, :].reshape(-1)
        values = values[np.isfinite(values)]
        phase_stats.append(_percentile_stats(values, percentiles))

    agg_values = x_np.sum(axis=1).reshape(-1)
    agg_values = agg_values[np.isfinite(agg_values)]
    agg_stats = _percentile_stats(agg_values, percentiles)

    return {"phase": phase_stats, "aggregate": agg_stats}


def _format_stats(stats: dict[str, object]) -> str:
    phase_stats = stats.get("phase", [])
    agg_stats = stats.get("aggregate", {})
    phase_lines = []
    for idx, phase in enumerate(phase_stats):
        phase_lines.append(
            f"phase{idx + 1}={{min:{phase['min']:.4f},p1:{phase['p1']:.4f},p5:{phase['p5']:.4f},p50:{phase['p50']:.4f}}}"
        )
    agg_line = (
        f"aggregate={{min:{agg_stats['min']:.4f},p1:{agg_stats['p1']:.4f},p5:{agg_stats['p5']:.4f},p50:{agg_stats['p50']:.4f}}}"
    )
    return "; ".join(phase_lines + [agg_line])


def _load_raw_windows(
    parquet_files: list[str],
    *,
    window_len: int,
    window_stride: int,
    hourly_aggregation: bool,
    aggregation_divisor: float,
    nan_as_zero: bool,
    max_windows: int,
    max_hour_value: float | None,
) -> torch.Tensor:
    window_spec = WindowSpec(input_len=window_len, stride=window_stride)
    dataset = StreamingParquetBuffer(
        sm_files=parquet_files,
        replay_buffer_capacity=max(1000, max_windows),
        min_buffer_size=0,
        window_spec=window_spec,
        normalizer=None,
        num_workers=0,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        output_aggregate_stream=False,
        sample_random=False,
        nan_as_zero=nan_as_zero,
        phased_mode=True,
        max_hour_value=max_hour_value,
    )

    loader = DataLoader(dataset, batch_size=min(32, max_windows), num_workers=0)
    batches = []
    seen = 0
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            continue
        x, _, _, _ = batch
        if x is None:
            continue
        batches.append(x.cpu())
        seen += x.shape[0]
        if seen >= max_windows:
            break

    if not batches:
        return torch.empty((0, 3, window_len), dtype=torch.float32)

    x_all = torch.cat(batches, dim=0)
    return x_all[:max_windows]


def _load_first_batch(
    parquet_files: list[str],
    *,
    window_len: int,
    window_stride: int,
    hourly_aggregation: bool,
    aggregation_divisor: float,
    nan_as_zero: bool,
    max_hour_value: float | None,
    output_aggregate_stream: bool,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    window_spec = WindowSpec(input_len=window_len, stride=window_stride)
    dataset = StreamingParquetBuffer(
        sm_files=parquet_files,
        replay_buffer_capacity=max(1000, batch_size),
        min_buffer_size=0,
        window_spec=window_spec,
        normalizer=None,
        num_workers=0,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        output_aggregate_stream=output_aggregate_stream,
        sample_random=False,
        nan_as_zero=nan_as_zero,
        phased_mode=True,
        max_hour_value=max_hour_value,
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            continue
        x, meter_id, timestamps, agg = batch
        return x, meter_id, timestamps, agg
    return None


def test_phased_raw_floor_and_round_trip_smoke():
    default_data_dir = r"E:\Data\processed_data\second_batch\individual_sm_datasets\raw_profiles"
    data_dir = Path(os.getenv("PHASED_DATA_DIR", default_data_dir))
    if not data_dir.exists():
        pytest.skip("PHASED_DATA_DIR not available; skipping phased data smoke test.")

    print("[phased-smoke] data_dir=", data_dir)

    max_files = _env_int("PHASED_MAX_FILES", 5)
    if max_files <= 0:
        max_files = None
    parquet_files = _collect_parquet_files(data_dir, max_files)
    if not parquet_files:
        pytest.skip("No phased parquet files found; skipping phased data smoke test.")

    print("[phased-smoke] parquet_files=", len(parquet_files))

    window_len = _env_int("PHASED_WINDOW_LEN", 24)
    window_stride = _env_int("PHASED_WINDOW_STRIDE", 24)
    if window_len <= 0 or window_stride <= 0:
        pytest.skip("Invalid window length/stride; skipping phased data smoke test.")
    hourly_aggregation = _env_bool("PHASED_HOURLY_AGG", True)
    aggregation_divisor = _env_float("PHASED_AGG_DIVISOR", 1000.0) or 1.0
    nan_as_zero = _env_bool("PHASED_NAN_AS_ZERO", False)
    config_max_hour_value, config_path = _get_config_max_hour_value()
    max_hour_value = _env_float("PHASED_MAX_HOUR_VALUE", config_max_hour_value)
    max_windows = _env_int("PHASED_MAX_WINDOWS", 128)
    if max_windows <= 0:
        pytest.skip("PHASED_MAX_WINDOWS is non-positive; skipping phased data smoke test.")

    max_hour_source = "env" if os.getenv("PHASED_MAX_HOUR_VALUE") else "config"
    if max_hour_source == "config" and config_path is None:
        max_hour_source = "default"
    print(
        "[phased-smoke] window_len=", window_len,
        "window_stride=", window_stride,
        "hourly_aggregation=", hourly_aggregation,
        "aggregation_divisor=", aggregation_divisor,
        "nan_as_zero=", nan_as_zero,
        "max_hour_value=", max_hour_value,
        "max_hour_source=", max_hour_source,
        "max_windows=", max_windows,
    )

    x_raw = _load_raw_windows(
        parquet_files,
        window_len=window_len,
        window_stride=window_stride,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        nan_as_zero=nan_as_zero,
        max_windows=max_windows,
        max_hour_value=max_hour_value,
    )
    if x_raw.numel() == 0:
        pytest.skip("No phased windows produced; check data or window settings.")

    raw_stats = _summarize_windows(x_raw)
    print("[phased-smoke] raw_windows=", tuple(x_raw.shape))
    print("[phased-smoke] raw_stats=", _format_stats(raw_stats))

    normalizer_path = os.getenv("PHASED_NORMALIZER_PATH")
    if normalizer_path and Path(normalizer_path).exists():
        normalizer = Normalizer.load(normalizer_path, device="cpu")
        print("[phased-smoke] normalizer=loaded", normalizer_path)
    else:
        normalizer = Normalizer(method="log1p_phase")
        normalizer.fit_normalizer_on_files(
            filenames=parquet_files,
            window_spec=WindowSpec(input_len=window_len, stride=window_stride),
            phased=True,
            aggregation_divisor=aggregation_divisor,
            hourly_aggregation=hourly_aggregation,
            nan_as_zero=nan_as_zero,
            device="cpu",
        )
        print("[phased-smoke] normalizer=recomputed")

    x_norm = normalizer.transform(x_raw)
    x_roundtrip = normalizer.inverse_transform(x_norm)

    max_abs_error = (x_roundtrip - x_raw).abs().max().item()
    mean_abs_error = (x_roundtrip - x_raw).abs().mean().item()

    roundtrip_stats = _summarize_windows(x_roundtrip)
    atol = _env_float("PHASED_ROUNDTRIP_ATOL", 1e-3) or 1e-3
    print("[phased-smoke] roundtrip_stats=", _format_stats(roundtrip_stats))
    print(
        "[phased-smoke] max_abs_error=",
        f"{max_abs_error:.6f}",
        "mean_abs_error=",
        f"{mean_abs_error:.6f}",
        "atol=",
        f"{atol:.6f}",
    )
    assert max_abs_error <= atol, (
        "Round-trip error exceeded tolerance. "
        f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}, "
        f"raw_stats={_format_stats(raw_stats)}, roundtrip_stats={_format_stats(roundtrip_stats)}"
    )


def test_phased_dataloader_contract_smoke():
    default_data_dir = r"E:\Data\processed_data\second_batch\individual_sm_datasets\raw_profiles"
    data_dir = Path(os.getenv("PHASED_DATA_DIR", default_data_dir))
    if not data_dir.exists():
        pytest.skip("PHASED_DATA_DIR not available; skipping phased dataloader test.")

    print("[phased-dataloader] data_dir=", data_dir)

    max_files = _env_int("PHASED_MAX_FILES", 5)
    if max_files <= 0:
        max_files = None
    parquet_files = _collect_parquet_files(data_dir, max_files)
    if not parquet_files:
        pytest.skip("No phased parquet files found; skipping phased dataloader test.")

    print("[phased-dataloader] parquet_files=", len(parquet_files))

    window_len = _env_int("PHASED_WINDOW_LEN", 24)
    window_stride = _env_int("PHASED_WINDOW_STRIDE", 24)
    if window_len <= 0 or window_stride <= 0:
        pytest.skip("Invalid window length/stride; skipping phased dataloader test.")
    hourly_aggregation = _env_bool("PHASED_HOURLY_AGG", True)
    aggregation_divisor = _env_float("PHASED_AGG_DIVISOR", 1000.0) or 1.0
    nan_as_zero = _env_bool("PHASED_NAN_AS_ZERO", False)
    config_max_hour_value, config_path = _get_config_max_hour_value()
    max_hour_value = _env_float("PHASED_MAX_HOUR_VALUE", config_max_hour_value)
    batch_size = _env_int("PHASED_DATALOADER_BATCH", 8)
    if batch_size <= 0:
        pytest.skip("PHASED_DATALOADER_BATCH is non-positive; skipping phased dataloader test.")

    max_hour_source = "env" if os.getenv("PHASED_MAX_HOUR_VALUE") else "config"
    if max_hour_source == "config" and config_path is None:
        max_hour_source = "default"
    print(
        "[phased-dataloader] window_len=", window_len,
        "window_stride=", window_stride,
        "hourly_aggregation=", hourly_aggregation,
        "aggregation_divisor=", aggregation_divisor,
        "nan_as_zero=", nan_as_zero,
        "max_hour_value=", max_hour_value,
        "max_hour_source=", max_hour_source,
        "batch_size=", batch_size,
    )

    batch = _load_first_batch(
        parquet_files,
        window_len=window_len,
        window_stride=window_stride,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        nan_as_zero=nan_as_zero,
        max_hour_value=max_hour_value,
        output_aggregate_stream=True,
        batch_size=batch_size,
    )
    if batch is None:
        pytest.skip("No batches produced by dataloader; check data or window settings.")

    x, meter_id, timestamps, agg = batch
    print("[phased-dataloader] x_shape=", tuple(x.shape))
    print("[phased-dataloader] meter_id_shape=", tuple(meter_id.shape))
    print("[phased-dataloader] timestamps_shape=", tuple(timestamps.shape))
    print("[phased-dataloader] agg_shape=", tuple(agg.shape))

    assert x.dim() == 3, "Expected x as [B, F, L]."
    assert x.shape[1] == 3, "Expected 3 phase features in x."
    assert x.shape[2] == window_len, "Expected window_len in x."
    assert meter_id.dim() == 1 and meter_id.shape[0] == x.shape[0], "meter_id shape mismatch."
    assert timestamps.dim() == 1 and timestamps.shape[0] == x.shape[0], "timestamps shape mismatch."
    assert agg.dim() == 3 and agg.shape[1] == 1 and agg.shape[2] == window_len, "agg shape mismatch."

    agg_expected = x.sum(dim=1, keepdim=True)
    max_agg_error = (agg - agg_expected).abs().max().item()
    print("[phased-dataloader] max_agg_error=", f"{max_agg_error:.6f}")
    assert max_agg_error <= 1e-5, "Aggregate stream does not match phase sum."
 
if __name__ == "__main__":
    pytest.main([str(Path(__file__).resolve()), "-s"])
