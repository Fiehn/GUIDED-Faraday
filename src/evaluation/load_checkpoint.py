import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model.set import SetGuideVAE
from src.data.dataloader_phased import WindowSpec, PhasedReplayDataloaders
from src.visualization.writer import Writer
from src.visualization.visualizer import Visualizer
from src.visualization.plot_styling import PlotStyle
from evaluation.eval_framework import PhasedEvaluationSuite


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_config(config_path: str) -> SimpleNamespace:
    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = ROOT_DIR / cfg_path
    cfg_path = cfg_path.resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _to_namespace(raw)
    cfg._config_path = str(cfg_path)

    if not hasattr(cfg, "data"):
        raise ValueError("Config must contain a top-level 'data' section")

    cfg.window = cfg.data.window
    cfg.dataloader = cfg.data.dataloader
    cfg.data_dir = cfg.data.data_dir
    cfg.data_files = list(getattr(cfg.data, "data_files", []) or [])

    if hasattr(cfg, "normalization"):
        cfg.normalizer = cfg.normalization
    elif not hasattr(cfg, "normalizer"):
        cfg.normalizer = SimpleNamespace(method="log1p")

    return cfg


def _resolve_checkpoint_file(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()

    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = sorted(path.glob("*_final.pt"))
    if candidates:
        return candidates[-1]

    candidates = sorted(path.glob("*.pt"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"No .pt checkpoint found in directory: {path}")


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _fix_and_preload_conditioners(conditioner, project_root: Path) -> None:
    if conditioner is None or not hasattr(conditioner, "condition_items"):
        return

    for tag, item in conditioner.condition_items.items():
        if not (hasattr(item, "path") and hasattr(item, "load")):
            continue

        p = Path(str(item.path))
        if not p.is_absolute():
            item.path = str((project_root / str(p).lstrip("./")).resolve())

        try:
            item.load(item.path)
            print(f"Loaded {item.__class__.__name__} for '{tag}' from: {item.path}")
        except Exception as exc:
            print(f"Warning: could not preload {item.__class__.__name__} ({item.path}): {exc}")


def _resolve_parquet_paths(config: SimpleNamespace, project_root: Path, max_files: int | None, seed: int):
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    data_dir = data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    parquet_paths = []
    configured_files = list(getattr(config, "data_files", []) or [])

    if configured_files:
        for f in configured_files:
            p = Path(f)
            if not p.is_absolute():
                p = data_dir / p
            p = p.resolve()
            if p.exists() and p.suffix == ".parquet":
                parquet_paths.append(str(p))
    else:
        parquet_paths = sorted(str(p) for p in data_dir.rglob("*.parquet"))

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for phased mode in: {data_dir}")

    if max_files is not None and max_files > 0 and max_files < len(parquet_paths):
        rng = random.Random(seed)
        parquet_paths = sorted(rng.sample(parquet_paths, max_files))
        print(f"Using sampled parquet files: {len(parquet_paths)}")
    else:
        print(f"Using parquet files: {len(parquet_paths)}")

    return parquet_paths


def _prepare_phased_window(config: SimpleNamespace, model: SetGuideVAE, phased_cfg) -> tuple[WindowSpec, bool, float]:
    seq_len = int(getattr(model, "sequence_length", config.window.input_len))
    hourly_aggregation = bool(getattr(config.window, "aggregated", False))
    aggregation_energy_unit = str(getattr(phased_cfg, "aggregation_energy_unit", "wh")).lower()
    if aggregation_energy_unit not in {"wh", "kwh"}:
        print(f"Warning: unknown data.phased.aggregation_energy_unit='{aggregation_energy_unit}'. Falling back to 'wh'.")
        aggregation_energy_unit = "wh"

    if seq_len == 24 and not hourly_aggregation:
        hourly_aggregation = True
        print("Overriding hourly_aggregation=True to match checkpoint sequence_length=24.")
    elif seq_len == 96 and hourly_aggregation:
        hourly_aggregation = False
        print("Overriding hourly_aggregation=False to match checkpoint sequence_length=96.")

    config.window.input_len = seq_len
    config.window.stride = seq_len
    config.window.aggregated = hourly_aggregation

    aggregation_divisor = 1000.0 if hourly_aggregation and aggregation_energy_unit == "kwh" else 1.0
    if hourly_aggregation:
        print(f"Hourly aggregation unit: {aggregation_energy_unit} (divisor={aggregation_divisor:g})")

    return WindowSpec(input_len=seq_len, stride=seq_len), hourly_aggregation, aggregation_divisor


def _build_eval_loader(
    config: SimpleNamespace,
    model: SetGuideVAE,
    normalizer,
    project_root: Path,
    max_files: int | None,
    seed: int,
    eval_device: torch.device,
):
    phased_cfg = config.data.phased
    parquet_paths = _resolve_parquet_paths(config, project_root, max_files, seed)
    window, hourly_aggregation, aggregation_divisor = _prepare_phased_window(config, model, phased_cfg)

    batch_size = int(config.dataloader.batch_size)
    phased_workers = int(getattr(phased_cfg, "num_workers", config.dataloader.num_workers))
    eval_batches = int(getattr(phased_cfg, "eval_batches", 5))
    eval_cache_on_gpu = bool(
        getattr(phased_cfg, "eval_cache_on_gpu", str(eval_device) != "cpu")
    )
    eval_cache_device = eval_device if eval_cache_on_gpu and str(eval_device) != "cpu" else None
    cache_scan_batches = int(getattr(phased_cfg, "cache_scan_batches", 0)) or None

    replay = PhasedReplayDataloaders(
        sm_files=parquet_paths,
        window_spec=window,
        batch_size=batch_size,
        replay_buffer_capacity=int(phased_cfg.replay_buffer_capacity),
        min_buffer_size=int(phased_cfg.min_buffer_size),
        normalizer=normalizer,
        num_workers=phased_workers,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        eval_cache_device=eval_cache_device,
        eval_cache_batches=eval_batches,
        cache_scan_batches=cache_scan_batches,
    )

    replay.refresh_eval_cache(eval_batches)
    test_loader = replay.get_eval_loader()
    if test_loader is None:
        raise RuntimeError("Could not build a cached evaluation loader.")
    return test_loader


def _get_visualize_config(config: SimpleNamespace):
    vis = getattr(config, "visualize", SimpleNamespace())
    return SimpleNamespace(
        num_samples=int(getattr(vis, "num_samples", 5)),
        num_recon_per=int(getattr(vis, "num_recon_per", 5)),
        num_synthetic=int(getattr(vis, "num_synthetic", 40)),
        num_originals=int(getattr(vis, "num_originals", 4)),
        num_synthetic_per=int(getattr(vis, "num_synthetic_per", 10)),
    )


def _run_evaluation_and_visualization(
    *,
    model,
    test_loader,
    normalizer,
    conditioner,
    device: torch.device,
    log_path: Path,
    config: SimpleNamespace,
):
    log_path.mkdir(parents=True, exist_ok=True)
    model.eval()

    evaluator = PhasedEvaluationSuite(
        model,
        test_loader,
        normalizer=normalizer,
        conditioner=conditioner,
        device=str(device),
        window_size=config.window.input_len,
        output_path=str(log_path),
    )
    evaluator.evaluate()

    writer = Writer(log_dir=str(log_path))
    vis_cfg = _get_visualize_config(config)
    visualizer = Visualizer(
        model=model,
        writer=writer,
        plot_style=PlotStyle(),
        log_dir=str(log_path),
        device=device,
        normalizer=normalizer,
        conditioner=conditioner,
    )
    visualizer.visualize(
        test_loader=test_loader,
        num_samples=vis_cfg.num_samples,
        num_recon_per=vis_cfg.num_recon_per,
        num_synthetic=vis_cfg.num_synthetic,
        num_originals=vis_cfg.num_originals,
        num_synthetic_per=vis_cfg.num_synthetic_per,
    )


def run_checkpoint_evaluation(
    checkpoint_path: str,
    config_path: str,
    device: str = "auto",
    max_files: int | None = None,
):
    config = _load_config(config_path)
    data_mode = str(getattr(config.data, "mode", "")).lower()
    phased_mode = data_mode in {"phased", "phase", "parquet", "phased_parquet"}
    if not phased_mode:
        raise ValueError(
            f"This script currently supports phased parquet mode only. Found mode='{data_mode}'."
        )

    ckpt_file = _resolve_checkpoint_file(checkpoint_path)
    model_dir = ckpt_file.parent
    selected_device = _select_device(device)

    print(f"Config: {config._config_path}")
    print(f"Checkpoint: {ckpt_file}")
    print(f"Device: {selected_device}")

    model, normalizer_ckpt = SetGuideVAE.load_from_checkpoint(str(ckpt_file), device=selected_device)
    model.eval()

    conditioner = getattr(model, "conditioner", None)
    _fix_and_preload_conditioners(conditioner, ROOT_DIR)

    if normalizer_ckpt is None:
        raise RuntimeError(
            "Checkpoint did not return a normalizer from load_from_checkpoint(). "
            "To avoid normalization mismatches, this script requires a checkpoint-embedded normalizer."
        )

    normalizer = normalizer_ckpt
    print(f"Using normalizer loaded from checkpoint (method={getattr(normalizer, 'method', 'unknown')}).")

    test_loader = _build_eval_loader(
        config=config,
        model=model,
        normalizer=normalizer,
        project_root=ROOT_DIR,
        max_files=max_files,
        seed=42,
        eval_device=selected_device,
    )

    _run_evaluation_and_visualization(
        model=model,
        test_loader=test_loader,
        normalizer=normalizer,
        conditioner=conditioner,
        device=selected_device,
        log_path=model_dir,
        config=config,
    )

    print("Done. Metrics + visualizations saved under:")
    print(model_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a SetGuideVAE checkpoint and run evaluation + visualizations in one command."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "models" / "SetGuideVAE_DictionaryCoder_20260325_222832"),
        help="Path to checkpoint file (.pt) or model directory containing .pt files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "config" / "config.yaml"),
        help="Path to training/evaluation config YAML.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of parquet files used for eval (random sample).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_checkpoint_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        max_files=args.max_files,
    )