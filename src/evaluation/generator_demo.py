import argparse
import os
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.evaluation.load_model import load_model
from src.model.factory import _load_train_config
from src.model.generators import SyntheticGenerator
from src.visualization.plot_styling import PlotStyle
from src.data.dataloader_phased import StreamingParquetBuffer, WindowSpec
from src.data.dataloader import load_csv_and_cache_as_parquet


def _resolve_config_path(model_path: str) -> str:
    preferred = os.path.join(model_path, "training_config_used.yaml")
    if os.path.exists(preferred):
        return preferred
    for name in os.listdir(model_path):
        if name.endswith(".yaml"):
            return os.path.join(model_path, name)
    return os.path.join("config", "config.yaml")


def _plot_week(series: np.ndarray, seq_len: int, output_path: str, title: str):
    style = PlotStyle()
    total_steps, num_features = series.shape
    hours = np.arange(total_steps)

    fig, ax = plt.subplots(figsize=style.get_figsize())
    if num_features > 1:
        agg = series.sum(axis=1)
        ax.plot(hours, agg, label="Aggregated", color=style.color_original, linewidth=2.5)
        for idx in range(num_features):
            ax.plot(
                hours,
                series[:, idx],
                label=f"Phase {idx + 1}",
                color=style.get_color(idx),
                alpha=0.8,
                linewidth=style.linewidth,
            )
    else:
        ax.plot(hours, series[:, 0], label="Generated", color=style.get_color(0), linewidth=2.0)

    for day in range(1, max(1, total_steps // seq_len)):
        ax.axvline(day * seq_len, color="#dddddd", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Hour Index")
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstruction(original: np.ndarray, recon_a: np.ndarray, recon_b: np.ndarray, output_path: str, title_a: str, title_b: str):
    style = PlotStyle()
    L, F = original.shape

    if F > 1:
        orig_series = original.sum(axis=1)
        recon_a_series = recon_a.sum(axis=3)[:, 0]
        recon_b_series = recon_b.sum(axis=3)[:, 0]
    else:
        orig_series = original[:, 0]
        recon_a_series = recon_a[:, 0, :, 0]
        recon_b_series = recon_b[:, 0, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    hours = np.arange(L)

    axes[0].plot(hours, orig_series, label="Original", color=style.color_original, linewidth=2.0)
    for s in range(recon_a_series.shape[0]):
        axes[0].plot(hours, recon_a_series[s], color=style.get_color(s), alpha=0.6, linewidth=1.5)
    axes[0].set_title(title_a)
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, orig_series, label="Original", color=style.color_original, linewidth=2.0)
    for s in range(recon_b_series.shape[0]):
        axes[1].plot(hours, recon_b_series[s], color=style.get_color(s + 1), alpha=0.6, linewidth=1.5)
    axes[1].set_title(title_b)
    axes[1].set_xlabel("Hour")
    axes[1].grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Demo generator capabilities: week generation + EV-conditioned reconstruction.")
    parser.add_argument("--model-path", required=True, help="Path to model folder containing config + checkpoints.")
    parser.add_argument("--output-dir", default="evaluation_comparison/generator_demo", help="Output directory for plots.")
    parser.add_argument("--meter-id", type=int, default=None, help="Meter ID for conditioning.")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD) for week generation.")
    parser.add_argument("--days", type=int, default=7, help="Number of days to generate.")
    parser.add_argument("--input-path", default=None, help="CSV or parquet path for reconstruction input.")
    parser.add_argument("--input-date", default=None, help="Date (YYYY-MM-DD) for reconstruction window.")
    parser.add_argument("--decode-ev-class", type=int, default=3, help="EV class for decoding (1, 2, or 3).")
    parser.add_argument("--num-recon-samples", type=int, default=6, help="Number of decoder samples for reconstruction.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--ev-path", default=None, help="Override path to EV CSV used by conditioner.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model, normalizer, conditioner = load_model(args.model_path, device=device, ev_path=args.ev_path)

    config_path = _resolve_config_path(args.model_path)
    config = _load_train_config(config_path)

    generator = SyntheticGenerator(
        model=model,
        conditioner=conditioner,
        normalizer=normalizer,
        device=device,
    )

    seq_len = int(getattr(model, "sequence_length", 24))
    data_cfg = getattr(config, "data", SimpleNamespace())
    data_mode = str(getattr(data_cfg, "mode", "csv")).lower()

    # --- Week generation ---
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime.utcnow()
    if args.meter_id is None:
        print("Warning: meter_id not provided; week generation will use random conditions.")
        week_samples, _ = generator.generate_continous(
            num_days=args.days,
            meter_id=0,
            start_date=start_date,
            num_candidates=16,
            temperature=args.temperature,
        )
    else:
        week_samples, _ = generator.generate_continous(
            num_days=args.days,
            meter_id=args.meter_id,
            start_date=start_date,
            num_candidates=16,
            temperature=args.temperature,
        )

    week_series = week_samples.detach().cpu().numpy()
    if week_series.ndim == 1:
        week_series = week_series[:, None]

    week_path = os.path.join(args.output_dir, "week_generation.png")
    _plot_week(
        week_series,
        seq_len=seq_len,
        output_path=week_path,
        title=f"Week Generation ({args.days} days)",
    )

    # --- Reconstruction with EV toggle ---
    if conditioner is None or not hasattr(conditioner, "get_condition_tags"):
        raise RuntimeError("Model has no conditioner; EV-conditioned reconstruction is not available.")

    tags = conditioner.get_condition_tags()
    if "ev_phase" not in tags and "electric_vehicle_phase" not in tags:
        raise RuntimeError("Model conditioner does not include EV conditioning (ev_phase).")

    input_path = args.input_path
    if input_path is None:
        if data_mode in {"phased", "phased_parquet"}:
            data_dir = os.path.join("data", "3p")
            candidates = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
            if not candidates:
                raise FileNotFoundError("No phased parquet files found under data/3p.")
            input_path = sorted(candidates)[0]
        else:
            input_path = os.path.join("data", "test_data_small.csv")

    phased_mode = data_mode in {"phased", "phased_parquet"}
    if input_path.endswith(".csv"):
        columns_cfg = getattr(data_cfg, "columns", SimpleNamespace())
        meter_col = getattr(columns_cfg, "meter_id", "meter_number")
        ts_dst_col = getattr(columns_cfg, "timestamp_dst", "timestamp_dst")
        ts_utc_col = getattr(columns_cfg, "timestamp_utc", None)
        value_cols = list(getattr(columns_cfg, "value_cols", ["hourly_energy_consumption"]))

        cache_dir = os.path.join(args.output_dir, "parquet_cache")
        os.makedirs(cache_dir, exist_ok=True)
        parquet_name = os.path.splitext(os.path.basename(input_path))[0] + ".parquet"
        load_csv_and_cache_as_parquet(
            filename=os.path.basename(input_path),
            cache_dirs=[cache_dir],
            parquet_name=parquet_name,
            data_dir=os.path.dirname(input_path),
            csv_path=os.path.dirname(input_path),
            meter_col=meter_col,
            timestamp_dst_col=ts_dst_col,
            value_cols=value_cols,
            timestamp_utc_col=ts_utc_col,
        )
        input_path = os.path.join(cache_dir, parquet_name)
        phased_mode = False

    window = WindowSpec(input_len=seq_len, stride=seq_len)
    columns_cfg = getattr(data_cfg, "columns", SimpleNamespace())
    meter_col = getattr(columns_cfg, "meter_id", "meter_number")
    value_cols = list(getattr(columns_cfg, "value_cols", ["hourly_energy_consumption"]))

    if phased_mode:
        hourly_aggregation = bool(getattr(config.window, "aggregated", False))
        aggregation_unit = str(getattr(data_cfg, "aggregation_energy_unit", "wh")).lower()
        aggregation_divisor = 1000.0 if aggregation_unit == "kwh" else 1.0
    else:
        hourly_aggregation = True
        aggregation_divisor = 1.0

    stream = StreamingParquetBuffer(
        sm_files=[input_path],
        replay_buffer_capacity=64,
        min_buffer_size=0,
        window_spec=window,
        normalizer=None,
        num_workers=0,
        hourly_aggregation=hourly_aggregation,
        aggregation_divisor=aggregation_divisor,
        sample_random=False,
        phased_mode=phased_mode,
        meter_col=meter_col,
        feature_cols=value_cols,
    )

    target_date = pd.Timestamp(args.input_date).date() if args.input_date else None
    selected = None
    for seq, meter_id_t, timestamp_t, _agg in stream:
        meter_val = int(meter_id_t.item())
        ts_val = int(timestamp_t.item())
        if args.meter_id is not None and meter_val != args.meter_id:
            continue
        if target_date is not None:
            if pd.to_datetime(ts_val, unit="s").date() != target_date:
                continue
        selected = (seq, meter_id_t, timestamp_t)
        break

    if selected is None:
        raise ValueError("No matching window found for the requested meter/date.")

    seq, meter_id_t, timestamp_t = selected
    values = seq.transpose(0, 1).cpu().numpy()
    meter_id = int(meter_id_t.item())
    start_ts = int(timestamp_t.item())

    input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    meter_tensor = torch.tensor([meter_id], dtype=torch.long)
    ts_tensor = torch.tensor([start_ts], dtype=torch.long)

    if hasattr(model, "_to_phase_first") and hasattr(model, "_build_condition_input"):
        cond_input = model._build_condition_input(model._to_phase_first(input_tensor))
    else:
        cond_input = input_tensor

    base_conditions = conditioner.get_conditions(cond_input, meter_id=meter_tensor, timestamps=ts_tensor)
    ev_start, ev_end = conditioner.get_condition_indexes("ev_phase")

    encode_conditions = base_conditions.clone()
    decode_conditions = base_conditions.clone()

    encode_conditions[:, ev_start:ev_end] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=encode_conditions.dtype)
    ev_class = int(args.decode_ev_class)
    if ev_class not in (1, 2, 3):
        raise ValueError("decode_ev_class must be 1, 2, or 3")
    decode_vec = torch.zeros(4, dtype=decode_conditions.dtype)
    decode_vec[ev_class] = 1.0
    decode_conditions[:, ev_start:ev_end] = decode_vec.unsqueeze(0)

    recon_base = generator.reconstruct(
        input_tensor,
        num_reconstructions=args.num_recon_samples,
        meter_id=meter_tensor,
        timestamp=ts_tensor,
        is_denormalized=False,
        temperature=args.temperature,
    )

    recon_ev = generator.reconstruct(
        input_tensor,
        num_reconstructions=args.num_recon_samples,
        meter_id=meter_tensor,
        timestamp=ts_tensor,
        new_conditions=decode_conditions,
        encode_conditions=encode_conditions,
        is_denormalized=False,
        temperature=args.temperature,
    )

    if recon_base is None or recon_ev is None:
        raise RuntimeError("Generator returned None for reconstruction.")

    S_base, B_base, D_base = recon_base.shape
    S_ev, B_ev, D_ev = recon_ev.shape
    F = values.shape[1]

    recon_base = recon_base.reshape(S_base, B_base, seq_len, F).detach().cpu().numpy()
    recon_ev = recon_ev.reshape(S_ev, B_ev, seq_len, F).detach().cpu().numpy()

    recon_path = os.path.join(args.output_dir, "reconstruction_ev_toggle.png")
    _plot_reconstruction(
        original=values,
        recon_a=recon_base,
        recon_b=recon_ev,
        output_path=recon_path,
        title_a="Reconstruction (Original Conditions)",
        title_b="Reconstruction (Decode with EV)",
    )

    print(f"Saved week generation plot to {week_path}")
    print(f"Saved reconstruction plot to {recon_path}")


if __name__ == "__main__":
    main()
