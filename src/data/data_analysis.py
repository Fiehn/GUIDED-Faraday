import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import re
import glob
import yaml
from types import SimpleNamespace

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

from data.dataloader_phased import SMPhaseData, WindowSpec
from data.normalizer import Normalizer
from visualization.plot_styling import PlotStyle

# Add paths to imports
current_dir = os.getcwd()
if 'src' not in current_dir:
    sys.path.insert(0, os.path.join(current_dir, 'src'))
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

print("Python path setup complete")
print(f"Current directory: {current_dir}")


def _to_namespace(obj):
    """Convert nested dicts to SimpleNamespace recursively."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_config(config_path: str = None):
    """Load YAML config and convert to SimpleNamespace."""
    if config_path is None:
        config_path = os.path.join(current_dir, "config", "config.yaml")
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    
    return _to_namespace(raw)


config = _load_config()



##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Load data 


# Analysis data source configuration
# Options: "phased_parquet" or "hourly_csv"
analysis_data_mode = "hourly_csv"
data_dir_parquet = "data/raw_profiles"
# hourly_csv_path = "data/test_data_eval.csv"
hourly_csv_path = "data/hourly_measurements_2023_10.csv"
analysis_freq = "1h" if analysis_data_mode == "hourly_csv" else "15min"

output_dir_data_analysis = "output/data_analysis"
os.makedirs(output_dir_data_analysis, exist_ok=True)
window_spec = WindowSpec(input_len=96, stride=96)  # used for phased loader only
plot_style = PlotStyle()


def save_df(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(output_dir_data_analysis, filename)
    df.to_csv(path, index=False)
    return path


def save_json(payload: dict, filename: str) -> str:
    path = os.path.join(output_dir_data_analysis, filename)
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, indent=2, default=str)
    return path

if analysis_data_mode == "phased_parquet":
    normalizer_3phase = Normalizer(method="log1p_phase")
    parquet_files = glob.glob(f"{data_dir_parquet}/*.parquet")
    print(f"Found {len(parquet_files)} parquet files")

    if len(parquet_files) > 0:
        normalizer_3phase.fit(X=None, sm_files=parquet_files)
        print(f"  Normalizer fitted on {len(parquet_files)} files")
        print(f"  Normalizer mean: {normalizer_3phase.mean}")
        print(f"  Normalizer std: {normalizer_3phase.std}")
    else:
        print(f"  No parquet files found in {data_dir_parquet}")
        print(f"  Available files: {glob.glob(f'{data_dir_parquet}/*')[:5]}")
        normalizer_3phase = None

    sm_phase_data = SMPhaseData(
        data_dir=data_dir_parquet,
        filename=None,
        window_spec=window_spec,
        batch_size=32,
        replay_buffer_capacity=5000,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        normalizer=normalizer_3phase,
        num_workers=config.data.phased.num_workers,
        seed=42
    )
    sm_phase_data.setup()
    print("Loaded 3-phase data from parquet files")
    print(f"Train loader: {sm_phase_data.train_loader}")
else:
    normalizer_3phase = None
    sm_phase_data = None
    if not os.path.exists(hourly_csv_path):
        raise FileNotFoundError(f"Hourly CSV file not found: {hourly_csv_path}")
    print(f"Loaded hourly CSV mode from: {hourly_csv_path}")

# Preview from phased train loader.
# Keep disabled by default for small datasets where the replay buffer may not
# reach the minimum sampling threshold.
preview_train_loader_sample = False
if preview_train_loader_sample and sm_phase_data is not None:
    train_batch = None
    try:
        train_batch = next(iter(sm_phase_data.train_loader))
    except StopIteration:
        train_batch = None

    if train_batch is not None:
        print("=" * 60)
        print("3-PHASE DATA SAMPLE")
        print("=" * 60)

        # Unpack the batch
        if isinstance(train_batch, (tuple, list)):
            if len(train_batch) >= 3:
                x, meter_ids, timestamps = train_batch[:3]
            else:
                x = train_batch[0]
                meter_ids = None
                timestamps = None
        else:
            x = train_batch
            meter_ids = None
            timestamps = None

        print(f"\nBatch tensor shape: {x.shape}")
        print(f"  - Interpretation: [batch_size, features*timesteps] or [batch_size, timesteps, features]")
        if meter_ids is not None:
            print(f"Meter IDs shape: {meter_ids.shape}")
            print(f"Meter IDs (first 5): {meter_ids[:5]}")
        if timestamps is not None:
            print(f"Timestamps shape: {timestamps.shape}")
            print(f"Timestamps (first 5): {timestamps[:5]}")

        print(f"\nBatch tensor statistics:")
        print(f"  Min: {x.min().item():.6f}")
        print(f"  Max: {x.max().item():.6f}")
        print(f"  Mean: {x.mean().item():.6f}")
        print(f"  Std: {x.std().item():.6f}")

        print(f"\nFirst sample from batch:\n{x[0]}")



##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Plotting each phase & aggregated
# Plotting: `selected_meter` and `selected_day` to plot a specific smart meter and exact day.

class SmartMeterDayPlotter:
    def __init__(
        self,
        data_dir_parquet: str,
        verbose: bool = False,
        data_mode: str = "phased_parquet",
        hourly_csv_path: str = None,
        interval_freq: str = "15min",
        hourly_energy_col: str = "hourly_energy_consumption",
    ):
        self.data_dir_parquet = data_dir_parquet
        self.verbose = verbose
        self.data_mode = data_mode
        self.hourly_csv_path = hourly_csv_path
        self.interval_freq = interval_freq
        self.hourly_energy_col = hourly_energy_col
        self.hourly_df = None
        self.meter_to_file = self._build_meter_file_map()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _build_meter_file_map(self):
        if self.data_mode == "hourly_csv":
            if self.hourly_csv_path is None or (not os.path.exists(self.hourly_csv_path)):
                raise FileNotFoundError(f"Hourly CSV not found: {self.hourly_csv_path}")

            self.hourly_df = pd.read_csv(self.hourly_csv_path)
            if "meter_number" not in self.hourly_df.columns:
                raise ValueError("Hourly CSV must contain 'meter_number' column")

            meters = sorted(pd.to_numeric(self.hourly_df["meter_number"], errors="coerce").dropna().astype(int).unique().tolist())
            return {meter: self.hourly_csv_path for meter in meters}

        parquet_files = sorted(glob.glob(f"{self.data_dir_parquet}/*.parquet"))
        meter_to_file = {}
        for p in parquet_files:
            m = re.search(r"sm_(\d+)\.parquet$", os.path.basename(p))
            if m:
                meter_to_file[int(m.group(1))] = p
        return meter_to_file

    def available_meters(self):
        return sorted(self.meter_to_file.keys())

    @staticmethod
    def _pick_col(cols, base):
        matches = [c for c in cols if c.startswith(base)]
        return matches[0] if matches else None

    def _pick_time_column(self, df):
        cols = list(df.columns)

        for c in ["timestamp_utc", "timestamp_dst", "timestamp"]:
            if c in cols:
                return c

        pref = []
# Plotting each phase & aggregated
# Plotting: `selected_meter` and `selected_day` to plot a specific smart meter and exact day.

class SmartMeterDayPlotter:
    def __init__(
        self,
        data_dir_parquet: str,
        verbose: bool = False,
        data_mode: str = "phased_parquet",
        hourly_csv_path: str = None,
        interval_freq: str = "15min",
        hourly_energy_col: str = "hourly_energy_consumption",
    ):
        self.data_dir_parquet = data_dir_parquet
        self.verbose = verbose
        self.data_mode = data_mode
        self.hourly_csv_path = hourly_csv_path
        self.interval_freq = interval_freq
        self.hourly_energy_col = hourly_energy_col
        self.hourly_df = None
        self.meter_to_file = self._build_meter_file_map()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _build_meter_file_map(self):
        if self.data_mode == "hourly_csv":
            if self.hourly_csv_path is None or (not os.path.exists(self.hourly_csv_path)):
                raise FileNotFoundError(f"Hourly CSV not found: {self.hourly_csv_path}")

            self.hourly_df = pd.read_csv(self.hourly_csv_path)
            if "meter_number" not in self.hourly_df.columns:
                raise ValueError("Hourly CSV must contain 'meter_number' column")

            meters = sorted(pd.to_numeric(self.hourly_df["meter_number"], errors="coerce").dropna().astype(int).unique().tolist())
            return {meter: self.hourly_csv_path for meter in meters}

        parquet_files = sorted(glob.glob(f"{self.data_dir_parquet}/*.parquet"))
        meter_to_file = {}
        for p in parquet_files:
            m = re.search(r"sm_(\d+)\.parquet$", os.path.basename(p))
            if m:
                meter_to_file[int(m.group(1))] = p
        return meter_to_file

    def available_meters(self):
        return sorted(self.meter_to_file.keys())

    @staticmethod
    def _pick_col(cols, base):
        matches = [c for c in cols if c.startswith(base)]
        return matches[0] if matches else None

    def _pick_time_column(self, df):
        cols = list(df.columns)

        for c in ["timestamp_utc", "timestamp_dst", "timestamp"]:
            if c in cols:
                return c

        pref = []
        for c in cols:
            lc = c.lower()
            if lc.startswith("timestamp_utc") or lc.startswith("timestamp_dst") or lc.startswith("timestamp"):
                pref.append(c)

        if "__index_level_0__" in cols:
            pref.append("__index_level_0__")

        broad = [c for c in cols if ("time" in c.lower() or "date" in c.lower()) and c not in pref]
        candidates = pref + broad

        best_col = None
        best_score = -1.0
        for c in candidates:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            score = parsed.notna().mean()
            if score > best_score:
                best_score = score
                best_col = c

        if best_col is not None and best_score > 0:
            self._log(f"Detected time column: {best_col} (parse success={best_score:.1%})")
            return best_col

        idx_parsed = pd.to_datetime(df.index, errors="coerce", utc=True)
        idx_score = idx_parsed.notna().mean() if len(idx_parsed) > 0 else 0.0
        if idx_score > 0:
            self._log(f"Using dataframe index as timestamp (parse success={idx_score:.1%})")
            return "__index__"

        return None

    def _load_meter_dataframe(self, selected_meter: int):
        available_meters = self.available_meters()
        self._log(f"Found {len(available_meters)} meters in parquet files")
        self._log(f"Available meters: {available_meters}")

        if selected_meter not in self.meter_to_file:
            raise ValueError(f"Meter {selected_meter} not found. Choose one of: {available_meters}")

        if self.data_mode == "hourly_csv":
            if self.hourly_df is None:
                self.hourly_df = pd.read_csv(self.hourly_csv_path)

            df_meter = self.hourly_df[pd.to_numeric(self.hourly_df["meter_number"], errors="coerce") == selected_meter].copy()
            if df_meter.empty:
                raise ValueError(f"No rows found for meter {selected_meter} in hourly CSV")

            time_col = self._pick_time_column(df_meter)
            if time_col is None:
                raise ValueError("No usable timestamp source found in hourly CSV")

            if self.hourly_energy_col not in df_meter.columns:
                raise ValueError(f"Hourly CSV missing required column: {self.hourly_energy_col}")

            timestamp_series = df_meter.index if time_col == "__index__" else df_meter[time_col]
            agg_kwh = pd.to_numeric(df_meter[self.hourly_energy_col], errors="coerce").fillna(0.0)
            agg_wh = agg_kwh * 1000.0

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamp_series, utc=True, errors="coerce"),
                "L1": agg_wh,
                "L2": 0.0,
                "L3": 0.0,
            }).dropna(subset=["timestamp"]).sort_values("timestamp")
        else:
            parquet_path = self.meter_to_file[selected_meter]
            self._log(f"Using file: {parquet_path}")

            df_raw = pd.read_parquet(parquet_path)
            self._log(f"Columns in file: {list(df_raw.columns)}")
            self._log(f"Index type: {type(df_raw.index).__name__}")

            time_col = self._pick_time_column(df_raw)
            if time_col is None:
                raise ValueError("No usable timestamp source found. " + f"Columns were: {list(df_raw.columns)}")

            l1_col = self._pick_col(df_raw.columns, "active_power_p14_l1")
            l2_col = self._pick_col(df_raw.columns, "active_power_p14_l2")
            l3_col = self._pick_col(df_raw.columns, "active_power_p14_l3")

            if l1_col is None and l2_col is None and l3_col is None:
                raise ValueError("No phase columns found (active_power_p14_l1/l2/l3). " + f"Columns were: {list(df_raw.columns)}")

            timestamp_series = df_raw.index if time_col == "__index__" else df_raw[time_col]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamp_series, utc=True, errors="coerce"),
                "L1": pd.to_numeric(df_raw[l1_col], errors="coerce") if l1_col else np.nan,
                "L2": pd.to_numeric(df_raw[l2_col], errors="coerce") if l2_col else np.nan,
                "L3": pd.to_numeric(df_raw[l3_col], errors="coerce") if l3_col else np.nan,
            }).dropna(subset=["timestamp"]).sort_values("timestamp")

        df = df.set_index("timestamp")
        df[["L1", "L2", "L3"]] = df[["L1", "L2", "L3"]].fillna(0.0)
        df["Aggregated"] = df[["L1", "L2", "L3"]].sum(axis=1)

        self._log(f"Overall date range (UTC): {df.index.min()} -> {df.index.max()}")
        self._log(f"Non-zero rows (aggregated): {(df['Aggregated'] > 0).sum()} / {len(df)}")
        return df

    def _resolve_day(self, df, selected_day="auto", top_n_days=10):
        daily = df["Aggregated"].groupby(df.index.floor("D")).sum().sort_values(ascending=False)
        non_zero_daily = daily[daily > 0]

        if len(non_zero_daily) == 0:
            self._log("Warning: all daily totals are zero for this meter.")
        else:
            self._log(f"Top {top_n_days} non-zero days by aggregated daily energy (kWh):")
            self._log(str((non_zero_daily.head(top_n_days) / 1000.0).round(3)))

        if selected_day == "auto":
            selected_day_ts = non_zero_daily.index[0] if len(non_zero_daily) > 0 else daily.index[0]
            selected_day = selected_day_ts.strftime("%Y-%m-%d")
            self._log(f"Auto-selected day: {selected_day}")

        return selected_day

    def plot_day(self, selected_meter: int, selected_day="auto", show_distribution=True, top_n_days=10):
        df = self._load_meter_dataframe(selected_meter)
        selected_day = self._resolve_day(df, selected_day=selected_day, top_n_days=top_n_days)

        day_start = pd.Timestamp(selected_day, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)

        day_df = df[(df.index >= day_start) & (df.index < day_end)][["L1", "L2", "L3"]].copy()

        if day_df.empty:
            min_day = df.index.min().strftime("%Y-%m-%d")
            max_day = df.index.max().strftime("%Y-%m-%d")
            raise ValueError(f"No data for meter {selected_meter} on {selected_day}. Available date range: {min_day} to {max_day}")

        full_idx = pd.date_range(start=day_start, end=day_end, freq=self.interval_freq, tz="UTC", inclusive="left")
        day_df = day_df.reindex(full_idx).fillna(0.0)
        day_df["Aggregated"] = day_df[["L1", "L2", "L3"]].sum(axis=1)

        self._log(f"Selected meter: {selected_meter}")
        self._log(f"Selected day (UTC): {selected_day}")
        self._log(f"Rows in selected day (after reindex): {len(day_df)}")
        self._log(str(day_df.head()))
        self._log(f"Day total (aggregated): {day_df['Aggregated'].sum() / 1000.0:.3f} kWh")

        if self.data_mode == "hourly_csv":
            fig, ax = plt.subplots(1, 1, figsize=plot_style.get_figsize(height_multiplier=1.2), sharex=True)
            ax.plot(
                day_df.index,
                day_df["Aggregated"].values,
                linewidth=plot_style.linewidth,
                alpha=0.9,
                marker=plot_style.marker,
                markersize=4,
                linestyle=plot_style.linestyle,
                color=plot_style.get_color_original(),
            )
            ax.set_title(f"SM {selected_meter} | {selected_day} | Aggregated")
            ax.set_ylabel("Energy (Wh)")
            ax.set_xlabel("Time (UTC)")
            ax.grid(True, alpha=0.3)
            ax.legend([f"Total: {day_df['Aggregated'].sum() / 1000.0:.2f} kWh"], loc="upper right")
            labels = ["Aggregated"]
            line_colors = [plot_style.get_color_original()]
        else:
            fig, axes = plt.subplots(4, 1, figsize=plot_style.get_figsize(height_multiplier=2.2), sharex=True)
            labels = ["L1", "L2", "L3", "Aggregated"]
            line_colors = [plot_style.get_color(0), plot_style.get_color(1), plot_style.get_color(2), plot_style.get_color_original()]

            for i, col in enumerate(labels):
                axes[i].plot(
                    day_df.index,
                    day_df[col].values,
                    linewidth=plot_style.linewidth,
                    alpha=0.9,
                    marker=plot_style.marker,
                    markersize=4,
                    linestyle=plot_style.linestyle,
                    color=line_colors[i],
                )
                axes[i].set_title(f"SM {selected_meter} | {selected_day} | {col}")
                axes[i].set_ylabel("Energy (Wh)")
                axes[i].grid(True, alpha=0.3)
                axes[i].legend([f"Total: {day_df[col].sum() / 1000.0:.2f} kWh"], loc="upper right")

            axes[-1].set_xlabel("Time (UTC)")

        plt.tight_layout()
        day_plot_path = os.path.join(output_dir_data_analysis, f"sm_{selected_meter}_{selected_day}_timeseries.png")
        fig.savefig(day_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {day_plot_path}")

        if show_distribution:
            if self.data_mode == "hourly_csv":
                fig, ax = plt.subplots(1, 1, figsize=plot_style.get_figsize(height_multiplier=1.0))
                ax.hist(day_df["Aggregated"].values, bins=40, edgecolor="black", alpha=0.75, color=plot_style.get_color_original())
                ax.set_title("Aggregated distribution")
                ax.set_xlabel("Energy (Wh)")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
            else:
                fig, axes = plt.subplots(2, 2, figsize=plot_style.get_figsize(height_multiplier=1.5))
                for idx, (ax, col) in enumerate(zip(axes.flatten(), labels)):
                    hist_color = line_colors[idx]
                    ax.hist(day_df[col].values, bins=40, edgecolor="black", alpha=0.75, color=hist_color)
                    ax.set_title(f"{col} distribution")
                    ax.set_xlabel("Energy (Wh)")
                    ax.set_ylabel("Frequency")
                    ax.grid(True, alpha=0.3)
            plt.tight_layout()
            dist_plot_path = os.path.join(output_dir_data_analysis, f"sm_{selected_meter}_{selected_day}_distribution.png")
            fig.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {dist_plot_path}")

        return day_df
    

##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
#  Coverage/Completeness Analysis (Per Meter)
# Computes per meter:
# - missing-rate in 15-minute intervals
# - number of full days (96/96 intervals observed)
# - gap statistics in the 15-minute timeline

class CoverageAnalyzer:
    def __init__(self, plotter: SmartMeterDayPlotter, freq: str = '15min'):
        self.plotter = plotter
        self.freq = freq

    def _interval_label(self) -> str:
        return str(self.freq).replace(" ", "")

    def _intervals_per_day(self) -> int:
        delta = pd.to_timedelta(self.freq)
        return int(pd.Timedelta(days=1) / delta)

    def compute_meter_coverage_metrics(self, meter_id: int):
        source_path = self.plotter.meter_to_file[meter_id]
        df = self.plotter._load_meter_dataframe(meter_id)
        ts = pd.DatetimeIndex(df.index)
        if len(ts) == 0:
            return {
                'meter_id': meter_id,
                'file': os.path.basename(source_path),
                'error': 'No valid timestamps'
            }

        ts = pd.DatetimeIndex(ts).floor(self.freq).unique().sort_values()
        expected_idx = pd.date_range(start=ts.min(), end=ts.max(), freq=self.freq, tz='UTC')

        observed = pd.Series(1, index=ts, dtype='int8')
        observed_aligned = observed.reindex(expected_idx).fillna(0).astype('int8')

        expected_intervals = int(len(expected_idx))
        observed_intervals = int(observed_aligned.sum())
        missing_intervals = int(expected_intervals - observed_intervals)
        missing_rate = float(missing_intervals / expected_intervals) if expected_intervals > 0 else np.nan
        interval_label = self._interval_label()
        intervals_per_day = self._intervals_per_day()

        # Gaps: contiguous runs of missing 15-min intervals
        missing_mask = observed_aligned.eq(0)
        if missing_mask.any():
            groups = (missing_mask != missing_mask.shift(fill_value=False)).cumsum()
            run_lengths = missing_mask.groupby(groups).sum()
            gap_run_lengths = run_lengths[run_lengths > 0].astype(int)
            num_gap_runs = int(len(gap_run_lengths))
            max_gap_intervals = int(gap_run_lengths.max())
            avg_gap_intervals = float(gap_run_lengths.mean())
        else:
            num_gap_runs = 0
            max_gap_intervals = 0
            avg_gap_intervals = 0.0

        # Full days = expected intervals/day and all observed
        per_day_expected = observed_aligned.groupby(observed_aligned.index.floor('D')).size()
        per_day_observed = observed_aligned.groupby(observed_aligned.index.floor('D')).sum()
        full_days = int(((per_day_expected == intervals_per_day) & (per_day_observed == intervals_per_day)).sum())
        complete_days = int((per_day_expected == intervals_per_day).sum())

        metrics = {
            'meter_id': int(meter_id),
            'file': os.path.basename(source_path),
            'start_utc': ts.min(),
            'end_utc': ts.max(),
            f'expected_intervals_{interval_label}': expected_intervals,
            f'observed_intervals_{interval_label}': observed_intervals,
            f'missing_intervals_{interval_label}': missing_intervals,
            'missing_rate': missing_rate,
            'coverage_rate': float(1.0 - missing_rate) if pd.notna(missing_rate) else np.nan,
            f'num_gap_runs_{interval_label}': num_gap_runs,
            f'max_gap_intervals_{interval_label}': max_gap_intervals,
            f'avg_gap_intervals_{interval_label}': avg_gap_intervals,
            f'full_days_{intervals_per_day}_of_{intervals_per_day}': full_days,
            f'days_with_{intervals_per_day}_expected': complete_days,
        }
        return metrics

    def report_all_meters(self, sort_by='missing_rate', ascending=False):
        rows = []
        for meter in self.plotter.available_meters():
            rows.append(self.compute_meter_coverage_metrics(meter_id=meter))

        report = pd.DataFrame(rows)
        if sort_by in report.columns:
            report = report.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        return report


##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Daily profile stats per meter
# Computes daily profile metrics from aggregated 15-minute load (Wh per interval):
# - per-day total energy (reported in kWh)
# - peak load and time-of-peak (Wh per 15-min interval)
# - load factor (mean / peak)

class DailyProfileAnalyzer:
    def __init__(self, plotter: SmartMeterDayPlotter, freq: str = '15min'):
        self.plotter = plotter
        self.freq = freq

    def compute_daily_profile_stats(self, meter_id: int):
        df = self.plotter._load_meter_dataframe(meter_id)
        df = df.sort_index()

        # Use aggregated 15-minute profile in Wh per interval
        agg = df['Aggregated'].copy()
        day_key = agg.index.floor('D')

        daily_total_wh = agg.groupby(day_key).sum()
        daily_total_kwh = daily_total_wh / 1000.0
        daily_mean_wh = agg.groupby(day_key).mean()
        daily_peak_wh = agg.groupby(day_key).max()

        # Time of peak within each day (UTC)
        peak_timestamp = agg.groupby(day_key).idxmax()
        peak_time_utc = peak_timestamp.dt.strftime('%H:%M')

        # Load factor = mean / peak (safe divide)
        load_factor = (daily_mean_wh / daily_peak_wh.replace(0, np.nan)).fillna(0.0)

        daily_stats = pd.DataFrame({
            'meter_id': int(meter_id),
            'date_utc': daily_total_wh.index,
            'daily_total_wh': daily_total_wh.values,
            'daily_total_kwh': daily_total_kwh.values,
            'peak_load_wh_15min': daily_peak_wh.values,
            'time_of_peak_utc': peak_time_utc.values,
            'load_factor_mean_over_peak': load_factor.values,
        })

        return daily_stats.sort_values('date_utc').reset_index(drop=True)

    def report_all_meters(self):
        all_daily = []
        for meter_id in self.plotter.available_meters():
            meter_daily = self.compute_daily_profile_stats(meter_id=meter_id)
            all_daily.append(meter_daily)

        if len(all_daily) == 0:
            return pd.DataFrame(), pd.DataFrame()

        all_daily_df = pd.concat(all_daily, ignore_index=True)

        summary = all_daily_df.groupby('meter_id').agg(
            days=('date_utc', 'count'),
            avg_daily_total_kwh=('daily_total_kwh', 'mean'),
            median_daily_total_kwh=('daily_total_kwh', 'median'),
            avg_peak_load_wh_15min=('peak_load_wh_15min', 'mean'),
            avg_load_factor=('load_factor_mean_over_peak', 'mean'),
        ).reset_index().sort_values('meter_id')

        return all_daily_df, summary


##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Phase Balance Metrics (Per Meter)
# Analyzes 3-phase load balance across the entire dataset:
# - per-phase contribution shares (L1%, L2%, L3%) and average Wh per interval
# - phase dominance: which phase carries the most load on average
# - imbalance index: measure of phase imbalance (0=perfect balance, 1=completely unbalanced)

class PhaseBalanceAnalyzer:
    def __init__(self, plotter: SmartMeterDayPlotter):
        self.plotter = plotter

    def compute_meter_phase_balance_stats(self, meter_id: int):
        """Compute phase balance metrics for all timesteps in a meter's data."""
        df = self.plotter._load_meter_dataframe(meter_id)

        # Extract L1, L2, L3 data
        l1 = df['L1'].values
        l2 = df['L2'].values
        l3 = df['L3'].values
        agg = l1 + l2 + l3

        # Avoid divide by zero
        agg_safe = agg.copy()
        agg_safe[agg_safe == 0] = 1.0

        # Per-phase contribution shares (percentage)
        l1_share = (l1 / agg_safe * 100.0).astype(float)
        l2_share = (l2 / agg_safe * 100.0).astype(float)
        l3_share = (l3 / agg_safe * 100.0).astype(float)

        # Average per-phase values where aggregated > 0
        mask_nonzero = agg > 0
        if mask_nonzero.sum() > 0:
            avg_l1_wh = l1[mask_nonzero].mean()
            avg_l2_wh = l2[mask_nonzero].mean()
            avg_l3_wh = l3[mask_nonzero].mean()
            avg_agg_wh = agg[mask_nonzero].mean()
        else:
            avg_l1_wh = avg_l2_wh = avg_l3_wh = avg_agg_wh = 0.0

        # Phase dominance: which phase is dominant on average across nonzero intervals
        if mask_nonzero.sum() > 0:
            dom_l1_avg_share = l1_share[mask_nonzero].mean()
            dom_l2_avg_share = l2_share[mask_nonzero].mean()
            dom_l3_avg_share = l3_share[mask_nonzero].mean()
            dominant_phase = ['L1', 'L2', 'L3'][np.argmax([dom_l1_avg_share, dom_l2_avg_share, dom_l3_avg_share])]
        else:
            dom_l1_avg_share = dom_l2_avg_share = dom_l3_avg_share = 0.0
            dominant_phase = 'N/A'

        # Imbalance index: standard deviation of phase shares on nonzero intervals
        if mask_nonzero.sum() > 0:
            phase_shares_nonzero = np.array([l1_share[mask_nonzero], l2_share[mask_nonzero], l3_share[mask_nonzero]])
            imbalance_index = phase_shares_nonzero.std() / (100.0 / np.sqrt(3))
        else:
            imbalance_index = 0.0

        return {
            'meter_id': int(meter_id),
            'avg_l1_wh_15min': float(avg_l1_wh),
            'avg_l2_wh_15min': float(avg_l2_wh),
            'avg_l3_wh_15min': float(avg_l3_wh),
            'avg_agg_wh_15min': float(avg_agg_wh),
            'avg_l1_share_pct': float(dom_l1_avg_share),
            'avg_l2_share_pct': float(dom_l2_avg_share),
            'avg_l3_share_pct': float(dom_l3_avg_share),
            'dominant_phase': dominant_phase,
            'imbalance_index': float(imbalance_index),
        }

    def report_all_meters(self):
        """Compute phase balance metrics for all meters."""
        rows = []
        for meter_id in self.plotter.available_meters():
            stats = self.compute_meter_phase_balance_stats(meter_id)
            rows.append(stats)

        return pd.DataFrame(rows).sort_values('meter_id').reset_index(drop=True)


##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Plots distributions across all data
# This plots the distributions using all available timesteps from all parquet files (L1, L2, L3, and Aggregated).

class DistributionAnalyzer:
    def __init__(self, plotter: SmartMeterDayPlotter, exclude_zeros: bool = True, verbose: bool = False):
        self.plotter = plotter
        self.exclude_zeros = exclude_zeros
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def compute_and_save(self):
        all_l1, all_l2, all_l3 = [], [], []
        processed = 0

        for meter_id in self.plotter.available_meters():
            df_meter = self.plotter._load_meter_dataframe(meter_id)
            s1 = pd.to_numeric(df_meter["L1"], errors="coerce").fillna(0.0).to_numpy()
            s2 = pd.to_numeric(df_meter["L2"], errors="coerce").fillna(0.0).to_numpy()
            s3 = pd.to_numeric(df_meter["L3"], errors="coerce").fillna(0.0).to_numpy()

            all_l1.append(s1)
            all_l2.append(s2)
            all_l3.append(s3)
            processed += 1

        if processed == 0:
            raise ValueError("No meter data was processed for distribution analysis.")

        L1_all = np.concatenate(all_l1)
        L2_all = np.concatenate(all_l2)
        L3_all = np.concatenate(all_l3)
        Agg_all = L1_all + L2_all + L3_all

        if self.plotter.data_mode == "hourly_csv":
            raw_series_list = [Agg_all]
            titles = ["Aggregated full-series distribution (all files)"]
        else:
            raw_series_list = [L1_all, L2_all, L3_all, Agg_all]
            titles = [
                "L1 full-series distribution (all files)",
                "L2 full-series distribution (all files)",
                "L3 full-series distribution (all files)",
                "Aggregated full-series distribution (all files)",
            ]

        series_list = []
        kept_counts = {}
        for series, title in zip(raw_series_list, titles):
            if self.exclude_zeros:
                filtered = series[series > 0]
                kept_counts[title] = {"kept": int(len(filtered)), "total": int(len(series))}
                series_list.append(filtered)
            else:
                kept_counts[title] = {"kept": int(len(series)), "total": int(len(series))}
                series_list.append(series)

        if self.plotter.data_mode == "hourly_csv":
            fig, ax = plt.subplots(1, 1, figsize=plot_style.get_figsize(height_multiplier=1.0))
            data = series_list[0]
            title = titles[0]
            ax.hist(data, bins=120, edgecolor="black", alpha=0.75, color=plot_style.get_color_original())
            suffix = " (zeros excluded)" if self.exclude_zeros else ""
            ax.set_title(f"{title}{suffix}")
            ax.set_xlabel("Energy (Wh)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(2, 2, figsize=plot_style.get_figsize(height_multiplier=1.7))
            dist_colors = [plot_style.get_color(0), plot_style.get_color(1), plot_style.get_color(2), plot_style.get_color_original()]
            for idx, (ax, data, title) in enumerate(zip(axes.flatten(), series_list, titles)):
                ax.hist(data, bins=120, edgecolor="black", alpha=0.75, color=dist_colors[idx])
                suffix = " (zeros excluded)" if self.exclude_zeros else ""
                ax.set_title(f"{title}{suffix}")
                ax.set_xlabel("Energy (Wh)")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        full_dist_plot_path = os.path.join(output_dir_data_analysis, "full_series_phase_distributions.png")
        fig.savefig(full_dist_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        distribution_summary_path = save_json(
            {
                "processed_files": int(processed),
                "total_samples_per_phase": int(len(L1_all)),
                "non_zero_counts": {
                    "L1": int((L1_all > 0).sum()),
                    "L2": int((L2_all > 0).sum()),
                    "L3": int((L3_all > 0).sum()),
                    "Aggregated": int((Agg_all > 0).sum()),
                },
                "exclude_zeros": bool(self.exclude_zeros),
                "kept_counts": kept_counts,
            },
            "full_series_distribution_summary.json",
        )

        return full_dist_plot_path, distribution_summary_path


##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# Seasonality 
# Analyzes temporal patterns in aggregated load:
# - Monthly distributions and average load by month
# - Weekday vs weekend patterns (what it looks like)
# - Statistical comparison of calendar effects (weekday load vs weekend load)

class SeasonalityAnalyzer:
    def __init__(self, plotter: SmartMeterDayPlotter):
        self.plotter = plotter

    def compute_stats(self):
        """Analyze calendar effects across aggregated load data."""
        all_data = []
        for meter_id in self.plotter.available_meters():
            df_meter = self.plotter._load_meter_dataframe(meter_id)
            df_temp = pd.DataFrame({
                'timestamp': df_meter.index,
                'agg_wh': pd.to_numeric(df_meter['Aggregated'], errors='coerce').fillna(0.0).values,
            }).dropna(subset=['timestamp']).sort_values('timestamp')

            df_temp['month'] = df_temp['timestamp'].dt.month
            df_temp['weekday'] = df_temp['timestamp'].dt.dayofweek
            df_temp['is_weekend'] = df_temp['weekday'].isin([5, 6])

            all_data.append(df_temp)

        if len(all_data) == 0:
            return None, None, None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        # Monthly statistics
        month_stats = combined_df.groupby('month').agg({
            'agg_wh': ['mean', 'median', 'std', 'count']
        }).round(4)
        month_stats.columns = ['mean_wh', 'median_wh', 'std_wh', 'count_intervals']
        month_stats = month_stats.reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_stats['month_name'] = month_stats['month'].map(lambda m: month_names[m-1] if 1 <= m <= 12 else 'Unknown')

        # Weekday vs Weekend statistics
        wd_we_stats = combined_df.groupby('is_weekend').agg({
            'agg_wh': ['mean', 'median', 'std', 'count']
        }).round(4)
        wd_we_stats.columns = ['mean_wh', 'median_wh', 'std_wh', 'count_intervals']
        wd_we_stats.index = ['Weekday', 'Weekend']
        wd_we_stats = wd_we_stats.reset_index().rename(columns={'index': 'day_type'})

        # Detailed weekday breakdown
        weekday_stats = combined_df.groupby('weekday').agg({
            'agg_wh': ['mean', 'median', 'std', 'count']
        }).round(4)
        weekday_stats.columns = ['mean_wh', 'median_wh', 'std_wh', 'count_intervals']
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_stats['weekday_name'] = [weekday_names[i] for i in weekday_stats.index]
        weekday_stats = weekday_stats.reset_index(drop=True)

        return month_stats, wd_we_stats, weekday_stats



##############################################################################
# ----------------------------------------------------------------------------
##############################################################################
# OUTPUT 


def make_plotter(verbose: bool = False) -> SmartMeterDayPlotter:
    return SmartMeterDayPlotter(
        data_dir_parquet=data_dir_parquet,
        verbose=verbose,
        data_mode=analysis_data_mode,
        hourly_csv_path=hourly_csv_path,
        interval_freq=analysis_freq,
        hourly_energy_col="hourly_energy_consumption",
    )




# Plotting each phase & aggregated
plotter = make_plotter(verbose=False)
available_meters = plotter.available_meters()
if len(available_meters) == 0:
    raise ValueError("No meters available for analysis in current data source")

selected_meter = 116 if 116 in available_meters else available_meters[0]
selected_day = "auto"  # or e.g. "2024-02-01"
day_df = plotter.plot_day(
    selected_meter=selected_meter,
    selected_day=selected_day,
    show_distribution=False,
    top_n_days=10,
    )

day_df_path = save_df(day_df.reset_index().rename(columns={"index": "timestamp_utc"}), "selected_day_profile.csv")




# Coverage/completeness analysis
plotter = make_plotter(verbose=False)
coverage_analyzer = CoverageAnalyzer(plotter=plotter, freq=analysis_freq)
coverage_df = coverage_analyzer.report_all_meters()
interval_label = analysis_freq.replace(" ", "")
intervals_per_day = int(pd.Timedelta(days=1) / pd.to_timedelta(analysis_freq))
display_cols = [
    'meter_id', 'missing_rate',
    f'missing_intervals_{interval_label}',
    f'num_gap_runs_{interval_label}',
    f'max_gap_intervals_{interval_label}',
    f'full_days_{intervals_per_day}_of_{intervals_per_day}',
    f'days_with_{intervals_per_day}_expected',
    f'expected_intervals_{interval_label}',
    f'observed_intervals_{interval_label}',
]
coverage_display_df = coverage_df[[col for col in display_cols if col in coverage_df.columns]]

coverage_path = save_df(coverage_df, "coverage_all_meters.csv")
coverage_display_path = save_df(coverage_display_df, "coverage_display_subset.csv")

if 'missing_rate' in coverage_df.columns:
    coverage_summary_path = save_json(
        {
            "average_missing_rate": float(coverage_df['missing_rate'].mean()),
            "best_coverage_meter": int(coverage_df.loc[coverage_df['missing_rate'].idxmin(), 'meter_id']),
            "worst_coverage_meter": int(coverage_df.loc[coverage_df['missing_rate'].idxmax(), 'meter_id']),
        },
        "coverage_summary.json",
    )
else:
    coverage_summary_path = None




# Daily profile stats for all meters
plotter = make_plotter(verbose=False)
daily_profile_analyzer = DailyProfileAnalyzer(plotter=plotter, freq=analysis_freq)
all_daily_stats_df, daily_summary_df = daily_profile_analyzer.report_all_meters()

daily_all_path = save_df(all_daily_stats_df, "daily_profile_all_meter_days.csv")
daily_summary_path = save_df(daily_summary_df, "daily_profile_summary_per_meter.csv")

daily_global_summary_path = save_json(
    {
        "rows_meter_days": int(len(all_daily_stats_df)),
        "average_load_factor": float(all_daily_stats_df['load_factor_mean_over_peak'].mean()),
    },
    "daily_profile_global_summary.json",
)




# Phase balance analysis for all meters
plotter = make_plotter(verbose=False)
phase_balance_analyzer = PhaseBalanceAnalyzer(plotter=plotter)
phase_balance_df = phase_balance_analyzer.report_all_meters()
phase_balance_path = save_df(phase_balance_df, "phase_balance_per_meter.csv")
phase_balance_summary_path = save_json(
    {
        "average_imbalance_index": float(phase_balance_df['imbalance_index'].mean()),
        "dominant_phase_counts": phase_balance_df['dominant_phase'].value_counts().to_dict(),
    },
    "phase_balance_summary.json",
)


# Full-series distributions across all meters
distribution_analyzer = DistributionAnalyzer(
    plotter=plotter,
    exclude_zeros=True,
    verbose=False,
)
full_dist_plot_path, distribution_summary_path = distribution_analyzer.compute_and_save()



# Run seasonality analysis
plotter = make_plotter(verbose=False)
seasonality_analyzer = SeasonalityAnalyzer(plotter=plotter)
month_stats, wd_we_stats, weekday_stats = seasonality_analyzer.compute_stats()

if month_stats is not None:
    month_stats_out = month_stats[['month_name', 'mean_wh', 'median_wh', 'std_wh', 'count_intervals']].copy()
    month_stats_path = save_df(month_stats_out, "seasonality_monthly_stats.csv")
else:
    month_stats_path = None

if wd_we_stats is not None:
    wd_we_out = wd_we_stats[['day_type', 'mean_wh', 'median_wh', 'std_wh', 'count_intervals']].copy()
    wd_we_path = save_df(wd_we_out, "seasonality_weekday_vs_weekend.csv")
    weekday_mean = wd_we_stats[wd_we_stats['day_type'] == 'Weekday']['mean_wh'].values[0]
    weekend_mean = wd_we_stats[wd_we_stats['day_type'] == 'Weekend']['mean_wh'].values[0]
    pct_diff = ((weekday_mean - weekend_mean) / weekend_mean * 100) if weekend_mean > 0 else 0
    seasonality_summary_path = save_json(
        {
            "weekday_mean_wh": float(weekday_mean),
            "weekend_mean_wh": float(weekend_mean),
            "weekday_vs_weekend_pct_diff": float(pct_diff),
        },
        "seasonality_summary.json",
    )
else:
    wd_we_path = None
    seasonality_summary_path = None

if weekday_stats is not None:
    weekday_stats_out = weekday_stats[['weekday_name', 'mean_wh', 'median_wh', 'std_wh', 'count_intervals']].copy()
    weekday_stats_path = save_df(weekday_stats_out, "seasonality_weekday_detail.csv")
else:
    weekday_stats_path = None





# Summary 
run_summary_path = save_json(
    {
        "day_profile_csv": day_df_path,
        "coverage_csv": coverage_path,
        "coverage_display_csv": coverage_display_path,
        "coverage_summary_json": coverage_summary_path,
        "daily_profile_all_csv": daily_all_path,
        "daily_profile_summary_csv": daily_summary_path,
        "daily_profile_global_summary_json": daily_global_summary_path,
        "phase_balance_csv": phase_balance_path,
        "phase_balance_summary_json": phase_balance_summary_path,
        "seasonality_monthly_csv": month_stats_path,
        "seasonality_weekday_weekend_csv": wd_we_path,
        "seasonality_weekday_detail_csv": weekday_stats_path,
        "seasonality_summary_json": seasonality_summary_path,
        "distribution_summary_json": distribution_summary_path,
        "distribution_plot": full_dist_plot_path,
    },
    "analysis_outputs.json",
)

print(f"Saved analysis outputs to: {output_dir_data_analysis}")
print(f"Data analysis output file: {run_summary_path}")
