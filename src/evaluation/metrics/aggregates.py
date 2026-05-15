import numpy as np
import os
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List
from matplotlib.ticker import AutoMinorLocator

from datetime import datetime

from src.evaluation.metrics.base.base import SyntheticMetric, _save_standalone_metric_result, _sanitize_numeric_array, _resolve_output_directory, _to_json_serializable
from src.evaluation.metrics.flexibility_distance import FlexibilityDistance
from src.evaluation.metrics.tstr import TSTR
from src.evaluation.metrics.acf import ACFDistribution
from src.evaluation.metrics.mmd import MaximumMeanDiscrepancy
from src.visualization.plot_styling import PlotStyle

class AggregatedProfiles(SyntheticMetric):
    """Aggregate samples from real and synthetic data and compute Flexibility Distance between aggregated profiles.
     This metric evaluates how well the synthetic data captures the overall consumption patterns when multiple samples are aggregated together, 
     which is important for understanding collective behavior and flexibility in energy consumption.
     Steps:
     1. Randomly sample N time series from real and synthetic datasets (e.g.,
     2. Aggregate by summing consumption across the sampled series at each time step to create an "aggregated profile" for real and synthetic data.
     3. Compute Flexibility Distance between the two aggregated profiles to assess similarity in overall patterns and flexibility.
     4. Optionally, plot the aggregated profiles for visual comparison.
     Lower Flexibility Distance indicates that the synthetic data captures the collective consumption patterns of the real data more closely, with 0 indicating identical aggregated profiles.
    
    Args:
    n_samples (int): Number of time series to sample and aggregate (default: 100)
    window_size (int): Length of each time series window (default: 24)
    tau (int or 'auto'): Time delay for Flexibility Distance (default: 'auto, which uses median heuristic)
    """
    
    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    output_path: str = None
    
    def __init__(self, n_samples: int = 100, window_size: int = 24, tau: str = 'auto',
                 plot_filename: str = 'aggregated_profiles_comparison.pdf'):
        self.n_samples = n_samples
        self.window_size = window_size
        self.tau = tau
        self.plot_filename = plot_filename
    
    @property
    def name(self) -> str:
        return "Aggregated Profiles (Flexibility Distance)"

    def _uses_phase_first_flatten(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, '_to_phase_first')

    def _extract_consumption_feature(self, data: np.ndarray) -> np.ndarray:
        arr = _sanitize_numeric_array(np.asarray(data))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D flattened array [N, D], got {arr.shape}")

        if arr.shape[1] == self.window_size:
            return arr

        if arr.shape[1] % self.window_size != 0:
            raise ValueError(
                f"Flattened dimension {arr.shape[1]} is not divisible by window_size {self.window_size}"
            )

        n_features = arr.shape[1] // self.window_size
        if n_features == 1:
            return arr.reshape(-1, self.window_size)

        if self._uses_phase_first_flatten():
            # Flatten order is [f0_t0..f0_tL-1, f1_t0..], so feature 0 is contiguous.
            return arr.reshape(-1, n_features, self.window_size)[:, 0, :]

        # Fallback for time-first flatten order [t0_f0, t0_f1, ..., t1_f0, ...].
        return arr.reshape(-1, self.window_size, n_features)[:, :, 0]
    
    def compute(self) -> float:
        """Aggregate real and synthetic data, then compute Flexibility Distance."""
        real_data = self._extract_consumption_feature(self.real_data)
        synthetic_data = self._extract_consumption_feature(self.synthetic_data)
        
        # Use all available samples (match the size to the smaller dataset)
        n_agg = min(len(real_data), len(synthetic_data))
        if n_agg < 1:
            raise ValueError("AggregatedProfiles requires at least one sample in both real and synthetic data")

        real_samples = real_data[:n_agg]
        synth_samples = synthetic_data[:n_agg]
        
        # Aggregate by summing
        real_aggregated = np.sum(real_samples, axis=0)
        synth_aggregated = np.sum(synth_samples, axis=0)
        
        # Plot aggregated profiles
        self._plot_aggregated_profiles(real_aggregated, synth_aggregated, n_agg=n_agg)
        
        # Compute Flexibility Distance
        fd_score = FlexibilityDistance.compute_fd_between_series(
            real_aggregated, synth_aggregated, tau=self.tau
        )

        _save_standalone_metric_result(
            metric_key='AggregatedProfiles',
            payload={
                'value': float(fd_score),
                'n_samples_requested': int(self.n_samples),
                'n_samples_used': int(n_agg),
                'window_size': int(self.window_size),
                'tau': self.tau,
                'real_aggregated': real_aggregated,
                'synthetic_aggregated': synth_aggregated,
                'updated_at_utc': datetime.utcnow().isoformat() + 'Z',
            },
            output_file=self.output_path or 'models/eval_results.json'
        )
        
        return fd_score
    
    def _plot_aggregated_profiles(self, real_aggregated: np.ndarray, synth_aggregated: np.ndarray, n_agg: int = None):
        """Plot aggregated real and synthetic profiles.
        Args:
            real_aggregated (np.ndarray): Aggregated profile from real data
            synth_aggregated (np.ndarray): Aggregated profile from synthetic data
            n_agg (int): Number of samples aggregated (used for plot title)
        """
        if n_agg is None:
            n_agg = self.n_samples
        fig, ax = plt.subplots(figsize=(18, 6))
        timesteps = np.arange(len(real_aggregated))
        ax.plot(timesteps, real_aggregated, label='Real Aggregated', 
                color='blue', linewidth=2, alpha=0.8)
        ax.plot(timesteps, synth_aggregated, label='Synthetic Aggregated', 
                color='red', linewidth=2, alpha=0.8, linestyle='--')
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Aggregated Consumption [kWh]', fontsize=12)
        ax.set_title(f'Aggregated Consumption Profiles (all n={n_agg} samples)', fontsize=14)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='x', which='major', length=6)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        base_dir = _resolve_output_directory(self.output_path)
        os.makedirs(base_dir, exist_ok=True)
        plot_filename = self.plot_filename or 'aggregated_profiles_comparison.pdf'
        plt.savefig(os.path.join(base_dir, plot_filename), dpi=150, bbox_inches='tight')
        plt.close()
        

class PerConditionEval(SyntheticMetric):
    """Compute per-condition evaluation for MMD, ACFDistribution, TSTR, ClassicalStatistics, and AggregatedProfiles.

    Includes:
    - MMD per day-of-week and month (using full 24-hour windows)
    - ACFDistribution per day-of-week and month (using full 24-hour windows)
    - ClassicalStatistics per day-of-week and month
    - TSTR per day-of-week and month (train/test on real, train on synthetic, test on real holdout)
    - AggregatedProfiles per day-of-week and month (aggregate samples within each group and compare profiles)

        Also includes AggregatedProfiles per feeder (aggregate samples by feeder and compare profiles), which is important for understanding how well the synthetic data captures patterns at the feeder level, which can be critical for grid management and planning.

        This comprehensive per-condition evaluation framework allows us to assess synthetic data quality across multiple dimensions and conditions, providing insights into where the synthetic data performs well and where it may need improvement.

    Args:
    window_size (int): Length of time series windows (default: 24 for hourly data)
    max_lag (int): Maximum lag for ACFDistribution (default: 10)
    min_samples_per_group (int): Minimum samples required in each group to compute metrics (default: 2)
    verbose (bool): Whether to print detailed debug information during computation (default: False)
    predictor_type (str): Predictor type for TSTR ('linear', 'mlp', etc.) (default: 'linear')
    tstr_epochs (int): Training epochs for TSTR predictor (default: 50)
    feeder_mapping_path (str): Path to JSON file mapping meter IDs to feeder IDs (default: 'topology/cleaned/Cleaned_SM_Topology_mapping.json')
    min_samples_per_feeder (int): Minimum samples required per feeder to compute AggregatedProfiles (default: 5)
    """

    real_data: np.ndarray = None
    synthetic_data: np.ndarray = None
    timestamps: np.ndarray = None
    meter_ids: np.ndarray = None
    feeder_ids: np.ndarray = None
    output_path: str = None

    def __init__(self, window_size: int = 24, max_lag: int = 10, min_samples_per_group: int = 2,
                 verbose: bool = False, predictor_type: str = 'linear', tstr_epochs: int = 50,
                 feeder_mapping_path: str = 'data/topology/cleaned/Cleaned_SM_Topology_mapping.json',
                 min_samples_per_feeder: int = 5):
        self.window_size = window_size
        self.max_lag = max_lag
        self.min_samples_per_group = min_samples_per_group
        self.verbose = verbose
        self.predictor_type = predictor_type
        self.tstr_epochs = tstr_epochs
        self.feeder_mapping_path = feeder_mapping_path
        self.min_samples_per_feeder = min_samples_per_feeder
        self._meter_to_feeder = None

        self.results_by_condition = {
            'mmd': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'acf': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'classical_statistics': {
                'day_of_week': {},
                'month_of_year': {},
            },
            'aggregated_profiles': {
                'day_of_week': {},
                'month_of_year': {},
                'feeder': {},
            },
            'tstr': {
                'day_of_week': {},
                'month_of_year': {},
            }
        }

        self.mmd_calculator = MaximumMeanDiscrepancy(sigma=None, seed=42, max_samples=2000)
        self.acf_calculator = ACFDistribution(window_size=window_size, max_lag=max_lag)
        self.tstr_calculator = TSTR(
            task='next_step',
            window_size=window_size,
            n_features=1,
            predictor_type=predictor_type,
            epochs=tstr_epochs,
            random_state=42
        )
        self.aggregated_profiles_calculator = AggregatedProfiles(
            n_samples=100,
            window_size=window_size,
            tau='auto',
        )

    @property
    def name(self) -> str:
        return "Per-Condition Evaluation"

    def _compute_subset_acf(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute ACF score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None
        self.acf_calculator.real_data = real_subset
        self.acf_calculator.synthetic_data = synthetic_subset
        return float(self.acf_calculator.compute())

    def _compute_subset_tstr(self, real_subset: np.ndarray, synthetic_subset: np.ndarray,
                             group_label: str = None) -> float:
        """Compute TSTR ratio for one subset using the TSTR metric; return None if too few samples."""
        if len(real_subset) < 4 or len(synthetic_subset) < 2:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} -> insufficient samples (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            return None

        try:
            if self.verbose and group_label is not None:
                print(f"\n  [TSTR] {group_label} (real={len(real_subset)}, synthetic={len(synthetic_subset)})")
            self.tstr_calculator.real_data = real_subset
            self.tstr_calculator.synthetic_data = synthetic_subset
            tstr_ratio = self.tstr_calculator.compute()
            return float(tstr_ratio)
        except Exception as e:
            if self.verbose:
                label_suffix = f" for {group_label}" if group_label is not None else ""
                print(f"Warning: TSTR computation failed{label_suffix}: {e}")
            return None

    def _compute_subset_aggregated_profiles(self, real_subset: np.ndarray, synthetic_subset: np.ndarray,
                                            group_type: str = 'all', group_value: Any = 'all'):
        """Compute AggregatedProfiles score for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            group_key = self._sanitize_label_for_filename(group_type)
            group_val = self._sanitize_label_for_filename(group_value)
            plot_filename = f"aggregated_profiles_comparison_{group_key}_{group_val}.pdf"

            metric = AggregatedProfiles(
                n_samples=self.aggregated_profiles_calculator.n_samples,
                window_size=self.aggregated_profiles_calculator.window_size,
                tau=self.aggregated_profiles_calculator.tau,
                plot_filename=plot_filename,
            )
            metric.model = getattr(self, 'model', None)
            metric.real_data = real_subset
            metric.synthetic_data = synthetic_subset
            metric.output_path = self.output_path
            return float(metric.compute())
        except Exception as e:
            if self.verbose:
                print(f"Warning: AggregatedProfiles computation failed: {e}")
            return None

    def _load_meter_to_feeder_mapping(self) -> Dict[str, str]:
        """Load meter->feeder mapping from Cleaned_SM_Topology_mapping.json."""
        if self._meter_to_feeder is not None:
            return self._meter_to_feeder

        mapping_path = self.feeder_mapping_path
        if not os.path.isabs(mapping_path):
            mapping_path = os.path.join(os.getcwd(), mapping_path)

        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)

        meter_to_feeder = {}
        for meter, topo in raw_mapping.items():
            feeder_id = topo.get('Feeder ID') if isinstance(topo, dict) else None
            if feeder_id is None:
                meter_to_feeder[str(meter)] = None
            else:
                meter_to_feeder[str(meter)] = f"LvFeeder.{feeder_id}"

        self._meter_to_feeder = meter_to_feeder
        return meter_to_feeder

    def _resolve_feeder_ids(self, n_expected: int) -> np.ndarray:
        """Resolve feeder IDs either from injected feeder_ids or from injected meter_ids + mapping file."""
        if self.feeder_ids is not None:
            feeder_ids = np.asarray(self.feeder_ids)
        elif self.meter_ids is not None:
            meter_ids = np.asarray(self.meter_ids)
            if len(meter_ids) != n_expected:
                raise ValueError(f"meter_ids length ({len(meter_ids)}) must match data length ({n_expected})")

            meter_to_feeder = self._load_meter_to_feeder_mapping()
            feeder_ids = []
            for meter in meter_ids:
                if isinstance(meter, np.generic):
                    meter = meter.item()
                meter_key = str(meter)
                feeder_ids.append(meter_to_feeder.get(meter_key, None))
            feeder_ids = np.array(feeder_ids, dtype=object)
        else:
            return None

        if len(feeder_ids) != n_expected:
            raise ValueError(f"feeder_ids length ({len(feeder_ids)}) must match data length ({n_expected})")
        return self._normalize_feeder_ids(feeder_ids)

    @staticmethod
    def _normalize_feeder_ids(feeder_ids: np.ndarray) -> np.ndarray:
        labels = []
        for raw in np.asarray(feeder_ids).reshape(-1):
            if isinstance(raw, np.generic):
                raw = raw.item()
            if raw is None:
                labels.append(None)
                continue
            sval = str(raw).strip()
            if sval == "" or sval.lower() in {"nan", "none"}:
                labels.append(None)
            else:
                labels.append(sval)
        return np.array(labels, dtype=object)

    @staticmethod
    def _unique_valid_labels(labels: np.ndarray) -> List[str]:
        unique = []
        seen = set()
        for label in np.asarray(labels).reshape(-1):
            if label is None:
                continue
            sval = str(label)
            if sval not in seen:
                seen.add(sval)
                unique.append(sval)
        return unique

    @staticmethod
    def _sanitize_label_for_filename(label: Any) -> str:
        text = str(label).strip().lower()
        safe = []
        for ch in text:
            if ch.isalnum():
                safe.append(ch)
            else:
                safe.append('_')
        out = ''.join(safe)
        while '__' in out:
            out = out.replace('__', '_')
        out = out.strip('_')
        return out or 'group'

    def _compute_aggregated_profiles_per_feeder(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feeder_ids: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute AggregatedProfiles FD per feeder."""
        feeder_scores = {}
        feeder_ids = self._normalize_feeder_ids(feeder_ids)
        unique_feeders = self._unique_valid_labels(feeder_ids)

        for feeder in unique_feeders:
            n_group = int(np.sum(feeder_ids == feeder))
            if n_group < self.min_samples_per_feeder:
                continue

            score = self._compute_subset_aggregated_profiles(
                real_data[feeder_ids == feeder],
                synthetic_data[feeder_ids == feeder],
                group_type='feeder',
                group_value=str(feeder),
            )
            if score is None:
                continue

            feeder_scores[str(feeder)] = {
                'fd': float(score),
                'n_samples': n_group,
            }

        return feeder_scores

    def _compute_classical_stats_for_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute classical stats for [N, window_size] consumption windows."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected [N, {self.window_size}] array for classical stats, got {data.shape}"
            )

        mean_profile = np.mean(data, axis=0)
        std_profile = np.std(data, axis=0)

        return {
            'mean_profile': mean_profile.tolist(),
            'std_profile': std_profile.tolist(),
            'overall_mean': float(np.mean(data)),
            'overall_std': float(np.std(data)),
            'overall_min': float(np.min(data)),
            'overall_max': float(np.max(data)),
            'quantiles': {
                '25%': float(np.percentile(data, 25)),
                '50%': float(np.percentile(data, 50)),
                '75%': float(np.percentile(data, 75)),
            }
        }

    def _compute_subset_classical_stats(self, real_subset: np.ndarray, synthetic_subset: np.ndarray):
        """Compute full classical stats for one subset; return None if too few samples."""
        if len(real_subset) < self.min_samples_per_group or len(synthetic_subset) < self.min_samples_per_group:
            return None

        try:
            real_stats = self._compute_classical_stats_for_data(real_subset)
            synthetic_stats = self._compute_classical_stats_for_data(synthetic_subset)
            real_mean_profile = np.array(real_stats['mean_profile'])
            synth_mean_profile = np.array(synthetic_stats['mean_profile'])
            real_std_profile = np.array(real_stats['std_profile'])
            synth_std_profile = np.array(synthetic_stats['std_profile'])

            return {
                'n_profiles': int(min(len(real_subset), len(synthetic_subset))),
                'real': real_stats,
                'synthetic': synthetic_stats,
                'differences': {
                    'overall_mean_abs_diff': float(abs(real_stats['overall_mean'] - synthetic_stats['overall_mean'])),
                    'overall_std_abs_diff': float(abs(real_stats['overall_std'] - synthetic_stats['overall_std'])),
                    'mean_profile_mae': float(np.mean(np.abs(real_mean_profile - synth_mean_profile))),
                    'std_profile_mae': float(np.mean(np.abs(real_std_profile - synth_std_profile))),
                }
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: ClassicalStatistics computation failed: {e}")
            return None

    def _plot_aggregated_profiles_bars(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        days_of_week: np.ndarray,
        months_of_year: np.ndarray,
        feeder_ids: np.ndarray = None,
    ) -> Dict[str, str]:
        """Create bar charts of aggregated real vs synthetic consumption by day and month."""
        real_data = _sanitize_numeric_array(real_data)
        synthetic_data = _sanitize_numeric_array(synthetic_data)

        base_dir = _resolve_output_directory(self.output_path)
        # plots_dir = os.path.join(base_dir, 'Plots') if base_dir else 'Plots'
        plots_dir = base_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot style
        plot_style = PlotStyle(figsize=(14, 6))
        color_real = plot_style.palette_colors[0]
        color_synthetic = plot_style.palette_colors[4]
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        # Sum total consumption (all windows + all 24 hours) per day-of-week
        day_labels = []
        day_real_totals = []
        day_synth_totals = []
        day_counts = []
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                day_labels.append(day_names[day])
                day_real_totals.append(float(np.sum(real_data[mask])))
                day_synth_totals.append(float(np.sum(synthetic_data[mask])))
                day_counts.append(int(np.sum(mask)))

        day_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_day_of_week.pdf')
        if day_labels:
            x = np.arange(len(day_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, day_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, day_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Day of Week ({sum(day_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(day_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(day_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption (all windows + all 24 hours) per month
        month_labels = []
        month_real_totals = []
        month_synth_totals = []
        month_counts = []
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                month_labels.append(month_names[month])
                month_real_totals.append(float(np.sum(real_data[mask])))
                month_synth_totals.append(float(np.sum(synthetic_data[mask])))
                month_counts.append(int(np.sum(mask)))

        month_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_month_of_year.pdf')
        if month_labels:
            x = np.arange(len(month_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, month_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, month_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Month of Year ({sum(month_counts)} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month of Year', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(month_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(month_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Sum total consumption per feeder (if feeder_ids provided)
        feeder_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_feeder.pdf')
        if feeder_ids is not None:
            feeder_ids = self._normalize_feeder_ids(feeder_ids)
            unique_feeders = self._unique_valid_labels(feeder_ids)
            
            if unique_feeders:
                feeder_labels = []
                feeder_real_totals = []
                feeder_synth_totals = []
                feeder_counts = []
                for feeder in unique_feeders:
                    mask = feeder_ids == feeder
                    if np.sum(mask) > 0:
                        feeder_labels.append(str(feeder))
                        feeder_real_totals.append(float(np.sum(real_data[mask])))
                        feeder_synth_totals.append(float(np.sum(synthetic_data[mask])))
                        feeder_counts.append(int(np.sum(mask)))
                
                if feeder_labels:
                    x = np.arange(len(feeder_labels))
                    width = 0.38
                    # fig, ax = plt.subplots(figsize=(max(12, len(feeder_labels)), 6))
                    fig, ax = plt.subplots(figsize=plot_style.figsize)
                    ax.bar(x - width / 2, feeder_real_totals, width, label='Real', 
                           color=color_real, alpha=0.9, linewidth=1.2)
                    ax.bar(x + width / 2, feeder_synth_totals, width, label='Synthetic', 
                           color=color_synthetic, alpha=0.9, linewidth=1.2)
                    ax.set_title(f'Aggregated Consumption by Feeder ({sum(feeder_counts)} profiles)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Feeder ID', fontsize=12)
                    ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
                    ax.set_xticks(x)
                    ax.set_xticklabels(feeder_labels, rotation=45, ha='right', fontsize=10)
                    ax.legend(fontsize=11)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    plt.savefig(feeder_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

        # Split into input timesteps to see if it has an easier time predicting certain hours of the day (e.g., morning vs evening) 
        hour_plot_path = os.path.join(plots_dir, 'aggregated_profiles_bar_hour_of_day.pdf')

        input_length = real_data.shape[1]
        if input_length == self.window_size:
            hour_labels = [f"Hour {i}" for i in range(self.window_size)]
            hour_real_totals = np.sum(real_data, axis=0)
            hour_synth_totals = np.sum(synthetic_data, axis=0)

            x = np.arange(len(hour_labels))
            width = 0.38
            fig, ax = plt.subplots(figsize=plot_style.figsize)
            ax.bar(x - width / 2, hour_real_totals, width, label='Real', 
                   color=color_real, alpha=0.9, linewidth=1.2)
            ax.bar(x + width / 2, hour_synth_totals, width, label='Synthetic', 
                   color=color_synthetic, alpha=0.9, linewidth=1.2)
            ax.set_title(f'Aggregated Consumption by Hour of Day ({real_data.shape[0]} profiles)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Aggregated Consumption (sum)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(hour_labels, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(hour_plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        return {
            'day_of_week': day_plot_path,
            'month_of_year': month_plot_path,
            'feeder': feeder_plot_path if feeder_ids is not None else None,
            'hour_of_day': hour_plot_path,
        }

    def compute(self) -> float:
        """Run per-condition MMD, ACFDistribution, and TSTR and save results."""
        if self.real_data is None or self.synthetic_data is None:
            raise ValueError("PerConditionEval requires injected real_data and synthetic_data.")
        if self.timestamps is None:
            raise ValueError("PerConditionEval requires injected timestamps for per-condition grouping.")

        self.mmd_calculator.output_path = self.output_path
        self.acf_calculator.output_path = self.output_path
        self.tstr_calculator.output_path = self.output_path
        self.aggregated_profiles_calculator.output_path = self.output_path

        real_data = _sanitize_numeric_array(np.asarray(self.real_data))
        synthetic_data = _sanitize_numeric_array(np.asarray(self.synthetic_data))
        timestamps = np.asarray(self.timestamps)

        if len(real_data) != len(synthetic_data):
            synthetic_data = synthetic_data[:len(real_data)]

        # Ensure lengths match before creating masking arrays
        min_len = min(len(real_data), len(timestamps))
        real_data = real_data[:min_len]
        synthetic_data = synthetic_data[:min_len]
        timestamps = timestamps[:min_len]

        if real_data.ndim != 2 or synthetic_data.ndim != 2:
            raise ValueError(f"Expected 2D arrays [N, D], got real={real_data.shape}, synthetic={synthetic_data.shape}")
        if real_data.shape[1] != self.window_size or synthetic_data.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window shape [N, {self.window_size}], got real={real_data.shape}, synthetic={synthetic_data.shape}"
            )
        if self.verbose:
            self._print_input_debug(real_data, synthetic_data, timestamps)

        days_of_week = []
        months_of_year = []
        for ts in timestamps:
            dt = datetime.utcfromtimestamp(ts)
            days_of_week.append(dt.weekday())
            months_of_year.append(dt.month)
        days_of_week = np.array(days_of_week)
        months_of_year = np.array(months_of_year)

        # Resolve feeder IDs if available
        feeder_ids = self._resolve_feeder_ids(len(real_data))

        aggregated_profile_plot_paths = self._plot_aggregated_profiles_bars(
            real_data=real_data,
            synthetic_data=synthetic_data,
            days_of_week=days_of_week,
            months_of_year=months_of_year,
            feeder_ids=feeder_ids,
        )

        # -------------------------
        # MMD 
        # -------------------------
        mmd_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_day[day] = self.mmd_calculator.compute()
                else:
                    mmd_per_day[day] = 0.0

        mmd_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                if len(real_data[mask]) >= 2 and len(synthetic_data[mask]) >= 2:
                    self.mmd_calculator.real_data = real_data[mask]
                    self.mmd_calculator.synthetic_data = synthetic_data[mask]
                    mmd_per_month[month] = self.mmd_calculator.compute()
                else:
                    mmd_per_month[month] = 0.0

        # -------------------------
        # ACF Distribution 
        # -------------------------
        acf_per_day = {}
        for day in range(7):
            mask = days_of_week == day
            if np.sum(mask) > 0:
                acf_per_day[day] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        acf_per_month = {}
        for month in range(1, 13):
            mask = months_of_year == month
            if np.sum(mask) > 0:
                acf_per_month[month] = self._compute_subset_acf(real_data[mask], synthetic_data[mask])

        # -------------------------
        # Classical Statistics
        # -------------------------
        classical_stats_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                classical_stats_per_day[day] = self._compute_subset_classical_stats(
                    real_data[days_of_week == day], synthetic_data[days_of_week == day]
                )

        classical_stats_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                classical_stats_per_month[month] = self._compute_subset_classical_stats(
                    real_data[months_of_year == month], synthetic_data[months_of_year == month]
                )

        # -------------------------
        # Aggregated Profiles (Flexibility Distance)
        # -------------------------
        aggregated_profiles_per_day = {}
        for day in range(7):
            if np.sum( days_of_week == day) > 0:
                aggregated_profiles_per_day[day] = self._compute_subset_aggregated_profiles(
                    real_data[ days_of_week == day],
                    synthetic_data[ days_of_week == day],
                    group_type='day_of_week',
                    group_value=day,
                )

        aggregated_profiles_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                aggregated_profiles_per_month[month] = self._compute_subset_aggregated_profiles(
                    real_data[months_of_year == month],
                    synthetic_data[months_of_year == month],
                    group_type='month_of_year',
                    group_value=month,
                )

        aggregated_profiles_per_feeder = {}
        feeder_ids = self._resolve_feeder_ids(len(real_data))
        if feeder_ids is not None:
            aggregated_profiles_per_feeder = self._compute_aggregated_profiles_per_feeder(
                real_data=real_data,
                synthetic_data=synthetic_data,
                feeder_ids=feeder_ids,
            )

        # -------------------------
        # TSTR (Train on Synthetic, Test on Real)
        # -------------------------
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        tstr_per_day = {}
        for day in range(7):
            if np.sum(days_of_week == day) > 0:
                tstr_per_day[day] = self._compute_subset_tstr(
                    real_data[days_of_week == day],
                    synthetic_data[days_of_week == day],
                    group_label=f"day_of_week={day_names[day]} (day {day})"
                )

        tstr_per_month = {}
        for month in range(1, 13):
            if np.sum(months_of_year == month) > 0:
                tstr_per_month[month] = self._compute_subset_tstr(
                    real_data[months_of_year == month],
                    synthetic_data[months_of_year == month],
                    group_label=f"month_of_year={month_names[month]} (month {month})"
                )


        # -------------------------
        # Save results and print
        # -------------------------
        self.results_by_condition['mmd']['day_of_week'] = mmd_per_day
        self.results_by_condition['mmd']['month_of_year'] = mmd_per_month
        self.results_by_condition['acf']['day_of_week'] = acf_per_day
        self.results_by_condition['acf']['month_of_year'] = acf_per_month
        self.results_by_condition['classical_statistics']['day_of_week'] = classical_stats_per_day
        self.results_by_condition['classical_statistics']['month_of_year'] = classical_stats_per_month
        self.results_by_condition['aggregated_profiles']['day_of_week'] = aggregated_profiles_per_day
        self.results_by_condition['aggregated_profiles']['month_of_year'] = aggregated_profiles_per_month
        self.results_by_condition['aggregated_profiles']['feeder'] = aggregated_profiles_per_feeder
        self.results_by_condition['tstr']['day_of_week'] = tstr_per_day
        self.results_by_condition['tstr']['month_of_year'] = tstr_per_month

        # Save file
        output_dir = _resolve_output_directory(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        combined_output_file = os.path.join(output_dir, 'eval_results_condition.json')
        existing_payload = {}
        if os.path.exists(combined_output_file):
            try:
                with open(combined_output_file, 'r') as f:
                    existing_payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_payload = {}

        output_payload = {
            'mmd': self.results_by_condition['mmd'],
            'acf': self.results_by_condition['acf'],
            'classical_statistics': self.results_by_condition['classical_statistics'],
            'aggregated_profiles': self.results_by_condition['aggregated_profiles'],
            'tstr': self.results_by_condition['tstr'],
            'aggregated_profile_plots': aggregated_profile_plot_paths,
            'feeder_mapping_path': self.feeder_mapping_path,
            'min_samples_per_feeder': int(self.min_samples_per_feeder),
            'window_size': int(self.window_size),
            'max_lag': int(self.max_lag),
            'min_samples_per_group': int(self.min_samples_per_group),
            'predictor_type': self.predictor_type,
        }

        # Preserve standalone metric entries written by individual metric classes
        if isinstance(existing_payload, dict) and 'standalone_metrics' in existing_payload:
            output_payload['standalone_metrics'] = existing_payload['standalone_metrics']

        with open(combined_output_file, 'w') as f:
            json.dump(_to_json_serializable(output_payload), f, indent=2)

        all_mmd_scores = list(mmd_per_day.values()) + list(mmd_per_month.values())
        avg_mmd = np.mean(all_mmd_scores) if all_mmd_scores else 0.0
        return float(avg_mmd)
