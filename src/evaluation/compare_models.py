

import os
import sys
import torch
import random
import json
import numpy as np
from pathlib import Path
import argparse

# Add project roots for imports
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.evaluation.load_model import load_model
from src.model.factory import _load_train_config
from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
from src.visualization.plot_styling import PlotStyle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not installed
    def tabulate(data, headers="keys", **kwargs):
        return str(data)

class PhasedModelComparator:
    """Class to compare evaluation results across multiple phased and unphased models."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, output_dir: str = "evaluation_comparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_data = {}
        # We define a superset of possible phases.
        self.phases = ['full_eval', 'aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']

    def _metric_target_label(self, metric: str) -> str:
        """Return target direction/target value for plot titles."""
        display_name = self._display_metric_name(metric)

        if metric.startswith("Avg MinMax") or "BEST Overall" in display_name:
            return "↑"

        if "TSTR" in metric or "Train on Synthetic" in display_name:
            return "target=1"

        if "MMD" in metric or "MaximumMeanDiscrepancy" in metric or "Maximum Mean Discrepancy" in display_name:
            return "target=0"

        if any(term in metric for term in ["LogLikelihood", "Log-Likelihood", "Accuracy", "Score", "R2"]):
            return "↑"

        return "↓"

    def _display_metric_title(self, metric: str) -> str:
        return f"{self._display_metric_name(metric)} ({self._metric_target_label(metric)})"

    def _display_metric_name(self, metric: str) -> str:

        """Pretty metric names for tables and plots."""
        rename_map = {
            "Flexibility Distance": "Reconstruction Flexibility Distance (FD)",
            "Avg MinMax Reconstruction Score": "BEST Overall Reconstruction Score",
            "Avg MinMax Synthetic Score": "BEST Overall Synthetic Score",
            "TSTR (Train on Synthetic, Test on Real)": "Train on Synthetic, Test on Real (TSTR)",
            "Aggregated Profiles (Flexibility Distance)": "FD on Real & Synth Aggregated Profiles",
            "Set-MMD (Permutation Invariant)": "Maximum Mean Discrepancy (MMD)",
        }
        return rename_map.get(metric, metric)
    
    def _include_metric(self, metric: str) -> bool:
        excluded_metrics = {
            "Classical Statistics",
        }
        return metric not in excluded_metrics

    def _model_color(self, model_name: str, index: int, plot_style):
        name = model_name.lower()

        if "avg baseline" in name or "mean baseline" in name or "averagebaseline" in name:
            return "0.55"   # medium grey

        if "quantile baseline" in name or "quantilebaseline" in name:
            return "0.25"   # dark grey

        return plot_style.get_color(index)
    

    def _canonical_metric_key(self, metric: str) -> str:
        """Normalize metric names for robust matching."""
        return (
            str(metric)
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )


    def _metric_goal(self, metric: str) -> str:
        """
        Return one of:
            'lower'     -> lower is better
            'higher'    -> higher is better
            'target_0'  -> closer to 0 is better
            'target_1'  -> closer to 1 is better
        """
        key = self._canonical_metric_key(metric)

        goal_map = {
            # Reconstruction metrics
            "flexibilitydistance": "lower",
            "reconstructionflexibilitydistancefd": "lower",
            "reconstructionloglikelihood": "higher",
            "reconstructionloglikelihood": "higher",
            "reconstructionrmse": "lower",

            # Synthetic metrics
            "aggregatedprofilesflexibilitydistance": "lower",
            "perconditionevaluation": "lower",
            "acfdistributionmatchks": "lower",
            "tstrtrainonsynthetictestonreal": "target_1",
            "trainonsynthetictestonrealtstr": "target_1",
            "tstr": "target_1",
            "maximummeandiscrepancymmd": "target_0",
            "setmmdpermutationinvariant": "target_0",
            "mmd": "target_0",
            "hourlyconsumptionbymonth": "lower",
            "hourlyconsumptionbyday": "lower",
            "peakanalysis": "lower",

            # Summary scores
            "avgminmaxreconstructionscore": "higher",
            "avgminmaxsyntheticscore": "higher",
        }

        return goal_map.get(key, "lower")


    def _metric_group(self, metric: str) -> str | None:
        """
        Return:
            'reconstruction'
            'synthetic'
            None -> excluded from AvgMinMax scores
        """
        key = self._canonical_metric_key(metric)

        reconstruction_metrics = {
            "flexibilitydistance",
            "reconstructionflexibilitydistancefd",
            "reconstructionloglikelihood",
            "reconstructionrmse",
        }

        synthetic_metrics = {
            "aggregatedprofilesflexibilitydistance",
            "tstrtrainonsynthetictestonreal",
            "trainonsynthetictestonrealtstr",
            "tstr",
            "maximummeandiscrepancymmd",
            "setmmdpermutationinvariant",
            "mmd",
            "acfdistributionmatchks",
            "hourlyconsumptionbymonth",
            "hourlyconsumptionbyday",
            "peakanalysis"
        }

        if key in reconstruction_metrics:
            return "reconstruction"

        if key in synthetic_metrics:
            return "synthetic"

        return None


    def _metric_target_label(self, metric: str) -> str:
        goal = self._metric_goal(metric)

        if goal == "higher":
            return "↑"
        if goal == "lower":
            return "↓"
        if goal == "target_0":
            return "target=0"
        if goal == "target_1":
            return "target=1"

        return "↓"


    def _display_metric_title(self, metric: str) -> str:
        return f"{self._display_metric_name(metric)} ({self._metric_target_label(metric)})"

    def add_model(self, model_name: str, model_eval_dir: str):
        """Add a model's evaluation results to the comparator, handling both phased and unphased output."""
        self.models_data[model_name] = {}
        
        # 1. Check for phased results in subdirectories
        found_phased = False
        for phase in ['aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']:
            eval_json_path = os.path.join(model_eval_dir, f"{phase}_eval", "eval_results.json")
            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, 'r') as f:
                        data = json.load(f)
                    self.models_data[model_name][phase] = data
                    found_phased = True
                except Exception as e:
                    print(f"Error loading {eval_json_path}: {e}")
        
        # 2. Check for unphased results at root (either EvalFramework or PhasedEvaluationSuite summary)
        root_eval_path = os.path.join(model_eval_dir, "eval_results.json")
        if os.path.exists(root_eval_path):
            try:
                with open(root_eval_path, 'r') as f:
                    data = json.load(f)
                
                # If it's a simple flat dict (from EvalFramework), add it as 'full_eval'
                if not any(k in data for k in ['aggregated_phase', 'ranked_phases']):
                    self.models_data[model_name]['full_eval'] = data
            except Exception as e:
                print(f"Error loading {root_eval_path}: {e}")
        
        if not self.models_data[model_name]:
            print(f"No valid evaluation results found in {model_eval_dir} for model {model_name}")

    def compare(self):
        """Generate comparison tables and plots for each phase."""
        if not self.models_data:
            print("No models added. Exiting.")
            return

        for phase in self.phases:
            # Only show a phase in the comparison if ALL loaded models have data for it.
            # This prevents partial 'N/A' columns from appearing when one model ran in
            # unphased mode and another in phased mode.
            models_with_phase = [m for m in self.models_data if phase in self.models_data[m]]
            if models_with_phase and len(models_with_phase) == len(self.models_data):
                print(f"\n--- Comparing models for phase: {phase} ---")
                self._compute_averages(phase)
                self._generate_table(phase)
                self._generate_plots(phase)
            elif models_with_phase:
                missing = [m for m in self.models_data if phase not in self.models_data[m]]
                print(f"\n[Skipping phase '{phase}': missing for models {missing}]")

    def _compute_averages(self, phase: str):
        """Compute normalized min-max average scores for reconstruction and synthetic metrics."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)
                        
        scalar_metrics = list(all_metrics)
        models = list(self.models_data.keys())

        preferred_last = [
            "Avg Baseline",
            "Mean Baseline",
            "Quantile Baseline",
        ]
        non_last = [m for m in models if m not in preferred_last]
        last = [m for m in preferred_last if m in models]
        models = non_last + last
        
        # --- Normalization for Average Plotting ---
        scores_dict = {m: {} for m in models}
        for metric in scalar_metrics:
            goal = self._metric_goal(metric)

            raw_vals = []
            for m in models:
                v = self.models_data[m].get(phase, {}).get(metric, None)
                raw_vals.append((m, v))

            mapped_vals = {}
            for m, v in raw_vals:
                if isinstance(v, (int, float)):
                    v = float(v)

                    # Convert everything so lower is internally better
                    if goal == "higher":
                        mapped_vals[m] = -v
                    elif goal == "target_1":
                        mapped_vals[m] = abs(v - 1.0)
                    elif goal == "target_0":
                        mapped_vals[m] = abs(v - 0.0)
                    else:  # lower
                        mapped_vals[m] = v
                                
            if mapped_vals:
                vmin = min(mapped_vals.values())
                vmax = max(mapped_vals.values())
                for m, val in mapped_vals.items():
                    if vmax == vmin:
                        scores_dict[m][metric] = 1.0
                    else:
                        # 1.0 is the best score, 0.0 is the worst score
                        scores_dict[m][metric] = 1.0 - (val - vmin) / (vmax - vmin)

        recon_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synth_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]
        
        for m in models:
            if phase not in self.models_data[m]:
                continue
            recon_scores = [scores_dict[m][met] for met in recon_metrics if met in scores_dict[m]]
            synth_scores = [scores_dict[m][met] for met in synth_metrics if met in scores_dict[m]]
            
            if recon_scores:
                self.models_data[m][phase]['Avg MinMax Reconstruction Score'] = float(np.mean(recon_scores))
            if synth_scores:
                self.models_data[m][phase]['Avg MinMax Synthetic Score'] = float(np.mean(synth_scores))

    def _generate_table(self, phase: str):
        """Generate and print a comparison table of scalar metrics for a given phase."""
        headers = ["Metric"] + list(self.models_data.keys())
        
        # Collect all unique metric names
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)
                        
        # scalar_metrics = sorted(list(all_metrics))
        scalar_metrics = sorted(all_metrics, key=lambda m: (0 if "Reconstruction" in m or "Flexibility" in m else 1, m))
        
        # Move averages to the end
        averages = [m for m in scalar_metrics if m.startswith('Avg MinMax')]
        non_averages = [m for m in scalar_metrics if not m.startswith('Avg MinMax')]
        scalar_metrics = non_averages + averages
                
        table_data = []
        for metric in scalar_metrics:
            # row = [metric]
            row = [self._display_metric_name(metric)]
            for model_name, data in self.models_data.items():
                if phase in data and metric in data[phase]:
                    val = data[phase][metric]
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("N/A")
            table_data.append(row)
            
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nModel Comparison Table ({phase}):")
        print(table_str)
        
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        table_path = os.path.join(phase_output_dir, "comparison_table.txt")
        with open(table_path, "w") as f:
            f.write(table_str)
        print(f"\nTable saved to {table_path}")

    def _generate_plots(self, phase: str):
        """Generate bar charts for scalar metrics for a given phase, split by reconstruction and synthetic metrics."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)
                
        scalar_metrics = list(all_metrics)
        if not scalar_metrics:
            return
        
        # Separate metrics by group
        reconstruction_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synthetic_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]
        
        # Add their respective BEST metrics at the end
        reconstruction_best = "Avg MinMax Reconstruction Score"
        synthetic_best = "Avg MinMax Synthetic Score"
        
        if reconstruction_best in scalar_metrics:
            reconstruction_metrics.append(reconstruction_best)
        if synthetic_best in scalar_metrics:
            synthetic_metrics.append(synthetic_best)
        
        # Sort each group (BEST metrics come last)
        reconstruction_metrics = sorted(reconstruction_metrics, key=lambda m: (1 if m == reconstruction_best else 0, m))
        synthetic_metrics = sorted(synthetic_metrics, key=lambda m: (1 if m == synthetic_best else 0, m))
        
        models = list(self.models_data.keys())
        preferred_last = [
            "Avg Baseline",
            "Mean Baseline",
            "Quantile Baseline",
        ]
        non_last = [m for m in models if m not in preferred_last]
        last = [m for m in preferred_last if m in models]
        models = non_last + last
        
        # Generate plots for each metric group
        plot_style = PlotStyle()
        for metric_group, group_name in [(reconstruction_metrics, "Reconstruction"), (synthetic_metrics, "Synthetic")]:
            if not metric_group:
                continue
            
            self._generate_metric_group_plot(phase, metric_group, group_name, models, plot_style)
    
    def _generate_metric_group_plot(self, phase: str, metric_group: list, group_name: str, models: list, plot_style):
        """Generate a single bar chart figure for a metric group."""
        n_metrics = len(metric_group)
        n_cols = 3
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle the case of a single subplot or 1D array of subplots
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Add header title to the figure
        title_map = {
            "Reconstruction": "Reconstruction Evaluation",
            "Synthetic": "Synthetic Data Evaluation",
        }
        fig.suptitle(title_map.get(group_name, group_name), fontsize=16, fontweight='bold', y=0.995)
            
        width = 0.8 / len(models)
        
        for idx, metric in enumerate(metric_group):
            ax = axes[idx]
            
            # Determine if we want to min/max this metric
            goal = self._metric_goal(metric)
            
            m_values = []
            valid_models = []
            
            for i, model in enumerate(models):
                if phase in self.models_data[model]:
                    if metric in self.models_data[model][phase]:
                        val = self.models_data[model][phase][metric]
                        if isinstance(val, (int, float)):
                            m_values.append(val)
                            valid_models.append(i)
                        else:
                            m_values.append(np.nan)
                    else:
                        m_values.append(np.nan)
                else:
                    m_values.append(np.nan)
            
            # Find best value based on criteria
            best_idx = -1
            best_val = None
            if len(valid_models) > 0:
                valid_vals = [m_values[i] for i in valid_models]
                if goal == "target_1":
                    best_val = min(valid_vals, key=lambda v: abs(v - 1.0))
                elif goal == "target_0":
                    best_val = min(valid_vals, key=lambda v: abs(v - 0.0))
                elif goal == "higher":
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)
                    
                best_idx = valid_models[valid_vals.index(best_val)]
            
            # One x-position for the metric
            x = np.arange(1) 
            
            for i in range(len(models)):
                if not np.isnan(m_values[i]):
                    val = m_values[i]
                    color = self._model_color(models[i], i, plot_style)
                    is_best = (i == best_idx)
                    
                    ec = 'black' if is_best else 'none'
                    lw = 1.5 if is_best else 0
                    hatch = '///' if is_best else ''
                    
                    # Offset the bars around the center
                    bar_x = x + i * width - (len(models)-1)*width/2
                    ax.bar(bar_x, [val], width, color=color, edgecolor=ec, linewidth=lw, hatch=hatch)
                    
                    # Annotate the best bar with a star
                    if is_best:
                        va = 'bottom' if val >= 0 else 'top'
                        offset_sign = 1 if val >= 0 else -1
                        ax.annotate('★', xy=(bar_x[0], val), xytext=(0, 3 * offset_sign), 
                                    textcoords='offset points', ha='center', va=va, 
                                    fontsize=12, color='black', alpha=0.8)
            
            ax.set_title(self._display_metric_title(metric), fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Create a custom legend
        legend_handles = [
            mpatches.Patch(color=self._model_color(model, i, plot_style), label=model)
            for i, model in enumerate(models)
        ]
        legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='★ Best Value'))
        
        # Position the legend above the plots
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(models)+1))
        
        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        plot_path = os.path.join(phase_output_dir, f"metric_comparison_{group_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {plot_path}")

def get_eval_files(data_dir: str, data_files: list[str] | None = None):
    """
    Simplified file discovery: if data_files is None, 
    grab all .parquet files in data_dir that don't start with 'test_'.
    """
    if data_files is not None and len(data_files) > 0:
        # Ensure absolute paths
        return [os.path.join(data_dir, f) if not os.path.isabs(f) else f for f in data_files]
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    discovered = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".parquet") and not f.startswith("test_"):
                discovered.append(os.path.join(root, f))
    
    if not discovered:
        raise FileNotFoundError(f"No .parquet files (not starting with 'test_') found in {data_dir}")
        
    # Deterministic ordering by default to avoid data variance across runs/models.
    return sorted(discovered)

def _find_model_config_path(model_dir: str) -> str | None:
    """Return a config yaml path inside a trained model directory if present."""
    if not os.path.isdir(model_dir):
        return None
    # Prefer the snapshot if training wrote it.
    snap = os.path.join(model_dir, "training_config_used.yaml")
    if os.path.exists(snap):
        return snap
    for f in os.listdir(model_dir):
        if f.endswith(".yaml") or f.endswith(".yml"):
            return os.path.join(model_dir, f)
    return None

def _infer_split_from_first_model(model_paths: list[str]) -> tuple[int, dict]:
    """Infer split seed/ratios from the first model's config as a sensible default."""
    if not model_paths:
        return 0, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    cfg_path = _find_model_config_path(model_paths[0])
    if cfg_path is None:
        return 0, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    cfg = _load_train_config(cfg_path).data
    seed = int(getattr(cfg.splits, "random_seed", 0))
    ratios = {
        "train_ratio": float(getattr(cfg.splits, "train_ratio", 0.8)),
        "val_ratio": float(getattr(cfg.splits, "val_ratio", 0.1)),
        "test_ratio": float(getattr(cfg.splits, "test_ratio", 0.1)),
    }
    return seed, ratios

def _has_existing_eval_outputs(model_output_dir: str, phased: bool) -> bool:
    """Return True if evaluation artifacts already exist on disk for this model output folder."""
    if phased:
        phases = ["aggregated_phase", "ranked_phase_1", "ranked_phase_2", "ranked_phase_3"]
        for phase in phases:
            p = os.path.join(model_output_dir, f"{phase}_eval", "eval_results.json")
            if os.path.exists(p):
                return True
        return os.path.exists(os.path.join(model_output_dir, "eval_results.json"))
    return os.path.exists(os.path.join(model_output_dir, "eval_results.json"))

def compare_models(model_paths: list[str], 
                   model_names: list[str], 
                   secondary_model_paths: list[str] | None = None,
                   data_dir: str = "data/_parquet_cache",
                   data_files: list[str] | None = None,
                   num_samples: int = 100,
                   output_dir: str = "evaluation_comparison",
                   phased: bool = False,
                   agg_divisor = 1000.0,
                   recompute: bool = True,
                   split_seed: int | None = None,
                   split_ratios: dict | None = None,
                   max_value: float | None = 200.0,
                   ):
    """
    Run evaluation for multiple models and compare results.
    Can handle both phased and unphased models.
    """

    # 1. Prepare evaluation data
    if data_files is None:
        data_files = get_eval_files(data_dir)

    # 2. Pick a single global test split definition so all models see identical windows.
    if split_seed is None or split_ratios is None:
        inferred_seed, inferred_ratios = _infer_split_from_first_model(model_paths)
        split_seed = inferred_seed if split_seed is None else int(split_seed)
        split_ratios = inferred_ratios if split_ratios is None else split_ratios
    split_seed = int(split_seed)
    split_ratios = dict(split_ratios)
    
    # 4. Loop through models and evaluate
    for i, (path, name, secondary) in enumerate(zip(model_paths, model_names, secondary_model_paths if secondary_model_paths is not None else [None]*len(model_paths))):
        print(f"\nEvaluating model: {name}")
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

        if (not recompute) and _has_existing_eval_outputs(model_output_dir, phased=phased) and secondary is not None:
            print(f"Found existing evaluation outputs in {model_output_dir}; skipping recompute.")
            continue


        model, normalizer, conditioner = load_model(path, device="cuda" if torch.cuda.is_available() else "cpu")
        sec_model = None
        if secondary is not None:
            sec_model, normalizer, conditioner = load_model(secondary, device="cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load Data
        window = WindowSpec(input_len=24, stride=24)
        loader_obj = PhasedReplayDataloaders(
            sm_files=data_files,
            window_spec=window,
            batch_size=32,
            replay_buffer_capacity=1000,
            min_buffer_size=0,
            normalizer=normalizer,
            num_workers=0,
            phased_mode=phased,
            # Deterministic test selection + ordering.
            sample_random=False,
            feature_cols=["hourly_energy_consumption",],
            meter_col = "meter_number",
            aggregation_divisor=agg_divisor,
            hourly_aggregation=True if agg_divisor > 1.0 else False,
            split_seed=split_seed,
            split_ratios=split_ratios,
            max_value=max_value,
            pin_memory=False
        )
        
        # 3. Grab a fixed number of samples for evaluation
        n_batches = max(1, num_samples // 32)
        # True held-out test split (not used during training).
        loader_obj.refresh_test_cache(n_batches)
        test_loader = loader_obj.get_test_loader()

        if test_loader is None and hasattr(loader_obj, "test_dataset"):
            print("[WARN] get_test_loader() returned None, constructing DataLoader from test_dataset directly.")
            from torch.utils.data import DataLoader
            test_loader = DataLoader(
                loader_obj.test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

        if phased:
            # Phased models use PhasedEvaluationSuite which creates sub-phase directories
            evaluator = PhasedEvaluationSuite(
                model=model,
                dataloader=test_loader,
                normalizer=normalizer,
                conditioner=conditioner,
                device="cuda" if torch.cuda.is_available() else "cpu",
                window_size=window.input_len,
                output_path=model_output_dir,
                secondary_model=sec_model
            )
            evaluator.evaluate()
        else:
            print("\nDEBUG BEFORE EVAL")
            print("model_name:", name)
            print("model_path:", path)
            print("loaded_model_type:", type(model))
            print("data_dir:", data_dir)
            print("data_files:", data_files)
            print("phased:", phased)
            print("recompute:", recompute)
            print("n_batches:", n_batches)
            print("loader_obj_type:", type(loader_obj))
            print("test_loader:", test_loader)
            print("type(test_loader):", type(test_loader))

            # Unphased models use EvalFramework directly
            evaluator = EvalFramework(
                model=model,
                dataloader=test_loader,
                normalizer=normalizer,
                conditioner=conditioner,
                device="cuda" if torch.cuda.is_available() else "cpu",
                window_size=window.input_len,
                output_path=os.path.join(model_output_dir, "eval_results.json"),
                secondary_model=sec_model
            )
            # evaluator.register_all(phased=False)
            # evaluator.evaluate()
            # evaluator.save_results(os.path.join(model_output_dir, "eval_results.json"))
            evaluator.register_all(phased=False)
            evaluator.evaluate()

            # ------------------------------------------------------------
            # DEBUG: print example real / reconstructed / synthetic data
            # ------------------------------------------------------------
            print("\n" + "=" * 80)
            print(f"DATA EXAMPLES USED IN EVALUATION: {name}")
            print("=" * 80)

            for attr in [
                "_real_data",
                "_reconstructed_data",
                "_synthetic_data",
                "real_data",
                "reconstructed_data",
                "synthetic_data",
            ]:
                if hasattr(evaluator, attr):
                    arr = getattr(evaluator, attr)

                    if arr is None:
                        print(f"{attr}: None")
                        continue

                    try:
                        print(f"\n{attr}")
                        print("type:", type(arr))
                        print("shape:", arr.shape)

                        # Print first sample
                        example = arr[0]
                        print("first sample shape:", example.shape if hasattr(example, "shape") else type(example))
                        print(example)

                        # If it is long, also print only first 24 values flattened
                        flat = np.asarray(example).reshape(-1)
                        print("first sample flattened first 24 values:")
                        print(flat[:24])

                    except Exception as e:
                        print(f"{attr}: could not print, error={e}")

            print("=" * 80 + "\n")

            evaluator.save_results(os.path.join(model_output_dir, "eval_results.json"))
    
    # 5. Use PhasedModelComparator to generate comparison tables and plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON")
    print("="*50)
    
    comparator = PhasedModelComparator(output_dir=output_dir)
    for name in model_names:
        comparator.add_model(name, os.path.join(output_dir, name))
    
    comparator.compare()
    
    print("\nComparison complete. Results saved to:", output_dir)
    return comparator.models_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model evaluation results (reuse outputs or recompute).")
    parser.add_argument("--models", nargs="+", required=True, help="List of model directories (each contains config + checkpoint).")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for models (same length as --models).")
    parser.add_argument("--secondary-models", nargs="*", default=None, help="Optional secondary model dirs aligned with --models. Use 'None' to skip.")
    parser.add_argument("--data-dir", type=str, default="data/_parquet_cache", help="Directory with parquet files.")
    parser.add_argument("--data-files", nargs="*", default=None, help="Optional explicit parquet files (relative to --data-dir unless absolute).")
    parser.add_argument("--num-samples", type=int, default=100, help="Approx number of windows to evaluate per model.")
    parser.add_argument("--output-dir", type=str, default="evaluation_comparison", help="Where to write/read evaluation outputs.")
    parser.add_argument("--phased", action="store_true", help="Use phased evaluation suite.")
    parser.add_argument("--recompute", action="store_true", help="Recompute evaluation even if outputs already exist.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed used for deterministic window-level train/val/test split. Defaults to first model's training config.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train ratio for split (defaults to first model's training config).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Val ratio for split (defaults to first model's training config).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio for split (defaults to first model's training config).")
    parser.add_argument("--max-value", type=float, default=200.0, help="Maximum value for filtering windows.")
    args = parser.parse_args()

    if len(args.models) != len(args.names):
        raise ValueError("--models and --names must have the same length.")
    
    secondary_model_paths = None
    if args.secondary_models is not None and len(args.secondary_models) > 0:
        secondary_model_paths = []
        for s in args.secondary_models:
            if s is None or str(s).lower() == "none":
                secondary_model_paths.append(None)
            else:
                secondary_model_paths.append(s)
        if len(secondary_model_paths) < len(args.models):
            secondary_model_paths.extend([None] * (len(args.models) - len(secondary_model_paths)))

    split_ratios = None
    if args.train_ratio is not None or args.val_ratio is not None or args.test_ratio is not None:
        split_ratios = {
            "train_ratio": 0.8 if args.train_ratio is None else float(args.train_ratio),
            "val_ratio": 0.1 if args.val_ratio is None else float(args.val_ratio),
            "test_ratio": 0.1 if args.test_ratio is None else float(args.test_ratio),
        }

    compare_models(
        model_paths=args.models,
        model_names=args.names,
        secondary_model_paths=secondary_model_paths,
        data_dir=args.data_dir,
        data_files=args.data_files,
        num_samples=int(args.num_samples),
        output_dir=args.output_dir,
        phased=bool(args.phased),
        recompute=bool(args.recompute),
        split_seed=args.split_seed,
        split_ratios=split_ratios,
        max_value=args.max_value
    )