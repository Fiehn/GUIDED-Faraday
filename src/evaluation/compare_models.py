

import os
import sys
import shutil
import torch
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
from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec, InMemoryCSVDataloaders
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite
from src.visualization.plot_styling import PlotStyle
from src.data.feeder_holdout import plot_feeder_aggregated_profile, plot_feeder_holdout_analysis, run_coincidence_factor_analysis

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
            "Avg MinMax Reconstruction Score": "MinMax Score (Reconstruction)",
            "Avg MinMax Synthetic Score": "MinMax Score (Synthetic)",
            "Avg Rank Reconstruction Score": "Avg Rank (Reconstruction)",
            "Avg Rank Synthetic Score": "Avg Rank (Synthetic)",
            "Win Count Reconstruction": "Win Count (Reconstruction)",
            "Win Count Synthetic": "Win Count (Synthetic)",
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
            .replace(",", "")
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
            "peakmagnitude": "lower",
            "peaktiming": "lower",

            # Summary scores
            "avgminmaxreconstructionscore": "higher",
            "avgminmaxsyntheticscore": "higher",
            "avgrankреconstructionscore": "lower",
            "avgrankreconstructionscore": "lower",
            "avgranksyntheticscore": "lower",
            "wincountreconstruction": "higher",
            "wincountsynthetic": "higher",
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
            "peakmagnitude"
        }

        summary_metrics = {
            "avgminmaxreconstructionscore", "avgminmaxsyntheticscore",
            "avgrankreconstructionscore", "avgranksyntheticscore",
            "wincountreconstruction", "wincountsynthetic",
        }

        if key in summary_metrics:
            return None  # never feed summary metrics back into themselves

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

    def add_model(self, model_name: str, model_dir: str):
        """Add a model's evaluation results to the comparator, handling both phased and unphased output."""
        self.models_data[model_name] = {}
        
        # 1. Check for phased results in subdirectories
        for phase in ['aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']:
            eval_json_path = os.path.join(model_dir, f"{phase}_eval", "eval_results.json")
            
            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, 'r') as f:
                        data = json.load(f)
                    self.models_data[model_name][phase] = data
                except Exception as e:
                    print(f"Error loading {eval_json_path}: {e}")
        
        # 2. Check for unphased results at root (either EvalFramework or PhasedEvaluationSuite summary)
        root_eval_path = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(root_eval_path):
            try:
                with open(root_eval_path, 'r') as f:
                    data = json.load(f)
                
                # If it's a simple flat dict (from EvalFramework), add it as 'full_eval'
                if not any(k in data for k in ['aggregated_phase', 'ranked_phases']):
                    self.models_data[model_name]['full_eval'] = data
            except Exception as e:
                print(f"Error loading {root_eval_path}: {e}")
        
        # 3. Check for feeder holdout results — stored as a separate 'holdout_eval' key,
        #    never injected into regular phase dicts.
        holdout_path = os.path.join(model_dir, "eval_results_feeder_holdout.json")
        if os.path.exists(holdout_path):
            import traceback as _tb
            try:
                with open(holdout_path, 'r') as f:
                    holdout_data = json.load(f)
                scalar_holdout = {k: v for k, v in holdout_data.items() if isinstance(v, (int, float))}
                if scalar_holdout:
                    self.models_data[model_name]['holdout_eval'] = scalar_holdout
                else:
                    print(f"Warning: {holdout_path} contains no scalar metrics.")
            except Exception as e:
                print(f"Error loading {holdout_path}: {e}")
                _tb.print_exc()

        if not self.models_data[model_name]:
            print(f"No valid evaluation results found in {model_dir} for model {model_name}")

    def _sort_models_by_synthetic_score(self, phase: str) -> list[str]:
        """Return model names sorted by Avg Rank Synthetic Score ascending (rank 1 = best first)."""
        score_key = "Avg Rank Synthetic Score"

        def _score(model_name: str) -> float:
            # Missing entries sort to the end (worst)
            return float(self.models_data.get(model_name, {}).get(phase, {}).get(score_key, float("inf")))

        return sorted(self.models_data.keys(), key=_score)

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
                sorted_models = self._sort_models_by_synthetic_score(phase)
                self._generate_table(phase, sorted_models)
                self._generate_plots(phase, sorted_models)
            elif models_with_phase:
                missing = [m for m in self.models_data if phase not in self.models_data[m]]
                print(f"\n[Skipping phase '{phase}': missing for models {missing}]")

        self._compare_holdout()

    def _compare_holdout(self):
        """Generate a separate comparison table and plots for feeder holdout results.

        Models without holdout results are shown as N/A — they are not excluded.
        No aggregate scores (MinMax/Rank/WinCount) are computed because not all
        models are guaranteed to have holdout data.
        """
        models_with_holdout = [m for m in self.models_data if 'holdout_eval' in self.models_data[m]]
        if not models_with_holdout:
            print("\n[No holdout eval results found for any model; skipping holdout comparison.]")
            return

        missing = [m for m in self.models_data if 'holdout_eval' not in self.models_data[m]]
        if missing:
            print(f"\n--- Holdout comparison (N/A for: {missing}) ---")
        else:
            print(f"\n--- Holdout comparison ({len(models_with_holdout)} models) ---")

        # Collect all scalar metrics present in any model's holdout results.
        all_metrics: set[str] = set()
        for m in models_with_holdout:
            all_metrics.update(self.models_data[m]['holdout_eval'].keys())

        if not all_metrics:
            print("  [holdout] No scalar metrics to compare.")
            return

        all_models = list(self.models_data.keys())
        scalar_metrics = sorted(all_metrics, key=lambda x: (0 if "Reconstruction" in x or "Flexibility" in x else 1, x))

        # --- Table ---
        headers = ["Metric"] + all_models
        table_data = []
        for metric in scalar_metrics:
            goal = self._metric_goal(metric)
            model_vals = {
                m: float(self.models_data[m]['holdout_eval'][metric])
                for m in models_with_holdout
                if metric in self.models_data[m]['holdout_eval']
            }
            best_models: set = set()
            if model_vals:
                if goal == "target_1":
                    best_raw = min(abs(v - 1.0) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v - 1.0) - best_raw) <= 1e-9}
                elif goal == "target_0":
                    best_raw = min(abs(v) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v) - best_raw) <= 1e-9}
                elif goal == "higher":
                    best_raw = max(model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= abs(best_raw) * 1e-9 + 1e-12}
                else:
                    best_raw = min(model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= abs(best_raw) * 1e-9 + 1e-12}

            row = [self._display_metric_name(metric)]
            for m in all_models:
                if 'holdout_eval' in self.models_data[m] and metric in self.models_data[m]['holdout_eval']:
                    val = self.models_data[m]['holdout_eval'][metric]
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    row.append(f"★ {fmt}" if m in best_models else fmt)
                else:
                    row.append("N/A")
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nHoldout Comparison Table:")
        print(table_str)

        holdout_dir = os.path.join(self.output_dir, "holdout_eval")
        os.makedirs(holdout_dir, exist_ok=True)
        table_path = os.path.join(holdout_dir, "comparison_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"Holdout table saved to {table_path}")

        # --- Plots (only models with holdout data) ---
        plot_style = PlotStyle()
        reconstruction_metrics = sorted(m for m in scalar_metrics if self._metric_group(m) == "reconstruction")
        synthetic_metrics = sorted(m for m in scalar_metrics if self._metric_group(m) == "synthetic")
        for metric_group, group_name in [(reconstruction_metrics, "Reconstruction"), (synthetic_metrics, "Synthetic")]:
            if metric_group:
                self._generate_metric_group_plot("holdout_eval", metric_group, group_name, models_with_holdout, plot_style)

    def _compute_averages(self, phase: str):
        """Compute three aggregate scores (min-max, rank, win count) for reconstruction and synthetic metrics."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)

        scalar_metrics = list(all_metrics)
        models = list(self.models_data.keys())

        preferred_last = ["Avg Baseline", "Mean Baseline", "Quantile Baseline"]
        non_last = [m for m in models if m not in preferred_last]
        last = [m for m in preferred_last if m in models]
        models = non_last + last

        def _comparable(v: float, goal: str) -> float:
            """Map to a value where lower is always better."""
            if goal == "higher":    return -v
            if goal == "target_1":  return abs(v - 1.0)
            if goal == "target_0":  return abs(v)
            return v  # lower

        minmax_scores: dict[str, dict] = {m: {} for m in models}
        rank_scores:   dict[str, dict] = {m: {} for m in models}
        win_counts:    dict[str, dict] = {m: {} for m in models}

        for metric in scalar_metrics:
            goal = self._metric_goal(metric)

            valid = {
                m: float(v)
                for m in models
                for v in [self.models_data[m].get(phase, {}).get(metric)]
                if isinstance(v, (int, float))
            }
            if not valid:
                continue

            mapped = {m: _comparable(v, goal) for m, v in valid.items()}

            # Min-max: 1.0 = best, 0.0 = worst
            vmin, vmax = min(mapped.values()), max(mapped.values())
            for m, c in mapped.items():
                minmax_scores[m][metric] = 1.0 if vmax == vmin else 1.0 - (c - vmin) / (vmax - vmin)

            # Rank-based: 1 = first place (best), ties share the midrank position
            sorted_ms = sorted(mapped, key=mapped.__getitem__)
            n = len(sorted_ms)
            i = 0
            while i < n:
                j = i + 1
                while j < n and mapped[sorted_ms[j]] == mapped[sorted_ms[i]]:
                    j += 1
                avg_rank = (i + j - 1) / 2  # 0-indexed midrank; +1 gives 1-indexed position
                for k in range(i, j):
                    rank_scores[sorted_ms[k]][metric] = avg_rank + 1.0  # 1-indexed (lower = better)
                i = j

            # Win count: 1.0 for sole winner, split equally on ties
            best = min(mapped.values())
            winners = [m for m, c in mapped.items() if c == best]
            share = 1.0 / len(winners)
            for m in valid:
                win_counts[m][metric] = share if m in winners else 0.0

        recon_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synth_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]

        for m in models:
            if phase not in self.models_data[m]:
                continue

            r_mm   = [minmax_scores[m][met] for met in recon_metrics if met in minmax_scores[m]]
            s_mm   = [minmax_scores[m][met] for met in synth_metrics  if met in minmax_scores[m]]
            r_rank = [rank_scores[m][met]   for met in recon_metrics  if met in rank_scores[m]]
            s_rank = [rank_scores[m][met]   for met in synth_metrics  if met in rank_scores[m]]
            r_wins = [win_counts[m][met]    for met in recon_metrics  if met in win_counts[m]]
            s_wins = [win_counts[m][met]    for met in synth_metrics  if met in win_counts[m]]

            if r_mm:   self.models_data[m][phase]['Avg MinMax Reconstruction Score'] = float(np.mean(r_mm))
            if s_mm:   self.models_data[m][phase]['Avg MinMax Synthetic Score']      = float(np.mean(s_mm))
            if r_rank: self.models_data[m][phase]['Avg Rank Reconstruction Score']   = float(np.mean(r_rank))
            if s_rank: self.models_data[m][phase]['Avg Rank Synthetic Score']         = float(np.mean(s_rank))
            if r_wins: self.models_data[m][phase]['Win Count Reconstruction']         = float(sum(r_wins))
            if s_wins: self.models_data[m][phase]['Win Count Synthetic']              = float(sum(s_wins))

    def _generate_table(self, phase: str, sorted_models: list[str] | None = None):
        """Generate and print a comparison table of scalar metrics for a given phase."""
        models_order = sorted_models if sorted_models is not None else list(self.models_data.keys())
        headers = ["Metric"] + models_order
        
        # Collect all unique metric names
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)
                        
        # scalar_metrics = sorted(list(all_metrics))
        scalar_metrics = sorted(all_metrics, key=lambda m: (0 if "Reconstruction" in m or "Flexibility" in m else 1, m))
        
        # Move all summary metrics to the end
        _SUMMARY = {'Avg MinMax Reconstruction Score', 'Avg MinMax Synthetic Score',
                    'Avg Rank Reconstruction Score',   'Avg Rank Synthetic Score',
                    'Win Count Reconstruction',        'Win Count Synthetic'}
        summaries    = [m for m in scalar_metrics if m in _SUMMARY]
        non_summary  = [m for m in scalar_metrics if m not in _SUMMARY]
        scalar_metrics = non_summary + summaries
                
        table_data = []
        for metric in scalar_metrics:
            goal = self._metric_goal(metric)

            # Find the winning model(s) for this metric — ties share ★
            model_vals = {
                m: float(self.models_data[m][phase][metric])
                for m in models_order
                if phase in self.models_data[m] and metric in self.models_data[m][phase]
                and isinstance(self.models_data[m][phase][metric], (int, float))
            }
            best_models: set = set()
            if model_vals:
                if goal == "target_1":
                    best_raw = min(abs(v - 1.0) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v - 1.0) - best_raw) <= 1e-9}
                elif goal == "target_0":
                    best_raw = min(abs(v) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v) - best_raw) <= 1e-9}
                elif goal == "higher":
                    best_raw = max(model_vals.values())
                    tol = abs(best_raw) * 1e-9 + 1e-12
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= tol}
                else:  # lower
                    best_raw = min(model_vals.values())
                    tol = abs(best_raw) * 1e-9 + 1e-12
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= tol}

            row = [self._display_metric_name(metric)]
            for model_name in models_order:
                data = self.models_data[model_name]
                if phase in data and metric in data[phase]:
                    val = data[phase][metric]
                    formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                    if model_name in best_models:
                        formatted = f"★ {formatted}"
                else:
                    formatted = "N/A"
                row.append(formatted)
            table_data.append(row)
            
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nModel Comparison Table ({phase}):")
        print(table_str)
        
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        table_path = os.path.join(phase_output_dir, "comparison_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"\nTable saved to {table_path}")

    def _generate_plots(self, phase: str, sorted_models: list[str] | None = None):
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

        # Append summary metrics at the end of each group in a fixed order
        _RECON_SUMMARY = [
            "Avg MinMax Reconstruction Score",
            "Avg Rank Reconstruction Score",
            "Win Count Reconstruction",
        ]
        _SYNTH_SUMMARY = [
            "Avg MinMax Synthetic Score",
            "Avg Rank Synthetic Score",
            "Win Count Synthetic",
        ]
        reconstruction_metrics = sorted(
            [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        ) + [m for m in _RECON_SUMMARY if m in scalar_metrics]

        synthetic_metrics = sorted(
            [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]
        ) + [m for m in _SYNTH_SUMMARY if m in scalar_metrics]

        models = sorted_models if sorted_models is not None else list(self.models_data.keys())
        
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

    phases = ["aggregated_phase", "ranked_phase_1", "ranked_phase_2", "ranked_phase_3"]
    for phase in phases:
        p = os.path.join(model_output_dir, f"{phase}_eval", "eval_results.json")
        if os.path.exists(p):
            return True
    return os.path.exists(os.path.join(model_output_dir, "eval_results.json"))

class _DenormalizingLoader:
    """Wraps a normalized DataLoader and inverse-transforms x on each batch.

    Used so Faraday-family models (trained on raw kWh, normalizer=None) receive
    the same test windows as every other model while seeing un-normalized values.
    """

    def __init__(self, loader, normalizer):
        self.loader = loader
        self.normalizer = normalizer

    def __iter__(self):
        for x, meter_id, timestamps, agg in self.loader:
            yield self.normalizer.inverse_transform(x), meter_id, timestamps, agg

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def _run_holdout_eval(
    model,
    normalizer,
    conditioner,
    data_files: list[str],
    holdout_feeder_id: int,
    topology_path: str,
    split_seed: int,
    split_ratios: dict,
    max_value: float | None,
    num_samples: int,
    device: str,
    window_size: int,
    model_output_dir: str,
    n_synthetic_draws: int = 1,
) -> None:
    """Run EvalFramework on the holdout feeder and save results + profile plot.

    n_synthetic_draws: how many independent synthetic draws to average for the
    profile plot. Metrics are computed on the first draw only; the averaged
    synthetic is used exclusively for the feeder-level profile plot.
    """
    import copy
    loader = InMemoryCSVDataloaders(
        sm_files=data_files,
        window_len=window_size,
        stride=window_size,
        batch_size=1024,
        split_ratios=split_ratios,
        split_seed=split_seed,
        max_value=max_value,
        normalizer=normalizer,
        holdout_feeder_id=holdout_feeder_id,
        topology_path=topology_path,
        pin_memory=False,
    )
    holdout_loader = loader.get_holdout_loader()
    if holdout_loader is None:
        print(f"  [holdout] No windows found for feeder {holdout_feeder_id}; skipping.")
        return

    # Use the full holdout dataset as the collection target so every calendar date
    # and every meter is represented in the feeder-level aggregated profile plot.
    n_holdout_windows = len(holdout_loader.dataset)
    print(f"  [holdout] {n_holdout_windows} real windows from feeder {holdout_feeder_id}.")

    eval_norm = copy.deepcopy(normalizer)
    evaluator = EvalFramework(
        model=model,
        dataloader=holdout_loader,
        normalizer=eval_norm,
        conditioner=conditioner,
        device=device,
        window_size=window_size,
        output_path=model_output_dir,
        n_synthetic_samples=n_holdout_windows,
        preserve_condition_order=True,
    )
    evaluator.register_all(phased=False)
    try:
        evaluator.evaluate_averaged(n_draws=n_synthetic_draws)
    except RuntimeError as e:
        if device != "cpu" and "out of memory" in str(e).lower():
            print("  [holdout] GPU OOM — retrying on CPU.")
            cpu_norm = copy.deepcopy(normalizer)
            evaluator = EvalFramework(
                model=model, dataloader=holdout_loader, normalizer=cpu_norm,
                conditioner=conditioner, device="cpu", window_size=window_size,
                output_path=model_output_dir, n_synthetic_samples=n_holdout_windows,
                preserve_condition_order=True,
            )
            evaluator.register_all(phased=False)
            evaluator.evaluate_averaged(n_draws=n_synthetic_draws)
        else:
            raise

    holdout_json = os.path.join(model_output_dir, "eval_results_feeder_holdout.json")
    evaluator.save_results(holdout_json)
    print(f"  [holdout] Metrics saved to {holdout_json}")

    _rw = getattr(evaluator, '_real_data', None)
    _sw = getattr(evaluator, '_synthetic_data', None)
    _rt = getattr(evaluator, '_timestamps', None)
    _rm = getattr(evaluator, '_meter_ids', None)
    _missing_plot = [n for n, v in [("_real_data", _rw), ("_synthetic_data", _sw), ("_timestamps", _rt), ("_meter_ids", _rm)] if v is None]
    if _missing_plot:
        print(f"  [holdout] Profile plots skipped — EvalFramework did not populate: {_missing_plot}")
    if all(x is not None for x in [_rw, _sw, _rt, _rm]):
        _rw, _sw = np.asarray(_rw), np.asarray(_sw)
        _rt, _rm = np.asarray(_rt), np.asarray(_rm)
        _raw_st = getattr(evaluator, '_synthetic_timestamps', None)
        _raw_sm = getattr(evaluator, '_synthetic_meter_ids', None)
        if _raw_st is not None:
            _st = np.asarray(_raw_st)
            _sm = np.asarray(_raw_sm) if _raw_sm is not None else np.tile(_rm, (_sw.shape[0] // max(_rm.shape[0], 1)) + 1)[:_sw.shape[0]]
        else:
            _n, _reps = _sw.shape[0], (_sw.shape[0] // max(_rt.shape[0], 1)) + 1
            _st = np.tile(_rt, _reps)[:_n]
            _sm = np.tile(_rm, _reps)[:_n]

        # For the profile plot, pool daily aggregates across K independent draws
        # (not pixel-wise averaged), so bands reflect inter-draw AND inter-day variance.
        _plot_extra_draws = None
        if n_synthetic_draws > 1:
            with torch.no_grad():
                _fresh = [
                    np.asarray(evaluator._generate_synthetic(n_holdout_windows))
                    for _ in range(n_synthetic_draws)
                ]
            _sw = _fresh[0]
            _plot_extra_draws = _fresh[1:]

        _holdout_analysis_dir = os.path.join(model_output_dir, "holdout_analysis")
        plot_feeder_holdout_analysis(
            real_windows=_rw, real_timestamps=_rt, real_meter_ids=_rm,
            synth_windows=_sw, synth_timestamps=_st, synth_meter_ids=_sm,
            output_dir=_holdout_analysis_dir,
            conditioner=conditioner,
            synth_extra_draws=_plot_extra_draws,
        )
        _cf_scalars = run_coincidence_factor_analysis(
            real_windows=_rw, real_meter_ids=_rm, real_timestamps=_rt,
            synth_windows=_sw, synth_meter_ids=_sm, synth_timestamps=_st,
            output_dir=_holdout_analysis_dir,
        )
        if _cf_scalars:
            try:
                with open(holdout_json) as _f:
                    _holdout_results = json.load(_f)
                _holdout_results.update(_cf_scalars)
                with open(holdout_json, "w") as _f:
                    json.dump(_holdout_results, _f, indent=2)
            except Exception as _e:
                print(f"  [holdout] Warning: could not merge CF scalars into holdout JSON: {_e}")
                import traceback as _tb; _tb.print_exc()


def compare_models(model_paths: list[str],
                   model_names: list[str],
                   secondary_model_paths: list[str] | None = None,
                   eval_result_dirs: list[str] | None = None,
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
                   holdout_n_draws: int = 1,
                   ):
    """
    Run evaluation for multiple models and compare results.
    Can handle both phased and unphased models.

    Args:
        model_paths: Paths to trained model folders (used for loading checkpoints).
        model_names: Display names for comparison.
        secondary_model_paths: Optional secondary model paths aligned with model_paths.
        eval_result_dirs: Optional paths to existing evaluation result folders aligned with
            model_paths. If provided and recompute=False, these folders are used as the
            source for comparison (for example rerun_evaluation_YYYYMMDD_HHMMSS folders).
    """

    # Names-only mode: when no model_paths are given, resolve each name to output_dir/<name>.
    if not model_paths:
        model_paths = [os.path.join(output_dir, name) for name in model_names]
        if not recompute:
            missing = [name for name, p in zip(model_names, model_paths) if not os.path.isdir(p)]
            if missing:
                raise FileNotFoundError(
                    f"Pre-computed results not found for: {missing}. "
                    f"Expected directories under '{output_dir}'."
                )

    if eval_result_dirs is not None and len(eval_result_dirs) < len(model_paths):
        eval_result_dirs = list(eval_result_dirs) + [None] * (len(model_paths) - len(eval_result_dirs))

    # 1. Prepare evaluation data — skip file discovery when not recomputing
    if data_files is None and recompute:
        data_files = get_eval_files(data_dir)
    elif data_files is None:
        data_files = []

    # 2. Pick a single global test split definition so all models see identical windows.
    if split_seed is None or split_ratios is None:
        inferred_seed, inferred_ratios = _infer_split_from_first_model(model_paths)
        split_seed = inferred_seed if split_seed is None else int(split_seed)
        split_ratios = inferred_ratios if split_ratios is None else split_ratios
    split_seed = int(split_seed)
    split_ratios = dict(split_ratios)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = WindowSpec(input_len=24, stride=24)

    # 3. Pre-scan: decide which models need recomputation and resolve output dirs.
    #    This lets us build the shared test loader exactly once, before the eval loop.
    secondaries = (
        secondary_model_paths
        if secondary_model_paths is not None
        else [None] * len(model_paths)
    )
    eval_plan: list[dict] = []
    for i, (path, name, secondary) in enumerate(zip(model_paths, model_names, secondaries)):
        model_output_dir = os.path.join(output_dir, name)
        requested_eval_dir = eval_result_dirs[i] if eval_result_dirs is not None else None
        eval_source_dir = requested_eval_dir if requested_eval_dir else path
        skip = (
            (not recompute)
            and _has_existing_eval_outputs(eval_source_dir, phased=phased)
            and secondary is None
        )
        eval_plan.append({
            "path": path,
            "name": name,
            "secondary": secondary,
            "model_output_dir": model_output_dir,
            "eval_source_dir": eval_source_dir,
            "skip": skip,
        })

    # 4. Build ONE shared test loader for all models that need recomputation.
    #    Uses the first recomputing model's normalizer (same "infer from first model"
    #    convention already used for split_seed above).  All models are expected to
    #    have been trained on the same dataset with a compatible normalizer.
    shared_test_loader = None
    loader_obj = None
    # Cache (model, normalizer, conditioner) for the first recomputing model so we
    # don't load its weights twice (once here for the normalizer, once in the loop).
    _first_model_cache: dict[int, tuple] = {}

    needs_recompute_indices = [i for i, p in enumerate(eval_plan) if not p["skip"]]
    if needs_recompute_indices:
        first_idx = needs_recompute_indices[0]
        _first_model, _first_norm, _first_cond = load_model(
            eval_plan[first_idx]["path"], device=device
        )
        _first_model_cache[first_idx] = (_first_model, _first_norm, _first_cond)

        # replay_buffer_capacity sized to num_samples: the circular buffer can hold
        # all requested unique test windows without evicting earlier ones.
        # This loader is evaluation-only, so a large buffer is safe.
        loader_obj = PhasedReplayDataloaders(
            sm_files=data_files,
            window_spec=window,
            batch_size=1024,
            replay_buffer_capacity=num_samples,
            min_buffer_size=0,
            normalizer=_first_norm,
            num_workers=0,
            phased_mode=phased,
            sample_random=False,          # deterministic test selection + ordering
            feature_cols=["hourly_energy_consumption"],
            meter_col="meter_number",
            aggregation_divisor=agg_divisor,
            hourly_aggregation=agg_divisor > 1.0,
            split_seed=split_seed,
            split_ratios=split_ratios,
            max_value=max_value,
            pin_memory=False,
        )

        n_batches = max(1, num_samples // 32)
        loader_obj.refresh_test_cache(n_batches)
        shared_test_loader = loader_obj.get_test_loader()

        if shared_test_loader is None and hasattr(loader_obj, "test_dataset"):
            print("[WARN] get_test_loader() returned None; constructing DataLoader from test_dataset directly.")
            from torch.utils.data import DataLoader as _DL
            shared_test_loader = _DL(
                loader_obj.test_dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

        if shared_test_loader is None:
            print("[ERROR] Could not build a test loader. No evaluation will be run.")
        else:
            n_test = len(shared_test_loader.dataset)  # type: ignore[arg-type]
            print(f"\nShared test loader ready: {n_test} windows "
                  f"({n_test * 32 // 32} samples, batch_size=1024).")

    # 5. Loop through models and evaluate, reusing the shared test loader.
    used_eval_dirs: list[str] = []

    for i, plan in enumerate(eval_plan):
        path             = plan["path"]
        name             = plan["name"]
        secondary        = plan["secondary"]
        model_output_dir = plan["model_output_dir"]
        eval_source_dir  = plan["eval_source_dir"]

        print(f"\nEvaluating model: {name}")
        os.makedirs(model_output_dir, exist_ok=True)

        if plan["skip"]:
            print(f"  Found existing evaluation outputs in {eval_source_dir}; skipping recompute.")
            used_eval_dirs.append(eval_source_dir)
            continue

        if shared_test_loader is None:
            print(f"  [ERROR] No test data available; skipping '{name}'.")
            used_eval_dirs.append(model_output_dir)
            continue

        # Reuse cached weights for the first model; load fresh for all others.
        if i in _first_model_cache:
            model, normalizer, conditioner = _first_model_cache[i]
        else:
            model, normalizer, conditioner = load_model(path, device=device)

        sec_model = None
        if secondary is not None:
            # Only the model object is needed from the secondary load; the secondary
            # model carries its own conditioner and normalizer internally and will
            # build its own conditions from real phased data / metadata at inference
            # time.  Overwriting normalizer/conditioner here was causing the primary
            # model's encoder to receive wrong-dimensioned condition vectors.
            sec_model, *_ = load_model(secondary, device=device)

        # Faraday-family models have no normalizer and were trained on raw kWh.
        # The shared loader yields normalized data, so wrap it to denormalize on the fly.
        _loader_norm = getattr(loader_obj, 'normalizer', None)
        dataloader = (
            _DenormalizingLoader(shared_test_loader, _loader_norm)
            if normalizer is None and _loader_norm is not None
            else shared_test_loader
        )

        if phased:
            # Phased models use PhasedEvaluationSuite which creates sub-phase directories
            evaluator = PhasedEvaluationSuite(
                model=model,
                dataloader=dataloader,  # type: ignore[arg-type]
                normalizer=normalizer,
                conditioner=conditioner,
                device=device,
                window_size=window.input_len,
                output_path=model_output_dir,
                secondary_model=sec_model,
                n_synthetic_samples=num_samples,
            )
            evaluator.evaluate()
        else:
            # Unphased models use EvalFramework directly
            evaluator = EvalFramework(
                model=model,
                dataloader=dataloader,  # type: ignore[arg-type]
                normalizer=normalizer,
                conditioner=conditioner,
                device=device,
                window_size=window.input_len,
                output_path=os.path.join(model_output_dir, "eval_results.json"),
                secondary_model=sec_model,
                n_synthetic_samples=num_samples,
            )
            evaluator.register_all(phased=False)
            evaluator.evaluate()
            evaluator.save_results(os.path.join(model_output_dir, "eval_results.json"))

        # Run holdout eval if the model's training config has holdout_feeder_id set.
        # When recompute=True, always re-run (overwrites JSON + plot).
        # When recompute=False, copy from training dir if available; only run if still missing.
        _holdout_dst = os.path.join(model_output_dir, "eval_results_feeder_holdout.json")
        _holdout_src = os.path.join(path, "eval_results_feeder_holdout.json")
        if not recompute and os.path.exists(_holdout_src) and not os.path.exists(_holdout_dst):
            shutil.copy2(_holdout_src, _holdout_dst)
        if (recompute or not os.path.exists(_holdout_dst)) and data_files:
            _cfg_path = _find_model_config_path(path)
            if _cfg_path is None:
                print(f"  [holdout] No training config found in {path}; cannot run holdout eval.")
            else:
                import traceback as _tb
                try:
                    _mcfg = _load_train_config(_cfg_path)
                    _hid = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                    _topo = getattr(getattr(_mcfg, "data", None), "topology_path", None)
                    if _hid is None:
                        print(f"  [holdout] holdout_feeder_id not set in {_cfg_path}; skipping holdout eval.")
                    elif _topo is None:
                        print(f"  [holdout] holdout_feeder_id={_hid} is set but topology_path is missing in {_cfg_path}; skipping holdout eval.")
                    else:
                        print(f"  Running holdout eval (feeder {_hid})...")
                        _run_holdout_eval(
                            model=model,
                            normalizer=normalizer,
                            conditioner=conditioner,
                            data_files=data_files,
                            holdout_feeder_id=int(_hid),
                            topology_path=str(_topo),
                            split_seed=split_seed,
                            split_ratios=split_ratios,
                            max_value=max_value,
                            num_samples=num_samples,
                            device=device,
                            window_size=window.input_len,
                            model_output_dir=model_output_dir,
                            n_synthetic_draws=holdout_n_draws,
                        )
                except Exception as _e:
                    print(f"  [holdout] FAILED with exception: {_e}")
                    _tb.print_exc()

        used_eval_dirs.append(model_output_dir)

    # 6. Use PhasedModelComparator to generate comparison tables and plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON")
    print("="*50)
    
    comparator = PhasedModelComparator(output_dir=output_dir)
    if len(used_eval_dirs) != len(model_names):
        raise ValueError(
            f"Internal mismatch: have {len(used_eval_dirs)} eval dirs for {len(model_names)} model names."
        )
    for name, eval_dir in zip(model_names, used_eval_dirs):
        comparator.add_model(name, eval_dir)
    
    comparator.compare()
    
    print("\nComparison complete. Results saved to:", output_dir)
    return comparator.models_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model evaluation results (reuse outputs or recompute).")
    parser.add_argument("--models", nargs="+", required=False, help="List of model directories (each contains config + checkpoint).")
    parser.add_argument("--names", nargs="+", required=False, help="Display names for models (same length as --models).")
    parser.add_argument("--secondary-models", nargs="*", default=None, help="Optional secondary model dirs aligned with --models. Use 'None' to skip.")
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=None,
        help="Optional existing evaluation result dirs aligned with --models (for example rerun_evaluation_* folders).",
    )
    parser.add_argument("--data-dir", type=str, default="data/_parquet_cache", help="Directory with parquet files.")
    parser.add_argument("--data-files", nargs="*", default=None, help="Optional explicit parquet files (relative to --data-dir unless absolute).")
    parser.add_argument("--num-samples", type=int, default=10000, help="Approx number of windows to evaluate per model.")
    parser.add_argument("--output-dir", type=str, default="evaluation_comparison", help="Where to write/read evaluation outputs.")
    parser.add_argument("--phased", action="store_true", help="Use phased evaluation suite.")
    parser.add_argument("--recompute", action="store_true", help="Recompute evaluation even if outputs already exist.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed used for deterministic window-level train/val/test split. Defaults to first model's training config.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train ratio for split (defaults to first model's training config).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Val ratio for split (defaults to first model's training config).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio for split (defaults to first model's training config).")
    parser.add_argument("--max-value", type=float, default=200.0, help="Maximum value for filtering windows.")
    parser.add_argument("--holdout-n-draws", type=int, default=1, help="Number of synthetic draws to average for holdout feeder profile plot.")
    args = parser.parse_args()
    holdout_n_draws = int(args.holdout_n_draws)

    args.models = ["E:/models/new3/FaradayModel_20260529_021321", "E:/models/new3/GuidedFaradayModel_20260529_000300"]
    args.names = ["Faraday-NEW", "Guided-Faraday-NEW"]

    # args.models = ["E:/models/new2/AverageBaselineModel_20260526_234356",
    #     "E:/models/new1/CVAE_20260522_041326_guide_cnn", "E:/models/new1/CVAE_20260522_051533_h_guide_cnn",
    #                "E:/models/new2/CVAE_20260526_154037", "E:/models/new2/FaradayModel_20260526_102007", #"E:/models/new2/GuidedFaradayModel_20260526_110124",
    #                "E:/models/new2/DiffusionModel_20260526_070858", "E:/models/new1/DiffusionModel_20260522_070459", 
    #                #"E:/models/new2/GuidedFaradayModel_20260526_114407", #"E:/models/new2/GuidedFaradayModel_20260526_122401",
    #                 "E:/models/new2/QuantileBaselineModel_20260527_000045", 
    #                "E:/models/new2/CVAE_20260525_233836","E:/models/new2/CVAE_20260526_011125",
    #                 "E:/models/new2/CVAE_20260526_091116",
    #                "E:/models/new2/GuidedFaradayModel_20260527_152449", "E:/models/new2/GuidedFaradayModel_20260527_162635"
    #                #"E:/models/new2/GuidedFaradayModel_20260527_141620",
    #                ]

    # args.names = ["Baseline-Average",
    #     "Guide-CNN", "H-Guide-CNN",
    #               "Guide-Vamp", "Faraday", #"Guided-Faraday", 
    #               "Diffusion", "Diffusion Old",
    #               #"Guided-Faraday-CNN", #"Guided-Faraday-CNN-Deterministic",
    #                "Baseline Quantile",
    #               "H-Guide", "Guide", 
    #                "CVAE", 
    #               "Guided-Faraday-CNN-FIX", "Guided-Faraday"
    #               #"Guided-Faraday-cnn-new", 
    #               ]
    
    # args.models = ["E:/models/new2/FaradayModel_20260526_102007","E:/models/new2/GuidedFaradayModel_20260527_152449", "E:/models/new2/GuidedFaradayModel_20260527_162635"
    #                ]
    # args.names = ["Faraday", "Guided-Faraday-CNN-FIX", "Guided-Faraday"]

    if not args.names:
        raise ValueError("--names is required.")
    if args.models and len(args.models) != len(args.names):
        raise ValueError("--models and --names must have the same length.")
    if not args.models and args.recompute:
        raise ValueError("--models is required when --recompute is set.")
    if not args.models:
        args.models = []
    
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

    eval_result_dirs = None
    if args.eval_dirs is not None and len(args.eval_dirs) > 0:
        eval_result_dirs = []
        for d in args.eval_dirs:
            if d is None or str(d).strip() == "" or str(d).lower() == "none":
                eval_result_dirs.append(None)
            else:
                eval_result_dirs.append(d)
        if len(eval_result_dirs) < len(args.models):
            eval_result_dirs.extend([None] * (len(args.models) - len(eval_result_dirs)))

    split_ratios = None
    if args.train_ratio is not None or args.val_ratio is not None or args.test_ratio is not None:
        split_ratios = {
            "train_ratio": 0.8 if args.train_ratio is None else float(args.train_ratio),
            "val_ratio": 0.1 if args.val_ratio is None else float(args.val_ratio),
            "test_ratio": 0.1 if args.test_ratio is None else float(args.test_ratio),
        }

    # args.models = [#"E:/models/new1/CVAE_20260521_234622_guide",
    #                #"E:/models/new1/CVAE_20260522_013003_cvae",  
    #                "E:/models/new1/CVAE_20260522_041326_guide_cnn",
    #                "E:/models/new1/CVAE_20260522_051533_h_guide_cnn", 
    #                "E:/models/new1/QuantileBaselineModel_20260522_034728", 
    #             #    "E:/models/old/Guide Vamp", "E:/models/old/Guide Hierarchy", "E:/models/old/Guide CNN Hierarchy",
    #             #     "E:/models/old/AverageBaselineModel",
    #             #    "E:/models/old/AutoRegressiveTransformerVAE",
    #                "E:/models/new1/DiffusionModel_20260522_070459",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_113725",
    #                "E:/models/new1/FaradayModel_20260522_083016",
    #                "C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_134604",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_143604",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_154532",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_165636",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_175145",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_205135",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_213329",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_222016",
    #                "E:/models/new2/CVAE_20260525_233836","E:/models/new2/CVAE_20260526_011125","E:/models/new2/CVAE_20260526_024444",
    #                "E:/models/new2/CVAE_20260526_053436","E:/models/new2/CVAE_20260526_091116","E:/models/new2/DiffusionModel_20260526_070858",
    #                "E:/models/new2/CVAE_20260526_154037", "E:/models/new2/FaradayModel_20260526_102007", "E:/models/new2/GuidedFaradayModel_20260526_110124",
    #                "E:/models/new2/GuidedFaradayModel_20260526_114407", "E:/models/new2/GuidedFaradayModel_20260526_122401",
    #                "E:/models/new2/GuidedFaradayModel_20260527_080406_maybe_tanh", "E:/models/new2/GuidedFaradayModel_20260527_090218",
    #                "E:/models/new2/GuidedFaradayModel_20260527_100405"
    #                ]
    # args.names = [#"Guide", 
    #               #"CVAE", 
    #               "Guide-CNN",
    #                 "H-Guide-CNN", 
    #                 "Quantiles Baseline",
    #             #   "Guide Vamp", "Guide Hierarchy", "Guide CNN Hierarchy",  "Average Baseline",
    #             #     "AutoRegressiveTransformerVAE", 
    #                 "Diffusion",    
    #                 #"Guided-Faraday 2", 
    #                 "Faraday",
    #                 "Guided-Faraday 3", 
    #                 # "Guide CNN", 
    #                 # "Guide CNN 2",
    #                 #"Guided-Faraday 4",
    #                 #"Guide CNN 3",
    #                 #"Guided-Faraday 5",
    #                 #"Guided-Faraday 6",
    #                 #"Guided-Faraday 7",
    #                 "H-Guide-n", "Guide-n", "Guide-CNN-n", 
    #               "H-Guide-CNN-n", "CVAE-n", "Diffusion-n", 
    #               "Guide-Vamp-n", "Faraday-n", "Guided-Faraday-n", 
    #               "Guided-Faraday-CNN-n", "Guided-Faraday-CNN-2-n",
    #               "Guided-Faraday-CNN-3-n", "Guided-Faraday-CNN-TanH-n",
    #               "Guided-Faraday-CNN-4-n-100PDCC"
    #               ]
    # Guided-Faraday 2: used the tanh. 
    # Guided-Faraday 3: used softplus with beta=10. Non-stochastic loss  (2,363,544 params)
    # Guide-Faraday 4: will be stochastic in the loss (see loss function)
    # Guide-Faraday 5: using mu activaton = none, still stochastic in the loss
    #                   Using larger pdcc average max std (4) (2,363,544 params)
    # Guide-Faraday 6: using mu activation = None, still stochastic in the loss, User-Profiles included (2,363,544 params)
    # Guide-Faraday 7: using mu activation = None, non-stochastic, User-profiles (2,363,544 params)
    # Guide-Faraday 8: Will not early stop before 15 patience. Using mu activation = None, stochastic (2,363,544 params)

    # E:/models/new2/GuidedFaradayModel_20260527_141620, cnn concat on input (2,838,640 params)

    # Guide-CNN: used the tanh activation in forward (see git)
    # Guide CNN: used softplus but with beta=1. 
    # Guide CNN 2: used softplus with beta=15. (1,153,153 params)
    # Guide CNN 3: is scaled up to the other models size and used softplus with beta=100 in sampling and None in forward. (2,114,433 params
    # Guide CNN 4: using tanh in forward like back in the day but with actual parameter count (2,114,433 params)

    

    compare_models(
        model_paths=args.models,
        model_names=args.names,
        secondary_model_paths=secondary_model_paths,
        eval_result_dirs=eval_result_dirs,
        data_dir=args.data_dir,
        data_files=args.data_files,
        num_samples=int(args.num_samples),
        output_dir=args.output_dir,
        phased=bool(args.phased),
        recompute=bool(args.recompute),
        split_seed=args.split_seed,
        split_ratios=split_ratios,
        max_value=args.max_value,
        agg_divisor=1000.0 if args.phased else 1.0,
        holdout_n_draws=holdout_n_draws,
        
    )

# Example run in CLI: 
# python src/evaluation/compare_models.py \
#     --models models/hpc_runs/AutoRegressiveTransformerVAE_20260424_170101 \
#     --names art \
#     --eval-dirs models/hpc_runs/AutoRegressiveTransformerVAE_20260424_170101/rerun_evaluation_20260511_101040 \
#     --phased