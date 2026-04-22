

import os
import sys
import torch
import random
import json
import numpy as np
from pathlib import Path

# Add project roots for imports
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.evaluation.load_model import load_model
from src.train import _load_train_config
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

    def __init__(self, output_dir: str = "evaluation_comparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_data = {}
        # We define a superset of possible phases.
        self.phases = ['full_eval', 'aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']

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
                    if isinstance(val, (int, float)):
                        all_metrics.add(metric)
                        
        scalar_metrics = list(all_metrics)
        models = list(self.models_data.keys())
        
        # --- Normalization for Average Plotting ---
        scores_dict = {m: {} for m in models}
        for metric in scalar_metrics:
            is_max = any(term in metric for term in ['LogLikelihood', 'Accuracy', 'Score', 'R2'])
            is_one = ('TSTR' in metric)
            is_zero = ('MMD' in metric or 'MaximumMeanDiscrepancy' in metric)
            
            raw_vals = []
            for m in models:
                v = self.models_data[m].get(phase, {}).get(metric, None)
                raw_vals.append((m, v))
                
            mapped_vals = {}
            for m, v in raw_vals:
                if isinstance(v, (int, float)):
                    if is_one:
                        mapped_vals[m] = abs(v - 1.0)
                    elif is_zero:
                        mapped_vals[m] = abs(v)
                    elif is_max:
                        mapped_vals[m] = -v # invert max so lower is always better
                    else:
                        mapped_vals[m] = float(v)
                        
            if mapped_vals:
                vmin = min(mapped_vals.values())
                vmax = max(mapped_vals.values())
                for m, val in mapped_vals.items():
                    if vmax == vmin:
                        scores_dict[m][metric] = 1.0
                    else:
                        # 1.0 is the best score, 0.0 is the worst score
                        scores_dict[m][metric] = 1.0 - (val - vmin) / (vmax - vmin)

        recon_metrics = [m for m in scalar_metrics if 'Reconstruction' in m or 'FlexibilityDistance' in m]
        synth_metrics = [m for m in scalar_metrics if m not in recon_metrics]
        
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
                    if isinstance(val, (int, float)):
                        all_metrics.add(metric)
                        
        scalar_metrics = sorted(list(all_metrics))
        
        # Move averages to the end
        averages = [m for m in scalar_metrics if m.startswith('Avg MinMax')]
        non_averages = [m for m in scalar_metrics if not m.startswith('Avg MinMax')]
        scalar_metrics = non_averages + averages
                
        table_data = []
        for metric in scalar_metrics:
            row = [metric]
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
        """Generate bar charts for scalar metrics for a given phase."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)):
                        all_metrics.add(metric)
                
        scalar_metrics = list(all_metrics)
        if not scalar_metrics:
            return
            
        scalar_metrics = sorted(scalar_metrics)
        
        # Move averages to the end
        averages = [m for m in scalar_metrics if m.startswith('Avg MinMax')]
        non_averages = [m for m in scalar_metrics if not m.startswith('Avg MinMax')]
        scalar_metrics = non_averages + averages
        
        models = list(self.models_data.keys())
        
        n_metrics = len(scalar_metrics)
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle the case of a single subplot or 1D array of subplots
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        width = 0.8 / len(models)
        
        # Use centralized plot styling
        plot_style = PlotStyle()
        
        for idx, metric in enumerate(scalar_metrics):
            ax = axes[idx]
            
            # Determine if we want to min/max this metric
            is_maximize = any(term in metric for term in ['LogLikelihood', 'Accuracy', 'Score', 'R2'])
            is_closest_to_one = ('TSTR' in metric)
            is_closest_to_zero = ('MMD' in metric or 'MaximumMeanDiscrepancy' in metric)
            
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
                if is_closest_to_one:
                    best_val = min(valid_vals, key=lambda v: abs(v - 1.0))
                elif is_closest_to_zero:
                    best_val = min(valid_vals, key=lambda v: abs(v - 0.0))
                elif is_maximize:
                    best_val = max(valid_vals)
                else:
                    # Default: lower is better
                    best_val = min(valid_vals)
                    
                best_idx = valid_models[valid_vals.index(best_val)]
            
            # One x-position for the metric
            x = np.arange(1) 
            
            for i in range(len(models)):
                if not np.isnan(m_values[i]):
                    val = m_values[i]
                    color = plot_style.get_color(i)
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
            
            ax.set_title(metric, fontsize=11, fontweight='bold')
            ax.set_xticks([]) # Remove x-axis tick completely for cleaner look
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        # Create a custom legend
        legend_handles = [mpatches.Patch(color=plot_style.get_color(i), label=model) for i, model in enumerate(models)]
        legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='★ Best Value'))
        
        # Position the legend above the plots
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(models)+1))
        
        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        plot_path = os.path.join(phase_output_dir, "metric_comparison.png")
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
        
    # randomly shuffle available files:
    random.shuffle(discovered)
    return discovered

def compare_models(model_paths: list[str], 
                   model_names: list[str], 
                   secondary_model_paths: list[str] | None = None,
                   data_dir: str = "data/_parquet_cache",
                   data_files: list[str] | None = None,
                   num_samples: int = 100,
                   seed: int = 42,
                   output_dir: str = "evaluation_comparison",
                   phased: bool = False,
                   agg_divisor = 1000.0,
                   is_agg = True,
                   ):
    """
    Run evaluation for multiple models and compare results.
    Can handle both phased and unphased models.
    """

    # 1. Prepare evaluation data
    if data_files is None:
        data_files = get_eval_files(data_dir)
    
    # 4. Loop through models and evaluate
    for i, (path, name) in enumerate(zip(model_paths, model_names)):
        print(f"\nEvaluating model: {name}")
        model, normalizer, conditioner = load_model(path, device="cuda")
        
        # reloading with the same seed each model
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

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
            sample_random=True,
            feature_cols=["hourly_energy_consumption",],
            meter_col = "meter_number",
            aggregation_divisor=agg_divisor,
            hourly_aggregation=True if agg_divisor > 1.0 else False
        )
        
        # 3. Grab a fixed number of samples for evaluation
        n_batches = max(1, num_samples // 32)
        loader_obj.refresh_eval_cache(n_batches)
        test_loader = loader_obj.get_eval_loader()
        
        sec_model = None
        if secondary_model_paths is not None and i < len(secondary_model_paths):
            if secondary_model_paths[i] is not None:
                sec_model, _, _ = load_model(secondary_model_paths[i], device="cuda")
        
        model_output_dir = os.path.join(output_dir, name)
        os.makedirs(model_output_dir, exist_ok=True)

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
            evaluator.register_all(phased=False)
            evaluator.evaluate()
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
    model_paths = [
        "models/SetGuideVAE_20260422_143202",
        "models/DiffusionModel_20260422_162452"
    ]
    model_names = [
        "Set-guide",
        "Diffusion"
    ]
    secondary_model_paths = [
        None,
        None,
    ]
    data_dir = "F:/Data/processed_data/second_batch/individual_sm_datasets/raw_profiles"
    compare_models(model_paths, 
                   model_names, 
                   secondary_model_paths, 
                   data_dir=data_dir, 
                   phased=True)