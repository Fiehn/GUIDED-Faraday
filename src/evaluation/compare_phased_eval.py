import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.visualization.plot_styling import PlotStyle

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers="keys", **kwargs):
        return str(data)

class PhasedModelComparator:
    """Class to compare evaluation results across multiple phased models."""

    def __init__(self, output_dir: str = "evaluation_comparison_phased"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_data = {}
        self.phases = ['aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']

    def add_model(self, model_name: str, model_eval_dir: str):
        """Add a model's phased evaluation results to the comparator."""
        self.models_data[model_name] = {}
        for phase in self.phases:
            eval_json_path = os.path.join(model_eval_dir, f"{phase}_eval", "eval_results.json")
            if not os.path.exists(eval_json_path):
                 print(f"File not found: {eval_json_path}")
                 continue
            
            with open(eval_json_path, 'r') as f:
                data = json.load(f)
            
            self.models_data[model_name][phase] = data

    def compare(self):
        """Generate comparison tables and plots for each phase."""
        if not self.models_data:
            print("No models added. Exiting.")
            return

        for phase in self.phases:
            # Check if we have data for this phase
            has_data = any(phase in self.models_data[m] for m in self.models_data)
            if has_data:
                self._compute_averages(phase)
                self._generate_table(phase)
                self._generate_plots(phase)

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
            
        # Create a custom legend so we can include the "Best Value" marker style
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluations between different phased models.")
    parser.add_argument("--eval_dirs", nargs='+', required=True, help="List of paths to directories containing phased eval results.")
    parser.add_argument("--names", nargs='+', required=True, help="List of model names corresponding to the eval dirs.")
    parser.add_argument("--output_dir", type=str, default="evaluation_comparison_phased", help="Directory to save the comparison plots and table.")
    
    args = parser.parse_args()
    
    args.eval_dirs = ["models/SetGuideVAE_DictionaryCoder_20260320_141714_clamping", "models/SetGuideVAE_DictionaryCoder_20260320_165707_phase_tags", "models/SetGuideVAE_DictionaryCoder_20260321_103830_lossfunction_change"]
    args.names = ["Clamping", "Phase Tags", "Loss Function Change"]

    if len(args.eval_dirs) != len(args.names):
        print("Error: The number of evaluation directories must match the number of model names.")
        exit(1)
        
    comparator = PhasedModelComparator(args.output_dir)
    
    for name, path in zip(args.names, args.eval_dirs):
        comparator.add_model(name, path)
        
    comparator.compare()
