import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.visualization.plot_styling import PlotStyle

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers="keys", **kwargs):
        return str(data)

class ModelComparator:
    """Class to compare evaluation results across multiple models."""

    def __init__(self, output_dir: str = "evaluation_comparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_data = {}

    def add_model(self, model_name: str, eval_json_path: str):
        """Add a model's evaluation results to the comparator."""
        if not os.path.exists(eval_json_path):
            print(f"File not found: {eval_json_path}")
            return
            
        with open(eval_json_path, 'r') as f:
            data = json.load(f)
            
        self.models_data[model_name] = data

    def compare(self):
        """Generate comparison table and plots."""
        if not self.models_data:
            print("No models added. Exiting.")
            return
            
        self._generate_table()
        self._generate_plots()

    def _generate_table(self):
        """Generate and print a comparison table of scalar metrics."""
        headers = ["Metric"] + list(self.models_data.keys())
        
        # Collect all unique metric names
        all_metrics = set()
        for data in self.models_data.values():
            if 'standalone_metrics' in data:
                all_metrics.update(data['standalone_metrics'].keys())
                
        table_data = []
        for metric in sorted(all_metrics):
            row = [metric]
            for model_name, data in self.models_data.items():
                val = data.get('standalone_metrics', {}).get(metric, {}).get('value', 'N/A')
                if isinstance(val, float):
                    val = f"{val:.4f}"
                row.append(val)
            table_data.append(row)
            
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print("\nModel Comparison Table:")
        print(table_str)
        
        table_path = os.path.join(self.output_dir, "comparison_table.txt")
        with open(table_path, "w") as f:
            f.write(table_str)
        print(f"\nTable saved to {table_path}")

    def _generate_plots(self):
        """Generate bar charts for scalar metrics."""
        all_metrics = set()
        for data in self.models_data.values():
            if 'standalone_metrics' in data:
                all_metrics.update(data['standalone_metrics'].keys())
                
        scalar_metrics = []
        for metric in all_metrics:
            is_scalar = True
            for model_name, data in self.models_data.items():
                if 'standalone_metrics' in data and metric in data['standalone_metrics']:
                    val = data['standalone_metrics'][metric].get('value', None)
                    if val is not None and not isinstance(val, (int, float)):
                        is_scalar = False
            
            if is_scalar:
                scalar_metrics.append(metric)
                
        if not scalar_metrics:
            return
            
        scalar_metrics = sorted(scalar_metrics)
        models = list(self.models_data.keys())
        
        # --- Normalization for Average Plotting ---
        scores_dict = {m: {} for m in models} # model -> {metric -> normalized_score (0 to 1)}
        for metric in scalar_metrics:
            is_max = any(term in metric for term in ['LogLikelihood', 'Accuracy', 'Score', 'R2'])
            is_one = ('TSTR' in metric)
            is_zero = ('MMD' in metric or 'MaximumMeanDiscrepancy' in metric)
            
            raw_vals = []
            for m in models:
                v = self.models_data[m].get('standalone_metrics', {}).get(metric, {}).get('value', None)
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
            recon_scores = [scores_dict[m][met] for met in recon_metrics if met in scores_dict[m]]
            synth_scores = [scores_dict[m][met] for met in synth_metrics if met in scores_dict[m]]
            
            self.models_data[m].setdefault('standalone_metrics', {})
            if recon_scores:
                self.models_data[m]['standalone_metrics']['Avg MinMax Reconstruction Score'] = {'value': float(np.mean(recon_scores))}
            if synth_scores:
                self.models_data[m]['standalone_metrics']['Avg MinMax Synthetic Score'] = {'value': float(np.mean(synth_scores))}
                
        if any(recon_metrics) and 'Avg MinMax Reconstruction Score' not in scalar_metrics:
            scalar_metrics.append('Avg MinMax Reconstruction Score')
        if any(synth_metrics) and 'Avg MinMax Synthetic Score' not in scalar_metrics:
            scalar_metrics.append('Avg MinMax Synthetic Score')
        # --- END Normalization ---
        
        n_metrics = len(scalar_metrics)
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle the case of a single subplot or 1D array of subplots
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        import matplotlib.patches as mpatches
        width = 0.8 / len(models)
        
        # Use centralized plot styling
        plot_style = PlotStyle()
        
        for idx, metric in enumerate(scalar_metrics):
            ax = axes[idx]
            
            # Determine if we want to min/max this metric
            # By default, lower error/distance is better (needs to be closest to 0). 
            # Higher is better for LogLikelihood, Accuracy, etc.
            # TSTR should be closest to 1.
            is_maximize = any(term in metric for term in ['LogLikelihood', 'Accuracy', 'Score', 'R2'])
            is_closest_to_one = ('TSTR' in metric)
            is_closest_to_zero = ('MMD' in metric or 'MaximumMeanDiscrepancy' in metric)
            
            m_values = []
            for model in models:
                v = self.models_data[model].get('standalone_metrics', {}).get(metric, {}).get('value', None)
                m_values.append(v)
                
            valid_vals = [v for v in m_values if isinstance(v, (int, float))]
            best_val = None
            if valid_vals:
                if is_closest_to_one:
                    # Find value with minimum absolute difference from 1
                    best_val = min(valid_vals, key=lambda x: abs(x - 1.0))
                elif is_closest_to_zero:
                    # Find value with minimum absolute difference from 0
                    best_val = min(valid_vals, key=lambda x: abs(x))
                elif is_maximize:
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)
            
            # One x-position for the metric
            x = np.arange(1) 
            
            for i, model in enumerate(models):
                val = m_values[i]
                if not isinstance(val, (int, float)):
                    val = 0
                
                is_best = (val == best_val) and (best_val is not None)
                ec = 'black' if is_best else 'none'
                lw = 1.5 if is_best else 0
                hatch = '///' if is_best else ''
                color = plot_style.get_color(i)
                
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
            
            # Format y-axis minimally
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
        # Create a custom legend so we can include the "Best Value" marker style
        legend_handles = [mpatches.Patch(color=plot_style.get_color(i), label=model) for i, model in enumerate(models)]
        legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='★ Best Value'))
        
        # Position the legend above the plots
        # Using 'lower center' ensures the legend grows *upwards* into the bounding box, preventing overlap!
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(models)+1))
        
        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        # Remove empty space
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "metric_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluations between different models.")
    parser.add_argument("--eval_files", nargs='+', required=True, help="List of paths to eval_results.json files.")
    parser.add_argument("--names", nargs='+', required=True, help="List of model names corresponding to the eval files.")
    parser.add_argument("--output_dir", type=str, default="evaluation_comparison", help="Directory to save the comparison plots and table.")
    
    args = parser.parse_args()
    
    if len(args.eval_files) != len(args.names):
        print("Error: The number of evaluation files must match the number of model names.")
        exit(1)
        
    comparator = ModelComparator(args.output_dir)

    args.eval_files = ["models/CVAE_DictionaryCoder_20260318_111021_vamp_userprofiler/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_120522_vamp_softmax_userprofiler/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_144932_vamp_softmax_noUserprofiler/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_154738_leanableDict_softmax_noUserprofiler/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_170149_learnableDict_softmax_Userprofiler/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_172437_learnableDict_softmax_user_profiler_weightedlossSmall/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_175057_learnableDict_softmax_userProfiler_weightedlossLarge/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_183902_learnableDict_softmax_userprofiler_noBetaAnnealing/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_202010_learnableDict_softmax_userprofiler_heirarchy/eval_results.json",
                  "models/CVAE_DictionaryCoder_20260318_212808_vamp_softmax_userprofiler_heirarchy/eval_results.json"
                  ]
    args.names = ["Vamp UserProfiler", "Vamp Softmax UserProfiler", "Vamp Softmax NoUserProfiler", "LeanableDict Softmax NoUserProfiler",
                  "LeanableDict Softmax UserProfiler", "LeanableDict Softmax UserProfiler WeightedLossSmall",
                  "LeanableDict Softmax UserProfiler WeightedLossLarge", "LeanableDict Softmax UserProfiler NoBetaAnnealing",
                    "LeanableDict Softmax UserProfiler Heirarchy", "Vamp Softmax UserProfiler Heirarchy"
                  ]
    
    for name, path in zip(args.names, args.eval_files):
        comparator.add_model(name, path)
        
    comparator.compare()
