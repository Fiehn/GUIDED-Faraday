import os
import glob
import argparse
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))
from src.model.factory import _load_train_config

CONFIG_MAPPING = {
    # Core Model
    "model.type": ("Model Architecture", "The core architecture class used for generative modeling"),
    "model.latent_dim": ("Latent Dimension ($Z$)", "Dimensionality of the bottleneck latent representation"),

    # Encoder / Decoder
    "model.encoder.type": ("Encoder Type", "Architecture of the encoder network"),
    "model.encoder.hidden_dims": ("Encoder Hidden Layers", "Number of units per hidden layer in the encoder"),
    "model.encoder.dropout": ("Encoder Dropout", "Dropout probability applied in the encoder"),
    "model.decoder.type": ("Decoder Type", "Architecture of the decoder network"),
    "model.decoder.hidden_dims": ("Decoder Hidden Layers", "Number of units per hidden layer in the decoder"),
    "model.decoder.dropout": ("Decoder Dropout", "Dropout probability applied in the decoder"),
    "model.decoder.mu_activation": ("Decoder $\\mu$ Activation", "Activation function applied to the output mean"),

    # Prior
    "model.prior.type": ("Prior Distribution Type", "Type of prior distribution (e.g., standard Gaussian, learnable dictionary, VampPrior)"),
    "model.prior.dictionary_size": ("Prior Dictionary Size", "Number of components in the learnable prior dictionary"),
    "model.prior.vamp_num_components": ("VampPrior Components", "Number of pseudo-inputs used in VampPrior"),

    # Hierarchy
    "hierarchy.enabled": ("Hierarchy Enabled", "Whether hierarchical latent variables are utilized"),
    "hierarchy.levels": ("Hierarchy Layer Types", "Type of mechanism at each hierarchy level (e.g., linear)"),
    "hierarchy.latent_dims": ("Hierarchy Latent Dims", "Dimensionality of latent variables at each hierarchical level"),
    "hierarchy.loss_weights": ("Hierarchy Loss Weights", "KL-divergence penalty weighting for each hierarchical level"),

    # Guided Faraday & Faraday
    "model.faraday.mse_weight": ("MSE Loss Weight", "Multiplier for the Mean Squared Error reconstruction loss"),
    "model.faraday.mmd_weight": ("MMD Loss Weight", "Multiplier for the Maximum Mean Discrepancy (MMD) distribution loss"),
    "model.faraday.upper_quantile_weight": ("Upper Quantile Weight", "Multiplier for the pinball loss on the upper quantile"),
    "model.faraday.lower_quantile_weight": ("Lower Quantile Weight", "Multiplier for the pinball loss on the lower quantile"),
    "model.faraday.median_quantile_weight": ("Median Quantile Weight", "Multiplier for the pinball loss on the median"),
    "model.gmm.n_components": ("GMM Components", "Number of Gaussian components in the fitted feature mixture model"),
    "model.gmm.max_iter": ("GMM Max Iterations", "Maximum Expectation-Maximization steps for GMM fitting"),
    "model.guided_faraday.pdcc_average_max_std": ("PDCC Average Max STD", "Maximum average PDCC value"),
    "model.guided_faraday.dictionary_size": ("PDCC Dictionary Size", "Number of conditional embeddings in the guidance mechanism"),
    "model.guided_faraday.stochastic_loss": ("Stochastic Guidance", "Whether loss evaluation samples from distributions or relies on expected values"),

    # Phase Splitter
    "model.phase_splitter.base_channels": ("Splitter Base Channels", "Initial number of convolutional channels in the phase splitter"),
    "model.phase_splitter.temperature": ("Splitter Temperature", "Softmax temperature scaling applied during phase attribution"),

    # Set VAE
    "model.set.hidden_dim": ("Set VAE Hidden Dim", "Dimensionality of representations inside Set VAE Transformer blocks"),
    "model.set.num_heads": ("Set VAE Attention Heads", "Number of attention heads used in multi-head attention layers"),
    "model.set.num_inds": ("Set VAE Inducing Points", "Number of inducing points utilized in the Induced Set Attention Blocks (ISAB)"),
    "model.set.num_blocks": ("Set VAE Blocks", "Number of stacked Set VAE attention layers"),

    # Diffusion
    "model.diffusion.num_timesteps": ("Diffusion Timesteps", "Number of discrete steps in the forward/reverse diffusion process"),
    "model.diffusion.schedule_type": ("Noise Schedule", "Mathematical schedule dictating noise variance across timesteps (e.g., cosine)"),
    "model.diffusion.base_channels": ("UNet Base Channels", "Number of initial channels in the Conditional UNet denoiser"),
    "model.diffusion.reconstruction_noise_fraction": ("Evaluation Noise Fraction", "Fraction of initial noise applied when performing synthetic generation"),
    
    # Conditioning
    "conditioning.tags": ("Conditioning Tags", "List of metadata features utilized for conditional generation"),
    "conditioning.user_profile.num_clusters": ("User Profile Clusters", "Number of distinct behavioral clusters utilized in the profile conditioner"),
    "conditioning.user_profile.n_topics": ("User Profile Topics", "Number of latent topics utilized in the profile conditioner"),
}

def _get_nested_val(ns, key_path):
    keys = key_path.split('.')
    current = ns
    for k in keys:
        if isinstance(current, SimpleNamespace):
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                return None
        elif isinstance(current, dict):
            if k in current:
                current = current[k]
            else:
                return None
        else:
            return None
    return current

def extract_curated_parameters(cfg):
    """Extract parameters from config based on CONFIG_MAPPING."""
    extracted = []
    
    # We always iterate through our predefined list of interesting params.
    for key, (latex_name, description) in CONFIG_MAPPING.items():
        val = _get_nested_val(cfg, key)
        
        # We only add the row if the value is defined and not explicitly "None" (in a way that implies unused)
        # However, boolean False is valid, 0 is valid.
        if val is not None:
            # Clean up the value representation for LaTeX
            if isinstance(val, list):
                val_str = f"[{', '.join(str(v) for v in val)}]"
            elif isinstance(val, bool):
                val_str = "True" if val else "False"
            elif isinstance(val, float):
                # Format small floats nicely
                if val < 0.001 and val > 0:
                    val_str = f"{val:.1e}"
                else:
                    val_str = str(val)
            else:
                val_str = str(val)

            # Replace underscores with escaped underscores for LaTeX
            val_str = val_str.replace("_", "\\_")

            extracted.append((latex_name, val_str, description))
            
    return extracted

def generate_latex_table(params, model_name):
    """Generate a booktabs LaTeX table from a list of parameter tuples."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llp{8cm}@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Parameter} & \\textbf{Value} & \\textbf{Description} \\\\")
    lines.append("\\midrule")
    
    for param_name, value, description in params:
        # Escape specific LaTeX characters if needed, although mostly we control the names.
        lines.append(f"{param_name} & {value} & {description} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{Configuration parameters used for training the \\textbf{{{model_name}}} model.}}")
    lines.append(f"\\label{{tab:config_{model_name.replace(' ', '_').replace('-', '_').lower()}}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def _find_model_config_path(model_dir: str) -> str | None:
    yaml_path = os.path.join(model_dir, "training_config_used.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    
    # Fallback to any yaml if multiple exist, but usually it's training_config_used.yaml
    yamls = glob.glob(os.path.join(model_dir, "*.yaml"))
    for y in yamls:
        if "base" not in os.path.basename(y).lower():
            return y
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX configuration tables for models.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model directories (each contains config + checkpoint).")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for models (same length as --models).")
    parser.add_argument("--output-dir", type=str, default="evaluation_comparison", help="Central directory where evaluation comparisons are saved.")
    
    args = parser.parse_args()

    args.models = [
                #    "E:/models/new4/CVAE_20260601_034519_guide_cnn",
                #    "E:/models/new4/FaradayModel_20260531_124255",
                #    "E:/models/new4/FaradayModel_20260601_060319",
                #    "E:/models/new4/GuidedFaradayModel_20260531_115215",
                #    "E:/models/new4/GuidedFaradayModel_20260531_131653_relu",
                #    "E:/models/new4/GuidedFaradayModel_20260531_143252_peaks",
                #    "E:/models/new4/GuidedFaradayModel_20260531_204015",
                #    "E:/models/new4/GuidedFaradayModel_20260531_211801_cnn",
                # "E:/models/new4/GuidedFaradayModel_20260531_225823_userprofile",
                # "E:/models/new3/FaradayModel_20260529_021321", # NOT for holdout
                # "E:/models/new3/GuidedFaradayModel_20260528_223126", # NOt for holdout
                # "E:/models/new3/GuidedFaradayModel_20260529_000300_cnn", # NOT for holdout
                # "E:/models/new3/GuidedFaradayModel_20260529_012840_cnn_mmd" # NOT for holdout
                #"E:/models/new4/GuidedFaradayModel_20260604_102425_deterministic", # use
                #"E:/models/new4/GuidedFaradayModel_20260603_123450", # use
                "E:/models/new4/GuidedFaradayModel_20260605_105946",# "E:/models/new4/GuidedFaradayModel_20260605_112623",
                "E:/models/new4/GuidedFaradayModel_20260605_201145",
                #"E:/models/new4/GuidedFaradayModel_20260605_173422",
                # "E:/models/new4/GuidedFaradayModel_20260603_120241", # use
                # "E:/models/new4/GuidedFaradayModel_20260603_102551", # use
                "E:/models/new4/CVAE_20260531_185015_guide_cnn_h_broken",
                "E:/models/new4/DiffusionModel_20260601_002842", # use
                "E:/models/new4/FaradayModel_20260601_060319", # USe
                "E:/models/new4/CVAE_20260601_022452_guide", # USE
                "E:/models/new4/CVAE_20260531_180003_cvae",
                "E:/models/new4/QuantileBaselineModel_20260601_005326",
                "E:/models/new4/AverageBaselineModel_20260531_173051",
                   ]
    args.names = [
                    # "Guide-CNN",
                    # "Faraday",
                    # "Faraday-new-fix",
                    # "Guided-Faraday",
                    # "Guided-Faraday-RELU",
                    # "Guided-Faraday-PEAKS",
                    # "Guided-Faraday-Deterministic",
                    # "Guided-Faraday-CNN-2",
                    # "Guided-Faraday-UserProfile",
                    # "Faraday-OLD",
                    # "Guided-Faraday-OLD",
                    # "Guided-Faraday-CNN-OLD",
                    # "Guided-Faraday-CNN-MMD-OLD",
                    #"Guided-Faraday-CNN-Deterministic-big", # use
                    #"Guided-Faraday-CNN-Stochastic-big", # Use
                    "Guided-Faraday-CNN",# "Guided-Faraday-CNN-Stochastic",
                    "Guided-Faraday-CNN-UserProfiles",
                    #"Guided-Faraday-CNN-newGMM",
                    #"Guided-Faraday-UserProfiles", # use
                    #"Guided-Faraday", # USe
                    "H-Guide-CNN", # use
                    "Diffusion", # use
                    "Faraday", # USE
                    "Guide", # USE
                    "CVAE", # use
                    "Baseline-Quantile", # use
                    "Baseline-Average", # use
    ]

    if len(args.models) != len(args.names):
        raise ValueError("--models and --names must have the same length.")

    for path, name in zip(args.models, args.names):
        print(f"Processing model: {name} at {path}")
        
        cfg_path = _find_model_config_path(path)
        if cfg_path is None:
            print(f"  [ERROR] No configuration YAML found in {path}. Skipping.")
            continue
            
        try:
            cfg = _load_train_config(cfg_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load config from {cfg_path}: {e}")
            continue
            
        # Extract the values
        params = extract_curated_parameters(cfg)
        
        if not params:
            print("  [WARNING] No relevant configuration parameters extracted.")
            continue
            
        # Generate LaTeX
        latex_str = generate_latex_table(params, name)
        
        # Save to model directory
        output_file = os.path.join(path, "config_latex_table.txt")
        with open(output_file, "w") as f:
            f.write(latex_str)
            
        # Save to evaluation comparison directory
        # We replace spaces to align with standard folder naming or just use name directly.
        # compare_models usually uses the raw name but lets be safe.
        safe_name = name.replace("/", "_").replace("\\", "_")
        eval_model_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(eval_model_dir, exist_ok=True)
        eval_output_file = os.path.join(eval_model_dir, "config_latex_table.txt")
        with open(eval_output_file, "w") as f:
            f.write(latex_str)
            
        print(f"  [SUCCESS] Wrote LaTeX table to {output_file} and {eval_output_file}")

if __name__ == "__main__":
    main()
