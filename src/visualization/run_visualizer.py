import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))

from src.evaluation.load_model import load_model
from src.model.factory import _load_train_config
from src.evaluation.compare_models import _build_eval_test_loader, get_eval_files
from src.visualization.visualizer import Visualizer
from src.visualization.plot_styling import PlotStyle

def main():
    parser = argparse.ArgumentParser(description="Run the Visualizer on a model checkpoint.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the model directory (containing config and checkpoint).")
    parser.add_argument("--data-files", nargs="*", default=None, help="Optional specific data files to run on. If none, runs on default files from the config.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional specific data directory containing parquet files.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of windows to evaluate (controls the size of the test loader cache).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots. Defaults to the model directory.")

    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} is not a valid directory.")
        sys.exit(1)

    print(f"Loading model from {model_dir} ...")
    try:
        model, normalizer, conditioner = load_model(model_dir, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Find the config to figure out data loading details
    yaml_path = os.path.join(model_dir, "training_config_used.yaml")
    if not os.path.exists(yaml_path):
        import glob
        yamls = glob.glob(os.path.join(model_dir, "*.yaml"))
        for y in yamls:
            if "base" not in os.path.basename(y).lower():
                yaml_path = y
                break

    print(f"Loading config from {yaml_path} ...")
    try:
        cfg = _load_train_config(yaml_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    data_files = args.data_files
    data_dir = args.data_dir if args.data_dir else getattr(cfg.data, "data_dir", None)
    
    if data_files is None and data_dir:
        data_files = get_eval_files(data_dir)
    elif data_files is None:
        data_files = getattr(cfg.data, "data_files", [])
    elif data_files and len(data_files) == 1 and os.path.isdir(data_files[0]):
        # The user accidentally passed a directory to --data-files
        data_files = get_eval_files(data_files[0])
    window_len = getattr(cfg.window, "input_len", 24)
    split_seed = getattr(cfg.splits, "random_seed", 42)
    split_ratios = {
        "train_ratio": getattr(cfg.splits, "train_ratio", 0.8),
        "val_ratio": getattr(cfg.splits, "val_ratio", 0.1),
        "test_ratio": getattr(cfg.splits, "test_ratio", 0.1),
    }
    max_value = getattr(cfg.data, "max_value", 200.0)
    data_mode = str(getattr(cfg.data, "mode", "csv")).lower()
    phased = data_mode in {"phased", "phased_parquet"}
    agg_divisor = 1.0 # Default fallback, compare_models typically handles hourly vs sub-hourly here.

    out_dir = args.output_dir if args.output_dir else model_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Building dataloader (this may take a minute) ...")
    test_loader = _build_eval_test_loader(
        data_files=data_files,
        split_seed=split_seed,
        split_ratios=split_ratios,
        max_value=max_value,
        window_len=window_len,
        normalizer=normalizer,
        num_samples=args.num_samples,
        output_dir=out_dir,
        phased=phased,
        agg_divisor=agg_divisor,
        output_aggregate=False,
    )

    if test_loader is None:
        print("Warning: Could not build test_loader. Some visualization functions will be skipped or may crash.")

    print("Instantiating Visualizer ...")
    plot_style = PlotStyle()
    visualizer = Visualizer(
        plot_style=plot_style,
        log_dir=out_dir,
        model=model,
        normalizer=normalizer,
        device=args.device,
        conditioner=conditioner
    )

    print("Running Visualizations ...")

    # Functions that don't need test_loader
    print("- visualize_user_dict")
    try:
        visualizer.visualize_user_dict()
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        print("- visualize_sigma_mapper")
        visualizer.visualize_sigma_mapper()
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        print("- visualize_synthetic")
        visualizer.visualize_synthetic(num_samples=50)
    except Exception as e:
        print(f"  Failed: {e}")

    # Functions that DO need test_loader
    if test_loader is not None:
        try:
            print("- visualize_reconstructions")
            visualizer.visualize_reconstructions(test_loader, num_recon_per=10, num_samples=10)
        except Exception as e:
            print(f"  Failed: {e}")

        try:
            print("- visualize_latent_space")
            visualizer.visualize_latent_space(test_loader, PCA=True, tSNE=True, max_samples=500)
        except Exception as e:
            print(f"  Failed: {e}")

        try:
            print("- visualize_synthetic_recreations")
            visualizer.visualize_synthetic_recreations(test_loader, num_originals=4, num_synthetic_per=50)
        except Exception as e:
            print(f"  Failed: {e}")

        try:
            print("- visualize_5synth_5real_and_mean")
            visualizer.visualize_5synth_5real_and_mean(test_loader, num_samples=5)
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\nVisualizations complete! All plots were saved to: {out_dir}")

if __name__ == "__main__":
    main()
