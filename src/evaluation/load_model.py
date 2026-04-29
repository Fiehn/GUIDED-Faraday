from pathlib import Path
import sys
import os
import torch
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from src.model.factory import _load_train_config
from src.model.factory import create_model_from_config

def load_model(model_path: str, device: str = "cpu"):
    """
    The model_path is to the folder containing a config file and a few checkpoint files 
    We check first for .yaml files and use that config to construct the model. 
    Then we load the checkpoint file with starts with "best_model_" and ends in .pt
    We return the model, normalizer and conditioner
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Find config file
    config_path = None
    for file in os.listdir(model_path):
        if file.endswith(".yaml"):
            config_path = os.path.join(model_path, file)
            break
    
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {model_path}")
    
    # 2. Load config
    config = _load_train_config(config_path)

    # 3. Load the model
    model = create_model_from_config(config, device)

    # 4. Load the checkpoint
    checkpoint_path = None
    for file in os.listdir(model_path):
        if file.startswith("best_model_") and file.endswith(".pt"):
            checkpoint_path = os.path.join(model_path, file)
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint file found in {model_path}")
    
    normalizer, conditioner = model.load_model(checkpoint_path, device)
    model.eval() # Set model to evaluation mode for inference since it is the assumed state of load
    return model, normalizer, conditioner
    
if __name__ == "__main__":
    model, normalizer, conditioner = load_model("models/CVAE_20260422_133408")
    
    print("Running a random sample through model: ")
    print(model.sample(conditioner.get_random_conditions(1)))
