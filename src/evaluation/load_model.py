from pathlib import Path
import sys
import os
from types import SimpleNamespace
import torch
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from model.base.factory import _load_train_config
from model.base.factory import create_model_from_config

def load_model(model_path: str, device: str = "cpu", ev_path: str | None = None):
    """
    The model_path is to the folder containing a config file and a few checkpoint files 
    We check first for .yaml files and use that config to construct the model. 
    Then we load the checkpoint file with starts with "best_model_" and ends in .pt
    We return the model, normalizer and conditioner
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Find config file (prefer training_config_used.yaml)
    config_path = None
    preferred = os.path.join(model_path, "training_config_used.yaml")
    if os.path.exists(preferred):
        config_path = preferred
    else:
        for file in os.listdir(model_path):
            if file.endswith(".yaml"):
                config_path = os.path.join(model_path, file)
                break
    
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {model_path}")
    
    # 2. Load config
    config = _load_train_config(config_path)
    if ev_path is not None:
        conditioning_cfg = getattr(config, "conditioning", None)
        if conditioning_cfg is not None:
            ev_cfg = getattr(conditioning_cfg, "ev_phase", None)
            if ev_cfg is None:
                setattr(conditioning_cfg, "ev_phase", SimpleNamespace(path=ev_path))
            else:
                setattr(ev_cfg, "path", ev_path)

    # 3. Load the model
    model = create_model_from_config(config, device)

    # 4. Load the checkpoint — prefer best_model_* with highest epoch number
    import re

    checkpoint_path = None
    max_epoch = -1
    pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]

    # best_model_<type>_<ts1>_<ts2>_<epoch>.pt — take the one with the largest epoch
    for file in pt_files:
        m = re.match(r"best_model_.+_(\d+)\.pt$", file)
        if m:
            epoch = int(m.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                checkpoint_path = os.path.join(model_path, file)

    # Fall back to the most recently modified .pt file if no best_model_ found
    if checkpoint_path is None and pt_files:
        pt_files_full = [os.path.join(model_path, f) for f in pt_files]
        checkpoint_path = max(pt_files_full, key=os.path.getmtime)
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint file found in {model_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    normalizer, conditioner = model.load_model(checkpoint_path, device)
    model.eval() # Set model to evaluation mode for inference since it is the assumed state of load
    return model, normalizer, conditioner
    
if __name__ == "__main__":
    model, normalizer, conditioner = load_model("models/CVAE_20260422_133408")
    
    print("Running a random sample through model: ")
    print(model.sample(conditioner.get_random_conditions(1)))
