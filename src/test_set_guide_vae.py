"""
Smoke-test runner for SetCVAE with phased parquet data.

Usage (from project root, venv active):
    python src/test_set_guide_vae.py
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure src/ and project root are on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import _load_train_config, train_set_vae_phased_smoke
from data.dataloader_phased import StreamingParquetBuffer, WindowSpec
from evaluation.eval_framework import EvalFramework

root = _ROOT
config_path = os.path.join(root, "config", "config.yaml")
out_dir = os.path.join(root, "output", "smoke_set_vae")
os.makedirs(out_dir, exist_ok=True)

# Load config as a SimpleNamespace (same format train() uses internally)
cfg = _load_train_config(config_path)
cfg.skip_validation = True  # never run full CSV validation in a smoke test

# Resolve absolute data_dir
data_dir = cfg.data.data_dir
if not os.path.isabs(data_dir):
    data_dir = os.path.join(root, data_dir)

# Train tiny smoke model (returns trained SetCVAE)
model = train_set_vae_phased_smoke(config=cfg, data_dir=data_dir)

if model is None:
    print("train_set_vae_phased_smoke returned None — check config/parquet files.")
    sys.exit(1)

# Save checkpoint
model.save_model(save_path=out_dir, model_name="set_vae_smoke", device="cpu", normalizer=None)

# ── Build a small standalone eval loader ─────────────────────────────────────
parquet_files = []
for r, _, fs in os.walk(os.path.join(root, "data")):
    for f in fs:
        if f.endswith(".parquet"):
            parquet_files.append(os.path.join(r, f))
parquet_files = sorted(parquet_files)

smoke_cfg = getattr(getattr(cfg, "training", object()), "smoke", None)
max_files = int(getattr(smoke_cfg, "max_files", 1)) if smoke_cfg else 1
eval_batch_size = int(getattr(smoke_cfg, "batch_size", 8)) if smoke_cfg else 8
parquet_files = parquet_files[:max_files]

input_len = int(cfg.window.input_len)
stride = int(cfg.window.stride)
phased = getattr(cfg.data, "phased", None)
replay_buffer_capacity = int(getattr(phased, "replay_buffer_capacity", 512)) if phased else 512
min_buffer_size = int(getattr(phased, "min_buffer_size", 8)) if phased else 8
num_workers = int(getattr(phased, "num_workers", 0)) if phased else 0

ds = StreamingParquetBuffer(
    sm_files=parquet_files,
    replay_buffer_capacity=replay_buffer_capacity,
    min_buffer_size=min_buffer_size,
    window_spec=WindowSpec(input_len=input_len, stride=stride),
    normalizer=None,
    num_workers=num_workers,
)
loader = DataLoader(ds, batch_size=eval_batch_size, num_workers=num_workers)

xs, ms, ts = [], [], []
for i, (x, cond) in enumerate(loader):
    # cond shape: [B, 2, L] — meter_id at [:,0,0], timestamp at [:,1,0]
    meter_id = cond[:, 0, 0].long()
    timestamps = cond[:, 1, 0].long()
    xs.append(x.float())
    ms.append(meter_id)
    ts.append(timestamps)
    if i >= 4:  # only a few batches for a quick smoke test
        break

if not xs:
    print("No batches collected from eval loader. Check parquet files and buffer settings.")
    sys.exit(1)

eval_loader = DataLoader(
    TensorDataset(torch.cat(xs, 0), torch.cat(ms, 0), torch.cat(ts, 0)),
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=num_workers,
)

evaluator = EvalFramework(model, eval_loader, normalizer=None, device="cpu", output_dir=out_dir)
evaluator.register_all()
results = evaluator.evaluate()
print(results)
evaluator.save_results(os.path.join(out_dir, "smoke_eval_results.json"))

print("Saved model + eval results in:", out_dir)
