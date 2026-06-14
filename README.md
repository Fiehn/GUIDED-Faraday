# Conditional Generative Modeling for Statistically Representative Per-Phase Smart Meter Time Series

Thesis project generating synthetic 3-phase smart meter time series using conditional generative models (VAE family, Faraday family, diffusion). Models are trained on Danish household smart meter data and evaluated on statistical fidelity to real distributions, including cross-customer coincidence structure and grid-level power flow metrics.

## Repository structure

```
src/
  config/          # Config schema and loader (base + override YAML composition)
  data/            # Dataloader, normalizer, conditioners
    scripts/       # Helper scripts: subset creation, dataloader caching
  evaluation/      # EvalFramework, metrics, compare_models entry point
    metrics/       # Individual metric implementations (MMD, ACF, TSTR, MIA, ...)
    scripts/       # Helper scripts: checkpoint sweep, condition mixing analysis
  model/           # All generative models and priors
    hierarchical/  # Hierarchy support for VAE-family models
    scripts/       # Helper scripts: parameter budget counting
  power_flow/      # Pandapower network builder and load flow integration
  test/            # Pytest test suite
  visualization/   # Plot styling and output writers
config/            # Base config and per-condition YAML overrides
slurmjobs/         # SLURM job scripts for DTU HPC cluster
scripts/           # One-off helper scripts (e.g. find_holdout_feeder)
```

Domain vocabulary (Window, Meter, LV Feeder, Holdout feeder pool, Conditioner, etc.) is defined in `CONTEXT.md`. Architecture decisions are in `docs/adr/`.

## Installation

Requires Python 3.11+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

For CUDA on Linux, edit `pyproject.toml` to add the `pytorch-cu130` index marker for `sys_platform == 'linux'` (see comment in the file).

## Training a model

```bash
uv run src/train.py --config config/config.yaml
```

The base config at `config/config.yaml` sets data paths, normalization, conditioning, and training hyperparameters. Per-model architecture is supplied via an override config that declares `base: config/config.yaml` and adds model-specific keys:

```bash
uv run src/train.py --config config/set/set.yaml
```

On the DTU HPC cluster, use the SLURM scripts in `slurmjobs/`:

```bash
sbatch slurmjobs/faraday.slurm
```

## Evaluation

Evaluation and model comparison are run through `compare_models.py`:

```bash
uv run src/evaluation/compare_models.py \
  --models models/faraday models/diffusion models/cvae \
  --names "Faraday" "Diffusion" "CVAE" \
  --output-dir evaluation_comparison \
  --phased
```

Results are written to `evaluation_comparison/` as JSON files and PDF figures. For phased models, per-phase breakdowns are included automatically.

## Running tests

```bash
uv run pytest
```

## Key concepts

- **Window**: fixed-length 24-hour time slice, the atomic unit of model input/output
- **Phased data**: 15-minute resolution 3-phase readings (distinct from hourly aggregated data)
- **Conditioner**: produces a fixed-size conditioning vector per window (day-of-week, EV presence, user profile, ...)
- **Holdout feeder pool**: LV Feeders jointly excluded from training for OOD evaluation