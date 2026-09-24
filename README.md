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
    balanced_phase/    # VAE family + Faraday family models (joint 3-phase output)
      hierarchical/    # Hierarchy support for VAE-family models
    three_single_phased/ # Models that process each phase independently
    base/            # Base classes, factory, priors, GMM, baselines
    scripts/         # Helper scripts: parameter budget counting
  power_flow/      # Pandapower network builder and load flow integration
  test/            # Pytest test suite
  visualization/   # Plot styling and output writers
config/tests/      # Per-model experiment configs (base + override YAML pairs)
slurmjobs/         # SLURM job scripts for DTU HPC cluster
```

## Installation

Requires Python 3.11+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

For CUDA on Linux, edit `pyproject.toml` to add the `pytorch-cu130` index marker for `sys_platform == 'linux'` (see comment in the file).

## Training a model

```bash
uv run src/train.py --config config/tests/agg/guided_faraday_cnn.yaml
```

Each config in `config/tests/` is an override config: it declares a `base:` path pointing to the shared base config for that experiment set, and adds model-specific architecture settings. The base config sets data paths, normalisation, conditioning, and training hyperparameters. Phased models use configs under `config/tests/phased/`:

```bash
uv run src/train.py --config config/tests/phased/diffusion.yaml
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

## Cite
If using the GUIDED-Faraday model for your work please cite the following.
@thesis{kold2026a,
  author = {Kold, Clara Regine Hoeg and Fiehn, Rasmus Lyhne},
  title = {Conditional generative modeling for statistically representative per phase smart meter time series},
  language = {eng},
  format = {thesis},
  year = {2026},
  publisher = {DTU Department of Wind and Energy Systems}
}
