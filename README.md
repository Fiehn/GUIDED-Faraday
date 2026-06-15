# Synthetic Smart-Meter Load Profile Generator

Generate realistic 24-hour electricity consumption profiles from trained generative models.

## Installation

```bash
uv sync # installs: torch, numpy, pandas, pydantic, pyyaml, scikit-learn, scipy
```
or
```bash
pip install pyproject.toml # installs: torch, numpy, pandas, pydantic, pyyaml, scikit-learn, scipy
```

## Quick start

```python
from synth import load_model, generate, get_condition_info

# Load by name -- resolves to models/<name>/ automatically
bundle = load_model("GUIDED_Faraday_CNN")

# 100 independent 24-hour profiles  ->  np.ndarray  [100, 24, 1]
samples = generate(bundle, n=100)

# Pin temporal and user-profile conditions
samples = generate(bundle, n=100, meter_id=42, date="2024-06-14")

# Override appliance conditions
samples = generate(bundle, n=100, ev=True, hp=2, pv=False)

# 7-day continuous sequence with smooth midnight transitions  ->  [7*24, 1]
week = generate(bundle, n=7, meter_id=42, date="2024-06-14", continuous=True)

# Full path also accepted
bundle = load_model("/absolute/path/to/checkpoint/folder/")
```

---

## Available models

Place model folders from trained models under `models/`.

| Name (folder in `models/`) | Architecture | Phase | Notes |
|---|---|---|---|
| `GUIDED_Faraday_CNN` | Guided-Faraday-CNN | Balanced | RECOMENDED |
| `GUIDED_Faraday_CNN_UserProfiles` | Guided-Faraday-CNN-UserProfiles | Balanced | |
| `Diffusion` | Diffusion | Balanced | |
| `Faraday` | Faraday | Balanced | |
| `GUIDE_VAE` | GUIDE-VAE | Balanced | |
| `Baseline_Quantile` | Quantile Baseline | Balanced | |
| `Baseline_Average` | Average Baseline | Balanced | |
| `Set_GUIDE_VAE` | Set-Guide-VAE | 3-phase | |
| `Set_GUIDE_VAE_CNN` | Set-Guide-VAE-CNN | 3-phase | RECOMENDED |
| `Sequential` | Sequential | 3-phase | |
| `Diffusion_phased` | Diffusion | 3-phase | |
| `Hierarchical Splitter` | Hierarchical Splitter | 3-phase | |
| `Average_Baseline` | Average Baseline | 3-phase | |
| `Quantile_Baseline` | Quantile Baseline | 3-phase | |

---

## Conditions reference

All condition parameters are **optional** -- omit any to sample it randomly from the
training distribution. Not every model uses every condition; call
`get_condition_info(bundle)` to see what a specific model supports.

### `meter_id` — user profile

Pins consumption patterns to a specific meter seen during training.

```python
samples = generate(bundle, n=50, meter_id=42)   # specific user
samples = generate(bundle, n=50)                 # random user
```

### `date` — temporal features

Sets day-of-week and month, and holiday flag. Accepts `"YYYY-MM-DD"` or `datetime`.

```python
samples = generate(bundle, n=50, date="2024-12-25")   # Christmas
samples = generate(bundle, n=50)                        # random date
```

### `ev` — electric vehicle (binary)

```python
samples = generate(bundle, n=50, ev=True)    # has EV
samples = generate(bundle, n=50, ev=False)   # no EV
```

### `hp` — heat pump (multi-class)

`False` = no heat pump, `True` = class 1, or integer for specific class.

```python
samples = generate(bundle, n=50, hp=False)   # no heat pump
samples = generate(bundle, n=50, hp=1)       # heat pump on 1 phase
samples = generate(bundle, n=50, hp=2)       # heat pump on 2 phase
samples = generate(bundle, n=50, hp=3)       # heat pump on 3 phase
```

### `pv` — solar panels (binary)

```python
samples = generate(bundle, n=50, pv=True)    # has solar panels
samples = generate(bundle, n=50, pv=False)   # no solar panels
```

### Combining conditions

Any combination is valid. Omitted conditions are sampled randomly.

```python
# Summer day, no EV, heat pump class 1, no solar
samples = generate(bundle, n=100,
                   date="2024-07-01", ev=False, hp=1, pv=False)

# A week of continuous generation, EV user with class-2 heat pump
week = generate(bundle, n=7,
                meter_id=42, date="2024-07-01",
                ev=True, hp=2, continuous=True)
```

### Inspecting a loaded model

```python
info = get_condition_info(bundle)
# Example output:
# {
#   "day_of_week":   [0, 1, 2, 3, 4, 5, 6],
#   "month_of_year": [7, 8, ..., 18],
#   "holidays":      [19],
#   "ev_phase":      [20, 21],              # binary: no-EV, has-EV
#   "hp":            [22, 23, 24, 25],      # 4 classes: none + 3 HP types
#   "pv":            [26, 27],              # binary: no-PV, has-PV
#   "user_profile":  [28, ..., 37],
# }
```

---

## Output shape

| Call | Output shape |
|---|---|
| `generate(bundle, n=100)` | `[100, 24, 1]` — 100 profiles, kWh per hour |
| `generate(bundle, n=100)` on 3-phase model | `[100, 24, 3]` — one column per phase |
| `generate(bundle, n=7, continuous=True)` | `[168, 1]` — 7 days end-to-end |

Values are in kWh (original scale, inverse-normalised by the loaded model).