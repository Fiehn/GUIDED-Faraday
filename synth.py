"""Inference API for 3-phase synthetic smart-meter load profiles.

Quick start::

    from synth import load_model, generate, get_condition_info

    # Single model
    bundle  = load_model("GUIDED_Faraday_CNN")
    samples = generate(bundle, n=100)                         # [100, 24, 1]

    # Phase splitter (uses GUIDED_Faraday_CNN as primary by default)
    bundle = load_model("HierarchicalSplitter")
    samples = generate(bundle, n=100)                         # [100, 24, 3]

    # Phase splitter with explicit primary
    bundle = load_model("HierarchicalSplitter", primary="Diffusion")
    samples = generate(bundle, n=100)                         # [100, 24, 3]

    # Condition overrides (randomly sampled if omitted)
    samples = generate(bundle, n=50, ev=True, hp=2, pv=False)

    week = generate(bundle, n=7, meter_id=42,
                    date="2024-06-14", continuous=True)       # [7*24, 3]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEFAULT_PRIMARY = "GUIDED_Faraday_CNN"


@dataclass
class ModelBundle:
    """Everything needed to generate from a trained model (or model pair)."""
    model: object
    normalizer: object
    conditioner: object
    device: str
    secondary_model: object = field(default=None)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _resolve_path(name_or_path: str) -> str:
    p = Path(name_or_path)
    if not p.exists():
        p = Path(__file__).parent / "models" / name_or_path
    return str(p)


def _load_single(path_or_name: str, device: str) -> ModelBundle:
    from evaluation.load_model import load_model as _load
    model, norm, cond = _load(_resolve_path(path_or_name), device=device)
    return ModelBundle(model=model, normalizer=norm, conditioner=cond, device=device)


def load_model(
    path_or_name: str,
    device: str = "auto",
    *,
    primary: str | ModelBundle | None = None,
) -> ModelBundle:
    """Load a trained model by name or by path.

    For the :class:`HierarchicalSplitter` (phase splitter), the loaded model
    becomes the *secondary* model and a primary generator is loaded alongside it.
    Pass ``primary=`` to choose the primary; omit it to default to
    ``"GUIDED_Faraday_CNN"``.

    Args:
        path_or_name: Bare model name (resolved to ``models/<name>/``) or full path.
        device: ``"auto"`` picks CUDA if available, otherwise CPU.
        primary: Primary generator name or :class:`ModelBundle` to pair with a
                 phase splitter.  Only used when the loaded model is a splitter.

    Returns:
        :class:`ModelBundle` ready to pass to :func:`generate`.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = _load_single(path_or_name, device)

    is_splitter = (
        type(bundle.model).__name__ == "HierarchicalSplitter"
        or primary is not None
    )

    if is_splitter:
        secondary_model = bundle.model
        if primary is None:
            primary = _DEFAULT_PRIMARY
        primary_bundle = primary if isinstance(primary, ModelBundle) else _load_single(primary, device)
        return ModelBundle(
            model=primary_bundle.model,
            normalizer=primary_bundle.normalizer,
            conditioner=primary_bundle.conditioner,
            device=device,
            secondary_model=secondary_model,
        )

    return bundle


# ---------------------------------------------------------------------------
# Internal: condition override helper
# ---------------------------------------------------------------------------

def _find_tag(conditioner, hint: str) -> str | None:
    if conditioner is None or not hasattr(conditioner, "get_condition_tags"):
        return None
    return next((t for t in conditioner.get_condition_tags() if hint in t.lower()), None)


def _apply_overrides(
    conditions: torch.Tensor,
    conditioner,
    *,
    ev: bool | None,
    hp: bool | int | None,
    pv: bool | None,
) -> torch.Tensor:
    cond = conditions.clone()

    def _set(hint: str, value: bool | int) -> None:
        tag = _find_tag(conditioner, hint)
        if tag is None:
            return
        start, end = conditioner.get_condition_indexes(tag)
        dim = end - start
        if dim == 1:
            cond[:, start] = float(bool(value))
        else:
            cond[:, start:end] = 0.0
            idx = 1 if value is True else (0 if value is False else int(value))
            if 0 <= idx < dim:
                cond[:, start + idx] = 1.0

    if ev is not None:
        _set("ev", ev)
    if hp is not None:
        _set("hp", hp)
    if pv is not None:
        for hint in ("pv", "solar"):
            if _find_tag(conditioner, hint) is not None:
                _set(hint, pv)
                break

    return cond


def _build_conditions_with_overrides(
    gen,
    bundle: ModelBundle,
    n: int,
    meter_id: int | None,
    date: datetime | None,
    ev: bool | None,
    hp: bool | int | None,
    pv: bool | None,
) -> torch.Tensor | None:
    if bundle.conditioner is None:
        return None
    mid = meter_id if meter_id is not None else 0
    ts = date if date is not None else datetime.now()
    base = gen._build_conditions(meter_id=mid, timestamp=ts)  # [1, C]
    if base.shape[0] == 1:
        base = base.expand(n, -1).clone()
    return _apply_overrides(base, bundle.conditioner, ev=ev, hp=hp, pv=pv)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    bundle: ModelBundle,
    n: int,
    *,
    meter_id: int | None = None,
    date: str | datetime | None = None,
    ev: bool | None = None,
    hp: bool | int | None = None,
    pv: bool | None = None,
    continuous: bool = False,
) -> np.ndarray:
    """Generate synthetic 24-hour load profiles.

    Args:
        bundle: Returned by :func:`load_model`.
        n: Number of independent profiles, or number of days when
           ``continuous=True``.
        meter_id: Meter ID for user-profile conditioning. Omit for random.
        date: Temporal conditioning as ``"YYYY-MM-DD"`` or :class:`datetime`.
              Omit for random.
        ev: Electric-vehicle presence. ``True`` = has EV, ``False`` = no EV.
            Omit to sample randomly.
        hp: Heat pump class. ``False`` = no HP, ``True`` = class 1,
            integer for specific class. Omit to sample randomly.
        pv: Solar panel presence. ``True`` = has PV, ``False`` = no PV.
            Omit to sample randomly.
        continuous: Stitch *n* days end-to-end with smooth midnight transitions.

    Returns:
        ``np.ndarray``:

        * ``continuous=False`` → ``[n, 24, features]``
        * ``continuous=True``  → ``[n * 24, features]``

        *features* is 1 for aggregated models, 3 for 3-phase models.
    """
    from model.base.generators import SyntheticGenerator

    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    gen = SyntheticGenerator(
        model=bundle.model,
        conditioner=bundle.conditioner,
        normalizer=bundle.normalizer,
        secondary_model=bundle.secondary_model,
        device=bundle.device,
    )

    needs_override = any(x is not None for x in [ev, hp, pv])

    if continuous:
        if date is None:
            date = datetime.now()
        cond_tensor = None
        if needs_override:
            cond_tensor = _build_conditions_with_overrides(
                gen, bundle, 1, meter_id, date, ev=ev, hp=hp, pv=pv
            )
        samples, _ = gen.generate_continuous(
            num_days=n,
            meter_id=meter_id if meter_id is not None else 0,
            start_date=date,
            conditions=cond_tensor,
        )
    else:
        if needs_override:
            cond_tensor = _build_conditions_with_overrides(
                gen, bundle, n, meter_id, date, ev=ev, hp=hp, pv=pv
            )
            samples, _ = gen.generate(num_samples=n, conditions=cond_tensor)
        else:
            samples, _ = gen.generate(num_samples=n, meter_id=meter_id, date=date)

    if samples is None:
        raise RuntimeError("Model returned no samples.")
    return samples.detach().cpu().numpy()


def get_condition_info(bundle: ModelBundle) -> dict[str, list[int]]:
    """Return the conditioning slots the model understands.

    Args:
        bundle: Returned by :func:`load_model`.

    Returns:
        Dict mapping each condition tag to its indices in the condition vector.
        Empty dict for unconditioned models.
    """
    c = bundle.conditioner
    if c is None or not hasattr(c, "get_condition_tags"):
        return {}
    return {
        tag: list(range(*c.get_condition_indexes(tag)))
        for tag in c.get_condition_tags()
    }