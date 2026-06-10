

import os
import sys
import shutil
import torch
import json
import numpy as np
from pathlib import Path
import argparse

# Add project roots for imports
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.evaluation.load_model import load_model
from src.model.factory import _load_train_config
from src.data.dataloader_phased import PhasedReplayDataloaders, WindowSpec, _extract_meter_id_from_filename
from src.evaluation.eval_framework import EvalFramework, PhasedEvaluationSuite, VisualEvaluation
from src.visualization.plot_styling import PlotStyle
from src.data.feeder_holdout import (
    HoldoutWindows,
    HoldoutPhasedWindows,
    load_holdout_windows_raw,
    load_holdout_windows_phased,
    plot_feeder_aggregated_profile,
    plot_feeder_holdout_analysis,
    run_coincidence_factor_analysis,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not installed
    def tabulate(data, headers="keys", **kwargs):
        return str(data)

class PhasedModelComparator:
    """Class to compare evaluation results across multiple phased and unphased models."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, output_dir: str = "evaluation_comparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_data = {}
        self._model_eval_dirs: dict[str, str] = {}
        self._param_counts: dict[str, int | None] = {}
        # We define a superset of possible phases.
        self.phases = ['full_eval', 'aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3', 'cross_phase', 'c2st']

    def _metric_target_label(self, metric: str) -> str:
        """Return target direction/target value for plot titles."""
        display_name = self._display_metric_name(metric)

        if metric.startswith("Avg MinMax") or "BEST Overall" in display_name:
            return "↑"

        if "TSTR" in metric or "Train on Synthetic" in display_name:
            return "target=1"

        if "MMD" in metric or "MaximumMeanDiscrepancy" in metric or "Maximum Mean Discrepancy" in display_name:
            return "target=0"

        if any(term in metric for term in ["LogLikelihood", "Log-Likelihood", "Accuracy", "Score", "R2"]):
            return "↑"

        return "↓"

    def _display_metric_title(self, metric: str) -> str:
        return f"{self._display_metric_name(metric)} ({self._metric_target_label(metric)})"

    def _display_metric_name(self, metric: str) -> str:

        """Pretty metric names for tables and plots."""
        rename_map = {
            "Flexibility Distance": "Reconstruction Flexibility Distance (FD)",
            "Avg MinMax Reconstruction Score": "MinMax Score (Reconstruction)",
            "Avg MinMax Synthetic Score": "MinMax Score (Synthetic)",
            "Avg Rank Reconstruction Score": "Avg Rank (Reconstruction)",
            "Avg Rank Synthetic Score": "Avg Rank (Synthetic)",
            "Win Count Reconstruction": "Win Count (Reconstruction)",
            "Win Count Synthetic": "Win Count (Synthetic)",
            "TSTR (Train on Synthetic, Test on Real)": "Train on Synthetic, Test on Real (TSTR)",
            "Aggregated Profiles (Flexibility Distance)": "FD on Real & Synth Aggregated Profiles",
            "Set-MMD (Permutation Invariant)": "Maximum Mean Discrepancy (MMD)",
            "CF_error_N_max_independent": "CF Error at N_max (Independent)",
            "CF_area_independent": "CF Area Error (Independent)",
            "CF_error_N_max_shared_cond": "CF Error at N_max (Shared Cond.)",
            "CF_area_shared_cond": "CF Area Error (Shared Cond.)",
        }
        return rename_map.get(metric, metric)
    
    def _include_metric(self, metric: str) -> bool:
        excluded_metrics = {
            "Classical Statistics",
        }
        return metric not in excluded_metrics

    def _model_color(self, model_name: str, index: int, plot_style):
        name = model_name.lower()

        if "avg baseline" in name or "mean baseline" in name or "averagebaseline" in name:
            return "0.55"   # medium grey

        if "quantile baseline" in name or "quantilebaseline" in name:
            return "0.25"   # dark grey

        return plot_style.get_color(index)
    

    def _canonical_metric_key(self, metric: str) -> str:
        """Normalize metric names for robust matching."""
        return (
            str(metric)
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )


    def _metric_goal(self, metric: str) -> str:
        """
        Return one of:
            'lower'     -> lower is better
            'higher'    -> higher is better
            'target_0'  -> closer to 0 is better
            'target_1'  -> closer to 1 is better
        """
        key = self._canonical_metric_key(metric)

        goal_map = {
            # Reconstruction metrics
            "flexibilitydistance": "lower",
            "reconstructionflexibilitydistancefd": "lower",
            "reconstructionloglikelihood": "higher",
            "reconstructionloglikelihood": "higher",
            "reconstructionrmse": "lower",

            # Synthetic metrics
            "aggregatedprofilesflexibilitydistance": "lower",
            "perconditionevaluation": "lower",
            "acfdistributionmatchks": "lower",
            "tstrtrainonsynthetictestonreal": "target_1",
            "trainonsynthetictestonrealtstr": "target_1",
            "tstr": "target_1",
            "maximummeandiscrepancymmd": "target_0",
            "setmmdpermutationinvariant": "target_0",
            "mmd": "target_0",
            "hourlyconsumptionbymonth": "lower",
            "hourlyconsumptionbyday": "lower",
            "hourlyprofilesbymonthanddayofweek": "lower",
            "peakmagnitude": "lower",
            "peaktiming": "lower",

            # Cross-phase metrics (ADR-0018)
            "crossphasecorrelationerror": "lower",
            "crosscorrelationdistributionks": "lower",
            "slicedwassersteindistance": "lower",
            "classifiertwosampletestauc": "lower",

            # Single-channel C2ST analysis
            "c2staucaggregate": "lower",
            "c2staucphase1low": "lower",
            "c2staucphase2mid": "lower",
            "c2staucphase3high": "lower",
            "c2staucsinglechannel": "lower",

            # Coincidence factor metrics
            "cferrornmaxindependent": "lower",
            "cfareaindependent": "lower",
            "cferrornmaxsharedcond": "lower",
            "cfareasharedcond": "lower",

            # Grid utility metrics (feeder-only, lower RMSE vs real is better)
            "voltagenadir": "lower",
            "transformerutilisation": "lower",
            # [holdout]-suffix variants injected into full_eval
            "voltagenadirholdout": "lower",
            "transformerutilisationholdout": "lower",

            # Summary scores
            "avgminmaxreconstructionscore": "higher",
            "avgminmaxsyntheticscore": "higher",
            "avgrankреconstructionscore": "lower",
            "avgrankreconstructionscore": "lower",
            "avgranksyntheticscore": "lower",
            "wincountreconstruction": "higher",
            "wincountsynthetic": "higher",
        }

        return goal_map.get(key, "lower")


    def _metric_group(self, metric: str) -> str | None:
        """
        Return:
            'reconstruction'
            'synthetic'
            None -> excluded from AvgMinMax scores
        """
        key = self._canonical_metric_key(metric)

        reconstruction_metrics = {
            "flexibilitydistance",
            "reconstructionflexibilitydistancefd",
            "reconstructionloglikelihood",
            "reconstructionrmse",
        }

        synthetic_metrics = {
            "aggregatedprofilesflexibilitydistance",
            "tstrtrainonsynthetictestonreal",
            "trainonsynthetictestonrealtstr",
            "tstr",
            "maximummeandiscrepancymmd",
            "setmmdpermutationinvariant",
            "mmd",
            "acfdistributionmatchks",
            # "hourlyconsumptionbymonth",
            # "hourlyconsumptionbyday",
            "hourlyprofilesbymonthanddayofweek",
            "peakmagnitude",
            "peaktiming",
            "cferrornmaxindependent",
            "cfareaindependent",
            "cferrornmaxsharedcond",
            "cfareasharedcond",
        }

        summary_metrics = {
            "avgminmaxreconstructionscore", "avgminmaxsyntheticscore",
            "avgrankreconstructionscore", "avgranksyntheticscore",
            "wincountreconstruction", "wincountsynthetic",
        }

        # Grid utility metrics are feeder-only by design and must never contribute
        # to Avg MinMax Synthetic Score, Avg Rank Synthetic Score, or any other composite.
        grid_utility_metrics = {
            "voltagenadir",
            "transformerutilisation",
            "voltagenadirholdout",        # full_eval [holdout]-suffix variant
            "transformerutilisationholdout",
        }

        if key in summary_metrics:
            return None  # never feed summary metrics back into themselves

        if key in grid_utility_metrics:
            return None

        if key in reconstruction_metrics:
            return "reconstruction"

        if key in synthetic_metrics:
            return "synthetic"

        return None


    def _metric_target_label(self, metric: str) -> str:
        goal = self._metric_goal(metric)

        if goal == "higher":
            return "↑"
        if goal == "lower":
            return "↓"
        if goal == "target_0":
            return "target=0"
        if goal == "target_1":
            return "target=1"

        return "↓"


    def _display_metric_title(self, metric: str) -> str:
        return f"{self._display_metric_name(metric)} ({self._metric_target_label(metric)})"

    def add_model(self, model_name: str, model_dir: str, trainable_params: int | None = None):
        """Add a model's evaluation results to the comparator, handling both phased and unphased output."""
        self.models_data[model_name] = {}
        self._model_eval_dirs[model_name] = model_dir
        self._param_counts[model_name] = trainable_params
        
        # 1. Check for phased results in subdirectories.
        #    Entries may be flat scalars (n_draws=1) or {mean,std,values} dicts (n_draws>1,
        #    Intra-eval variance).  Extract means into [phase]; stds into [phase_meta].
        for phase in ['aggregated_phase', 'ranked_phase_1', 'ranked_phase_2', 'ranked_phase_3']:
            eval_json_path = os.path.join(model_dir, f"{phase}_eval", "eval_results.json")

            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, 'r') as f:
                        data = json.load(f)
                    phase_eval: dict[str, float] = {}
                    phase_meta: dict[str, dict] = {}
                    for key, val in data.items():
                        if isinstance(val, dict) and "mean" in val:
                            phase_eval[key] = float(val["mean"])
                            phase_meta[key] = {
                                "std": float(val["std"]) if val.get("std") is not None else None,
                                "values": val.get("values"),
                            }
                        else:
                            try:
                                phase_eval[key] = float(val)
                            except (TypeError, ValueError):
                                phase_eval[key] = val
                    self.models_data[model_name][phase] = phase_eval
                    if phase_meta:
                        self.models_data[model_name][f"{phase}_meta"] = phase_meta
                except Exception as e:
                    print(f"Error loading {eval_json_path}: {e}")
        
        # 1b. Fallback: load from phased_eval_summary.json when per-phase eval_results.json files
        #     were not found (e.g. only the consolidated summary was written).
        def _extract_phase_dict(raw: dict) -> tuple[dict, dict]:
            phase_eval: dict[str, float] = {}
            phase_meta: dict[str, dict] = {}
            for key, val in raw.items():
                if isinstance(val, dict) and "mean" in val:
                    phase_eval[key] = float(val["mean"])
                    phase_meta[key] = {
                        "std": float(val["std"]) if val.get("std") is not None else None,
                        "values": val.get("values"),
                    }
                else:
                    try:
                        phase_eval[key] = float(val)
                    except (TypeError, ValueError):
                        phase_eval[key] = val
            return phase_eval, phase_meta

        phased_summary_path = os.path.join(model_dir, "phased_eval_summary.json")
        if os.path.exists(phased_summary_path):
            try:
                with open(phased_summary_path, 'r') as f:
                    summary = json.load(f)
                _phase_map = {
                    'aggregated_phase': summary.get('aggregated_phase', {}),
                    'ranked_phase_1': summary.get('ranked_phases', {}).get('phase_1', {}),
                    'ranked_phase_2': summary.get('ranked_phases', {}).get('phase_2', {}),
                    'ranked_phase_3': summary.get('ranked_phases', {}).get('phase_3', {}),
                    'cross_phase': summary.get('cross_phase', {}),
                    'c2st': summary.get('c2st', {}),
                }
                for phase_key, raw in _phase_map.items():
                    if raw and phase_key not in self.models_data[model_name]:
                        pe, pm = _extract_phase_dict(raw)
                        self.models_data[model_name][phase_key] = pe
                        if pm:
                            self.models_data[model_name][f"{phase_key}_meta"] = pm
            except Exception as e:
                print(f"Error loading {phased_summary_path}: {e}")

        # 2. Check for unphased results at root (either EvalFramework or PhasedEvaluationSuite summary)
        root_eval_path = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(root_eval_path):
            try:
                with open(root_eval_path, 'r') as f:
                    data = json.load(f)
                
                # If it's a simple flat dict (from EvalFramework), add it as 'full_eval'.
                # Entries may be flat scalars (K=1 run) or {mean,std,values} dicts (K>1 run,
                # Intra-eval variance).  Extract means into full_eval; stds into full_eval_meta.
                if not any(k in data for k in ['aggregated_phase', 'ranked_phases']):
                    full_eval = {}
                    full_eval_meta = {}
                    for key, val in data.items():
                        if isinstance(val, dict) and "mean" in val:
                            full_eval[key] = float(val["mean"])
                            full_eval_meta[key] = {
                                "std": float(val["std"]) if val.get("std") is not None else None,
                                "values": val.get("values"),
                            }
                        else:
                            try:
                                full_eval[key] = float(val)
                            except (TypeError, ValueError):
                                full_eval[key] = val
                    self.models_data[model_name]['full_eval'] = full_eval
                    if full_eval_meta:
                        self.models_data[model_name]['full_eval_meta'] = full_eval_meta
            except Exception as e:
                print(f"Error loading {root_eval_path}: {e}")

        # 2b. Standalone C2ST results (non-phased models, or as fallback for phased).
        #     For phased models 'c2st' is already loaded from phased_eval_summary.json above.
        c2st_path = os.path.join(model_dir, 'c2st_results.json')
        if os.path.exists(c2st_path) and 'c2st' not in self.models_data.get(model_name, {}):
            try:
                with open(c2st_path, 'r') as f:
                    c2st_raw = json.load(f)
                pe, pm = _extract_phase_dict(c2st_raw)
                if pe:
                    self.models_data[model_name]['c2st'] = pe
                if pm:
                    self.models_data[model_name]['c2st_meta'] = pm
            except Exception as e:
                print(f"Error loading {c2st_path}: {e}")

        # 3. Feeder holdout results.
        #    3a. Summary JSON → holdout_eval (mean scalars) + holdout_summary_meta (std, per_feeder).
        #    3b. Legacy flat JSON → holdout_eval only (backward compat, no meta).
        #    3c. Inject means into full_eval with [holdout] suffix (existing behaviour).
        summary_path = os.path.join(model_dir, "eval_results_feeder_holdout_summary.json")
        flat_path = os.path.join(model_dir, "eval_results_feeder_holdout.json")

        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                holdout_eval: dict[str, float] = {}
                holdout_meta: dict[str, dict] = {}
                for metric, entry in summary.items():
                    mean_val = float(entry["mean"])
                    holdout_eval[metric] = mean_val
                    holdout_meta[metric] = {
                        "std": float(entry["std"]) if entry.get("std") is not None else None,
                        "q95": float(entry["q95"]) if entry.get("q95") is not None else None,
                        "per_feeder": entry.get("per_feeder", {}),
                    }
                self.models_data[model_name]["holdout_eval"] = holdout_eval
                self.models_data[model_name]["holdout_summary_meta"] = holdout_meta
                self.models_data[model_name]["holdout_eval_meta"] = holdout_meta
            except Exception as e:
                print(f"Error loading {summary_path}: {e}")
        elif os.path.exists(flat_path):
            try:
                with open(flat_path) as f:
                    flat = json.load(f)
                holdout_eval = {}
                for k, v in flat.items():
                    if isinstance(v, dict) and "mean" in v:
                        holdout_eval[k] = float(v["mean"])
                    else:
                        holdout_eval[k] = float(v)  # type: ignore[arg-type]
                self.models_data[model_name]["holdout_eval"] = holdout_eval
            except Exception as e:
                print(f"Error loading {flat_path}: {e}")

        if not self.models_data[model_name]:
            print(f"No valid evaluation results found in {model_dir} for model {model_name}")

    def _sort_models_by_synthetic_score(self, phase: str) -> list[str]:
        """Return model names sorted by Avg Rank Synthetic Score ascending (rank 1 = best first)."""
        score_key = "Avg Rank Synthetic Score"

        def _score(model_name: str) -> float:
            # Missing entries sort to the end (worst)
            return float(self.models_data.get(model_name, {}).get(phase, {}).get(score_key, float("inf")))

        return sorted(self.models_data.keys(), key=_score)

    def _generate_mia_plots(self, models: list[str]) -> None:
        """Generate MIA ROC curve and bar chart for all models that have eval_mia_curve.json."""
        curves: dict[str, dict] = {}
        for name in models:
            eval_dir = self._model_eval_dirs.get(name)
            if not eval_dir:
                continue
            sidecar = os.path.join(eval_dir, "eval_mia_curve.json")
            if os.path.exists(sidecar):
                try:
                    with open(sidecar) as f:
                        curves[name] = json.load(f)
                except Exception as e:
                    print(f"  [MIA] Could not load sidecar for {name}: {e}")
                    import traceback; traceback.print_exc()

        if not curves:
            print("  [MIA] No curve sidecar (eval_mia_curve.json) found for any model; skipping MIA plots.")
            print(f"  [MIA] Searched in: { {n: self._model_eval_dirs.get(n) for n in models} }")
            return

        mia_dir = os.path.join(self.output_dir, "mia")
        os.makedirs(mia_dir, exist_ok=True)
        print(f"  [MIA] Plotting {len(curves)} model(s) → {mia_dir}")
        plot_style = PlotStyle()

        # --- ROC curve ---
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", lw=1, label="Random (AUC = 0.50)")
            for i, name in enumerate(models):
                if name not in curves:
                    continue
                c = curves[name]
                fpr = c.get("fpr", [])
                tpr = c.get("tpr", [])
                auc = c.get("auc_roc", float("nan"))
                color = self._model_color(name, i, plot_style)
                ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={auc:.3f})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Membership Inference Attack — ROC Curves")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            fig.tight_layout()
            roc_path = os.path.join(mia_dir, "mia_roc.pdf")
            fig.savefig(roc_path, dpi=300, bbox_inches="tight")
            print(f"  [MIA] ROC curve saved to {roc_path}")
        except Exception as e:
            print(f"  [MIA] Warning: could not save ROC curve: {e}")
            import traceback; traceback.print_exc()
        finally:
            plt.close("all")

        # --- Bar chart: AUC-ROC / Precision / TPR@FPR=0.1 ---
        try:
            scalar_specs = [
                ("auc_roc",            "AUC-ROC"),
                ("precision_balanced", "Precision (balanced)"),
                ("tpr_at_fpr_0.1",     "TPR @ FPR=0.1"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                "Membership Inference Attack — Privacy Metrics  (↑ = more leakage, 0.5 = random guess)",
                fontsize=13, fontweight="bold",
            )
            for ax, (key, title) in zip(axes, scalar_specs):
                vals, labels, colors = [], [], []
                for i, name in enumerate(models):
                    if name not in curves:
                        continue
                    val = curves[name].get(key)
                    if val is None:
                        continue
                    vals.append(float(val))
                    labels.append(name)
                    colors.append(self._model_color(name, i, plot_style))
                if not vals:
                    ax.set_visible(False)
                    continue
                x = np.arange(len(vals))
                ax.bar(x, vals, color=colors)
                ax.axhline(0.5, color="black", linestyle="--", lw=1, label="Random (0.5)")
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                ax.legend(fontsize=8)
            fig.tight_layout()
            bar_path = os.path.join(mia_dir, "mia_bar.pdf")
            fig.savefig(bar_path, dpi=300, bbox_inches="tight")
            print(f"  [MIA] Bar chart saved to {bar_path}")
        except Exception as e:
            print(f"  [MIA] Warning: could not save bar chart: {e}")
            import traceback; traceback.print_exc()
        finally:
            plt.close("all")

    def _print_param_table(self) -> None:
        """Print and save a one-row-per-model trainable parameter count table."""
        rows = [
            [name, f"{self._param_counts[name]:,}" if self._param_counts.get(name) is not None else "N/A"]
            for name in self.models_data
        ]
        table_str = tabulate(rows, headers=["Model", "Trainable Params"], tablefmt="grid")
        print("\n=== Parameter Budget ===")
        print(table_str)

        table_path = os.path.join(self.output_dir, "parameter_count.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"Parameter count table saved to {table_path}")

    def _generate_ev_hp_plots(self) -> None:
        """Generate EV/HP breakdown comparison bar charts from ev_hp_breakdown.json sidecars.

        Produces ev_hp_breakdown/ev_hp_ev_fd.pdf and ev_hp_breakdown/ev_hp_hp_fd.pdf.
        When ev_hp_breakdown_random_profiles.json sidecars are also present, produces a
        separate set of figures with a _random_profiles suffix.
        Skipped entirely when no model has a non-empty sidecar or all FD values are null.
        """
        ev_hp_dir = os.path.join(self.output_dir, "ev_hp_breakdown")
        os.makedirs(ev_hp_dir, exist_ok=True)

        for sidecar_name, out_suffix in [
            ("ev_hp_breakdown.json", ""),
            ("ev_hp_breakdown_random_profiles.json", "_random_profiles"),
        ]:
            self._generate_ev_hp_plots_for_sidecar(sidecar_name, out_suffix, ev_hp_dir)

    def _generate_ev_hp_plots_for_sidecar(
        self,
        sidecar_name: str,
        out_suffix: str,
        ev_hp_dir: str,
    ) -> None:
        """Render bar charts for one ev_hp_breakdown sidecar variant."""
        breakdown: dict[str, dict] = {}
        for name in self.models_data:
            path = os.path.join(self._model_eval_dirs.get(name, ""), sidecar_name)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data:
                        breakdown[name] = data
                except Exception as e:
                    print(f"  [ev_hp] Could not load sidecar '{sidecar_name}' for {name}: {e}")

        if not breakdown:
            return

        all_groups: set[str] = set()
        for bd in breakdown.values():
            for grp, info in bd.items():
                if info.get("fd") is not None:
                    all_groups.add(grp)
        if not all_groups:
            print(f"  [ev_hp] All FD values are null in '{sidecar_name}'; skipping plots.")
            return

        models_order = [n for n in self.models_data if n in breakdown]
        plot_style = PlotStyle()

        for pair_name, pair_groups in [("EV", ["EV", "No-EV"]), ("HP", ["HP", "No-HP"])]:
            valid_groups = [g for g in pair_groups if g in all_groups]
            if not valid_groups:
                continue

            n_groups = len(valid_groups)
            fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), squeeze=False)
            axes_flat = axes.flatten()
            title_suffix = " (Random User Profiles)" if out_suffix else ""
            fig.suptitle(
                f"Aggregated Profiles FD — {pair_name} Condition Breakdown{title_suffix}  (↓)",
                fontsize=14, fontweight="bold", y=1.02,
            )

            for col, group_name in enumerate(valid_groups):
                ax = axes_flat[col]
                vals, labels, colors, n_reals = [], [], [], []
                for i, name in enumerate(models_order):
                    info = breakdown[name].get(group_name, {})
                    fd = info.get("fd")
                    if fd is None:
                        continue
                    vals.append(float(fd))
                    labels.append(name)
                    colors.append(self._model_color(name, i, plot_style))
                    n_reals.append(info.get("n_real", "?"))

                if not vals:
                    ax.set_visible(False)
                    continue

                best_val = min(vals)
                x = np.arange(len(vals))
                for xi, (val, color) in enumerate(zip(vals, colors)):
                    is_best = abs(val - best_val) < abs(best_val) * 1e-9 + 1e-12
                    ec = "black" if is_best else "none"
                    lw = 1.5 if is_best else 0
                    hatch = "///" if is_best else ""
                    ax.bar([xi], [val], color=color, edgecolor=ec, linewidth=lw, hatch=hatch)
                    if is_best:
                        ax.annotate("★", xy=(xi, val), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=12, color="black", alpha=0.8)

                ax.set_title(group_name, fontsize=11, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
                ax.grid(axis="y", linestyle="--", alpha=0.6)

                for xi, (n, val) in enumerate(zip(n_reals, vals)):
                    ax.annotate(f"n={n}", xy=(xi, 0), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=7, color="0.5")

            legend_handles = [
                mpatches.Patch(color=self._model_color(name, i, plot_style), label=name)
                for i, name in enumerate(models_order)
            ]
            legend_handles.append(
                mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="★ Best Value")
            )
            fig.legend(handles=legend_handles, loc="lower center",
                       bbox_to_anchor=(0.5, 1.04), ncol=min(3, len(models_order) + 1))

            plt.tight_layout()
            out_path = os.path.join(ev_hp_dir, f"ev_hp_{pair_name.lower()}_fd{out_suffix}.pdf")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [ev_hp] Plot saved to {out_path}")

    def compare(self, mia_models: list[str] | None = None, sort_models: bool = True):
        """Generate comparison tables and plots for each phase.

        mia_models: if given, only these model names appear in MIA plots.
                    Defaults to all models.
        sort_models: if True (default), models are sorted by Avg Rank Synthetic Score.
                     If False, models appear in the order they were added via add_model().
        """
        if not self.models_data:
            print("No models added. Exiting.")
            return

        self._print_param_table()

        last_sorted_models: list[str] | None = None
        for phase in self.phases:
            # Only show a phase in the comparison if ALL loaded models have data for it.
            # This prevents partial 'N/A' columns from appearing when one model ran in
            # unphased mode and another in phased mode.
            models_with_phase = [m for m in self.models_data if phase in self.models_data[m]]
            if models_with_phase and len(models_with_phase) == len(self.models_data):
                print(f"\n--- Comparing models for phase: {phase} ---")
                self._compute_averages(phase)
                sorted_models = self._sort_models_by_synthetic_score(phase) if sort_models else list(self.models_data.keys())
                last_sorted_models = sorted_models
                self._generate_table(phase, sorted_models)
                self._generate_plots(phase, sorted_models)
                if phase == 'cross_phase':
                    try:
                        self._generate_cross_phase_radar(sorted_models)
                    except Exception as _rdr_e:
                        print(f"  [compare] cross-phase radar failed: {_rdr_e}")
            elif models_with_phase:
                missing = [m for m in self.models_data if phase not in self.models_data[m]]
                print(f"\n[Skipping phase '{phase}': missing for models {missing}]")

        try:
            self._compare_holdout(sorted_models=last_sorted_models)
        except Exception as _hoe:
            print(f"\n[compare] WARNING: _compare_holdout raised an exception; holdout plots may be incomplete.")
            import traceback as _ho_tb; _ho_tb.print_exc()
        mia_candidates = list(self.models_data.keys()) if mia_models is None else [
            m for m in mia_models if m in self.models_data
        ]
        self._generate_mia_plots(mia_candidates)

    def _compute_holdout_averages(self, models_with_holdout: list[str]) -> None:
        """Compute MinMax/Rank/WinCount aggregate scores for holdout_eval metrics.

        Mirrors _compute_averages but operates on holdout_eval instead of a phase dict,
        and only considers models that actually have holdout data.
        """
        phase = 'holdout_eval'
        all_metrics: set[str] = set()
        for m in models_with_holdout:
            for metric, val in self.models_data[m][phase].items():
                if isinstance(val, (int, float)) and self._include_metric(metric):
                    all_metrics.add(metric)

        scalar_metrics = list(all_metrics)

        def _comparable(v: float, goal: str) -> float:
            if goal == "higher":   return -v
            if goal == "target_1": return abs(v - 1.0)
            if goal == "target_0": return abs(v)
            return v

        minmax_scores: dict[str, dict] = {m: {} for m in models_with_holdout}
        rank_scores:   dict[str, dict] = {m: {} for m in models_with_holdout}
        win_counts:    dict[str, dict] = {m: {} for m in models_with_holdout}

        for metric in scalar_metrics:
            goal = self._metric_goal(metric)
            valid = {
                m: float(self.models_data[m][phase][metric])
                for m in models_with_holdout
                if metric in self.models_data[m][phase]
                and isinstance(self.models_data[m][phase][metric], (int, float))
            }
            if not valid:
                continue

            mapped = {m: _comparable(v, goal) for m, v in valid.items()}

            vmin, vmax = min(mapped.values()), max(mapped.values())
            for m, c in mapped.items():
                minmax_scores[m][metric] = 1.0 if vmax == vmin else 1.0 - (c - vmin) / (vmax - vmin)

            sorted_ms = sorted(mapped, key=mapped.__getitem__)
            n = len(sorted_ms)
            i = 0
            while i < n:
                j = i + 1
                while j < n and mapped[sorted_ms[j]] == mapped[sorted_ms[i]]:
                    j += 1
                avg_rank = (i + j - 1) / 2
                for k in range(i, j):
                    rank_scores[sorted_ms[k]][metric] = avg_rank + 1.0
                i = j

            best = min(mapped.values())
            winners = [m for m, c in mapped.items() if c == best]
            share = 1.0 / len(winners)
            for m in valid:
                win_counts[m][metric] = share if m in winners else 0.0

        recon_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synth_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]

        for m in models_with_holdout:
            r_mm   = [minmax_scores[m][met] for met in recon_metrics if met in minmax_scores[m]]
            s_mm   = [minmax_scores[m][met] for met in synth_metrics  if met in minmax_scores[m]]
            r_rank = [rank_scores[m][met]   for met in recon_metrics  if met in rank_scores[m]]
            s_rank = [rank_scores[m][met]   for met in synth_metrics  if met in rank_scores[m]]
            r_wins = [win_counts[m][met]    for met in recon_metrics  if met in win_counts[m]]
            s_wins = [win_counts[m][met]    for met in synth_metrics  if met in win_counts[m]]

            if r_mm:   self.models_data[m][phase]['Avg MinMax Reconstruction Score'] = float(np.mean(r_mm))
            if s_mm:   self.models_data[m][phase]['Avg MinMax Synthetic Score']      = float(np.mean(s_mm))
            if r_rank: self.models_data[m][phase]['Avg Rank Reconstruction Score']   = float(np.mean(r_rank))
            if s_rank: self.models_data[m][phase]['Avg Rank Synthetic Score']        = float(np.mean(s_rank))
            if r_wins: self.models_data[m][phase]['Win Count Reconstruction']        = float(sum(r_wins))
            if s_wins: self.models_data[m][phase]['Win Count Synthetic']             = float(sum(s_wins))

    def _compare_holdout(self, sorted_models: list[str] | None = None):
        """Generate a separate comparison table and plots for feeder holdout results.

        Models without holdout results are shown as N/A — they are not excluded.
        Aggregate scores (MinMax/Rank/WinCount) are computed over models that have holdout data.
        If sorted_models is provided (the order from the main comparison), that order is
        preserved; otherwise models are sorted by Avg Rank Synthetic Score.
        """
        models_with_holdout = [m for m in self.models_data if 'holdout_eval' in self.models_data[m]]
        if not models_with_holdout:
            print("\n[No holdout eval results found for any model; skipping holdout comparison.]")
            return

        self._compute_holdout_averages(models_with_holdout)

        holdout_set = set(models_with_holdout)
        if sorted_models is not None:
            sorted_holdout = [m for m in sorted_models if m in holdout_set]
            # models with holdout data not in sorted_models (edge case)
            sorted_holdout += [m for m in models_with_holdout if m not in sorted_models]
            missing = [m for m in sorted_models if m not in holdout_set]
            missing += [m for m in self.models_data if m not in sorted_models and m not in holdout_set]
        else:
            sorted_holdout = sorted(
                models_with_holdout,
                key=lambda m: float(self.models_data[m].get('holdout_eval', {}).get('Avg Rank Synthetic Score', float('inf')))
            )
            missing = [m for m in self.models_data if 'holdout_eval' not in self.models_data[m]]
        all_models = sorted_holdout + missing

        if missing:
            print(f"\n--- Holdout comparison (N/A for: {missing}) ---")
        else:
            print(f"\n--- Holdout comparison ({len(models_with_holdout)} models) ---")

        all_metrics: set[str] = set()
        for m in models_with_holdout:
            for metric, val in self.models_data[m]['holdout_eval'].items():
                if isinstance(val, (int, float)) and self._include_metric(metric):
                    all_metrics.add(metric)

        if not all_metrics:
            print("  [holdout] No scalar metrics to compare.")
            return

        scalar_metrics = sorted(all_metrics, key=lambda x: (0 if "Reconstruction" in x or "Flexibility" in x else 1, x))

        _SUMMARY = {'Avg MinMax Reconstruction Score', 'Avg MinMax Synthetic Score',
                    'Avg Rank Reconstruction Score',   'Avg Rank Synthetic Score',
                    'Win Count Reconstruction',        'Win Count Synthetic'}
        non_summary = [m for m in scalar_metrics if m not in _SUMMARY]
        summaries   = [m for m in scalar_metrics if m in _SUMMARY]
        scalar_metrics = non_summary + summaries

        # --- Table ---
        headers = ["Metric"] + all_models
        table_data = []
        for metric in scalar_metrics:
            goal = self._metric_goal(metric)
            model_vals = {
                m: float(self.models_data[m]['holdout_eval'][metric])
                for m in models_with_holdout
                if metric in self.models_data[m]['holdout_eval']
                and isinstance(self.models_data[m]['holdout_eval'][metric], (int, float))
            }
            best_models: set = set()
            if model_vals:
                if goal == "target_1":
                    best_raw = min(abs(v - 1.0) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v - 1.0) - best_raw) <= 1e-9}
                elif goal == "target_0":
                    best_raw = min(abs(v) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v) - best_raw) <= 1e-9}
                elif goal == "higher":
                    best_raw = max(model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= abs(best_raw) * 1e-9 + 1e-12}
                else:
                    best_raw = min(model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= abs(best_raw) * 1e-9 + 1e-12}

            row = [self._display_metric_name(metric)]
            for m in all_models:
                if 'holdout_eval' in self.models_data[m] and metric in self.models_data[m]['holdout_eval']:
                    val = self.models_data[m]['holdout_eval'][metric]
                    if isinstance(val, (int, float)):
                        meta = self.models_data[m].get('holdout_summary_meta', {}).get(metric, {})
                        std = meta.get('std') if meta else None
                        mean_str = f"{val:.4f}"
                        fmt = f"{mean_str} ± {std:.4f}" if std is not None else mean_str
                        row.append(f"★ {fmt}" if m in best_models else fmt)
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nHoldout Comparison Table:")
        print(table_str)

        holdout_dir = os.path.join(self.output_dir, "holdout_eval")
        os.makedirs(holdout_dir, exist_ok=True)
        table_path = os.path.join(holdout_dir, "comparison_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"Holdout table saved to {table_path}")

        # --- Bar plots (only models with holdout data) ---
        plot_style = PlotStyle()
        _RECON_SUMMARY = [
            "Avg MinMax Reconstruction Score",
            "Avg Rank Reconstruction Score",
        ]
        _SYNTH_SUMMARY = [
            "Avg MinMax Synthetic Score",
            "Avg Rank Synthetic Score",
        ]
        reconstruction_metrics = sorted(m for m in scalar_metrics if self._metric_group(m) == "reconstruction") \
            + [m for m in _RECON_SUMMARY if m in all_metrics]
        synthetic_metrics = sorted(m for m in scalar_metrics if self._metric_group(m) == "synthetic") \
            + [m for m in _SYNTH_SUMMARY if m in all_metrics]
        # For the holdout plot keep only CF_error_N_max_shared_cond from CF metrics and rename it.
        _CF_PLOT_EXCLUDED = {"CF_error_N_max_independent", "CF_area_independent", "CF_area_shared_cond"}
        _HOLDOUT_NAME_OVERRIDES = {"CF_error_N_max_shared_cond": "CF Error"}
        synthetic_metrics_plot = [m for m in synthetic_metrics if m not in _CF_PLOT_EXCLUDED]
        for metric_group, group_name in [(reconstruction_metrics, "Reconstruction"), (synthetic_metrics_plot, "Synthetic")]:
            if metric_group:
                self._generate_metric_group_plot("holdout_eval", metric_group, group_name, models_with_holdout, plot_style, name_overrides=_HOLDOUT_NAME_OVERRIDES)

        # --- Distribution plots (box + violin) per metric group ---
        # Only when at least one model has ≥ 2 feeders in its summary meta.
        for metric_group, group_name in [(reconstruction_metrics, "Reconstruction"), (synthetic_metrics, "Synthetic")]:
            if metric_group:
                self._generate_holdout_distribution_plot(metric_group, group_name, models_with_holdout, holdout_dir)

    def _generate_holdout_distribution_plot(
        self,
        metric_group: list[str],
        group_name: str,
        models_with_holdout: list[str],
        holdout_dir: str,
    ) -> None:
        """Save distribution_<group_name>.png with box + violin subplots per metric.

        Each metric gets two adjacent subplots (box left, violin right).
        Skipped entirely when no model has ≥ 2 feeders.
        """
        per_feeder_vals: dict[str, dict[str, list[float]]] = {}
        for m in models_with_holdout:
            meta = self.models_data[m].get("holdout_summary_meta", {})
            if not meta:
                continue
            model_vals: dict[str, list[float]] = {}
            for metric in metric_group:
                if metric in meta and meta[metric].get("per_feeder"):
                    model_vals[metric] = list(meta[metric]["per_feeder"].values())
            if model_vals:
                per_feeder_vals[m] = model_vals

        if not per_feeder_vals:
            return

        any_multi = any(
            len(vals) >= 2
            for model_vals in per_feeder_vals.values()
            for vals in model_vals.values()
        )
        if not any_multi:
            return

        plot_style = PlotStyle()
        n_metrics = len(metric_group)
        n_cols = min(n_metrics, 4)
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        title_map = {
            "Reconstruction": "Reconstruction — Inter-Feeder Distribution",
            "Synthetic": "Synthetic — Inter-Feeder Distribution",
        }
        fig.suptitle(title_map.get(group_name, group_name), fontsize=14, fontweight="bold", y=0.995)

        subplot_idx = 0
        for metric in metric_group:
            if subplot_idx >= len(axes_flat):
                break
            ax = axes_flat[subplot_idx]
            subplot_idx += 1

            data_by_model = []
            labels = []
            colors = []
            for i, m in enumerate(models_with_holdout):
                vals = per_feeder_vals.get(m, {}).get(metric)
                if vals:
                    data_by_model.append(vals)
                    labels.append(m)
                    colors.append(self._model_color(m, i, plot_style))

            if not data_by_model:
                ax.set_visible(False)
                continue

            positions = list(range(1, len(data_by_model) + 1))
            bp = ax.boxplot(data_by_model, positions=positions, patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(self._display_metric_name(metric), fontsize=9, fontweight="bold")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

        for i in range(subplot_idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        os.makedirs(holdout_dir, exist_ok=True)
        out_path = os.path.join(holdout_dir, f"distribution_{group_name}.pdf")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Distribution plot saved to {out_path}")

    def _compute_averages(self, phase: str):
        """Compute three aggregate scores (min-max, rank, win count) for reconstruction and synthetic metrics."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)

        scalar_metrics = list(all_metrics)
        models = list(self.models_data.keys())

        preferred_last = ["Avg Baseline", "Mean Baseline", "Quantile Baseline"]
        non_last = [m for m in models if m not in preferred_last]
        last = [m for m in preferred_last if m in models]
        models = non_last + last

        def _comparable(v: float, goal: str) -> float:
            """Map to a value where lower is always better."""
            if goal == "higher":    return -v
            if goal == "target_1":  return abs(v - 1.0)
            if goal == "target_0":  return abs(v)
            return v  # lower

        minmax_scores: dict[str, dict] = {m: {} for m in models}
        rank_scores:   dict[str, dict] = {m: {} for m in models}
        win_counts:    dict[str, dict] = {m: {} for m in models}

        for metric in scalar_metrics:
            goal = self._metric_goal(metric)

            valid = {
                m: float(v)
                for m in models
                for v in [self.models_data[m].get(phase, {}).get(metric)]
                if isinstance(v, (int, float))
            }
            if not valid:
                continue

            mapped = {m: _comparable(v, goal) for m, v in valid.items()}

            # Min-max: 1.0 = best, 0.0 = worst
            vmin, vmax = min(mapped.values()), max(mapped.values())
            for m, c in mapped.items():
                minmax_scores[m][metric] = 1.0 if vmax == vmin else 1.0 - (c - vmin) / (vmax - vmin)

            # Rank-based: 1 = first place (best), ties share the midrank position
            sorted_ms = sorted(mapped, key=mapped.__getitem__)
            n = len(sorted_ms)
            i = 0
            while i < n:
                j = i + 1
                while j < n and mapped[sorted_ms[j]] == mapped[sorted_ms[i]]:
                    j += 1
                avg_rank = (i + j - 1) / 2  # 0-indexed midrank; +1 gives 1-indexed position
                for k in range(i, j):
                    rank_scores[sorted_ms[k]][metric] = avg_rank + 1.0  # 1-indexed (lower = better)
                i = j

            # Win count: 1.0 for sole winner, split equally on ties
            best = min(mapped.values())
            winners = [m for m, c in mapped.items() if c == best]
            share = 1.0 / len(winners)
            for m in valid:
                win_counts[m][metric] = share if m in winners else 0.0

        recon_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synth_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]

        for m in models:
            if phase not in self.models_data[m]:
                continue

            r_mm   = [minmax_scores[m][met] for met in recon_metrics if met in minmax_scores[m]]
            s_mm   = [minmax_scores[m][met] for met in synth_metrics  if met in minmax_scores[m]]
            r_rank = [rank_scores[m][met]   for met in recon_metrics  if met in rank_scores[m]]
            s_rank = [rank_scores[m][met]   for met in synth_metrics  if met in rank_scores[m]]
            r_wins = [win_counts[m][met]    for met in recon_metrics  if met in win_counts[m]]
            s_wins = [win_counts[m][met]    for met in synth_metrics  if met in win_counts[m]]

            if r_mm:   self.models_data[m][phase]['Avg MinMax Reconstruction Score'] = float(np.mean(r_mm))
            if s_mm:   self.models_data[m][phase]['Avg MinMax Synthetic Score']      = float(np.mean(s_mm))
            if r_rank: self.models_data[m][phase]['Avg Rank Reconstruction Score']   = float(np.mean(r_rank))
            if s_rank: self.models_data[m][phase]['Avg Rank Synthetic Score']         = float(np.mean(s_rank))
            if r_wins: self.models_data[m][phase]['Win Count Reconstruction']         = float(sum(r_wins))
            if s_wins: self.models_data[m][phase]['Win Count Synthetic']              = float(sum(s_wins))

    def _generate_table(self, phase: str, sorted_models: list[str] | None = None):
        """Generate and print a comparison table of scalar metrics for a given phase."""
        models_order = sorted_models if sorted_models is not None else list(self.models_data.keys())
        headers = ["Metric"] + models_order
        
        # Collect all unique metric names
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)
                        
        # scalar_metrics = sorted(list(all_metrics))
        scalar_metrics = sorted(all_metrics, key=lambda m: (0 if "Reconstruction" in m or "Flexibility" in m else 1, m))
        
        # Move all summary metrics to the end
        _SUMMARY = {'Avg MinMax Reconstruction Score', 'Avg MinMax Synthetic Score',
                    'Avg Rank Reconstruction Score',   'Avg Rank Synthetic Score',
                    'Win Count Reconstruction',        'Win Count Synthetic'}
        summaries    = [m for m in scalar_metrics if m in _SUMMARY]
        non_summary  = [m for m in scalar_metrics if m not in _SUMMARY]
        scalar_metrics = non_summary + summaries
                
        table_data = []
        for metric in scalar_metrics:
            goal = self._metric_goal(metric)

            # Find the winning model(s) for this metric — ties share ★
            model_vals = {
                m: float(self.models_data[m][phase][metric])
                for m in models_order
                if phase in self.models_data[m] and metric in self.models_data[m][phase]
                and isinstance(self.models_data[m][phase][metric], (int, float))
            }
            best_models: set = set()
            if model_vals:
                if goal == "target_1":
                    best_raw = min(abs(v - 1.0) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v - 1.0) - best_raw) <= 1e-9}
                elif goal == "target_0":
                    best_raw = min(abs(v) for v in model_vals.values())
                    best_models = {m for m, v in model_vals.items() if abs(abs(v) - best_raw) <= 1e-9}
                elif goal == "higher":
                    best_raw = max(model_vals.values())
                    tol = abs(best_raw) * 1e-9 + 1e-12
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= tol}
                else:  # lower
                    best_raw = min(model_vals.values())
                    tol = abs(best_raw) * 1e-9 + 1e-12
                    best_models = {m for m, v in model_vals.items() if abs(v - best_raw) <= tol}

            row = [self._display_metric_name(metric)]
            for model_name in models_order:
                data = self.models_data[model_name]
                if phase in data and metric in data[phase]:
                    val = data[phase][metric]
                    mean_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    meta = self.models_data[model_name].get(f"{phase}_meta", {}).get(metric, {})
                    std = meta.get("std") if meta else None
                    formatted = f"{mean_str} ± {std:.4f}" if std is not None else mean_str
                    if model_name in best_models:
                        formatted = f"★ {formatted}"
                else:
                    formatted = "N/A"
                row.append(formatted)
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(f"\nModel Comparison Table ({phase}):")
        print(table_str)
        
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        table_path = os.path.join(phase_output_dir, "comparison_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"\nTable saved to {table_path}")

    def _generate_plots(self, phase: str, sorted_models: list[str] | None = None):
        """Generate bar charts for scalar metrics for a given phase, split by reconstruction and synthetic metrics."""
        all_metrics = set()
        for model_name, data in self.models_data.items():
            if phase in data:
                for metric, val in data[phase].items():
                    if isinstance(val, (int, float)) and self._include_metric(metric):
                        all_metrics.add(metric)

        scalar_metrics = list(all_metrics)
        if not scalar_metrics:
            return

        # Separate metrics by group
        reconstruction_metrics = [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        synthetic_metrics = [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]

        # Append summary metrics at the end of each group in a fixed order
        _RECON_SUMMARY = [
            "Avg MinMax Reconstruction Score",
            "Avg Rank Reconstruction Score",
            "Win Count Reconstruction",
        ]
        _SYNTH_SUMMARY = [
            "Avg MinMax Synthetic Score",
            "Avg Rank Synthetic Score",
            "Win Count Synthetic",
        ]
        reconstruction_metrics = sorted(
            [m for m in scalar_metrics if self._metric_group(m) == "reconstruction"]
        ) + [m for m in _RECON_SUMMARY if m in scalar_metrics]

        synthetic_metrics = sorted(
            [m for m in scalar_metrics if self._metric_group(m) == "synthetic"]
        ) + [m for m in _SYNTH_SUMMARY if m in scalar_metrics]

        models = sorted_models if sorted_models is not None else list(self.models_data.keys())
        
        # Generate plots for each metric group
        plot_style = PlotStyle()
        for metric_group, group_name in [(reconstruction_metrics, "Reconstruction"), (synthetic_metrics, "Synthetic")]:
            if not metric_group:
                continue

            self._generate_metric_group_plot(phase, metric_group, group_name, models, plot_style)
            self._generate_metric_group_boxplot(phase, metric_group, group_name, models, plot_style)
    
    def _generate_metric_group_plot(self, phase: str, metric_group: list, group_name: str, models: list, plot_style, name_overrides: dict | None = None):
        """Generate a single bar chart figure for a metric group."""
        n_metrics = len(metric_group)
        n_cols = 3
        n_rows = int(np.ceil(n_metrics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        # Add header title to the figure
        title_map = {
            "Reconstruction": "Reconstruction Evaluation",
            "Synthetic": "Synthetic Data Evaluation",
        }
        fig.suptitle(title_map.get(group_name, group_name), fontsize=16, fontweight='bold', y=0.995)
            
        width = 0.8 / len(models)
        
        for idx, metric in enumerate(metric_group):
            ax = axes[idx]
            
            # Determine if we want to min/max this metric
            goal = self._metric_goal(metric)
            
            m_values = []
            m_stds = []
            valid_models = []

            for i, model in enumerate(models):
                if phase in self.models_data[model]:
                    if metric in self.models_data[model][phase]:
                        val = self.models_data[model][phase][metric]
                        if isinstance(val, (int, float)):
                            m_values.append(val)
                            valid_models.append(i)
                            meta = self.models_data[model].get(f"{phase}_meta", {}).get(metric, {})
                            m_stds.append(meta.get("std") if meta else None)
                        else:
                            m_values.append(np.nan)
                            m_stds.append(None)
                    else:
                        m_values.append(np.nan)
                        m_stds.append(None)
                else:
                    m_values.append(np.nan)
                    m_stds.append(None)

            # Find best value based on criteria
            best_idx = -1
            best_val = None
            if len(valid_models) > 0:
                valid_vals = [m_values[i] for i in valid_models]
                if goal == "target_1":
                    best_val = min(valid_vals, key=lambda v: abs(v - 1.0))
                elif goal == "target_0":
                    best_val = min(valid_vals, key=lambda v: abs(v - 0.0))
                elif goal == "higher":
                    best_val = max(valid_vals)
                else:
                    best_val = min(valid_vals)

                best_idx = valid_models[valid_vals.index(best_val)]

            # One x-position for the metric
            x = np.arange(1)

            for i in range(len(models)):
                if not np.isnan(m_values[i]):
                    val = m_values[i]
                    std = m_stds[i]
                    color = self._model_color(models[i], i, plot_style)
                    is_best = (i == best_idx)

                    ec = 'black' if is_best else 'none'
                    lw = 1.5 if is_best else 0
                    hatch = '///' if is_best else ''

                    # Offset the bars around the center
                    bar_x = x + i * width - (len(models)-1)*width/2
                    yerr = [[std], [std]] if std is not None else None
                    ax.bar(bar_x, [val], width, color=color, edgecolor=ec, linewidth=lw,
                           hatch=hatch, yerr=yerr, capsize=3,
                           error_kw={"ecolor": "black", "elinewidth": 1.0})
                    
                    # Annotate the best bar with a star
                    if is_best:
                        va = 'bottom' if val >= 0 else 'top'
                        offset_sign = 1 if val >= 0 else -1
                        ax.annotate('★', xy=(bar_x[0], val), xytext=(0, 3 * offset_sign), 
                                    textcoords='offset points', ha='center', va=va, 
                                    fontsize=12, color='black', alpha=0.8)
            
            _display_name = name_overrides.get(metric, self._display_metric_name(metric)) if name_overrides else self._display_metric_name(metric)
            ax.set_title(f"{_display_name} ({self._metric_target_label(metric)})", fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Create a custom legend
        legend_handles = [
            mpatches.Patch(color=self._model_color(model, i, plot_style), label=model)
            for i, model in enumerate(models)
        ]
        legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='★ Best Value'))
        
        # Position the legend above the plots
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(models)+1))
        
        # Hide any unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()

        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        plot_path = os.path.join(phase_output_dir, f"metric_comparison_{group_name}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {plot_path}")

    def _generate_metric_group_boxplot(self, phase: str, metric_group: list, group_name: str, models: list, plot_style):
        """Save metric_comparison_{group_name}_boxplot.pdf — one subplot per metric.

        Models that ran evaluate_repeated(k>1) show a full box plot of their K draw
        values. Models with only a mean scalar (k=1) show a single diamond marker.
        Skipped entirely when no (metric, model) pair has a values list with ≥ 2 entries.
        """
        has_any_values = any(
            len(((self.models_data[m].get(f"{phase}_meta") or {}).get(metric) or {}).get("values") or []) >= 2
            for metric in metric_group
            for m in models
        )
        if not has_any_values:
            return

        n_metrics = len(metric_group)
        n_cols = 3
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        title_map = {
            "Reconstruction": "Reconstruction Evaluation — Draw Distribution",
            "Synthetic": "Synthetic Data Evaluation — Draw Distribution",
        }
        fig.suptitle(title_map.get(group_name, f"{group_name} — Draw Distribution"),
                     fontsize=16, fontweight='bold', y=0.995)

        for idx, metric in enumerate(metric_group):
            ax = axes_flat[idx]

            bp_positions, bp_data, bp_colors = [], [], []
            dot_positions, dot_values, dot_colors = [], [], []
            tick_positions, tick_labels = [], []

            model_pos = 0
            for i, model in enumerate(models):
                phase_data = self.models_data[model].get(phase)
                if phase_data is None or metric not in phase_data:
                    continue
                val = phase_data[metric]
                if not isinstance(val, (int, float)):
                    continue

                model_pos += 1
                color = self._model_color(model, i, plot_style)
                tick_positions.append(model_pos)
                tick_labels.append(model)

                meta = ((self.models_data[model].get(f"{phase}_meta") or {}).get(metric)) or {}
                values = meta.get("values")
                if values and len(values) >= 2:
                    bp_positions.append(model_pos)
                    bp_data.append([float(v) for v in values])
                    bp_colors.append(color)
                else:
                    dot_positions.append(model_pos)
                    dot_values.append(float(val))
                    dot_colors.append(color)

            if bp_data:
                bp = ax.boxplot(bp_data, positions=bp_positions, patch_artist=True, widths=0.5)
                for patch, color in zip(bp["boxes"], bp_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            for pos, val, color in zip(dot_positions, dot_values, dot_colors):
                ax.scatter([pos], [val], color=color, zorder=5, s=80, marker='D')

            ax.set_title(self._display_metric_title(metric), fontsize=11, fontweight='bold')
            if tick_positions:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=20, ha='right', fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        for i in range(n_metrics, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        legend_handles = [
            mpatches.Patch(color=self._model_color(m, i, plot_style), label=m)
            for i, m in enumerate(models)
        ]
        fig.legend(handles=legend_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(models) + 1))

        plt.tight_layout()
        phase_output_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_output_dir, exist_ok=True)
        plot_path = os.path.join(phase_output_dir, f"metric_comparison_{group_name}_boxplot.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Boxplot saved to {plot_path}")

    def _generate_cross_phase_radar(self, models: list[str]) -> None:
        """Spider/radar chart normalising all 4 cross-phase metrics so higher = better."""
        METRIC_KEYS = [
            "Cross-Phase Correlation Error",
            "Cross-Correlation Distribution (KS)",
            "Sliced Wasserstein Distance",
            "Classifier Two-Sample Test (AUC)",
        ]
        AXIS_LABELS = ["Corr.\nError", "CCF KS", "Sliced\nWasserstein", "C2ST\nAUC"]

        raw: dict[str, np.ndarray] = {}
        for model in models:
            cp_data = self.models_data[model].get('cross_phase', {})
            row = []
            for key in METRIC_KEYS:
                v = cp_data.get(key)
                if isinstance(v, dict):
                    v = v.get('mean')
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    row.append(np.nan)
            raw[model] = np.array(row, dtype=float)

        all_vals = np.array([raw[m] for m in models], dtype=float)
        v_min = np.nanmin(all_vals, axis=0)
        v_max = np.nanmax(all_vals, axis=0)
        denom = np.where(v_max > v_min, v_max - v_min, 1e-12)
        # All metrics are lower-better → invert so 1.0 = best on the spider
        normalized = {
            m: np.where(np.isfinite(raw[m]), (v_max - raw[m]) / denom, 0.0)
            for m in models
        }

        n_axes = len(METRIC_KEYS)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'polar': True})
        plot_style = PlotStyle()

        for i, model in enumerate(models):
            vals = normalized[model].tolist() + [normalized[model][0]]
            color = self._model_color(model, i, plot_style)
            ax.plot(angles, vals, color=color, lw=2, label=model)
            ax.fill(angles, vals, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(AXIS_LABELS, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_title(
            "Cross-Phase Metrics — Radar Chart\n(higher = better; relative to loaded models)",
            fontsize=12, pad=22,
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=9)

        out_dir = os.path.join(self.output_dir, 'cross_phase')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'cross_phase_radar.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Cross-phase radar chart → {path}")


def _build_eval_test_loader(
    data_files: list[str],
    split_seed: int,
    split_ratios: dict,
    max_value: float | None,
    window_len: int,
    normalizer,
    num_samples: int,
    output_dir: str,
    phased: bool,
    agg_divisor: float = 1.0,
    output_aggregate: bool = False,
    preprocessed_cache_dir: str | None = None,
) -> "torch.utils.data.DataLoader | None":
    import gc
    import hashlib
    import os
    import torch
    from torch.utils.data import DataLoader as _DL, TensorDataset as _TDS

    # Cache key covers everything that affects WHICH windows land in the test split
    # and their raw values — NOT num_samples (applied at load time from the cache).
    if not phased:
        h = hashlib.md5()
        for f in sorted(data_files):
            h.update(f.encode())
            try:
                h.update(str(os.path.getmtime(f)).encode())
            except OSError:
                pass
        h.update(str(split_seed).encode())
        h.update(str(sorted(split_ratios.items())).encode())
        h.update(str(max_value).encode())
        h.update(str(window_len).encode())
        h.update(str(phased).encode())
        h.update(str(agg_divisor).encode())
        # Normalizer excluded: cache stores raw windows, normalization is at load time.
        cache_path = os.path.join(output_dir, f"_eval_test_cache_{h.hexdigest()[:16]}.pt")
    else:
        cache_path = preprocessed_cache_dir

    def _sample_and_wrap(x_raw, meter_ids, timestamps):
        """Apply normalizer + draw num_samples, return DataLoader."""
        _n_total = len(x_raw)
        _n_cap = min(num_samples, _n_total)
        _g = torch.Generator()
        _g.manual_seed(split_seed)
        _idx = torch.randperm(_n_total, generator=_g)[:_n_cap]
        _x = x_raw[_idx]; _m = meter_ids[_idx]; _t = timestamps[_idx]
        if output_aggregate:
            _x = _x.float()
            if _x.dim() == 3:
                _x = _x.sum(dim=1, keepdim=True)
            if normalizer is not None and getattr(normalizer, "fitted", False):
                _x = normalizer.transform(_x.reshape(-1, 1)).reshape(len(_x), 1, window_len)
        elif normalizer is not None and getattr(normalizer, "fitted", False):
            if phased:
                _x = normalizer.transform(_x.float())
            else:
                _x2 = _x.reshape(-1, 1).float()
                _x2 = normalizer.transform(_x2)
                _x = _x2.reshape(len(_x), window_len).unsqueeze(1)
        else:
            _x = _x.float() if phased else _x.float().unsqueeze(1)
        _agg = torch.zeros((len(_x), 0, window_len), dtype=_x.dtype)
        print(f"[eval cache] Using {_n_cap}/{_n_total} test windows (seed={split_seed}).")
        return _DL(_TDS(_x, _m, _t, _agg), batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)

    # --- Try loading from cache ----------------------------------------------
    if os.path.exists(cache_path) and not phased:
        try:
            cached = torch.load(cache_path, weights_only=True)
            print(f"[eval cache] Cache hit: {len(cached['x'])} total test windows.")
            return _sample_and_wrap(cached["x"], cached["meter_ids"], cached["timestamps"])
        except Exception as _ce:
            print(f"[eval cache] Cache load failed ({_ce}); rebuilding.")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    # --- Build (phased: via InMemoryPhasedDataloaders with disk cache support) ---
    print(f"[eval cache] Scanning {len(data_files)} file(s) for test windows (runs once, then cached)...")

    if phased:
        print(f"[eval cache] Phased data, using InMemoryPhasedDataloaders with disk cache support. Cache path: {cache_path}")
        from src.data.dataloader_phased import InMemoryPhasedDataloaders as _PhasedDL
        _PHASED_FEATURE_COLS = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        _dl = _PhasedDL(
            sm_files=data_files,
            feature_cols=_PHASED_FEATURE_COLS,
            window_len=window_len,
            stride=window_len,
            batch_size=1024,
            aggregation=agg_divisor > 1.0,
            aggregation_divisor=agg_divisor,
            split_ratios=split_ratios,
            split_seed=split_seed,
            max_hour_value=20.0,
            max_value=max_value,
            nan_as_zero=False,
            normalizer=None,
            output_aggregate_stream=False,
            num_workers=1,
            preprocessed_cache_dir=preprocessed_cache_dir,
        )
        test_ds = _dl.test_dataset
        if len(test_ds) == 0:
            print("[eval cache] ERROR: no test windows found — check data_dir and split ratios.")
            return None
        _xc = test_ds.x
        _mc = test_ds.meter_ids
        _tc = test_ds.timestamps
        del _dl, test_ds; gc.collect()

        try:
            os.makedirs(output_dir, exist_ok=True)
            torch.save({"x": _xc, "meter_ids": _mc, "timestamps": _tc}, cache_path)
            print(f"[eval cache] Saved {len(_xc)} test windows → {cache_path}")
        except Exception as _se:
            print(f"[eval cache] Warning: could not save cache: {_se}")

        return _sample_and_wrap(_xc, _mc, _tc)

    else:
        from src.data.dataloader_phased import InMemoryCSVDataset as _FileDS
        def _load_file(fpath):
            return _FileDS(
                sm_files=[fpath], split="test",
                split_ratios=split_ratios, split_seed=split_seed,
                max_hour_value=20.0, max_value=max_value,
                nan_as_zero=False, normalizer=None,
                output_aggregate_stream=False,
                meter_col="meter_number",
                value_cols=["hourly_energy_consumption"],
                window_len=window_len, stride=window_len,
            )

        xs: list[torch.Tensor] = []
        ms: list[torch.Tensor] = []
        ts: list[torch.Tensor] = []

        for fpath in data_files:
            try:
                file_ds = _load_file(fpath)
                if len(file_ds) == 0:
                    del file_ds; gc.collect()
                    continue
                xs.append(file_ds.x.squeeze(1))
                ms.append(file_ds.meter_ids)
                ts.append(file_ds.timestamps)
                del file_ds; gc.collect()
            except Exception as _fe:
                print(f"[eval cache] Warning: skipping {fpath}: {_fe}")

        if not xs:
            print("[eval cache] ERROR: no test windows found — check data_dir and split ratios.")
            return None

        _xc = torch.cat(xs); _mc = torch.cat(ms); _tc = torch.cat(ts)
        del xs, ms, ts; gc.collect()

        try:
            os.makedirs(output_dir, exist_ok=True)
            torch.save({"x": _xc, "meter_ids": _mc, "timestamps": _tc}, cache_path)
            print(f"[eval cache] Saved {len(_xc)} test windows → {cache_path}")
        except Exception as _se:
            print(f"[eval cache] Warning: could not save cache: {_se}")

        return _sample_and_wrap(_xc, _mc, _tc)


def get_eval_files(data_dir: str, data_files: list[str] | None = None):
    """
    Simplified file discovery: if data_files is None,
    grab all .parquet files in data_dir that don't start with 'test_'.
    """
    if data_files is not None and len(data_files) > 0:
        # Ensure absolute paths
        return [os.path.join(data_dir, f) if not os.path.isabs(f) else f for f in data_files]

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    discovered = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".parquet") and not f.startswith("test_"):
                discovered.append(os.path.join(root, f))

    if not discovered:
        raise FileNotFoundError(f"No .parquet files (not starting with 'test_') found in {data_dir}")

    # Deterministic ordering by default to avoid data variance across runs/models.
    return sorted(discovered)

def _find_model_config_path(model_dir: str) -> str | None:
    """Return a config yaml path inside a trained model directory if present."""
    if not os.path.isdir(model_dir):
        return None
    # Prefer the snapshot if training wrote it.
    snap = os.path.join(model_dir, "training_config_used.yaml")
    if os.path.exists(snap):
        return snap
    for f in os.listdir(model_dir):
        if f.endswith(".yaml") or f.endswith(".yml"):
            return os.path.join(model_dir, f)
    return None

def _infer_split_from_first_model(model_paths: list[str]) -> tuple[int, dict]:
    """Infer split seed/ratios from the first model's config as a sensible default."""
    if not model_paths:
        return 0, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    cfg_path = _find_model_config_path(model_paths[0])
    if cfg_path is None:
        return 0, {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    cfg = _load_train_config(cfg_path).data
    seed = int(getattr(cfg.splits, "random_seed", 0))
    ratios = {
        "train_ratio": float(getattr(cfg.splits, "train_ratio", 0.8)),
        "val_ratio": float(getattr(cfg.splits, "val_ratio", 0.1)),
        "test_ratio": float(getattr(cfg.splits, "test_ratio", 0.1)),
    }
    return seed, ratios

def _has_existing_eval_outputs(model_output_dir: str, phased: bool) -> bool:
    """Return True if evaluation artifacts already exist on disk for this model output folder."""

    phases = ["aggregated_phase", "ranked_phase_1", "ranked_phase_2", "ranked_phase_3"]
    for phase in phases:
        p = os.path.join(model_output_dir, f"{phase}_eval", "eval_results.json")
        if os.path.exists(p):
            return True
    if os.path.exists(os.path.join(model_output_dir, "phased_eval_summary.json")):
        return True
    if phased:
        # A flat eval_results.json is from a non-phased run and must not satisfy a phased check.
        return False
    return os.path.exists(os.path.join(model_output_dir, "eval_results.json"))

class _DenormalizingLoader:
    """Wraps a normalized DataLoader and inverse-transforms x on each batch.

    Used so Faraday-family models (trained on raw kWh, normalizer=None) receive
    the same test windows as every other model while seeing un-normalized values.
    """

    def __init__(self, loader, normalizer):
        self.loader = loader
        self.normalizer = normalizer

    def __iter__(self):
        for x, meter_id, timestamps, agg in self.loader:
            yield self.normalizer.inverse_transform(x), meter_id, timestamps, agg

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def _dispatch_holdout_feeder_ids(hid) -> list[int]:
    """Normalise holdout_feeder_id from config to a list of ints.

    Raises TypeError for anything other than int or list[int].
    """
    if isinstance(hid, int):
        return [hid]
    if isinstance(hid, list):
        return hid
    raise TypeError(
        f"holdout_feeder_id must be int or list[int], got {type(hid).__name__!r}: {hid!r}"
    )


def _write_holdout_summary(model_output_dir: str, feeder_ids: list[int]) -> None:
    """Aggregate per-feeder holdout JSONs into an inter-feeder summary file.

    Reads eval_results_feeder_holdout_{id}.json for each feeder ID, collects the
    per-feeder mean for every metric, then writes eval_results_feeder_holdout_summary.json
    with schema: {metric: {mean, std, q95, per_feeder: {id_str: mean}}}.

    std and q95 are None when fewer than two feeders contribute (not meaningful).
    Missing per-feeder JSONs are excluded with a printed warning.
    Writes {} if no feeder JSONs are found.
    """
    per_feeder_data: dict[str, dict[str, float]] = {}  # {metric: {feeder_id_str: mean}}

    for fid in feeder_ids:
        path = os.path.join(model_output_dir, f"eval_results_feeder_holdout_{fid}.json")
        if not os.path.exists(path):
            print(f"[holdout summary] WARNING: JSON for feeder {fid} not found; excluding from summary.")
            continue
        with open(path) as f:
            data = json.load(f)
        for metric, entry in data.items():
            if isinstance(entry, dict) and "mean" not in entry:
                print(f"[holdout summary] WARNING: skipping '{metric}' in feeder {fid}: "
                      f"dict has no 'mean' key (keys={list(entry.keys())[:5]}). "
                      f"This is a diagnostic sidecar that should not be in a holdout JSON.")
                continue
            mean_val = entry["mean"] if isinstance(entry, dict) else float(entry)
            per_feeder_data.setdefault(metric, {})[str(fid)] = mean_val

    summary: dict[str, dict] = {}
    for metric, feeder_means in per_feeder_data.items():
        values = list(feeder_means.values())
        mean = float(np.mean(values))
        std = float(np.std(values)) if len(values) >= 2 else None
        q95 = float(np.percentile(values, 95)) if len(values) >= 2 else None
        summary[metric] = {"mean": mean, "std": std, "q95": q95, "per_feeder": feeder_means}

    out_path = os.path.join(model_output_dir, "eval_results_feeder_holdout_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[holdout summary] Written: {out_path}")


def _load_holdout_scalars(model_dir: str) -> dict[str, float]:
    """Return flat {metric: float} dict from per-feeder or legacy holdout JSON files.

    Looks for eval_results_feeder_holdout_*.json first; falls back to
    eval_results_feeder_holdout.json.  Handles both the new {mean, std, values}
    schema and the legacy flat-scalar schema.
    """
    import glob as _glob
    per_feeder = sorted(
        _glob.glob(os.path.join(model_dir, "eval_results_feeder_holdout_*.json"))
    )
    legacy = os.path.join(model_dir, "eval_results_feeder_holdout.json")
    files = per_feeder if per_feeder else ([legacy] if os.path.exists(legacy) else [])

    result: dict[str, float] = {}
    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            for k, v in data.items():
                if isinstance(v, dict) and "mean" in v:
                    result[k] = float(v["mean"])
                elif isinstance(v, (int, float)):
                    result[k] = float(v)
        except Exception as e:
            print(f"  [holdout] Error loading {filepath}: {e}")
    return result


def _copy_results_to_checkpoint_dir(model_output_dir: str, checkpoint_dir: str) -> None:
    """Copy all eval_results*.json from model_output_dir to checkpoint_dir.

    Called after a recompute so the checkpoint dir holds a durable cache of the
    last compare_models run results (ADR-0011).  Errors are printed but not raised.
    """
    import glob as _glob
    for src in _glob.glob(os.path.join(model_output_dir, "eval_results*.json")):
        dst = os.path.join(checkpoint_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"  [cache] Warning: could not copy {src} → {dst}: {e}")


def _restore_results_from_checkpoint(checkpoint_dir: str, model_output_dir: str) -> bool:
    """Copy eval_results*.json from checkpoint_dir to model_output_dir when available.

    Returns True if at least one file was copied (caller can skip recompute).
    Prints a warning so callers know they are using cached results (ADR-0011).
    """
    import glob as _glob
    sources = _glob.glob(os.path.join(checkpoint_dir, "eval_results*.json"))
    if not sources:
        return False
    print(
        f"  [cache] WARNING: using eval results cached in checkpoint dir by a prior "
        f"compare_models run ({checkpoint_dir}). Pass --recompute if data or settings have changed."
    )
    os.makedirs(model_output_dir, exist_ok=True)
    for src in sources:
        dst = os.path.join(model_output_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"  [cache] Warning: could not restore {src} → {dst}: {e}")
    return True


def _inject_random_holdout_profiles(conditioner, meter_ids: np.ndarray, seed: int = 42) -> set:
    """Overwrite UserProfiler cache entries for holdout meters with random Dirichlet profiles.

    User profiles are computed from all meters before training, so holdout meters already
    have real profiles in the cache derived from their own consumption data. This leaks
    holdout behavior into the condition. We must unconditionally overwrite their entries
    with random profiles before the eval pass. See ADR-0015.

    Returns the set of meter IDs that were overwritten so the caller can clean them up.
    """
    if conditioner is None:
        return set()
    user_profiler = getattr(conditioner, "condition_items", {}).get("user_profile")
    if user_profiler is None:
        return set()

    unique_ids = np.unique(meter_ids).tolist()
    rng = np.random.default_rng(seed)
    profiles = rng.dirichlet(np.ones(user_profiler.n_topics), size=len(unique_ids)).astype(np.float32)

    injected: set = set()
    for i, mid in enumerate(unique_ids):
        mid_int = int(mid)
        user_profiler.user_profiles_cache[mid_int] = profiles[i]
        injected.add(mid_int)

    user_profiler._invalidate_profile_lookup()
    print(
        f"  [holdout] Overwrote user profiles with random Dirichlet for {len(injected)} holdout meters "
        f"(seed={seed}, n_topics={user_profiler.n_topics})."
    )
    return injected


def _remove_injected_holdout_profiles(conditioner, injected_ids: set) -> None:
    """Remove random profiles injected by _inject_random_holdout_profiles."""
    if not injected_ids or conditioner is None:
        return
    user_profiler = getattr(conditioner, "condition_items", {}).get("user_profile")
    if user_profiler is None:
        return
    for mid in injected_ids:
        user_profiler.user_profiles_cache.pop(mid, None)
    user_profiler._invalidate_profile_lookup()


def _build_holdout_loader(
    raw_holdout: HoldoutWindows,
    normalizer,
    window_size: int,
) -> "torch.utils.data.DataLoader":
    """Apply normalizer to cached raw windows and wrap in a DataLoader."""
    from torch.utils.data import TensorDataset, DataLoader as _DL

    x_np = raw_holdout.x
    if normalizer is not None and getattr(normalizer, "fitted", False):
        x_2d = torch.from_numpy(x_np.reshape(-1, 1)).float()
        x_2d = normalizer.transform(x_2d)
        x_t = x_2d.reshape(len(x_np), window_size).unsqueeze(1)
    else:
        x_t = torch.from_numpy(x_np).float().unsqueeze(1)

    agg_t = torch.zeros((len(x_np), 0, window_size), dtype=torch.float32)
    ds = TensorDataset(
        x_t,
        torch.from_numpy(raw_holdout.meter_ids),
        torch.from_numpy(raw_holdout.timestamps),
        agg_t,
    )
    return _DL(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)


def _run_holdout_eval(
    model,
    normalizer,
    conditioner,
    raw_holdout: HoldoutWindows,
    feeder_id: int,
    device: str,
    window_size: int,
    model_output_dir: str,
    n_synthetic_draws: int = 5,
) -> None:
    """Run EvalFramework on one holdout feeder and save per-feeder results.

    raw_holdout: pre-loaded un-normalised windows from load_holdout_windows_raw().
    n_synthetic_draws: number of independent synthetic draws for evaluate_repeated().
    """
    import copy

    _injected = _inject_random_holdout_profiles(conditioner, raw_holdout.meter_ids)
    try:
        _run_holdout_eval_inner(
            model=model, normalizer=normalizer, conditioner=conditioner,
            raw_holdout=raw_holdout, feeder_id=feeder_id, device=device,
            window_size=window_size, model_output_dir=model_output_dir,
            n_synthetic_draws=n_synthetic_draws,
        )
    finally:
        _remove_injected_holdout_profiles(conditioner, _injected)


def _run_holdout_eval_inner(
    model,
    normalizer,
    conditioner,
    raw_holdout: HoldoutWindows,
    feeder_id: int,
    device: str,
    window_size: int,
    model_output_dir: str,
    n_synthetic_draws: int = 5,
) -> None:
    import copy

    holdout_loader = _build_holdout_loader(raw_holdout, normalizer, window_size)
    n_holdout_windows = len(raw_holdout.x)
    print(f"  [holdout] {n_holdout_windows} real windows from feeder {feeder_id}.")

    # All metric plots go into a feeder-specific subdirectory so each feeder's output is
    # isolated and never clobbers the global eval or another feeder's plots.
    # The per-feeder result JSON is written explicitly to model_output_dir (not affected).
    _feeder_plot_dir = os.path.join(model_output_dir, f"feeder_{feeder_id}")
    os.makedirs(_feeder_plot_dir, exist_ok=True)
    eval_norm = copy.deepcopy(normalizer)
    evaluator = EvalFramework(
        model=model,
        dataloader=holdout_loader,
        normalizer=eval_norm,
        conditioner=conditioner,
        device=device,
        window_size=window_size,
        output_path=_feeder_plot_dir,
        n_synthetic_samples=n_holdout_windows,
        preserve_condition_order=True,
    )
    evaluator.register_all(phased=False)
    try:
        repeated_results = evaluator.evaluate_repeated(k=n_synthetic_draws)
    except RuntimeError as e:
        if device != "cpu" and "out of memory" in str(e).lower():
            print("  [holdout] GPU OOM — retrying on CPU.")
            cpu_norm = copy.deepcopy(normalizer)
            evaluator = EvalFramework(
                model=model, dataloader=holdout_loader, normalizer=cpu_norm,
                conditioner=conditioner, device="cpu", window_size=window_size,
                output_path=_feeder_plot_dir, n_synthetic_samples=n_holdout_windows,
                preserve_condition_order=True,
            )
            evaluator.register_all(phased=False)
            repeated_results = evaluator.evaluate_repeated(k=n_synthetic_draws)
        else:
            raise

    holdout_json = os.path.join(model_output_dir, f"eval_results_feeder_holdout_{feeder_id}.json")
    with open(holdout_json, "w") as f:
        json.dump(repeated_results, f, indent=2)
    print(f"  [holdout] Metrics saved to {holdout_json}")

    _rw = getattr(evaluator, '_real_data', None)
    _sw = getattr(evaluator, '_synthetic_data', None)
    _rt = getattr(evaluator, '_timestamps', None)
    _rm = getattr(evaluator, '_meter_ids', None)
    _missing_plot = [n for n, v in [("_real_data", _rw), ("_synthetic_data", _sw), ("_timestamps", _rt), ("_meter_ids", _rm)] if v is None]
    if _missing_plot:
        print(f"  [holdout] Profile plots skipped — EvalFramework did not populate: {_missing_plot}")
    if all(x is not None for x in [_rw, _sw, _rt, _rm]):
        _rw, _sw = np.asarray(_rw), np.asarray(_sw)
        _rt, _rm = np.asarray(_rt), np.asarray(_rm)
        _raw_st = getattr(evaluator, '_synthetic_timestamps', None)
        _raw_sm = getattr(evaluator, '_synthetic_meter_ids', None)
        if _raw_st is not None:
            _st = np.asarray(_raw_st)
            _sm = np.asarray(_raw_sm) if _raw_sm is not None else np.tile(_rm, (_sw.shape[0] // max(_rm.shape[0], 1)) + 1)[:_sw.shape[0]]
        else:
            _n, _reps = _sw.shape[0], (_sw.shape[0] // max(_rt.shape[0], 1)) + 1
            _st = np.tile(_rt, _reps)[:_n]
            _sm = np.tile(_rm, _reps)[:_n]

        # Use the K draws already generated by evaluate_repeated (cached in
        # _all_synthetic_draws) so we don't pay model inference cost again.
        _cached = getattr(evaluator, "_all_synthetic_draws", None) or []
        if _cached:
            _sw = np.asarray(_cached[0])
            _plot_extra_draws = [np.asarray(d) for d in _cached[1:]] or None
        else:
            _plot_extra_draws = None

        _holdout_analysis_dir = os.path.join(model_output_dir, f"feeder_{feeder_id}")
        plot_feeder_holdout_analysis(
            real_windows=_rw, real_timestamps=_rt, real_meter_ids=_rm,
            synth_windows=_sw, synth_timestamps=_st, synth_meter_ids=_sm,
            output_dir=_holdout_analysis_dir,
            conditioner=conditioner,
            synth_extra_draws=_plot_extra_draws,
        )
        _cf_scalars = run_coincidence_factor_analysis(
            real_windows=_rw, real_meter_ids=_rm, real_timestamps=_rt,
            synth_windows=_sw, synth_meter_ids=_sm, synth_timestamps=_st,
            output_dir=_holdout_analysis_dir,
        )
        if _cf_scalars:
            try:
                with open(holdout_json) as _f:
                    _holdout_results = json.load(_f)
                _holdout_results.update(_cf_scalars)
                with open(holdout_json, "w") as _f:
                    json.dump(_holdout_results, _f, indent=2)
            except Exception as _e:
                print(f"  [holdout] Warning: could not merge CF scalars into holdout JSON: {_e}")
                import traceback as _tb; _tb.print_exc()

        # --- Power flow scalars (voltage_nadir, transformer_utilisation) ---
        if True:
            return # Temporarily disable power flow analysis to save time during development; re-enable when needed.
        try:
            from src.power_flow.network_builder import build_feeder_network
            from src.data.feeder_holdout import (
                compute_power_flow_daily_scalars,
                run_power_flow_analysis,
                plot_power_flow_cdf,
            )
            print(f"  [power_flow] Building pandapower network for feeder {feeder_id}...", flush=True)
            from pathlib import Path as _Path
            _pf_net, _pf_m2l = build_feeder_network(
                str(feeder_id),
                mapping_path=_Path(topology_path),
            )
            print(f"  [power_flow] Computing real power flow scalars...", flush=True)
            _real_nadirs, _real_utils = compute_power_flow_daily_scalars(
                _pf_net, _pf_m2l, _rw, _rt, _rm, feeder_id=feeder_id
            )
            _pf_synth_draws = [_sw] + (_plot_extra_draws or [])
            _synth_nadirs_per_draw: list[np.ndarray] = []
            _synth_utils_per_draw: list[np.ndarray] = []
            for _di, _draw in enumerate(_pf_synth_draws):
                print(f"  [power_flow] Synthetic draw {_di + 1}/{len(_pf_synth_draws)}...", flush=True)
                _sn, _su = compute_power_flow_daily_scalars(
                    _pf_net, _pf_m2l, _draw, _st, _sm, feeder_id=feeder_id
                )
                _synth_nadirs_per_draw.append(_sn)
                _synth_utils_per_draw.append(_su)
            _pf_result = run_power_flow_analysis(
                real_nadirs=_real_nadirs,
                real_utils=_real_utils,
                synth_nadirs_per_draw=_synth_nadirs_per_draw,
                synth_utils_per_draw=_synth_utils_per_draw,
            )
            with open(holdout_json) as _f:
                _holdout_results = json.load(_f)
            _holdout_results.update(_pf_result)
            with open(holdout_json, "w") as _f:
                json.dump(_holdout_results, _f, indent=2)
            print(f"  [power_flow] voltage_nadir RMSE={_pf_result['voltage_nadir']['rmse_vs_real']:.4f}  "
                  f"transformer_utilisation RMSE={_pf_result['transformer_utilisation']['rmse_vs_real']:.4f}", flush=True)
            plot_power_flow_cdf(
                real_scalars=_real_nadirs,
                synth_scalars_per_draw=_synth_nadirs_per_draw,
                output_path=os.path.join(_holdout_analysis_dir, "power_flow_voltage_nadir_cdf.png"),
                x_label="Voltage nadir (pu)",
                reference_x=0.95,
                reference_label="0.95 pu limit",
            )
            plot_power_flow_cdf(
                real_scalars=_real_utils,
                synth_scalars_per_draw=_synth_utils_per_draw,
                output_path=os.path.join(_holdout_analysis_dir, "power_flow_transformer_utilisation_cdf.png"),
                x_label="Transformer utilisation (fraction)",
                reference_x=1.0,
                reference_label="100% rated capacity",
            )
        except Exception as _pf_exc:
            print(f"  [power_flow] Skipped: {_pf_exc}")
            import traceback as _tb_pf; _tb_pf.print_exc()


def _build_holdout_loader_phased(
    raw_holdout: HoldoutPhasedWindows,
    normalizer,
    window_size: int,
) -> "torch.utils.data.DataLoader":
    """Apply normalizer to raw phased windows and wrap in a DataLoader."""
    from torch.utils.data import TensorDataset, DataLoader as _DL

    x_np = raw_holdout.x  # [N, n_phases, window_len]
    n, p, l = x_np.shape
    if normalizer is not None and getattr(normalizer, "fitted", False):
        x_t = torch.from_numpy(x_np).float()
        x_t = normalizer.transform(x_t)
    else:
        x_t = torch.from_numpy(x_np).float()

    agg_t = torch.zeros((n, 0, window_size), dtype=torch.float32)
    ds = TensorDataset(
        x_t,
        torch.from_numpy(raw_holdout.meter_ids),
        torch.from_numpy(raw_holdout.timestamps),
        agg_t,
    )
    return _DL(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)


def _run_holdout_eval_phased(
    model,
    normalizer,
    conditioner,
    raw_holdout: HoldoutPhasedWindows,
    holdout_id: int,
    device: str,
    window_size: int,
    model_output_dir: str,
    n_synthetic_draws: int = 5,
    secondary_model=None,
) -> None:
    """Run EvalFramework on a phased holdout set and save results.

    Mirrors _run_holdout_eval but uses register_all(phased=True) and handles
    3-phase windows [N, 3, window_len] instead of 1-channel [N, 1, window_len].
    """
    _injected = _inject_random_holdout_profiles(conditioner, raw_holdout.meter_ids)
    try:
        _run_holdout_eval_phased_inner(
            model=model, normalizer=normalizer, conditioner=conditioner,
            raw_holdout=raw_holdout, holdout_id=holdout_id, device=device,
            window_size=window_size, model_output_dir=model_output_dir,
            n_synthetic_draws=n_synthetic_draws, secondary_model=secondary_model,
        )
    finally:
        _remove_injected_holdout_profiles(conditioner, _injected)


def _run_holdout_eval_phased_inner(
    model,
    normalizer,
    conditioner,
    raw_holdout: HoldoutPhasedWindows,
    holdout_id: int,
    device: str,
    window_size: int,
    model_output_dir: str,
    n_synthetic_draws: int = 5,
    secondary_model=None,
) -> None:
    import copy

    holdout_loader = _build_holdout_loader_phased(raw_holdout, normalizer, window_size)
    n_windows = len(raw_holdout.x)
    n_meters = len(np.unique(raw_holdout.meter_ids))
    print(f"  [holdout phased] {n_windows} windows from {n_meters} meters.")

    _plot_dir = os.path.join(model_output_dir, f"phased_holdout_{holdout_id}")
    os.makedirs(_plot_dir, exist_ok=True)
    eval_norm = copy.deepcopy(normalizer)
    evaluator = EvalFramework(
        model=model,
        dataloader=holdout_loader,
        normalizer=eval_norm,
        conditioner=conditioner,
        device=device,
        window_size=window_size,
        output_path=_plot_dir,
        n_synthetic_samples=n_windows,
        preserve_condition_order=True,
        secondary_model=secondary_model,
    )
    evaluator.register_all(phased=True)
    try:
        repeated_results = evaluator.evaluate_repeated(k=n_synthetic_draws)
    except RuntimeError as e:
        if device != "cpu" and "out of memory" in str(e).lower():
            print("  [holdout phased] GPU OOM — retrying on CPU.")
            cpu_norm = copy.deepcopy(normalizer)
            evaluator = EvalFramework(
                model=model, dataloader=holdout_loader, normalizer=cpu_norm,
                conditioner=conditioner, device="cpu", window_size=window_size,
                output_path=_plot_dir, n_synthetic_samples=n_windows,
                preserve_condition_order=True, secondary_model=secondary_model,
            )
            evaluator.register_all(phased=True)
            repeated_results = evaluator.evaluate_repeated(k=n_synthetic_draws)
        else:
            raise

    holdout_json = os.path.join(model_output_dir, f"eval_results_feeder_holdout_{holdout_id}.json")
    with open(holdout_json, "w") as f:
        json.dump(repeated_results, f, indent=2)
    print(f"  [holdout phased] Metrics saved to {holdout_json}")

    # Profile plots: sum the 3 phases to get the aggregated load profile.
    _rw_raw = getattr(evaluator, '_real_data', None)
    _sw_raw = getattr(evaluator, '_synthetic_data', None)
    _rt = getattr(evaluator, '_timestamps', None)
    _rm = getattr(evaluator, '_meter_ids', None)
    if all(x is not None for x in [_rw_raw, _sw_raw, _rt, _rm]):
        _rw = np.asarray(_rw_raw)
        _sw = np.asarray(_sw_raw)
        # Sum phases: [N, n_phases, window_len] -> [N, window_len]
        if _rw.ndim == 3:
            _rw = _rw.sum(axis=1)
        if _sw.ndim == 3:
            _sw = _sw.sum(axis=1)
        _rt = np.asarray(_rt)
        _rm = np.asarray(_rm)
        _raw_st = getattr(evaluator, '_synthetic_timestamps', None)
        _raw_sm = getattr(evaluator, '_synthetic_meter_ids', None)
        if _raw_st is not None:
            _st = np.asarray(_raw_st)
            _sm = np.asarray(_raw_sm) if _raw_sm is not None else np.tile(_rm, (_sw.shape[0] // max(_rm.shape[0], 1)) + 1)[:_sw.shape[0]]
        else:
            _n = _sw.shape[0]
            _st = np.tile(_rt, (_n // max(_rt.shape[0], 1)) + 1)[:_n]
            _sm = np.tile(_rm, (_n // max(_rm.shape[0], 1)) + 1)[:_n]
        _cached = getattr(evaluator, "_all_synthetic_draws", None) or []
        if _cached:
            _sw = np.asarray(_cached[0])
            if _sw.ndim == 3:
                _sw = _sw.sum(axis=1)
            _plot_extra_draws = [
                np.asarray(d).sum(axis=1) if np.asarray(d).ndim == 3 else np.asarray(d)
                for d in _cached[1:]
            ] or None
        else:
            _plot_extra_draws = None
        try:
            plot_feeder_holdout_analysis(
                real_windows=_rw, real_timestamps=_rt, real_meter_ids=_rm,
                synth_windows=_sw, synth_timestamps=_st, synth_meter_ids=_sm,
                output_dir=_plot_dir,
                conditioner=conditioner,
                synth_extra_draws=_plot_extra_draws,
            )
        except Exception as _pe:
            print(f"  [holdout phased] Profile plots failed: {_pe}")


def _build_mia_train_loader(
    data_files: list[str],
    normalizer,
    split_seed: int,
    split_ratios: dict,
    max_value: float | None,
    window_size: int,
    num_samples: int,
    phased: bool = False,
    agg_divisor: float = 1.0,
    preprocessed_cache_dir: str | None = None,
):
    """Build a capped DataLoader over the training split for MIA.

    Returns None when data_files is empty or the training split contains no windows.
    Caps at num_samples windows (deterministic, no shuffle) so MIA does not dominate
    runtime on large datasets.
    """
    if not data_files:
        return None
    import torch
    import os
    from torch.utils.data import DataLoader as _DL, Subset as _Sub

    if phased:
        from src.data.dataloader_phased import InMemoryPhasedDataloaders
        _PHASED_FEATURE_COLS = ['active_power_p14_l1', 'active_power_p14_l2', 'active_power_p14_l3']
        dl = InMemoryPhasedDataloaders(
            sm_files=data_files,
            feature_cols=_PHASED_FEATURE_COLS,
            window_len=window_size,
            stride=window_size,
            batch_size=1024,
            aggregation=agg_divisor > 1.0,
            aggregation_divisor=agg_divisor,
            split_ratios=split_ratios,
            split_seed=split_seed,
            max_hour_value=20.0,
            max_value=max_value,
            nan_as_zero=False,
            normalizer=None,
            output_aggregate_stream=False,
            num_workers=min(16, (os.cpu_count() or 4)),
            preprocessed_cache_dir=preprocessed_cache_dir,
        )
        train_ds = dl.train_dataset
    else:
        from src.data.dataloader_phased import InMemoryCSVDataloaders
        dl = InMemoryCSVDataloaders(
            sm_files=data_files,
            window_len=window_size,
            stride=window_size,
            batch_size=1024,
            split_ratios=split_ratios,
            split_seed=split_seed,
            max_hour_value=20.0,
            max_value=max_value,
            nan_as_zero=False,
            normalizer=normalizer,
            output_aggregate_stream=False,
            meter_col="meter_number",
            value_cols=["hourly_energy_consumption"],
            pin_memory=False,
        )
        train_ds = dl.train_dataset

    n_cap = min(num_samples, len(train_ds))
    if n_cap == 0:
        return None
    return _DL(
        _Sub(train_ds, list(range(n_cap))),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def _run_tstr_only(
    model,
    normalizer,
    conditioner,
    dataloader,
    output_json: str,
    n_draws: int,
    device: str,
    window_size: int,
    n_synthetic_samples: int,
    preserve_condition_order: bool = False,
    framework_output_path: str | None = None,
) -> None:
    """Build a minimal EvalFramework with only TSTR registered, run it, and patch output_json.

    Reads the existing JSON, updates only 'TSTR (Train on Synthetic, Test on Real)',
    and writes back — all other metric keys are preserved.  Skips with a warning if
    output_json does not already exist (full eval must be run first).

    framework_output_path: where EvalFramework writes its diagnostic sidecar
        (e.g. the TSTR {value, trtr_mse, ...} entry via _save_standalone_metric_result).
        Defaults to output_json.  For holdout feeder calls pass the feeder plot dir so
        the diagnostic does not land in the per-feeder holdout JSON — matching the
        behaviour of _run_holdout_eval which uses output_path=_feeder_plot_dir.

    Always uses evaluate_repeated(k=max(1, n_draws)) so the result is always stored
    as {mean, std, values}, matching the holdout format and accepted by add_model().
    """
    if not os.path.exists(output_json):
        print(f"  [recompute-tstr] SKIP: {output_json} does not exist; run full eval first.")
        return

    import copy
    from src.evaluation.metrics.tstr import TSTR

    tstr_key = "TSTR (Train on Synthetic, Test on Real)"
    k = max(1, n_draws)
    fw_path = framework_output_path if framework_output_path is not None else output_json

    def _do_run(dev: str):
        ev = EvalFramework(
            model=model,
            dataloader=dataloader,
            normalizer=copy.deepcopy(normalizer),
            conditioner=conditioner,
            device=dev,
            window_size=window_size,
            output_path=fw_path,
            n_synthetic_samples=n_synthetic_samples,
            preserve_condition_order=preserve_condition_order,
        )
        ev.register_metric(TSTR(window_size=window_size))
        results = ev.evaluate_repeated(k=k)
        return results.get(tstr_key)

    try:
        tstr_val = _do_run(device)
    except RuntimeError as e:
        if device != "cpu" and "out of memory" in str(e).lower():
            print("  [recompute-tstr] GPU OOM — retrying on CPU.")
            tstr_val = _do_run("cpu")
        else:
            raise

    if tstr_val is None:
        print(f"  [recompute-tstr] WARNING: TSTR result is None; {output_json} not updated.")
        return

    try:
        with open(output_json) as f:
            existing = json.load(f)
    except Exception as e:
        print(f"  [recompute-tstr] WARNING: could not read {output_json} for patching: {e}")
        import traceback; traceback.print_exc()
        return

    if framework_output_path is not None:
        # Diagnostic sidecar is redirected away from output_json; remove any copy
        # that landed here during a prior run before this redirect was in place.
        # "TSTR" is the top-level sidecar key; "standalone_metrics" is a legacy nested
        # variant from older code versions.
        existing.pop("TSTR", None)
        existing.pop("standalone_metrics", None)
    existing[tstr_key] = tstr_val
    with open(output_json, "w") as f:
        json.dump(existing, f, indent=2)

    if isinstance(tstr_val, dict):
        print(f"  [recompute-tstr] Patched: TSTR mean={tstr_val['mean']:.4f} → {output_json}")
    else:
        print(f"  [recompute-tstr] Patched: TSTR={float(tstr_val):.4f} → {output_json}")


def _run_mia_only(
    model,
    normalizer,
    train_loader,
    test_loader,
    output_dir: str,
    device: str,
    window_size: int,
    loader_normalizer=None,
    n_member_samples: int = 1000,
) -> None:
    """Run MIA and save eval_mia_curve.json without triggering synthetic generation.

    Drives reconstruction directly via EvalFramework helper methods so that
    _generate_synthetic() is never called — avoiding failures specific to
    certain model types (diffusion stochasticity, Faraday PDCC decoder, etc.).
    """
    import copy
    import numpy as np
    from src.evaluation.metrics.mia import MembershipInferenceAttack as _MIA

    _test = (
        _DenormalizingLoader(test_loader, loader_normalizer)
        if normalizer is None and loader_normalizer is not None
        else test_loader
    )
    _train = (
        _DenormalizingLoader(train_loader, loader_normalizer)
        if normalizer is None and loader_normalizer is not None
        else train_loader
    )
    eval_norm = copy.deepcopy(normalizer)

    # Build EvalFramework only for its helper methods (_unpack_batch,
    # _inverse_transform_for_metrics, _flatten_for_metrics, _reconstruct_mean).
    # We never call evaluate() so _generate_synthetic() never runs.
    ev = EvalFramework(
        model=model,
        dataloader=_test,
        normalizer=eval_norm,
        device=device,
        window_size=window_size,
        output_path=os.path.join(output_dir, "eval_results.json"),
        n_synthetic_samples=0,
    )
    k = 5 if ev.generator._is_diffusion else 1

    def _collect(loader, cap=None):
        real_list, recon_list = [], []
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if cap is not None and n >= cap:
                    break
                x, meter_ids, timestamps, _agg = ev._unpack_batch(batch)
                x = x.to(device)
                x_inv = ev._inverse_transform_for_metrics(x)
                x_flat = ev._flatten_for_metrics(x_inv)
                recon = ev._reconstruct_mean(x, n_samples=k, meter_id=meter_ids, timestamps=timestamps)
                real_list.append(x_flat.cpu().numpy())
                recon_list.append(recon.cpu().numpy())
                n += x.shape[0]
        if not real_list:
            return None, None
        return np.concatenate(real_list), np.concatenate(recon_list)

    non_member_data, non_member_recon = _collect(_test, cap=n_member_samples)
    member_data, member_recon = _collect(_train, cap=n_member_samples)

    if non_member_data is None or member_data is None:
        print(f"  [MIA] No data collected for {output_dir}; skipping.")
        return

    mia = _MIA(n_reconstruction_samples=k)
    mia.non_member_data = non_member_data
    mia.non_member_reconstructed_data = non_member_recon
    mia.member_data = member_data
    mia.member_reconstructed_data = member_recon
    mia.output_path = os.path.join(output_dir, "eval_results.json")
    mia.compute()


def _run_visual_only(
    model,
    normalizer,
    conditioner,
    dataloader,
    output_dir: str,
    device: str,
    window_size: int,
    n_synthetic_samples: int,
    loader_normalizer=None,
) -> None:
    """Build a minimal EvalFramework with only VisualEvaluation registered and run it.

    Saves pca_comparison.pdf and tsne_comparison.pdf to output_dir/visual_evaluation/.
    Does not read or write eval_results.json.

    loader_normalizer: the normalizer that was applied to dataloader's data (i.e. the
        shared loader's normalizer, _first_norm).  When provided, the dataloader is
        wrapped with _DenormalizingLoader so that real data reaching EvalFramework is
        already in raw-kWh units — ensuring all models see the same real distribution
        in t-SNE/PCA plots.  real_data_preprocessed=True tells EvalFramework not to
        apply a second inverse transform.
    """
    import copy

    # Pre-denormalize real data with the loader's normalizer so every model sees raw
    # kWh.  Without this, each model would inverse-transform the shared normalised
    # loader data with its *own* normalizer, causing the real cluster to look
    # different across models when their normalizers differ.
    raw_loader = (
        _DenormalizingLoader(dataloader, loader_normalizer)
        if loader_normalizer is not None
        else dataloader
    )

    ev = EvalFramework(
        model=model,
        dataloader=raw_loader,
        normalizer=copy.deepcopy(normalizer),
        conditioner=conditioner,
        device=device,
        window_size=window_size,
        output_path=output_dir,
        n_synthetic_samples=n_synthetic_samples,
        real_data_preprocessed=True,
    )
    ev.register_metric(VisualEvaluation(output_path=output_dir))
    ev._prepare_data()
    ev._inject_data_into_metrics()
    ev.evaluate_synthetic_only()


def _run_ev_hp_breakdown(
    model,
    normalizer,
    conditioner,
    dataloader,
    model_output_dir: str,
    device: str,
    window_size: int,
    n_synthetic_samples: int,
    min_group_size: int = 30,
    _shared_real: "dict | None" = None,
    random_user_profiles: bool = False,
) -> "dict | None":
    """Compute aggregated-profile FD for EV/no-EV and HP/no-HP test-set groups.

    Saves ev_hp_breakdown.json (or ev_hp_breakdown_random_profiles.json when
    random_user_profiles=True):
      { "EV": {"fd": float|null, "n_real": int}, "No-EV": ..., "HP": ..., "No-HP": ... }
    Also saves three EV Variance Profile PDFs (ev_variance_profile_*.pdf, or
    ev_variance_profile_*_random_profiles.pdf) when both EV and no-EV groups are
    present and large enough.
    Groups with < min_group_size real windows are recorded with fd=null.
    When the conditioner has no EV or HP items, an empty dict is saved.

    random_user_profiles: when True, inject one random Dirichlet user profile per
    unique test-split meter into the conditioner cache before generation.  This
    isolates the EV condition's effect from per-meter behavioural-type confounding.
    Profiles are removed from the cache after the call (try/finally).

    _shared_real: pre-computed real EV/no-EV denormalised windows from a previous
    call, keyed as {"ev": np.ndarray, "noev": np.ndarray}.  When provided the real
    data is reused so every model plot shows identical real curves.  On the first
    call (None) it is computed here and returned so the caller can pass it forward.
    Returns the shared_real dict (or None on early exit).
    """
    import copy
    import traceback as _evhp_tb
    from torch.utils.data import TensorDataset, DataLoader as _DL
    from src.data.feeder_holdout import _get_ev_hp_flags_per_window, plot_ev_variance_profile
    from src.evaluation.metrics.aggregates import AggregatedProfiles as _AFP

    _AFP_KEY = "Aggregated Profiles (Flexibility Distance)"
    print(f"  [ev_hp] Collecting test windows for EV/HP breakdown...")

    all_x, all_m, all_t, all_agg = [], [], [], []
    try:
        for batch in dataloader:
            x, m, t, agg = batch
            all_x.append(x.cpu())
            all_m.append(m.cpu())
            all_t.append(t.cpu())
            all_agg.append(agg.cpu())
    except Exception as e:
        print(f"  [ev_hp] Data collection failed: {e}")
        return

    if not all_x:
        print("  [ev_hp] No data collected; skipping.")
        return

    x_all   = torch.cat(all_x,   dim=0)
    m_all   = torch.cat(all_m,   dim=0)
    t_all   = torch.cat(all_t,   dim=0)
    agg_all = torch.cat(all_agg, dim=0)

    ev_flags, hp_flags = _get_ev_hp_flags_per_window(m_all.numpy(), conditioner)

    has_ev = bool(ev_flags.any())
    has_hp = bool(hp_flags.any())
    if not has_ev and not has_hp:
        print("  [ev_hp] Conditioner has no EV or HP items; writing empty sidecar.")
        os.makedirs(model_output_dir, exist_ok=True)
        with open(os.path.join(model_output_dir, "ev_hp_breakdown.json"), "w") as f:
            json.dump({}, f)
        return _shared_real

    # Compute real EV/no-EV denormalised windows directly from x_all.
    # Done once on the first call (_shared_real is None) and passed back to the
    # caller so all subsequent model plots use identical real curves — bypasses
    # per-model EvalFramework denormalisation which would vary across models.
    if _shared_real is None:
        _inv = getattr(normalizer, 'inverse_transform', None)
        with torch.no_grad():
            _xd = _inv(x_all.float()) if _inv is not None else x_all.float()
        if isinstance(_xd, torch.Tensor):
            _xd = _xd.detach().cpu().numpy()
        _xd = np.asarray(_xd, dtype=np.float32)
        if _xd.ndim == 3:
            _xd = _xd[:, :, 0] if _xd.shape[2] == 1 else _xd[:, 0, :]
        elif _xd.ndim == 2 and _xd.shape[1] != window_size:
            _xd = _xd[:, :window_size]
        _shared_real = {"ev": _xd[ev_flags], "noev": _xd[~ev_flags]}
        print(f"  [ev_hp] Real windows: EV={len(_shared_real['ev'])}, No-EV={len(_shared_real['noev'])}")

    groups: list[tuple[str, np.ndarray]] = []
    if has_ev:
        groups += [("EV", ev_flags), ("No-EV", ~ev_flags)]
    if has_hp:
        groups += [("HP", hp_flags), ("No-HP", ~hp_flags)]

    _sidecar_name = "ev_hp_breakdown_random_profiles.json" if random_user_profiles else "ev_hp_breakdown.json"
    _vp_suffix = "_random_profiles" if random_user_profiles else ""

    _injected_ev: set = set()
    if random_user_profiles:
        _injected_ev = _inject_random_holdout_profiles(conditioner, m_all.numpy(), seed=42)
        if not _injected_ev:
            print("  [ev_hp] random_user_profiles=True but conditioner has no user_profile item; proceeding without injection.")

    try:
        results: dict[str, dict] = {}
        # Collect synthetic windows per EV/no-EV group for variance profile plots
        _evvp_synth: dict[str, "np.ndarray | None"] = {"EV": None, "No-EV": None}

        for group_name, mask in groups:
            n_group = int(mask.sum())
            results[group_name] = {"fd": None, "n_real": n_group}
            if n_group < min_group_size:
                print(f"  [ev_hp] '{group_name}': {n_group} windows < {min_group_size}; skipping.")
                continue

            mask_t = torch.from_numpy(mask)
            x_g   = x_all[mask_t]
            m_g   = m_all[mask_t]
            t_g   = t_all[mask_t]
            agg_g = agg_all[mask_t]
            n_synth_g = min(n_synthetic_samples, n_group)

            group_loader = _DL(
                TensorDataset(x_g, m_g, t_g, agg_g),
                batch_size=512, shuffle=False, num_workers=0,
            )
            out_subdir = os.path.join(
                model_output_dir,
                f"ev_hp_{group_name.lower().replace(' ', '_').replace('-', '_')}",
            )
            try:
                ev = EvalFramework(
                    model=model,
                    dataloader=group_loader,
                    normalizer=copy.deepcopy(normalizer),
                    conditioner=conditioner,
                    device=device,
                    window_size=window_size,
                    output_path=out_subdir,
                    n_synthetic_samples=n_synth_g,
                )
                ev.register_metric(_AFP(n_samples=min(100, n_group), window_size=window_size))
                ev._prepare_data()
                ev._inject_data_into_metrics()
                ev.evaluate_synthetic_only()
                fd = ev.results.get(_AFP_KEY)
                results[group_name]["fd"] = float(fd) if fd is not None else None
                _fd_str = f"{fd:.4f}" if fd is not None else "None"
                print(f"  [ev_hp] '{group_name}': n={n_group}, FD={_fd_str}")

                # Capture synthetic windows for EV variance profile
                if group_name in _evvp_synth:
                    _evvp_synth[group_name] = ev._synthetic_data  # [N_synth, window_size]
            except Exception as e:
                print(f"  [ev_hp] '{group_name}' failed: {e}")
                _evhp_tb.print_exc()

        out_path = os.path.join(model_output_dir, _sidecar_name)
        os.makedirs(model_output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [ev_hp] Breakdown saved to {out_path}")

        # EV Variance Profile plots — use shared real data so curves are identical across models
        if has_ev and _shared_real is not None:
            if _evvp_synth["EV"] is not None or _evvp_synth["No-EV"] is not None:
                try:
                    plot_ev_variance_profile(
                        real_ev=_shared_real["ev"],
                        synth_ev=_evvp_synth["EV"],
                        real_noev=_shared_real["noev"],
                        synth_noev=_evvp_synth["No-EV"],
                        output_dir=model_output_dir,
                        suffix=_vp_suffix,
                    )
                except Exception as e:
                    print(f"  [ev_hp] EV variance profile plot failed: {e}")
                    _evhp_tb.print_exc()
            else:
                print("  [ev_hp] Skipping EV variance profile — EV groups had too few windows.")
    finally:
        _remove_injected_holdout_profiles(conditioner, _injected_ev)

    return _shared_real


def compare_models(model_paths: list[str],
                   model_names: list[str],
                   secondary_model_paths: list[str] | None = None,
                   eval_result_dirs: list[str] | None = None,
                   data_dir: str = "data/_parquet_cache",
                   data_files: list[str] | None = None,
                   num_samples: int = 100,
                   output_dir: str = "evaluation_comparison",
                   phased: bool = False,
                   agg_divisor = 1000.0,
                   recompute: bool = True,
                   split_seed: int | None = None,
                   split_ratios: dict | None = None,
                   max_value: float | None = 200.0,
                   holdout_n_draws: int = 1,
                   n_draws: int = 1,
                   recompute_tstr: bool = False,
                   recompute_visual: bool = False,
                   mia_model_names: list[str] | None = None,
                   sort_models: bool = True,
                   only_holdout: bool = False,
                   only_cross_phase: bool = False,
                   cross_phase_samples: int | None = None,
                   only_c2st: bool = False,
                   c2st_samples: int | None = None,
                   ev_hp_breakdown: bool = False,
                   ev_random_user_profiles: bool = False,
                   preprocessed_cache_dir: str | None = None,
                   ):
    """
    Run evaluation for multiple models and compare results.
    Can handle both phased and unphased models.

    Args:
        model_paths: Paths to trained model folders (used for loading checkpoints).
        model_names: Display names for comparison.
        secondary_model_paths: Optional secondary model paths aligned with model_paths.
        eval_result_dirs: Optional paths to existing evaluation result folders aligned with
            model_paths. If provided and recompute=False, these folders are used as the
            source for comparison (for example rerun_evaluation_YYYYMMDD_HHMMSS folders).
        n_draws: Number of independent synthetic draws for the main test-split eval.
            n_draws=1 (default) uses a single pass and writes flat scalars.
            n_draws>1 calls evaluate_repeated(k=n_draws) and writes {mean,std,values}
            per synthetic metric (Intra-eval variance).
    """

    # Names-only mode: when no model_paths are given, resolve each name to output_dir/<name>.
    if not model_paths:
        model_paths = [os.path.join(output_dir, name) for name in model_names]
        if not recompute:
            missing = [name for name, p in zip(model_names, model_paths) if not os.path.isdir(p)]
            if missing:
                raise FileNotFoundError(
                    f"Pre-computed results not found for: {missing}. "
                    f"Expected directories under '{output_dir}'."
                )

    if eval_result_dirs is not None and len(eval_result_dirs) < len(model_paths):
        eval_result_dirs = list(eval_result_dirs) + [None] * (len(model_paths) - len(eval_result_dirs))

    # 1. Prepare evaluation data — always discover files when data_dir is provided,
    #    regardless of recompute flag; we may need to compute even without --recompute
    #    when no cached outputs exist yet.
    if data_files is None and data_dir:
        data_files = get_eval_files(data_dir)
    elif data_files is None:
        data_files = []

    # Removed sequential file validation loop, datasets gracefully handle skipping bad files.

    # 2. Pick a single global test split definition so all models see identical windows.
    if split_seed is None or split_ratios is None:
        inferred_seed, inferred_ratios = _infer_split_from_first_model(model_paths)
        split_seed = inferred_seed if split_seed is None else int(split_seed)
        split_ratios = inferred_ratios if split_ratios is None else split_ratios
    split_seed = int(split_seed)
    split_ratios = dict(split_ratios)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = WindowSpec(input_len=24, stride=24)

    # 3. Pre-scan: decide which models need recomputation and resolve output dirs.
    #    This lets us build the shared test loader exactly once, before the eval loop.
    secondaries = (
        secondary_model_paths
        if secondary_model_paths is not None
        else [None] * len(model_paths)
    )
    eval_plan: list[dict] = []
    for i, (path, name, secondary) in enumerate(zip(model_paths, model_names, secondaries)):
        model_output_dir = os.path.join(output_dir, name)
        requested_eval_dir = eval_result_dirs[i] if eval_result_dirs is not None else None
        eval_source_dir = requested_eval_dir if requested_eval_dir else path
        skip = (
            (not recompute and not recompute_tstr and not recompute_visual and not only_cross_phase and not only_c2st)
            and (
                # Checkpoint dir results only count for standalone models — a secondary
                # combo produces different outputs than the primary alone.
                (secondary is None and _has_existing_eval_outputs(eval_source_dir, phased=phased))
                or _has_existing_eval_outputs(model_output_dir, phased=phased)
            )
            and only_holdout is False
        )
        # Checkpoint-dir fallback (ADR-0011): when output_dir/<name>/ is empty and
        # recompute=False, restore cached results from the model checkpoint dir.
        # When recompute_tstr=True or recompute_visual=True we still restore (so any
        # existing JSON is present) but do NOT set skip=True — we need to fall through
        # to the patch-only branch.
        if not skip and not recompute and secondary is None and path != model_output_dir:
            restored = _restore_results_from_checkpoint(path, model_output_dir)
            if restored and not recompute_tstr and not recompute_visual and only_holdout is True:
                skip = True
        eval_plan.append({
            "path": path,
            "name": name,
            "secondary": secondary,
            "model_output_dir": model_output_dir,
            "eval_source_dir": eval_source_dir,
            "skip": skip,
        })

    # 4. Build ONE shared test loader for all models that need recomputation.
    #    Uses the first recomputing model's normalizer (same "infer from first model"
    #    convention already used for split_seed above).  All models are expected to
    #    have been trained on the same dataset with a compatible normalizer.
    shared_test_loader = None
    shared_train_loader = None
    _first_norm = None
    # Cache (model, normalizer, conditioner) for the first recomputing model so we
    # don't load its weights twice (once here for the normalizer, once in the loop).
    _first_model_cache: dict[int, tuple] = {}

    needs_recompute_indices = [i for i, p in enumerate(eval_plan) if not p["skip"]]
    if needs_recompute_indices:
        first_idx = needs_recompute_indices[0]
        _first_model, _first_norm, _first_cond = load_model(
            eval_plan[first_idx]["path"], device=device
        )
        _first_model_cache[first_idx] = (_first_model, _first_norm, _first_cond)

        _cache_dir = os.path.join(output_dir, "_parquet_cache")
        if preprocessed_cache_dir:
            _cache_dir = preprocessed_cache_dir
        else:
            _first_cfg_path = _find_model_config_path(eval_plan[first_idx]["path"])
            if _first_cfg_path is not None:
                try:
                    _hcfg = _load_train_config(_first_cfg_path)
                    if hasattr(_hcfg, "data") and getattr(_hcfg.data, "preprocessed_cache_dir", None):
                        _cache_dir = _hcfg.data.preprocessed_cache_dir
                except Exception:
                    pass
        print(f"[eval cache] Using preprocessed cache dir: {_cache_dir}")

        # When phased, a single-phase normalizer (feature dim 1) cannot transform [n,3,24]
        # data.  Clear _first_norm so the shared loader is built raw and all downstream
        # _loader_norm / _DenormalizingLoader logic remains consistent.
        if phased and _first_norm is not None and getattr(_first_norm, "fitted", False):
            _feat_dim = _first_norm._feature_dim_hint()
            if _feat_dim is not None and _feat_dim != 3:
                _first_norm = None

        shared_test_loader = _build_eval_test_loader(
            data_files=data_files,
            split_seed=split_seed,
            split_ratios=split_ratios,
            max_value=max_value,
            window_len=window.input_len,
            normalizer=_first_norm,
            num_samples=num_samples,
            output_dir=output_dir,
            phased=phased,
            agg_divisor=agg_divisor,
            preprocessed_cache_dir=_cache_dir,
        )
        import gc as _gc; _gc.collect()  # release 1M-tensor heap before model loading
        print(f"Shared test loader ready: {len(shared_test_loader.dataset)} windows for evaluation.")

        if not recompute_tstr and not recompute_visual and only_holdout is False:
            # MIA only needs a few thousand member windows — cap independently of
            # num_samples so that large num_samples values don't trigger loading
            # the entire training split.
            _mia_n = min(num_samples, 5000)
            shared_train_loader = _build_mia_train_loader(
                data_files=data_files,
                normalizer=_first_norm,
                split_seed=split_seed,
                split_ratios=split_ratios,
                max_value=max_value,
                window_size=window.input_len,
                num_samples=_mia_n,
                phased=phased,
                agg_divisor=agg_divisor,
                preprocessed_cache_dir=_cache_dir,
            )
            if shared_train_loader is not None:
                print(f"Shared train loader ready: {len(shared_train_loader.dataset)} member windows for MIA.")

    # 5a. Pre-build holdout window cache (one read per feeder, shared across all models).
    #     Raw un-normalised windows are cached; each model applies its own normalizer.
    holdout_cache: dict[int, HoldoutWindows] = {}
    if needs_recompute_indices and data_files:
        _first_cfg_path = _find_model_config_path(eval_plan[needs_recompute_indices[0]]["path"])
        if _first_cfg_path is not None:
            try:
                _hcfg = _load_train_config(_first_cfg_path)
                _hid_raw = getattr(getattr(_hcfg, "data", None), "holdout_feeder_id", None)
                _htopo  = getattr(getattr(_hcfg, "data", None), "topology_path", None)
                if _hid_raw is not None and _htopo is not None:
                    _hfids = _dispatch_holdout_feeder_ids(_hid_raw)
                    print(f"\nPre-loading holdout windows for {len(_hfids)} feeder(s)...")
                    for _hfid in _hfids:
                        print(f"  Loading feeder {_hfid}...", end=" ", flush=True)
                        try:
                            _h_max_hr = getattr(_hcfg.data, "max_hour_value", None)
                            if _h_max_hr is not None: _h_max_hr = float(_h_max_hr)
                            _h_nan = bool(getattr(_hcfg.data, "nan_as_zero", False))

                            holdout_cache[_hfid] = load_holdout_windows_raw(
                                feeder_id=_hfid,
                                data_files=data_files,
                                topology_path=str(_htopo),
                                window_size=window.input_len,
                                max_value=max_value,
                                max_hour_value=_h_max_hr,
                                nan_as_zero=_h_nan,
                            )
                            print(f"{len(holdout_cache[_hfid].x)} windows.")
                        except Exception as _he:
                            print(f"FAILED: {_he}")
            except Exception as _hce:
                print(f"  [holdout] Could not pre-load holdout cache: {_hce}")

    # 5b. Pre-build phased holdout window cache (phased mode, holdout_feeder_id).
    phased_holdout_cache: dict[int, HoldoutPhasedWindows] = {}
    if needs_recompute_indices and data_files and phased:
        _first_cfg_path = _find_model_config_path(eval_plan[needs_recompute_indices[0]]["path"])
        if _first_cfg_path is not None:
            try:
                _hcfg = _load_train_config(_first_cfg_path)
                _hid_raw = getattr(getattr(_hcfg, "data", None), "holdout_feeder_id", None)
                if _hid_raw is None:
                    _hid_raw = getattr(getattr(_hcfg, "data", None), "holdout_meter_ids", None)
                _htopo  = getattr(getattr(_hcfg, "data", None), "topology_path", None)
                if _hid_raw is not None and _htopo is not None:
                    _phased_feature_cols = ["active_power_p14_l1", "active_power_p14_l2", "active_power_p14_l3"]
                    _hfids = _dispatch_holdout_feeder_ids(_hid_raw)
                    from src.data.feeder_holdout import resolve_holdout_meter_ids
                    print(f"\nPre-loading phased holdout windows for {len(_hfids)} feeder(s)...")
                    for _hfid in _hfids:
                        print(f"  Loading feeder {_hfid}...", end=" ", flush=True)
                        try:
                            _hm_ids = resolve_holdout_meter_ids(str(_htopo), _hfid)
                            _holdout_files = [
                                f for f in data_files
                                if _extract_meter_id_from_filename(os.path.basename(f)) in _hm_ids
                            ]
                            if not _holdout_files:
                                print(f"FAILED: No valid files found for feeder {_hfid}.")
                                continue
                            _h_agg = bool(getattr(getattr(_hcfg.data, "window", None), "aggregated", True))
                            _h_agg_unit = str(getattr(_hcfg.data, "aggregation_energy_unit", "wh")).lower()
                            _h_agg_div = 1000.0 if _h_agg and _h_agg_unit == "kwh" else 1.0
                            _h_max_hr = getattr(_hcfg.data, "max_hour_value", None)
                            if _h_max_hr is not None: _h_max_hr = float(_h_max_hr)
                            _h_nan = bool(getattr(_hcfg.data, "nan_as_zero", False))

                            phased_holdout_cache[_hfid] = load_holdout_windows_phased(
                                files=_holdout_files,
                                feature_cols=_phased_feature_cols,
                                window_size=window.input_len,
                                max_value=max_value,
                                max_hour_value=_h_max_hr,
                                aggregation=_h_agg,
                                aggregation_divisor=_h_agg_div,
                                nan_as_zero=_h_nan,
                            )
                            print(f"{len(phased_holdout_cache[_hfid].x)} windows loaded.")
                        except Exception as _phe:
                            print(f"FAILED: {_phe}")
            except Exception as _phce:
                print(f"  [holdout phased] Could not pre-load phased holdout cache: {_phce}")

    # 5. Loop through models and evaluate, reusing the shared test loader.
    used_eval_dirs: list[str] = []
    _ev_shared_real: "dict | None" = None  # real EV/no-EV arrays, computed once and reused

    for i, plan in enumerate(eval_plan):
        path             = plan["path"]
        name             = plan["name"]
        secondary        = plan["secondary"]
        model_output_dir = plan["model_output_dir"]
        eval_source_dir  = plan["eval_source_dir"]

        print(f"\nEvaluating model: {name}")
        os.makedirs(model_output_dir, exist_ok=True)

        if plan["skip"]:
            _skip_src = model_output_dir if _has_existing_eval_outputs(model_output_dir, phased=phased) else eval_source_dir
            print(f"  Found existing evaluation outputs in {_skip_src}; skipping recompute.")
            # Ensure model_output_dir has the main eval JSON so the comparator can read
            # from one canonical location alongside the holdout JSONs that live there.
            # Only restore from checkpoint dir when model_output_dir itself has no results
            # — avoids overwriting correct phased results with stale single-phase JSONs.
            if eval_source_dir != model_output_dir and not _has_existing_eval_outputs(model_output_dir, phased=phased):
                _restore_results_from_checkpoint(eval_source_dir, model_output_dir)
            used_eval_dirs.append(model_output_dir)
            # Feeder holdout JSONs and summary always live in model_output_dir.
            # If the summary is missing, either build it from existing per-feeder JSONs
            # (when >5 are found) or run the missing Feeder holdout eval passes first.
            _summary_path = os.path.join(model_output_dir, "eval_results_feeder_holdout_summary.json")
            if not os.path.exists(_summary_path) and data_files:
                import glob as _glob
                import traceback as _tb
                _existing_feeder_jsons = sorted(
                    _glob.glob(os.path.join(model_output_dir, "eval_results_feeder_holdout_*.json"))
                )
                _found_ids: list[int] = []
                for _fp in _existing_feeder_jsons:
                    try:
                        _fid_str = os.path.basename(_fp).replace("eval_results_feeder_holdout_", "").replace(".json", "")
                        _found_ids.append(int(_fid_str))
                    except ValueError:
                        continue  # skip summary file and any non-numeric names
                if len(_found_ids) > 5:
                    print(f"  [holdout] {len(_found_ids)} feeder JSONs found; creating holdout summary.")
                    _write_holdout_summary(model_output_dir, _found_ids)
                else:
                    print(
                        f"  [holdout] Only {len(_found_ids)} feeder JSONs found (threshold 5); "
                        f"running missing Feeder holdout eval passes."
                    )
                    _cfg_path = _find_model_config_path(path)
                    if _cfg_path is None:
                        print(f"  [holdout] No training config found in {path}; cannot run holdout eval.")
                    else:
                        try:
                            _mcfg = _load_train_config(_cfg_path)

                            # --- Phased holdout (holdout_meter_ids) ---
                            _phm_raw = getattr(getattr(_mcfg, "data", None), "holdout_meter_ids", None)
                            if _phm_raw is None:
                                _phm_raw = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                            if phased and _phm_raw and 0 not in _found_ids:
                                print(f"  [holdout phased] Running missing phased holdout eval...")
                                # phased_holdout_cache is only pre-loaded when needs_recompute_indices
                                # is non-empty. When all models were skipped it stays empty, so we
                                # populate it lazily here.
                                if not phased_holdout_cache:
                                    _htopo_lazy = getattr(getattr(_mcfg, "data", None), "topology_path", None)
                                    if _htopo_lazy is not None:
                                        _phased_feature_cols = ["active_power_p14_l1", "active_power_p14_l2", "active_power_p14_l3"]
                                        _hfids_lazy = _dispatch_holdout_feeder_ids(_phm_raw)
                                        from src.data.feeder_holdout import resolve_holdout_meter_ids
                                        for _hfid_lazy in _hfids_lazy:
                                            try:
                                                _hm_ids_lazy = resolve_holdout_meter_ids(str(_htopo_lazy), _hfid_lazy)
                                                _holdout_files_lazy = [
                                                    f for f in data_files
                                                    if _extract_meter_id_from_filename(os.path.basename(f)) in _hm_ids_lazy
                                                ]
                                                if not _holdout_files_lazy:
                                                    print(f"  [holdout phased] No files for feeder {_hfid_lazy}; skipping.")
                                                    continue
                                                _h_agg_lazy = bool(getattr(getattr(_mcfg.data, "window", None), "aggregated", True))
                                                _h_agg_unit_lazy = str(getattr(_mcfg.data, "aggregation_energy_unit", "wh")).lower()
                                                _h_agg_div_lazy = 1000.0 if _h_agg_lazy and _h_agg_unit_lazy == "kwh" else 1.0
                                                _h_max_hr_lazy = getattr(_mcfg.data, "max_hour_value", None)
                                                if _h_max_hr_lazy is not None: _h_max_hr_lazy = float(_h_max_hr_lazy)
                                                _h_nan_lazy = bool(getattr(_mcfg.data, "nan_as_zero", False))
                                                phased_holdout_cache[_hfid_lazy] = load_holdout_windows_phased(
                                                    files=_holdout_files_lazy,
                                                    feature_cols=_phased_feature_cols,
                                                    window_size=window.input_len,
                                                    max_value=max_value,
                                                    max_hour_value=_h_max_hr_lazy,
                                                    aggregation=_h_agg_lazy,
                                                    aggregation_divisor=_h_agg_div_lazy,
                                                    nan_as_zero=_h_nan_lazy,
                                                )
                                                print(f"  [holdout phased] Loaded {len(phased_holdout_cache[_hfid_lazy].x)} windows for feeder {_hfid_lazy}.")
                                            except Exception as _lphe:
                                                print(f"  [holdout phased] Load failed for feeder {_hfid_lazy}: {_lphe}")
                                    else:
                                        print(f"  [holdout phased] topology_path not in config; cannot load holdout windows.")
                                _skip_model, _skip_norm, _skip_cond = load_model(path, device=device)
                                _skip_sec_model = None
                                if secondary is not None:
                                    _skip_sec_model, *_ = load_model(secondary, device=device)
                                for _phoid, _phraw in phased_holdout_cache.items():
                                    _run_holdout_eval_phased(
                                        model=_skip_model,
                                        normalizer=_skip_norm,
                                        conditioner=_skip_cond,
                                        raw_holdout=_phraw,
                                        holdout_id=_phoid,
                                        device=device,
                                        window_size=window.input_len,
                                        model_output_dir=model_output_dir,
                                        n_synthetic_draws=holdout_n_draws,
                                        secondary_model=_skip_sec_model,
                                    )
                                if phased_holdout_cache:
                                    _write_holdout_summary(model_output_dir, list(phased_holdout_cache.keys()))

                            # --- Topology-based feeder holdout (unphased) ---
                            _hid = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                            _topo = getattr(getattr(_mcfg, "data", None), "topology_path", None)
                            if not phased and _hid is None:
                                print(f"  [holdout] holdout_feeder_id not set in {_cfg_path}; skipping holdout eval.")
                            elif not phased and _topo is None:
                                print(f"  [holdout] holdout_feeder_id={_hid} set but topology_path missing in {_cfg_path}; skipping.")
                            elif not phased:
                                _fids = _dispatch_holdout_feeder_ids(_hid)
                                _missing_fids = [fid for fid in _fids if fid not in _found_ids]
                                if recompute_tstr:
                                    _missing_fids = _fids # when recomputing TSTR, we want to re-run all holdout feeders to patch their JSONs with updated TSTR values — no skipping
                                print(f"  [holdout] Running {len(_missing_fids)} missing feeder holdout(s) out of {len(_fids)} configured.")
                                _skip_model, _skip_norm, _skip_cond = load_model(path, device=device)
                                for _fid in _missing_fids:
                                    if _fid not in holdout_cache:
                                        try:
                                            holdout_cache[_fid] = load_holdout_windows_raw(
                                                feeder_id=_fid,
                                                data_files=data_files,
                                                topology_path=str(_topo),
                                                window_size=window.input_len,
                                                max_value=max_value,
                                            )
                                        except Exception as _lhe:
                                            print(f"  [holdout] Load failed for feeder {_fid}: {_lhe}")
                                            continue
                                    print(f"  Running holdout eval (feeder {_fid})...")
                                    _run_holdout_eval(
                                        model=_skip_model,
                                        normalizer=_skip_norm,
                                        conditioner=_skip_cond,
                                        raw_holdout=holdout_cache[_fid],
                                        feeder_id=_fid,
                                        device=device,
                                        window_size=window.input_len,
                                        model_output_dir=model_output_dir,
                                        n_synthetic_draws=holdout_n_draws,
                                    )
                                # Re-scan model_output_dir to include both pre-existing and newly computed.
                                _all_jsons = sorted(
                                    _glob.glob(os.path.join(model_output_dir, "eval_results_feeder_holdout_*.json"))
                                )
                                _all_ids: list[int] = []
                                for _fp in _all_jsons:
                                    try:
                                        _fid_str = os.path.basename(_fp).replace("eval_results_feeder_holdout_", "").replace(".json", "")
                                        _all_ids.append(int(_fid_str))
                                    except ValueError:
                                        continue
                                _write_holdout_summary(model_output_dir, _all_ids)
                        except Exception as _e:
                            print(f"  [holdout] FAILED with exception: {_e}")
                            _tb.print_exc()
            continue

        if shared_test_loader is None:
            print(f"  [ERROR] No test data available; skipping '{name}'.")
            used_eval_dirs.append(model_output_dir)
            continue

        # Reuse cached weights for the first model; load fresh for all others.
        if i in _first_model_cache:
            model, normalizer, conditioner = _first_model_cache[i]
        else:
            model, normalizer, conditioner = load_model(path, device=device)

        sec_model = None
        if secondary is not None:
            # Only the model object is needed from the secondary load; the secondary
            # model carries its own conditioner and normalizer internally and will
            # build its own conditions from real phased data / metadata at inference
            # time.  Overwriting normalizer/conditioner here was causing the primary
            # model's encoder to receive wrong-dimensioned condition vectors.
            sec_model, *_ = load_model(secondary, device=device)

        # Faraday-family models have no normalizer and were trained on raw kWh.
        # The in-memory loader yields data normalized with _first_norm, so wrap it
        # to denormalize on the fly for models that have no normalizer of their own.
        _loader_norm = _first_norm
        dataloader = (
            _DenormalizingLoader(shared_test_loader, _loader_norm)
            if normalizer is None and _loader_norm is not None
            else shared_test_loader
        )

        # Guard: when a secondary model is present the primary model is single-phase
        # (e.g. CVAE + HierarchicalSplitter). The shared_test_loader was built with
        # phased=True and provides [n, 3, 24] tensors, but the primary's normalizer
        # expects 1-feature data. Build a per-model unphased loader so PhasedEvaluationSuite
        # can create its inner EvalFramework without a shape mismatch.
        if secondary is not None and phased:
            # CVAE+Splitter composite: provide raw un-normalized 3-phase data so that
            # PhasedEvaluationSuite gets real_phase=[n,3,W]. The generator normalizes
            # the aggregate internally after phase summation (is_denormalized=False path).
            # Reuse the already-loaded shared_test_loader tensors — no disk I/O needed.
            print(f"  '{name}': secondary model present — using pre-loaded data for PhasedEvaluationSuite.")
            try:
                from torch.utils.data import DataLoader as _DL2, TensorDataset as _TDS2
                _ds = shared_test_loader.dataset
                _x_norm = _ds.tensors[0].float()
                if _first_norm is not None and getattr(_first_norm, "fitted", False):
                    _x_raw = _first_norm.inverse_transform(_x_norm)
                else:
                    _x_raw = _x_norm
                dataloader = _DL2(
                    _TDS2(_x_raw, _ds.tensors[1], _ds.tensors[2], _ds.tensors[3]),
                    batch_size=1024, shuffle=False, num_workers=0, pin_memory=False,
                )
                _loader_norm = None
            except Exception as _sgl_e:
                print(f"  [ERROR] Could not build raw phased loader for '{name}': {_sgl_e}; skipping.")
                used_eval_dirs.append(model_output_dir)
                continue

        if recompute_tstr:
            # TSTR-only recompute: patch TSTR in all existing result files without
            # running any other metric (no reconstruction, no MIA, no other synthetic metrics).
            import glob as _tstr_glob, traceback as _tstr_tb

            # Ensure eval results exist in model_output_dir (restore from checkpoint if needed).
            _main_json = os.path.join(model_output_dir, "eval_results.json")
            if not os.path.exists(_main_json) and path != model_output_dir:
                _restore_results_from_checkpoint(path, model_output_dir)

            # 1. Main test-split eval_results.json.
            if not phased:
                print(f"  [recompute-tstr] Patching main eval for '{name}'...")
                if only_holdout:
                    print(f"  [recompute-tstr] only_holdout=True: skipping main test-split TSTR recompute and patch; only holdout feeders will be updated.")
                else:
                    _run_tstr_only(
                        model=model, normalizer=normalizer, conditioner=conditioner,
                        dataloader=dataloader,
                        output_json=_main_json,
                        n_draws=n_draws, device=device,
                        window_size=window.input_len, n_synthetic_samples=num_samples,
                    )
            else:
                print(f"  [recompute-tstr] WARNING: phased TSTR recompute not implemented; "
                      f"skipping phase-specific eval dirs for '{name}'.")

            # 2. Warn about any phased eval dirs that may exist even in non-phased runs.
            for _phase_dir in ['aggregated_phase_eval', 'ranked_phase_1_eval',
                               'ranked_phase_2_eval', 'ranked_phase_3_eval']:
                _pjson = os.path.join(model_output_dir, _phase_dir, "eval_results.json")
                if os.path.exists(_pjson):
                    print(f"  [recompute-tstr] WARNING: {_phase_dir}/eval_results.json exists "
                          f"but phased TSTR recompute is not implemented; that file is NOT updated.")

            # 3. Holdout per-feeder JSONs.
            _cfg_path = _find_model_config_path(path)
            if _cfg_path is None:
                print(f"  [recompute-tstr] No config found in {path}; skipping holdout TSTR recompute.")
            else:
                try:
                    _mcfg = _load_train_config(_cfg_path)
                    _hid = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                    if _hid is None:
                        print(f"  [recompute-tstr] holdout_feeder_id not set; skipping holdout TSTR recompute.")
                    else:
                        _fids = _dispatch_holdout_feeder_ids(_hid)
                        for _fid in _fids:
                            _holdout_json = os.path.join(
                                model_output_dir, f"eval_results_feeder_holdout_{_fid}.json"
                            )
                            if _fid not in holdout_cache:
                                print(f"  [recompute-tstr] WARNING: feeder {_fid} not in holdout cache; skipping.")
                                continue
                            _hl = _build_holdout_loader(holdout_cache[_fid], normalizer, window.input_len)
                            _n_hw = len(holdout_cache[_fid].x)
                            _feeder_plot_dir = os.path.join(model_output_dir, f"feeder_{_fid}")
                            os.makedirs(_feeder_plot_dir, exist_ok=True)
                            print(f"  [recompute-tstr] Patching holdout feeder {_fid} ({_n_hw} windows, {holdout_n_draws} draw(s))...")
                            _run_tstr_only(
                                model=model, normalizer=normalizer, conditioner=conditioner,
                                dataloader=_hl,
                                output_json=_holdout_json,
                                n_draws=holdout_n_draws, device=device,
                                window_size=window.input_len, n_synthetic_samples=_n_hw,
                                preserve_condition_order=True,
                                framework_output_path=_feeder_plot_dir,
                            )
                        _write_holdout_summary(model_output_dir, _fids)
                except Exception as _tstr_e:
                    print(f"  [recompute-tstr] Holdout TSTR failed for '{name}': {_tstr_e}")
                    _tstr_tb.print_exc()

            _copy_results_to_checkpoint_dir(model_output_dir, path)
            used_eval_dirs.append(model_output_dir)
            continue

        if recompute_visual:
            if phased:
                print(f"  [recompute-visual] Skipping '{name}': visual patch not supported for phased models.")
            else:
                print(f"  [recompute-visual] Generating visual evaluation for '{name}'...")
                _n_vis = num_samples  # collect full eval budget so windows span all meters; VisualEvaluation subsamples to 5000
                try:
                    _run_visual_only(
                        model=model,
                        normalizer=normalizer,
                        conditioner=conditioner,
                        dataloader=shared_test_loader,
                        output_dir=model_output_dir,
                        device=device,
                        window_size=window.input_len,
                        n_synthetic_samples=_n_vis,
                        loader_normalizer=_first_norm,
                    )
                except RuntimeError as _ve:
                    if device != "cpu" and "out of memory" in str(_ve).lower():
                        print("  [recompute-visual] GPU OOM — retrying on CPU.")
                        _run_visual_only(
                            model=model, normalizer=normalizer, conditioner=conditioner,
                            dataloader=shared_test_loader, output_dir=model_output_dir,
                            device="cpu", window_size=window.input_len,
                            n_synthetic_samples=_n_vis,
                            loader_normalizer=_first_norm,
                        )
                    else:
                        raise
            used_eval_dirs.append(model_output_dir)
            continue

        if only_cross_phase:
            if not phased:
                print(f"  [only-cross-phase] Skipping '{name}': --only-cross-phase requires --phased.")
            else:
                _cp_samples = cross_phase_samples if cross_phase_samples is not None else num_samples
                print(f"  [only-cross-phase] Running cross-phase metrics only for '{name}' "
                      f"({_cp_samples} samples, {n_draws} draw(s))...")
                evaluator = PhasedEvaluationSuite(
                    model=model,
                    dataloader=dataloader,  # type: ignore[arg-type]
                    normalizer=normalizer,
                    conditioner=conditioner,
                    device=device,
                    window_size=window.input_len,
                    output_path=model_output_dir,
                    secondary_model=sec_model,
                    n_synthetic_samples=_cp_samples,
                    n_draws=n_draws,
                )
                evaluator.evaluate_cross_phase_only()
                _copy_results_to_checkpoint_dir(model_output_dir, path)
            used_eval_dirs.append(model_output_dir)
            continue

        if only_c2st:
            _c2st_samples = c2st_samples if c2st_samples is not None else num_samples
            print(f"  [only-c2st] Running C2ST analysis for '{name}' "
                  f"({_c2st_samples} samples, {n_draws} draw(s), phased={phased})...")
            if phased:
                evaluator = PhasedEvaluationSuite(
                    model=model,
                    dataloader=dataloader,  # type: ignore[arg-type]
                    normalizer=normalizer,
                    conditioner=conditioner,
                    device=device,
                    window_size=window.input_len,
                    output_path=model_output_dir,
                    secondary_model=sec_model,
                    n_synthetic_samples=_c2st_samples,
                    n_draws=n_draws,
                )
                evaluator.evaluate_c2st_only()
            else:
                # Non-phased: lean single-channel C2ST — no reconstruction pass needed.
                import time as _time
                from src.evaluation.metrics.c2st import C2STSuite
                from src.evaluation.metrics.base.base import _sanitize_numeric_array as _san

                _fw = EvalFramework(
                    model,
                    dataloader,  # type: ignore[arg-type]
                    normalizer=normalizer,
                    conditioner=conditioner,
                    device=device,
                    window_size=window.input_len,
                    n_synthetic_samples=_c2st_samples,
                    output_path=model_output_dir,
                    evaluation_title='C2ST DATA COLLECTION',
                    verbose=False,
                )
                _fw._prepare_data()
                _real   = _san(_fw._real_data)
                _synths = [_san(_fw._synthetic_data)]
                for _di in range(1, n_draws):
                    print(f"    Generating draw {_di + 1}/{n_draws}...", end="", flush=True)
                    _t0 = _time.time()
                    with torch.no_grad():
                        _synths.append(_san(_fw._generate_synthetic(_c2st_samples)))
                    print(f" done ({_time.time() - _t0:.1f}s)")
                C2STSuite(
                    output_path=model_output_dir,
                    n_draws=n_draws,
                    device=device,
                ).evaluate(_real, _synths)
            _copy_results_to_checkpoint_dir(model_output_dir, path)
            used_eval_dirs.append(model_output_dir)
            continue

        if phased:
            # Phased models use PhasedEvaluationSuite which creates sub-phase directories
            evaluator = PhasedEvaluationSuite(
                model=model,
                dataloader=dataloader,  # type: ignore[arg-type]
                normalizer=normalizer,
                conditioner=conditioner,
                device=device,
                window_size=window.input_len,
                output_path=model_output_dir,
                secondary_model=sec_model,
                n_synthetic_samples=num_samples,
                n_draws=n_draws,
            )
            evaluator.evaluate()
        else:
            # Unphased models use EvalFramework directly
            evaluator = EvalFramework(
                model=model,
                dataloader=dataloader,  # type: ignore[arg-type]
                normalizer=normalizer,
                conditioner=conditioner,
                device=device,
                window_size=window.input_len,
                output_path=os.path.join(model_output_dir, "eval_results.json"),
                secondary_model=sec_model,
                n_synthetic_samples=num_samples,
                train_dataloader=shared_train_loader,
            )
            evaluator.register_all(phased=False)
            out_json = os.path.join(model_output_dir, "eval_results.json")
            if n_draws > 1:
                results = evaluator.evaluate_repeated(k=n_draws)
            else:
                evaluator.evaluate()
                results = evaluator.results
            with open(out_json, "w") as _f:
                json.dump(results, _f, indent=2)

        # Cache results back to checkpoint dir so recompute=False can find them later (ADR-0011).
        _copy_results_to_checkpoint_dir(model_output_dir, path)

        # Run holdout eval if the model's training config has holdout_feeder_id set.
        import glob as _glob
        _any_holdout_exists = bool(
            _glob.glob(os.path.join(model_output_dir, "eval_results_feeder_holdout_*.json"))
        )
        if (recompute or not _any_holdout_exists) and data_files:
            _cfg_path = _find_model_config_path(path)
            if _cfg_path is None:
                print(f"  [holdout] No training config found in {path}; cannot run holdout eval.")
            else:
                import traceback as _tb
                try:
                    _mcfg = _load_train_config(_cfg_path)

                    # --- Phased file-based holdout (holdout_feeder_id) ---
                    _hid = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                    if _hid is None:
                        _hid = getattr(getattr(_mcfg, "data", None), "holdout_meter_ids", None)
                    if phased and _hid is not None:
                        _fids = _dispatch_holdout_feeder_ids(_hid)
                        for _fid in _fids:
                            if _fid in phased_holdout_cache:
                                _phraw = phased_holdout_cache[_fid]
                                print(f"  Running phased holdout eval for feeder {_fid} ({len(_phraw.x)} windows)...")
                                _run_holdout_eval_phased(
                                    model=model,
                                    normalizer=normalizer,
                                    conditioner=conditioner,
                                    raw_holdout=_phraw,
                                    holdout_id=_fid,
                                    device=device,
                                    window_size=window.input_len,
                                    model_output_dir=model_output_dir,
                                    n_synthetic_draws=holdout_n_draws,
                                    secondary_model=sec_model,
                                )
                        if _fids and any(f in phased_holdout_cache for f in _fids):
                            _write_holdout_summary(model_output_dir, _fids)

                    # --- Topology-based feeder holdout (unphased) ---
                    _hid = getattr(getattr(_mcfg, "data", None), "holdout_feeder_id", None)
                    _topo = getattr(getattr(_mcfg, "data", None), "topology_path", None)
                    if not phased and _hid is None:
                        print(f"  [holdout] holdout_feeder_id not set in {_cfg_path}; skipping holdout eval.")
                    elif not phased and _topo is None:
                        print(f"  [holdout] holdout_feeder_id={_hid} is set but topology_path is missing in {_cfg_path}; skipping holdout eval.")
                    elif not phased:
                        _fids = _dispatch_holdout_feeder_ids(_hid)
                        for _fid in _fids:
                            if _fid not in holdout_cache:
                                print(f"  [holdout] Feeder {_fid} not in cache; loading now...")
                                try:
                                    holdout_cache[_fid] = load_holdout_windows_raw(
                                        feeder_id=_fid,
                                        data_files=data_files,
                                        topology_path=str(_topo),
                                        window_size=window.input_len,
                                        max_value=max_value,
                                    )
                                except Exception as _lhe:
                                    print(f"  [holdout] Load failed for feeder {_fid}: {_lhe}")
                                    continue
                            print(f"  Running holdout eval (feeder {_fid})...")
                            _run_holdout_eval(
                                model=model,
                                normalizer=normalizer,
                                conditioner=conditioner,
                                raw_holdout=holdout_cache[_fid],
                                feeder_id=_fid,
                                device=device,
                                window_size=window.input_len,
                                model_output_dir=model_output_dir,
                                n_synthetic_draws=holdout_n_draws,
                            )
                        _write_holdout_summary(model_output_dir, _fids)
                except Exception as _e:
                    print(f"  [holdout] FAILED with exception: {_e}")
                    _tb.print_exc()

        used_eval_dirs.append(model_output_dir)

        # EV/HP breakdown (optional, unphased only — phased uses a different evaluator structure)
        # Always recomputes when the flag is set — no sidecar caching.
        if ev_hp_breakdown and not phased:
            try:
                _ev_shared_real = _run_ev_hp_breakdown(
                    model=model,
                    normalizer=normalizer,
                    conditioner=conditioner,
                    dataloader=dataloader,
                    model_output_dir=model_output_dir,
                    device=device,
                    window_size=window.input_len,
                    n_synthetic_samples=num_samples,
                    _shared_real=_ev_shared_real,
                    random_user_profiles=ev_random_user_profiles,
                ) or _ev_shared_real
            except Exception as _evhp_e:
                print(f"  [ev_hp] EV/HP breakdown failed for '{name}': {_evhp_e}")
                import traceback as _evhp_loop_tb; _evhp_loop_tb.print_exc()

        # Release GPU memory between models so each starts with a clean VRAM state.
        import gc as _gc2
        _gc2.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5b. MIA fallback: for each model missing eval_mia_curve.json, run MIA-only.
    #     Builds loaders on demand when the main loop skipped them (recompute=False).
    #     When mia_model_names is set, only those models get sidecar computation.
    _mia_missing = [
        (i, plan) for i, plan in enumerate(eval_plan)
        if not os.path.exists(os.path.join(plan["model_output_dir"], "eval_mia_curve.json"))
        and (mia_model_names is None or plan["name"] in mia_model_names)
    ]
    if _mia_missing and data_files:
        _mia_test_loader = shared_test_loader
        _mia_train_loader = shared_train_loader
        # Phased loaders provide raw un-normalized data; applying a 1-feature normalizer
        # to 3-phase tensors would fail. MIA still runs, just on raw data.
        _mia_loader_norm = None if phased else _first_norm
        _mia_model_cache: dict[int, tuple] = dict(_first_model_cache)

        if _mia_test_loader is None or _mia_train_loader is None:
            print("\n[MIA fallback] Building data loaders (main loop was skipped)...")
            try:
                _mia_first_i, _mia_first_plan = _mia_missing[0]
                _mia_fm, _mia_fn, _mia_fc = load_model(_mia_first_plan["path"], device=device)
                _mia_model_cache[_mia_first_i] = (_mia_fm, _mia_fn, _mia_fc)
                _mia_cache_dir = preprocessed_cache_dir or os.path.join(output_dir, "_parquet_cache")
                _mia_norm = _mia_fn
                if phased and _mia_norm is not None and getattr(_mia_norm, "fitted", False):
                    _feat_dim = _mia_norm._feature_dim_hint()
                    if _feat_dim is not None and _feat_dim != 3:
                        _mia_norm = None
                _mia_test_loader = _build_eval_test_loader(
                    data_files=data_files, split_seed=split_seed,
                    split_ratios=split_ratios, max_value=max_value,
                    window_len=window.input_len, normalizer=_mia_norm,
                    num_samples=num_samples, output_dir=output_dir,
                    phased=phased, agg_divisor=agg_divisor,
                    preprocessed_cache_dir=_mia_cache_dir,
                )
                _mia_loader_norm = _mia_norm
                _mia_train_loader = _build_mia_train_loader(
                    data_files=data_files, normalizer=_mia_norm,
                    split_seed=split_seed, split_ratios=split_ratios,
                    max_value=max_value, window_size=window.input_len,
                    num_samples=num_samples, phased=phased, agg_divisor=agg_divisor,
                    preprocessed_cache_dir=_mia_cache_dir,
                )
            except Exception as _mia_build_e:
                print(f"  [MIA fallback] Could not build loaders: {_mia_build_e}")
                import traceback as _mia_build_tb; _mia_build_tb.print_exc()
                _mia_test_loader = None
                _mia_train_loader = None

        for _mia_i, _mia_plan in _mia_missing:
            if _mia_test_loader is None or _mia_train_loader is None:
                print(f"  [MIA fallback] No loaders available; skipping '{_mia_plan['name']}'.")
                continue
            print(f"\n[MIA fallback] Running MIA-only for '{_mia_plan['name']}'...")
            try:
                if _mia_i in _mia_model_cache:
                    _fm, _fn, _fc = _mia_model_cache[_mia_i]
                else:
                    _fm, _fn, _fc = load_model(_mia_plan["path"], device=device)
                _run_mia_only(
                    model=_fm, normalizer=_fn,
                    train_loader=_mia_train_loader,
                    test_loader=_mia_test_loader,
                    output_dir=_mia_plan["model_output_dir"],
                    device=device, window_size=window.input_len,
                    loader_normalizer=_mia_loader_norm,
                )
            except Exception as _mia_e:
                print(f"  [MIA fallback] Failed for '{_mia_plan['name']}': {_mia_e}")
                import traceback as _mia_tb; _mia_tb.print_exc()

    # 5c. EV/HP breakdown fallback: compute for any model skipped in the main loop.
    # Always recomputes when the flag is set — no sidecar caching.
    if ev_hp_breakdown and not phased and data_files:
        _evhp_missing = list(enumerate(eval_plan))
        if _evhp_missing:
            _evhp_loader = shared_test_loader
            _evhp_loader_norm = _first_norm
            _evhp_model_cache2: dict[int, tuple] = dict(_first_model_cache)

            if _evhp_loader is None:
                print("\n[ev_hp fallback] Building data loader (main loop was skipped)...")
                try:
                    _evhp_fi, _evhp_fp = _evhp_missing[0]
                    _evhp_bfm, _evhp_bfn, _evhp_bfc = load_model(_evhp_fp["path"], device=device)
                    _evhp_model_cache2[_evhp_fi] = (_evhp_bfm, _evhp_bfn, _evhp_bfc)
                    _evhp_loader = _build_eval_test_loader(
                        data_files=data_files, split_seed=split_seed,
                        split_ratios=split_ratios, max_value=max_value,
                        window_len=window.input_len, normalizer=_evhp_bfn,
                        num_samples=num_samples, output_dir=output_dir,
                        phased=phased, agg_divisor=agg_divisor,
                    )
                    _evhp_loader_norm = _evhp_bfn
                except Exception as _evhp_bl:
                    print(f"  [ev_hp fallback] Could not build loader: {_evhp_bl}")
                    _evhp_loader = None

            for _evhp_fi2, _evhp_fp2 in _evhp_missing:
                if _evhp_loader is None:
                    print(f"  [ev_hp fallback] No loader; skipping '{_evhp_fp2['name']}'.")
                    continue
                print(f"\n[ev_hp fallback] Computing breakdown for '{_evhp_fp2['name']}'...")
                try:
                    if _evhp_fi2 in _evhp_model_cache2:
                        _efm, _efn, _efc = _evhp_model_cache2[_evhp_fi2]
                    else:
                        _efm, _efn, _efc = load_model(_evhp_fp2["path"], device=device)
                    _evhp_dl = (
                        _DenormalizingLoader(_evhp_loader, _evhp_loader_norm)
                        if _efn is None and _evhp_loader_norm is not None
                        else _evhp_loader
                    )
                    _ev_shared_real = _run_ev_hp_breakdown(
                        model=_efm, normalizer=_efn, conditioner=_efc,
                        dataloader=_evhp_dl,
                        model_output_dir=_evhp_fp2["model_output_dir"],
                        device=device, window_size=window.input_len,
                        n_synthetic_samples=num_samples,
                        _shared_real=_ev_shared_real,
                        random_user_profiles=ev_random_user_profiles,
                    ) or _ev_shared_real
                except Exception as _evhp_re:
                    print(f"  [ev_hp fallback] Failed for '{_evhp_fp2['name']}': {_evhp_re}")
                    import traceback as _evhp_ftb; _evhp_ftb.print_exc()

    # 6. Use PhasedModelComparator to generate comparison tables and plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON")
    print("="*50)

    # Count trainable parameters for each model from its config (best-effort; N/A on failure).
    _param_counts: dict[str, int | None] = {}
    for plan in eval_plan:
        _name = plan["name"]
        _cfg_path = _find_model_config_path(plan["path"])
        if _cfg_path is None:
            print(f"  [params] No config found for '{_name}'; parameter count unavailable.")
            _param_counts[_name] = None
            continue
        try:
            from src.model.usable.count_learnable_params import (  # noqa: PLC0415
                _load_config as _clp_load_config,
                count_from_config as _clp_count,
            )
            _cfg = _clp_load_config(_cfg_path)
            _, _trainable, _ = _clp_count(_cfg)
            _param_counts[_name] = _trainable
        except Exception as _pe:
            print(f"  [params] Could not count params for '{_name}': {_pe}")
            import traceback as _params_tb; _params_tb.print_exc()
            _param_counts[_name] = None

    comparator = PhasedModelComparator(output_dir=output_dir)
    if len(used_eval_dirs) != len(model_names):
        raise ValueError(
            f"Internal mismatch: have {len(used_eval_dirs)} eval dirs for {len(model_names)} model names."
        )
    for name, eval_dir in zip(model_names, used_eval_dirs):
        comparator.add_model(name, eval_dir, trainable_params=_param_counts.get(name))

    if ev_hp_breakdown:
        comparator._generate_ev_hp_plots()
        print("\nEV/HP breakdown complete. Results saved to:", output_dir)
        return comparator.models_data

    comparator.compare(mia_models=mia_model_names, sort_models=sort_models)

    print("\nComparison complete. Results saved to:", output_dir)
    return comparator.models_data

if __name__ == "__main__":
    import faulthandler as _fh, sys as _sys
    # Write C-level crash dumps (segfaults, OOM kills) to a file alongside the log.
    _crash_log = open("compare_models_crash.log", "w")
    _fh.enable(file=_crash_log, all_threads=True)

    parser = argparse.ArgumentParser(description="Compare model evaluation results (reuse outputs or recompute).")
    parser.add_argument("--models", nargs="+", required=False, help="List of model directories (each contains config + checkpoint).")
    parser.add_argument("--names", nargs="+", required=False, help="Display names for models (same length as --models).")
    parser.add_argument("--secondary-models", nargs="*", default=None, help="Optional secondary model dirs aligned with --models. Use 'None' to skip.")
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=None,
        help="Optional existing evaluation result dirs aligned with --models (for example rerun_evaluation_* folders).",
    )
    parser.add_argument("--data-dir", type=str, default="data/_parquet_cache", help="Directory with parquet files.")
    parser.add_argument("--data-files", nargs="*", default=None, help="Optional explicit parquet files (relative to --data-dir unless absolute).")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for preprocessed window cache (overrides preprocessed_cache_dir from model config).")
    parser.add_argument("--num-samples", type=int, default=10000, help="Approx number of windows to evaluate per model.")
    parser.add_argument("--output-dir", type=str, default="evaluation_comparison", help="Where to write/read evaluation outputs.")
    parser.add_argument("--phased", action="store_true", help="Use phased evaluation suite.")
    parser.add_argument("--recompute", action="store_true", help="Recompute evaluation even if outputs already exist.")
    parser.add_argument("--recompute-tstr", action="store_true", help="Recompute only TSTR in all existing eval result files (main eval + all holdout feeders). Skips reconstruction, MIA, and all other synthetic metrics.")
    parser.add_argument("--split-seed", type=int, default=None, help="Seed used for deterministic window-level train/val/test split. Defaults to first model's training config.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train ratio for split (defaults to first model's training config).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Val ratio for split (defaults to first model's training config).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio for split (defaults to first model's training config).")
    parser.add_argument("--max-value", type=float, default=200.0, help="Maximum value for filtering windows.")
    parser.add_argument("--holdout-n-draws", type=int, default=1, help="Number of synthetic draws to average for holdout feeder profile plot.")
    parser.add_argument("--n-draws", type=int, default=1, help="Number of independent synthetic draws for main test-split eval (Intra-eval variance). 1=single pass (default).")
    parser.add_argument("--mia-models", nargs="*", default=None, help="Subset of model names to include in MIA plots. Defaults to all models with a sidecar.")
    parser.add_argument("--recompute-visual", action="store_true", help="Recompute only visual evaluation (reconstruction plots) for all models, skipping all other metrics. Not compatible with phased evaluation.")
    parser.add_argument("--no-sort", action="store_true", help="Disable automatic model sorting; models appear in the order listed in --names.")
    parser.add_argument("--only-holdout", action="store_true", help="Only run holdout evaluation (and summary) for models with holdout_feeder_id set in their training config. Skips main test-split evaluation and all metrics except TSTR on the holdout.")
    parser.add_argument("--ev-hp-breakdown", action="store_true", help="Compute and plot EV/no-EV and HP/no-HP aggregated-profile FD breakdown and EV variance profile plots on the full test set. Always recomputes and overwrites existing sidecars and PDFs.")
    parser.add_argument("--ev-random-user-profiles", action="store_true", help="When --ev-hp-breakdown is active, inject random Dirichlet user profiles for all test-split meters before synthetic generation. Isolates the EV condition effect from per-meter behavioural-type confounding. Writes separate _random_profiles sidecar files and PDFs alongside the real-profile outputs.")
    parser.add_argument("--only-cross-phase", action="store_true", help="Compute only the four cross-phase metrics (CrossPhaseCorrelationError, CrossCorrelationDistribution, SlicedWassersteinDistance, ClassifierTwoSampleTest) for phased models. Patches existing phased_eval_summary.json; skips all per-phase metric computation and reconstruction passes.")
    parser.add_argument("--cross-phase-samples", type=int, default=None, help="Number of synthetic samples for --only-cross-phase (defaults to --num-samples). Reduce to speed up generation.")
    parser.add_argument("--only-c2st", action="store_true", help="Run only the single-channel C2ST analysis (aggregate + 3 ranked phases) for phased models. Patches existing phased_eval_summary.json; skips all per-phase metric computation and reconstruction passes.")
    parser.add_argument("--c2st-samples", type=int, default=None, help="Number of synthetic samples for --only-c2st (defaults to --num-samples).")
    args = parser.parse_args()
    holdout_n_draws = int(args.holdout_n_draws)
    n_draws = int(args.n_draws)


    # args.models = [
    #             #    "E:/models/new4/CVAE_20260601_034519_guide_cnn",
    #             #    "E:/models/new4/FaradayModel_20260531_124255",
    #             #    "E:/models/new4/FaradayModel_20260601_060319",
    #             #    "E:/models/new4/GuidedFaradayModel_20260531_115215",
    #             #    "E:/models/new4/GuidedFaradayModel_20260531_131653_relu",
    #             #    "E:/models/new4/GuidedFaradayModel_20260531_143252_peaks",
    #             #    "E:/models/new4/GuidedFaradayModel_20260531_204015",
    #             #    "E:/models/new4/GuidedFaradayModel_20260531_211801_cnn",
    #             # "E:/models/new4/GuidedFaradayModel_20260531_225823_userprofile",
    #             # "E:/models/new3/FaradayModel_20260529_021321", # NOT for holdout
    #             # "E:/models/new3/GuidedFaradayModel_20260528_223126", # NOt for holdout
    #             # "E:/models/new3/GuidedFaradayModel_20260529_000300_cnn", # NOT for holdout
    #             # "E:/models/new3/GuidedFaradayModel_20260529_012840_cnn_mmd" # NOT for holdout
    #             #"E:/models/new4/GuidedFaradayModel_20260604_102425_deterministic", # use
    #             #"E:/models/new4/GuidedFaradayModel_20260603_123450", # use
    #             #"E:/models/new4/GuidedFaradayModel_20260605_173422",
    #             # "E:/models/new4/GuidedFaradayModel_20260603_120241", # use
    #             # "E:/models/new4/GuidedFaradayModel_20260603_102551", # use
    #             "E:/models/new4/GuidedFaradayModel_20260605_105946",# "E:/models/new4/GuidedFaradayModel_20260605_112623",
    #             "E:/models/new4/GuidedFaradayModel_20260605_201145",
    #             "E:/models/new4/CVAE_20260531_185015_guide_cnn_h_broken",
    #             "E:/models/new4/DiffusionModel_20260601_002842", # use
    #             "E:/models/new4/FaradayModel_20260601_060319", # USe
    #             "E:/models/new4/CVAE_20260601_022452_guide", # USE
    #             "E:/models/new4/CVAE_20260531_180003_cvae",
    #             "E:/models/new4/QuantileBaselineModel_20260601_005326",
    #             "E:/models/new4/AverageBaselineModel_20260531_173051",
    #             "E:/models/new4/RealDataBaselineModel_20260608_143634",
    #                ]
    # args.names = [
    #                 # "Guide-CNN",
    #                 # "Faraday",
    #                 # "Faraday-new-fix",
    #                 # "Guided-Faraday",
    #                 # "Guided-Faraday-RELU",
    #                 # "Guided-Faraday-PEAKS",
    #                 # "Guided-Faraday-Deterministic",
    #                 # "Guided-Faraday-CNN-2",
    #                 # "Guided-Faraday-UserProfile",
    #                 # "Faraday-OLD",
    #                 # "Guided-Faraday-OLD",
    #                 # "Guided-Faraday-CNN-OLD",
    #                 # "Guided-Faraday-CNN-MMD-OLD",
    #                 #"Guided-Faraday-CNN-Deterministic-big", # use
    #                 #"Guided-Faraday-CNN-Stochastic-big", # Use
    #                 #"Guided-Faraday-CNN-newGMM",
    #                 #"Guided-Faraday-UserProfiles", # use
    #                 #"Guided-Faraday", # USe
    #                 "Guided-Faraday-CNN",# "Guided-Faraday-CNN-Stochastic",
    #                 "Guided-Faraday-CNN-UserProfiles",
    #                 "H-Guide-CNN", # use
    #                 "Diffusion", # use
    #                 "Faraday", # USE
    #                 "Guide", # USE
    #                 "CVAE", # use
    #                 "Baseline-Quantile", # use
    #                 "Baseline-Average", # use
    #                 "Baseline-RealData", # use
    # ]
    # args.mia_models = [
    #     "Guided-Faraday-CNN",
    #     "Diffusion",
    #     "H-Guide-CNN",
    # ]

    # PHASED:
    # args.models = ["E:/models/phased2/SetGuideVAE_20260607_124614",
    #                "E:/models/phased2/SetGuideVAE_20260607_132508_CNN",
    #               "E:/models/phased2/DiffusionModel_20260607_080656",
    #               "E:/models/phased2/SequentialPhaseVAE_20260607_082110",
    #                 "E:/models/new4/GuidedFaradayModel_20260605_105946",
    #                 "E:/models/phased2/QuantileBaselineModel_20260607_112946",
    #                 "E:/models/phased2/AverageBaselineModel_20260607_113136"
    #                ]
    # args.secondary_models = [None, 
    #                          None, 
    #                          None,
    #                          None, 
    #                          "E:/models/phased/HierarchicalSplitter_20260604_171559", 
    #                          None,
    #                          None
    #                          ]
    # args.names = [ "Set-Guide-VAE",
    #                 "Set-Guide-VAE-CNN",
    #                 "Diffusion_phased", 
    #                 "Sequential",
    #                 "Guide-CNN (with Hierarchical Splitter)",
    #                 "Quantile Baseline",
    #                 "Average Baseline"
    #               ]
    # uv run .\src\evaluation\compare_models.py --models "s" --names "s" --holdout-n-draws 5 --num-samples 50000 --data-dir "E:\Data\processed_data\second_batch\individual_sm_datasets\raw_profiles" --n-draws 10 --phased --cache-dir "E:/Data/processed_data/cached_data" --no-sort


    # args.names = ["Diffusion"]
    # args.models = ["E:/models/new4/DiffusionModel_20260601_002842"]
    # uv run src/evaluation/compare_models.py --models "s" --names "s" --num-samples 50000 --holdout-n-draws 5 --n-draws 10 --recompute --only-holdout

    # args.models = ["E:/models/new4/CVAE_20260531_185015_guide_cnn_h_broken"]
    # args.secondary_models = ["E:/models/phased/HierarchicalSplitter_20260604_171559"]
    # args.names = ["Guide-CNN (with Hierarchical Splitter)"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260605_201145"]
    # args.names = ["Guided-Faraday-CNN-UserProfiles-newMMD"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260605_173422"]
    # args.names = ["Guided-Faraday-CNN-newGMM-userProfiles"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260603_120241"]
    # args.names = ["Guided-Faraday-UserProfiles"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260604_102425_deterministic"]
    # args.names = ["Guided-Faraday-CNN-Deterministic-big"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260605_105946"]
    # args.names = ["Guided-Faraday-CNN"]


    # args.models = ["E:/models/new4/FaradayModel_20260603_133808"]
    # args.names = ["Faraday-NEW-FIX2"]

    # args.models = ["E:/models/new4/GuidedFaradayModel_20260603_102551",
    #                "E:/models/new4/GuidedFaradayModel_20260603_120241",
    #                "E:/models/new4/GuidedFaradayModel_20260603_123450" ]
    # args.names = ["Guided-Faraday-UserProfiles", 
    #               "Guided-Faraday", 
    #               "Guided-Faraday-CNN"]

    # args.models = ["E:/models/new4/FaradayModel_20260601_060319",]
    # args.names = ["Faraday-new-fix",]

    #args.models = ["E:/models/new3/FaradayModel_20260529_021321", "E:/models/new3/GuidedFaradayModel_20260529_000300"]
    #args.names = ["Faraday-NEW", "Guided-Faraday-NEW"]

    # args.models = ["E:/models/new2/AverageBaselineModel_20260526_234356",
    #     "E:/models/new1/CVAE_20260522_041326_guide_cnn", "E:/models/new1/CVAE_20260522_051533_h_guide_cnn",
    #                "E:/models/new2/CVAE_20260526_154037", "E:/models/new2/FaradayModel_20260526_102007", #"E:/models/new2/GuidedFaradayModel_20260526_110124",
    #                "E:/models/new2/DiffusionModel_20260526_070858", "E:/models/new1/DiffusionModel_20260522_070459", 
    #                #"E:/models/new2/GuidedFaradayModel_20260526_114407", #"E:/models/new2/GuidedFaradayModel_20260526_122401",
    #                 "E:/models/new2/QuantileBaselineModel_20260527_000045", 
    #                "E:/models/new2/CVAE_20260525_233836","E:/models/new2/CVAE_20260526_011125",
    #                 "E:/models/new2/CVAE_20260526_091116",
    #                "E:/models/new2/GuidedFaradayModel_20260527_152449", "E:/models/new2/GuidedFaradayModel_20260527_162635"
    #                #"E:/models/new2/GuidedFaradayModel_20260527_141620",
    #                ]

    # args.names = ["Baseline-Average",
    #     "Guide-CNN", "H-Guide-CNN",
    #               "Guide-Vamp", "Faraday", #"Guided-Faraday", 
    #               "Diffusion", "Diffusion Old",
    #               #"Guided-Faraday-CNN", #"Guided-Faraday-CNN-Deterministic",
    #                "Baseline Quantile",
    #               "H-Guide", "Guide", 
    #                "CVAE", 
    #               "Guided-Faraday-CNN-FIX", "Guided-Faraday"
    #               #"Guided-Faraday-cnn-new", 
    #               ]
    
    # args.models = ["E:/models/new2/FaradayModel_20260526_102007","E:/models/new2/GuidedFaradayModel_20260527_152449", "E:/models/new2/GuidedFaradayModel_20260527_162635"
    #                ]
    # args.names = ["Faraday", "Guided-Faraday-CNN-FIX", "Guided-Faraday"]

    if not args.names:
        raise ValueError("--names is required.")
    if args.models and len(args.models) != len(args.names):
        raise ValueError("--models and --names must have the same length.")
    if not args.models and args.recompute:
        raise ValueError("--models is required when --recompute is set.")
    if not args.models:
        args.models = []
    
    secondary_model_paths = None
    if args.secondary_models is not None and len(args.secondary_models) > 0:
        secondary_model_paths = []
        for s in args.secondary_models:
            if s is None or str(s).lower() == "none":
                secondary_model_paths.append(None)
            else:
                secondary_model_paths.append(s)
        if len(secondary_model_paths) < len(args.models):
            secondary_model_paths.extend([None] * (len(args.models) - len(secondary_model_paths)))

    eval_result_dirs = None
    if args.eval_dirs is not None and len(args.eval_dirs) > 0:
        eval_result_dirs = []
        for d in args.eval_dirs:
            if d is None or str(d).strip() == "" or str(d).lower() == "none":
                eval_result_dirs.append(None)
            else:
                eval_result_dirs.append(d)
        if len(eval_result_dirs) < len(args.models):
            eval_result_dirs.extend([None] * (len(args.models) - len(eval_result_dirs)))

    split_ratios = None
    if args.train_ratio is not None or args.val_ratio is not None or args.test_ratio is not None:
        split_ratios = {
            "train_ratio": 0.8 if args.train_ratio is None else float(args.train_ratio),
            "val_ratio": 0.1 if args.val_ratio is None else float(args.val_ratio),
            "test_ratio": 0.1 if args.test_ratio is None else float(args.test_ratio),
        }

    # args.models = [#"E:/models/new1/CVAE_20260521_234622_guide",
    #                #"E:/models/new1/CVAE_20260522_013003_cvae",  
    #                "E:/models/new1/CVAE_20260522_041326_guide_cnn",
    #                "E:/models/new1/CVAE_20260522_051533_h_guide_cnn", 
    #                "E:/models/new1/QuantileBaselineModel_20260522_034728", 
    #             #    "E:/models/old/Guide Vamp", "E:/models/old/Guide Hierarchy", "E:/models/old/Guide CNN Hierarchy",
    #             #     "E:/models/old/AverageBaselineModel",
    #             #    "E:/models/old/AutoRegressiveTransformerVAE",
    #                "E:/models/new1/DiffusionModel_20260522_070459",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_113725",
    #                "E:/models/new1/FaradayModel_20260522_083016",
    #                "C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_134604",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_143604",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_154532",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_165636",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/CVAE_20260525_175145",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_205135",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_213329",
    #                #"C:/Users/rasmu/Desktop/Thesis/models/new/GuidedFaradayModel_20260525_222016",
    #                "E:/models/new2/CVAE_20260525_233836","E:/models/new2/CVAE_20260526_011125","E:/models/new2/CVAE_20260526_024444",
    #                "E:/models/new2/CVAE_20260526_053436","E:/models/new2/CVAE_20260526_091116","E:/models/new2/DiffusionModel_20260526_070858",
    #                "E:/models/new2/CVAE_20260526_154037", "E:/models/new2/FaradayModel_20260526_102007", "E:/models/new2/GuidedFaradayModel_20260526_110124",
    #                "E:/models/new2/GuidedFaradayModel_20260526_114407", "E:/models/new2/GuidedFaradayModel_20260526_122401",
    #                "E:/models/new2/GuidedFaradayModel_20260527_080406_maybe_tanh", "E:/models/new2/GuidedFaradayModel_20260527_090218",
    #                "E:/models/new2/GuidedFaradayModel_20260527_100405"
    #                ]
    # args.names = [#"Guide", 
    #               #"CVAE", 
    #               "Guide-CNN",
    #                 "H-Guide-CNN", 
    #                 "Quantiles Baseline",
    #             #   "Guide Vamp", "Guide Hierarchy", "Guide CNN Hierarchy",  "Average Baseline",
    #             #     "AutoRegressiveTransformerVAE", 
    #                 "Diffusion",    
    #                 #"Guided-Faraday 2", 
    #                 "Faraday",
    #                 "Guided-Faraday 3", 
    #                 # "Guide CNN", 
    #                 # "Guide CNN 2",
    #                 #"Guided-Faraday 4",
    #                 #"Guide CNN 3",
    #                 #"Guided-Faraday 5",
    #                 #"Guided-Faraday 6",
    #                 #"Guided-Faraday 7",
    #                 "H-Guide-n", "Guide-n", "Guide-CNN-n", 
    #               "H-Guide-CNN-n", "CVAE-n", "Diffusion-n", 
    #               "Guide-Vamp-n", "Faraday-n", "Guided-Faraday-n", 
    #               "Guided-Faraday-CNN-n", "Guided-Faraday-CNN-2-n",
    #               "Guided-Faraday-CNN-3-n", "Guided-Faraday-CNN-TanH-n",
    #               "Guided-Faraday-CNN-4-n-100PDCC"
    #               ]
    # Guided-Faraday 2: used the tanh. 
    # Guided-Faraday 3: used softplus with beta=10. Non-stochastic loss  (2,363,544 params)
    # Guide-Faraday 4: will be stochastic in the loss (see loss function)
    # Guide-Faraday 5: using mu activaton = none, still stochastic in the loss
    #                   Using larger pdcc average max std (4) (2,363,544 params)
    # Guide-Faraday 6: using mu activation = None, still stochastic in the loss, User-Profiles included (2,363,544 params)
    # Guide-Faraday 7: using mu activation = None, non-stochastic, User-profiles (2,363,544 params)
    # Guide-Faraday 8: Will not early stop before 15 patience. Using mu activation = None, stochastic (2,363,544 params)

    # E:/models/new2/GuidedFaradayModel_20260527_141620, cnn concat on input (2,838,640 params)

    # Guide-CNN: used the tanh activation in forward (see git)
    # Guide CNN: used softplus but with beta=1. 
    # Guide CNN 2: used softplus with beta=15. (1,153,153 params)
    # Guide CNN 3: is scaled up to the other models size and used softplus with beta=100 in sampling and None in forward. (2,114,433 params
    # Guide CNN 4: using tanh in forward like back in the day but with actual parameter count (2,114,433 params)

    

    try:
        compare_models(
            model_paths=args.models,
            model_names=args.names,
            secondary_model_paths=secondary_model_paths,
            eval_result_dirs=eval_result_dirs,
            data_dir=args.data_dir,
            data_files=args.data_files,
            num_samples=int(args.num_samples),
            output_dir=args.output_dir,
            phased=bool(args.phased),
            recompute=bool(args.recompute),
            split_seed=args.split_seed,
            split_ratios=split_ratios,
            max_value=args.max_value,
            agg_divisor=1000.0 if args.phased else 1.0,
            holdout_n_draws=holdout_n_draws,
            n_draws=n_draws,
            recompute_tstr=bool(args.recompute_tstr),
            mia_model_names=args.mia_models if args.mia_models else None,
            recompute_visual=bool(args.recompute_visual) if hasattr(args, "recompute_visual") else False,
            sort_models=not bool(args.no_sort),
            only_holdout=bool(args.only_holdout) if hasattr(args, "only_holdout") else False,
            ev_hp_breakdown=bool(args.ev_hp_breakdown) if hasattr(args, "ev_hp_breakdown") else False,
            ev_random_user_profiles=bool(args.ev_random_user_profiles) if hasattr(args, "ev_random_user_profiles") else False,
            preprocessed_cache_dir=args.cache_dir if hasattr(args, "cache_dir") and args.cache_dir else None,
            only_cross_phase=bool(args.only_cross_phase) if hasattr(args, "only_cross_phase") else False,
            cross_phase_samples=int(args.cross_phase_samples) if hasattr(args, "cross_phase_samples") and args.cross_phase_samples else None,
            only_c2st=bool(args.only_c2st) if hasattr(args, "only_c2st") else False,
            c2st_samples=int(args.c2st_samples) if hasattr(args, "c2st_samples") and args.c2st_samples else None,
        )
    except BaseException as _e:
        import traceback as _tb
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL: {type(_e).__name__}: {_e}", flush=True)
        print("="*60, flush=True)
        _tb.print_exc()
        _tb.print_exc(file=_crash_log)
        _crash_log.flush()
        raise

# Example run in CLI: 
# python src/evaluation/compare_models.py \
#     --models models/hpc_runs/AutoRegressiveTransformerVAE_20260424_170101 \
#     --names art \
#     --eval-dirs models/hpc_runs/AutoRegressiveTransformerVAE_20260424_170101/rerun_evaluation_20260511_101040 \
#     --phased