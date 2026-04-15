"""Count model parameters for a given YAML configuration.

This script instantiates the architecture described by a config file and prints
the number of total, trainable, and frozen parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

# Keep imports consistent with the existing codebase style in src/train.py.
for _p in (SRC_DIR, ROOT_DIR):
	if _p not in sys.path:
		sys.path.insert(0, _p)

from data.conditioner import (  # noqa: E402
	ActivePhaseCountConditioner,
	Conditioner,
	DayOfWeekConditioner,
	HeatpumpPhaseConditioner,
	HolidayConditioner,
	MonthOfYearConditioner,
	UserProfiler,
)
from src.model.art import AutoRegressiveTransformerVAE  # noqa: E402
from src.model.cvae import CVAE  # noqa: E402
from src.model.diffusion import (  # noqa: E402
	ConditionalUNet1D,
	DiffusionModel,
	NoiseScheduler,
)
from src.model.hierarchical.config import (  # noqa: E402
	HierarchicalConfig,
	LevelConfig,
)
from src.model.phase_splitter import HierarchicalSplitter  # noqa: E402
from src.model.set import SetGuideVAE  # noqa: E402
from src.model.vae import VAE  # noqa: E402


class _StaticConditionItem:
	"""Fallback condition item used when external assets are unavailable."""

	def __init__(self, condition_dim: int):
		self.condition_dim = int(condition_dim)

	def get_conditions(self, x=None, meter_id=None, timestamps=None):
		device = torch.device("cpu")
		batch_size = 1

		if isinstance(x, torch.Tensor):
			device = x.device
			batch_size = int(x.shape[0]) if x.dim() >= 1 else 1
		elif isinstance(meter_id, torch.Tensor):
			device = meter_id.device
			batch_size = int(meter_id.shape[0]) if meter_id.dim() >= 1 else 1
		elif isinstance(timestamps, torch.Tensor):
			device = timestamps.device
			batch_size = int(timestamps.shape[0]) if timestamps.dim() >= 1 else 1

		return torch.zeros((batch_size, self.condition_dim), dtype=torch.float32, device=device)

	def get_random_conditions(self, batch_size: int):
		return torch.zeros((int(batch_size), self.condition_dim), dtype=torch.float32)


def _warn(msg: str) -> None:
	print(f"Warning: {msg}", file=sys.stderr)


def _to_namespace(obj):
	if isinstance(obj, dict):
		return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
	if isinstance(obj, list):
		return [_to_namespace(v) for v in obj]
	return obj


def _load_config(config_path: str):
	with open(config_path, "r", encoding="utf-8") as f:
		raw = yaml.safe_load(f) or {}

	cfg = _to_namespace(raw)
	cfg._config_path = os.path.abspath(config_path)

	if not hasattr(cfg, "data"):
		raise ValueError("Config must contain a top-level 'data' section")
	if not hasattr(cfg, "model"):
		raise ValueError("Config must contain a top-level 'model' section")
	if not hasattr(cfg.data, "window"):
		raise ValueError("Config must contain 'data.window' section")

	cfg.window = cfg.data.window
	cfg.columns = getattr(cfg.data, "columns", SimpleNamespace(value_cols=[]))
	cfg.model_type = str(getattr(cfg.model, "type", "vae")).lower()

	return cfg


def _get_list(cfg_list, n: int, default):
	values = list(cfg_list) if cfg_list is not None else []
	if len(values) < n:
		values.extend([default] * (n - len(values)))
	return values[:n]


def _build_hierarchy_config(config) -> HierarchicalConfig | None:
	hierarchy_cfg = getattr(config, "hierarchy", SimpleNamespace(enabled=False))
	if not bool(getattr(hierarchy_cfg, "enabled", False)):
		return None

	level_types = list(getattr(hierarchy_cfg, "levels", []))
	latent_dims = [int(v) for v in list(getattr(hierarchy_cfg, "latent_dims", []))]
	if len(level_types) == 0:
		raise ValueError("hierarchy.enabled=true but no hierarchy.levels were provided")
	if len(latent_dims) != len(level_types):
		raise ValueError("hierarchy.latent_dims must have same length as hierarchy.levels")

	top_down = [bool(v) for v in _get_list(getattr(hierarchy_cfg, "top_down", []), len(level_types), False)]
	bottom_up = [bool(v) for v in _get_list(getattr(hierarchy_cfg, "bottom_up", []), len(level_types), False)]
	loss_weights = [float(v) for v in _get_list(getattr(hierarchy_cfg, "loss_weights", []), len(level_types), 1.0)]

	levels = []
	for i, level_type in enumerate(level_types):
		input_dim = int(config.model.latent_dim) if i == 0 else int(latent_dims[i - 1])
		levels.append(
			LevelConfig(
				name=f"Level {i + 1}",
				strategy=str(level_type),
				input_dim=input_dim,
				latent_dim=int(latent_dims[i]),
				top_down=top_down[i],
				bottom_up=bottom_up[i],
				loss_weight=loss_weights[i],
			)
		)

	return HierarchicalConfig(
		levels=levels,
		aggregation=str(getattr(hierarchy_cfg, "aggregation", "concat")),
		bidirectional=bool(getattr(hierarchy_cfg, "bidirectional", False)),
		residual_connections=bool(getattr(hierarchy_cfg, "residual_connections", False)),
		kl_annealing=bool(getattr(hierarchy_cfg, "kl_annealing", True)),
		kl_warmup_steps=int(getattr(hierarchy_cfg, "kl_warmup_steps", 1000)),
		free_bits=float(getattr(hierarchy_cfg, "free_bits", 0.0)),
	)


def _safe_add_condition(conditioner: Conditioner, tag: str, factory, fallback_dim: int):
	try:
		item = factory()
	except Exception as exc:
		_warn(f"Could not initialize condition tag '{tag}' ({exc}). Using static dim={fallback_dim} fallback.")
		item = _StaticConditionItem(condition_dim=fallback_dim)
	conditioner.add_condition_item(tag, item)


def _build_conditioner(config) -> Conditioner:
	conditioner = Conditioner()
	cond_cfg = getattr(config, "conditioning", SimpleNamespace(tags=[]))
	tags = list(getattr(cond_cfg, "tags", []) or [])

	for tag in tags:
		if tag == "day_of_week":
			conditioner.add_condition_item("day_of_week", DayOfWeekConditioner())
		elif tag == "month_of_year":
			conditioner.add_condition_item("month_of_year", MonthOfYearConditioner())
		elif tag == "holiday":
			holiday_path = "data/holidays/denmark_holidays.json"
			_safe_add_condition(
				conditioner,
				"holiday",
				lambda: HolidayConditioner(holiday_path=holiday_path),
				fallback_dim=1,
			)
		elif tag == "nr_phases_active":
			conditioner.add_condition_item("nr_phases_active", ActivePhaseCountConditioner())
		elif tag == "heatpump_phase":
			hp_path = "data/HP/label_propagation_predictions.csv"
			_safe_add_condition(
				conditioner,
				"heatpump_phase",
				lambda: HeatpumpPhaseConditioner(topology_path=hp_path),
				fallback_dim=4,
			)
		elif tag == "user_profile":
			user_profile_cfg = getattr(cond_cfg, "user_profile", SimpleNamespace())
			online_cfg = getattr(cond_cfg, "online_user_profile", SimpleNamespace(decay=1.0))

			online = bool(getattr(user_profile_cfg, "online", False))
			decay = float(getattr(online_cfg, "decay", 0.95)) if online else 1.0

			conditioner.add_condition_item(
				"user_profile",
				UserProfiler(
					n_topics=int(getattr(user_profile_cfg, "n_topics", 10)),
					num_clusters=int(getattr(user_profile_cfg, "num_clusters", 100)),
					random_state=getattr(user_profile_cfg, "random_state", None),
					path=str(getattr(user_profile_cfg, "cache_path", "./models/UserProfiles/user_profiles.pkl")),
					verbose=bool(getattr(user_profile_cfg, "verbose", False)),
					decay=decay,
				),
			)
		else:
			_warn(f"Unknown condition tag '{tag}' in config. Skipping.")

	return conditioner


def _resolve_input_dimensions(config, model_type: str):
	data_mode = str(getattr(config.data, "mode", "csv")).lower()
	phased_mode = data_mode in {"phased", "phase", "parquet", "phased_parquet"}

	csv_num_features = len(getattr(config.columns, "value_cols", []) or [])
	set_cfg = getattr(config.model, "set", SimpleNamespace(num_features=None))

	if phased_mode and model_type in {"set_guide_vae", "art", "diffusion", "phase_splitter"}:
		configured = getattr(set_cfg, "num_features", None)
		num_features = int(configured) if configured is not None else 3
	else:
		if csv_num_features < 1:
			raise ValueError(
				"Could not infer number of features from config.data.columns.value_cols. "
				"Please provide at least one value column."
			)
		num_features = int(csv_num_features)

	input_len = int(getattr(config.window, "input_len", 24))
	input_dim = int(input_len * num_features)

	return {
		"input_len": input_len,
		"num_features": num_features,
		"input_dim": input_dim,
		"phased_mode": phased_mode,
	}


def build_model_from_config(config):
	model_type = str(getattr(config.model, "type", "vae")).lower()
	hierarchy = _build_hierarchy_config(config)

	if model_type == "diffusion" and hierarchy is not None:
		raise ValueError("DiffusionModel currently does not support hierarchy.enabled=true.")

	dims = _resolve_input_dimensions(config, model_type=model_type)
	input_dim = int(dims["input_dim"])
	input_len = int(dims["input_len"])
	num_features = int(dims["num_features"])

	prior_cfg = getattr(config.model, "prior", SimpleNamespace())
	encoder_cfg = getattr(config.model, "encoder", SimpleNamespace(type="dictionary"))
	decoder_cfg = getattr(config.model, "decoder", SimpleNamespace(type="dictionary"))
	model_arch_kwargs = {
		"encoder_hidden_dims": list(getattr(encoder_cfg, "hidden_dims", []) or []),
		"encoder_dropout": getattr(encoder_cfg, "dropout", None),
		"decoder_hidden_dims": list(getattr(decoder_cfg, "hidden_dims", []) or []),
		"decoder_dropout": getattr(decoder_cfg, "dropout", None),
	}
	latent_dim = int(getattr(config.model, "latent_dim", 32))
	prior_type = getattr(prior_cfg, "type", None)
	dictionary_size = int(getattr(prior_cfg, "dictionary_size", 100))
	vamp_components = int(getattr(prior_cfg, "vamp_num_components", 20))

	conditioner = None
	if model_type in {"cvae", "set_guide_vae", "art", "diffusion", "phase_splitter"}:
		conditioner = _build_conditioner(config)

	if model_type == "vae":
		model = VAE(
			input_dim=input_dim,
			latent_dim=latent_dim,
			encoder_type=str(getattr(encoder_cfg, "type", "dictionary")),
			decoder_type=str(getattr(decoder_cfg, "type", "dictionary")),
			prior_type=prior_type,
			dictionary_size=dictionary_size,
			vamp_num_components=vamp_components,
			hierarchy_config=hierarchy,
			**model_arch_kwargs,
		)
	elif model_type == "cvae":
		model = CVAE(
			input_dim=input_dim,
			latent_dim=latent_dim,
			conditioner=conditioner,
			encoder_type=str(getattr(encoder_cfg, "type", "dictionary")),
			decoder_type=str(getattr(decoder_cfg, "type", "dictionary")),
			prior_type=prior_type,
			dictionary_size=dictionary_size,
			vamp_num_components=vamp_components,
			hierarchy_config=hierarchy,
			**model_arch_kwargs,
		)
	elif model_type == "set_guide_vae":
		set_cfg = getattr(config.model, "set", SimpleNamespace())
		model = SetGuideVAE(
			input_dim=input_dim,
			latent_dim=latent_dim,
			conditioner=conditioner,
			encoder_type=str(getattr(encoder_cfg, "type", "dictionary")),
			decoder_type=str(getattr(decoder_cfg, "type", "dictionary")),
			prior_type=prior_type,
			dictionary_size=dictionary_size,
			vamp_num_components=vamp_components,
			hierarchy_config=hierarchy,
			sequence_length=input_len,
			num_features=num_features,
			set_hidden_dim=int(getattr(set_cfg, "hidden_dim", 128)),
			set_num_heads=int(getattr(set_cfg, "num_heads", 4)),
			set_num_inds=int(getattr(set_cfg, "num_inds", 16)),
			set_num_blocks=int(getattr(set_cfg, "num_blocks", 2)),
			is_isab=bool(getattr(set_cfg, "is_isab", True)),
			use_abl_pool=bool(getattr(set_cfg, "use_abl_pool", False)),
			**model_arch_kwargs,
		)
	elif model_type == "art":
		art_cfg = getattr(config.model, "art", SimpleNamespace())
		model = AutoRegressiveTransformerVAE(
			input_dim=input_dim,
			latent_dim=latent_dim,
			conditioner=conditioner,
			encoder_type=str(getattr(encoder_cfg, "type", "dictionary")),
			decoder_type=str(getattr(decoder_cfg, "type", "dictionary")),
			prior_type=prior_type,
			dictionary_size=dictionary_size,
			vamp_num_components=vamp_components,
			hierarchy_config=hierarchy,
			bin_size=int(getattr(art_cfg, "num_bins", 64)),
			nhead=int(getattr(art_cfg, "transformer_num_heads", 4)),
			num_layers=int(getattr(art_cfg, "transformer_num_layers", 4)),
			d_model=int(getattr(art_cfg, "d_model", 256)),
			sequence_length=input_len,
			num_features=num_features,
			**model_arch_kwargs,
		)
	elif model_type == "diffusion":
		diff_cfg = getattr(config.model, "diffusion", SimpleNamespace())
		condition_dim = int(conditioner.get_condition_dim())

		scheduler = NoiseScheduler(
			num_timesteps=int(getattr(diff_cfg, "num_timesteps", 1000)),
			beta_start=float(getattr(diff_cfg, "beta_start", 1.0e-4)),
			beta_end=float(getattr(diff_cfg, "beta_end", 0.02)),
			schedule_type=str(getattr(diff_cfg, "schedule_type", "linear")),
			cosine_s=float(getattr(diff_cfg, "cosine_s", 0.008)),
			max_beta=float(getattr(diff_cfg, "max_beta", 0.999)),
			device="cpu",
		)

		denoiser = ConditionalUNet1D(
			channels=num_features,
			condition_dim=condition_dim,
			time_dim=int(getattr(diff_cfg, "time_dim", 128)),
			base_channels=int(getattr(diff_cfg, "base_channels", 128)),
			num_blocks=int(getattr(diff_cfg, "num_blocks", 4)),
			self_attention_heads=int(getattr(diff_cfg, "self_attention_heads", 4)),
			use_condition_cross_attention=bool(getattr(diff_cfg, "use_condition_cross_attention", False)),
			condition_attention_heads=int(
				getattr(
					diff_cfg,
					"condition_cross_attention_heads",
					getattr(diff_cfg, "condition_attention_heads", 4),
				)
			),
			condition_tokens=int(getattr(diff_cfg, "condition_tokens", 4)),
		)

		model = DiffusionModel(
			noise_scheduler=scheduler,
			model=denoiser,
			conditioner=conditioner,
			sequence_length=input_len,
			num_features=num_features,
			condition_dropout_prob=float(getattr(diff_cfg, "condition_dropout_prob", 0.0)),
			reconstruction_noise_fraction=float(getattr(diff_cfg, "reconstruction_noise_fraction", 0.3)),
			deterministic_reconstruction=bool(getattr(diff_cfg, "deterministic_reconstruction", True)),
		)
	elif model_type == "phase_splitter":
		model = HierarchicalSplitter(
			conditioner=conditioner,
			sequence_length=input_len,
			condition_dim=int(conditioner.get_condition_dim()),
		)
	else:
		raise ValueError(f"Unknown model type '{model_type}' in config")

	metadata = {
		"model_type": model_type,
		"input_dim": input_dim,
		"input_len": input_len,
		"num_features": num_features,
		"condition_dim": int(conditioner.get_condition_dim()) if conditioner is not None else 0,
		"phased_mode": bool(dims["phased_mode"]),
	}
	return model, metadata


def count_parameters(model: torch.nn.Module):
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen = total - trainable
	return total, trainable, frozen


def module_trainable_breakdown(model: torch.nn.Module):
	rows = []
	for name, module in model.named_modules():
		if name == "":
			continue
		params = list(module.parameters(recurse=False))
		if len(params) == 0:
			continue
		total = sum(p.numel() for p in params)
		trainable = sum(p.numel() for p in params if p.requires_grad)
		if total == 0:
			continue
		rows.append({"module": name, "total": total, "trainable": trainable})

	rows.sort(key=lambda r: r["trainable"], reverse=True)
	return rows


def _fmt_int(value: int) -> str:
	return f"{int(value):,}"


def _default_config_path() -> str:
	return os.path.join(ROOT_DIR, "config", "config.yaml")


def main():
	parser = argparse.ArgumentParser(description="Count model parameters from YAML config")
	parser.add_argument(
		"--config",
		type=str,
		default=_default_config_path(),
		help="Path to YAML config file",
	)
	parser.add_argument(
		"--detailed",
		action="store_true",
		default=False,
		help="Print per-module trainable parameter breakdown",
	)
	parser.add_argument(
		"--top-modules",
		type=int,
		default=25,
		help="Number of modules to show in --detailed output",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		default=False,
		help="Print output as JSON",
	)
	args = parser.parse_args()

	config_path = os.path.abspath(args.config)
	config = _load_config(config_path)
	model, metadata = build_model_from_config(config)

	total, trainable, frozen = count_parameters(model)

	payload = {
		"config_path": config_path,
		"model_class": model.__class__.__name__,
		"metadata": metadata,
		"parameters": {
			"total": int(total),
			"trainable": int(trainable),
			"frozen": int(frozen),
		},
	}

	if args.detailed:
		payload["module_breakdown"] = module_trainable_breakdown(model)

	if args.json:
		print(json.dumps(payload, indent=2))
		return

	print("=== Model Parameter Count ===")
	print(f"Config:         {payload['config_path']}")
	print(f"Model class:    {payload['model_class']}")
	print(f"Model type:     {metadata['model_type']}")
	print(f"Input dim:      {_fmt_int(metadata['input_dim'])}")
	print(f"Input length:   {_fmt_int(metadata['input_len'])}")
	print(f"Num features:   {_fmt_int(metadata['num_features'])}")
	print(f"Condition dim:  {_fmt_int(metadata['condition_dim'])}")
	print(f"Phased mode:    {metadata['phased_mode']}")
	print("")
	print(f"Total params:      {_fmt_int(total)}")
	print(f"Trainable params:  {_fmt_int(trainable)}")
	print(f"Frozen params:     {_fmt_int(frozen)}")

	if args.detailed:
		rows = payload.get("module_breakdown", [])[: max(1, int(args.top_modules))]
		print("")
		print(f"Top {len(rows)} modules by trainable parameters")
		for row in rows:
			print(
				f"  - {row['module']}: trainable={_fmt_int(row['trainable'])}, "
				f"total={_fmt_int(row['total'])}"
			)


if __name__ == "__main__":
	main()

