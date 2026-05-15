from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class EncoderConfig(BaseModel):
    model_config = {"extra": "allow"}
    type: str
    hidden_dims: list[int]
    dropout: float = 0.0


class DecoderConfig(BaseModel):
    model_config = {"extra": "allow"}
    type: str
    hidden_dims: list[int]
    dropout: float = 0.0


class PriorConfig(BaseModel):
    model_config = {"extra": "allow"}
    type: str


class HierarchyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool = False


class ArtConfig(BaseModel):
    model_config = {"extra": "allow"}
    num_bins: int
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    d_model: int = 64


class SetConfig(BaseModel):
    model_config = {"extra": "allow"}
    hidden_dim: int = 256
    num_heads: int = 4


class _VAEBase(BaseModel):
    latent_dim: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    prior: PriorConfig
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)


class VaeModelConfig(_VAEBase):
    type: Literal["vae"]


class CvaeModelConfig(_VAEBase):
    type: Literal["cvae"]


class ArtModelConfig(_VAEBase):
    type: Literal["art"]
    art: ArtConfig


class SetGuideVaeModelConfig(_VAEBase):
    type: Literal["set_guide_vae"]
    model_config = {"extra": "allow"}


class SequentialVaeModelConfig(_VAEBase):
    type: Literal["sequential_vae"]
    model_config = {"extra": "allow"}


class DiffusionConfig(BaseModel):
    model_config = {"extra": "allow"}
    num_timesteps: int
    schedule_type: str = "cosine"


class DiffusionModelConfig(BaseModel):
    type: Literal["diffusion"]
    diffusion: DiffusionConfig


class PhaseSplitterConfig(BaseModel):
    model_config = {"extra": "allow"}
    base_channels: int = 64
    temperature: float = 2.0


class PhaseSplitterModelConfig(BaseModel):
    type: Literal["phase_splitter"]
    phase_splitter: PhaseSplitterConfig = Field(default_factory=PhaseSplitterConfig)


class BaselineAverageModelConfig(BaseModel):
    type: Literal["baseline_average"]


class BaselineQuantileModelConfig(BaseModel):
    type: Literal["baseline_quantile"]


class BaselineMeanModelConfig(BaseModel):
    type: Literal["baseline_mean"]


ModelConfig = Annotated[
    Union[
        ArtModelConfig,
        VaeModelConfig,
        CvaeModelConfig,
        SetGuideVaeModelConfig,
        SequentialVaeModelConfig,
        DiffusionModelConfig,
        PhaseSplitterModelConfig,
        BaselineAverageModelConfig,
        BaselineQuantileModelConfig,
        BaselineMeanModelConfig,
    ],
    Field(discriminator="type"),
]


class ExperimentConfig(BaseModel):
    model_config = {"extra": "allow"}
    model: ModelConfig

    @model_validator(mode="before")
    @classmethod
    def _migrate_top_level_hierarchy(cls, values: dict) -> dict:
        if "hierarchy" in values:
            import warnings
            warnings.warn(
                "Top-level 'hierarchy' key is deprecated. Move it inside 'model.hierarchy' "
                "(VAE-family only). It has been auto-migrated for this run.",
                DeprecationWarning,
                stacklevel=2,
            )
            hierarchy = values.pop("hierarchy")
            model_section = values.get("model", {})
            if isinstance(model_section, dict) and "hierarchy" not in model_section:
                model_section = {**model_section, "hierarchy": hierarchy}
                values = {**values, "model": model_section}
        return values