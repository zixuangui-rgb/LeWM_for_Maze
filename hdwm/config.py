"""Pydantic configuration models for Hydra entrypoints."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigModel(BaseModel):
    """Base config model for Hydra dictionaries."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class StrictConfigModel(BaseModel):
    """Base config model for sections that should reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectionStage(str, Enum):
    """Stage in projection pipeline to select for output.

    Pipeline: x1 -> LN -> x2 -> BN -> x3 -> L2 -> x4
    """

    PRE_NORM = "pre_norm"
    POST_LN = "post_ln"
    POST_BN = "post_bn"
    POST_L2 = "post_l2"


class TemporalPositionEncoding(str, Enum):
    """Temporal position encoding used by model sequence transformers."""

    ABSOLUTE = "absolute"
    ROTARY = "rotary"


class BatchSampleStrategy(str, Enum):
    """How same-map and different-map data batches are sampled."""

    SAME_WITHIN_BATCH = "same_within_batch"
    DIFFERENT_WITHIN_BATCH = "different_within_batch"


ContextSampleStrategy = BatchSampleStrategy


class ObsCMINegativeSource(str, Enum):
    """Negative source for BWM observation CMI."""

    BATCH = "batch"
    NOISE = "noise"


class PriorCMINegativeSource(str, Enum):
    """Negative source for BWM prior CMI."""

    ONE_HOT = "one_hot"
    NOISE = "noise"


class BWMPriorForm(str, Enum):
    """How the BWM predictor prior conditions the target branch."""

    ADALN = "adaln"
    OBSERVATION_SOFTMAX_MASK = "observation_softmax_mask"
    FEATURE_SOFTMAX_MULTIPLIER = "feature_softmax_multiplier"


class RingWorldAction(IntEnum):
    """Readable ring-world action meanings."""

    LEFT = 0
    STAY = 1
    RIGHT = 2


class GridWorld2DAction(IntEnum):
    """Readable 2D grid-world action meanings."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def _omegaconf_to_dict(cfg: DictConfig) -> dict[str, Any]:
    # Resolve Hydra interpolations before Pydantic validation so runtime code
    # only sees plain Python values; Hydra metadata such as `_target_` is
    # ignored by ConfigModel.
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    return dict(container)


class RingWorldConfig(ConfigModel):
    """Validated ring-world config loaded from Hydra."""

    type: Literal["ring_world"] = "ring_world"
    length: int = Field(default=16, gt=1)
    p_noop: float = Field(default=0.1, ge=0.0, le=1.0)
    p_noise: float = Field(default=0.2, ge=0.0, le=1.0)
    p_action_turn: float = Field(default=0.1, ge=0.0, le=1.0)
    p_action_stay: float = Field(default=0.05, ge=0.0, le=1.0)
    num_actions: int = Field(default=3, gt=0)
    observation_dtype: str = "float32"
    max_episode_steps: int | None = Field(default=None, gt=0)
    render_mode: Literal["ansi", "human"] | None = "ansi"

    @model_validator(mode="after")
    def validate_observation_dtype(self) -> RingWorldConfig:
        if not hasattr(np, self.observation_dtype):
            raise ValueError(f"unsupported numpy dtype: {self.observation_dtype}")
        if self.num_actions != len(RingWorldAction):
            raise ValueError("num_actions must match RingWorldAction")
        return self

    @property
    def numpy_observation_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.observation_dtype)

    @property
    def observation_size(self) -> int:
        return self.length

    @property
    def observation_spatial_shape(self) -> tuple[int]:
        return (self.length,)

    @property
    def observation_channels(self) -> int:
        return 1

    @property
    def action_vocab_size(self) -> int:
        return self.num_actions


class GridNoisePlacement(str, Enum):
    """Where observation distractor noise may be placed in a 2D grid."""

    EMPTY = "empty"
    EMPTY_AND_OBSTACLE = "empty_and_obstacle"


class GridWorld2DConfig(ConfigModel):
    """Validated 2D grid-world config loaded from Hydra."""

    type: Literal["grid_world_2d"] = "grid_world_2d"
    height: int = Field(default=8, gt=1)
    width: int = Field(default=8, gt=1)
    p_obstacle: float = Field(default=0.1, ge=0.0, le=1.0)
    obstacles: tuple[int, ...] = ()
    resample_obstacles_per_sequence: bool = True
    train_virtual_border: tuple[int, int, int, int] | None = None
    validation_virtual_border: tuple[int, int, int, int] | None = None
    virtual_border_pass_through: float = Field(default=0.95, ge=0.0, le=1.0)
    p_noop: float = Field(default=0.1, ge=0.0, le=1.0)
    p_noise: float = Field(default=0.2, ge=0.0, le=1.0)
    p_action_turn: float = Field(default=0.1, ge=0.0, le=1.0)
    p_action_stay: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_placement: GridNoisePlacement = GridNoisePlacement.EMPTY
    num_actions: int = Field(default=5, gt=0)
    observation_dtype: str = "float32"
    max_episode_steps: int | None = Field(default=None, gt=0)
    render_mode: Literal["ansi", "human"] | None = "ansi"

    @property
    def observation_size(self) -> int:
        return self.height * self.width

    @model_validator(mode="after")
    def validate_grid_config(self) -> GridWorld2DConfig:
        expected_size = self.height * self.width
        if not hasattr(np, self.observation_dtype):
            raise ValueError(f"unsupported numpy dtype: {self.observation_dtype}")
        if any(
            position < 0 or position >= expected_size for position in self.obstacles
        ):
            raise ValueError(f"obstacles must be in [0, {expected_size - 1}]")
        if self.num_actions != len(GridWorld2DAction):
            raise ValueError("num_actions must match GridWorld2DAction")
        if (self.train_virtual_border is None) != (
            self.validation_virtual_border is None
        ):
            raise ValueError(
                "train_virtual_border and validation_virtual_border "
                "must be set together"
            )
        if self.train_virtual_border is not None:
            self._validate_virtual_border(
                self.train_virtual_border,
                name="train_virtual_border",
            )
            self._validate_virtual_border(
                self.validation_virtual_border,
                name="validation_virtual_border",
            )
            if self._virtual_borders_overlap(
                self.train_virtual_border,
                self.validation_virtual_border,
            ):
                raise ValueError(
                    "train_virtual_border and validation_virtual_border "
                    "must not overlap"
                )
        return self

    def _validate_virtual_border(
        self,
        border: tuple[int, int, int, int] | None,
        name: str,
    ) -> None:
        if border is None:
            raise ValueError(f"{name} must be set")
        top, left, bottom, right = border
        if not (0 <= top < bottom <= self.height):
            raise ValueError(f"{name} rows must satisfy 0 <= top < bottom <= height")
        if not (0 <= left < right <= self.width):
            raise ValueError(f"{name} columns must satisfy 0 <= left < right <= width")

    @staticmethod
    def _virtual_borders_overlap(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int] | None,
    ) -> bool:
        if second is None:
            raise ValueError("second virtual border must be set")
        first_top, first_left, first_bottom, first_right = first
        second_top, second_left, second_bottom, second_right = second
        return max(first_top, second_top) < min(first_bottom, second_bottom) and max(
            first_left, second_left
        ) < min(first_right, second_right)

    @property
    def numpy_observation_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.observation_dtype)

    @property
    def observation_spatial_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def observation_channels(self) -> int:
        return 3

    @property
    def action_vocab_size(self) -> int:
        return self.num_actions


class IceWorld2DConfig(GridWorld2DConfig):
    """Validated ice-world config loaded from Hydra."""

    type: Literal["ice_world_2d"] = "ice_world_2d"
    p_action_turn: float = Field(default=0.85, ge=0.0, le=1.0)
    p_action_stay: float = Field(default=0.0, ge=0.0, le=1.0)


class FourRoomsConfig(ConfigModel):
    """Validated FourRooms environment config loaded from Hydra.

    A 2D grid divided into 4 rooms by cross-shaped walls, connected by
    doorways whose positions can vary randomly per sequence.
    """

    type: Literal["four_rooms"] = "four_rooms"
    size: int = Field(default=11, gt=4)
    doorway_size: int = Field(default=1, gt=0)
    doorway_offset_min: int = Field(default=1, ge=0)
    doorway_offset_max: int = Field(default=4, ge=1)
    asymmetric: bool = False
    p_noop: float = Field(default=0.0, ge=0.0, le=1.0)
    p_noise: float = Field(default=0.1, ge=0.0, le=1.0)
    p_action_turn: float = Field(default=0.1, ge=0.0, le=1.0)
    p_action_stay: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_placement: GridNoisePlacement = GridNoisePlacement.EMPTY
    num_actions: int = Field(default=5, gt=0)
    observation_dtype: str = "float32"
    max_episode_steps: int | None = Field(default=None, gt=0)
    render_mode: Literal["ansi", "human"] | None = "ansi"
    resample_per_sequence: bool = True

    @model_validator(mode="after")
    def validate(self) -> FourRoomsConfig:
        if not hasattr(np, self.observation_dtype):
            raise ValueError(f"unsupported numpy dtype: {self.observation_dtype}")
        if self.num_actions != len(GridWorld2DAction):
            raise ValueError("num_actions must match GridWorld2DAction")
        if self.doorway_offset_min < 0:
            raise ValueError("doorway_offset_min must be >= 0")
        if self.doorway_offset_max + self.doorway_size > self.size // 2:
            raise ValueError("doorway_offset_max + doorway_size must be <= size/2")
        return self

    @property
    def numpy_observation_dtype(self):
        return np.dtype(self.observation_dtype)

    @property
    def height(self) -> int:
        return self.size

    @property
    def width(self) -> int:
        return self.size

    @property
    def observation_size(self) -> int:
        return self.size * self.size

    @property
    def observation_spatial_shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def observation_channels(self) -> int:
        return 3

    @property
    def action_vocab_size(self) -> int:
        return self.num_actions


class ProcgenMazeConfig(ConfigModel):
    """Validated procgen-maze config loaded from Hydra."""

    type: Literal["procgen_maze"] = "procgen_maze"
    height: int = Field(default=9, gt=2)
    width: int = Field(default=9, gt=2)
    observation_channels: int = Field(default=5, gt=0)
    p_noop: float = Field(default=0.1, ge=0.0, le=1.0)
    p_noise: float = Field(default=0.2, ge=0.0, le=1.0)
    p_action_turn: float = Field(default=0.1, ge=0.0, le=1.0)
    p_action_stay: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_placement: GridNoisePlacement = GridNoisePlacement.EMPTY
    num_actions: int = Field(default=5, gt=0)
    observation_dtype: str = "float32"
    max_episode_steps: int | None = Field(default=None, gt=0)
    render_mode: Literal["ansi", "human"] | None = "ansi"
    resample_maze_per_sequence: bool = True
    topology_seed: int | None = Field(default=None)
    walls: tuple[int, ...] = ()
    train_virtual_border: tuple[int, int, int, int] | None = None
    validation_virtual_border: tuple[int, int, int, int] | None = None
    virtual_border_pass_through: float = Field(default=0.95, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_maze_config(self) -> ProcgenMazeConfig:
        if not hasattr(np, self.observation_dtype):
            raise ValueError(f"unsupported numpy dtype: {self.observation_dtype}")
        if self.num_actions != len(GridWorld2DAction):
            raise ValueError("num_actions must match GridWorld2DAction")
        expected_size = self.width * self.height
        if any(pos < 0 or pos >= expected_size for pos in self.walls):
            raise ValueError(f"walls must be in [0, {expected_size - 1}]")
        return self

    @property
    def numpy_observation_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.observation_dtype)

    @property
    def observation_size(self) -> int:
        return self.height * self.width

    @property
    def observation_spatial_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def action_vocab_size(self) -> int:
        return self.num_actions


EnvConfig = (
    RingWorldConfig
    | IceWorld2DConfig
    | GridWorld2DConfig
    | ProcgenMazeConfig
    | FourRoomsConfig
)


def _coerce_env_config(value: Any) -> EnvConfig:
    if isinstance(
        value,
        (
            RingWorldConfig,
            IceWorld2DConfig,
            GridWorld2DConfig,
            ProcgenMazeConfig,
            FourRoomsConfig,
        ),
    ):
        return value
    if isinstance(value, dict):
        env_type = value.get("type")
        if env_type == "four_rooms":
            return FourRoomsConfig.model_validate(value)
        if env_type == "procgen_maze":
            return ProcgenMazeConfig.model_validate(value)
        if env_type == "ice_world_2d":
            return IceWorld2DConfig.model_validate(value)
        if env_type == "grid_world_2d" or "height" in value or "width" in value:
            return GridWorld2DConfig.model_validate(value)
        return RingWorldConfig.model_validate(value)
    return value


class EncoderModelConfig(ConfigModel):
    """Shared observation encoder config for world-model baselines."""

    env_config: EnvConfig
    latent_dim: int = Field(default=64, gt=0)
    model_dim: int | None = Field(default=None, gt=0)
    latent_batch_norm: bool = True
    latent_layer_norm: bool = False
    latent_l2_norm: bool = False
    latent_batch_norm_affine: bool = True
    embedding_stage: ProjectionStage = ProjectionStage.POST_BN
    sigreg_stage: ProjectionStage = ProjectionStage.POST_BN
    latent_temporal_fusion_enabled: bool = False
    latent_temporal_context_window: int | None = Field(default=None, ge=0)
    temporal_position_encoding: TemporalPositionEncoding = (
        TemporalPositionEncoding.ABSOLUTE
    )
    encoder_layers: int = Field(default=1, gt=0)
    encoder_heads: int = Field(default=4, gt=0)
    mlp_ratio: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    max_sequence_length: int = Field(default=128, gt=0)

    @model_validator(mode="before")
    @classmethod
    def set_legacy_observation_size_env(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "env_config" in data or "observation_size" not in data:
            return data
        resolved = dict(data)
        spatial_shape = resolved.get("observation_spatial_shape")
        if isinstance(spatial_shape, (list, tuple)) and len(spatial_shape) == 2:
            resolved["env_config"] = {
                "type": "grid_world_2d",
                "height": spatial_shape[0],
                "width": spatial_shape[1],
            }
        else:
            resolved["env_config"] = {"length": resolved["observation_size"]}
        return resolved

    @field_validator("env_config", mode="before")
    @classmethod
    def validate_env_config_field(cls, value: Any) -> EnvConfig:
        return _coerce_env_config(value)

    @property
    def effective_model_dim(self) -> int:
        return self.model_dim if self.model_dim is not None else self.latent_dim

    @property
    def observation_size(self) -> int:
        return self.env_config.observation_size

    @property
    def observation_spatial_shape(self) -> tuple[int, ...]:
        return self.env_config.observation_spatial_shape

    @property
    def effective_observation_spatial_shape(self) -> tuple[int, ...]:
        return self.observation_spatial_shape

    @property
    def observation_channels(self) -> int:
        return self.env_config.observation_channels

    @property
    def num_actions(self) -> int:
        legacy_num_actions = self.__dict__.get("num_actions")
        if isinstance(legacy_num_actions, int):
            return legacy_num_actions
        return self.env_config.num_actions

    @property
    def action_vocab_size(self) -> int:
        return self.num_actions

    @model_validator(mode="after")
    def validate_encoder_head_divisibility(self) -> EncoderModelConfig:
        if self.effective_model_dim % self.encoder_heads != 0:
            raise ValueError("model_dim must be divisible by encoder_heads")
        return self

    @model_validator(mode="after")
    def validate_projection_stages(self) -> EncoderModelConfig:
        if (
            self.embedding_stage == ProjectionStage.POST_LN
            and not self.latent_layer_norm
        ):
            raise ValueError("embedding_stage=post_ln requires latent_layer_norm=True")
        if self.sigreg_stage == ProjectionStage.POST_LN and not self.latent_layer_norm:
            raise ValueError("sigreg_stage=post_ln requires latent_layer_norm=True")
        if (
            self.embedding_stage == ProjectionStage.POST_BN
            and not self.latent_batch_norm
        ):
            raise ValueError("embedding_stage=post_bn requires latent_batch_norm=True")
        if self.sigreg_stage == ProjectionStage.POST_BN and not self.latent_batch_norm:
            raise ValueError("sigreg_stage=post_bn requires latent_batch_norm=True")
        if self.embedding_stage == ProjectionStage.POST_L2 and not self.latent_l2_norm:
            raise ValueError("embedding_stage=post_l2 requires latent_l2_norm=True")
        if self.sigreg_stage == ProjectionStage.POST_L2 and not self.latent_l2_norm:
            raise ValueError("sigreg_stage=post_l2 requires latent_l2_norm=True")
        return self

    @property
    def uses_rotary_temporal_position_encoding(self) -> bool:
        return self.temporal_position_encoding == TemporalPositionEncoding.ROTARY

    def _validate_rotary_attention_head_dim(
        self,
        heads: int,
        name: str,
    ) -> None:
        if not self.uses_rotary_temporal_position_encoding:
            return
        if self.effective_model_dim // heads % 2 != 0:
            raise ValueError(
                f"model_dim / {name} must be even for rotary temporal positions"
            )


class HDWMConfig(EncoderModelConfig):
    """Validated HDWM model config loaded from Hydra."""

    type: Literal["hdwm"] = "hdwm"
    prior_layers: int = Field(default=2, gt=0)
    prior_heads: int = Field(default=4, gt=0)
    readout_heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_hdwm_head_divisibility(self) -> HDWMConfig:
        if self.effective_model_dim % self.prior_heads != 0:
            raise ValueError("model_dim must be divisible by prior_heads")
        if self.effective_model_dim % self.readout_heads != 0:
            raise ValueError("model_dim must be divisible by readout_heads")
        self._validate_rotary_attention_head_dim(self.prior_heads, "prior_heads")
        return self


class LEWMConfig(EncoderModelConfig):
    """Validated LE-WM baseline model config loaded from Hydra."""

    type: Literal["lewm"] = "lewm"
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_lewm_head_divisibility(self) -> LEWMConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads,
            "predictor_heads",
        )
        return self


class LEWMCNNConfig(EncoderModelConfig):
    """Validated LE-WM CNN-variant model config loaded from Hydra.

    Replaces the per-cell Transformer encoder with a CNN backbone while
    keeping the same projector, predictor, and loss architecture.
    """

    type: Literal["lewm_cnn"] = "lewm_cnn"
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_size: int = Field(default=3, gt=0)
    cnn_stride: int = Field(default=2, gt=0)
    cnn_padding: int | None = None

    @model_validator(mode="after")
    def validate_cnn_head_divisibility(self) -> LEWMCNNConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads,
            "predictor_heads",
        )
        return self


class LEWMViTConfig(EncoderModelConfig):
    """Validated LE-WM ViT config.

    Uses a Vision Transformer encoder that splits the grid into patches,
    processes them with a Transformer, and outputs a CLS token.
    Naturally supports variable input sizes via position embedding interpolation.
    """

    type: Literal["lewm_vit"] = "lewm_vit"
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)
    vit_patch_size: int = Field(default=3, gt=0)
    vit_layers: int = Field(default=4, gt=0)
    vit_heads: int = Field(default=4, gt=0)
    vit_mlp_ratio: int = Field(default=4, gt=0)
    vit_dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_vit_head_divisibility(self) -> LEWMViTConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        if self.effective_model_dim % self.vit_heads != 0:
            raise ValueError("model_dim must be divisible by vit_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads, "predictor_heads")
        return self


class ICWMConfig(LEWMConfig):
    """Validated in-context LE-WM trajectory-packing config."""

    type: Literal["icwm"] = "icwm"
    context_length: int = Field(default=4, gt=0)


class LEWMV2Config(LEWMConfig):
    """Validated LE-WMv2 dynamics-space model config loaded from Hydra."""

    type: Literal["lewm_v2"] = "lewm_v2"
    h_l2_norm: bool = True
    obs_decoder_layers: int = Field(default=1, gt=0)


class LEWMV3Config(LEWMV2Config):
    """Validated LE-WMv3 concept-conditioned dynamics model config."""

    type: Literal["lewm_v3"] = "lewm_v3"
    dynamics_mode: Literal["rotation", "transition"] = "rotation"
    concept_dim: int = Field(default=8, gt=0)
    concept_hidden_dim: int | None = Field(default=None, gt=0)
    transition_hidden_dim: int | None = Field(default=None, gt=0)
    rotation_hidden_dim: int | None = Field(default=None, gt=0)
    concept_mlp_layers: int = Field(default=2, gt=0)
    transition_mlp_layers: int = Field(default=2, gt=0)
    rotation_mlp_layers: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_rotation_dim_even(self) -> LEWMV3Config:
        if self.dynamics_mode == "rotation" and self.effective_model_dim % 2 != 0:
            raise ValueError("model_dim must be even for paired-plane rotations")
        return self


class LIWMConfig(LEWMConfig):
    """Validated Li-group world-model config."""

    type: Literal["liwm"] = "liwm"
    pos_dim: int = Field(default=3, gt=1)
    num_generators: int = Field(default=6, gt=0)


class PRISMConfig(EncoderModelConfig):
    """Validated PRISM model config loaded from Hydra."""

    type: Literal["prism"] = "prism"
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def validate_prism_head_divisibility(self) -> PRISMConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads,
            "predictor_heads",
        )
        return self


class PRISMV2Config(PRISMConfig):
    """Validated PRISMv2 belief-space model config loaded from Hydra."""

    type: Literal["prismv2"] = "prismv2"
    bottleneck_dim: int | None = Field(default=4, gt=0)
    obs_decoder_layers: int = Field(default=1, gt=0)

    @property
    def effective_bottleneck_dim(self) -> int:
        return (
            self.bottleneck_dim
            if self.bottleneck_dim is not None
            else self.effective_model_dim
        )


class BWMConfig(EncoderModelConfig):
    """Validated Bayes world model config loaded from Hydra."""

    type: Literal["bwm"] = "bwm"
    shared_encoder_layers: int = Field(default=1, gt=0)
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)
    prior_dim: int = Field(default=16, gt=0)
    prior_form: BWMPriorForm = BWMPriorForm.ADALN
    modulate_shift: bool = True
    prior_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_bwm_assumptions(self) -> BWMConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads,
            "predictor_heads",
        )
        if self.prior_dim >= self.latent_dim:
            raise ValueError("prior_dim must be smaller than latent_dim")
        return self


class BWMV2Config(BWMConfig):
    """Validated BWMv2 config loaded from Hydra."""

    type: Literal["bwmv2"] = "bwmv2"
    bwm_branch_enabled: bool = True


class HWMConfig(EncoderModelConfig):
    """Validated heteroscedastic dynamics model config loaded from Hydra."""

    type: Literal["hwm"] = "hwm"
    predictor_layers: int = Field(default=2, gt=0)
    predictor_heads: int = Field(default=4, gt=0)
    logvar_min: float = -4.0
    logvar_max: float = 2.0

    @model_validator(mode="after")
    def validate_hwm_assumptions(self) -> HWMConfig:
        if self.effective_model_dim % self.predictor_heads != 0:
            raise ValueError("model_dim must be divisible by predictor_heads")
        self._validate_rotary_attention_head_dim(
            self.predictor_heads,
            "predictor_heads",
        )
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        return self


WorldModelConfig = (
    HDWMConfig
    | LIWMConfig
    | ICWMConfig
    | LEWMV3Config
    | LEWMV2Config
    | LEWMConfig
    | LEWMCNNConfig
    | LEWMViTConfig
    | PRISMConfig
    | PRISMV2Config
    | BWMConfig
    | BWMV2Config
    | HWMConfig
)


class SequenceDataConfig(ConfigModel):
    """Validated training data config loaded from Hydra."""

    batch_size: int = Field(default=512, gt=0)
    sequence_length: int = Field(default=4, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    batch_sample_strategy: BatchSampleStrategy = BatchSampleStrategy.SAME_WITHIN_BATCH
    num_workers: int = Field(default=0, ge=0)
    persistent_workers: bool = True
    prefetch_factor: int | None = Field(default=None, gt=0)
    validation_batches: int = Field(default=1, gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_context_sample_strategy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "batch_sample_strategy" not in data and "context_sample_strategy" in data:
            data = dict(data)
            data["batch_sample_strategy"] = data["context_sample_strategy"]
        return data

    @property
    def context_sample_strategy(self) -> BatchSampleStrategy:
        return self.batch_sample_strategy


class OptimizerConfig(ConfigModel):
    """Validated optimizer config loaded from Hydra."""

    lr: float = Field(default=1e-3, gt=0.0)
    variance_lr: float | None = Field(default=None, gt=0.0)
    sigreg_warmup_lr_factor: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip_val: float | None = Field(default=None, ge=0.0)


class HDWMLossConfig(StrictConfigModel):
    """Validated HDWM loss config loaded from Hydra."""

    type: Literal["hdwm"] = "hdwm"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    align_weight: float = Field(default=1.0, ge=0.0)
    cmi_weight: float = Field(default=1.0, ge=0.0)
    cmi_temperature: float = Field(default=0.1, gt=0.0)
    normalize_latents_for_align: bool = True
    normalize_prior_for_readout: bool = False
    normalize_posterior_for_cmi: bool = True
    normalize_evidence_for_cmi: bool = True


class LEWMLossConfig(StrictConfigModel):
    """Validated LE-WM loss config loaded from Hydra."""

    type: Literal["lewm"] = "lewm"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    prediction_weight: float = Field(default=1.0, ge=0.0)
    sigreg_weight: float = Field(default=0.09, ge=0.0)
    wasserstein_sigreg_weight: float = Field(default=0.0, ge=0.0)
    wasserstein_sigreg_only_warmup: bool = False
    sigreg_knots: int = Field(default=17, gt=1)
    sigreg_num_proj: int = Field(default=1024, gt=0)
    sigreg_only_warmup_steps: int = Field(default=0, ge=0)
    sigreg_warmup_jitter_size: float = Field(default=0.0, ge=0.0)
    sigreg_input_batch_norm: bool = False
    sigreg_input_scale_sqrt_dim: bool = False


class ICWMLossConfig(LEWMLossConfig):
    """Validated in-context LE-WM loss config."""

    type: Literal["icwm"] = "icwm"


class LEWMV2LossConfig(StrictConfigModel):
    """Validated LE-WMv2 dynamics-space loss config loaded from Hydra."""

    type: Literal["lewm_v2"] = "lewm_v2"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    prediction_steps: int = Field(default=1, gt=0)
    detach_rollout_hidden_targets: bool = False
    dynamics_weight: float = Field(default=1.0, ge=0.0)
    obs_weight: float = Field(default=1.0, ge=0.0)
    predicted_obs_weight: float = Field(default=1.0, ge=0.0)
    computed_obs_weight: float = Field(default=1.0, ge=0.0)
    sigreg_weight: float = Field(default=0.09, ge=0.0)
    wasserstein_sigreg_weight: float = Field(default=0.0, ge=0.0)
    wasserstein_sigreg_only_warmup: bool = False
    sigreg_knots: int = Field(default=17, gt=1)
    sigreg_num_proj: int = Field(default=1024, gt=0)
    sigreg_only_warmup_steps: int = Field(default=0, ge=0)
    sigreg_warmup_jitter_size: float = Field(default=0.0, ge=0.0)
    sigreg_input_batch_norm: bool = False
    sigreg_input_scale_sqrt_dim: bool = False


class LEWMV3LossConfig(LEWMV2LossConfig):
    """Validated LE-WMv3 concept-conditioned loss config loaded from Hydra."""

    type: Literal["lewm_v3"] = "lewm_v3"
    cauchy_weight: float = Field(default=0.1, ge=0.0)
    cauchy_tau: float | None = Field(default=None, gt=0.0)
    concept_pair_max_samples: int | None = Field(default=1024, gt=1)
    concept_monitor_every_n_steps: int = Field(default=500, gt=0)
    concept_jump_similarity_threshold: float = Field(default=0.8, ge=-1.0, le=1.0)


class LIWMLossConfig(LEWMLossConfig):
    """Validated Li-group world-model loss config."""

    type: Literal["liwm"] = "liwm"
    equivariance_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1e-3, ge=0.0)
    equivariance_epsilon: float = Field(default=0.05, gt=0.0)


class PRISMLossConfig(StrictConfigModel):
    """Validated PRISM loss config loaded from Hydra."""

    type: Literal["prism"] = "prism"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    prediction_weight: float = Field(default=1.0, ge=0.0)
    sigreg_weight: float = Field(default=0.09, ge=0.0)
    wasserstein_sigreg_weight: float = Field(default=0.0, ge=0.0)
    wasserstein_sigreg_only_warmup: bool = False
    sigreg_knots: int = Field(default=17, gt=1)
    sigreg_num_proj: int = Field(default=1024, gt=0)
    sigreg_only_warmup_steps: int = Field(default=0, ge=0)
    sigreg_warmup_jitter_size: float = Field(default=0.0, ge=0.0)
    sigreg_input_batch_norm: bool = False
    sigreg_input_scale_sqrt_dim: bool = False


class PRISMV2LossConfig(PRISMLossConfig):
    """Validated PRISMv2 belief-space loss config loaded from Hydra."""

    type: Literal["prismv2"] = "prismv2"
    obs_weight: float = Field(default=1.0, ge=0.0)
    posterior_obs_weight: float = Field(default=1.0, ge=0.0)
    prior_obs_weight: float = Field(default=1.0, ge=0.0)
    z_sigreg_weight: float = Field(default=1.0, ge=0.0)
    posterior_belief_sigreg_weight: float = Field(default=0.0, ge=0.0)
    posterior_belief_sigreg_enabled: bool | None = None

    @property
    def effective_posterior_belief_sigreg_enabled(self) -> bool:
        if self.posterior_belief_sigreg_enabled is not None:
            return self.posterior_belief_sigreg_enabled
        return self.posterior_belief_sigreg_weight != 0.0


class BWMLossConfig(StrictConfigModel):
    """Validated Bayes world model loss config loaded from Hydra."""

    type: Literal["bwm"] = "bwm"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    prediction_weight: float = Field(default=1.0, ge=0.0)
    detach_prediction_for_modulated_loss: bool = False
    original_pred_enabled: bool = False
    original_pred_weight: float = Field(default=0.0, ge=0.0)
    original_sigreg_enabled: bool = False
    original_sigreg_weight: float = Field(default=0.0, ge=0.0)
    sigreg_weight: float = Field(default=0.09, ge=0.0)
    wasserstein_sigreg_weight: float = Field(default=0.0, ge=0.0)
    wasserstein_sigreg_only_warmup: bool = False
    sigreg_knots: int = Field(default=17, gt=1)
    sigreg_num_proj: int = Field(default=1024, gt=0)
    sigreg_only_warmup_steps: int = Field(default=0, ge=0)
    sigreg_warmup_jitter_size: float = Field(default=0.0, ge=0.0)
    sigreg_input_batch_norm: bool = False
    sigreg_input_scale_sqrt_dim: bool = False
    obs_cmi_enabled: bool = False
    prior_cmi_enabled: bool = False
    obs_cmi_weight: float = Field(default=0.0, ge=0.0)
    prior_cmi_weight: float = Field(default=0.0, ge=0.0)
    cmi_temperature: float = Field(default=0.1, gt=0.0)
    obs_cmi_fixed_negatives: int = Field(default=16, ge=0)
    prior_cmi_fixed_negatives: int = Field(default=8, ge=0)
    obs_cmi_negative_source: ObsCMINegativeSource = ObsCMINegativeSource.BATCH
    obs_cmi_noise_ratio: float = Field(default=0.05, ge=0.0)
    prior_cmi_negative_source: PriorCMINegativeSource = PriorCMINegativeSource.ONE_HOT
    normalize_prediction_for_cmi: bool = True
    normalize_target_for_cmi: bool = True
    obs_mask_gt_weight: float = Field(default=0.0, ge=0.0)
    obs_mask_entropy_weight: float = Field(default=0.0, ge=0.0)


class BWMV2LossConfig(BWMLossConfig):
    """Validated BWMv2 loss config loaded from Hydra.

    BWMv2 uses LE-WM naming for the main branch:
    `prediction_weight` trains prediction -> encoded target, and
    `sigreg_weight` regularizes encoded embeddings. The prior-modulated BWM
    branch is controlled separately by `bwm_*` fields.
    """

    type: Literal["bwmv2"] = "bwmv2"
    bwm_prediction_enabled: bool = False
    bwm_prediction_weight: float = Field(default=0.0, ge=0.0)
    bwm_sigreg_enabled: bool = False
    bwm_sigreg_weight: float = Field(default=0.0, ge=0.0)


class HWMLossConfig(StrictConfigModel):
    """Validated heteroscedastic dynamics loss config loaded from Hydra."""

    type: Literal["hwm"] = "hwm"
    exclude_identical_frame_pairs_from_cosine_monitor: bool = True
    vicreg_enabled: bool = False
    vicreg_weight: float = Field(default=10.0, ge=0.0)
    vicreg_variance_target: float = Field(default=0.03, gt=0.0)
    vicreg_epsilon: float = Field(default=1e-4, gt=0.0)
    nll_weight: float = Field(default=1.0, ge=0.0)
    var_prior_weight: float = Field(default=1e-3, ge=0.0)
    sigreg_weight: float = Field(default=0.09, ge=0.0)
    wasserstein_sigreg_weight: float = Field(default=0.0, ge=0.0)
    wasserstein_sigreg_only_warmup: bool = False
    sigreg_knots: int = Field(default=17, gt=1)
    sigreg_num_proj: int = Field(default=1024, gt=0)
    sigreg_only_warmup_steps: int = Field(default=0, ge=0)
    sigreg_warmup_jitter_size: float = Field(default=0.0, ge=0.0)
    sigreg_input_batch_norm: bool = False
    sigreg_input_scale_sqrt_dim: bool = False


WorldModelLossConfig = Annotated[
    (
        HDWMLossConfig
        | LIWMLossConfig
        | ICWMLossConfig
        | LEWMV3LossConfig
        | LEWMV2LossConfig
        | LEWMLossConfig
        | PRISMLossConfig
        | PRISMV2LossConfig
        | BWMLossConfig
        | BWMV2LossConfig
        | HWMLossConfig
    ),
    Field(discriminator="type"),
]


class MetricsConfig(ConfigModel):
    """Validated metrics config loaded from Hydra."""

    log_gt_pos_distribution: bool = False
    log_layer_weight_stats: bool = False
    position_probe_enabled: bool = True
    position_probe_ridge: float = Field(default=1e-3, gt=0.0)


class RuntimeMonitorConfig(ConfigModel):
    """Validated runtime timing and resource monitor config loaded from Hydra."""

    enabled: bool = True
    log_every_n_steps: int = Field(default=1, gt=0)
    warmup_steps: int = Field(default=1, ge=0)
    synchronize_device: bool = True
    log_system: bool = True
    log_cuda_memory: bool = True
    log_gpu_utilization: bool = True


class TrainerConfig(ConfigModel):
    """Validated Lightning trainer config loaded from Hydra."""

    accelerator: str = "auto"
    devices: int | str = 1
    max_steps: int = Field(default=100, gt=0)
    log_every_n_steps: int = Field(default=1, gt=0)
    val_check_interval: int | float = Field(default=500, gt=0)
    enable_checkpointing: bool = False
    enable_model_summary: bool = True
    compile_model: bool = True
    compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ] = "reduce-overhead"


class SwanLabConfig(ConfigModel):
    """Validated SwanLab config loaded from Hydra."""

    enabled: bool = True
    project: str = "hdwm"
    workspace: str = ""
    experiment_name: str = "hdwm-ring-world"
    description: str = "Minimal HDWM training pipeline for ring-world."
    mode: str = "offline"
    logdir: str = "swanlog"
    save_dir: str = "swanlog/.swanlab"
    cache_dir: str = "swanlog/.cache"


class TrainConfig(ConfigModel):
    """Validated config for the training entrypoint."""

    seed: int = 0
    env: EnvConfig
    model: WorldModelConfig
    data: SequenceDataConfig = Field(default_factory=SequenceDataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: WorldModelLossConfig = Field(default_factory=HDWMLossConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    runtime_monitor: RuntimeMonitorConfig = Field(default_factory=RuntimeMonitorConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    swanlab: SwanLabConfig = Field(default_factory=SwanLabConfig)

    @model_validator(mode="before")
    @classmethod
    def set_model_env_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Set model's env_config before validation."""
        model_data = data.get("model", {})
        model_data["env_config"] = data.get("env", {})
        return data

    @field_validator("env", mode="before")
    @classmethod
    def validate_env_config(cls, value: Any) -> EnvConfig:
        return _coerce_env_config(value)

    @model_validator(mode="after")
    def validate_model_loss_match(self) -> TrainConfig:
        # CNN encoder variants use the same loss as their base model.
        _CNN_MODEL_TYPES = {"lewm_cnn", "lewm_vit"}
        model_type = self.model.type
        if model_type in _CNN_MODEL_TYPES:
            model_type = "lewm"
        if model_type != self.loss.type:
            raise ValueError(
                f"model type {self.model.type!r} requires matching loss type, "
                f"got {self.loss.type!r}"
            )
        return self

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> TrainConfig:
        return cls.model_validate(_omegaconf_to_dict(cfg))


class EnvRunConfig(ConfigModel):
    """Validated config for the ring-world smoke-run entrypoint."""

    seed: int = 0
    env: EnvConfig
    batch_size: int = Field(default=4, gt=0)
    sequence_length: int = Field(default=8, gt=0)
    print_first_sequence: bool = True

    @field_validator("env", mode="before")
    @classmethod
    def validate_env_config(cls, value: Any) -> EnvConfig:
        return _coerce_env_config(value)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> EnvRunConfig:
        return cls.model_validate(_omegaconf_to_dict(cfg))
