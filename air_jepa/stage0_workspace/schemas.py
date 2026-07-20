"""Strict configuration schemas for AIR-JEPA Stage 0."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from air_jepa.stage0_workspace import AIR_METHODS, SYSTEM_SEEDS


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class PathSpec(StrictModel):
    train_manifest: str
    historical_development_manifest: str
    historical_confirmatory_manifest: str
    preflight_manifest: str
    air_dev_manifest: str
    air_early_manifest: str
    air_select_manifest: str
    air_final_manifest: str
    protocol_lock: str
    package_lock: str
    source_lock: str
    run_root: str
    representation_checkpoint_template: str
    j0_checkpoint_template: str
    j1_checkpoint_template: str
    historical_j0_result_template: str
    historical_j1_result_template: str
    air_checkpoint_template: str
    result_template: str
    diagnostic_template: str
    release_template: str
    audit_output: str
    benchmark_output: str


class ModelSpec(StrictModel):
    input_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    ffn_expansion: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    cost_bins: int = Field(gt=1)
    max_distance: int = Field(gt=0)
    neighbor_offsets: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def validate_attention(self) -> ModelSpec:
        locked = {
            "input_dim": 64,
            "hidden_dim": 64,
            "attention_heads": 4,
            "ffn_expansion": 2,
            "dropout": 0.0,
            "cost_bins": 129,
            "max_distance": 128,
        }
        for name, expected_value in locked.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"AIR0-v1 locks model.{name}={expected_value!r}")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        expected = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
        if self.neighbor_offsets != expected:
            raise ValueError("AIR0 requires the locked four-neighbor-plus-self order")
        if self.cost_bins != self.max_distance + 1:
            raise ValueError("cost_bins must equal max_distance + 1")
        return self


class MethodLossSpec(StrictModel):
    action: float = Field(ge=0.0)
    future: float = Field(ge=0.0)
    cost: float = Field(ge=0.0)


class TrainingSpec(StrictModel):
    steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    betas: tuple[float, float]
    epsilon: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip: float = Field(gt=0.0)
    scheduler: Literal["cosine"]
    dtype: Literal["float32"]
    amp: Literal[False]
    log_every: int = Field(gt=0)
    gradient_audit_every: int = Field(gt=0)
    k_train: tuple[int, ...]
    phase_steps: int = Field(gt=0)
    deep_supervision_every: int = Field(gt=0)
    target_variance_epsilon: float = Field(gt=0.0)
    methods: dict[str, MethodLossSpec]

    @model_validator(mode="after")
    def validate_training_matrix(self) -> TrainingSpec:
        locked = {
            "steps": 30_000,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "betas": (0.9, 0.999),
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "log_every": 500,
            "gradient_audit_every": 500,
            "k_train": (4, 8, 16, 32, 64, 128),
            "phase_steps": 5_000,
            "deep_supervision_every": 16,
            "target_variance_epsilon": 1e-4,
        }
        for name, expected_value in locked.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"AIR0-v1 locks training.{name}={expected_value!r}")
        if tuple(sorted(self.k_train)) != self.k_train:
            raise ValueError("k_train must be strictly increasing")
        if self.steps != self.phase_steps * len(self.k_train):
            raise ValueError("steps must equal phase_steps * len(k_train)")
        if tuple(self.methods) != AIR_METHODS:
            raise ValueError(f"training methods must be ordered as {AIR_METHODS}")
        direct = self.methods["air0_direct"]
        treatment = self.methods["air0_jepa"]
        if direct != MethodLossSpec(action=1.0, future=0.0, cost=0.0):
            raise ValueError("air0_direct loss contract changed")
        if treatment != MethodLossSpec(action=1.0, future=1.0, cost=0.5):
            raise ValueError("air0_jepa loss contract changed")
        return self


class EvaluationSpec(StrictModel):
    max_steps: Literal[128]
    seen_max_size: Literal[21]
    primary_k: Literal[128]
    k_values: tuple[int, ...]
    action_ids: tuple[int, ...]
    primary_action_protocol: Literal["unmasked"]
    diagnostic_action_protocol: Literal["corrected"]
    local_states_per_maze: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    bootstrap_seed: int
    early_seen_per_size: int = Field(gt=0)
    early_ood_per_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_evaluation(self) -> EvaluationSpec:
        if self.k_values != (1, 4, 8, 16, 32, 64, 128):
            raise ValueError("locked K curve changed")
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k is absent from k_values")
        if self.action_ids != (1, 2, 3, 4):
            raise ValueError("Procgen action contract changed")
        locked = {
            "local_states_per_maze": 24,
            "bootstrap_samples": 20_000,
            "bootstrap_seed": 20_260_720,
            "early_seen_per_size": 20,
            "early_ood_per_size": 35,
        }
        for name, expected_value in locked.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"AIR0-v1 locks evaluation.{name}={expected_value!r}")
        return self


class GateSpec(StrictModel):
    green_overall_mean: float
    green_overall_each_seed: float
    green_ood_mean: float
    green_ood_each_seed: float
    j1_noninferiority_margin: float
    direct_noninferiority_margin: float
    k_scaling_min_delta: float
    future_copy_improvement: float
    permutation_local_top1_drop: float
    permutation_sr_drop: float
    yellow_overall_floor: float
    early_green_sr: float
    early_green_j1_delta: float
    early_green_k_delta: float
    early_red_sr: float
    collapse_variance_ratio: float
    collapse_candidate_ratio: float
    collapse_permutation_effect: float

    @model_validator(mode="after")
    def validate_gates(self) -> GateSpec:
        locked = {
            "green_overall_mean": 0.90,
            "green_overall_each_seed": 0.87,
            "green_ood_mean": 0.75,
            "green_ood_each_seed": 0.68,
            "j1_noninferiority_margin": -0.03,
            "direct_noninferiority_margin": -0.01,
            "k_scaling_min_delta": 0.05,
            "future_copy_improvement": 0.30,
            "permutation_local_top1_drop": 0.10,
            "permutation_sr_drop": 0.05,
            "yellow_overall_floor": 0.80,
            "early_green_sr": 0.88,
            "early_green_j1_delta": -0.05,
            "early_green_k_delta": 0.03,
            "early_red_sr": 0.75,
            "collapse_variance_ratio": 0.10,
            "collapse_candidate_ratio": 0.10,
            "collapse_permutation_effect": 0.01,
        }
        for name, expected_value in locked.items():
            if getattr(self, name) != expected_value:
                raise ValueError(f"AIR0-v1 locks gates.{name}={expected_value!r}")
        return self


class StatisticsSpec(StrictModel):
    familywise_alpha: float = Field(gt=0.0, lt=1.0)
    multiplicity: Literal["bonferroni_simultaneous_percentile_ci"]
    resampling: Literal["crossed_seed_size_stratified_task"]

    @model_validator(mode="after")
    def validate_statistics(self) -> StatisticsSpec:
        if self.familywise_alpha != 0.05:
            raise ValueError("AIR0-v1 locks statistics.familywise_alpha=0.05")
        return self


class Stage0Config(StrictModel):
    schema_name: Literal["air-jepa-stage0-config-v1"] = Field(alias="schema")
    experiment_id: Literal["procgen-maze-air0-workspace-v1"]
    worker_count: Literal[4]
    seeds: tuple[int, ...]
    paths: PathSpec
    model: ModelSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    statistics: StatisticsSpec
    gates: GateSpec

    @model_validator(mode="after")
    def validate_stage(self) -> Stage0Config:
        if self.seeds != SYSTEM_SEEDS:
            raise ValueError(f"system seeds must be {SYSTEM_SEEDS}")
        return self


__all__ = [
    "EvaluationSpec",
    "GateSpec",
    "MethodLossSpec",
    "ModelSpec",
    "PathSpec",
    "Stage0Config",
    "StatisticsSpec",
    "TrainingSpec",
]
