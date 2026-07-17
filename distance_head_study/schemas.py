"""Strict schemas for the staged DistanceHead experiment."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from distance_head_study import (
    ACTION_IDS,
    CONFIRMATION_SEED_START,
    DEVELOPMENT_BACKBONE_SEEDS,
    MODEL_ACTION_VOCAB_SIZE,
    PROTOCOL_ID,
)


class StrictModel(BaseModel):
    """Reject unknown fields and mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetMode(str, Enum):
    RAW = "raw"
    GLOBAL_NORM = "global_norm"
    LOG1P = "log1p"
    LEGACY_LOG_NORM = "legacy_log_norm"


class RegressionLoss(str, Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    ASYMMETRIC = "asymmetric"


class OutputKind(str, Enum):
    SCALAR = "scalar"
    ORDINAL = "ordinal"
    DISTRIBUTION = "distribution"
    QUANTILE = "quantile"
    MULTITASK = "multitask"


class ArchitectureKind(str, Enum):
    HISTORICAL_CONCAT = "historical_concat"
    ASYMMETRIC = "asymmetric"
    QUASIMETRIC = "quasimetric"
    HORIZON_CONDITIONED = "horizon_conditioned"
    HIERARCHICAL = "hierarchical"


class SamplerKind(str, Enum):
    UNIFORM = "uniform"
    DISTANCE_BALANCED = "distance_balanced"
    DECISION_BALANCED = "decision_balanced"
    FULL_HORIZON = "full_horizon"
    HARD_CROSSFIT = "hard_crossfit"


class TrainingScope(str, Enum):
    FROZEN = "frozen"
    PREDICTOR = "predictor"
    PROJECTOR_PREDICTOR = "projector_predictor"
    FULL = "full"


class InitializationMode(str, Enum):
    STRICT = "strict"
    COMPATIBLE_SHARED = "compatible_shared"


class PlannerKind(str, Enum):
    MODEL_FREE_GREEDY = "model_free_greedy"
    PREDICTOR_GREEDY = "predictor_greedy"
    CATEGORICAL_CEM = "categorical_cem"
    ICEM = "icem"
    BEAM = "beam"
    BEST_FIRST = "best_first"


class CostKind(str, Enum):
    LATENT_L2 = "latent_l2"
    TERMINAL_DISTANCE = "terminal_distance"
    PATH_INTEGRATED = "path_integrated"
    HYBRID = "hybrid"
    REACHABILITY = "reachability"
    RISK_LOOP = "risk_loop"


class LabelMode(str, Enum):
    TRUE = "true"
    SHUFFLED = "shuffled"
    RANDOM = "random"


class PathsConfig(StrictModel):
    source_config: Path
    source_lock: Path
    train_manifest: Path
    legacy_manifest: Path
    cal_manifest: Path
    screen_manifest: Path
    select_manifest: Path
    confirm_manifest: Path
    stress_manifest: Path
    method_catalog: Path
    seed_registry: Path
    baseline_provenance: Path
    bootstrap_schedule: Path
    protocol_lock: Path
    checkpoint_root: Path
    cache_root: Path
    run_root: Path
    decision_root: Path
    seed_release_root: Path
    shortlist_lock: Path
    negative_shortlist_lock: Path
    confirmation_n_lock: Path
    confirm_opened: Path
    closure_gate: Path
    negative_closure_lock: Path
    legacy_backbone_template: str
    fresh_backbone_template: str
    head_checkpoint_template: str
    cache_index_template: str
    candidate_bank_template: str
    result_template: str


class SplitConfig(StrictModel):
    train_sizes: tuple[int, ...] = tuple(range(9, 22, 2))
    screen_per_size: Literal[20] = 20
    select_per_size: Literal[30] = 30
    confirm_per_size: Literal[100] = 100
    stress_sizes: tuple[int, ...] = (27, 29, 31)
    stress_per_size: int = Field(50, ge=10)
    cal_per_size: int = Field(20, ge=1)

    @field_validator("train_sizes")
    @classmethod
    def validate_train_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(range(9, 22, 2)):
            raise ValueError("training/development sizes are locked to 9..21 odd")
        return value


class SeedConfig(StrictModel):
    screen_backbones: tuple[int, ...] = (42,)
    select_backbones: tuple[int, ...] = DEVELOPMENT_BACKBONE_SEEDS
    historical_backbones: tuple[int, ...] = tuple(range(42, 62))
    ordered_confirmation_backbones: tuple[int, ...] = tuple(range(1001, 1033))
    screen_head_seeds: tuple[int, ...] = (0, 1, 2)
    select_head_seeds: tuple[int, ...] = (0, 1)
    confirmation_head_seed: int = 0
    sample_schedule_seed: int = 2_026_071_701
    run_order_seed: int = 2_026_071_702
    bootstrap_seed: int = 2_026_071_703

    @model_validator(mode="after")
    def validate_seed_hierarchy(self) -> SeedConfig:
        if self.screen_backbones != (42,):
            raise ValueError("Seed-1 must use backbone 42")
        if self.select_backbones != DEVELOPMENT_BACKBONE_SEEDS:
            raise ValueError("Seed-3 must use backbones 42/43/44")
        if self.screen_head_seeds != (0, 1, 2):
            raise ValueError("Seed-1 head seeds are fixed to 0/1/2")
        if self.select_head_seeds != (0, 1):
            raise ValueError("Seed-3 head seeds are fixed to 0/1")
        if len(set(self.ordered_confirmation_backbones)) != len(
            self.ordered_confirmation_backbones
        ):
            raise ValueError("confirmation backbone seeds must be unique")
        if len(self.ordered_confirmation_backbones) < 10:
            raise ValueError("at least ten ordered confirmation seeds are required")
        if self.ordered_confirmation_backbones[0] != CONFIRMATION_SEED_START:
            raise ValueError("fresh confirmation namespace must start at 1001")
        overlap = set(self.historical_backbones) & set(
            self.ordered_confirmation_backbones
        )
        if overlap:
            raise ValueError(f"confirmation seeds collide with history: {overlap}")
        return self


class PlannerProtocol(StrictModel):
    max_steps: Literal[128] = 128
    history_size: Literal[3] = 3
    horizon: Literal[12] = 12
    num_candidates: Literal[64] = 64
    num_elites: Literal[8] = 8
    cem_iters: Literal[1] = 1
    momentum: Literal[0.1] = 0.1
    action_ids: tuple[int, ...] = ACTION_IDS
    model_action_vocab_size: Literal[5] = MODEL_ACTION_VOCAB_SIZE
    action_protocols: tuple[Literal["corrected_v1", "unmasked"], ...] = (
        "corrected_v1",
        "unmasked",
    )
    rollout_semantics: Literal["legacy_warmup_v1"] = "legacy_warmup_v1"
    reference_transitions: Literal[768] = 768

    @field_validator("action_ids")
    @classmethod
    def validate_actions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != ACTION_IDS:
            raise ValueError(f"action space is locked to {ACTION_IDS}")
        return value


class TrainingConfig(StrictModel):
    steps: Literal[30_000] = 30_000
    effective_batch_size: Literal[512] = 512
    microbatch_size: int = Field(128, ge=16, le=512)
    pairs_per_topology: Literal[64] = 64
    learning_rate: Literal[0.001] = 0.001
    joint_backbone_learning_rate: Literal[0.0001] = 0.0001
    weight_decay: Literal[0.00001] = 0.00001
    grad_clip: Literal[1.0] = 1.0
    checkpoint_selection: Literal["final_step"] = "final_step"
    warmup_fraction: Literal[0.05] = 0.05
    final_lr_ratio: Literal[0.1] = 0.1
    deterministic: Literal[True] = True
    calibration_gradient_ratio: Literal[0.5] = 0.5
    calibration_weight_clip: tuple[float, float] = (0.1, 10.0)
    distance_bins: tuple[int, ...] = (2, 4, 8, 12, 20, 32, 48, 128)
    reachability_budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    horizons: tuple[int, ...] = (1, 3, 5, 8, 11)
    candidate_sets_per_backbone: int = Field(2048, ge=128)
    trajectory_contexts_per_step: Literal[16] = 16
    trajectory_candidates: Literal[64] = 64
    hard_negative_folds: Literal[5] = 5

    @model_validator(mode="after")
    def validate_accumulation(self) -> TrainingConfig:
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("effective batch size must divide by microbatch size")
        if self.calibration_weight_clip != (0.1, 10.0):
            raise ValueError("gradient-calibrated loss-weight clip is locked")
        accumulation = self.effective_batch_size // self.microbatch_size
        if self.trajectory_contexts_per_step % accumulation:
            raise ValueError("trajectory contexts must divide across microbatches")
        return self


class AnalysisConfig(StrictModel):
    familywise_alpha: Literal[0.05] = 0.05
    bootstrap_samples: int = Field(10_000, ge=10_000)
    minimum_overall_delta: Literal[0.04] = 0.04
    minimum_ood_delta: Literal[0.05] = 0.05
    screen_strong_delta: Literal[0.06] = 0.06
    screen_regular_delta: Literal[0.04] = 0.04
    borderline_delta: Literal[0.02] = 0.02
    max_secondary_drop: Literal[0.02] = 0.02
    required_power: Literal[0.8] = 0.8
    minimum_confirmation_backbones: Literal[10] = 10
    diagnostic_batches: Literal[32] = 32
    trajectory_diagnostic_batches: Literal[8] = 8
    trajectory_diagnostic_contexts: Literal[8] = 8
    reachability_min_auroc: Literal[0.65] = 0.65
    reachability_max_brier: Literal[0.25] = 0.25
    reachability_max_ece: Literal[0.15] = 0.15
    reachability_max_monotonic_violation: Literal[0.05] = 0.05
    primary_endpoints: tuple[str, ...] = (
        "corrected_overall_sr",
        "corrected_ood_sr",
    )


class HeadSpec(StrictModel):
    architecture: ArchitectureKind = ArchitectureKind.HISTORICAL_CONCAT
    output: OutputKind = OutputKind.SCALAR
    target: TargetMode = TargetMode.LEGACY_LOG_NORM
    regression_loss: RegressionLoss = RegressionLoss.HUBER
    latent_dim: Literal[256] = 256
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    global_distance_scale: Literal[128.0] = 128.0
    ordinal_thresholds: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    distribution_edges: tuple[int, ...] = (2, 4, 8, 12, 20, 32, 48, 128)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    reachability_budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    horizon_conditioned: bool = False
    domain_adapter: bool = False
    uncertainty: bool = False

    @field_validator("ordinal_thresholds", "distribution_edges", "reachability_budgets")
    @classmethod
    def validate_ordered_positive_grid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("distance grids must be nonempty and positive")
        if tuple(sorted(set(value))) != value:
            raise ValueError("distance grids must be strictly increasing")
        return value

    @model_validator(mode="after")
    def validate_architecture(self) -> HeadSpec:
        if self.architecture in (
            ArchitectureKind.HORIZON_CONDITIONED,
            ArchitectureKind.HIERARCHICAL,
        ):
            if not self.horizon_conditioned:
                raise ValueError("selected architecture needs its horizon input")
        if self.output == OutputKind.MULTITASK and not self.reachability_budgets:
            raise ValueError("multitask output requires reachability budgets")
        if self.output == OutputKind.QUANTILE:
            if tuple(sorted(set(self.quantiles))) != self.quantiles:
                raise ValueError("quantiles must be strictly ordered")
            if any(value <= 0.0 or value >= 1.0 for value in self.quantiles):
                raise ValueError("quantiles must lie strictly inside (0, 1)")
        if self.architecture == ArchitectureKind.QUASIMETRIC:
            if self.output != OutputKind.SCALAR:
                raise ValueError("quasimetric architecture currently has scalar output")
            if self.horizon_conditioned or self.domain_adapter or self.uncertainty:
                raise ValueError(
                    "quasimetric forbids horizon/domain/uncertainty additions"
                )
        return self


class ObjectiveWeights(StrictModel):
    absolute: float = Field(1.0, ge=0.0)
    anchor: float = Field(0.0, ge=0.0)
    pairwise: float = Field(0.0, ge=0.0)
    listwise: float = Field(0.0, ge=0.0)
    all_action: float = Field(0.0, ge=0.0)
    delta: float = Field(0.0, ge=0.0)
    bellman: float = Field(0.0, ge=0.0)
    eikonal: float = Field(0.0, ge=0.0)
    multistep: float = Field(0.0, ge=0.0)
    triangle: float = Field(0.0, ge=0.0)
    successor_contrastive: float = Field(0.0, ge=0.0)
    predicted_listwise: float = Field(0.0, ge=0.0)
    predicted_consistency: float = Field(0.0, ge=0.0)
    trajectory_listwise: float = Field(0.0, ge=0.0)
    reachability: float = Field(0.0, ge=0.0)
    uncertainty: float = Field(0.0, ge=0.0)
    original_jepa: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_nonempty(self) -> ObjectiveWeights:
        if sum(self.model_dump().values()) <= 0.0:
            raise ValueError("at least one objective must be active")
        return self


class PlannerSpec(StrictModel):
    kind: PlannerKind = PlannerKind.CATEGORICAL_CEM
    cost: CostKind = CostKind.TERMINAL_DISTANCE
    path_weight: float = Field(0.0, ge=0.0)
    latent_l2_weight: float = Field(0.0, ge=0.0)
    reachability_weight: float = Field(0.0, ge=0.0)
    uncertainty_weight: float = Field(0.0, ge=0.0)
    loop_weight: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_cost(self) -> PlannerSpec:
        if self.cost == CostKind.HYBRID and self.latent_l2_weight <= 0.0:
            raise ValueError("hybrid cost needs a positive latent-L2 weight")
        if self.cost == CostKind.REACHABILITY and self.reachability_weight <= 0.0:
            raise ValueError("reachability cost needs a positive weight")
        if self.cost == CostKind.RISK_LOOP and (
            self.uncertainty_weight + self.loop_weight <= 0.0
        ):
            raise ValueError("risk-loop cost needs uncertainty or loop weight")
        return self


class ResolvedMethod(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    stage: Literal["P0", "P1", "P2", "reserve", "confirm"]
    role: Literal["baseline", "candidate", "control", "oracle", "calibration"]
    description: str = Field(min_length=1)
    head: HeadSpec | None = None
    objectives: ObjectiveWeights | None = None
    sampler: SamplerKind = SamplerKind.UNIFORM
    training_scope: TrainingScope = TrainingScope.FROZEN
    distance_gradients_to_backbone: bool = False
    planner: PlannerSpec = PlannerSpec()
    label_mode: LabelMode = LabelMode.TRUE
    trajectory_horizons: tuple[int, ...] = (11,)
    trainable: bool = True
    update_head: bool = True
    checkpoint_owner: str | None = None
    reuse_parent_checkpoint: bool = False
    checkpoint_decision_alias: str | None = None
    initialization_parent: str | None = None
    initialization_mode: InitializationMode = InitializationMode.STRICT
    confirmatory_eligible: bool = True
    uses_test_bfs: bool = False
    checkpoint_selection: Literal["final_step"] = "final_step"
    required_artifacts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_method_boundary(self) -> ResolvedMethod:
        if not self.trajectory_horizons or any(
            horizon not in (1, 3, 5, 8, 11) for horizon in self.trajectory_horizons
        ):
            raise ValueError("executed-action horizons must be a subset of 1/3/5/8/11")
        if self.planner.cost == CostKind.LATENT_L2:
            if self.head is not None or self.trainable or self.checkpoint_owner:
                raise ValueError("latent-L2 baseline has no DistanceHead training")
        elif self.role != "oracle" and self.head is None:
            raise ValueError("learned distance methods require a head")
        if self.trainable and self.objectives is None:
            raise ValueError("trainable method requires objective weights")
        if not self.trainable:
            if self.objectives is not None:
                raise ValueError("non-trainable method cannot declare objectives")
            if self.head is not None and not self.checkpoint_owner:
                raise ValueError(
                    "diagnostic planner needs an explicit checkpoint owner"
                )
        if self.role == "oracle" and self.confirmatory_eligible:
            raise ValueError("oracle methods cannot enter confirmatory ranking")
        if self.uses_test_bfs and self.role != "oracle":
            raise ValueError(
                "test-time BFS is restricted to non-ranking oracle methods"
            )
        if self.label_mode != LabelMode.TRUE and self.role != "control":
            raise ValueError("shuffled/random labels are control-only")
        if self.training_scope != TrainingScope.FROZEN:
            if self.objectives is None or self.objectives.original_jepa <= 0.0:
                raise ValueError(
                    "joint training must retain the original JEPA objective"
                )
        elif self.distance_gradients_to_backbone:
            raise ValueError("frozen-backbone methods cannot enable distance gradients")
        if self.distance_gradients_to_backbone and not self.update_head:
            raise ValueError("joint distance treatment must also update its head")
        if self.reuse_parent_checkpoint and self.initialization_parent is not None:
            raise ValueError(
                "planner reuse and joint-training initialization are exclusive"
            )
        if self.reuse_parent_checkpoint:
            owners = int(self.checkpoint_decision_alias is not None) + int(
                self.checkpoint_owner is not None
            )
            if owners != 1:
                raise ValueError(
                    "planner checkpoint reuse needs exactly one owner source"
                )
        elif self.checkpoint_decision_alias is not None:
            raise ValueError("checkpoint decision alias is planner-reuse only")
        if (
            self.initialization_mode != InitializationMode.STRICT
            and self.initialization_parent is None
        ):
            raise ValueError(
                "compatible initialization requires an initialization parent"
            )
        return self


ALLOWED_OVERRIDE_PATHS = frozenset(
    {
        "head.architecture",
        "head.output",
        "head.target",
        "head.regression_loss",
        "head.horizon_conditioned",
        "head.domain_adapter",
        "head.uncertainty",
        "objectives.absolute",
        "objectives.anchor",
        "objectives.pairwise",
        "objectives.listwise",
        "objectives.all_action",
        "objectives.delta",
        "objectives.bellman",
        "objectives.eikonal",
        "objectives.multistep",
        "objectives.triangle",
        "objectives.successor_contrastive",
        "objectives.predicted_listwise",
        "objectives.predicted_consistency",
        "objectives.trajectory_listwise",
        "objectives.reachability",
        "objectives.uncertainty",
        "objectives.original_jepa",
        "sampler",
        "training_scope",
        "distance_gradients_to_backbone",
        "trajectory_horizons",
        "update_head",
        "reuse_parent_checkpoint",
        "checkpoint_decision_alias",
        "checkpoint_owner",
        "planner.kind",
        "planner.cost",
        "planner.path_weight",
        "planner.latent_l2_weight",
        "planner.reachability_weight",
        "planner.uncertainty_weight",
        "planner.loop_weight",
        "label_mode",
        "required_artifacts",
        "initialization_mode",
    }
)


class MethodTemplate(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    stage: Literal["P0", "P1", "P2", "reserve", "confirm"]
    role: Literal["baseline", "candidate", "control", "oracle", "calibration"]
    description: str = Field(min_length=1)
    parent: str | None = None
    resolved: ResolvedMethod | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    declared_changes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_template(self) -> MethodTemplate:
        if self.parent is None:
            if self.resolved is None or self.overrides or self.declared_changes:
                raise ValueError("root method needs only a complete resolved spec")
            if (
                self.resolved.name != self.name
                or self.resolved.stage != self.stage
                or self.resolved.role != self.role
                or self.resolved.description != self.description
            ):
                raise ValueError("root method metadata and resolved spec differ")
        else:
            if self.resolved is not None or not self.overrides:
                raise ValueError("derived method needs parent plus overrides")
            if set(self.overrides) != set(self.declared_changes):
                raise ValueError("declared changes must exactly match override keys")
            unknown = set(self.overrides) - ALLOWED_OVERRIDE_PATHS
            if unknown:
                raise ValueError(f"unsupported scientific override paths: {unknown}")
        return self


class MethodCatalog(StrictModel):
    schema_version: Literal[1] = 1
    decision_aliases: dict[str, Path]
    methods: tuple[MethodTemplate, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> MethodCatalog:
        names = [method.name for method in self.methods]
        if len(names) != len(set(names)):
            raise ValueError("method catalog contains duplicate names")
        known = set(names)
        for method in self.methods:
            if method.parent is None:
                continue
            if method.parent.startswith("@"):
                if method.parent[1:] not in self.decision_aliases:
                    raise ValueError(f"unknown decision alias: {method.parent}")
            elif method.parent not in known:
                raise ValueError(f"unknown method parent: {method.parent}")
        return self


class StudyConfig(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: Literal[PROTOCOL_ID] = PROTOCOL_ID
    study_role: Literal["staged_distance_head_confirmatory_study"]
    device: str = "auto"
    paths: PathsConfig
    splits: SplitConfig = SplitConfig()
    seeds: SeedConfig = SeedConfig()
    planner: PlannerProtocol = PlannerProtocol()
    training: TrainingConfig = TrainingConfig()
    analysis: AnalysisConfig = AnalysisConfig()


__all__ = [
    "ALLOWED_OVERRIDE_PATHS",
    "AnalysisConfig",
    "ArchitectureKind",
    "CostKind",
    "HeadSpec",
    "InitializationMode",
    "LabelMode",
    "MethodCatalog",
    "MethodTemplate",
    "ObjectiveWeights",
    "OutputKind",
    "PathsConfig",
    "PlannerKind",
    "PlannerProtocol",
    "PlannerSpec",
    "RegressionLoss",
    "ResolvedMethod",
    "SamplerKind",
    "SeedConfig",
    "SplitConfig",
    "StudyConfig",
    "TargetMode",
    "TrainingConfig",
    "TrainingScope",
]
