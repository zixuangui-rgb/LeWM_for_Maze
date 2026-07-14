"""Strict configuration schemas for the planner-frontier study."""

from __future__ import annotations

import copy
import itertools
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vector_jepa_planner_frontier import ACTION_IDS, PROTOCOL_ID


class StrictModel(BaseModel):
    """Base schema that rejects silent configuration drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RolloutSemantics(str, Enum):
    LEGACY_WARMUP_V1 = "legacy_warmup_v1"
    ACTION_ALIGNED_V2 = "action_aligned_v2"


class PlannerKind(str, Enum):
    LEGACY_CEM = "legacy_cem"
    CATEGORICAL_CEM = "categorical_cem"
    ICEM = "icem"
    BEAM = "beam"
    BEST_FIRST = "best_first"
    MCTS = "mcts"
    VECTOR_DTS = "vector_dts"
    BIDIRECTIONAL = "bidirectional"


class ProposalKind(str, Enum):
    UNIFORM = "uniform"
    MIXTURE = "mixture"
    DISCRETE_DENOISING = "discrete_denoising"


class PathsConfig(StrictModel):
    source_config: Path
    source_lock: Path
    amendments: Path
    amendment_document: Path
    amendment_before: Path
    amendment_after: Path
    train_manifest: Path
    development_manifest: Path
    validation_manifest: Path
    confirmatory_manifest: Path
    checkpoint_template: str
    component_training_template: str
    component_checkpoint_template: str
    retrieval_bank_template: str
    counterexample_dataset_template: str
    counterexample_round_template: str
    result_template: str
    oracle_result_template: str
    run_root: Path
    protocol_lock: Path
    confirmation_power: Path
    confirmation_lock: Path
    confirmation_opened: Path
    confirmation_mapping: Path
    confirmation_schedule: Path
    confirmation_unblinded: Path
    schedule_dir: Path
    p2_selection: Path
    p5_advancement: Path
    p7_selection: Path
    p8_selection: Path


class ProtocolConfig(StrictModel):
    max_steps: int = Field(128, ge=1)
    seen_max_size: int = Field(21, ge=3)
    evaluation_seed: int = 42
    run_order_seed: int = 20260714
    action_ids: tuple[int, ...] = ACTION_IDS
    training_seeds: tuple[int, ...] = tuple(range(42, 62))
    planner_seeds: tuple[int, ...] = (104_729, 130_363)
    search_seeds: tuple[int, ...] = (155_921, 196_613)
    primary_action_selection: Literal["corrected_v1"] = "corrected_v1"
    paired_action_selection: Literal["unmasked"] = "unmasked"
    allow_confirmatory_model_selection: Literal[False] = False
    allow_score_triggered_reruns: Literal[False] = False
    validation_count: int = Field(700, ge=1)
    confirmatory_count: int = Field(900, ge=1)

    @field_validator("action_ids")
    @classmethod
    def validate_action_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != ACTION_IDS:
            raise ValueError(f"the locked action space is exactly {ACTION_IDS}")
        return value

    @field_validator("training_seeds")
    @classmethod
    def validate_training_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 8 or len(set(value)) != len(value):
            raise ValueError(
                "training_seeds are backbone seeds and require at least eight "
                "unique values"
            )
        return value

    @field_validator("planner_seeds", "search_seeds")
    @classmethod
    def validate_nested_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != 2 or len(set(value)) != 2:
            raise ValueError("planner and search seed levels require two unique values")
        return value


class BudgetConfig(StrictModel):
    multiplier: Literal[0.5, 1.0, 4.0, 16.0] = 1.0
    reference_transitions: int = Field(768, ge=1)
    hard_limit: bool = True

    @property
    def transition_limit(self) -> int:
        return int(round(self.multiplier * self.reference_transitions))


class PlannerConfig(StrictModel):
    kind: PlannerKind
    rollout_semantics: RolloutSemantics = RolloutSemantics.LEGACY_WARMUP_V1
    horizon: int = Field(12, ge=1)
    history_size: int = Field(3, ge=2)
    num_candidates: int = Field(64, ge=1)
    num_elites: int = Field(8, ge=1)
    cem_iters: int = Field(1, ge=1)
    momentum: float = Field(0.1, ge=0.0, lt=1.0)
    beam_width: int = Field(16, ge=1)
    exploration_constant: float = Field(1.4, ge=0.0)
    elite_reuse_fraction: float = Field(0.25, ge=0.0, le=0.75)
    proposal_seed_fraction: Literal[0.25] = 0.25
    diversity_penalty: float = Field(0.0, ge=0.0)
    budget: BudgetConfig = BudgetConfig()

    @model_validator(mode="after")
    def validate_elites(self) -> PlannerConfig:
        if self.num_elites > self.num_candidates:
            raise ValueError("num_elites cannot exceed num_candidates")
        if self.kind == PlannerKind.LEGACY_CEM:
            locked = (
                self.rollout_semantics == RolloutSemantics.LEGACY_WARMUP_V1
                and self.horizon == 12
                and self.history_size == 3
                and self.num_candidates == 64
                and self.num_elites == 8
                and self.cem_iters == 1
                and self.momentum == 0.1
                and self.budget.multiplier == 1.0
            )
            if not locked:
                raise ValueError("legacy_cem is the exact historical B0 configuration")
        if self.kind == PlannerKind.CATEGORICAL_CEM and (
            self.num_candidates * self.horizon * self.cem_iters
            > self.budget.transition_limit
        ):
            raise ValueError("categorical CEM exceeds its predictor-transition budget")
        return self


class ScorerConfig(StrictModel):
    goal_l2_weight: float = Field(1.0, ge=0.0)
    verifier_weight: float = Field(0.0, ge=0.0)
    reachability_weight: float = Field(0.0, ge=0.0)
    counterexample_ranker_weight: float = Field(0.0, ge=0.0)
    verifier_scale_min: float = Field(0.1, gt=0.0)
    verifier_scale_max: float = Field(10.0, gt=0.0)
    eps: float = Field(1e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_scale(self) -> ScorerConfig:
        if self.verifier_scale_min > self.verifier_scale_max:
            raise ValueError("verifier scale bounds are reversed")
        if (
            self.goal_l2_weight
            + self.verifier_weight
            + self.reachability_weight
            + self.counterexample_ranker_weight
            <= 0.0
        ):
            raise ValueError("at least one scorer term must be active")
        return self


class ProposalConfig(StrictModel):
    kind: ProposalKind = ProposalKind.UNIFORM
    uniform_weight: float = Field(1.0, ge=0.25, le=1.0)
    retrieval_weight: float = Field(0.0, ge=0.0, le=0.75)
    learned_weight: float = Field(0.0, ge=0.0, le=0.75)
    temperature: float = Field(1.0, gt=0.0)
    retrieval_top_k: int = Field(32, ge=1)
    denoising_steps: int = Field(8, ge=1)

    @model_validator(mode="after")
    def validate_mixture(self) -> ProposalConfig:
        total = self.uniform_weight + self.retrieval_weight + self.learned_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("proposal mixture weights must sum to one")
        if self.kind == ProposalKind.UNIFORM and self.uniform_weight != 1.0:
            raise ValueError("uniform proposal must have weight one")
        return self


class MemoryConfig(StrictModel):
    enabled: bool = False
    join_threshold: float = Field(0.95, ge=0.0, le=1.0)
    required_validation_precision: float = Field(0.95, ge=0.95, le=1.0)
    dominance_delta: float = Field(0.0, ge=0.0)
    hard_pruning: bool = False
    soft_priority_penalty: float = Field(1.0, ge=0.0)


class ControlConfig(StrictModel):
    verifier_targets: Literal[
        "true", "action_shuffle", "pair_shuffle", "random_untrained"
    ] = "true"
    predictor_association: Literal["true", "candidate_shuffle"] = "true"
    proposal_execution: Literal["search", "proposal_only"] = "search"
    dts_expansion: Literal["learned", "direct", "random", "fixed_breadth"] = "learned"
    ranker_negatives: Literal["hard_three_rounds", "random"] = "hard_three_rounds"


class JointHyperparameters(StrictModel):
    planner_learning_rate: Literal[0.0001, 0.0003]
    backbone_lr_multiplier: Literal[0.01, 0.03, 0.1]
    planner_loss_weight: Literal[0.1, 0.3, 1.0]
    sigreg_multiplier: Literal[0.5, 1.0, 2.0]


class JointGridConfig(StrictModel):
    planner_learning_rates: tuple[Literal[0.0001, 0.0003], ...] = (
        0.0001,
        0.0003,
    )
    backbone_lr_multipliers: tuple[Literal[0.01, 0.03, 0.1], ...] = (
        0.01,
        0.03,
        0.1,
    )
    planner_loss_weights: tuple[Literal[0.1, 0.3, 1.0], ...] = (0.1, 0.3, 1.0)
    sigreg_multipliers: tuple[Literal[0.5, 1.0, 2.0], ...] = (0.5, 1.0, 2.0)

    @model_validator(mode="after")
    def validate_complete_grid(self) -> JointGridConfig:
        expected = {
            "planner_learning_rates": (0.0001, 0.0003),
            "backbone_lr_multipliers": (0.01, 0.03, 0.1),
            "planner_loss_weights": (0.1, 0.3, 1.0),
            "sigreg_multipliers": (0.5, 1.0, 2.0),
        }
        for name, values in expected.items():
            if tuple(getattr(self, name)) != values:
                raise ValueError(f"joint grid drifted: {name}")
        return self


class MethodConfig(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    stage: Literal["P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    track: Literal["F", "J"]
    planner: PlannerConfig
    scorer: ScorerConfig = ScorerConfig()
    proposal: ProposalConfig = ProposalConfig()
    memory: MemoryConfig = MemoryConfig()
    control: ControlConfig = ControlConfig()
    component_checkpoint_required: bool = False
    confirmatory_eligible: bool = False
    initialization_parent: str | None = None
    reuse_component_from: str | None = None
    adaptive_role: Literal[
        "static",
        "p2_selected_planner",
        "p5_selected_architecture",
        "p8_selected_parent",
    ] = "static"
    joint_hyperparameters: JointHyperparameters | None = None
    effective_decision_sha256s: tuple[str, ...] = ()
    trainable_components: (
        tuple[
            Literal[
                "verifier",
                "reachability",
                "join",
                "autoregressive_proposal",
                "denoising_proposal",
                "dts",
                "ranker",
            ],
            ...,
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_baseline_boundary(self) -> MethodConfig:
        if self.planner.kind == PlannerKind.LEGACY_CEM:
            if self.track != "F":
                raise ValueError("the historical B0 baseline is Track F")
            if self.scorer != ScorerConfig():
                raise ValueError("the historical B0 baseline uses latent L2 only")
            if self.proposal != ProposalConfig():
                raise ValueError("the historical B0 baseline uses uniform proposals")
            if self.memory.enabled or self.component_checkpoint_required:
                raise ValueError(
                    "the historical B0 baseline has no learned planner add-on"
                )
            if self.control != ControlConfig():
                raise ValueError("the historical B0 baseline has no control mutation")
        if self.trainable_components == () and self.initialization_parent is None:
            raise ValueError(
                "a fully frozen component set requires a parent checkpoint"
            )
        if self.reuse_component_from is not None:
            if self.stage not in {"P7", "P8"}:
                raise ValueError("only P7/P8 may reuse a frozen component checkpoint")
            if not self.component_checkpoint_required:
                raise ValueError("component reuse requires a component checkpoint")
            if self.initialization_parent is not None:
                raise ValueError(
                    "component reuse and training inheritance are exclusive"
                )
            if self.trainable_components is not None:
                raise ValueError("a checkpoint reuse alias cannot train any component")
        if self.stage == "P7" and self.track == "J":
            if self.joint_hyperparameters is None:
                raise ValueError(
                    "every Track J grid cell needs explicit hyperparameters"
                )
        elif self.joint_hyperparameters is not None:
            raise ValueError("joint hyperparameters are valid only for P7 Track J")
        return self


class TrainingConfig(StrictModel):
    verifier_steps: int = Field(30_000, ge=1)
    reachability_steps: int = Field(30_000, ge=1)
    join_steps: int = Field(30_000, ge=1)
    proposal_steps: int = Field(60_000, ge=1)
    denoising_steps: int = Field(100_000, ge=1)
    dts_steps: int = Field(100_000, ge=1)
    ranker_initial_steps: int = Field(30_000, ge=1)
    joint_steps: int = Field(30_000, ge=1)
    transition_batch_size: int = Field(512, ge=1)
    proposal_batch_size: int = Field(256, ge=1)
    dts_batch_size: int = Field(64, ge=1)
    joint_batch_size: int = Field(128, ge=1)
    sequence_length: int = Field(8, ge=2)
    verifier_learning_rate: float = Field(3e-4, gt=0.0)
    reachability_learning_rate: float = Field(3e-4, gt=0.0)
    join_learning_rate: float = Field(3e-4, gt=0.0)
    proposal_learning_rate: float = Field(1e-4, gt=0.0)
    dts_learning_rate: float = Field(1e-4, gt=0.0)
    ranker_learning_rate: float = Field(3e-4, gt=0.0)
    joint_planner_learning_rate: float = Field(3e-4, gt=0.0)
    joint_backbone_lr_multiplier: Literal[0.01, 0.03, 0.1] = 0.03
    joint_planner_loss_weight: Literal[0.1, 0.3, 1.0] = 0.3
    warmup_fraction: Literal[0.05] = 0.05
    final_learning_rate_ratio: Literal[0.1] = 0.1
    weight_decay: float = Field(1e-4, ge=0.0)
    grad_clip: float = Field(1.0, gt=0.0)
    checkpoint_selection: Literal["final_step"] = "final_step"
    counterexample_rounds: Literal[3] = 3
    counterexample_round_steps: int = Field(20_000, ge=1)
    calibration_batches: int = Field(32, ge=1)
    calibration_seed: int = 20260714
    retrieval_bank_chunks: int = Field(8192, ge=1)
    reachability_bins: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    sigreg_weight: float = Field(0.09, ge=0.0)
    prediction_weight: float = Field(1.0, ge=0.0)
    abs_position_weight: float = Field(0.1, ge=0.0)
    relative_position_weight: float = Field(1.0, ge=0.0)
    goal_position_weight: float = Field(0.5, ge=0.0)


class AnalysisConfig(StrictModel):
    primary_metric: Literal["success"] = "success"
    secondary_metrics: tuple[str, ...] = ("spl",)
    bootstrap_samples: int = Field(20_000, ge=1_000)
    bootstrap_seed: int = 20260714
    familywise_alpha: float = Field(0.05, gt=0.0, lt=1.0)
    multiplicity_method: Literal["bonferroni_simultaneous_percentile_ci"] = (
        "bonferroni_simultaneous_percentile_ci"
    )
    task_strata_key: Literal["maze_size"] = "maze_size"
    seed_resampling: Literal["paired_same_backbone"] = "paired_same_backbone"
    decision_rule_min_delta: float = 0.03
    candidate_trace_fraction: Literal[0.1] = 0.1
    candidate_trace_k: Literal[64] = 64
    candidate_prefix_lengths: tuple[int, ...] = (2, 4, 8)

    @field_validator("candidate_prefix_lengths")
    @classmethod
    def validate_candidate_prefix_lengths(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if value != (2, 4, 8):
            raise ValueError("candidate prefix diagnostics are locked to (2, 4, 8)")
        return value


class StudyConfig(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: Literal[PROTOCOL_ID] = PROTOCOL_ID
    study_role: Literal["preregistered_planner_frontier"]
    device: str = "auto"
    paths: PathsConfig
    protocol: ProtocolConfig = ProtocolConfig()
    training: TrainingConfig = TrainingConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    joint_grid: JointGridConfig = JointGridConfig()
    methods: tuple[MethodConfig, ...]

    @model_validator(mode="before")
    @classmethod
    def expand_joint_grid(cls, value: Any) -> Any:
        """Expand the one checked-in P7 template into the full 2x3x3x3 grid."""

        if not isinstance(value, dict) or not isinstance(value.get("methods"), list):
            return value
        methods = copy.deepcopy(value["methods"])
        adaptive_by_stage = {
            "P3": "p2_selected_planner",
            "P5": "p5_selected_architecture",
            "P6": "p5_selected_architecture",
            "P7": "p5_selected_architecture",
            "P8": "p8_selected_parent",
        }
        for method in methods:
            role = adaptive_by_stage.get(str(method.get("stage")))
            if role is not None:
                method.setdefault("adaptive_role", role)
        track_j = [
            method
            for method in methods
            if method.get("stage") == "P7" and method.get("track") == "J"
        ]
        if len(track_j) > 1:
            return value
        if len(track_j) != 1 or track_j[0].get("name") != "p7_track_j_joint_all":
            raise ValueError("configuration needs exactly one canonical P7 template")
        raw_grid = value.get("joint_grid", {})
        planner_lrs = tuple(raw_grid.get("planner_learning_rates", (0.0001, 0.0003)))
        backbone_multipliers = tuple(
            raw_grid.get("backbone_lr_multipliers", (0.01, 0.03, 0.1))
        )
        planner_weights = tuple(raw_grid.get("planner_loss_weights", (0.1, 0.3, 1.0)))
        sigreg_multipliers = tuple(raw_grid.get("sigreg_multipliers", (0.5, 1.0, 2.0)))
        template_index = methods.index(track_j[0])
        expanded: list[dict[str, Any]] = []
        for lr_index, backbone_index, weight_index, sigreg_index in itertools.product(
            range(len(planner_lrs)),
            range(len(backbone_multipliers)),
            range(len(planner_weights)),
            range(len(sigreg_multipliers)),
        ):
            method = copy.deepcopy(track_j[0])
            hyperparameters = {
                "planner_learning_rate": planner_lrs[lr_index],
                "backbone_lr_multiplier": backbone_multipliers[backbone_index],
                "planner_loss_weight": planner_weights[weight_index],
                "sigreg_multiplier": sigreg_multipliers[sigreg_index],
            }
            center = hyperparameters == {
                "planner_learning_rate": 0.0003,
                "backbone_lr_multiplier": 0.03,
                "planner_loss_weight": 0.3,
                "sigreg_multiplier": 1.0,
            }
            if not center:
                method["name"] = (
                    "p7_track_j_g"
                    f"{lr_index}{backbone_index}{weight_index}{sigreg_index}"
                )
            method["joint_hyperparameters"] = hyperparameters
            method["adaptive_role"] = "p5_selected_architecture"
            expanded.append(method)
        methods[template_index : template_index + 1] = expanded
        output = dict(value)
        output["methods"] = methods
        return output

    @field_validator("methods")
    @classmethod
    def validate_methods(
        cls, value: tuple[MethodConfig, ...]
    ) -> tuple[MethodConfig, ...]:
        names = [method.name for method in value]
        if len(names) != len(set(names)):
            raise ValueError("method names must be unique")
        baselines = [
            method for method in value if method.planner.kind == PlannerKind.LEGACY_CEM
        ]
        if len(baselines) != 1:
            raise ValueError(
                "the matrix must contain exactly one historical B0 baseline"
            )
        by_name = {method.name: method for method in value}
        stage_index = {f"P{index}": index for index in range(2, 9)}
        for method in value:
            if method.initialization_parent is not None:
                if method.initialization_parent not in by_name:
                    raise ValueError(
                        f"unknown initialization parent for {method.name}: "
                        f"{method.initialization_parent}"
                    )
                parent = by_name[method.initialization_parent]
                if stage_index[parent.stage] >= stage_index[method.stage]:
                    raise ValueError(
                        "component parents must come from an earlier stage"
                    )
            if method.reuse_component_from is not None:
                if method.reuse_component_from not in by_name:
                    raise ValueError(
                        f"unknown component reuse source for {method.name}: "
                        f"{method.reuse_component_from}"
                    )
                source = by_name[method.reuse_component_from]
                allowed_source_stages = (
                    {"P6"} if method.stage == "P7" else {"P5", "P6", "P7"}
                )
                if source.stage not in allowed_source_stages:
                    raise ValueError(
                        f"invalid {method.stage} component reuse source: {source.stage}"
                    )
                if not source.component_checkpoint_required:
                    raise ValueError(
                        "a checkpoint alias cannot reuse a headless method"
                    )
                for field in ("track", "scorer", "proposal", "memory", "control"):
                    if getattr(method, field) != getattr(source, field):
                        raise ValueError(
                            f"checkpoint reuse alias may not change {field}: "
                            f"{method.name}"
                        )
                alias_planner = method.planner.model_dump(mode="json")
                source_planner = source.planner.model_dump(mode="json")
                varying_field = (
                    "rollout_semantics" if method.stage == "P7" else "budget"
                )
                alias_planner.pop(varying_field)
                source_planner.pop(varying_field)
                if alias_planner != source_planner:
                    raise ValueError(
                        f"{method.stage} reuse alias changed more than "
                        f"{varying_field}: "
                        f"{method.name}"
                    )
                if method.stage == "P7" and (
                    source.planner.rollout_semantics
                    != RolloutSemantics.LEGACY_WARMUP_V1
                    or method.planner.rollout_semantics
                    != RolloutSemantics.ACTION_ALIGNED_V2
                ):
                    raise ValueError("P7 reuse control must isolate action alignment")
        required_chain = {
            "p5_track_f_all_hard_memory": "p3_factorial_v1r1p1m1",
            "p6_track_f_counterexample_ranked": "p5_track_f_all_hard_memory",
        }
        for child, parent in required_chain.items():
            if by_name[child].initialization_parent != parent:
                raise ValueError(f"{child} must inherit the locked {parent} checkpoint")
        joint_methods = [
            method for method in value if method.stage == "P7" and method.track == "J"
        ]
        if len(joint_methods) != 54:
            raise ValueError("P7 must expand to the complete 54-cell joint grid")
        if any(
            method.initialization_parent != "p6_track_f_counterexample_ranked"
            for method in joint_methods
        ):
            raise ValueError("every Track J cell must inherit the locked P6 checkpoint")
        return value


__all__ = [
    "AnalysisConfig",
    "BudgetConfig",
    "ControlConfig",
    "JointGridConfig",
    "JointHyperparameters",
    "MemoryConfig",
    "MethodConfig",
    "PathsConfig",
    "PlannerConfig",
    "PlannerKind",
    "ProposalConfig",
    "ProposalKind",
    "ProtocolConfig",
    "RolloutSemantics",
    "ScorerConfig",
    "StudyConfig",
    "TrainingConfig",
]
