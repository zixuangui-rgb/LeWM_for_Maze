"""Strict schemas for the full-900 paired planner screen."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from vector_jepa_planner_frontier.schemas import (
    AnalysisConfig,
    MethodConfig,
    PathsConfig,
    PlannerKind,
    ProtocolConfig,
    RolloutSemantics,
    StrictModel,
    TrainingConfig,
)
from vector_jepa_planner_full900_screen import PROTOCOL_ID

Phase = Literal["Q0", "Q1", "Q2A", "Q2B", "Q2C"]
Role = Literal["baseline", "candidate", "bridge_control", "matched_control"]
Parent = Literal["fixed", "q1_winner", "q1_best_first"]


class ReplicationConfig(StrictModel):
    """Predeclared seed escalation; every tier still uses all 900 tasks."""

    evaluation_manifest_role: Literal["development"] = "development"
    task_count: Literal[900] = 900
    screen_backbone_seeds: tuple[int, ...] = (42,)
    expansion_backbone_seeds: tuple[int, ...] = (42, 43, 44)
    final_backbone_seeds: tuple[int, ...] = tuple(range(42, 52))
    screen_planner_seeds: tuple[int, ...] = (104_729,)
    final_planner_seeds: tuple[int, ...] = (104_729, 130_363)
    action_selections: tuple[Literal["corrected_v1", "unmasked"], ...] = (
        "corrected_v1",
        "unmasked",
    )

    @model_validator(mode="after")
    def validate_exact_schedule(self) -> ReplicationConfig:
        expected = {
            "screen_backbone_seeds": (42,),
            "expansion_backbone_seeds": (42, 43, 44),
            "final_backbone_seeds": tuple(range(42, 52)),
            "screen_planner_seeds": (104_729,),
            "final_planner_seeds": (104_729, 130_363),
            "action_selections": ("corrected_v1", "unmasked"),
        }
        for field, value in expected.items():
            if tuple(getattr(self, field)) != value:
                raise ValueError(f"locked replication schedule drifted: {field}")
        return self


class GateConfig(StrictModel):
    """Exploratory advancement gates fixed before any new score is observed."""

    system_min_delta_sr: Literal[0.03] = 0.03
    mechanism_min_delta_sr: Literal[0.02] = 0.02
    max_protocol_regression_sr: Literal[0.03] = 0.03
    max_ood_regression_sr: Literal[0.03] = 0.03
    required_positive_backbones: Literal[2] = 2
    expansion_backbone_count: Literal[3] = 3
    max_shortlist_size: Literal[2] = 2
    bonferroni_comparison_count: Literal[48] = 48
    screen_interval_engine: Literal["exact_stratified_empirical_bootstrap"] = (
        "exact_stratified_empirical_bootstrap"
    )
    corrected_is_parent_selection_metric: Literal[True] = True
    blend_action_protocols: Literal[False] = False


class MethodRoleConfig(StrictModel):
    """Scientific role and direct control for one executable method."""

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    phase: Phase
    role: Role
    parent: Parent = "fixed"
    direct_control: str
    advancement_eligible: bool
    mechanism: str = Field(min_length=3)


class QuickStudyConfig(StrictModel):
    """A small method matrix with paper-grade compatibility invariants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_id: Literal[PROTOCOL_ID] = PROTOCOL_ID
    study_role: Literal["paired_full900_exploratory_method_screen"]
    device: str = "auto"
    paths: PathsConfig
    protocol: ProtocolConfig
    replication: ReplicationConfig = ReplicationConfig()
    gates: GateConfig = GateConfig()
    training: TrainingConfig
    analysis: AnalysisConfig
    methods: tuple[MethodConfig, ...]
    method_roles: tuple[MethodRoleConfig, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> QuickStudyConfig:
        if self.protocol.training_seeds != tuple(range(42, 52)):
            raise ValueError("the screen must reuse historical backbone seeds 42-51")
        if self.protocol.planner_seeds != (104_729, 130_363):
            raise ValueError(
                "planner-head seeds must match the locked frontier protocol"
            )
        if self.protocol.max_steps != 128 or self.protocol.evaluation_seed != 42:
            raise ValueError("legacy max_steps/evaluation_seed drifted")

        methods = {method.name: method for method in self.methods}
        roles = {role.name: role for role in self.method_roles}
        if len(methods) != len(self.methods) or len(roles) != len(self.method_roles):
            raise ValueError("method and role names must be unique")
        if set(methods) != set(roles):
            raise ValueError("every method needs exactly one scientific role")

        expected_names = {
            "b0_legacy_l2_cem",
            "q1_control_categorical_cem_1x",
            "q1_icem_1x",
            "q1_beam_1x",
            "q1_best_first_1x",
            "q1_mcts_1x",
            "q2a_reachability",
            "q2a_verifier",
            "q2a_autoregressive_proposal",
            "q2a_transposition_memory",
            "q2b_vector_dts",
            "q2b_control_dts_direct",
            "q2b_control_dts_uniform_expansion",
            "q2b_bidirectional",
            "q2b_control_bidirectional_forward",
            "q2b_denoising_icem",
            "q2b_control_denoising_uniform",
            "q2c_hard_negative_ranker",
            "q2c_control_random_negative_ranker",
        }
        if set(methods) != expected_names:
            raise ValueError("quick screen method family is incomplete or has drifted")

        baselines = [
            method
            for method in self.methods
            if method.planner.kind == PlannerKind.LEGACY_CEM
        ]
        if [method.name for method in baselines] != ["b0_legacy_l2_cem"]:
            raise ValueError("there must be one exact historical B0")
        if any(method.track != "F" for method in self.methods):
            raise ValueError("quick screening is frozen-backbone Track F only")
        if any(
            method.planner.rollout_semantics != RolloutSemantics.LEGACY_WARMUP_V1
            for method in self.methods
        ):
            raise ValueError(
                "quick screening must preserve historical rollout semantics"
            )
        if any(method.planner.budget.multiplier != 1.0 for method in self.methods):
            raise ValueError("all quick-screen planners use the same 1x budget")

        q1_kinds = {
            methods[name].planner.kind
            for name, role in roles.items()
            if role.phase == "Q1" and role.role == "candidate"
        }
        if q1_kinds != {
            PlannerKind.ICEM,
            PlannerKind.BEAM,
            PlannerKind.BEST_FIRST,
            PlannerKind.MCTS,
        }:
            raise ValueError("Q1 must contain the four preregistered search candidates")
        eligible = {role.name for role in roles.values() if role.advancement_eligible}
        if len(eligible) != 12:
            raise ValueError(
                "exactly twelve scientific candidate families must advance"
            )
        if roles["q2a_transposition_memory"].parent != "q1_best_first":
            raise ValueError(
                "memory must be contrasted with the planner that consumes it"
            )
        for name, role in roles.items():
            if role.parent == "q1_winner" and role.phase not in {"Q2A", "Q2C"}:
                raise ValueError(f"invalid adaptive parent for {name}")
            if (
                role.direct_control not in methods
                and role.direct_control != "__q1_winner__"
            ):
                raise ValueError(f"unknown direct control for {name}")
        return self


__all__ = [
    "GateConfig",
    "MethodRoleConfig",
    "QuickStudyConfig",
    "ReplicationConfig",
]
