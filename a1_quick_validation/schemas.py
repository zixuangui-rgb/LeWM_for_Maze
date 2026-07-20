"""Strict schema and invariants for the fixed quick-validation execution profile."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from a1_quick_validation import (
    ALL_METHODS,
    NEW_METHODS,
    PROFILE_ID,
    PROFILE_SCHEMA,
    REFERENCE_METHODS,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfilePaths(StrictModel):
    quick_config: str
    quick_protocol_lock: str
    package_lock: str
    reproduction_contract: str
    source_config: str
    source_protocol_lock: str
    source_run_root: str
    run_root: str


class PhaseSpec(StrictModel):
    split_role: Literal["screen", "select", "legacy"]
    backbone_seeds: tuple[int, ...]
    head_seeds: tuple[int, ...]
    action_protocols: tuple[Literal["corrected_v1", "unmasked"], ...]
    methods: tuple[str, ...]
    dynamic_method_source: Literal["none", "q1_shortlist", "q2_winner"]
    run_diagnostics: bool


class Q1Thresholds(StrictModel):
    sr_safety_delta: Literal[-0.01] = -0.01
    local_top1_delta: Literal[0.03] = 0.03
    regret_relative_reduction: Literal[0.15] = 0.15
    reachability_min_auroc: Literal[0.65] = 0.65
    reachability_max_brier: Literal[0.25] = 0.25
    reachability_max_ece: Literal[0.15] = 0.15
    reachability_max_monotonic_violation: Literal[0.05] = 0.05
    max_promoted_new_methods: Literal[2] = 2


class Q2Thresholds(StrictModel):
    minimum_mean_sr_delta: Literal[0.02] = 0.02
    minimum_each_head_sr_delta: Literal[0.0] = 0.0
    maximum_secondary_drop: Literal[0.02] = 0.02


class Q3Thresholds(StrictModel):
    minimum_overall_sr_delta: Literal[0.02] = 0.02
    maximum_secondary_drop: Literal[0.02] = 0.02
    require_seen_and_ood_nonnegative: Literal[True] = True


class QuickProfile(StrictModel):
    schema_name: Literal[PROFILE_SCHEMA] = Field(PROFILE_SCHEMA, alias="schema")
    profile_id: Literal[PROFILE_ID] = PROFILE_ID
    evidence_status: Literal["exploratory_fast_validation"]
    worker_count: Literal[4] = 4
    paths: ProfilePaths
    q1: PhaseSpec
    q2: PhaseSpec
    q3: PhaseSpec
    q1_thresholds: Q1Thresholds = Q1Thresholds()
    q2_thresholds: Q2Thresholds = Q2Thresholds()
    q3_thresholds: Q3Thresholds = Q3Thresholds()
    reference_methods: tuple[str, ...]
    new_methods: tuple[str, ...]
    no_adaptive_training_budget: Literal[True] = True
    no_test_bfs_in_action_selection: Literal[True] = True
    full900_is_exploratory_not_confirmatory: Literal[True] = True
    claim_boundary: str = Field(min_length=20)

    @model_validator(mode="after")
    def validate_locked_matrix(self) -> QuickProfile:
        if self.reference_methods != REFERENCE_METHODS:
            raise ValueError("reference method order is locked")
        if self.new_methods != NEW_METHODS:
            raise ValueError("new method order is locked")
        expected = {
            "q1": PhaseSpec(
                split_role="screen",
                backbone_seeds=(42,),
                head_seeds=(0,),
                action_protocols=("corrected_v1",),
                methods=ALL_METHODS,
                dynamic_method_source="none",
                run_diagnostics=True,
            ),
            "q2": PhaseSpec(
                split_role="select",
                backbone_seeds=(42,),
                head_seeds=(0, 1),
                action_protocols=("corrected_v1", "unmasked"),
                methods=("b_dh_cem", "a1_log"),
                dynamic_method_source="q1_shortlist",
                run_diagnostics=True,
            ),
            "q3": PhaseSpec(
                split_role="legacy",
                backbone_seeds=(42,),
                head_seeds=(0,),
                action_protocols=("corrected_v1", "unmasked"),
                methods=REFERENCE_METHODS,
                dynamic_method_source="q2_winner",
                run_diagnostics=False,
            ),
        }
        observed = {"q1": self.q1, "q2": self.q2, "q3": self.q3}
        for name, spec in expected.items():
            if observed[name] != spec:
                raise ValueError(f"{name} execution matrix differs from the lock")
        return self


__all__ = [
    "PhaseSpec",
    "ProfilePaths",
    "Q1Thresholds",
    "Q2Thresholds",
    "Q3Thresholds",
    "QuickProfile",
]
