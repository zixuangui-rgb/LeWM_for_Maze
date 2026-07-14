from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import vector_jepa_planner_frontier.effective_methods as effective_methods
import vector_jepa_planner_frontier.stage_gates as stage_gates
from final_closure.common import next_state, read_jsonl
from final_closure.evaluate import LeWMCEMController
from hdwm.planning import _latent_rollout_cost, cem_plan
from spatial_jepa_planning.common import observe_state, validate_manifest_entry
from vector_jepa_planner_frontier import ACTION_IDS
from vector_jepa_planner_frontier.calibrate import (
    binary_auroc,
    binary_reliability,
    select_precision_threshold,
)
from vector_jepa_planner_frontier.common import (
    ComputeLedger,
    component_checkpoint_owner,
    component_checkpoint_path,
    load_json,
    load_study_config,
    validate_compute_ledger,
    validate_manifest_isolation,
)
from vector_jepa_planner_frontier.counterexamples import execute_candidate, mining_fold
from vector_jepa_planner_frontier.data import (
    CounterexampleBatchSampler,
    JEPATrajectorySampler,
    PlannerBatchSampler,
    encode_planner_batch,
)
from vector_jepa_planner_frontier.evaluate import (
    CandidateTraceSink,
    FrontierController,
    _normalized_edit_distance,
    _path_revisit_metrics,
    analyze_candidate_result,
    candidate_trace_selected,
    exact_candidate_trace_keys,
)
from vector_jepa_planner_frontier.freeze_p2_selection import select_winner
from vector_jepa_planner_frontier.freeze_p5_advancement import (
    _validate_evidence,
    select_radical,
)
from vector_jepa_planner_frontier.freeze_p7_selection import select_joint_winner
from vector_jepa_planner_frontier.frontier_selection import (
    FRONTIER_BASES,
    FRONTIER_BUDGETS,
    aggregate_frontier_metrics,
    frontier_families,
    select_near_optimal_budget,
    select_track_f_family,
)
from vector_jepa_planner_frontier.heads import (
    ActionConsistencyVerifier,
    AutoregressiveProposal,
    DistributionalReachability,
    HeadConfig,
    StateJoinHead,
    VectorDTSHead,
    required_head_names,
)
from vector_jepa_planner_frontier.oracle_ladder import ORACLES, OracleController
from vector_jepa_planner_frontier.planners import (
    CandidateBatch,
    CompositeScorer,
    LegacyCEMPlanner,
    PlannerResult,
    build_planner,
)
from vector_jepa_planner_frontier.power_analysis import required_backbone_count
from vector_jepa_planner_frontier.proposals import (
    MixtureProposal,
    RetrievalBank,
    RetrievalProposal,
    UniformProposal,
)
from vector_jepa_planner_frontier.run_plan import (
    blocked_oracle_jobs,
    blocked_stage_jobs,
    component_jobs,
    evaluation_jobs,
    selected_methods,
    stage_schedule_text,
)
from vector_jepa_planner_frontier.schemas import (
    BudgetConfig,
    MethodConfig,
    PlannerConfig,
    PlannerKind,
    ProposalConfig,
    ProposalKind,
    RolloutSemantics,
    ScorerConfig,
    StudyConfig,
)
from vector_jepa_planner_frontier.smoke_test import confirmatory_gate_record
from vector_jepa_planner_frontier.summarize import (
    average_nested_task_rows,
    csv_text,
    exact_sign_flip_pvalue,
    nested_paired_bootstrap,
)
from vector_jepa_planner_frontier.train import (
    component_losses,
    component_stochastic_rngs,
    required_heads,
    rollout_training_batch,
)
from vector_jepa_planner_frontier.world_model import VectorContext, VectorWorldModel

ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder(nn.Module):
    def forward(self, observations: torch.Tensor, size: int) -> torch.Tensor:
        del size
        channel_means = observations.mean(dim=(-3, -2))
        return channel_means[..., :4]


class FakeProjector(nn.Module):
    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, None]:
        return encoded, None


class FakePredictor(nn.Module):
    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        increments = actions.to(embeddings.dtype).unsqueeze(-1)
        offsets = torch.tensor(
            [0.01, 0.02, 0.03, 0.04],
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return embeddings[:, 1:] + increments * offsets


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.embedding_projector = FakeProjector()
        self.predictor = FakePredictor()


def context() -> VectorContext:
    embeddings = torch.tensor([[[0.0, 0.1, 0.2, 0.3]]]).repeat(1, 3, 1)
    return VectorContext(
        embeddings=embeddings,
        actions=torch.full((1, 3), 4, dtype=torch.long),
        goal=torch.tensor([[[0.4, 0.5, 0.6, 0.7]]]),
        maze_size=11,
    )


def test_default_config_contains_exact_b0_and_complete_factorial() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    b0 = [
        method
        for method in config.methods
        if method.planner.kind == PlannerKind.LEGACY_CEM
    ]
    assert len(config.methods) == 118
    assert config.protocol.primary_action_selection == "corrected_v1"
    assert config.protocol.paired_action_selection == "unmasked"
    assert len(config.protocol.training_seeds) == 20
    assert len(config.protocol.planner_seeds) == 2
    assert len(config.protocol.search_seeds) == 2
    assert [method.name for method in b0] == ["b0_legacy_l2_cem"]
    codes = {
        method.name.rsplit("_", 1)[-1]
        for method in config.methods
        if method.name.startswith("p3_factorial_")
    }
    assert codes == {
        f"v{v}r{r}p{p}m{m}"
        for v in (0, 1)
        for r in (0, 1)
        for p in (0, 1)
        for m in (0, 1)
    }
    assert all(
        method.planner.budget.multiplier == 4.0
        for method in config.methods
        if method.stage == "P3"
    )
    p2_budget_matrix = {
        (
            PlannerKind.CATEGORICAL_CEM
            if method.planner.kind == PlannerKind.LEGACY_CEM
            else method.planner.kind,
            method.planner.budget.multiplier,
        )
        for method in config.methods
        if method.stage == "P2"
    }
    assert p2_budget_matrix == {
        (kind, budget)
        for kind in (
            PlannerKind.CATEGORICAL_CEM,
            PlannerKind.ICEM,
            PlannerKind.BEAM,
            PlannerKind.BEST_FIRST,
            PlannerKind.MCTS,
        )
        for budget in (0.5, 1.0, 4.0, 16.0)
    }
    joint = [
        method
        for method in config.methods
        if method.stage == "P7" and method.track == "J"
    ]
    assert len(joint) == 54
    assert {
        (
            method.joint_hyperparameters.planner_learning_rate,
            method.joint_hyperparameters.backbone_lr_multiplier,
            method.joint_hyperparameters.planner_loss_weight,
            method.joint_hyperparameters.sigreg_multiplier,
        )
        for method in joint
    } == {
        (planner_lr, backbone_lr, planner_weight, sigreg)
        for planner_lr in (0.0001, 0.0003)
        for backbone_lr in (0.01, 0.03, 0.1)
        for planner_weight in (0.1, 0.3, 1.0)
        for sigreg in (0.5, 1.0, 2.0)
    }


def test_radical_stage_contains_matched_non_oracle_controls() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    p4 = {method.name for method in config.methods if method.stage == "P4"}
    assert p4 == {
        "p4_vector_dts",
        "p4_control_dts_direct",
        "p4_control_dts_random_expansion",
        "p4_control_dts_fixed_breadth",
        "p4_bidirectional",
        "p4_control_bidirectional_forward_only",
        "p4_denoising_icem",
        "p4_control_denoising_uniform",
        "p4_control_denoising_retrieval",
        "p4_control_denoising_proposal_only",
    }


def test_oracle_ladder_matches_protocol_and_valid_future_never_hits_a_wall() -> None:
    assert ORACLES == (
        "O0",
        "O1_PROP",
        "O2_SELECT",
        "O3_DYN",
        "O4_VALUE",
        "O5_JOIN",
        "O6_VALID_FUTURE",
    )
    entry = read_jsonl(ROOT / "data/splits/unisize_eval_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    start = int(entry["start_cell"])
    controller = OracleController(
        None,  # type: ignore[arg-type]
        oracle="O6_VALID_FUTURE",
        evaluation_seed=42,
        action_selection="corrected_v1",
    )
    candidates = controller._candidates(env, start, np.random.default_rng(123))
    for sequence in candidates:
        state = start
        for raw_action in sequence:
            successor = next_state(env, state, int(raw_action))
            assert successor != state
            state = successor


def test_oracle_schedule_is_complete_and_backbone_blocked() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    jobs = blocked_oracle_jobs(
        config,
        split_role="validation",
        config_path="config.json",
    )
    assert len(jobs) == 20 * len(ORACLES) * 2 * 2
    observed: list[int] = []
    for job in jobs:
        marker = job.label.split(":backbone", 1)[1].split(":", 1)[0]
        seed = int(marker)
        if not observed or observed[-1] != seed:
            assert seed not in observed
            observed.append(seed)
    assert set(observed) == set(config.protocol.training_seeds)
    assert all("oracle_ladder" in str(job.output) for job in jobs)


def test_finalists_inherit_the_exact_locked_parent_chain() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    methods = {method.name: method for method in config.methods}
    assert (
        methods["p5_track_f_all_hard_memory"].initialization_parent
        == "p3_factorial_v1r1p1m1"
    )
    assert methods["p5_track_f_all_hard_memory"].trainable_components == ()
    assert (
        methods["p6_track_f_counterexample_ranked"].initialization_parent
        == "p5_track_f_all_hard_memory"
    )
    assert methods["p6_track_f_counterexample_ranked"].trainable_components == (
        "ranker",
    )
    p7 = methods["p7_track_j_joint_all"]
    assert p7.initialization_parent == "p6_track_f_counterexample_ranked"
    assert p7.control.ranker_negatives == "hard_three_rounds"
    assert "ranker" in required_head_names(p7)
    aligned = methods["p7_control_action_aligned_frozen"]
    assert aligned.reuse_component_from == "p6_track_f_counterexample_ranked"
    assert aligned.trainable_components is None
    assert (
        component_checkpoint_owner(config, aligned)
        == methods["p6_track_f_counterexample_ranked"]
    )
    assert component_jobs(config, aligned, 42, 104729, "config.json") == []


def test_p6_hard_and_random_negative_controls_have_matched_round_budgets() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    methods = {method.name: method for method in config.methods}
    for name in (
        "p6_track_f_counterexample_ranked",
        "p6_control_random_negative_ranker",
    ):
        method = methods[name]
        jobs = component_jobs(config, method, 42, 104729, "config.json")
        rounds = [job for job in jobs if job.label.startswith("counterexample:")]
        assert len(rounds) == 3
        assert [f":round{index}" in rounds[index - 1].label for index in (1, 2, 3)] == [
            True,
            True,
            True,
        ]
        checkpoint = component_checkpoint_path(
            config,
            method,
            backbone_seed=42,
            planner_seed=104729,
        )
        assert checkpoint is not None
        assert checkpoint.name.endswith("_round3.pt")


def test_p8_is_a_checkpoint_reuse_only_compute_frontier() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    selected_track_j = "p7_track_j_joint_all"
    families = frontier_families(config, selected_track_j=selected_track_j)
    assert set(families) == {*FRONTIER_BASES, selected_track_j}
    assert all(set(family) == set(FRONTIER_BUDGETS) for family in families.values())
    for base_name, family in families.items():
        base = family[4.0]
        assert base.name == base_name
        for budget, method in family.items():
            if budget == 4.0:
                continue
            assert method.stage == "P8"
            assert method.reuse_component_from == base_name
            assert component_checkpoint_owner(config, method) == base
            assert component_checkpoint_path(
                config, method, backbone_seed=42, planner_seed=104729
            ) == component_checkpoint_path(
                config, base, backbone_seed=42, planner_seed=104729
            )
            assert component_jobs(config, method, 42, 104729, "config.json") == []


def test_p8_closes_track_j_frontier_after_grid_stability_failure() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    families = frontier_families(config, selected_track_j=None)
    assert set(families) == set(FRONTIER_BASES)
    assert all(set(family) == set(FRONTIER_BUDGETS) for family in families.values())


def test_dynamic_methods_follow_nondefault_frozen_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    configured = {method.name: method for method in config.methods}
    selected_p2 = next(
        method
        for method in config.methods
        if method.stage == "P2"
        and method.planner.kind == PlannerKind.MCTS
        and method.planner.budget.multiplier == 4.0
    )
    p2 = {"selected_4x_planner": selected_p2.planner.model_dump(mode="json")}
    p5 = {
        "selected_components": ["verifier", "memory"],
        "selected_radical": "bidirectional",
    }
    selected_p7 = "p7_track_j_g0000"
    monkeypatch.setattr(
        effective_methods,
        "_decision_sha256",
        lambda config, attribute: f"{attribute}:sha256",
    )
    monkeypatch.setattr(stage_gates, "validate_p2_selection", lambda config, lock: p2)
    monkeypatch.setattr(stage_gates, "validate_p5_advancement", lambda config, lock: p5)
    monkeypatch.setattr(
        stage_gates,
        "validate_p7_selection",
        lambda config, lock: {"selected_track_j": selected_p7},
    )

    p3 = effective_methods.resolve_effective_method(
        config,
        {},
        configured["p3_factorial_v1r0p0m0"],
    )
    assert p3.planner.kind == PlannerKind.MCTS
    assert p3.planner.budget.multiplier == 4.0

    derived_p5 = effective_methods.resolve_effective_method(
        config,
        {},
        configured["p5_track_f_all_hard_memory"],
    )
    assert derived_p5.initialization_parent == "p3_factorial_v1r0p0m1"
    assert derived_p5.planner.kind == PlannerKind.BIDIRECTIONAL
    assert derived_p5.memory.enabled is True
    assert derived_p5.scorer.reachability_weight == 0.0

    p8 = effective_methods.resolve_effective_method(
        config,
        {},
        configured["p8_p7_1x"],
    )
    assert p8.reuse_component_from == selected_p7
    assert p8.track == "J"
    assert p8.planner.kind == PlannerKind.BIDIRECTIONAL
    assert p8.planner.budget.multiplier == 1.0
    assert p8.joint_hyperparameters is None


def test_p5_radical_choice_is_reproducible_from_validation_summary() -> None:
    false_gates = {
        "mechanism_improved": False,
        "overall_noninferiority": False,
        "large_maze_noninferiority": False,
        "equal_compute": False,
        "negative_control_passed": False,
        "direction_consistency_passed": False,
        "evidence": "summary.json:fixture",
    }
    true_gates = {
        key: ("summary.json:fixture" if key == "evidence" else True)
        for key in false_gates
    }
    evidence = {
        "schema": "vector-jepa-p5-evidence-v1",
        "reviewer": "unit-test",
        "validation_results_viewed": True,
        "confirmatory_results_viewed": False,
        "selected_components": [],
        "selected_radical": "bidirectional",
        "component_gates": {
            name: dict(false_gates)
            for name in ("verifier", "reachability", "proposal", "memory")
        },
        "radical_gates": {
            "vector_dts": dict(true_gates),
            "bidirectional": dict(true_gates),
            "denoising": dict(false_gates),
        },
        "radical_decision_reason": "locked validation tie-break",
    }
    summary = {
        "primary": [
            {
                "method": "p4_vector_dts",
                "action_selection": "corrected_v1",
                "sr": 0.700,
            },
            {
                "method": "p4_bidirectional",
                "action_selection": "corrected_v1",
                "sr": 0.695,
            },
        ],
        "per_size": [
            {
                "method": method,
                "action_selection": "corrected_v1",
                "maze_size": size,
                "sr": sr,
            }
            for method, sr in (
                ("p4_vector_dts", 0.50),
                ("p4_bidirectional", 0.55),
            )
            for size in (19, 21)
        ],
    }
    _validate_evidence(evidence)
    selected, metrics = select_radical(
        evidence,
        summary,
        action_selection="corrected_v1",
    )
    assert selected == "bidirectional"
    assert {row["radical"] for row in metrics} == {
        "vector_dts",
        "bidirectional",
    }


def test_p7_grid_selection_fails_closed_or_prefers_stability_within_sr_band() -> None:
    assert (
        select_joint_winner(
            [
                {
                    "method": "unstable",
                    "all_checkpoints_stable": False,
                    "corrected_macro_sr": 0.9,
                }
            ]
        )
        is None
    )
    winner = select_joint_winner(
        [
            {
                "method": "higher_sr",
                "all_checkpoints_stable": True,
                "corrected_macro_sr": 0.80,
                "max_jepa_relative_change": 0.09,
                "mean_jepa_relative_change": 0.07,
                "corrected_size19_21_sr": 0.70,
            },
            {
                "method": "more_stable",
                "all_checkpoints_stable": True,
                "corrected_macro_sr": 0.795,
                "max_jepa_relative_change": 0.03,
                "mean_jepa_relative_change": 0.02,
                "corrected_size19_21_sr": 0.69,
            },
        ]
    )
    assert winner is not None
    assert winner["method"] == "more_stable"


def test_checkpoint_reuse_schema_rejects_causal_control_drift() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    payload = config.model_dump(mode="json")
    p8 = next(method for method in payload["methods"] if method["name"] == "p8_p6_1x")
    p8["scorer"]["verifier_weight"] = 0.4
    with pytest.raises(ValueError, match="may not change scorer"):
        StudyConfig.model_validate(payload)

    payload = config.model_dump(mode="json")
    aligned = next(
        method
        for method in payload["methods"]
        if method["name"] == "p7_control_action_aligned_frozen"
    )
    aligned["planner"]["budget"]["multiplier"] = 1.0
    with pytest.raises(ValueError, match="changed more than rollout_semantics"):
        StudyConfig.model_validate(payload)


def test_headless_retrieval_control_gets_a_reproducible_bank_job() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    method = next(
        method
        for method in config.methods
        if method.name == "p4_control_denoising_retrieval"
    )
    jobs = component_jobs(config, method, 42, 0, "config.json")
    assert [job.label.split(":", 1)[0] for job in jobs] == ["retrieval"]
    assert "planner0" in jobs[0].label


def test_blocked_schedule_never_reenters_a_completed_backbone() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    methods = selected_methods(config, "P2")
    jobs = blocked_stage_jobs(
        config,
        methods,
        stage="P2",
        split_role="validation",
        config_path="config.json",
    )
    observed: list[int] = []
    for job in jobs:
        marker = job.label.split(":backbone", 1)[1].split(":", 1)[0]
        seed = int(marker)
        if not observed or observed[-1] != seed:
            assert seed not in observed
            observed.append(seed)
    assert set(observed) == set(config.protocol.training_seeds)
    lock = load_json(ROOT / "vector_jepa_planner_frontier/configs/protocol_lock.json")
    for name in (
        "amendments",
        "amendment_document",
        "amendment_before",
        "amendment_after",
    ):
        lock[name] = {"sha256": "0" * 64}
    assert stage_schedule_text(
        config,
        lock,
        stage="P2",
        split_role="validation",
        jobs=jobs,
    ) == stage_schedule_text(
        config,
        lock,
        stage="P2",
        split_role="validation",
        jobs=jobs,
    )


def test_confirmatory_job_matrix_respects_nested_seed_levels() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    with pytest.raises(RuntimeError, match="frozen P8 selection"):
        selected_methods(config, "confirmatory")
    configured = {method.name: method for method in config.methods}
    methods = [
        configured["b0_legacy_l2_cem"],
        configured["p8_p6_1x"],
        configured["p8_p7_1x"],
    ]
    jobs = evaluation_jobs(
        config,
        methods,
        split_role="confirmatory",
        config_path="vector_jepa_planner_frontier/configs/default.json",
    )
    expected_planner_blocks = 1 + 2 * 2
    assert len(jobs) == expected_planner_blocks * 20 * 2 * 2
    b0 = [job for job in jobs if ":b0_legacy_l2_cem:" in job.label]
    assert len(b0) == 20 * 2 * 2
    assert all(":planner0:" in job.label for job in b0)


def test_smoke_accepts_the_locked_pre_p8_confirmatory_state(tmp_path: Path) -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    paths = config.paths.model_copy(
        update={"p8_selection": tmp_path / "not-yet-frozen.json"}
    )
    pre_p8 = config.model_copy(update={"paths": paths})
    record = confirmatory_gate_record(
        pre_p8,
        config_path="vector_jepa_planner_frontier/configs/default.json",
    )
    assert record == {
        "confirmatory_gate": "correctly_locked_pending_p8_selection",
        "confirmatory_job_count": None,
        "expected_after_p8": "240_if_K2_or_400_if_K4",
    }


def test_path_revisit_metrics_are_bounded_and_use_correct_opportunities() -> None:
    direct = _path_revisit_metrics([0, 1, 2, 3])
    assert direct["repeat_states"] == 0
    assert direct["revisit_rate"] == 0.0
    assert direct["unique_state_ratio"] == 1.0
    assert direct["two_cycle_rate"] == 0.0

    alternating = _path_revisit_metrics([0, 1, 0, 1, 0])
    assert alternating["repeat_states"] == 3
    assert alternating["revisit_rate"] == 0.75
    assert alternating["unique_state_ratio"] == 0.25
    assert alternating["two_cycle_rate"] == 1.0
    assert alternating["short_cycle_periods"] == [2, 4]
    for row in (direct, alternating):
        for name in ("revisit_rate", "unique_state_ratio", "two_cycle_rate"):
            assert 0.0 <= float(row[name]) <= 1.0


def test_nested_task_rows_average_before_backbone_inference() -> None:
    template = {
        "task_id": "a",
        "maze_size": 11,
        "shortest_path_bin": "le16",
        "dead_end_density": 0.2,
        "junction_count": 3,
        "mean_corridor_length": 2.0,
        "spl": 0.5,
        "invalid_actions": 1,
        "loop_or_cycle": False,
        "revisit_rate": 0.1,
        "two_cycle_rate": 0.0,
        "assistance_rate": 0.2,
        "invalid_correction_rate": 0.1,
        "backtrack_correction_rate": 0.1,
        "dead_end_recovery_rate": 0.5,
        "final_bfs_distance": 4,
        "decision_count": 10,
        "auxiliary": {"plan_transitions": 50, "planner_forward_calls": 5},
    }
    left = [{**template, "success": False}]
    right = [{**template, "success": True}]
    averaged = average_nested_task_rows([left, right])
    assert averaged[0]["success"] == pytest.approx(0.5)
    assert averaged[0]["maze_size"] == 11
    assert averaged[0]["decision_count"] == pytest.approx(10.0)
    assert averaged[0]["auxiliary"]["plan_transitions"] == pytest.approx(50.0)


def test_p8_selection_rules_are_compute_matched_and_deterministic() -> None:
    track_f = select_track_f_family(
        [
            {
                "method": FRONTIER_BASES[0],
                "budget_multiplier": 4.0,
                "corrected_macro_sr": 0.64,
                "plan_transitions_per_decision": 2500.0,
            },
            {
                "method": FRONTIER_BASES[1],
                "budget_multiplier": 4.0,
                "corrected_macro_sr": 0.645,
                "plan_transitions_per_decision": 2600.0,
            },
        ]
    )
    assert track_f["method"] == FRONTIER_BASES[0]
    selected = select_near_optimal_budget(
        [
            {
                "method": f"m{budget}",
                "budget_multiplier": budget,
                "corrected_macro_sr": sr,
                "plan_transitions_per_decision": budget * 700.0,
            }
            for budget, sr in zip(
                FRONTIER_BUDGETS, (0.63, 0.641, 0.65, 0.651), strict=True
            )
        ]
    )
    assert selected["budget_multiplier"] == 1.0


def test_p8_metric_aggregation_preserves_seed_level_and_hard_budget() -> None:
    rows = [
        [
            {
                "maze_size": 19,
                "success": success,
                "decision_count": 2.0,
                "auxiliary": {"plan_transitions": transitions},
            },
            {
                "maze_size": 21,
                "success": success,
                "decision_count": 2.0,
                "auxiliary": {"plan_transitions": transitions},
            },
        ]
        for success, transitions in ((1.0, 8.0), (0.0, 4.0))
    ]
    metrics = aggregate_frontier_metrics(rows, transition_limit=4)
    assert metrics["corrected_macro_sr"] == pytest.approx(0.5)
    assert metrics["corrected_size19_21_sr"] == pytest.approx(0.5)
    assert metrics["plan_transitions_per_decision"] == pytest.approx(3.0)
    rows[0][0]["auxiliary"]["plan_transitions"] = 9.0
    with pytest.raises(ValueError, match="hard per-decision budget"):
        aggregate_frontier_metrics(rows, transition_limit=4)


def test_b0_schema_rejects_any_historical_planner_drift() -> None:
    with pytest.raises(ValueError, match="exact historical B0"):
        MethodConfig.model_validate(
            {
                "name": "bad_b0",
                "stage": "P2",
                "track": "F",
                "planner": {"kind": "legacy_cem", "horizon": 11},
            }
        )


def test_legacy_rollout_matches_repository_private_reference() -> None:
    model = FakeModel()
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    candidates = np.asarray([[1, 2, 3], [4, 3, 2]], dtype=np.int64)
    rollout = adapter.rollout(
        context(), candidates, semantics=RolloutSemantics.LEGACY_WARMUP_V1
    )
    reference = _latent_rollout_cost(
        model,
        context().embeddings,
        context().actions,
        context().goal,
        candidates,
        3,
        torch.device("cpu"),
    )
    goal = context().goal.squeeze(1).expand_as(rollout.terminal)
    actual = (rollout.terminal - goal).square().sum(dim=-1).numpy()
    assert np.allclose(actual, reference, atol=0.0, rtol=0.0)


def test_action_aligned_rollout_uses_first_action_but_legacy_warmup_does_not() -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    candidates = np.asarray([[1, 2], [4, 2]], dtype=np.int64)
    legacy = adapter.rollout(
        context(), candidates, semantics=RolloutSemantics.LEGACY_WARMUP_V1
    )
    aligned = adapter.rollout(
        context(), candidates, semantics=RolloutSemantics.ACTION_ALIGNED_V2
    )
    assert torch.equal(legacy.states[0, 0], legacy.states[1, 0])
    assert not torch.equal(aligned.states[0, 0], aligned.states[1, 0])
    assert not torch.equal(aligned.states[0, 1], aligned.states[1, 1])


def test_ranker_training_rollout_uses_every_paired_root() -> None:
    source = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]],
        dtype=torch.float32,
    )
    actions = torch.tensor([[1, 2, 3], [4, 3, 2]], dtype=torch.long)
    terminal = rollout_training_batch(
        FakeModel(),
        source,
        actions,
        history_size=3,
        semantics=RolloutSemantics.ACTION_ALIGNED_V2,
    )
    assert terminal.shape == source.shape
    assert not torch.equal(terminal[0], terminal[1])


def test_exact_backbone_sign_flip_test_and_p2_tie_break() -> None:
    assert exact_sign_flip_pvalue(np.asarray([1.0, 1.0])) == pytest.approx(0.5)
    assert exact_sign_flip_pvalue(np.zeros(4)) == pytest.approx(1.0)
    winner = select_winner(
        [
            {
                "method": "slower",
                "assisted_macro_sr": 0.70,
                "assisted_size19_21_sr": 0.60,
                "predictor_serial_calls_per_decision": 20.0,
            },
            {
                "method": "faster",
                "assisted_macro_sr": 0.695,
                "assisted_size19_21_sr": 0.605,
                "predictor_serial_calls_per_decision": 10.0,
            },
        ]
    )
    assert winner["method"] == "faster"


def test_legacy_planner_matches_repository_cem_exactly() -> None:
    model = FakeModel()
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(kind=PlannerKind.LEGACY_CEM)
    scorer = CompositeScorer(ScorerConfig())
    result = LegacyCEMPlanner(adapter, config, scorer).plan(context(), seed=123)
    expected_sequence, expected_cost, _ = cem_plan(
        model,
        context().embeddings,
        context().actions,
        context().goal,
        horizon=12,
        history_size=3,
        num_candidates=64,
        num_elites=8,
        cem_iters=1,
        momentum=0.1,
        num_actions=5,
        device=torch.device("cpu"),
        seed=123,
        score_fn=lambda terminal, goal: (terminal - goal).square().sum(dim=-1),
        allowed_actions=np.asarray(ACTION_IDS, dtype=np.int64),
    )
    assert np.array_equal(result.sequence, expected_sequence)
    assert result.cost == expected_cost
    assert result.ledger.plan_transitions == 768
    assert len(result.candidate_batches) == 1
    assert result.candidate_batches[0].sequences.shape == (64, 12)
    assert result.candidate_batches[0].predicted_costs is None


def test_instrumented_categorical_cem_is_numerically_equivalent() -> None:
    model = FakeModel()
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(kind=PlannerKind.CATEGORICAL_CEM)
    result = build_planner(
        adapter,
        config,
        CompositeScorer(ScorerConfig()),
    ).plan(context(), seed=123)
    expected_sequence, expected_cost, _ = cem_plan(
        model,
        context().embeddings,
        context().actions,
        context().goal,
        horizon=12,
        history_size=3,
        num_candidates=64,
        num_elites=8,
        cem_iters=1,
        momentum=0.1,
        num_actions=5,
        device=torch.device("cpu"),
        seed=123,
        score_fn=lambda terminal, goal: (terminal - goal).square().sum(dim=-1),
        allowed_actions=np.asarray(ACTION_IDS, dtype=np.int64),
    )
    assert np.array_equal(result.sequence, expected_sequence)
    assert result.cost == expected_cost
    assert result.candidate_batches[0].predicted_costs is not None


def test_candidate_trace_sampling_is_deterministic_and_method_independent() -> None:
    observed = [candidate_trace_selected("task-abc", step, 0.1) for step in range(100)]
    repeated = [candidate_trace_selected("task-abc", step, 0.1) for step in range(100)]
    assert observed == repeated
    assert 1 <= sum(observed) <= 25


def test_formal_candidate_sampling_is_exact_within_maze_size() -> None:
    rows = [
        {
            "task_id": f"task-{size}-{task}",
            "maze_size": size,
            "decision_traces": [{"step": step} for step in range(10)],
        }
        for size in (9, 11)
        for task in range(3)
    ]
    selected, record = exact_candidate_trace_keys(rows, 0.1)
    assert len(selected) == 6
    assert record["strata"]["9"]["selected_count"] == 3
    assert record["strata"]["11"]["selected_count"] == 3


def test_candidate_truth_analysis_is_posthoc_and_hashable(tmp_path: Path) -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    state = int(entry["start_cell"])
    goal = int(entry["goal_cell"])
    from final_closure.common import bfs_distances_from, next_state

    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    optimal = next(
        action
        for action in ACTION_IDS
        if int(distances[next_state(env, state, action)]) == int(distances[state]) - 1
    )
    alternatives = [action for action in ACTION_IDS if action != optimal]
    candidates = np.asarray(
        [[optimal, optimal], [alternatives[0], alternatives[0]]], dtype=np.int64
    )
    result = PlannerResult(
        sequence=np.asarray([optimal, optimal], dtype=np.int64),
        cost=0.1,
        ledger=ComputeLedger(),
        diagnostics={},
        candidate_batches=(
            CandidateBatch(candidates, np.asarray([0.1, 1.0], dtype=np.float64)),
        ),
    )
    record = analyze_candidate_result(
        env,
        task_identifier="fixture",
        task_index=0,
        step=0,
        seed=7,
        root_state=state,
        result=result,
        candidate_k=64,
        prefix_lengths=(2, 4, 8),
    )
    assert record["analysis_only_no_action_influence"] is True
    assert record["metrics"]["first_action_coverage_at_k"] is True
    assert record["stored_candidate_count"] == 2
    output = tmp_path / "candidate.jsonl"
    sink = CandidateTraceSink(output)
    sink.write(record)
    artifact = sink.commit()
    assert artifact["decision_record_count"] == 1
    assert output.read_text(encoding="utf-8").count("\n") == 1


def test_candidate_edit_distance_accepts_mixed_search_prefix_lengths() -> None:
    assert _normalized_edit_distance((1,), (1, 2, 3)) == pytest.approx(2 / 3)
    assert _normalized_edit_distance((1, 2), (1, 3)) == pytest.approx(0.5)
    assert _normalized_edit_distance((), ()) == 0.0


def test_no_progress_is_false_optimism_only_when_a_better_candidate_exists() -> None:
    from final_closure.common import bfs_distances_from

    fixture = None
    for entry in read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl"):
        env = validate_manifest_entry(entry)
        state = int(entry["start_cell"])
        goal = int(entry["goal_cell"])
        distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
        improving = [
            action
            for action in ACTION_IDS
            if int(distances[next_state(env, state, action)]) < int(distances[state])
        ]
        nonprogress = [
            action
            for action in ACTION_IDS
            if next_state(env, state, action) != state
            and int(distances[next_state(env, state, action)]) >= int(distances[state])
        ]
        if improving and nonprogress:
            fixture = (env, state, nonprogress[0], improving[0])
            break
    assert fixture is not None, "fixture requires a junction with a worse valid move"
    env, state, selected_action, improving_action = fixture

    def analyze(actions: list[int]) -> dict[str, object]:
        candidates = np.asarray([[action] for action in actions], dtype=np.int64)
        result = PlannerResult(
            sequence=np.asarray([selected_action], dtype=np.int64),
            cost=0.1,
            ledger=ComputeLedger(),
            diagnostics={},
            candidate_batches=(
                CandidateBatch(
                    candidates,
                    np.arange(len(actions), dtype=np.float64),
                ),
            ),
        )
        return analyze_candidate_result(
            env,
            task_identifier="false-optimism-fixture",
            task_index=0,
            step=0,
            seed=7,
            root_state=state,
            result=result,
            candidate_k=64,
            prefix_lengths=(2, 4, 8),
        )["metrics"]

    no_alternative = analyze([selected_action])
    assert no_alternative["selected_no_progress"] is True
    assert no_alternative["selected_invalid"] is False
    assert no_alternative["selected_short_cycle"] is False
    assert no_alternative["false_optimistic"] is False

    with_better_alternative = analyze([selected_action, improving_action])
    assert with_better_alternative["false_optimistic"] is True

    goal = int(env._goal_position)
    mined_without_alternative = execute_candidate(
        env,
        state,
        goal,
        [selected_action],
        progress_candidate_available=False,
    )
    mined_with_alternative = execute_candidate(
        env,
        state,
        goal,
        [selected_action],
        progress_candidate_available=True,
    )
    assert mined_without_alternative["false_optimistic"] is False
    assert mined_with_alternative["false_optimistic"] is True


def test_new_and_old_b0_controllers_choose_the_same_action() -> None:
    source_config = json.loads(
        (ROOT / "final_closure/configs/default.json").read_text(encoding="utf-8")
    )
    planner_config = source_config["baselines"][1]["planner"]
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    observation, _ = env.reset()
    model = FakeModel()
    old = LeWMCEMController(
        model,
        planner_config,
        device=torch.device("cpu"),
        evaluation_seed=42,
        action_selection="unmasked",
    )
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    planner = LegacyCEMPlanner(
        adapter,
        PlannerConfig(kind=PlannerKind.LEGACY_CEM),
        CompositeScorer(ScorerConfig()),
    )
    new = FrontierController(
        adapter,
        planner,
        evaluation_seed=42,
        action_selection="unmasked",
    )
    old.reset(env, observation, 0)
    new.reset(env, observation, 0)
    state = int(env._state)
    old_action, _ = old.choose(env, observation, state, None)
    new_action, _ = new.choose(env, observation, state, None)
    assert new_action == old_action


def test_new_and_old_corrected_v1_fallback_are_identical() -> None:
    source_config = json.loads(
        (ROOT / "final_closure/configs/default.json").read_text(encoding="utf-8")
    )
    planner_config = source_config["baselines"][1]["planner"]
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    free = np.flatnonzero((~env._maze_mask).reshape(-1))
    state = next(
        int(value)
        for value in free
        if any(
            env._next_state(int(value), env._decode_action(action)) == int(value)
            for action in ACTION_IDS
        )
    )
    observation = observe_state(env, state)
    model = FakeModel()
    old = LeWMCEMController(
        model,
        planner_config,
        device=torch.device("cpu"),
        evaluation_seed=42,
        action_selection="corrected",
    )
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    new = FrontierController(
        adapter,
        LegacyCEMPlanner(
            adapter,
            PlannerConfig(kind=PlannerKind.LEGACY_CEM),
            CompositeScorer(ScorerConfig()),
        ),
        evaluation_seed=42,
        action_selection="corrected_v1",
    )
    old.reset(env, observation, 0)
    new.reset(env, observation, 0)
    ledger = ComputeLedger()
    assert old._corrected_one_step(env, state, None) == new._corrected_one_step(
        env, state, None, ledger
    )
    assert ledger.assist_transitions == 5


def test_compute_ledger_keeps_assistance_separate() -> None:
    ledger = ComputeLedger()
    ledger.record_plan(transitions=768, batch_size=64, calls=12)
    ledger.record_assist(transitions=5)
    assert ledger.total_transitions == 773
    validate_compute_ledger(ledger.to_dict())
    changed = ledger.to_dict()
    changed["total_transitions"] += 1
    with pytest.raises(ValueError, match="total compute"):
        validate_compute_ledger(changed)


def test_reachability_cdf_is_monotone_and_budget_lookup_is_ceil_binned() -> None:
    head = DistributionalReachability(HeadConfig(latent_dim=4, hidden_dim=16))
    source = torch.randn(7, 4)
    goal = torch.randn(7, 4)
    probabilities = head(source, goal)
    assert bool((probabilities[:, 1:] >= probabilities[:, :-1]).all())
    assert torch.equal(
        head.probability_for_budget(source, goal, 3), probabilities[:, 2]
    )


def test_calibration_metrics_handle_ties_and_probability_one() -> None:
    probability = np.asarray([0.0, 0.25, 0.25, 1.0])
    target = np.asarray([0, 0, 1, 1])
    auc = binary_auroc(probability, target)
    ece, reliability = binary_reliability(probability, target, bin_count=4)
    assert auc == pytest.approx(0.875)
    assert 0.0 <= ece <= 1.0
    assert sum(int(row["count"]) for row in reliability) == 4
    assert reliability[-1]["count"] == 1


def test_join_threshold_maximizes_recall_subject_to_precision_gate() -> None:
    selected = select_precision_threshold(
        np.asarray([0.95, 0.9, 0.8, 0.7]),
        np.asarray([1, 1, 0, 1]),
        required_precision=0.95,
    )
    assert selected["join_threshold"] == pytest.approx(0.9)
    assert selected["join_precision"] == pytest.approx(1.0)
    assert selected["join_recall"] == pytest.approx(2 / 3)
    assert selected["join_precision_gate_passed"] is True


def test_power_analysis_enforces_twenty_backbone_floor() -> None:
    result = required_backbone_count(
        overall_std=0.0,
        large_size_std=0.0,
        comparison_count=2,
    )
    assert result["n_overall"] == 0
    assert result["n_ood"] == 0
    assert result["required_backbones"] == 20


@pytest.mark.parametrize(
    "kind",
    [PlannerKind.ICEM, PlannerKind.BEAM, PlannerKind.BEST_FIRST, PlannerKind.MCTS],
)
def test_search_planners_return_valid_action_and_respect_budget(
    kind: PlannerKind,
) -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(
        kind=kind,
        horizon=3,
        history_size=3,
        num_candidates=8,
        num_elites=2,
        cem_iters=1,
        beam_width=4,
        budget=BudgetConfig(multiplier=1.0, reference_transitions=48),
    )
    planner = build_planner(adapter, config, CompositeScorer(ScorerConfig()))
    result = planner.plan(context(), seed=9)
    assert result.sequence.shape == (3,)
    assert set(result.sequence.tolist()) <= set(ACTION_IDS)
    assert result.ledger.plan_transitions <= config.budget.transition_limit
    if kind == PlannerKind.MCTS:
        assert result.diagnostics["root_visits"] == result.diagnostics["simulations"]
        assert (
            result.diagnostics["simulations"] <= result.diagnostics["max_simulations"]
        )
        assert result.diagnostics["termination_reason"] in {
            "transition_limit",
            "simulation_limit",
        }
        assert result.diagnostics["selected_root_action"] == int(result.sequence[0])
        assert result.diagnostics["selected_evaluated_prefix"][0] == int(
            result.sequence[0]
        )


def test_nested_bootstrap_resamples_planner_seeds_inside_backbones() -> None:
    candidate = np.asarray(
        [
            [[0.0] * 4, [1.0] * 4],
            [[0.0] * 4, [1.0] * 4],
        ],
        dtype=np.float64,
    )
    baseline = np.full((2, 1, 4), 0.5, dtype=np.float64)
    effect = nested_paired_bootstrap(
        candidate,
        baseline,
        samples=2000,
        alpha=0.05,
        seed=17,
        strata=("9", "9", "11", "11"),
        pair_planner_seeds=False,
    )
    assert effect["delta"] == pytest.approx(0.0)
    assert effect["ci_low"] < 0.0 < effect["ci_high"]
    assert effect["planner_seed_resampling"] == "independent_within_backbone"


def test_summary_csv_has_exactly_one_header_row() -> None:
    rendered = csv_text([{"method": "a", "sr": 0.5}])
    assert rendered.splitlines() == ["method,sr", "a,0.5"]


def test_proposal_only_control_never_calls_predictor() -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(
        kind=PlannerKind.BEST_FIRST,
        horizon=3,
        num_candidates=8,
        num_elites=2,
        budget=BudgetConfig(multiplier=1.0, reference_transitions=48),
    )
    planner = build_planner(
        adapter,
        config,
        CompositeScorer(ScorerConfig()),
        proposal=UniformProposal(),
        proposal_only=True,
    )
    result = planner.plan(context(), seed=13)
    assert result.ledger.plan_transitions == 0
    assert result.ledger.candidate_sequences == 8
    assert result.diagnostics["proposal_only"] is True


def test_best_first_proposal_candidates_are_predictor_scored_and_budgeted() -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(
        kind=PlannerKind.BEST_FIRST,
        horizon=3,
        num_candidates=8,
        num_elites=2,
        budget=BudgetConfig(multiplier=1.0, reference_transitions=48),
    )
    planner = build_planner(
        adapter,
        config,
        CompositeScorer(ScorerConfig()),
        proposal=UniformProposal(),
    )
    result = planner.plan(context(), seed=13)
    assert result.diagnostics["proposal_seed_candidates"] == 4
    assert result.diagnostics["proposal_seed_transitions"] == 12
    assert result.ledger.plan_transitions >= 12
    assert result.ledger.plan_transitions <= config.budget.transition_limit


@pytest.mark.parametrize("mode", ["direct", "random", "fixed_breadth", "learned"])
def test_vector_dts_controls_are_executable(mode: str) -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    config = PlannerConfig(
        kind=PlannerKind.VECTOR_DTS,
        horizon=3,
        num_candidates=8,
        num_elites=2,
        beam_width=4,
        budget=BudgetConfig(multiplier=1.0, reference_transitions=48),
    )
    head = VectorDTSHead(HeadConfig(latent_dim=4, hidden_dim=16, horizon=3))
    planner = build_planner(
        adapter,
        config,
        CompositeScorer(ScorerConfig()),
        dts_head=head,
        dts_expansion=mode,
    )
    result = planner.plan(context(), seed=4)
    assert set(result.sequence.tolist()) <= set(ACTION_IDS)
    assert result.ledger.plan_transitions <= config.budget.transition_limit
    if mode in {"random", "learned"}:
        assert result.diagnostics["root_visits"] == result.diagnostics["simulations"]
        assert (
            result.diagnostics["simulations"] <= result.diagnostics["max_simulations"]
        )
        assert result.diagnostics["selected_root_action"] == int(result.sequence[0])


def test_mixture_proposal_preserves_uniform_floor_and_exact_count() -> None:
    bank = RetrievalBank(
        source_latents=torch.zeros(4, 4),
        goal_latents=torch.zeros(4, 4),
        action_chunks=torch.tensor([[1, 2, 3]] * 4),
        task_hashes=("a", "b", "c", "d"),
    )
    config = ProposalConfig(
        kind=ProposalKind.MIXTURE,
        uniform_weight=0.5,
        retrieval_weight=0.5,
        learned_weight=0.0,
    )
    proposal = MixtureProposal(config, retrieval=RetrievalProposal(bank, top_k=4))
    values = proposal.sample(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        count=10,
        horizon=3,
        rng=np.random.default_rng(4),
    )
    assert values.shape == (10, 3)
    assert np.isin(values, ACTION_IDS).all()


def test_transposition_hard_pruning_requires_precision_gate() -> None:
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    join = StateJoinHead(HeadConfig(latent_dim=4, hidden_dim=16))
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    method = next(
        item for item in config.methods if item.name == "p5_track_f_all_hard_memory"
    )
    assert method.memory.hard_pruning
    with pytest.raises(ValueError, match="precision gate"):
        build_planner(
            adapter,
            method.planner,
            CompositeScorer(
                method.scorer,
                verifier=ActionConsistencyVerifier(
                    HeadConfig(latent_dim=4, hidden_dim=16)
                ),
                reachability=DistributionalReachability(
                    HeadConfig(latent_dim=4, hidden_dim=16)
                ),
            ),
            memory_config=method.memory,
            join_head=join,
            join_precision=0.94,
        )


def test_mining_folds_are_deterministic_disjoint_and_complete() -> None:
    hashes = [f"task-{index}" for index in range(1000)]
    assignments = {task_hash: mining_fold(task_hash) for task_hash in hashes}
    assert set(assignments.values()) == {1, 2, 3}
    assert assignments == {task_hash: mining_fold(task_hash) for task_hash in hashes}
    assert sum(list(assignments.values()).count(index) for index in (1, 2, 3)) == 1000


def test_locked_new_manifests_are_pairwise_disjoint() -> None:
    config = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    overlaps = validate_manifest_isolation(config)
    assert overlaps
    assert all(value == 0 for result in overlaps.values() for value in result.values())
    lock = load_json(config.paths.protocol_lock)
    assert lock["validation_manifest"]["count"] == 700
    assert lock["confirmatory_manifest"]["count"] == 900


def test_action_chunk_sampler_fails_closed_if_no_topology_supports_horizon() -> None:
    entries = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")
    known_short = entries[247]
    sampler = PlannerBatchSampler([known_short], horizon=12)
    assert sampler._materialize(0).eligible_states.size == 0
    with pytest.raises(ValueError, match="no topology supporting horizon"):
        sampler.sample(np.random.default_rng(1), batch_size=1)


def test_action_chunk_sampler_replaces_ineligible_topology_without_retry_risk() -> None:
    entries = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")
    known_short = entries[247]
    same_size = [
        entry
        for entry in entries
        if int(entry["maze_size"]) == int(known_short["maze_size"])
    ]
    known_eligible = next(
        entry
        for entry in same_size
        if PlannerBatchSampler([entry], horizon=12)._materialize(0).eligible_states.size
        > 0
    )
    sampler = PlannerBatchSampler([known_short, known_eligible], horizon=12)
    replacement = sampler._eligible_topology(
        0,
        size=int(known_short["maze_size"]),
        rng=np.random.default_rng(7),
    )
    assert replacement.eligible_states.size > 0
    assert replacement.entry["task_hash"] == known_eligible["task_hash"]
    assert sampler._eligible_indices_by_size[int(known_short["maze_size"])] == (1,)


def test_general_state_sampler_keeps_short_topology_without_fake_chunk_labels() -> None:
    entries = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")
    known_short = entries[247]
    sampler = PlannerBatchSampler([known_short], horizon=12, require_full_chunk=False)
    batch = sampler.sample(np.random.default_rng(11), batch_size=8)
    assert batch.batch_size == 8
    assert batch.optimal_chunks_are_full is False
    proposal = AutoregressiveProposal(
        HeadConfig(latent_dim=4, hidden_dim=16, horizon=12)
    )
    latents = encode_planner_batch(FakeModel(), batch, gradients=False)
    method = MethodConfig.model_validate(
        {
            "name": "chunk_guard",
            "stage": "P3",
            "track": "F",
            "planner": {"kind": "best_first", "budget": {"multiplier": 4.0}},
            "proposal": {
                "kind": "mixture",
                "uniform_weight": 0.5,
                "retrieval_weight": 0.0,
                "learned_weight": 0.5,
            },
            "component_checkpoint_required": True,
        }
    )
    with pytest.raises(ValueError, match="require full optimal chunks"):
        component_losses(
            {"autoregressive_proposal": proposal},
            latents,
            batch,
            world_model=VectorWorldModel(
                FakeModel(), device=torch.device("cpu"), history_size=3
            ),
            method=method,
            stochastic_rngs=component_stochastic_rngs(42, 104729),
        )


def test_join_labels_compare_imagined_successors_to_true_or_distinct_states() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    sampler = PlannerBatchSampler([entry], horizon=2, require_full_chunk=False)
    batch = sampler.sample(np.random.default_rng(19), batch_size=128)
    positive = batch.join_labels == 1
    negative = ~positive
    assert bool(positive.any()) and bool(negative.any())
    assert torch.equal(
        batch.comparison_observations[positive],
        batch.successor_observations[positive],
    )
    assert bool(
        (
            batch.comparison_observations[negative]
            != batch.successor_observations[negative]
        )
        .reshape(int(negative.sum()), -1)
        .any(dim=1)
        .all()
    )


def test_joint_counterexample_sampler_is_split_safe_and_type_strict() -> None:
    entry = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")[0]
    env = validate_manifest_entry(entry)
    source = int(np.flatnonzero((~env._maze_mask).reshape(-1))[0])
    record = {
        "task_hash": entry["task_hash"],
        "topology_seed": entry["topology_seed"],
        "maze_size": entry["maze_size"],
        "source_state": source,
        "goal_state": entry["goal_cell"],
        "good_actions": [1, 2],
        "false_optimistic_actions": [3, 4],
        "negative_source": "planner_false_optimistic",
        "outcome": {"false_optimistic": True},
    }
    sampler = CounterexampleBatchSampler(
        [record],
        [entry],
        horizon=2,
        expected_negative_source="planner_false_optimistic",
    )
    batch = sampler.sample(
        np.random.default_rng(23), batch_size=4, device=torch.device("cpu")
    )
    assert batch.source_observations.shape[0] == 4
    assert batch.good_actions.shape == batch.bad_actions.shape == (4, 2)
    assert set(batch.task_hashes) == {entry["task_hash"]}

    wrong_type = dict(record, negative_source="matched_round_random_actions")
    with pytest.raises(ValueError, match="negative type mismatch"):
        CounterexampleBatchSampler(
            [wrong_type],
            [entry],
            horizon=2,
            expected_negative_source="planner_false_optimistic",
        )
    with pytest.raises(ValueError, match="duplicate counterexample task"):
        CounterexampleBatchSampler(
            [record, record],
            [entry],
            horizon=2,
            expected_negative_source="planner_false_optimistic",
        )


def test_joint_jepa_sampler_emits_locked_eight_frame_balanced_trajectories() -> None:
    entries = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")
    sampler = JEPATrajectorySampler(entries, sequence_length=8)
    rng = np.random.default_rng(29)
    observed_sizes = []
    for slot in range(len(sampler.sizes)):
        batch = sampler.sample(
            rng,
            batch_size=2,
            size_slot=slot,
            device=torch.device("cpu"),
        )
        observed_sizes.append(batch.maze_size)
        assert batch.observations.shape[:2] == (2, 8)
        assert batch.actions.shape == (2, 7)
        assert batch.states.shape == (2, 8)
    assert observed_sizes == list(sampler.sizes)


def test_every_configured_method_constructs_with_its_declared_dependencies() -> None:
    study = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    adapter = VectorWorldModel(FakeModel(), device=torch.device("cpu"), history_size=3)
    for method in study.methods:
        head_config = HeadConfig(latent_dim=4, hidden_dim=16, horizon=12)
        modules = required_heads(method, head_config)
        if method.component_checkpoint_required:
            assert modules, method.name
        scorer = CompositeScorer(
            method.scorer,
            verifier=modules.get("verifier"),
            reachability=modules.get("reachability"),
            ranker=modules.get("ranker"),
            shuffle_candidate_association=(
                method.control.predictor_association == "candidate_shuffle"
            ),
        )
        planner = build_planner(
            adapter,
            method.planner,
            scorer,
            proposal=UniformProposal(),
            memory_config=method.memory,
            join_head=modules.get("join"),
            join_precision=1.0,
            dts_head=modules.get("dts"),
            proposal_only=method.control.proposal_execution == "proposal_only",
            dts_expansion=method.control.dts_expansion,
        )
        assert planner is not None


def test_head_initialization_is_component_stable_across_factorial_cells() -> None:
    study = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    methods = {method.name: method for method in study.methods}
    verifier_only = required_heads(
        methods["p3_factorial_v1r0p0m0"],
        HeadConfig(latent_dim=4, hidden_dim=16, horizon=3),
        backbone_seed=42,
        planner_seed=104729,
    )["verifier"]
    verifier_plus = required_heads(
        methods["p3_factorial_v1r1p1m1"],
        HeadConfig(latent_dim=4, hidden_dim=16, horizon=3),
        backbone_seed=42,
        planner_seed=104729,
    )["verifier"]
    for left, right in zip(
        verifier_only.state_dict().values(),
        verifier_plus.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)


def test_component_training_random_streams_are_factorially_isolated() -> None:
    reference = component_stochastic_rngs(42, 104729)
    perturbed = component_stochastic_rngs(42, 104729)
    perturbed["ranker"].integers(1, 5, size=10_000)
    assert np.array_equal(
        reference["denoising_proposal"].random(256),
        perturbed["denoising_proposal"].random(256),
    )

    repeated = component_stochastic_rngs(42, 104729)
    assert np.array_equal(
        component_stochastic_rngs(42, 104729)["ranker"].integers(1, 5, size=256),
        repeated["ranker"].integers(1, 5, size=256),
    )


def test_every_track_f_component_loss_is_finite_on_a_real_maze_batch() -> None:
    study = load_study_config(
        ROOT / "vector_jepa_planner_frontier/configs/default.json"
    )
    entries = read_jsonl(ROOT / "data/splits/unisize_train_manifest.jsonl")
    sampler = PlannerBatchSampler([entries[0]], horizon=12)
    batch = sampler.sample(np.random.default_rng(2), batch_size=4)
    model = FakeModel()
    adapter = VectorWorldModel(model, device=torch.device("cpu"), history_size=3)
    latents = encode_planner_batch(model, batch, gradients=False)
    for method in study.methods:
        if method.track != "F" or not method.component_checkpoint_required:
            continue
        modules = required_heads(
            method, HeadConfig(latent_dim=4, hidden_dim=16, horizon=12)
        )
        losses = component_losses(
            modules,
            latents,
            batch,
            world_model=adapter,
            method=method,
            stochastic_rngs=component_stochastic_rngs(42, 104729),
        )
        if method.control.verifier_targets == "random_untrained" and set(modules) == {
            "verifier"
        }:
            assert not losses
        else:
            assert losses, method.name
            assert all(torch.isfinite(loss) for loss in losses.values()), method.name
