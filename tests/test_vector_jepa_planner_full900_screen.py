from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from final_closure.common import read_jsonl, sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_frontier.schemas import PlannerKind
from vector_jepa_planner_full900_screen.analysis import (
    delta_sr,
    stratified_paired_bootstrap,
)
from vector_jepa_planner_full900_screen.audit_protocol import _field_differences
from vector_jepa_planner_full900_screen.common import (
    component_checkpoint_path,
    load_config,
    load_json,
    result_path,
    validate_lock,
)
from vector_jepa_planner_full900_screen.evaluate import _validate_budget
from vector_jepa_planner_full900_screen.methods import (
    direct_control_name,
    effective_method,
    validate_q1_selection,
)
from vector_jepa_planner_full900_screen.parity import compare
from vector_jepa_planner_full900_screen.run_plan import (
    _q0_jobs,
    schedule_text,
    stage_jobs,
)
from vector_jepa_planner_full900_screen.schemas import QuickStudyConfig
from vector_jepa_planner_full900_screen.summarize import _nested_bootstrap

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "vector_jepa_planner_full900_screen/configs/default.json"


def config() -> QuickStudyConfig:
    return load_config(CONFIG_PATH)


def with_temp_decisions(base: QuickStudyConfig, tmp_path: Path) -> QuickStudyConfig:
    paths = base.paths.model_copy(
        update={
            "run_root": tmp_path / "runs",
            "result_template": str(
                tmp_path
                / "runs/{method}/backbone{backbone_seed}/planner{planner_seed}/"
                "{split}_{action_selection}.json"
            ),
            "schedule_dir": tmp_path / "runs/schedules",
            "p2_selection": tmp_path / "runs/decisions/q1_parent.json",
            "p5_advancement": tmp_path / "runs/decisions/shortlist.json",
            "p7_selection": tmp_path / "runs/decisions/final.json",
            "p8_selection": tmp_path / "runs/decisions/closure.json",
        }
    )
    return base.model_copy(update={"paths": paths})


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_q0_parity(study: QuickStudyConfig) -> None:
    for action in study.replication.action_selections:
        reference_action = "corrected" if action == "corrected_v1" else "unmasked"
        reference_path = (
            Path(study.paths.run_root)
            / "parity"
            / f"reference_seed42_{reference_action}.json"
        )
        candidate_path = result_path(
            study,
            method="b0_legacy_l2_cem",
            backbone_seed=42,
            planner_seed=0,
            action_selection=action,
        )
        write_json(reference_path, {"action": reference_action})
        write_json(
            candidate_path,
            {"method": "b0_legacy_l2_cem", "action": action},
        )
        write_json(
            Path(study.paths.run_root) / "parity" / f"parity_{action}.json",
            {
                "status": "pass",
                "task_count": 900,
                "protocol_id": study.protocol_id,
                "quick_spec_sha256": "quick",
                "action_selection": action,
                "executed_actions_compared": True,
                "reference_sha256": sha256_file(reference_path),
                "candidate_sha256": sha256_file(candidate_path),
            },
        )


def write_q1_selection(
    study: QuickStudyConfig,
    *,
    quick_spec: str = "quick",
    selected: str = "q1_mcts_1x",
) -> None:
    write_q0_parity(study)
    parity_sha256s = {
        action: sha256_file(
            Path(study.paths.run_root) / "parity" / f"parity_{action}.json"
        )
        for action in study.replication.action_selections
    }
    names = (
        "b0_legacy_l2_cem",
        "q1_control_categorical_cem_1x",
        "q1_icem_1x",
        "q1_beam_1x",
        "q1_best_first_1x",
        "q1_mcts_1x",
    )
    inputs = {}
    for name in names:
        for action in study.replication.action_selections:
            path = result_path(
                study,
                method=name,
                backbone_seed=42,
                planner_seed=0,
                action_selection=action,
            )
            write_json(path, {"method": name, "action": action})
            inputs[f"{name}:{action}"] = sha256_file(path)
    write_json(
        Path(study.paths.p2_selection),
        {
            "schema": "vector-jepa-full900-q1-parent-v1",
            "protocol_id": study.protocol_id,
            "quick_spec_sha256": quick_spec,
            "q0_parity_sha256s": parity_sha256s,
            "selected_parent": selected,
            "categorical_bridge_exact_task_parity": True,
            "ranked_candidates": [
                {"method": name}
                for name in (
                    "q1_control_categorical_cem_1x",
                    "q1_icem_1x",
                    "q1_beam_1x",
                    "q1_best_first_1x",
                    "q1_mcts_1x",
                )
            ],
            "input_sha256s": inputs,
        },
    )


def write_shortlist(study: QuickStudyConfig, names: list[str] | None = None) -> None:
    names = names or ["q2b_denoising_icem"]
    method_name = names[0]
    method = method_by_name(study, method_name)
    planner_seed = 104_729 if method.component_checkpoint_required else 0
    path = result_path(
        study,
        method=method_name,
        backbone_seed=42,
        planner_seed=planner_seed,
        action_selection="corrected_v1",
    )
    write_json(path, {"method": method_name})
    eligible = [role.name for role in study.method_roles if role.advancement_eligible]
    write_json(
        Path(study.paths.p5_advancement),
        {
            "schema": "vector-jepa-full900-shortlist-v1",
            "protocol_id": study.protocol_id,
            "quick_spec_sha256": "quick",
            "q1_parent_sha256": sha256_file(study.paths.p2_selection),
            "shortlist": names,
            "candidate_audits": [{"method": name} for name in eligible],
            "input_sha256s": {
                f"{method_name}:b42:p{planner_seed}:corrected_v1": sha256_file(path)
            },
        },
    )


def write_final_selection(study: QuickStudyConfig, winner: str) -> None:
    method = method_by_name(study, winner)
    planner_seed = 104_729 if method.component_checkpoint_required else 0
    path = result_path(
        study,
        method=winner,
        backbone_seed=42,
        planner_seed=planner_seed,
        action_selection="unmasked",
    )
    write_json(path, {"method": winner})
    write_json(
        Path(study.paths.p7_selection),
        {
            "schema": "vector-jepa-full900-final-winner-v1",
            "protocol_id": study.protocol_id,
            "quick_spec_sha256": "quick",
            "shortlist_sha256": sha256_file(study.paths.p5_advancement),
            "winner": winner,
            "candidate_audits": [{"method": winner}],
            "input_sha256s": {
                f"{winner}:b42:p{planner_seed}:unmasked": sha256_file(path)
            },
        },
    )


def fake_tasks(*, positive_count: int = 0) -> list[dict]:
    rows = []
    index = 0
    for size in range(9, 26, 2):
        for _ in range(100):
            rows.append(
                {
                    "task_id": f"task-{index}",
                    "maze_size": size,
                    "success": index < positive_count,
                    "spl": float(index < positive_count),
                }
            )
            index += 1
    return rows


def fake_parity_rows() -> list[dict]:
    return [
        {
            "task_id": f"task-{index}",
            "maze_size": 9 + 2 * (index // 100),
            "topology_seed": index,
            "start_cell": 1,
            "goal_cell": 2,
            "optimal_length": 1,
            "success": True,
            "path_length": 1,
            "spl": 1.0,
            "invalid_actions": 0,
            "repeat_states": 0,
            "max_state_visits": 1,
            "loop_or_cycle": False,
            "final_bfs_distance": 0,
            "executed_actions": [1],
            "decision_traces": [{"executed_action": 1}],
        }
        for index in range(900)
    ]


def test_config_locks_twelve_candidates_and_eighteen_executable_methods() -> None:
    study = config()
    assert len(study.methods) == 18
    assert sum(role.advancement_eligible for role in study.method_roles) == 12
    assert study.replication.task_count == 900
    assert study.replication.final_backbone_seeds == tuple(range(42, 52))
    assert study.replication.final_planner_seeds == (104_729, 130_363)
    assert study.gates.bonferroni_comparison_count == 48


def test_all_methods_are_frozen_backbone_legacy_rollout_and_one_x() -> None:
    study = config()
    assert all(method.track == "F" for method in study.methods)
    assert all(
        method.planner.budget.transition_limit == 768 for method in study.methods
    )
    assert all(
        method.planner.rollout_semantics.value == "legacy_warmup_v1"
        for method in study.methods
    )


def test_q1_contains_exact_four_search_candidates_and_one_bridge() -> None:
    study = config()
    roles = {role.name: role for role in study.method_roles}
    candidates = {
        method.planner.kind
        for method in study.methods
        if roles[method.name].phase == "Q1" and roles[method.name].role == "candidate"
    }
    assert candidates == {
        PlannerKind.ICEM,
        PlannerKind.BEAM,
        PlannerKind.BEST_FIRST,
        PlannerKind.MCTS,
    }
    bridge = method_by_name(study, "q1_control_categorical_cem_1x")
    assert bridge.planner.kind == PlannerKind.CATEGORICAL_CEM


def test_denoising_head_discovery_does_not_construct_a_dummy_transformer() -> None:
    study = config()
    method = method_by_name(study, "q2b_denoising_icem")
    assert required_head_names(method) == {"denoising_proposal"}


def test_q1_candidates_change_only_name_and_planner() -> None:
    study = config()
    bridge = method_by_name(study, "q1_control_categorical_cem_1x")
    for name in ("q1_icem_1x", "q1_beam_1x", "q1_best_first_1x", "q1_mcts_1x"):
        assert _field_differences(bridge, method_by_name(study, name)) == {
            "name",
            "planner",
        }


def test_radical_and_ranker_controls_change_only_declared_fields() -> None:
    study = config()
    expected = {
        ("q2b_vector_dts", "q2b_control_dts_direct"): {"name", "control"},
        (
            "q2b_bidirectional",
            "q2b_control_bidirectional_forward",
        ): {"name", "planner"},
        (
            "q2b_denoising_icem",
            "q2b_control_denoising_uniform",
        ): {"name", "proposal", "component_checkpoint_required"},
        (
            "q2c_hard_negative_ranker",
            "q2c_control_random_negative_ranker",
        ): {"name", "control"},
    }
    for (candidate, control), fields in expected.items():
        assert (
            _field_differences(
                method_by_name(study, candidate), method_by_name(study, control)
            )
            == fields
        )


def test_manifest_is_full_900_with_one_hundred_tasks_per_size() -> None:
    rows = read_jsonl(ROOT / "data/splits/unisize_eval_manifest.jsonl")
    assert len(rows) == 900
    assert Counter(int(row["maze_size"]) for row in rows) == {
        size: 100 for size in range(9, 26, 2)
    }
    assert len({str(row["task_hash"]) for row in rows}) == 900


def test_quick_schema_rejects_a_missing_method_family() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["methods"] = raw["methods"][:-1]
    raw["method_roles"] = raw["method_roles"][:-1]
    with pytest.raises(ValueError, match="method family"):
        QuickStudyConfig.model_validate(raw)


def test_quick_schema_rejects_non_one_x_budget() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["methods"][1]["planner"]["budget"] = {"multiplier": 4.0}
    with pytest.raises(ValueError, match="same 1x budget"):
        QuickStudyConfig.model_validate(raw)


def test_q1_parent_resolution_is_hashed_into_effective_method(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q1_selection(study)
    method = effective_method(study, lock, "q2a_verifier")
    assert method.planner.kind == PlannerKind.MCTS
    assert method.effective_decision_sha256s == (sha256_file(study.paths.p2_selection),)
    assert direct_control_name(study, lock, method.name) == "q1_mcts_1x"


def test_memory_uses_best_first_even_when_q1_winner_is_mcts(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q1_selection(study, selected="q1_mcts_1x")
    memory = effective_method(study, lock, "q2a_transposition_memory")
    assert memory.planner.kind == PlannerKind.BEST_FIRST
    assert direct_control_name(study, lock, memory.name) == "q1_best_first_1x"


def test_q1_selection_rejects_another_protocol(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    write_q1_selection(study, quick_spec="wrong")
    with pytest.raises(ValueError, match="another protocol"):
        validate_q1_selection(study, {"quick_spec_sha256": "quick"})


def test_q1_selection_rejects_a_replaced_input_result(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    write_q1_selection(study)
    path = result_path(
        study,
        method="q1_mcts_1x",
        backbone_seed=42,
        planner_seed=0,
        action_selection="corrected_v1",
    )
    write_json(path, {"tampered": True})
    with pytest.raises(ValueError, match="input hash mismatch"):
        validate_q1_selection(study, {"quick_spec_sha256": "quick"})


def test_q1_selection_rejects_a_replaced_q0_parity(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    write_q1_selection(study)
    parity = Path(study.paths.run_root) / "parity/parity_corrected_v1.json"
    write_json(parity, {"tampered": True})
    with pytest.raises(RuntimeError, match="Q0 parity gate failed"):
        validate_q1_selection(study, {"quick_spec_sha256": "quick"})


def test_ranker_component_path_uses_round_three(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q1_selection(study)
    ranker = effective_method(study, lock, "q2c_hard_negative_ranker")
    path = component_checkpoint_path(
        study, ranker, backbone_seed=42, planner_seed=104_729
    )
    assert path is not None
    assert path.name.endswith("round3.pt")


def test_result_path_keeps_action_protocol_separate() -> None:
    study = config()
    corrected = result_path(
        study,
        method="q1_mcts_1x",
        backbone_seed=42,
        planner_seed=0,
        action_selection="corrected_v1",
    )
    unmasked = result_path(
        study,
        method="q1_mcts_1x",
        backbone_seed=42,
        planner_seed=0,
        action_selection="unmasked",
    )
    assert corrected != unmasked
    assert corrected.name == "development_corrected_v1.json"


def test_parity_requires_every_core_task_field_to_match() -> None:
    rows = fake_parity_rows()
    reference = {"results": {"task_rows": rows}}
    candidate = {"tasks": [dict(row) for row in rows]}
    assert compare(reference, candidate)["status"] == "pass"
    candidate["tasks"][13]["path_length"] = 2
    with pytest.raises(ValueError, match="parity failed"):
        compare(reference, candidate)


def test_parity_rejects_a_different_executed_action_sequence() -> None:
    rows = fake_parity_rows()
    reference = {"results": {"task_rows": rows}}
    candidate_rows = json.loads(json.dumps(rows))
    candidate_rows[13]["decision_traces"][0]["executed_action"] = 2
    with pytest.raises(ValueError, match="parity failed"):
        compare(reference, {"tasks": candidate_rows})


def test_paired_bootstrap_is_stratified_deterministic_and_paired() -> None:
    control = {"tasks": fake_tasks(positive_count=0)}
    candidate = {"tasks": fake_tasks(positive_count=90)}
    first = stratified_paired_bootstrap(candidate, control, samples=2000, seed=7)
    second = stratified_paired_bootstrap(candidate, control, samples=2000, seed=7)
    assert first == second
    assert first["delta"] == pytest.approx(0.1)
    assert first["ci_low"] > 0.0
    assert first["confidence_level"] == pytest.approx(0.95)
    reversed_control = {"tasks": list(reversed(control["tasks"]))}
    with pytest.raises(ValueError, match="task order"):
        delta_sr(candidate, reversed_control)


def test_budget_validation_rejects_overspend_and_legacy_underuse() -> None:
    rows = [
        {
            "decision_traces": [
                {"compute": {"plan_transitions": 768}},
            ]
        }
    ]
    _validate_budget(rows, legacy=True)
    rows[0]["decision_traces"][0]["compute"]["plan_transitions"] = 769
    with pytest.raises(ValueError, match="exceeded"):
        _validate_budget(rows, legacy=False)
    rows[0]["decision_traces"][0]["compute"]["plan_transitions"] = 767
    with pytest.raises(ValueError, match="exactly 768"):
        _validate_budget(rows, legacy=True)


def test_q0_schedule_contains_two_references_two_candidates_and_two_parities() -> None:
    jobs = _q0_jobs(config(), str(CONFIG_PATH), None)
    assert len(jobs) == 6
    assert Counter(job.label.split(":", 1)[0] for job in jobs) == {
        "q0-reference": 2,
        "q0-candidate": 2,
        "q0-parity": 2,
    }


def test_stage_job_counts_match_the_locked_method_matrix(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q0_parity(study)
    q1 = stage_jobs(
        study,
        lock,
        stage="Q1",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    assert len(q1) == 10
    write_q1_selection(study)
    q2a = stage_jobs(
        study,
        lock,
        stage="Q2A",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    q2b = stage_jobs(
        study,
        lock,
        stage="Q2B",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    q2c = stage_jobs(
        study,
        lock,
        stage="Q2C",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    assert len(q2a) == 16
    assert len(q2b) == 22
    assert len(q2c) == 14


def test_schedule_text_is_reproducible(tmp_path: Path) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q0_parity(study)
    jobs = stage_jobs(
        study,
        lock,
        stage="Q1",
        config_path=str(CONFIG_PATH),
        device="cuda",
    )
    assert schedule_text(study, lock, "Q1", jobs) == schedule_text(
        study, lock, "Q1", jobs
    )


def test_q3_and_q4_schedules_complete_the_declared_seed_ladder(
    tmp_path: Path,
) -> None:
    study = with_temp_decisions(config(), tmp_path)
    lock = {"quick_spec_sha256": "quick"}
    write_q0_parity(study)
    write_q1_selection(study)
    write_shortlist(study)
    q3 = stage_jobs(
        study,
        lock,
        stage="Q3",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    assert len(q3) == 16
    assert all(":b42:" not in job.label for job in q3)
    write_final_selection(study, "q2b_denoising_icem")
    q4 = stage_jobs(
        study,
        lock,
        stage="Q4",
        config_path=str(CONFIG_PATH),
        device=None,
    )
    assert len(q4) == 96
    second_seed = [job for job in q4 if ":p130363" in job.label]
    assert len(second_seed) == 40


def test_crossed_bootstrap_resamples_backbone_as_an_independent_axis() -> None:
    candidate = {
        seed: [
            {"task_id": f"t{i}", "maze_size": 9, "success": float(seed != 42)}
            for i in range(20)
        ]
        for seed in (42, 43, 44)
    }
    control = {
        seed: [{"task_id": f"t{i}", "maze_size": 9, "success": 0.0} for i in range(20)]
        for seed in (42, 43, 44)
    }
    result = _nested_bootstrap(
        candidate, control, split="overall", samples=2000, seed=19
    )
    assert result["delta"] == pytest.approx(2 / 3)
    assert result["positive_backbones"] == 2
    assert result["backbone_count"] == 3
    assert result["resampling_design"] == "crossed_backbone_by_task_stratified"


def test_crossed_bootstrap_rejects_different_task_panels_between_backbones() -> None:
    base = [
        {"task_id": f"t{i}", "maze_size": 9, "success": float(i % 2)} for i in range(20)
    ]
    candidate = {42: base, 43: list(reversed(base))}
    control_base = [dict(row, success=0.0) for row in base]
    control = {42: control_base, 43: list(reversed(control_base))}
    with pytest.raises(ValueError, match="crossed task panel"):
        _nested_bootstrap(candidate, control, split="overall", samples=200, seed=3)


def test_checked_in_protocol_lock_reproduces() -> None:
    study = config()
    lock_path = ROOT / "vector_jepa_planner_full900_screen/configs/protocol_lock.json"
    if not lock_path.exists():
        pytest.skip("protocol lock is generated after implementation review")
    validate_lock(study, load_json(lock_path))
