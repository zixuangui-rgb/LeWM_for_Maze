from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from final_closure.common import bfs_distances_from, next_state, read_jsonl, sha256_file
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier.common import ComputeLedger, method_by_name
from vector_jepa_planner_frontier.counterexamples import execute_candidate, mining_fold
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_frontier.schemas import PlannerKind
from vector_jepa_planner_full900_screen.analysis import (
    delta_sr,
    exact_stratified_paired_bootstrap,
    stratified_paired_bootstrap,
)
from vector_jepa_planner_full900_screen.audit_protocol import _leaf_differences
from vector_jepa_planner_full900_screen.common import (
    component_checkpoint_path,
    load_config,
    load_json,
    result_path,
    validate_lock,
)
from vector_jepa_planner_full900_screen.counterexamples import _validate_dataset
from vector_jepa_planner_full900_screen.evaluate import _validate_budget
from vector_jepa_planner_full900_screen.freeze_q1 import _assert_bridge_parity
from vector_jepa_planner_full900_screen.lock_protocol import build_lock
from vector_jepa_planner_full900_screen.methods import (
    COMPONENT_PARITY_GROUPS,
    component_parity_audits,
    direct_control_name,
    effective_method,
    validate_component_parity,
    validate_q1_selection,
)
from vector_jepa_planner_full900_screen.parity import PARITY_FIELDS, compare
from vector_jepa_planner_full900_screen.run_plan import (
    _q0_jobs,
    schedule_text,
    stage_jobs,
)
from vector_jepa_planner_full900_screen.schemas import QuickStudyConfig
from vector_jepa_planner_full900_screen.summarize import _aggregate, _nested_bootstrap

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
            "component_training_template": str(
                tmp_path
                / (
                    "checkpoints/{method}_backbone{backbone_seed}_"
                    "planner{planner_seed}_train.pt"
                )
            ),
            "component_checkpoint_template": str(
                tmp_path
                / (
                    "checkpoints/{method}_backbone{backbone_seed}_"
                    "planner{planner_seed}_calibrated.pt"
                )
            ),
            "counterexample_round_template": str(
                tmp_path
                / (
                    "checkpoints/{method}_backbone{backbone_seed}_"
                    "planner{planner_seed}_round{round}.pt"
                )
            ),
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


def write_component_parity_checkpoints(
    study: QuickStudyConfig,
    *,
    backbone_seeds: tuple[int, ...],
    planner_seeds: tuple[int, ...] = (104_729,),
) -> None:
    for group_index, names in enumerate(COMPONENT_PARITY_GROUPS.values()):
        for backbone_seed in backbone_seeds:
            for planner_seed in planner_seeds:
                for name in names:
                    method = method_by_name(study, name)
                    path = component_checkpoint_path(
                        study,
                        method,
                        backbone_seed=backbone_seed,
                        planner_seed=planner_seed,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "stage": "component_calibration",
                            "method_name": name,
                            "backbone_seed": backbone_seed,
                            "planner_seed": planner_seed,
                            "source_checkpoint_sha256": f"source-{backbone_seed}",
                            "train_manifest_sha256": "train",
                            "validation_manifest_sha256": "validation",
                            "head_config": {"group": group_index},
                            "head_state_dicts": {
                                "head": {
                                    "weight": torch.tensor(
                                        [group_index, backbone_seed, planner_seed],
                                        dtype=torch.float32,
                                    )
                                }
                            },
                            "training_summary": {
                                "loss": 0.5,
                                "elapsed_seconds": float(group_index + 1),
                            },
                            "validation_metrics": {"accuracy": 0.75},
                            "initialization_parent": None,
                            "joint_counterexample_provenance": [],
                        },
                        path,
                    )


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
                "schema": "vector-jepa-full900-b0-parity-v1",
                "status": "pass",
                "task_count": 900,
                "compared_fields": list(PARITY_FIELDS),
                "protocol_id": study.protocol_id,
                "quick_spec_sha256": "quick",
                "action_selection": action,
                "executed_actions_compared": True,
                "mismatch_count": 0,
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
    ranked_names = [selected]
    ranked_names.extend(
        name
        for name in (
            "q1_control_categorical_cem_1x",
            "q1_icem_1x",
            "q1_beam_1x",
            "q1_best_first_1x",
            "q1_mcts_1x",
        )
        if name != selected
    )
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
                {
                    "method": name,
                    "corrected_sr": 1.0 - index * 0.1,
                    "corrected_ood_sr": 1.0 - index * 0.1,
                    "unmasked_sr": 1.0 - index * 0.1,
                    "planner_forward_calls_per_decision": float(index + 1),
                }
                for index, name in enumerate(ranked_names)
            ],
            "input_sha256s": inputs,
        },
    )


def write_shortlist(study: QuickStudyConfig, names: list[str] | None = None) -> None:
    names = names or ["q2b_denoising_icem"]
    write_component_parity_checkpoints(study, backbone_seeds=(42,))
    eligible = [role.name for role in study.method_roles if role.advancement_eligible]
    lock = {"quick_spec_sha256": "quick"}
    required_methods = {"b0_legacy_l2_cem"}
    for name in eligible:
        method = effective_method(study, lock, name)
        required_methods.add(method.name)
        required_methods.add(direct_control_name(study, lock, method.name))
    inputs = {}
    for method_name in required_methods:
        method = effective_method(study, lock, method_name)
        planner_seed = 104_729 if method.component_checkpoint_required else 0
        for action in study.replication.action_selections:
            path = result_path(
                study,
                method=method_name,
                backbone_seed=42,
                planner_seed=planner_seed,
                action_selection=action,
            )
            write_json(path, {"method": method_name, "action": action})
            inputs[f"{method_name}:b42:p{planner_seed}:{action}"] = sha256_file(path)
    write_json(
        Path(study.paths.p5_advancement),
        {
            "schema": "vector-jepa-full900-shortlist-v1",
            "protocol_id": study.protocol_id,
            "quick_spec_sha256": "quick",
            "q1_parent_sha256": sha256_file(study.paths.p2_selection),
            "shortlist": names,
            "candidate_audits": [{"method": name} for name in eligible],
            "component_parity_audits": component_parity_audits(
                study,
                candidates=("q2b_vector_dts", "q2b_bidirectional"),
                backbone_seeds=(42,),
                planner_seeds=(104_729,),
                include_dts_secondary_control=True,
            ),
            "input_sha256s": inputs,
        },
    )


def write_final_selection(study: QuickStudyConfig, winner: str) -> None:
    lock = {"quick_spec_sha256": "quick"}
    write_component_parity_checkpoints(
        study,
        backbone_seeds=study.replication.expansion_backbone_seeds,
    )
    required_methods = {
        "b0_legacy_l2_cem",
        winner,
        direct_control_name(study, lock, winner),
    }
    inputs = {}
    for method_name in required_methods:
        method = effective_method(study, lock, method_name)
        planner_seed = 104_729 if method.component_checkpoint_required else 0
        for backbone_seed in study.replication.expansion_backbone_seeds:
            for action in study.replication.action_selections:
                path = result_path(
                    study,
                    method=method_name,
                    backbone_seed=backbone_seed,
                    planner_seed=planner_seed,
                    action_selection=action,
                )
                write_json(path, {"method": method_name, "action": action})
                key = f"{method_name}:b{backbone_seed}:p{planner_seed}:{action}"
                inputs[key] = sha256_file(path)
    write_json(
        Path(study.paths.p7_selection),
        {
            "schema": "vector-jepa-full900-final-winner-v1",
            "protocol_id": study.protocol_id,
            "quick_spec_sha256": "quick",
            "shortlist_sha256": sha256_file(study.paths.p5_advancement),
            "winner": winner,
            "candidate_audits": [{"method": winner}],
            "component_parity_audits": component_parity_audits(
                study,
                candidates=(winner,),
                backbone_seeds=study.replication.expansion_backbone_seeds,
                planner_seeds=(104_729,),
            ),
            "input_sha256s": inputs,
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


def test_config_locks_twelve_candidates_and_nineteen_executable_methods() -> None:
    study = config()
    assert len(study.methods) == 19
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
        assert _leaf_differences(bridge, method_by_name(study, name)) == {
            "name",
            "planner.kind",
        }


def test_radical_and_ranker_controls_change_only_declared_fields() -> None:
    study = config()
    expected = {
        (
            "q2b_vector_dts",
            "q2b_control_dts_uniform_expansion",
        ): {"name", "control.dts_expansion"},
        (
            "q2b_bidirectional",
            "q2b_control_bidirectional_forward",
        ): {"name", "planner.kind"},
        (
            "q2b_denoising_icem",
            "q2b_control_denoising_uniform",
        ): {
            "name",
            "proposal.kind",
            "proposal.learned_weight",
            "proposal.uniform_weight",
            "component_checkpoint_required",
        },
        (
            "q2c_hard_negative_ranker",
            "q2c_control_random_negative_ranker",
        ): {"name", "control.ranker_negatives"},
    }
    for (candidate, control), fields in expected.items():
        assert (
            _leaf_differences(
                method_by_name(study, candidate), method_by_name(study, control)
            )
            == fields
        )


def test_shared_component_parity_is_exact_and_ignores_only_wall_time(
    tmp_path: Path,
) -> None:
    study = with_temp_decisions(config(), tmp_path)
    write_component_parity_checkpoints(study, backbone_seeds=(42,))
    control = method_by_name(study, "q2b_control_dts_uniform_expansion")
    control_path = component_checkpoint_path(
        study,
        control,
        backbone_seed=42,
        planner_seed=104_729,
    )
    checkpoint = torch.load(control_path, map_location="cpu", weights_only=False)
    checkpoint["training_summary"]["elapsed_seconds"] = 999.0
    torch.save(checkpoint, control_path)
    audit = validate_component_parity(
        study,
        candidate="q2b_vector_dts",
        backbone_seed=42,
        planner_seed=104_729,
        include_secondary_controls=True,
    )
    assert audit["status"] == "exact_match"

    checkpoint["head_state_dicts"]["head"]["weight"][0] += 1
    torch.save(checkpoint, control_path)
    with pytest.raises(ValueError, match="shared learned components diverged"):
        validate_component_parity(
            study,
            candidate="q2b_vector_dts",
            backbone_seed=42,
            planner_seed=104_729,
            include_secondary_controls=True,
        )


def test_dts_advancement_control_matches_search_budget_not_direct_policy() -> None:
    study = config()
    lock = {"quick_spec_sha256": "unused"}
    role = next(role for role in study.method_roles if role.name == "q2b_vector_dts")
    assert role.direct_control == "q2b_control_dts_uniform_expansion"
    learned = method_by_name(study, "q2b_vector_dts")
    matched = method_by_name(study, "q2b_control_dts_uniform_expansion")
    direct = method_by_name(study, "q2b_control_dts_direct")
    assert (
        learned.planner.budget.transition_limit
        == matched.planner.budget.transition_limit
    )
    assert matched.control.dts_expansion == "random"
    assert direct.control.dts_expansion == "direct"
    assert direct_control_name(study, lock, learned.name) == matched.name


def test_bidirectional_screen_reserves_budget_for_stitched_reranking() -> None:
    study = config()
    candidate = method_by_name(study, "q2b_bidirectional")
    control = method_by_name(study, "q2b_control_bidirectional_forward")
    assert candidate.planner.num_candidates == control.planner.num_candidates == 48
    half = candidate.planner.horizon // 2
    two_sided_rollout = 2 * candidate.planner.num_candidates * half
    rerank_count = (
        candidate.planner.budget.transition_limit - two_sided_rollout
    ) // candidate.planner.horizon
    assert two_sided_rollout == 576
    assert rerank_count == 16


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


def test_counterexample_resume_rejects_foreign_or_wrong_fold_data(
    tmp_path: Path,
) -> None:
    study = config()
    method = method_by_name(study, "q2c_hard_negative_ranker")
    entry = next(
        row
        for row in read_jsonl(ROOT / study.paths.train_manifest)
        if mining_fold(str(row["task_hash"])) == 1
    )
    input_path = tmp_path / "round0.pt"
    input_path.write_bytes(b"checkpoint")
    env = validate_manifest_entry(entry)
    goal = int(entry["goal_cell"])
    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    source = next(
        int(state)
        for state in np.flatnonzero((~env._maze_mask).reshape(-1))
        if int(distances[int(state)]) >= method.planner.horizon
        and any(
            next_state(env, int(state), action) == int(state) for action in (1, 2, 3, 4)
        )
    )
    good_actions = []
    current = source
    for _ in range(method.planner.horizon):
        action = next(
            action
            for action in (1, 2, 3, 4)
            if int(distances[next_state(env, current, action)])
            == int(distances[current]) - 1
        )
        good_actions.append(action)
        current = next_state(env, current, action)
    invalid_action = next(
        action for action in (1, 2, 3, 4) if next_state(env, source, action) == source
    )
    trigger_actions = [invalid_action] * method.planner.horizon
    outcome = execute_candidate(
        env,
        source,
        goal,
        trigger_actions,
        progress_candidate_available=True,
    )
    record = {
        "task_hash": str(entry["task_hash"]),
        "topology_seed": int(entry["topology_seed"]),
        "maze_size": int(entry["maze_size"]),
        "source_state": source,
        "goal_state": int(entry["goal_cell"]),
        "good_actions": good_actions,
        "false_optimistic_actions": trigger_actions,
        "mining_trigger_actions": trigger_actions,
        "negative_source": "planner_false_optimistic",
        "outcome": outcome,
        "candidate_budget": ComputeLedger().to_dict(),
    }
    dataset = {
        "schema": "vector-jepa-full900-counterexamples-v1",
        "method": method.name,
        "backbone_seed": 42,
        "planner_seed": 104_729,
        "round": 1,
        "negative_source": "hard_three_rounds",
        "train_manifest_sha256": "train",
        "source_checkpoint": str(input_path),
        "source_checkpoint_sha256": sha256_file(input_path),
        "records": [record],
    }
    lock = {"train_manifest": {"sha256": "train"}}
    assert (
        len(
            _validate_dataset(
                dataset,
                config=study,
                lock=lock,
                method=method,
                backbone_seed=42,
                planner_seed=104_729,
                round_index=1,
                input_path=input_path,
            )
        )
        == 1
    )
    dataset["records"] = [{**record, "task_hash": "foreign"}]
    with pytest.raises(ValueError, match="foreign or duplicate"):
        _validate_dataset(
            dataset,
            config=study,
            lock=lock,
            method=method,
            backbone_seed=42,
            planner_seed=104_729,
            round_index=1,
            input_path=input_path,
        )
    over_budget = ComputeLedger().to_dict()
    over_budget["plan_transitions"] = 769
    over_budget["total_transitions"] = 769
    dataset["records"] = [{**record, "candidate_budget": over_budget}]
    with pytest.raises(ValueError, match="planner-only budget"):
        _validate_dataset(
            dataset,
            config=study,
            lock=lock,
            method=method,
            backbone_seed=42,
            planner_seed=104_729,
            round_index=1,
            input_path=input_path,
        )


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
    candidate["tasks"][13]["goal_cell"] = 3
    with pytest.raises(ValueError, match="parity failed"):
        compare(reference, candidate)


def test_parity_rejects_a_different_executed_action_sequence() -> None:
    rows = fake_parity_rows()
    reference = {"results": {"task_rows": rows}}
    candidate_rows = json.loads(json.dumps(rows))
    candidate_rows[13]["decision_traces"][0]["executed_action"] = 2
    with pytest.raises(ValueError, match="parity failed"):
        compare(reference, {"tasks": candidate_rows})


def test_q1_bridge_uses_the_complete_q0_parity_field_set() -> None:
    rows = fake_parity_rows()
    left = {"tasks": rows}
    right = {"tasks": json.loads(json.dumps(rows))}
    _assert_bridge_parity(left, right)
    right["tasks"][0]["repeat_states"] = 1
    with pytest.raises(ValueError, match="bridge diverged"):
        _assert_bridge_parity(left, right)


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


def test_seed42_interval_enumerates_extreme_bonferroni_tail_exactly() -> None:
    control = {"tasks": fake_tasks(positive_count=0)}
    candidate = {"tasks": fake_tasks(positive_count=90)}
    result = exact_stratified_paired_bootstrap(candidate, control, alpha=0.05 / 48)
    assert result["delta"] == pytest.approx(0.1)
    assert result["ci_low"] > 0.0
    assert result["interval_engine"] == "exact_stratified_empirical_bootstrap"
    assert result["monte_carlo_samples"] == 0
    assert result["support_points"] == 1801


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
    assert len(q2b) == 26
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


def test_final_aggregate_reports_size_sd_and_compute_per_decision() -> None:
    tasks = {
        seed: [
            {
                "task_id": f"t{index}",
                "maze_size": 9 if index == 0 else 23,
                "success": float(index == 0),
                "spl": float(index == 0),
                "loop_or_cycle": float(index == 1),
                "invalid_actions": 1.0,
                "path_length": 2.0,
                "assistance_rate": 0.25,
                "decision_count": 2.0,
                "auxiliary": {
                    "plan_transitions": 20.0,
                    "assist_transitions": 2.0,
                    "total_transitions": 22.0,
                },
                "episode_seconds": 0.5,
            }
            for index in range(2)
        ]
        for seed in (42, 43)
    }
    summary = _aggregate(tasks)
    assert summary["overall"]["sr"]["mean"] == pytest.approx(0.5)
    assert summary["overall"]["sr"]["sd"] == pytest.approx(0.0)
    assert summary["overall"]["plan_transitions_per_decision"]["mean"] == pytest.approx(
        10.0
    )
    assert summary["by_size"]["9"]["task_count_per_backbone"] == 1


def test_checked_in_protocol_lock_reproduces() -> None:
    study = config()
    lock_path = ROOT / "vector_jepa_planner_full900_screen/configs/protocol_lock.json"
    if not lock_path.exists():
        pytest.skip("protocol lock is generated after implementation review")
    validate_lock(study, load_json(lock_path))


def test_protocol_lock_captures_both_dependency_files() -> None:
    lock = build_lock(CONFIG_PATH)
    assert lock["environment_spec"]["path"] == "pyproject.toml"
    assert lock["environment_lock"]["path"] == "uv.lock"
    assert lock["environment_spec"]["sha256"] == sha256_file(ROOT / "pyproject.toml")
    assert lock["environment_lock"]["sha256"] == sha256_file(ROOT / "uv.lock")
