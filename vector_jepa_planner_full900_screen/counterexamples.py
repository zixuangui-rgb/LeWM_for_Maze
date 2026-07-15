"""Run one locked hard- or random-negative ranker refinement round."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch

from final_closure.common import (
    bfs_distances_from,
    next_state,
    read_jsonl,
    sha256_file,
)
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier.common import (
    atomic_json_dump,
    atomic_torch_save,
    hierarchical_seed,
    resolve_device,
    set_seed,
    validate_compute_ledger,
)
from vector_jepa_planner_frontier.compat import checkpoint_path
from vector_jepa_planner_frontier.counterexamples import (
    execute_candidate,
    mine_round,
    mining_fold,
    train_ranker,
)
from vector_jepa_planner_full900_screen.common import (
    analysis_spec_sha256,
    load_config,
    load_json,
    method_by_name,
    require_clean_worktree,
    resolve_path,
    training_spec_sha256,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import effective_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def _validate_dataset(
    dataset: dict[str, object],
    *,
    config: object,
    lock: dict[str, object],
    method: object,
    backbone_seed: int,
    planner_seed: int,
    round_index: int,
    input_path: object,
) -> list[dict[str, object]]:
    expected = {
        "schema": "vector-jepa-full900-counterexamples-v1",
        "method": method.name,
        "backbone_seed": backbone_seed,
        "planner_seed": planner_seed,
        "round": round_index,
        "negative_source": method.control.ranker_negatives,
        "train_manifest_sha256": lock["train_manifest"]["sha256"],
        "source_checkpoint_sha256": sha256_file(input_path),
    }
    if any(dataset.get(key) != value for key, value in expected.items()):
        raise ValueError("counterexample dataset provenance mismatch")
    recorded_source = resolve_path(str(dataset.get("source_checkpoint", "")))
    if recorded_source.resolve() != resolve_path(input_path).resolve():
        raise ValueError("counterexample dataset source path mismatch")
    raw_records = dataset.get("records")
    if not isinstance(raw_records, list) or not all(
        isinstance(record, dict) for record in raw_records
    ):
        raise ValueError("counterexample dataset records must be a list of objects")
    records = list(raw_records)
    train_entries = {
        str(entry["task_hash"]): entry
        for entry in read_jsonl(resolve_path(config.paths.train_manifest))
    }
    seen_tasks: set[str] = set()
    expected_negative = (
        "matched_round_random_actions"
        if method.control.ranker_negatives == "random"
        else "planner_false_optimistic"
    )
    for record in records:
        task_hash = str(record.get("task_hash", ""))
        entry = train_entries.get(task_hash)
        if entry is None or task_hash in seen_tasks:
            raise ValueError("counterexample dataset has a foreign or duplicate task")
        seen_tasks.add(task_hash)
        if mining_fold(task_hash) != round_index:
            raise ValueError("counterexample dataset crossed its locked mining fold")
        identity = {
            "topology_seed": int(entry["topology_seed"]),
            "maze_size": int(entry["maze_size"]),
            "goal_state": int(entry["goal_cell"]),
            "negative_source": expected_negative,
        }
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("counterexample record identity mismatch")
        for key in (
            "good_actions",
            "false_optimistic_actions",
            "mining_trigger_actions",
        ):
            actions = record.get(key)
            if (
                not isinstance(actions, list)
                or len(actions) != method.planner.horizon
                or any(int(action) not in (1, 2, 3, 4) for action in actions)
            ):
                raise ValueError(f"counterexample record has invalid {key}")
        outcome = record.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("false_optimistic") is not True:
            raise ValueError("counterexample record lacks its mining trigger")
        env = validate_manifest_entry(entry)
        source_state = int(record.get("source_state", -1))
        goal_state = int(record["goal_state"])
        width = int(env.config.width)
        if (
            source_state < 0
            or source_state >= width * width
            or bool(env._maze_mask.reshape(-1)[source_state])
        ):
            raise ValueError("counterexample source state is not a free maze cell")
        distances = bfs_distances_from(env._maze_mask, goal_state, width)
        current = source_state
        for action in record["good_actions"]:
            successor = next_state(env, current, int(action))
            if int(distances[successor]) != int(distances[current]) - 1:
                raise ValueError("counterexample positive chunk is not BFS-optimal")
            current = successor
        recomputed_outcome = execute_candidate(
            env,
            source_state,
            goal_state,
            [int(action) for action in record["mining_trigger_actions"]],
            progress_candidate_available=bool(
                outcome.get("progress_candidate_available")
            ),
        )
        if recomputed_outcome != outcome:
            raise ValueError("counterexample mining outcome does not reproduce")
        if (
            expected_negative == "matched_round_random_actions"
            and record["false_optimistic_actions"] == record["good_actions"]
        ):
            raise ValueError("random-negative control duplicated its positive chunk")
        budget = record.get("candidate_budget")
        if not isinstance(budget, dict):
            raise ValueError("counterexample record lacks compute provenance")
        validate_compute_ledger(budget)
        if (
            int(budget["assist_transitions"]) != 0
            or int(budget["total_transitions"]) > method.planner.budget.transition_limit
        ):
            raise ValueError("counterexample mining exceeded its planner-only budget")
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    method = effective_method(config, lock, method_by_name(config, args.method))
    if method.scorer.counterexample_ranker_weight <= 0.0:
        raise ValueError("selected method is not a ranker experiment")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside historical seeds 42-51")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked nested seeds")
    require_clean_worktree()
    input_path = resolve_path(
        config.paths.component_checkpoint_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
        if args.round == 1
        else config.paths.counterexample_round_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round=args.round - 1,
        )
    )
    dataset_path = resolve_path(
        config.paths.counterexample_dataset_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round=args.round,
        )
    )
    output_path = resolve_path(
        config.paths.counterexample_round_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round=args.round,
        )
    )
    if output_path.exists():
        raise FileExistsError("counterexample round checkpoint is immutable")
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed(
            "full900-counterexample-round",
            args.backbone_seed,
            args.planner_seed,
            args.round,
        ),
        deterministic=True,
    )
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    allowed_stages = {"component_calibration", "counterexample_training_round"}
    if checkpoint.get("stage") not in allowed_stages:
        raise ValueError("counterexample input is not a calibrated ranker checkpoint")
    if checkpoint.get("method_name") != method.name:
        raise ValueError("counterexample checkpoint method mismatch")
    expected_training = training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
    )
    if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("counterexample checkpoint analysis-spec mismatch")
    if checkpoint.get("training_spec_sha256") != expected_training:
        raise ValueError("counterexample checkpoint training-spec mismatch")
    if (
        checkpoint.get("protocol", {}).get("code_fingerprint")
        != lock["code_fingerprint"]
    ):
        raise ValueError("counterexample checkpoint code fingerprint mismatch")
    if args.round == 1 and checkpoint.get("stage") != "component_calibration":
        raise ValueError("round one must start from calibration")
    if args.round > 1 and checkpoint.get("counterexample_round") != args.round - 1:
        raise ValueError("counterexample rounds must be contiguous")
    source_path = checkpoint_path(config, seed=args.backbone_seed)
    if checkpoint.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("counterexample checkpoint and source backbone diverged")

    source_sha = sha256_file(input_path)
    started = time.perf_counter()
    if dataset_path.exists():
        dataset = load_json(dataset_path)
        records = _validate_dataset(
            dataset,
            config=config,
            lock=lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round_index=args.round,
            input_path=input_path,
        )
    else:
        records = mine_round(
            config,
            lock,
            method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round_index=args.round,
            checkpoint_path=input_path,
            device=device,
            diagnostic_limit=0,
        )
        dataset = {
            "schema": "vector-jepa-full900-counterexamples-v1",
            "method": method.name,
            "backbone_seed": args.backbone_seed,
            "planner_seed": args.planner_seed,
            "round": args.round,
            "negative_source": method.control.ranker_negatives,
            "train_manifest_sha256": lock["train_manifest"]["sha256"],
            "source_checkpoint": str(input_path),
            "source_checkpoint_sha256": source_sha,
            "records": records,
        }
        atomic_json_dump(dataset_path, dataset)
        records = _validate_dataset(
            dataset,
            config=config,
            lock=lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round_index=args.round,
            input_path=input_path,
        )
    training_summary = train_ranker(
        config,
        lock,
        method,
        checkpoint,
        records,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
        round_index=args.round,
        device=device,
    )
    checkpoint.update(
        {
            "stage": "counterexample_training_round",
            "counterexample_round": args.round,
            "counterexample_dataset": str(dataset_path),
            "counterexample_dataset_sha256": sha256_file(dataset_path),
            "counterexample_training_summary": training_summary,
            "counterexample_reason_counts": dict(
                Counter(
                    reason
                    for record in records
                    for reason in record["outcome"]["reasons"]
                )
            ),
            "counterexample_elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    atomic_torch_save(output_path, checkpoint)


if __name__ == "__main__":
    main()
