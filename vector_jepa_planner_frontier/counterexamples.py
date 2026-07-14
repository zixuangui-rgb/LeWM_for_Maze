"""Mine and train exactly three topology-disjoint false-optimism rounds."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_closure.common import (
    bfs_distances_from,
    next_state,
    observe_state,
    read_jsonl,
    sha256_file,
)
from spatial_jepa_planning.common import validate_manifest_entry
from vector_jepa_planner_frontier.calibrate import load_heads
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    atomic_torch_save,
    hierarchical_seed,
    load_json,
    load_study_config,
    method_by_name,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    training_spec_sha256,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.effective_methods import resolve_effective_method
from vector_jepa_planner_frontier.evaluate import build_controller
from vector_jepa_planner_frontier.heads import pairwise_ranking_loss
from vector_jepa_planner_frontier.world_model import VectorWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", default="p6_track_f_counterexample_ranked")
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input-checkpoint")
    parser.add_argument("--dataset-output")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--diagnostic-limit", type=int, default=0)
    return parser.parse_args()


def mining_fold(task_hash: str) -> int:
    digest = hashlib.sha256(f"vector-jepa-mining-v1:{task_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3 + 1


def optimal_chunk(
    env: Any,
    state: int,
    goal: int,
    horizon: int,
    rng: np.random.Generator,
) -> list[int]:
    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    output: list[int] = []
    current = int(state)
    for _ in range(horizon):
        distance = int(distances[current])
        actions = [
            action
            for action in (1, 2, 3, 4)
            if int(distances[next_state(env, current, action)]) == distance - 1
        ]
        if not actions:
            raise RuntimeError("mining root cannot supply a full BFS action chunk")
        action = int(rng.choice(actions))
        output.append(action)
        current = next_state(env, current, action)
    return output


def execute_candidate(
    env: Any,
    state: int,
    goal: int,
    actions: list[int],
    *,
    progress_candidate_available: bool,
) -> dict[str, Any]:
    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    root_distance = int(distances[state])
    current = int(state)
    path = [current]
    invalid = False
    cycle_lengths: list[int] = []
    for action in actions:
        successor = next_state(env, current, int(action))
        invalid = invalid or successor == current
        current = successor
        for prior_index in range(max(0, len(path) - 8), len(path)):
            cycle = len(path) - prior_index
            if 2 <= cycle <= 8 and path[prior_index] == current:
                cycle_lengths.append(cycle)
        path.append(current)
    final_distance = int(distances[current])
    no_progress = final_distance >= root_distance
    reasons = []
    if invalid:
        reasons.append("wall_or_no_move")
    if cycle_lengths:
        reasons.append("cycle_2_to_8")
    if no_progress and progress_candidate_available:
        reasons.append("no_bfs_progress")
    return {
        "false_optimistic": bool(reasons),
        "reasons": reasons,
        "root_bfs_distance": root_distance,
        "final_bfs_distance": final_distance,
        "progress_candidate_available": bool(progress_candidate_available),
        "invalid": invalid,
        "cycle_lengths": sorted(set(cycle_lengths)),
        "path": path,
    }


def mine_round(
    config: Any,
    lock: dict[str, Any],
    method: Any,
    *,
    backbone_seed: int,
    planner_seed: int,
    round_index: int,
    checkpoint_path: Path,
    device: torch.device,
    diagnostic_limit: int,
) -> list[dict[str, Any]]:
    controller, _ = build_controller(
        config,
        lock,
        method,
        seed=backbone_seed,
        planner_seed=planner_seed,
        search_seed=config.protocol.search_seeds[0],
        device=device,
        action_selection="unmasked",
        component_checkpoint=checkpoint_path,
    )
    entries = [
        entry
        for entry in read_jsonl(resolve_path(config.paths.train_manifest))
        if mining_fold(str(entry["task_hash"])) == round_index
    ]
    if diagnostic_limit:
        entries = entries[:diagnostic_limit]
    rng = np.random.default_rng(
        hierarchical_seed(
            "counterexample-mining",
            backbone_seed,
            planner_seed,
            round_index,
        )
    )
    records: list[dict[str, Any]] = []
    for task_index, entry in enumerate(entries):
        env = validate_manifest_entry(entry)
        goal = int(entry["goal_cell"])
        distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
        free = np.flatnonzero((~env._maze_mask).reshape(-1))
        eligible = free[distances[free] >= method.planner.horizon]
        if eligible.size == 0:
            continue
        state = int(rng.choice(eligible))
        observation = observe_state(env, state)
        controller.reset(env, observation, task_index)
        if controller.context is None:
            raise RuntimeError("mining controller did not initialize its context")
        result = controller.planner.plan(
            controller.context,
            seed=hierarchical_seed(
                "counterexample-planner",
                backbone_seed,
                planner_seed,
                round_index,
                task_index,
            ),
        )
        bad_actions = result.sequence.tolist()
        root_distance = int(distances[state])
        progress_candidate_available = False
        for batch in result.candidate_batches:
            for raw_actions in batch.sequences:
                candidate_state = state
                for action in raw_actions.tolist():
                    candidate_state = next_state(env, candidate_state, int(action))
                if int(distances[candidate_state]) < root_distance:
                    progress_candidate_available = True
                    break
            if progress_candidate_available:
                break
        outcome = execute_candidate(
            env,
            state,
            goal,
            bad_actions,
            progress_candidate_available=progress_candidate_available,
        )
        if not outcome["false_optimistic"]:
            continue
        good_actions = optimal_chunk(env, state, goal, method.planner.horizon, rng)
        negative_actions = bad_actions
        negative_source = "planner_false_optimistic"
        if method.control.ranker_negatives == "random":
            negative_actions = rng.choice(
                np.asarray((1, 2, 3, 4), dtype=np.int64),
                size=method.planner.horizon,
                replace=True,
            ).tolist()
            if negative_actions == good_actions:
                negative_actions[0] = negative_actions[0] % 4 + 1
            negative_source = "matched_round_random_actions"
        records.append(
            {
                "task_hash": str(entry["task_hash"]),
                "topology_seed": int(entry["topology_seed"]),
                "maze_size": int(entry["maze_size"]),
                "source_state": state,
                "goal_state": goal,
                "good_actions": good_actions,
                "false_optimistic_actions": negative_actions,
                "mining_trigger_actions": bad_actions,
                "negative_source": negative_source,
                "outcome": outcome,
                "candidate_budget": result.ledger.to_dict(),
            }
        )
    return records


def train_ranker(
    config: Any,
    lock: dict[str, Any],
    method: Any,
    checkpoint: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    backbone_seed: int,
    planner_seed: int,
    round_index: int,
    device: torch.device,
) -> dict[str, Any]:
    model, _, _ = load_source_lewm(config, lock, seed=backbone_seed, device=device)
    if method.track == "J" and checkpoint.get("model_state_dict") is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = load_heads(checkpoint, device)
    if "ranker" not in modules:
        raise ValueError("counterexample method has no ranker head")
    ranker = modules["ranker"]
    ranker.train()
    for parameter in ranker.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=config.training.ranker_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    world_model = VectorWorldModel(
        model, device=device, history_size=method.planner.history_size
    )
    entry_by_hash = {
        str(entry["task_hash"]): entry
        for entry in read_jsonl(resolve_path(config.paths.train_manifest))
    }
    rng = np.random.default_rng(
        hierarchical_seed(
            "counterexample-ranker",
            backbone_seed,
            planner_seed,
            round_index,
        )
    )
    losses: list[float] = []
    if records:
        for _ in range(config.training.counterexample_round_steps):
            record = records[int(rng.integers(len(records)))]
            env = validate_manifest_entry(entry_by_hash[record["task_hash"]])
            source = world_model.encode(
                observe_state(env, int(record["source_state"])),
                int(record["maze_size"]),
            )
            goal = world_model.encode(
                observe_state(env, int(record["goal_state"])),
                int(record["maze_size"]),
            )
            context = world_model.initial_context(
                source, goal, maze_size=int(record["maze_size"])
            )
            good = world_model.rollout(
                context,
                np.asarray([record["good_actions"]], dtype=np.int64),
                semantics=method.planner.rollout_semantics,
            )
            bad = world_model.rollout(
                context,
                np.asarray([record["false_optimistic_actions"]], dtype=np.int64),
                semantics=method.planner.rollout_semantics,
            )
            good_score = ranker(source.squeeze(1), good.terminal, good.actions)
            bad_score = ranker(source.squeeze(1), bad.terminal, bad.actions)
            loss = pairwise_ranking_loss(good_score, bad_score)
            if not torch.isfinite(loss):
                raise FloatingPointError("counterexample ranker loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ranker.parameters(), config.training.grad_clip
            )
            optimizer.step()
            losses.append(float(loss.detach()))
    checkpoint["head_state_dicts"]["ranker"] = ranker.state_dict()
    return {
        "steps": config.training.counterexample_round_steps if records else 0,
        "mined_count": len(records),
        "final_loss": float(np.mean(losses[-500:])) if losses else None,
        "reason_counts": dict(
            Counter(
                reason for record in records for reason in record["outcome"]["reasons"]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("counterexample rounds require a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    base_method = method_by_name(config, args.method)
    if base_method.reuse_component_from is not None:
        raise ValueError("checkpoint reuse aliases cannot run counterexample training")
    method = resolve_effective_method(config, lock, base_method)
    if args.allow_dirty_worktree and args.diagnostic_limit == 0:
        raise ValueError("dirty worktrees are allowed only for mining diagnostics")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked matrix")
    if method.scorer.counterexample_ranker_weight <= 0.0:
        raise ValueError("selected method does not use counterexample ranking")
    if args.round > config.training.counterexample_rounds:
        raise ValueError("round lies outside the fixed three-round protocol")
    default_input = (
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
    input_path = resolve_path(args.input_checkpoint or default_input)
    dataset_path = resolve_path(
        args.dataset_output
        or config.paths.counterexample_dataset_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round=args.round,
        )
    )
    output_path = resolve_path(
        args.output
        or config.paths.counterexample_round_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            round=args.round,
        )
    )
    if output_path.exists():
        raise FileExistsError("counterexample round checkpoint is immutable")
    if args.diagnostic_limit < 0:
        raise ValueError("diagnostic limit cannot be negative")
    if args.diagnostic_limit and (args.dataset_output is None or args.output is None):
        raise ValueError("diagnostic mining requires explicit isolated output paths")
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed(
            "counterexample-round",
            args.backbone_seed,
            args.planner_seed,
            args.round,
        ),
        deterministic=True,
    )
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    allowed_stages = {"component_calibration", "counterexample_training_round"}
    if checkpoint.get("stage") not in allowed_stages:
        raise ValueError("counterexample input is not calibrated")
    if (
        checkpoint.get("method_name") != method.name
        or int(checkpoint.get("backbone_seed", -1)) != args.backbone_seed
        or int(checkpoint.get("planner_seed", -1)) != args.planner_seed
    ):
        raise ValueError("counterexample checkpoint method/seed mismatch")
    if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("input checkpoint analysis-spec mismatch")
    if checkpoint.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
    ):
        raise ValueError("input checkpoint training-spec mismatch")
    checkpoint_protocol = checkpoint.get("protocol", {})
    if checkpoint_protocol.get("git_dirty") is not False:
        raise ValueError("counterexample training rejects a dirty checkpoint")
    if checkpoint_protocol.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError("counterexample checkpoint code fingerprint mismatch")
    if args.round == 1 and checkpoint.get("stage") != "component_calibration":
        raise ValueError("round one must start from the calibrated base checkpoint")
    if args.round > 1 and int(checkpoint.get("counterexample_round", -1)) != (
        args.round - 1
    ):
        raise ValueError("counterexample rounds must be executed without gaps")
    started = time.perf_counter()
    source_checkpoint_sha256 = sha256_file(input_path)
    if dataset_path.exists():
        dataset = load_json(dataset_path)
        expected = {
            "schema": "vector-jepa-counterexamples-v1",
            "method": method.name,
            "backbone_seed": args.backbone_seed,
            "planner_seed": args.planner_seed,
            "round": args.round,
            "train_manifest_sha256": lock["train_manifest"]["sha256"],
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "diagnostic_limit": args.diagnostic_limit,
        }
        if any(dataset.get(key) != value for key, value in expected.items()):
            raise ValueError("existing counterexample dataset provenance mismatch")
        records = list(dataset.get("records", []))
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
            diagnostic_limit=args.diagnostic_limit,
        )
        dataset = {
            "schema": "vector-jepa-counterexamples-v1",
            "method": method.name,
            "backbone_seed": args.backbone_seed,
            "planner_seed": args.planner_seed,
            "round": args.round,
            "fold_definition": (
                "sha256('vector-jepa-mining-v1:' + task_hash) mod 3 plus one"
            ),
            "train_manifest_sha256": lock["train_manifest"]["sha256"],
            "source_checkpoint": str(input_path),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "diagnostic_limit": args.diagnostic_limit,
            "records": records,
        }
        atomic_json_dump(dataset_path, dataset)
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
            "stage": (
                "counterexample_diagnostic_round"
                if args.diagnostic_limit
                else "counterexample_training_round"
            ),
            "counterexample_round": args.round,
            "counterexample_dataset": str(dataset_path),
            "counterexample_dataset_sha256": sha256_file(dataset_path),
            "counterexample_training_summary": training_summary,
            "counterexample_elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    atomic_torch_save(output_path, checkpoint)


if __name__ == "__main__":
    main()
