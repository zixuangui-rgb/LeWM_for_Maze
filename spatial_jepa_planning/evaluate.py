#!/usr/bin/env python3
"""Evaluate learned, decoded-map, and oracle planners on fixed Set-B tasks."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_jepa_planning.common import (
    ACTION_IDS,
    ACTION_TO_SLOT,
    build_map_targets,
    load_planner_checkpoint,
    load_representation_checkpoint,
    next_state,
    observe_state,
    parse_int_list,
    planner_features,
    protocol_metadata,
    read_jsonl,
    resolve_device,
    set_agent_state,
    set_seed,
    sha256_file,
    strict_json_dump,
    summarize_rows,
    task_id,
    validate_manifest_entry,
)
from spatial_jepa_planning.models import (
    OracleValueIteration,
    PlannerOutput,
    SpatialRepresentation,
    neighbor_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("learned", "decoded_bfs", "oracle_bfs", "oracle_vi"),
        required=True,
    )
    parser.add_argument("--planner-ckpt", default=None)
    parser.add_argument("--representation-ckpt", default=None)
    parser.add_argument(
        "--train-manifest", default="data/splits/unisize_train_manifest.jsonl"
    )
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", default="8,16,32,64,128,256")
    parser.add_argument(
        "--decision-source", choices=("policy", "value"), default="policy"
    )
    parser.add_argument(
        "--action-selection",
        choices=("corrected", "model_valid", "unmasked"),
        default="corrected",
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seen-max-size", type=int, default=21)
    parser.add_argument("--field-states-per-maze", type=int, default=24)
    parser.add_argument("--field-pairs-per-maze", type=int, default=128)
    parser.add_argument("--recompute-every-step", action="store_true")
    parser.add_argument(
        "--decoded-action-selection",
        choices=("predicted", "corrected"),
        default="predicted",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-protocol-mismatch", action="store_true")
    return parser.parse_args()


def select_entries(
    entries: list[dict[str, Any]],
    max_per_size: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["maze_size"])].append(entry)
    selected: list[dict[str, Any]] = []
    for size in sorted(grouped):
        group = list(grouped[size])
        rng.shuffle(group)
        selected.extend(group[:max_per_size] if max_per_size > 0 else group)
    return selected[:limit] if limit > 0 else selected


def verify_checkpoint_protocol(
    checkpoint: dict[str, Any],
    manifest: str | Path,
    allow_mismatch: bool,
) -> None:
    expected = checkpoint.get("protocol", {}).get("eval_manifest_sha256")
    actual = sha256_file(manifest)
    if expected and expected != actual and not allow_mismatch:
        raise ValueError(
            "checkpoint/evaluation manifest mismatch: "
            f"checkpoint={expected}, current={actual}. "
            "Use --allow-protocol-mismatch only for a labelled "
            "non-comparable diagnostic."
        )


def corrected_actions(env: Any, state: int, previous: int | None) -> list[int]:
    moving: list[int] = []
    non_backtracking: list[int] = []
    for action in ACTION_IDS:
        candidate = next_state(env, state, int(action))
        if candidate == state:
            continue
        moving.append(int(action))
        if previous is None or candidate != previous:
            non_backtracking.append(int(action))
    return non_backtracking or moving


def choose_learned_action(
    output: PlannerOutput,
    env: Any,
    state: int,
    previous: int | None,
    decision_source: str,
    action_selection: str,
) -> int:
    row, col = divmod(int(state), int(env.config.width))
    if action_selection == "corrected":
        candidates = corrected_actions(env, state, previous)
    elif action_selection == "model_valid":
        candidates = [
            int(action)
            for action in ACTION_IDS
            if float(output.valid_logits[0, ACTION_TO_SLOT[int(action)], row, col])
            > 0.0
        ]
    else:
        candidates = [int(action) for action in ACTION_IDS]
    if not candidates:
        candidates = [int(action) for action in ACTION_IDS]

    best_action = candidates[0]
    best_score = -math.inf if decision_source == "policy" else math.inf
    for action in candidates:
        if decision_source == "policy":
            score = float(output.policy_logits[0, ACTION_TO_SLOT[action], row, col])
            if score > best_score:
                best_score = score
                best_action = action
        else:
            candidate = next_state(env, state, action)
            next_row, next_col = divmod(candidate, int(env.config.width))
            score = float(output.value[0, next_row, next_col])
            if score < best_score:
                best_score = score
                best_action = action
    return int(best_action)


def decode_map(
    representation: SpatialRepresentation,
    observation: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        latent = representation.planning_latent(obs)
        decoded = representation.map_decoder(latent)
    wall_probability = torch.sigmoid(decoded["wall_logits"])[0]
    wall = (wall_probability >= 0.5).cpu().numpy().astype(bool)
    agent = int(decoded["agent_logits"][0].flatten().argmax())
    goal = int(decoded["goal_logits"][0].flatten().argmax())
    wall.flat[agent] = False
    wall.flat[goal] = False
    true_wall = observation[..., 1] > 0.5
    intersection = int(np.logical_and(wall, true_wall).sum())
    union = int(np.logical_or(wall, true_wall).sum())
    true_agent = int(observation[..., 2].reshape(-1).argmax())
    true_goal = int(observation[..., 3].reshape(-1).argmax())
    metrics = {
        "wall_intersection": float(intersection),
        "wall_union": float(union),
        "agent_correct": float(agent == true_agent),
        "goal_correct": float(goal == true_goal),
    }
    return wall, agent, goal, metrics


def choose_bfs_action(
    wall: np.ndarray,
    predicted_state: int,
    predicted_goal: int,
    candidates: list[int],
) -> tuple[int, bool]:
    from diagnostics.common import bfs_distances_from

    width = int(wall.shape[1])
    if not 0 <= predicted_state < wall.size or not 0 <= predicted_goal < wall.size:
        return candidates[0], False
    if wall.reshape(-1)[predicted_state] or wall.reshape(-1)[predicted_goal]:
        return candidates[0], False
    distances = bfs_distances_from(wall, predicted_goal, width)
    if int(distances[predicted_state]) < 0:
        return candidates[0], False
    best_action = candidates[0]
    best_distance = math.inf
    row, col = divmod(predicted_state, width)
    deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    for action in candidates:
        drow, dcol = deltas[action]
        next_row, next_col = row + drow, col + dcol
        if not (0 <= next_row < wall.shape[0] and 0 <= next_col < width):
            continue
        candidate = next_row * width + next_col
        distance = int(distances[candidate])
        if not wall[next_row, next_col] and 0 <= distance < best_distance:
            best_distance = distance
            best_action = action
    return int(best_action), math.isfinite(best_distance)


def run_navigation(
    entry: dict[str, Any],
    *,
    action_fn: Callable[
        [Any, np.ndarray, int, int | None], tuple[int, dict[str, float]]
    ],
    max_steps: int,
) -> dict[str, Any]:
    env = validate_manifest_entry(entry)
    start = int(entry["start_cell"])
    goal = int(entry["goal_cell"])
    optimal_length = int(entry["bfs_path_length"])
    observation = set_agent_state(env, start)
    state = start
    previous: int | None = None
    path = [state]
    invalid_actions = 0
    auxiliary: defaultdict[str, float] = defaultdict(float)
    for _ in range(max_steps):
        if state == goal:
            break
        action, action_metrics = action_fn(env, observation, state, previous)
        for name, value in action_metrics.items():
            auxiliary[name] += float(value)
        old_state = state
        observation, _, _, _, info = env.step(action)
        state = int(info["state"])
        previous = old_state
        invalid_actions += int(state == old_state)
        path.append(state)
    success = state == goal
    final_distances = build_map_targets(env, torch.device("cpu"))["distance"]
    final_row, final_col = divmod(state, int(env.config.width))
    visit_counts = Counter(path)
    return {
        "task_id": task_id(entry),
        "maze_size": int(entry["maze_size"]),
        "topology_seed": int(entry["topology_seed"]),
        "start_cell": start,
        "goal_cell": goal,
        "optimal_length": optimal_length,
        "success": bool(success),
        "path_length": len(path) - 1,
        "spl": float(optimal_length / max(optimal_length, len(path) - 1))
        if success
        else 0.0,
        "invalid_actions": invalid_actions,
        "repeat_states": int(sum(max(count - 1, 0) for count in visit_counts.values())),
        "max_state_visits": int(max(visit_counts.values())),
        "loop_or_cycle": bool(max(visit_counts.values()) >= 4),
        "final_bfs_distance": int(final_distances[final_row, final_col]),
        "auxiliary": dict(auxiliary),
    }


class FieldAccumulator:
    def __init__(self) -> None:
        self.correct = 0
        self.active = 0
        self.margins: list[np.ndarray] = []
        self.sampled_top1: list[float] = []
        self.sampled_margin: list[float] = []
        self.predicted_values: list[np.ndarray] = []
        self.target_values: list[np.ndarray] = []

    def update(
        self,
        output: PlannerOutput,
        targets: dict[str, torch.Tensor],
        decision_source: str,
        sampled_states: torch.Tensor,
    ) -> None:
        valid = targets["valid_action_mask"].bool()
        optimal = targets["optimal_action_mask"].bool()
        if decision_source == "value":
            scores = -neighbor_stack(
                output.value, float(output.value.shape[-1] ** 2 * 4 + 1)
            )
        else:
            scores = output.policy_logits
        negative = torch.finfo(scores.dtype).min
        prediction = scores.masked_fill(~valid, negative).argmax(dim=1)
        active = optimal.any(dim=1) & (valid.sum(dim=1) >= 2)
        correct = optimal.gather(1, prediction.unsqueeze(1)).squeeze(1) & active
        self.correct += int(correct.sum())
        self.active += int(active.sum())
        suboptimal = valid & ~optimal
        paired = active & suboptimal.any(dim=1)
        best_optimal = scores.masked_fill(~optimal, negative).max(dim=1).values
        best_bad = scores.masked_fill(~suboptimal, negative).max(dim=1).values
        margin = best_optimal - best_bad
        self.margins.append(margin[paired].detach().cpu().numpy())
        sampled_active = active & sampled_states.bool()
        if bool(sampled_active.any()):
            self.sampled_top1.append(
                float(correct[sampled_active].float().mean().detach().cpu())
            )
        sampled_paired = paired & sampled_states.bool()
        if bool(sampled_paired.any()):
            self.sampled_margin.append(
                float(margin[sampled_paired].mean().detach().cpu())
            )
        free = targets["free_mask"].bool()
        self.predicted_values.append(output.value[free].detach().cpu().numpy())
        self.target_values.append(targets["distance"][free].detach().cpu().numpy())

    def summary(self) -> dict[str, Any]:
        margins = np.concatenate(self.margins) if self.margins else np.array([])
        predicted = (
            np.concatenate(self.predicted_values)
            if self.predicted_values
            else np.array([])
        )
        target = (
            np.concatenate(self.target_values) if self.target_values else np.array([])
        )
        if len(predicted) >= 2 and float(np.std(predicted)) > 1e-12:
            pearson = float(np.corrcoef(predicted, target)[0, 1])
        else:
            pearson = None
        residual = (
            float(((predicted - target) ** 2).sum()) if len(predicted) else math.nan
        )
        total = float(((target - target.mean()) ** 2).sum()) if len(target) else 0.0
        return {
            "n_states": self.active,
            "local_top1": float(np.mean(self.sampled_top1))
            if self.sampled_top1
            else None,
            "local_margin": float(np.mean(self.sampled_margin))
            if self.sampled_margin
            else None,
            "all_cell_local_top1": self.correct / max(self.active, 1),
            "all_cell_local_margin": float(margins.mean()) if len(margins) else None,
            "value_pearson": pearson,
            "value_r2": 1.0 - residual / total if total > 0 else None,
        }


def evaluate_learned(
    entries: list[dict[str, Any]],
    planner: torch.nn.Module,
    representation: SpatialRepresentation | None,
    checkpoint: dict[str, Any],
    iteration_values: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    planner_type = str(checkpoint["planner_config"]["planner_type"])
    if planner_type.startswith("feedforward"):
        iteration_values = (int(checkpoint["planner_config"]["depth"]),)
    results: dict[str, Any] = {}
    for iterations in iteration_values:
        rows: list[dict[str, Any]] = []
        diagnostic_rng = np.random.default_rng(args.seed + 202)
        field_all = FieldAccumulator()
        field_seen = FieldAccumulator()
        field_ood = FieldAccumulator()
        by_size = defaultdict(FieldAccumulator)
        started = time.time()
        for index, entry in enumerate(entries):
            env_for_field = validate_manifest_entry(entry, check_bfs=False)
            start_obs = observe_state(env_for_field, int(entry["start_cell"]))
            obs_tensor = torch.as_tensor(
                start_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                features = planner_features(
                    obs_tensor,
                    str(checkpoint["input_mode"]),
                    representation,
                )
                field_output = planner(features, iterations=iterations)[-1]
            targets = {
                name: value.unsqueeze(0)
                for name, value in build_map_targets(env_for_field, device).items()
            }
            free_states = torch.nonzero(
                targets["free_mask"].reshape(-1),
                as_tuple=False,
            ).flatten()
            pair_count = min(
                args.field_pairs_per_maze,
                max(1, len(free_states) * 2),
            )
            diagnostic_rng.integers(
                0,
                len(free_states),
                size=(pair_count, 2),
                dtype=np.int64,
            )
            sample_count = min(args.field_states_per_maze, len(free_states))
            chosen_indices = diagnostic_rng.choice(
                len(free_states), size=sample_count, replace=False
            )
            sampled_states = torch.zeros_like(targets["free_mask"], dtype=torch.bool)
            sampled_states.view(-1)[
                free_states[
                    torch.as_tensor(chosen_indices, dtype=torch.long, device=device)
                ]
            ] = True
            field_all.update(
                field_output, targets, args.decision_source, sampled_states
            )
            bucket = (
                field_seen
                if int(entry["maze_size"]) <= args.seen_max_size
                else field_ood
            )
            bucket.update(field_output, targets, args.decision_source, sampled_states)
            by_size[int(entry["maze_size"])].update(
                field_output, targets, args.decision_source, sampled_states
            )

            def action_fn(
                env: Any,
                observation: np.ndarray,
                state: int,
                previous: int | None,
                _iterations: int = iterations,
                _field_output: PlannerOutput = field_output,
            ) -> tuple[int, dict[str, float]]:
                output = _field_output
                if args.recompute_every_step:
                    tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        current_features = planner_features(
                            tensor,
                            str(checkpoint["input_mode"]),
                            representation,
                        )
                        output = planner(current_features, iterations=_iterations)[-1]
                action = choose_learned_action(
                    output,
                    env,
                    state,
                    previous,
                    args.decision_source,
                    args.action_selection,
                )
                return action, {}

            rows.append(
                run_navigation(entry, action_fn=action_fn, max_steps=args.max_steps)
            )
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                sr = float(np.mean([row["success"] for row in rows]))
                print(
                    f"  K={iterations:<4d} {index + 1:>4d}/{len(entries)} SR={sr:.4f}"
                )
        results[str(iterations)] = {
            "navigation": summarize_rows(rows, args.seen_max_size),
            "field": {
                "overall": field_all.summary(),
                "seen": field_seen.summary(),
                "ood": field_ood.summary(),
                "by_size": {
                    str(size): accumulator.summary()
                    for size, accumulator in sorted(by_size.items())
                },
            },
            "task_rows": rows,
            "elapsed_seconds": float(time.time() - started),
        }
    return results


def evaluate_decoded_bfs(
    entries: list[dict[str, Any]],
    representation: SpatialRepresentation,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    map_totals: defaultdict[str, float] = defaultdict(float)
    for index, entry in enumerate(entries):

        def action_fn(
            env: Any,
            observation: np.ndarray,
            state: int,
            previous: int | None,
        ) -> tuple[int, dict[str, float]]:
            wall, predicted_state, predicted_goal, metrics = decode_map(
                representation, observation, device
            )
            candidates = (
                corrected_actions(env, state, previous)
                if args.decoded_action_selection == "corrected"
                else [int(action) for action in ACTION_IDS]
            )
            action, reachable = choose_bfs_action(
                wall, predicted_state, predicted_goal, candidates
            )
            metrics["decoded_reachable"] = float(reachable)
            return action, metrics

        row = run_navigation(entry, action_fn=action_fn, max_steps=args.max_steps)
        for name, value in row["auxiliary"].items():
            map_totals[name] += float(value)
        rows.append(row)
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"  decoded {index + 1:>4d}/{len(entries)}")
    total_steps = max(sum(row["path_length"] for row in rows), 1)
    return {
        "navigation": summarize_rows(rows, args.seen_max_size),
        "decoder": {
            "wall_iou": map_totals["wall_intersection"]
            / max(map_totals["wall_union"], 1.0),
            "agent_accuracy_per_step": map_totals["agent_correct"] / total_steps,
            "goal_accuracy_per_step": map_totals["goal_correct"] / total_steps,
            "reachable_rate_per_step": map_totals["decoded_reachable"] / total_steps,
        },
        "task_rows": rows,
    }


def evaluate_oracle(
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    iterations: int | None,
) -> dict[str, Any]:
    from diagnostics.common import bfs_distances_from

    oracle_vi = OracleValueIteration().to(device)
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        goal_cell = int(entry["goal_cell"])
        static_env = validate_manifest_entry(entry, check_bfs=False)
        goal_distances = bfs_distances_from(
            static_env._maze_mask,
            goal_cell,
            int(static_env.config.width),
        )
        precomputed_output: PlannerOutput | None = None
        if iterations is not None:
            wall = torch.as_tensor(
                static_env._maze_mask,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            goal = torch.zeros_like(wall)
            goal.view(-1)[goal_cell] = True
            with torch.no_grad():
                precomputed_output = oracle_vi(wall, goal, iterations)

        def action_fn(
            env: Any,
            observation: np.ndarray,
            state: int,
            previous: int | None,
            _goal_cell: int = goal_cell,
            _goal_distances: np.ndarray = goal_distances,
            _precomputed_output: PlannerOutput | None = precomputed_output,
        ) -> tuple[int, dict[str, float]]:
            del observation, _goal_cell
            candidates = corrected_actions(env, state, previous)
            if iterations is None:
                action = min(
                    candidates,
                    key=lambda candidate: int(
                        _goal_distances[next_state(env, state, candidate)]
                    ),
                )
                return int(action), {"oracle_reachable": 1.0}
            if _precomputed_output is None:
                raise RuntimeError("oracle VI output was not precomputed")
            action = choose_learned_action(
                _precomputed_output,
                env,
                state,
                previous,
                "value",
                "corrected",
            )
            return action, {}

        rows.append(
            run_navigation(entry, action_fn=action_fn, max_steps=args.max_steps)
        )
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            label = "bfs" if iterations is None else f"vi K={iterations}"
            print(f"  {label} {index + 1:>4d}/{len(entries)}")
    return {
        "navigation": summarize_rows(rows, args.seen_max_size),
        "task_rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    entries = select_entries(
        read_jsonl(args.manifest),
        args.max_per_size,
        args.limit,
        args.seed + 101,
    )
    iterations = parse_int_list(args.iterations)
    if not entries:
        raise ValueError("evaluation selected zero tasks")

    planner = None
    representation = None
    checkpoint: dict[str, Any] | None = None
    if args.mode == "learned":
        if not args.planner_ckpt:
            raise ValueError("learned mode requires --planner-ckpt")
        planner, representation, checkpoint = load_planner_checkpoint(
            args.planner_ckpt, device
        )
        verify_checkpoint_protocol(
            checkpoint, args.manifest, args.allow_protocol_mismatch
        )
    elif args.mode == "decoded_bfs":
        if not args.representation_ckpt:
            raise ValueError("decoded_bfs mode requires --representation-ckpt")
        representation, checkpoint = load_representation_checkpoint(
            args.representation_ckpt, device
        )
        training_args = checkpoint.get("training_args", {})
        required_decoder_losses = (
            "lambda_map_wall",
            "lambda_map_agent",
            "lambda_map_goal",
        )
        if any(
            float(training_args.get(name, 0.0) or 0.0) <= 0.0
            for name in required_decoder_losses
        ):
            raise ValueError(
                "decoded_bfs requires a checkpoint trained with wall/agent/goal losses"
            )
        representation.eval()
        for parameter in representation.parameters():
            parameter.requires_grad = False
        verify_checkpoint_protocol(
            checkpoint, args.manifest, args.allow_protocol_mismatch
        )

    if checkpoint is not None and args.training_seed is not None:
        checkpoint_seed = int(checkpoint.get("protocol", {}).get("seed", -1))
        if checkpoint_seed != args.training_seed:
            raise ValueError(
                f"checkpoint seed {checkpoint_seed} != labelled training seed "
                f"{args.training_seed}"
            )

    print("=" * 88)
    print("SPATIAL-JEPA ITERATIVE PLANNING EVALUATION")
    print("=" * 88)
    displayed_selection = (
        args.decoded_action_selection
        if args.mode == "decoded_bfs"
        else args.action_selection
    )
    print(
        f"mode={args.mode} tasks={len(entries)} max_steps={args.max_steps} "
        f"selection={displayed_selection} device={device}"
    )
    if args.mode == "learned":
        assert planner is not None and checkpoint is not None
        results = evaluate_learned(
            entries,
            planner,
            representation,
            checkpoint,
            iterations,
            args,
            device,
        )
    elif args.mode == "decoded_bfs":
        assert representation is not None
        results = evaluate_decoded_bfs(entries, representation, args, device)
    elif args.mode == "oracle_bfs":
        results = evaluate_oracle(entries, args, device, iterations=None)
    else:
        results = {
            str(count): evaluate_oracle(entries, args, device, iterations=count)
            for count in iterations
        }

    payload = {
        "metadata": {
            **protocol_metadata(
                train_manifest=args.train_manifest,
                eval_manifest=args.manifest,
                seed=args.seed,
                max_steps=args.max_steps,
            ),
            "mode": args.mode,
            "planner_ckpt": args.planner_ckpt,
            "representation_ckpt": args.representation_ckpt,
            "decision_source": args.decision_source,
            "action_selection": args.action_selection,
            "task_count": len(entries),
            "training_seed": args.training_seed,
            "max_per_size": args.max_per_size,
            "limit": args.limit,
            "field_states_per_maze": args.field_states_per_maze,
            "field_pairs_per_maze": args.field_pairs_per_maze,
            "recompute_every_step": args.recompute_every_step,
            "decoded_action_selection": args.decoded_action_selection,
            "device": str(device),
            "comparable_to_full900": bool(
                len(entries) == 900
                and args.max_steps == 128
                and args.action_selection == "corrected"
            ),
        },
        "results": results,
    }
    strict_json_dump(args.output, payload)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
