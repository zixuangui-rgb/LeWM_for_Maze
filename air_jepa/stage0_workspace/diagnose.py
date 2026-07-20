#!/usr/bin/env python3
"""Full-900 local future/energy causal diagnostics for AIR0-jepa."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from collections import Counter
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace.checkpoints import (
    load_air_checkpoint,
    load_frozen_representation,
    verify_source_lock,
)
from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_json_dump,
    code_fingerprint,
    format_template,
    git_commit,
    git_worktree_dirty,
    load_config,
    prepare_new_output,
    relative_path,
    require_clean_worktree,
    require_h800_device,
    resolve_device,
    resolve_path,
    runtime_metadata,
    set_seed,
    sha256_file,
)
from air_jepa.stage0_workspace.evaluate import FUTURE_PERMUTATION
from air_jepa.stage0_workspace.losses import future_prediction_loss
from air_jepa.stage0_workspace.models import require_finite_output
from air_jepa.stage0_workspace.protocol import (
    verify_package_lock,
    verify_protocol_lock,
)
from diagnostics.common import (
    ACTION_IDS,
    bfs_distances_from,
    next_state,
    observe_state,
)
from spatial_jepa_planning.common import read_jsonl, validate_manifest_entry

DISTANCE_ECE_BINS = 15
DISTANCE_MAX = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split-role", choices=("air_early", "air_dev"), default="air_dev"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def deterministic_states(
    entry: dict[str, Any],
    *,
    count: int,
) -> list[int]:
    env = validate_manifest_entry(entry, check_bfs=False)
    goal = int(env._goal_position)
    distances = bfs_distances_from(env._maze_mask, goal, int(env.config.width))
    candidates: list[int] = []
    for state in np.flatnonzero((~env._maze_mask).reshape(-1)).tolist():
        if state == goal or int(distances[state]) <= 0:
            continue
        successors = {next_state(env, int(state), action) for action in ACTION_IDS}
        if len(successors) >= 2:
            candidates.append(int(state))
    digest = hashlib.sha256(
        f"air-local-v1:{entry['task_hash']}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    rng.shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def _pairwise_distance(fields: torch.Tensor) -> torch.Tensor:
    values = []
    for left, right in itertools.combinations(range(4), 2):
        values.append((fields[:, left] - fields[:, right]).square().mean(dim=(1, 2, 3)))
    return torch.stack(values, dim=1).mean(dim=1)


def _ranking_rows(
    *,
    energy: torch.Tensor,
    candidate_distances: torch.Tensor,
    optimal: torch.Tensor,
) -> list[dict[str, Any]]:
    chosen = energy.argmin(dim=1)
    positive = torch.finfo(energy.dtype).max
    best_optimal = energy.masked_fill(~optimal, positive).min(dim=1).values
    best_bad = energy.masked_fill(optimal, positive).min(dim=1).values
    rows = []
    for index in range(energy.shape[0]):
        slot = int(chosen[index])
        rows.append(
            {
                "top1": bool(optimal[index, slot]),
                "regret": int(
                    candidate_distances[index, slot] - candidate_distances[index].min()
                ),
                "margin": (
                    float(best_bad[index] - best_optimal[index])
                    if bool((~optimal[index]).any())
                    else None
                ),
                "chosen_slot": slot,
                "energy": energy[index].detach().cpu().tolist(),
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("distance Spearman inputs must be aligned nontrivial vectors")
    left = _average_ranks(first)
    right = _average_ranks(second)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _top_class_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int,
) -> float:
    if confidence.shape != correct.shape or confidence.ndim != 1 or not len(confidence):
        raise ValueError("distance ECE inputs must be aligned non-empty vectors")
    if bins <= 0 or np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("distance ECE confidence/bins are invalid")
    bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    total = float(len(confidence))
    error = 0.0
    for index in range(bins):
        selected = bin_index == index
        if not bool(selected.any()):
            continue
        error += (
            float(selected.sum())
            / total
            * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
        )
    return error


def _distance_metrics(
    rows: list[dict[str, Any]],
    *,
    ranking_name: str,
    class_name: str,
    confidence_name: str,
) -> dict[str, Any]:
    expected = np.asarray(
        [float(value) for row in rows for value in row[ranking_name]["energy"]],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            min(int(value), DISTANCE_MAX)
            for row in rows
            for value in row["candidate_distances"]
        ],
        dtype=np.int64,
    )
    classes = np.asarray(
        [int(value) for row in rows for value in row[class_name]],
        dtype=np.int64,
    )
    confidence = np.asarray(
        [float(value) for row in rows for value in row[confidence_name]],
        dtype=np.float64,
    )
    if not (
        expected.shape == targets.shape == classes.shape == confidence.shape
        and expected.ndim == 1
        and len(expected)
    ):
        raise ValueError("distance diagnostic vectors are not aligned")
    error = expected - targets.astype(np.float64)
    correct = classes == targets
    return {
        "n_action_predictions": int(len(expected)),
        "expected_mae": float(np.mean(np.abs(error))),
        "expected_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "expected_spearman": _spearman(expected, targets.astype(np.float64)),
        "categorical_accuracy": float(correct.mean()),
        "top_class_ece_15": _top_class_ece(
            confidence,
            correct.astype(np.float64),
            bins=DISTANCE_ECE_BINS,
        ),
    }


def _cost_classification(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or logits.shape[1:] != (4, DISTANCE_MAX + 1):
        raise ValueError(
            f"distance logits must have shape [B,4,{DISTANCE_MAX + 1}]"
        )
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, category = probabilities.max(dim=-1)
    return category, confidence


def summarize_state_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("diagnostic summary requires at least one state row")

    def ranking(name: str) -> dict[str, Any]:
        values = [row[name] for row in rows]
        margins = [
            float(value["margin"]) for value in values if value["margin"] is not None
        ]
        return {
            "local_top1": _mean([float(value["top1"]) for value in values]),
            "regret": _mean([float(value["regret"]) for value in values]),
            "margin": _mean(margins),
        }

    return {
        "n_states": len(rows),
        "predicted": ranking("predicted_ranking"),
        "true_future": ranking("true_ranking"),
        "copy_current": ranking("copy_ranking"),
        "permuted": ranking("permuted_ranking"),
        "zero": ranking("zero_ranking"),
        "predicted_true_choice_agreement": _mean(
            [
                float(
                    row["predicted_ranking"]["chosen_slot"]
                    == row["true_ranking"]["chosen_slot"]
                )
                for row in rows
            ]
        ),
        "prediction_flip_rate": _mean(
            [
                float(
                    row["true_ranking"]["top1"] and not row["predicted_ranking"]["top1"]
                )
                for row in rows
            ]
        ),
        "energy_wrong_with_true_future_rate": _mean(
            [float(not row["true_ranking"]["top1"]) for row in rows]
        ),
        "future": {
            "normalized_field_error": _mean(
                [float(row["normalized_field_error"]) for row in rows]
            ),
            "normalized_delta_error": _mean(
                [float(row["normalized_delta_error"]) for row in rows]
            ),
            "copy_delta_normalized": _mean(
                [float(row["copy_delta_normalized"]) for row in rows]
            ),
            "predicted_candidate_pairwise": _mean(
                [float(row["predicted_candidate_pairwise"]) for row in rows]
            ),
            "target_candidate_pairwise": _mean(
                [float(row["target_candidate_pairwise"]) for row in rows]
            ),
            "predicted_variance": _mean(
                [float(row["predicted_variance"]) for row in rows]
            ),
            "target_variance": _mean([float(row["target_variance"]) for row in rows]),
        },
        "distance": {
            "max_distance": DISTANCE_MAX,
            "ece_bins": DISTANCE_ECE_BINS,
            "target_clipped_rate": float(
                np.mean(
                    [
                        float(int(value) > DISTANCE_MAX)
                        for row in rows
                        for value in row["candidate_distances"]
                    ]
                )
            ),
            "predicted": _distance_metrics(
                rows,
                ranking_name="predicted_ranking",
                class_name="predicted_cost_class",
                confidence_name="predicted_cost_confidence",
            ),
            "true_future": _distance_metrics(
                rows,
                ranking_name="true_ranking",
                class_name="true_future_cost_class",
                confidence_name="true_future_cost_confidence",
            ),
        },
        "error_taxonomy": dict(Counter(str(row["local_error_type"]) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed not in config.seeds:
        raise ValueError("diagnostic seed is outside the locked system seeds")
    if args.split_role == "air_early" and args.seed != 42:
        raise ValueError("the locked early diagnostic is restricted to seed 42")
    if args.allow_dirty_worktree:
        raise ValueError("formal full-900 diagnostics reject dirty-worktree overrides")
    if args.output:
        output = resolve_path(args.output)
    elif args.split_role == "air_dev":
        output = format_template(
            config.paths.diagnostic_template,
            method="air0_jepa",
            seed=args.seed,
        )
    else:
        output = resolve_path(config.paths.run_root) / (
            f"diagnostics/air_early/air0_jepa/seed{args.seed}.json"
        )
    prepare_new_output(output)
    require_clean_worktree(allow_dirty=False)
    protocol_lock = verify_protocol_lock(config)
    package_lock = verify_package_lock(config)
    source_lock = verify_source_lock(config)
    device = resolve_device(args.device)
    require_h800_device(device)
    set_seed(args.seed + 303, deterministic=True)
    checkpoint_path = format_template(
        config.paths.air_checkpoint_template,
        method="air0_jepa",
        seed=args.seed,
    )
    model, checkpoint = load_air_checkpoint(
        checkpoint_path,
        config=config,
        method="air0_jepa",
        seed=args.seed,
        device=device,
        require_formal=True,
    )
    for key, expected in (
        ("protocol_sha256", protocol_lock["protocol_sha256"]),
        ("package_sha256", package_lock["package_sha256"]),
        ("source_lock_sha256", source_lock["source_lock_sha256"]),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint {key} differs from current lock")
    representation, _ = load_frozen_representation(
        config,
        seed=args.seed,
        device=device,
        source_lock=source_lock,
    )
    target_mean = torch.as_tensor(
        checkpoint["future_target_channel_moments"]["mean"],
        dtype=torch.float32,
        device=device,
    )
    manifest = resolve_path(
        config.paths.air_early_manifest
        if args.split_role == "air_early"
        else config.paths.air_dev_manifest
    )
    entries = read_jsonl(manifest)
    state_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    permutation_index = torch.as_tensor(
        FUTURE_PERMUTATION, dtype=torch.long, device=device
    )
    started = time.perf_counter()
    for task_index, entry in enumerate(entries):
        env = validate_manifest_entry(entry, check_bfs=False)
        states = deterministic_states(
            entry, count=config.evaluation.local_states_per_maze
        )
        if not states:
            raise ValueError(
                f"no eligible local diagnostic states: {entry['task_hash']}"
            )
        goal_distances = bfs_distances_from(
            env._maze_mask, int(env._goal_position), int(env.config.width)
        )
        observations = np.stack([observe_state(env, state) for state in states])
        next_states = np.asarray(
            [
                [next_state(env, state, action) for action in ACTION_IDS]
                for state in states
            ],
            dtype=np.int64,
        )
        successor_observations = np.stack(
            [
                np.stack([observe_state(env, int(value)) for value in row])
                for row in next_states
            ]
        )
        candidate_distances = torch.as_tensor(
            goal_distances[next_states], dtype=torch.long, device=device
        )
        optimal = (
            candidate_distances == candidate_distances.min(dim=1, keepdim=True).values
        )
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        successor_tensor = torch.as_tensor(
            successor_observations, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            source = representation.planning_latent(obs_tensor)
            batch, actions, height, width, channels = successor_tensor.shape
            true_future = representation.planning_latent(
                successor_tensor.reshape(batch * actions, height, width, channels)
            ).reshape(batch, actions, -1, height, width)
            mask = torch.ones((batch, height, width), dtype=torch.bool, device=device)
            output = model(
                source,
                iterations=config.evaluation.primary_k,
                valid_mask=mask,
            )[-1]
            require_finite_output(output)
            predicted = output.predicted_future
            if predicted is None:
                raise RuntimeError("AIR diagnostic requires predicted futures")
            copy = source[:, None].expand_as(predicted)
            zero = target_mean[None, None, :, None, None].expand_as(predicted)
            true_logits, true_energy = model.score_external_futures(
                output, true_future, mask
            )
            _, copy_energy = model.score_external_futures(output, copy, mask)
            _, permuted_energy = model.score_external_futures(
                output, predicted.index_select(1, permutation_index), mask
            )
            _, zero_energy = model.score_external_futures(output, zero, mask)
            predicted_cost_class, predicted_cost_confidence = _cost_classification(
                output.cost_logits
            )
            true_cost_class, true_cost_confidence = _cost_classification(true_logits)
            state_future_losses = [
                future_prediction_loss(
                    predicted[state_index : state_index + 1],
                    true_future[state_index : state_index + 1],
                    source[state_index : state_index + 1],
                    valid_mask=mask[state_index : state_index + 1],
                    epsilon=config.training.target_variance_epsilon,
                )
                for state_index in range(batch)
            ]
        rankings = {
            "predicted_ranking": _ranking_rows(
                energy=output.energy,
                candidate_distances=candidate_distances,
                optimal=optimal,
            ),
            "true_ranking": _ranking_rows(
                energy=true_energy,
                candidate_distances=candidate_distances,
                optimal=optimal,
            ),
            "copy_ranking": _ranking_rows(
                energy=copy_energy,
                candidate_distances=candidate_distances,
                optimal=optimal,
            ),
            "permuted_ranking": _ranking_rows(
                energy=permuted_energy,
                candidate_distances=candidate_distances,
                optimal=optimal,
            ),
            "zero_ranking": _ranking_rows(
                energy=zero_energy,
                candidate_distances=candidate_distances,
                optimal=optimal,
            ),
        }
        predicted_pairwise = _pairwise_distance(predicted)
        target_pairwise = _pairwise_distance(true_future)
        for index, state in enumerate(states):
            predicted_correct = bool(rankings["predicted_ranking"][index]["top1"])
            true_correct = bool(rankings["true_ranking"][index]["top1"])
            error_type = (
                "correct"
                if predicted_correct
                else (
                    "prediction_flip"
                    if true_correct
                    else "energy_wrong_with_true_future"
                )
            )
            state_rows.append(
                {
                    "task_id": str(entry["task_hash"]),
                    "maze_size": int(entry["maze_size"]),
                    "state": int(state),
                    "current_distance": int(goal_distances[state]),
                    "candidate_distances": candidate_distances[index].cpu().tolist(),
                    "optimal_action_mask": optimal[index].cpu().tolist(),
                    **{name: values[index] for name, values in rankings.items()},
                    "predicted_cost_class": predicted_cost_class[index].cpu().tolist(),
                    "predicted_cost_confidence": predicted_cost_confidence[index]
                    .cpu()
                    .tolist(),
                    "true_future_cost_class": true_cost_class[index].cpu().tolist(),
                    "true_future_cost_confidence": true_cost_confidence[index]
                    .cpu()
                    .tolist(),
                    "normalized_field_error": float(
                        state_future_losses[index].normalized_field.detach().cpu()
                    ),
                    "normalized_delta_error": float(
                        state_future_losses[index].normalized_delta.detach().cpu()
                    ),
                    "copy_delta_normalized": float(
                        state_future_losses[index].copy_delta_normalized.detach().cpu()
                    ),
                    "predicted_candidate_pairwise": float(
                        predicted_pairwise[index].detach().cpu()
                    ),
                    "target_candidate_pairwise": float(
                        target_pairwise[index].detach().cpu()
                    ),
                    "predicted_variance": float(
                        predicted[index].float().var(unbiased=False).detach().cpu()
                    ),
                    "target_variance": float(
                        true_future[index].float().var(unbiased=False).detach().cpu()
                    ),
                    "local_error_type": error_type,
                }
            )
        task_rows.append(
            {
                "task_id": str(entry["task_hash"]),
                "maze_size": int(entry["maze_size"]),
                "sampled_states": len(states),
            }
        )
        if args.progress_every > 0 and (task_index + 1) % args.progress_every == 0:
            print(f"diagnostic seed={args.seed} {task_index + 1}/{len(entries)}")

    payload = {
        "schema": "air-jepa-stage0-local-diagnostic-v1",
        "metadata": {
            "experiment_id": config.experiment_id,
            "method": "air0_jepa",
            "seed": args.seed,
            "k": config.evaluation.primary_k,
            "split_role": args.split_role,
            "evidence_role": "MECHANISM_DIAGNOSTIC",
            "states_per_maze": config.evaluation.local_states_per_maze,
            "distance_max": DISTANCE_MAX,
            "distance_ece_bins": DISTANCE_ECE_BINS,
            "distance_calibration": (
                "top-class exact-bin ECE over 15 equal-width confidence bins"
            ),
            "manifest": relative_path(manifest),
            "manifest_sha256": sha256_file(manifest),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "protocol_sha256": protocol_lock["protocol_sha256"],
            "package_sha256": package_lock["package_sha256"],
            "source_lock_sha256": source_lock["source_lock_sha256"],
            "future_permutation": list(FUTURE_PERMUTATION),
            "git_commit": git_commit(),
            "git_dirty": git_worktree_dirty(),
            "code_fingerprint": code_fingerprint(),
            "runtime": runtime_metadata(device),
            "elapsed_seconds": time.perf_counter() - started,
            "formal": True,
        },
        "summary": summarize_state_rows(state_rows),
        "task_rows": task_rows,
        "state_rows": state_rows,
    }
    atomic_json_dump(output, payload)
    print(f"saved={relative_path(output)} states={len(state_rows)}")


if __name__ == "__main__":
    main()
