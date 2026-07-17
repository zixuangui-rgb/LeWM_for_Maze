"""Distance, local-order, trajectory-order, and rollout-drift diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distance_head_study.candidates import candidate_bank_path, load_candidate_bank
from distance_head_study.common import (
    atomic_json_dump,
    canonical_json_sha256,
    load_study_config,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    sha256_file,
)
from distance_head_study.data import (
    ShardedGoalDataset,
    cache_index_path,
    encode_observations,
    evenly_spaced_indices,
    refresh_joint_latents,
    sample_training_batch,
    true_candidate_distances,
    validate_cache_binding,
)
from distance_head_study.evaluate import _head_output, _load_models
from distance_head_study.gates import require_evaluation_gate, require_seed_released
from distance_head_study.losses import (
    reachability_logits_by_budget,
    score_in_raw_steps,
)
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.schemas import SamplerKind
from distance_head_study.train_head import _predict_all_actions
from vector_jepa_planner_frontier.schemas import RolloutSemantics
from vector_jepa_planner_frontier.world_model import VectorContext

DIAGNOSTIC_SCHEMA = "distance-head-diagnostics-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--split-role", choices=("cal", "screen", "select", "stress"), required=True
    )
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--diagnostic-batches", type=int, default=0)
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": 0.0, "std": 0.0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _binary_auc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    positive = labels.astype(bool)
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _average_ranks(probabilities)
    numerator = (
        float(ranks[positive].sum()) - positive_count * (positive_count - 1) / 2.0
    )
    return numerator / (positive_count * negative_count)


def _expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> float:
    total = max(len(probabilities), 1)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not bool(mask.any()):
            continue
        error += (
            float(mask.sum())
            / total
            * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
        )
    return error


def _candidate_endpoint_indices(
    dataset: ShardedGoalDataset,
    batch: Any,
    sequences: torch.Tensor,
    context_index: int,
    executed_action_count: int,
) -> np.ndarray:
    shard = dataset.get(int(batch.topology_positions[context_index]))
    transitions = shard["next_indices"].numpy()
    endpoints = []
    for sequence in sequences.tolist():
        state = int(batch.source_indices[context_index])
        for action in sequence[:executed_action_count]:
            state = int(transitions[state, int(action)])
        endpoints.append(state)
    return np.asarray(endpoints, dtype=np.int64)


def _candidate_drift_steps(
    predicted_terminal: torch.Tensor,
    latent_bank: torch.Tensor,
    endpoints: np.ndarray,
    all_pairs_bfs: torch.Tensor,
) -> list[float]:
    """Measure every candidate terminal, without candidate-index subsampling."""

    if predicted_terminal.ndim != 2 or latent_bank.ndim != 2:
        raise ValueError("drift latents must have shape [candidate/state, latent]")
    if predicted_terminal.shape[1] != latent_bank.shape[1]:
        raise ValueError("drift latent dimensions differ")
    if len(endpoints) != predicted_terminal.shape[0]:
        raise ValueError("drift endpoint count differs from candidate count")
    nearest = torch.cdist(predicted_terminal, latent_bank).argmin(dim=1).cpu().tolist()
    return [
        float(all_pairs_bfs[int(endpoint), int(nearest_index)])
        for endpoint, nearest_index in zip(endpoints.tolist(), nearest, strict=True)
    ]


def _require_diagnostic_gate(
    config: Any,
    *,
    split_role: str,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> dict[str, Any]:
    """Apply the same seed and sealed-split gates as formal evaluation."""

    gate = require_evaluation_gate(
        config,
        split_role=split_role,
        method=method,
        backbone_seed=backbone_seed,
        head_seed=head_seed,
    )
    if gate is not None:
        return gate
    return require_seed_released(
        config,
        backbone_seed=backbone_seed,
        head_seed=0 if method == "b_l2_cem" else head_seed,
    )


def main() -> None:
    args = parse_args()
    if args.diagnostic_batches < 0:
        raise ValueError("diagnostic batches must be non-negative")
    diagnostic_override = args.diagnostic_batches > 0
    if diagnostic_override and not args.allow_dirty_worktree:
        raise ValueError("diagnostic batch overrides require --allow-dirty-worktree")
    config = load_study_config(args.config)
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    lock = verify_protocol_lock(config)
    method, method_hash, decision_hashes = load_and_resolve_method(
        config.paths.method_catalog,
        args.method,
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    _require_diagnostic_gate(
        config,
        split_role=args.split_role,
        method=method.name,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
    )
    device = resolve_device(args.device or config.device)
    world_model, head, checkpoint = _load_models(
        config,
        method,
        backbone_seed=args.backbone_seed,
        head_seed=args.head_seed,
        device=device,
        expected_analysis_spec_sha256=lock["analysis_spec_sha256"],
        expected_protocol_lock_sha256=lock["protocol_lock_sha256"],
    )
    dataset = ShardedGoalDataset(
        cache_index_path(
            config,
            split_role=args.split_role,
            backbone_seed=args.backbone_seed,
        )
    )
    cache_binding = validate_cache_binding(
        dataset,
        config,
        split_role=args.split_role,
        backbone_seed=args.backbone_seed,
        protocol_lock=lock,
    )
    bank_path = candidate_bank_path(
        config, split_role="train", backbone_seed=args.backbone_seed
    )
    bank_metadata, candidate_bank = load_candidate_bank(bank_path)
    if bank_metadata.get("protocol_id") != config.protocol_id:
        raise ValueError("candidate bank protocol mismatch")
    if bank_metadata.get("analysis_spec_sha256") != lock["analysis_spec_sha256"]:
        raise ValueError("candidate bank analysis lock mismatch")
    if bank_metadata.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]:
        raise ValueError("candidate bank protocol-lock mismatch")
    if int(bank_metadata.get("backbone_seed", -1)) != args.backbone_seed:
        raise ValueError("candidate bank backbone mismatch")
    expected_bank_shape = (
        config.training.candidate_sets_per_backbone,
        config.training.trajectory_candidates,
        config.planner.horizon,
    )
    if tuple(candidate_bank.shape) != expected_bank_shape:
        raise ValueError("candidate bank shape differs from diagnostic protocol")
    batch_count = int(args.diagnostic_batches or config.analysis.diagnostic_batches)
    absolute_prediction: list[float] = []
    absolute_truth: list[float] = []
    true_correct: list[float] = []
    predicted_correct: list[float] = []
    true_regret: list[float] = []
    predicted_regret: list[float] = []
    true_margin: list[float] = []
    predicted_margin: list[float] = []
    trajectory_spearman: list[float] = []
    true_dynamics_spearman: list[float] = []
    trajectory_regret: list[float] = []
    true_dynamics_regret: list[float] = []
    drift: list[float] = []
    reachability_probabilities: list[np.ndarray] = []
    reachability_labels: list[np.ndarray] = []
    by_size: dict[int, dict[str, list[float]]] = {}
    current_shard_latents: dict[int, torch.Tensor] = {}
    joint_model = bool(checkpoint.get("joint_model_state_loaded", False))

    for step in range(batch_count):
        cpu_batch = sample_training_batch(
            dataset,
            sampler=SamplerKind.UNIFORM,
            effective_batch_size=config.training.effective_batch_size,
            pairs_per_topology=config.training.pairs_per_topology,
            schedule_seed=config.seeds.bootstrap_seed,
            backbone_seed=args.backbone_seed,
            step=step,
        )
        batch = (
            refresh_joint_latents(
                dataset,
                cpu_batch,
                world_model.model,
                device=device,
                gradients=False,
            ).to(device)
            if joint_model
            else cpu_batch.to(device)
        )
        size_metrics = by_size.setdefault(
            batch.maze_size,
            {
                "absolute_error": [],
                "true_top1": [],
                "predicted_top1": [],
                "true_regret": [],
                "predicted_regret": [],
            },
        )
        predicted_next = _predict_all_actions(world_model.model, batch, gradients=False)
        batch_size, actions, latent_dim = batch.next_latents.shape
        goal_actions = (
            batch.goal[:, None].expand(-1, actions, -1).reshape(-1, latent_dim)
        )
        if head is None:
            current_scores = None
            true_scores = (
                (batch.next_latents - batch.goal[:, None]).square().sum(dim=-1)
            )
            predicted_scores = (
                (predicted_next - batch.goal[:, None]).square().sum(dim=-1)
            )
        else:
            current_output = _head_output(
                head,
                batch.source,
                batch.goal,
                horizon=config.planner.horizon,
                predicted_domain=False,
            )
            current_scores = score_in_raw_steps(
                current_output,
                head,
                max_distance=batch.max_distance,
            )
            max_actions = batch.max_distance[:, None].expand(-1, actions).reshape(-1)
            true_output = _head_output(
                head,
                batch.next_latents.reshape(-1, latent_dim),
                goal_actions,
                horizon=1,
                predicted_domain=False,
            )
            predicted_output = _head_output(
                head,
                predicted_next.reshape(-1, latent_dim),
                goal_actions,
                horizon=1,
                predicted_domain=True,
            )
            true_scores = score_in_raw_steps(
                true_output, head, max_distance=max_actions
            ).reshape(batch_size, actions)
            predicted_scores = score_in_raw_steps(
                predicted_output, head, max_distance=max_actions
            ).reshape(batch_size, actions)
            absolute_prediction.extend(current_scores.detach().cpu().tolist())
            absolute_truth.extend(batch.raw_distance.detach().cpu().tolist())
            size_metrics["absolute_error"].extend(
                (current_scores - batch.raw_distance).abs().detach().cpu().tolist()
            )
            if current_output.reachability_logits is not None:
                logits = reachability_logits_by_budget(head, batch.source, batch.goal)
                budgets = torch.tensor(
                    head.spec.reachability_budgets,
                    dtype=batch.raw_distance.dtype,
                    device=device,
                )
                reachability_probabilities.append(
                    torch.sigmoid(logits).detach().cpu().numpy()
                )
                reachability_labels.append(
                    (batch.raw_distance[:, None] <= budgets[None, :])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

        for label, scores, correct_rows, regret_rows, margin_rows in (
            ("true", true_scores, true_correct, true_regret, true_margin),
            (
                "predicted",
                predicted_scores,
                predicted_correct,
                predicted_regret,
                predicted_margin,
            ),
        ):
            masked = scores.masked_fill(~batch.valid_actions, float("inf"))
            selected = masked.argmin(dim=1)
            row = torch.arange(batch_size, device=device)
            correct_values = batch.optimal_actions[row, selected].float().cpu().tolist()
            correct_rows.extend(correct_values)
            best_true = (
                batch.next_distances.masked_fill(~batch.valid_actions, float("inf"))
                .min(dim=1)
                .values
            )
            regret_values = (
                (batch.next_distances[row, selected] - best_true).cpu().tolist()
            )
            regret_rows.extend(regret_values)
            size_metrics[f"{label}_top1"].extend(correct_values)
            size_metrics[f"{label}_regret"].extend(regret_values)
            best_score = (
                scores.masked_fill(~batch.optimal_actions, float("inf"))
                .min(dim=1)
                .values
            )
            worse_mask = batch.valid_actions & ~batch.optimal_actions
            worse_score = (
                scores.masked_fill(~worse_mask, float("inf")).min(dim=1).values
            )
            finite = torch.isfinite(worse_score)
            margin_rows.extend(
                (worse_score[finite] - best_score[finite]).cpu().tolist()
            )

        if step < config.analysis.trajectory_diagnostic_batches:
            contexts = min(
                config.analysis.trajectory_diagnostic_contexts,
                batch.source.shape[0],
            )
            context_indices = evenly_spaced_indices(batch.source.shape[0], contexts)
            sequences = candidate_bank[step, :, : config.planner.horizon]
            true_distances = true_candidate_distances(
                dataset,
                cpu_batch,
                sequences[None].expand(contexts, -1, -1),
                context_indices=context_indices,
                executed_action_count=max(config.planner.horizon - 1, 0),
            ).numpy()
            for output_row, batch_index in enumerate(context_indices.tolist()):
                context = VectorContext(
                    embeddings=batch.history_latents[batch_index : batch_index + 1],
                    actions=batch.history_actions[batch_index : batch_index + 1],
                    goal=batch.goal[batch_index : batch_index + 1, None],
                    maze_size=batch.maze_size,
                    remaining_steps=128,
                )
                rollout = world_model.rollout(
                    context,
                    sequences,
                    semantics=RolloutSemantics.LEGACY_WARMUP_V1,
                )
                goal = batch.goal[batch_index : batch_index + 1].expand(
                    len(sequences), -1
                )
                if head is None:
                    predicted_cost = (rollout.terminal - goal).square().sum(dim=-1)
                else:
                    predicted_cost = _head_output(
                        head,
                        rollout.terminal,
                        goal,
                        horizon=config.planner.horizon,
                    ).score
                predicted_array = predicted_cost.detach().cpu().numpy()
                truth = true_distances[output_row]
                trajectory_spearman.append(_spearman(predicted_array, truth))
                chosen = int(np.argmin(predicted_array))
                trajectory_regret.append(float(truth[chosen] - truth.min()))
                endpoints = _candidate_endpoint_indices(
                    dataset,
                    cpu_batch,
                    sequences,
                    batch_index,
                    max(config.planner.horizon - 1, 0),
                )
                shard = dataset.get(int(cpu_batch.topology_positions[batch_index]))
                position = int(cpu_batch.topology_positions[batch_index])
                if position not in current_shard_latents:
                    current_shard_latents[position] = (
                        encode_observations(
                            world_model.model,
                            shard["observations"],
                            maze_size=batch.maze_size,
                            device=device,
                            gradients=False,
                        )
                        if joint_model
                        else shard["latents"].to(device)
                    )
                latent_bank = current_shard_latents[position]
                endpoint_latents = latent_bank.index_select(
                    0, torch.from_numpy(endpoints).to(device)
                )
                if head is None:
                    true_cost = (endpoint_latents - goal).square().sum(dim=-1)
                else:
                    true_cost = _head_output(
                        head,
                        endpoint_latents,
                        goal,
                        horizon=config.planner.horizon,
                        predicted_domain=False,
                    ).score
                true_array = true_cost.detach().cpu().numpy()
                true_dynamics_spearman.append(_spearman(true_array, truth))
                true_choice = int(np.argmin(true_array))
                true_dynamics_regret.append(float(truth[true_choice] - truth.min()))
                drift.extend(
                    _candidate_drift_steps(
                        rollout.terminal,
                        latent_bank,
                        endpoints,
                        shard["all_pairs_bfs"],
                    )
                )

    if head is not None:
        predictions = np.asarray(absolute_prediction)
        truth = np.asarray(absolute_truth)
        residual = predictions - truth
        absolute = {
            "n": int(len(truth)),
            "mae_steps": float(np.abs(residual).mean()),
            "rmse_steps": float(np.sqrt(np.square(residual).mean())),
            "bias_steps": float(residual.mean()),
            "spearman": _spearman(predictions, truth),
        }
        if predictions.std() > 0:
            slope, intercept = np.polyfit(predictions, truth, deg=1)
            absolute.update(
                calibration_slope=float(slope), calibration_intercept=float(intercept)
            )
    else:
        absolute = {
            "available": False,
            "reason": "latent-L2 is not a BFS-distance estimator",
        }
    if reachability_probabilities:
        probabilities = np.concatenate(reachability_probabilities, axis=0)
        labels = np.concatenate(reachability_labels, axis=0)
        per_budget = {}
        auc_values = []
        assert head is not None
        for index, budget in enumerate(head.spec.reachability_budgets):
            auc = _binary_auc(probabilities[:, index], labels[:, index])
            if auc is not None:
                auc_values.append(auc)
            per_budget[str(budget)] = {
                "brier": float(
                    np.square(probabilities[:, index] - labels[:, index]).mean()
                ),
                "ece10": _expected_calibration_error(
                    probabilities[:, index], labels[:, index]
                ),
                "auroc": auc,
            }
        reachability = {
            "available": True,
            "macro_brier": float(np.square(probabilities - labels).mean()),
            "macro_ece10": float(
                np.mean(
                    [
                        _expected_calibration_error(
                            probabilities[:, index], labels[:, index]
                        )
                        for index in range(probabilities.shape[1])
                    ]
                )
            ),
            "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
            "monotonic_violation_rate": float(
                (probabilities[:, :-1] > probabilities[:, 1:]).mean()
            ),
            "per_budget": per_budget,
        }
    else:
        reachability = {"available": False}
    output = {
        "schema": DIAGNOSTIC_SCHEMA,
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "method": method.name,
        "method_sha256": method_hash,
        "decision_sha256s": list(decision_hashes),
        "split_role": args.split_role,
        "backbone_seed": args.backbone_seed,
        "head_seed": args.head_seed,
        "sample_count": batch_count * config.training.effective_batch_size,
        "checkpoint": checkpoint,
        "cache_binding": cache_binding,
        "candidate_bank": {
            "path": bank_path.as_posix(),
            "sha256": sha256_file(bank_path),
            "metadata": bank_metadata,
        },
        "absolute_distance": absolute,
        "reachability": reachability,
        "true_latent_local": {
            "top1": _summary(true_correct),
            "regret_steps": _summary(true_regret),
            "score_margin": _summary(true_margin),
            "score_margin_unit": "latent_squared_l2" if head is None else "bfs_steps",
        },
        "predicted_latent_local": {
            "top1": _summary(predicted_correct),
            "regret_steps": _summary(predicted_regret),
            "score_margin": _summary(predicted_margin),
            "score_margin_unit": "latent_squared_l2" if head is None else "bfs_steps",
        },
        "candidate_order": {
            "predicted_dynamics_spearman": _summary(trajectory_spearman),
            "predicted_dynamics_regret_steps": _summary(trajectory_regret),
            "true_dynamics_spearman": _summary(true_dynamics_spearman),
            "true_dynamics_regret_steps": _summary(true_dynamics_regret),
        },
        "closed_loop_drift_bfs_steps": _summary(drift),
        "by_size": {
            str(size): {name: _summary(values) for name, values in metrics.items()}
            for size, metrics in sorted(by_size.items())
        },
        "interpretation_boundary": (
            "diagnostic BFS labels never enter test-time action selection"
        ),
    }
    output["diagnostic_sha256"] = canonical_json_sha256(output)
    path = resolve_path(
        (
            "distance_head_study_runs/smoke/diagnostics/"
            f"{args.split_role}/{method.name}/backbone{args.backbone_seed}_"
            f"head{args.head_seed}_batches{batch_count}.json"
        )
        if diagnostic_override
        else (
            f"distance_head_study_runs/diagnostics/{args.split_role}/{method.name}/"
            f"backbone{args.backbone_seed}_head{args.head_seed}.json"
        )
    )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {path}")
    atomic_json_dump(path, output)
    print(Path(path))


if __name__ == "__main__":
    main()
