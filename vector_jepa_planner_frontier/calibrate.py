"""Calibrate trained planner heads on validation topologies without gradient updates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from final_closure.common import read_jsonl, sha256_file
from hdwm.losses import SIGReg
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_torch_save,
    hierarchical_seed,
    load_json,
    load_study_config,
    method_by_name,
    prepare_formal_output,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    training_spec_sha256,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.data import (
    JEPATrajectorySampler,
    PlannerBatchSampler,
    encode_planner_batch,
)
from vector_jepa_planner_frontier.effective_methods import resolve_effective_method
from vector_jepa_planner_frontier.heads import (
    ActionConsistencyVerifier,
    AutoregressiveProposal,
    CounterexampleRanker,
    DiscreteDenoisingProposal,
    DistributionalReachability,
    HeadConfig,
    StateJoinHead,
    VectorDTSHead,
    required_head_names,
)
from vector_jepa_planner_frontier.proposals import RetrievalBank
from vector_jepa_planner_frontier.train import (
    original_jepa_losses,
    predicted_successor_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def load_heads(
    checkpoint: dict[str, Any], device: torch.device
) -> dict[str, torch.nn.Module]:
    head_config = HeadConfig.from_dict(checkpoint["head_config"])
    constructors: dict[str, type[torch.nn.Module]] = {
        "verifier": ActionConsistencyVerifier,
        "reachability": DistributionalReachability,
        "join": StateJoinHead,
        "autoregressive_proposal": AutoregressiveProposal,
        "denoising_proposal": DiscreteDenoisingProposal,
        "dts": VectorDTSHead,
        "ranker": CounterexampleRanker,
    }
    modules: dict[str, torch.nn.Module] = {}
    for name, state_dict in checkpoint["head_state_dicts"].items():
        if name not in constructors:
            raise ValueError(f"unknown trained component: {name}")
        module = constructors[name](head_config).to(device)
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
        modules[name] = module
    return modules


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def binary_auroc(probability: np.ndarray, target: np.ndarray) -> float | None:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability, kind="stable")
    sorted_values = probability[order]
    ranks = np.empty(target.size, dtype=np.float64)
    start = 0
    while start < target.size:
        end = start + 1
        while end < target.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[target == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def binary_reliability(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    bin_count: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bin_count + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bin_count):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        selected = (probability >= lower) & (
            probability <= upper if index == bin_count - 1 else probability < upper
        )
        count = int(selected.sum())
        if count:
            confidence = float(probability[selected].mean())
            empirical = float(target[selected].mean())
            ece += count / target.size * abs(confidence - empirical)
        else:
            confidence = 0.0
            empirical = 0.0
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": confidence,
                "empirical_probability": empirical,
            }
        )
    return float(ece), rows


def select_precision_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    required_precision: float,
) -> dict[str, float | int | bool]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    candidates: list[dict[str, float | int | bool]] = []
    for threshold in np.unique(probability):
        predicted = probability >= threshold
        predicted_count = int(predicted.sum())
        if predicted_count == 0:
            continue
        true_positive = int((predicted & (target == 1)).sum())
        precision = true_positive / predicted_count
        recall = true_positive / max(int((target == 1).sum()), 1)
        candidates.append(
            {
                "join_threshold": float(threshold),
                "join_precision": float(precision),
                "join_recall": float(recall),
                "join_predicted_positive": predicted_count,
                "join_precision_gate_passed": precision >= required_precision,
            }
        )
    if not candidates:
        raise ValueError("join calibration requires at least one prediction")
    eligible = [row for row in candidates if bool(row["join_precision_gate_passed"])]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["join_recall"]) if eligible else float(row["join_precision"]),
            float(row["join_precision"]),
            float(row["join_threshold"]),
        ),
    )


def calibrate_heads(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    entries: list[dict[str, Any]],
    *,
    horizon: int,
    history_size: int,
    rollout_semantics: Any,
    batch_size: int,
    chunk_batch_size: int,
    dts_batch_size: int,
    batches: int,
    seed: int,
    required_join_precision: float,
    device: torch.device,
) -> dict[str, Any]:
    general_sampler = PlannerBatchSampler(
        entries, horizon=horizon, require_full_chunk=False
    )
    chunk_sampler = PlannerBatchSampler(
        entries, horizon=horizon, require_full_chunk=True
    )
    general_rng = np.random.default_rng(hierarchical_seed("calibration-general", seed))
    chunk_rng = np.random.default_rng(hierarchical_seed("calibration-chunk", seed))
    dts_rng = np.random.default_rng(hierarchical_seed("calibration-dts", seed))
    totals: defaultdict[str, float] = defaultdict(float)
    reach_squared: torch.Tensor | None = None
    reach_count = 0
    reach_probabilities: list[torch.Tensor] = []
    reach_targets: list[torch.Tensor] = []
    join_probabilities: list[torch.Tensor] = []
    join_targets: list[torch.Tensor] = []
    for _ in range(batches):
        batch = general_sampler.sample(
            general_rng, batch_size=batch_size, device=device
        )
        latents = encode_planner_batch(model, batch, gradients=False)
        if "verifier" in modules:
            with torch.no_grad():
                prediction = modules["verifier"](
                    latents["source"], latents["successor"]
                ).argmax(dim=-1)
            totals["verifier_correct"] += float(
                (prediction == batch.action_ids - 1).sum()
            )
            totals["verifier_count"] += batch.batch_size
        if "join" in modules:
            imagined_successor = predicted_successor_batch(
                model,
                latents["source"],
                batch.action_ids,
                history_size=history_size,
                semantics=rollout_semantics,
            )
            with torch.no_grad():
                probability = modules["join"].probability(
                    imagined_successor, latents["comparison"]
                )
            join_probabilities.append(probability.cpu())
            join_targets.append(batch.join_labels.cpu())
        if "reachability" in modules:
            with torch.no_grad():
                probability = modules["reachability"](
                    latents["source"], latents["goal"]
                )
            bins_tensor = torch.tensor(
                modules["reachability"].config.reachability_bins,
                dtype=batch.bfs_distances.dtype,
                device=device,
            )
            target = (
                batch.bfs_distances.reshape(-1, 1) <= bins_tensor.reshape(1, -1)
            ).float()
            squared = (probability - target).square().sum(dim=0).cpu()
            reach_squared = (
                squared if reach_squared is None else reach_squared + squared
            )
            reach_count += batch.batch_size
            reach_probabilities.append(probability.cpu())
            reach_targets.append(target.cpu())
        if "autoregressive_proposal" in modules or "denoising_proposal" in modules:
            chunk_batch = chunk_sampler.sample(
                chunk_rng, batch_size=chunk_batch_size, device=device
            )
            chunk_latents = encode_planner_batch(model, chunk_batch, gradients=False)
        if "autoregressive_proposal" in modules:
            with torch.no_grad():
                logits = modules["autoregressive_proposal"](
                    chunk_latents["source"],
                    chunk_latents["goal"],
                    chunk_batch.optimal_action_chunks,
                )
            totals["proposal_correct"] += float(
                (logits.argmax(dim=-1) == chunk_batch.optimal_action_chunks - 1).sum()
            )
            totals["proposal_count"] += int(chunk_batch.optimal_action_chunks.numel())
        if "denoising_proposal" in modules:
            denoising = modules["denoising_proposal"]
            masked = torch.full_like(
                chunk_batch.optimal_action_chunks,
                denoising.mask_slot,
            )
            with torch.no_grad():
                logits = denoising(
                    chunk_latents["source"],
                    chunk_latents["goal"],
                    masked,
                )
            totals["proposal_correct"] += float(
                (logits.argmax(dim=-1) == chunk_batch.optimal_action_chunks - 1).sum()
            )
            totals["proposal_count"] += int(chunk_batch.optimal_action_chunks.numel())
        if "dts" in modules:
            dts_batch = chunk_sampler.sample(
                dts_rng, batch_size=dts_batch_size, device=device
            )
            dts_latents = encode_planner_batch(model, dts_batch, gradients=False)
            with torch.no_grad():
                output = modules["dts"](dts_latents["source"], dts_latents["goal"])
            totals["dts_policy_correct"] += float(
                (
                    output["policy_logits"].argmax(dim=-1)
                    == dts_batch.optimal_action_chunks[:, 0] - 1
                ).sum()
            )
            totals["dts_policy_count"] += dts_batch.batch_size
    metrics: dict[str, Any] = {
        "calibration_batches": batches,
        "calibration_batch_size": batch_size,
        "calibration_chunk_batch_size": chunk_batch_size,
        "calibration_dts_batch_size": dts_batch_size,
        "calibration_seed": seed,
    }
    if totals["verifier_count"]:
        metrics["verifier_accuracy"] = safe_divide(
            totals["verifier_correct"], totals["verifier_count"]
        )
    if join_probabilities:
        metrics.update(
            select_precision_threshold(
                torch.cat(join_probabilities).numpy(),
                torch.cat(join_targets).numpy(),
                required_precision=required_join_precision,
            )
        )
    if reach_squared is not None:
        boundaries = modules["reachability"].config.reachability_bins
        probability_matrix = torch.cat(reach_probabilities).numpy()
        target_matrix = torch.cat(reach_targets).numpy()
        brier: dict[str, float] = {}
        ece: dict[str, float] = {}
        auroc: dict[str, float | None] = {}
        reliability: dict[str, list[dict[str, float | int]]] = {}
        for index, boundary in enumerate(boundaries):
            label = str(boundary)
            brier[label] = float(reach_squared[index] / reach_count)
            ece[label], reliability[label] = binary_reliability(
                probability_matrix[:, index], target_matrix[:, index]
            )
            auroc[label] = binary_auroc(
                probability_matrix[:, index], target_matrix[:, index]
            )
        metrics["reachability_brier_by_bin"] = brier
        metrics["reachability_ece_by_bin"] = ece
        metrics["reachability_auroc_by_bin"] = auroc
        metrics["reachability_reliability_by_bin"] = reliability
    if totals["proposal_count"]:
        metrics["proposal_teacher_forced_accuracy"] = safe_divide(
            totals["proposal_correct"], totals["proposal_count"]
        )
    if totals["dts_policy_count"]:
        metrics["dts_root_action_accuracy"] = safe_divide(
            totals["dts_policy_correct"], totals["dts_policy_count"]
        )
    return metrics


def jepa_stability_metrics(
    source_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    entries: list[dict[str, Any]],
    *,
    config: Any,
    batches: int,
    device: torch.device,
) -> dict[str, Any]:
    sampler = JEPATrajectorySampler(
        entries,
        sequence_length=config.training.sequence_length,
    )
    rng = np.random.default_rng(
        hierarchical_seed("joint-jepa-validation", config.training.calibration_seed)
    )
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    source_totals: defaultdict[str, float] = defaultdict(float)
    adapted_totals: defaultdict[str, float] = defaultdict(float)
    source_model.eval()
    adapted_model.eval()
    with torch.no_grad():
        for batch_index in range(batches):
            batch = sampler.sample(
                rng,
                batch_size=config.training.joint_batch_size,
                size_slot=batch_index,
                device=device,
            )
            projection_seed = hierarchical_seed(
                "joint-jepa-projection",
                config.training.calibration_seed,
                batch_index,
            )
            torch.manual_seed(projection_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(projection_seed)
            source_losses = original_jepa_losses(source_model, batch, sigreg, config)
            torch.manual_seed(projection_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(projection_seed)
            adapted_losses = original_jepa_losses(adapted_model, batch, sigreg, config)
            for name, value in source_losses.items():
                source_totals[name] += float(value)
            for name, value in adapted_losses.items():
                adapted_totals[name] += float(value)
    source_means = {
        name: value / batches for name, value in sorted(source_totals.items())
    }
    adapted_means = {
        name: value / batches for name, value in sorted(adapted_totals.items())
    }
    source_total = float(sum(source_means.values()))
    adapted_total = float(sum(adapted_means.values()))
    relative_change = (adapted_total - source_total) / max(abs(source_total), 1e-12)
    return {
        "jepa_validation_source": source_means,
        "jepa_validation_adapted": adapted_means,
        "jepa_validation_total_source": source_total,
        "jepa_validation_total_adapted": adapted_total,
        "jepa_validation_relative_change": float(relative_change),
        "jepa_stability_gate_passed": bool(relative_change <= 0.10),
        "jepa_stability_threshold": 0.10,
        "jepa_validation_sequence_length": config.training.sequence_length,
        "jepa_validation_size_schedule": "round_robin_over_validation_sizes",
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("calibration requires a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    base_method = method_by_name(config, args.method)
    if base_method.reuse_component_from is not None:
        raise ValueError("checkpoint reuse aliases cannot recalibrate")
    method = resolve_effective_method(config, lock, base_method)
    if args.allow_dirty_worktree:
        raise ValueError("formal calibration requires a clean worktree")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked matrix")
    if not method.component_checkpoint_required:
        raise ValueError("selected method has no trainable planner component")
    input_path = resolve_path(
        args.input
        or config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    output_path = resolve_path(
        args.output
        or config.paths.component_checkpoint_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    rerun = prepare_formal_output(
        output_path, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError("input is not a planner-frontier component checkpoint")
    if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported component checkpoint version")
    if checkpoint.get("stage") != "component_training":
        raise ValueError("calibration input must be an immutable training checkpoint")
    if (
        checkpoint.get("method_name") != method.name
        or int(checkpoint.get("backbone_seed", -1)) != args.backbone_seed
        or int(checkpoint.get("planner_seed", -1)) != args.planner_seed
    ):
        raise ValueError("calibration method/seed mismatch")
    if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("calibration checkpoint analysis-spec mismatch")
    if checkpoint.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
    ):
        raise ValueError("calibration checkpoint training-spec mismatch")
    checkpoint_protocol = checkpoint.get("protocol", {})
    if checkpoint_protocol.get("git_dirty") is not False:
        raise ValueError("calibration rejects a dirty training checkpoint")
    if checkpoint_protocol.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError("training checkpoint code fingerprint mismatch")
    device = resolve_device(args.device or config.device)
    set_seed(config.training.calibration_seed, deterministic=True)
    model, _, source_path = load_source_lewm(
        config, lock, seed=args.backbone_seed, device=device
    )
    if checkpoint.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("training and calibration source checkpoints differ")
    if method.track == "J":
        source_model, _, source_reference_path = load_source_lewm(
            config, lock, seed=args.backbone_seed, device=device
        )
        if source_reference_path != source_path:
            raise ValueError("joint stability reference uses another source checkpoint")
        source_model.eval()
        for parameter in source_model.parameters():
            parameter.requires_grad = False
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        source_model = None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = load_heads(checkpoint, device)
    if set(modules) != required_head_names(method):
        raise ValueError("training checkpoint does not contain the exact method heads")
    validation_path = resolve_path(config.paths.validation_manifest)
    if sha256_file(validation_path) != lock["validation_manifest"]["sha256"]:
        raise ValueError("validation manifest hash mismatch")
    validation_entries = read_jsonl(validation_path)
    metrics = calibrate_heads(
        model,
        modules,
        validation_entries,
        horizon=method.planner.horizon,
        history_size=method.planner.history_size,
        rollout_semantics=method.planner.rollout_semantics,
        batch_size=config.training.transition_batch_size,
        chunk_batch_size=config.training.proposal_batch_size,
        dts_batch_size=config.training.dts_batch_size,
        batches=config.training.calibration_batches,
        seed=config.training.calibration_seed,
        required_join_precision=method.memory.required_validation_precision,
        device=device,
    )
    if source_model is not None:
        metrics.update(
            jepa_stability_metrics(
                source_model,
                model,
                validation_entries,
                config=config,
                batches=config.training.calibration_batches,
                device=device,
            )
        )
    if method.memory.hard_pruning:
        metrics["hard_pruning_eligible"] = (
            metrics.get("join_precision", 0.0)
            >= method.memory.required_validation_precision
        )
    payload = dict(checkpoint)
    retrieval_bank_path = None
    if method.proposal.retrieval_weight > 0.0:
        retrieval_bank_path = resolve_path(
            config.paths.retrieval_bank_template.format(
                method=method.name,
                backbone_seed=args.backbone_seed,
                planner_seed=args.planner_seed,
            )
        )
        bank = RetrievalBank.load(retrieval_bank_path)
        train_task_hashes = {
            str(entry["task_hash"])
            for entry in read_jsonl(resolve_path(config.paths.train_manifest))
        }
        if not set(bank.task_hashes) <= train_task_hashes:
            raise ValueError("retrieval bank contains a non-training task hash")
        metrics["retrieval_bank_fingerprint"] = bank.fingerprint
    payload.update(
        {
            "stage": "component_calibration",
            "source_training_checkpoint": str(input_path),
            "source_training_checkpoint_sha256": sha256_file(input_path),
            "validation_manifest_sha256": sha256_file(validation_path),
            "validation_metrics": metrics,
            "retrieval_bank_path": str(retrieval_bank_path)
            if retrieval_bank_path is not None
            else None,
            "calibration_rerun": rerun,
        }
    )
    atomic_torch_save(output_path, payload)


if __name__ == "__main__":
    main()
