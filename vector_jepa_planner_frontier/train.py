"""Train frozen-backbone or jointly-updated pooled-vector planner components."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from final_closure.common import read_jsonl, sha256_file
from hdwm.losses import SIGReg
from scripts.train.train_canonical_lewm import compute_position_labels
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_torch_save,
    component_checkpoint_path,
    hierarchical_seed,
    load_json,
    load_study_config,
    method_by_name,
    parent_component_checkpoint_path,
    prepare_formal_output,
    protocol_metadata,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    training_spec_sha256,
    uses_counterexample_rounds,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.data import (
    CounterexampleBatchSampler,
    CounterexampleRawBatch,
    JEPATrajectoryBatch,
    JEPATrajectorySampler,
    PlannerBatchSampler,
    PlannerRawBatch,
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
    pairwise_ranking_loss,
    required_head_names,
)
from vector_jepa_planner_frontier.schemas import (
    MethodConfig,
    RolloutSemantics,
    StudyConfig,
)
from vector_jepa_planner_frontier.world_model import VectorWorldModel

CHUNK_HEADS = {
    "autoregressive_proposal",
    "denoising_proposal",
    "dts",
    "ranker",
}
HEAD_SEED_IDS = {
    "autoregressive_proposal": 1,
    "denoising_proposal": 2,
    "dts": 3,
    "join": 4,
    "ranker": 5,
    "reachability": 6,
    "verifier": 7,
}


def head_step_limits(config: StudyConfig) -> dict[str, int]:
    return {
        "verifier": config.training.verifier_steps,
        "reachability": config.training.reachability_steps,
        "join": config.training.join_steps,
        "autoregressive_proposal": config.training.proposal_steps,
        "denoising_proposal": config.training.denoising_steps,
        "dts": config.training.dts_steps,
        "ranker": config.training.ranker_initial_steps,
    }


def head_learning_rates(config: StudyConfig) -> dict[str, float]:
    return {
        "verifier": config.training.verifier_learning_rate,
        "reachability": config.training.reachability_learning_rate,
        "join": config.training.join_learning_rate,
        "autoregressive_proposal": config.training.proposal_learning_rate,
        "denoising_proposal": config.training.proposal_learning_rate,
        "dts": config.training.dts_learning_rate,
        "ranker": config.training.ranker_learning_rate,
    }


def component_stochastic_rngs(
    backbone_seed: int,
    planner_seed: int,
) -> dict[str, np.random.Generator]:
    """Independent streams prevent one factorial component perturbing another."""

    return {
        name: np.random.default_rng(
            hierarchical_seed(
                f"planner-{name}-stochastic",
                backbone_seed,
                planner_seed,
            )
        )
        for name in ("denoising_proposal", "ranker")
    }


def locked_training_steps(
    config: StudyConfig,
    method: MethodConfig,
    module_names: set[str],
) -> int:
    if method.trainable_components == ():
        return 0
    if method.track == "J":
        return config.training.joint_steps
    limits = head_step_limits(config)
    selected = (
        module_names
        if method.trainable_components is None
        else module_names.intersection(method.trainable_components)
    )
    if not selected:
        raise ValueError("the method selects no trainable planner component")
    return max(limits[name] for name in selected)


def learning_rate_factor(step: int, limit: int, config: StudyConfig) -> float:
    warmup = max(1, int(round(config.training.warmup_fraction * limit)))
    if step <= warmup:
        return step / warmup
    progress = min(max((step - warmup) / max(limit - warmup, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    floor = config.training.final_learning_rate_ratio
    return float(floor + (1.0 - floor) * cosine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def required_heads(
    method: MethodConfig,
    head_config: HeadConfig,
    *,
    backbone_seed: int | None = None,
    planner_seed: int | None = None,
) -> dict[str, torch.nn.Module]:
    constructors: dict[str, type[torch.nn.Module]] = {
        "verifier": ActionConsistencyVerifier,
        "reachability": DistributionalReachability,
        "join": StateJoinHead,
        "autoregressive_proposal": AutoregressiveProposal,
        "denoising_proposal": DiscreteDenoisingProposal,
        "dts": VectorDTSHead,
        "ranker": CounterexampleRanker,
    }
    if (backbone_seed is None) != (planner_seed is None):
        raise ValueError("head initialization requires both nested seed labels")
    modules: dict[str, torch.nn.Module] = {}
    for name in sorted(required_head_names(method)):
        if backbone_seed is None:
            modules[name] = constructors[name](head_config)
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                hierarchical_seed(
                    "planner-head-initialization",
                    int(backbone_seed),
                    int(planner_seed),
                    HEAD_SEED_IDS[name],
                )
            )
            modules[name] = constructors[name](head_config)
    if method.component_checkpoint_required and not modules:
        raise ValueError("method requires a component checkpoint but activates no head")
    return modules


def distance_class(distances: torch.Tensor, bins: tuple[int, ...]) -> torch.Tensor:
    boundaries = torch.tensor(bins, dtype=distances.dtype, device=distances.device)
    comparison = distances.reshape(-1, 1) <= boundaries.reshape(1, -1)
    any_match = comparison.any(dim=1)
    first = comparison.to(torch.int64).argmax(dim=1)
    return torch.where(any_match, first, torch.full_like(first, len(bins) - 1))


def rollout_training_batch(
    model: torch.nn.Module,
    source: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_size: int,
    semantics: RolloutSemantics,
) -> torch.Tensor:
    """Differentiable batched rollout from independently paired training roots."""

    if source.ndim != 2 or actions.ndim != 2 or source.shape[0] != actions.shape[0]:
        raise ValueError("batched ranker rollouts require paired roots and chunks")
    if bool(((actions < 1) | (actions > 4)).any()):
        raise ValueError("ranker action chunks must use environment actions [1, 4]")
    embeddings = source.unsqueeze(1).repeat(1, history_size, 1)
    action_history = torch.full(
        (source.shape[0], history_size - 1),
        4,
        dtype=torch.long,
        device=source.device,
    )
    terminal: torch.Tensor | None = None
    for step in range(actions.shape[1]):
        proposed = actions[:, step : step + 1]
        if semantics == RolloutSemantics.ACTION_ALIGNED_V2:
            predictor_embeddings = torch.cat(
                [embeddings[:, 1:], embeddings[:, -1:]], dim=1
            )
            predictor_actions = torch.cat([action_history[:, 1:], proposed], dim=1)
            prediction = model.predictor(predictor_embeddings, predictor_actions)
            action_history = predictor_actions
        else:
            prediction = model.predictor(embeddings, action_history)
            action_history = torch.cat([action_history[:, 1:], proposed], dim=1)
        terminal = prediction[:, -1]
        embeddings = torch.cat([embeddings[:, 1:], terminal.unsqueeze(1)], dim=1)
    if terminal is None or not torch.isfinite(terminal).all():
        raise FloatingPointError("ranker training rollout produced invalid latents")
    return terminal


def predicted_successor_batch(
    model: torch.nn.Module,
    source: torch.Tensor,
    action_ids: torch.Tensor,
    *,
    history_size: int,
    semantics: RolloutSemantics,
) -> torch.Tensor:
    """Predict one action-conditioned successor under the planner's semantics."""

    actions = action_ids.reshape(-1, 1)
    if semantics == RolloutSemantics.LEGACY_WARMUP_V1:
        actions = actions.repeat(1, 2)
    return rollout_training_batch(
        model,
        source,
        actions,
        history_size=history_size,
        semantics=semantics,
    )


def component_losses(
    modules: dict[str, torch.nn.Module],
    latents: dict[str, torch.Tensor],
    batch: PlannerRawBatch,
    *,
    world_model: VectorWorldModel,
    method: MethodConfig,
    stochastic_rngs: dict[str, np.random.Generator],
) -> dict[str, torch.Tensor]:
    if CHUNK_HEADS.intersection(modules) and not batch.optimal_chunks_are_full:
        raise ValueError("proposal/DTS/ranker losses require full optimal chunks")
    losses: dict[str, torch.Tensor] = {}
    if "verifier" in modules:
        if method.control.verifier_targets != "random_untrained":
            targets = batch.action_ids - 1
            successors = latents["successor"]
            if method.control.verifier_targets == "action_shuffle":
                targets = torch.roll(targets, shifts=1, dims=0)
            elif method.control.verifier_targets == "pair_shuffle":
                successors = torch.roll(successors, shifts=1, dims=0)
            logits = modules["verifier"](latents["source"], successors)
            losses["verifier"] = F.cross_entropy(logits, targets)
    if "reachability" in modules:
        reachability = modules["reachability"]
        reach_loss, _ = reachability.loss(
            latents["source"], latents["goal"], batch.bfs_distances
        )
        losses["reachability"] = reach_loss
    if "join" in modules:
        imagined_successor = predicted_successor_batch(
            world_model.model,
            latents["source"],
            batch.action_ids,
            history_size=method.planner.history_size,
            semantics=method.planner.rollout_semantics,
        )
        logits = modules["join"](imagined_successor, latents["comparison"])
        losses["join"] = F.binary_cross_entropy_with_logits(logits, batch.join_labels)
    if "autoregressive_proposal" in modules:
        logits = modules["autoregressive_proposal"](
            latents["source"], latents["goal"], batch.optimal_action_chunks
        )
        losses["autoregressive_proposal"] = F.cross_entropy(
            logits.reshape(-1, 4),
            (batch.optimal_action_chunks - 1).reshape(-1),
        )
    if "denoising_proposal" in modules:
        model = modules["denoising_proposal"]
        clean_slots = batch.optimal_action_chunks - 1
        if "denoising_proposal" not in stochastic_rngs:
            raise ValueError("denoising proposal requires its isolated random stream")
        mask = torch.as_tensor(
            stochastic_rngs["denoising_proposal"].random(clean_slots.shape) < 0.5,
            dtype=torch.bool,
            device=clean_slots.device,
        )
        noisy = torch.where(
            mask, torch.full_like(clean_slots, model.mask_slot), clean_slots
        )
        logits = model(latents["source"], latents["goal"], noisy)
        losses["denoising_proposal"] = (
            F.cross_entropy(logits[mask], clean_slots[mask])
            if bool(mask.any())
            else logits.sum() * 0.0
        )
    if "dts" in modules:
        output = modules["dts"](latents["source"], latents["goal"])
        losses["dts_policy"] = F.cross_entropy(
            output["policy_logits"], batch.optimal_action_chunks[:, 0] - 1
        )
        target = distance_class(
            batch.bfs_distances, modules["dts"].config.reachability_bins
        )
        losses["dts_value"] = F.cross_entropy(output["value_logits"], target)
        losses["dts_uncertainty"] = output["uncertainty"].mean() * 1e-3
    if "ranker" in modules:
        good_actions = batch.optimal_action_chunks
        if "ranker" not in stochastic_rngs:
            raise ValueError("ranker requires its isolated random-negative stream")
        bad_actions = torch.as_tensor(
            stochastic_rngs["ranker"].integers(
                1,
                5,
                size=tuple(good_actions.shape),
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=latents["source"].device,
        )
        good_terminal = rollout_training_batch(
            world_model.model,
            latents["source"],
            good_actions,
            history_size=method.planner.history_size,
            semantics=method.planner.rollout_semantics,
        )
        bad_terminal = rollout_training_batch(
            world_model.model,
            latents["source"],
            bad_actions,
            history_size=method.planner.history_size,
            semantics=method.planner.rollout_semantics,
        )
        ranker = modules["ranker"]
        losses["ranker"] = pairwise_ranking_loss(
            ranker(latents["source"], good_terminal, good_actions),
            ranker(latents["source"], bad_terminal, bad_actions),
        )
    return losses


def encode_counterexample_sources(
    model: torch.nn.Module,
    batch: CounterexampleRawBatch,
) -> torch.Tensor:
    encoded = model.encoder(batch.source_observations, batch.maze_size)
    embeddings, _ = model.embedding_projector(encoded)
    if embeddings.shape != (batch.source_observations.shape[0], 256):
        raise ValueError("unexpected counterexample embedding shape")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("counterexample encoder produced non-finite vectors")
    return embeddings


def joint_counterexample_loss(
    model: torch.nn.Module,
    ranker: torch.nn.Module,
    batch: CounterexampleRawBatch,
    *,
    method: MethodConfig,
) -> torch.Tensor:
    source = encode_counterexample_sources(model, batch)
    good_terminal = rollout_training_batch(
        model,
        source,
        batch.good_actions,
        history_size=method.planner.history_size,
        semantics=method.planner.rollout_semantics,
    )
    bad_terminal = rollout_training_batch(
        model,
        source,
        batch.bad_actions,
        history_size=method.planner.history_size,
        semantics=method.planner.rollout_semantics,
    )
    return pairwise_ranking_loss(
        ranker(source, good_terminal, batch.good_actions),
        ranker(source, bad_terminal, batch.bad_actions),
    )


def original_jepa_losses(
    model: torch.nn.Module,
    batch: JEPATrajectoryBatch,
    sigreg: SIGReg,
    config: StudyConfig,
    *,
    sigreg_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    observations = batch.observations
    actions = batch.actions
    output = model(observations, actions, batch.maze_size)
    prediction = F.mse_loss(output["prediction"], output["target"])
    regularization = sigreg(output["sigreg_embedding"].transpose(0, 1))
    x, y, dx, dy = compute_position_labels(
        batch.states,
        observations[:, :, :, :, 3],
        batch.maze_size,
    )
    absolute_target = torch.stack([x, y], dim=-1)
    relative_target = torch.stack([dx, dy], dim=-1)
    batch_size, sequence_length = observations.shape[:2]
    goal = observations[..., 3].reshape(batch_size, sequence_length, -1).argmax(-1)
    goal_target = torch.stack(
        [
            (goal % batch.maze_size).float() / max(batch.maze_size - 1, 1),
            (goal // batch.maze_size).float() / max(batch.maze_size - 1, 1),
        ],
        dim=-1,
    )
    return {
        "jepa_prediction": config.training.prediction_weight * prediction,
        "jepa_sigreg": config.training.sigreg_weight
        * float(sigreg_multiplier)
        * regularization,
        "jepa_abs_position": config.training.abs_position_weight
        * F.mse_loss(output["abs_pos_pred"], absolute_target),
        "jepa_relative_position": config.training.relative_position_weight
        * F.mse_loss(output["rel_pos_pred"], relative_target),
        "jepa_goal_position": config.training.goal_position_weight
        * F.mse_loss(output["goal_pos_pred"], goal_target),
    }


def load_parent_initialization(
    config: StudyConfig,
    lock: dict[str, Any],
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
    source_path: Path,
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
) -> dict[str, Any] | None:
    parent_path = parent_component_checkpoint_path(
        config,
        method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )
    if parent_path is None:
        return None
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    parent = resolve_effective_method(
        config, lock, method_by_name(config, str(method.initialization_parent))
    )
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    expected_stage = (
        "counterexample_training_round"
        if uses_counterexample_rounds(parent)
        else "component_calibration"
    )
    if checkpoint.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError("parent checkpoint belongs to another experiment family")
    if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("parent checkpoint format is unsupported")
    if checkpoint.get("stage") != expected_stage:
        raise ValueError("parent checkpoint is not at its locked final stage")
    if (
        expected_stage == "counterexample_training_round"
        and int(checkpoint.get("counterexample_round", -1))
        != config.training.counterexample_rounds
    ):
        raise ValueError("parent counterexample checkpoint is not round three")
    if (
        checkpoint.get("method_name") != parent.name
        or int(checkpoint.get("backbone_seed", -1)) != backbone_seed
        or int(checkpoint.get("planner_seed", -1)) != planner_seed
    ):
        raise ValueError("parent checkpoint method or seed label mismatch")
    if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("parent checkpoint analysis specification mismatch")
    if checkpoint.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=parent,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    ):
        raise ValueError("parent checkpoint training specification mismatch")
    if checkpoint.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("parent and child use different source backbones")
    parent_protocol = checkpoint.get("protocol", {})
    if parent_protocol.get("git_dirty") is not False:
        raise ValueError("parent checkpoint was produced from a dirty worktree")
    if parent_protocol.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError("parent checkpoint code fingerprint mismatch")
    parent_states = checkpoint.get("head_state_dicts", {})
    allowed_new = set(method.trainable_components or ())
    for name, module in modules.items():
        if name in parent_states:
            module.load_state_dict(parent_states[name], strict=True)
        elif name not in allowed_new:
            raise ValueError(f"parent checkpoint does not supply required head: {name}")
    unused = set(parent_states).difference(modules)
    if unused:
        raise ValueError(f"child silently drops parent heads: {sorted(unused)}")
    if checkpoint.get("model_state_dict") is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "method": parent.name,
        "path": str(parent_path),
        "sha256": sha256_file(parent_path),
        "stage": checkpoint["stage"],
    }


def counterexample_fold(task_hash: str) -> int:
    digest = hashlib.sha256(f"vector-jepa-mining-v1:{task_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3 + 1


def load_joint_counterexamples(
    config: StudyConfig,
    lock: dict[str, Any],
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
    train_entries: list[dict[str, Any]],
) -> tuple[CounterexampleBatchSampler, list[dict[str, Any]]]:
    """Load the exact frozen P6 hard-negative corpus inherited by Track J."""

    if method.track != "J" or method.initialization_parent is None:
        raise ValueError("joint counterexamples require a Track J inherited method")
    parent_base = method_by_name(config, method.initialization_parent)
    parent = resolve_effective_method(config, lock, parent_base)
    if not uses_counterexample_rounds(parent):
        raise ValueError("Track J parent lacks matched counterexample rounds")
    if (
        parent.control.ranker_negatives != "hard_three_rounds"
        or method.control.ranker_negatives != "hard_three_rounds"
    ):
        raise ValueError("Track J must preserve the P6 hard-negative definition")
    previous_path = resolve_path(
        config.paths.component_checkpoint_template.format(
            method=parent.name,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
    )
    if not previous_path.is_file():
        raise FileNotFoundError(previous_path)
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    expected_training_spec = training_spec_sha256(
        config,
        lock,
        method=parent,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )
    for round_index in range(1, config.training.counterexample_rounds + 1):
        dataset_path = resolve_path(
            config.paths.counterexample_dataset_template.format(
                method=parent.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        checkpoint_path = resolve_path(
            config.paths.counterexample_round_template.format(
                method=parent.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        dataset = load_json(dataset_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_dataset = {
            "schema": "vector-jepa-counterexamples-v1",
            "method": parent.name,
            "backbone_seed": backbone_seed,
            "planner_seed": planner_seed,
            "round": round_index,
            "train_manifest_sha256": lock["train_manifest"]["sha256"],
            "source_checkpoint_sha256": sha256_file(previous_path),
            "diagnostic_limit": 0,
        }
        if any(dataset.get(key) != value for key, value in expected_dataset.items()):
            raise ValueError(
                f"Track J counterexample dataset provenance mismatch: {dataset_path}"
            )
        if Path(str(dataset.get("source_checkpoint", ""))).resolve() != (
            previous_path.resolve()
        ):
            raise ValueError(
                f"Track J counterexample chain is broken at round {round_index}"
            )
        protocol = checkpoint.get("protocol", {})
        if (
            checkpoint.get("experiment_family") != EXPERIMENT_FAMILY
            or int(checkpoint.get("format_version", -1)) != FORMAT_VERSION
            or checkpoint.get("stage") != "counterexample_training_round"
            or checkpoint.get("method_name") != parent.name
            or int(checkpoint.get("backbone_seed", -1)) != backbone_seed
            or int(checkpoint.get("planner_seed", -1)) != planner_seed
            or int(checkpoint.get("counterexample_round", -1)) != round_index
            or checkpoint.get("analysis_spec_sha256")
            != analysis_spec_sha256(config, lock)
            or checkpoint.get("training_spec_sha256") != expected_training_spec
            or checkpoint.get("train_manifest_sha256")
            != lock["train_manifest"]["sha256"]
            or protocol.get("git_dirty") is not False
            or protocol.get("code_fingerprint") != lock["code_fingerprint"]
        ):
            raise ValueError(
                "Track J counterexample checkpoint provenance mismatch: "
                f"{checkpoint_path}"
            )
        if Path(str(checkpoint.get("counterexample_dataset", ""))).resolve() != (
            dataset_path.resolve()
        ) or checkpoint.get("counterexample_dataset_sha256") != sha256_file(
            dataset_path
        ):
            raise ValueError(
                "Track J checkpoint does not authenticate its dataset: "
                f"{checkpoint_path}"
            )
        round_records = dataset.get("records")
        if not isinstance(round_records, list):
            raise ValueError(f"counterexample records are not a list: {dataset_path}")
        summary = checkpoint.get("counterexample_training_summary", {})
        expected_steps = (
            config.training.counterexample_round_steps if round_records else 0
        )
        if (
            int(summary.get("mined_count", -1)) != len(round_records)
            or int(summary.get("steps", -1)) != expected_steps
        ):
            raise ValueError(
                f"Track J counterexample count/step mismatch: {checkpoint_path}"
            )
        if any(
            counterexample_fold(str(record.get("task_hash", ""))) != round_index
            for record in round_records
        ):
            raise ValueError(f"counterexample fold leakage detected: {dataset_path}")
        records.extend(round_records)
        provenance.append(
            {
                "round": round_index,
                "dataset_path": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_checkpoint_path": str(previous_path),
                "source_checkpoint_sha256": sha256_file(previous_path),
                "record_count": len(round_records),
            }
        )
        previous_path = checkpoint_path
    expected_parent_path = component_checkpoint_path(
        config,
        parent_base,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )
    if expected_parent_path is None or previous_path.resolve() != (
        expected_parent_path.resolve()
    ):
        raise ValueError("Track J data chain does not end at its inherited parent")
    sampler = CounterexampleBatchSampler(
        records,
        train_entries,
        horizon=method.planner.horizon,
        expected_negative_source="planner_false_optimistic",
    )
    return sampler, provenance


def train_components(
    config: StudyConfig,
    lock: dict[str, Any],
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
    device: torch.device,
    steps: int,
) -> tuple[torch.nn.Module, dict[str, torch.nn.Module], dict[str, Any], Path]:
    model, _, source_path = load_source_lewm(
        config, lock, seed=backbone_seed, device=device
    )
    head_config = HeadConfig(
        latent_dim=256,
        hidden_dim=512,
        action_count=4,
        horizon=method.planner.horizon,
        reachability_bins=config.training.reachability_bins,
    )
    modules = {
        name: module.to(device)
        for name, module in required_heads(
            method,
            head_config,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        ).items()
    }
    parent_record = load_parent_initialization(
        config,
        lock,
        method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
        source_path=source_path,
        model=model,
        modules=modules,
    )
    if method.track == "F":
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    else:
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad = True
    for module in modules.values():
        module.train()
    trainable_module_names = set(modules)
    if method.trainable_components is not None:
        requested = set(method.trainable_components)
        unknown = requested.difference(modules)
        if unknown:
            raise ValueError(f"unknown trainable component names: {sorted(unknown)}")
        trainable_module_names.intersection_update(requested)
    if method.control.verifier_targets == "random_untrained":
        trainable_module_names.discard("verifier")
    for name, module in modules.items():
        is_trainable = name in trainable_module_names
        module.train(is_trainable)
        for parameter in module.parameters():
            parameter.requires_grad = is_trainable
    limits = head_step_limits(config)
    learning_rates = head_learning_rates(config)
    module_limits = {
        name: (steps if method.track == "J" else limits[name])
        for name in trainable_module_names
    }
    parameter_groups: list[dict[str, Any]] = []
    for name in sorted(trainable_module_names):
        if method.track == "J" and method.joint_hyperparameters is None:
            raise ValueError("Track J method lacks its locked hyperparameters")
        base_lr = (
            method.joint_hyperparameters.planner_learning_rate
            if method.track == "J"
            else learning_rates[name]
        )
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in modules[name].parameters()
                    if parameter.requires_grad
                ],
                "lr": base_lr,
                "base_lr": base_lr,
                "step_limit": module_limits[name],
                "component_name": name,
            }
        )
    if method.track == "J":
        assert method.joint_hyperparameters is not None
        base_lr = (
            method.joint_hyperparameters.planner_learning_rate
            * method.joint_hyperparameters.backbone_lr_multiplier
        )
        parameter_groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                "lr": base_lr,
                "base_lr": base_lr,
                "step_limit": steps,
                "component_name": "jepa_backbone",
            }
        )
    parameters = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    random_untrained_control = (
        method.control.verifier_targets == "random_untrained"
        and set(modules) == {"verifier"}
        and method.track == "F"
    )
    inherited_frozen_control = parent_record is not None and (
        method.trainable_components == ()
    )
    if not parameters and not random_untrained_control and not inherited_frozen_control:
        raise ValueError("no trainable planner or world-model parameter was selected")
    optimizer = (
        torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.training.weight_decay,
        )
        if parameters
        else None
    )
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    train_entries = read_jsonl(resolve_path(config.paths.train_manifest))
    general_sampler = PlannerBatchSampler(
        train_entries,
        horizon=method.planner.horizon,
        require_full_chunk=False,
    )
    chunk_sampler = PlannerBatchSampler(
        train_entries,
        horizon=method.planner.horizon,
        require_full_chunk=True,
    )
    trajectory_sampler = (
        JEPATrajectorySampler(
            train_entries,
            sequence_length=config.training.sequence_length,
        )
        if method.track == "J"
        else None
    )
    counterexample_sampler: CounterexampleBatchSampler | None = None
    counterexample_provenance: list[dict[str, Any]] = []
    if method.track == "J":
        counterexample_sampler, counterexample_provenance = load_joint_counterexamples(
            config,
            lock,
            method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
            train_entries=train_entries,
        )
    world_model = VectorWorldModel(
        model, device=device, history_size=method.planner.history_size
    )
    general_rng = np.random.default_rng(
        hierarchical_seed("planner-general-batches", backbone_seed, planner_seed)
    )
    chunk_rng = np.random.default_rng(
        hierarchical_seed("planner-chunk-batches", backbone_seed, planner_seed)
    )
    dts_rng = np.random.default_rng(
        hierarchical_seed("planner-dts-batches", backbone_seed, planner_seed)
    )
    counterexample_rng = np.random.default_rng(
        hierarchical_seed("planner-hard-negative-batches", backbone_seed, planner_seed)
    )
    trajectory_rng = np.random.default_rng(
        hierarchical_seed("joint-jepa-trajectories", backbone_seed, planner_seed)
    )
    stochastic_rngs = component_stochastic_rngs(backbone_seed, planner_seed)
    history: defaultdict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    executed_steps = (
        0 if (random_untrained_control or inherited_frozen_control) else steps
    )
    for step in range(1, executed_steps + 1):
        assert optimizer is not None
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * learning_rate_factor(
                step, int(group["step_limit"]), config
            )
        active_names = {
            name for name in trainable_module_names if step <= module_limits[name]
        }
        general_modules = {
            name: modules[name] for name in active_names if name not in CHUNK_HEADS
        }
        losses: dict[str, torch.Tensor] = {}
        general_batch: PlannerRawBatch | None = None
        if general_modules:
            general_batch = general_sampler.sample(
                general_rng,
                batch_size=(
                    config.training.joint_batch_size
                    if method.track == "J"
                    else config.training.transition_batch_size
                ),
                device=device,
            )
            general_latents = encode_planner_batch(
                model, general_batch, gradients=method.track == "J"
            )
        if general_modules:
            losses.update(
                component_losses(
                    general_modules,
                    general_latents,
                    general_batch,
                    world_model=world_model,
                    method=method,
                    stochastic_rngs=stochastic_rngs,
                )
            )
        proposal_modules = {
            name: modules[name]
            for name in active_names
            if name in CHUNK_HEADS
            and name != "dts"
            and not (method.track == "J" and name == "ranker")
        }
        if proposal_modules:
            chunk_batch = chunk_sampler.sample(
                chunk_rng,
                batch_size=(
                    config.training.joint_batch_size
                    if method.track == "J"
                    else config.training.proposal_batch_size
                ),
                device=device,
            )
            chunk_latents = encode_planner_batch(
                model, chunk_batch, gradients=method.track == "J"
            )
            losses.update(
                component_losses(
                    proposal_modules,
                    chunk_latents,
                    chunk_batch,
                    world_model=world_model,
                    method=method,
                    stochastic_rngs=stochastic_rngs,
                )
            )
        if "dts" in active_names:
            dts_batch = chunk_sampler.sample(
                dts_rng,
                batch_size=config.training.dts_batch_size,
                device=device,
            )
            dts_latents = encode_planner_batch(
                model, dts_batch, gradients=method.track == "J"
            )
            losses.update(
                component_losses(
                    {"dts": modules["dts"]},
                    dts_latents,
                    dts_batch,
                    world_model=world_model,
                    method=method,
                    stochastic_rngs=stochastic_rngs,
                )
            )
        if method.track == "J" and "ranker" in active_names:
            assert counterexample_sampler is not None
            hard_batch = counterexample_sampler.sample(
                counterexample_rng,
                batch_size=config.training.joint_batch_size,
                device=device,
            )
            losses["ranker"] = joint_counterexample_loss(
                model,
                modules["ranker"],
                hard_batch,
                method=method,
            )
        if method.track == "J":
            assert method.joint_hyperparameters is not None
            losses = {
                name: method.joint_hyperparameters.planner_loss_weight * loss
                for name, loss in losses.items()
            }
            assert trajectory_sampler is not None
            trajectory_batch = trajectory_sampler.sample(
                trajectory_rng,
                batch_size=config.training.joint_batch_size,
                size_slot=step - 1,
                device=device,
            )
            losses.update(
                original_jepa_losses(
                    model,
                    trajectory_batch,
                    sigreg,
                    config,
                    sigreg_multiplier=method.joint_hyperparameters.sigreg_multiplier,
                )
            )
        total = torch.stack(list(losses.values())).sum()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.training.grad_clip)
        optimizer.step()
        history["total"].append(float(total.detach()))
        for name, loss in losses.items():
            history[name].append(float(loss.detach()))
    summary = {
        name: float(np.mean(values[-min(500, len(values)) :]))
        for name, values in sorted(history.items())
    }
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    summary["steps"] = int(executed_steps)
    summary["locked_steps"] = int(steps)
    summary["module_step_limits"] = module_limits
    summary["schedule"] = "5pct_linear_warmup_cosine_to_10pct"
    summary["random_untrained_control"] = random_untrained_control
    summary["inherited_frozen_control"] = inherited_frozen_control
    summary["joint_jepa_sequence_length"] = (
        config.training.sequence_length if method.track == "J" else None
    )
    summary["joint_jepa_size_schedule"] = (
        "round_robin_over_train_sizes" if method.track == "J" else None
    )
    return (
        model,
        modules,
        {
            "head_config": head_config.to_dict(),
            "losses": summary,
            "initialization_parent": parent_record,
            "joint_counterexample_provenance": counterexample_provenance,
        },
        source_path,
    )


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("component training requires a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    base_method = method_by_name(config, args.method)
    if base_method.reuse_component_from is not None:
        raise ValueError("checkpoint reuse aliases cannot train")
    method = resolve_effective_method(config, lock, base_method)
    if not method.component_checkpoint_required:
        raise ValueError("selected method has no trainable planner component")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in config.protocol.planner_seeds:
        raise ValueError("planner seed lies outside the locked matrix")
    module_names = set(
        required_heads(
            method,
            HeadConfig(latent_dim=1, hidden_dim=1, horizon=method.planner.horizon),
        ).keys()
    )
    locked_steps = locked_training_steps(config, method, module_names)
    steps = args.steps or locked_steps
    if args.steps is not None and args.steps != locked_steps and not args.diagnostic:
        raise ValueError("formal training cannot override the locked step budget")
    if args.diagnostic and (args.steps is None or args.output is None):
        raise ValueError("diagnostic training requires --steps and explicit --output")
    if args.diagnostic and not (0 < int(args.steps) < locked_steps):
        raise ValueError(
            "diagnostic steps must be positive and below the locked budget"
        )
    if args.allow_dirty_worktree and not args.diagnostic:
        raise ValueError("dirty worktrees are allowed only for isolated diagnostics")
    output = resolve_path(
        args.output
        or config.paths.component_training_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    rerun = prepare_formal_output(
        output, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed(
            "planner-component-initialization",
            args.backbone_seed,
            args.planner_seed,
        ),
        deterministic=True,
    )
    model, modules, training_summary, source_path = train_components(
        config,
        lock,
        method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
        device=device,
        steps=steps,
    )
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "stage": (
            "component_training_diagnostic" if args.diagnostic else "component_training"
        ),
        "diagnostic": bool(args.diagnostic),
        "method_name": method.name,
        "track": method.track,
        "seed": args.backbone_seed,
        "backbone_seed": args.backbone_seed,
        "planner_seed": args.planner_seed,
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "training_spec_sha256": training_spec_sha256(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        ),
        "train_manifest_sha256": sha256_file(resolve_path(config.paths.train_manifest)),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "initialization_parent": training_summary["initialization_parent"],
        "joint_counterexample_provenance": training_summary[
            "joint_counterexample_provenance"
        ],
        "head_config": training_summary["head_config"],
        "head_state_dicts": {
            name: module.state_dict() for name, module in modules.items()
        },
        "model_state_dict": model.state_dict() if method.track == "J" else None,
        "training_summary": training_summary["losses"],
        "validation_metrics": {},
        "protocol": protocol_metadata(
            config,
            lock,
            method=method,
            seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            device=device,
        ),
        "rerun": rerun,
    }
    atomic_torch_save(output, payload)


if __name__ == "__main__":
    main()
