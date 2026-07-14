"""Freeze checkpoint hashes, nested seeds, and an opaque confirmatory schedule."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_closure.common import sha256_file
from vector_jepa_planner_frontier import EXPERIMENT_FAMILY, FORMAT_VERSION
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    component_checkpoint_owner,
    component_checkpoint_path,
    load_json,
    load_study_config,
    parent_component_checkpoint_path,
    planner_seed_values,
    require_clean_worktree,
    resolve_path,
    training_spec_sha256,
    uses_counterexample_rounds,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import checkpoint_path
from vector_jepa_planner_frontier.effective_methods import (
    RADICAL_METHODS,
    effective_method_sha256,
    resolve_effective_method,
)
from vector_jepa_planner_frontier.freeze_p7_selection import (
    validate_joint_training_metadata,
)
from vector_jepa_planner_frontier.heads import required_head_names
from vector_jepa_planner_frontier.proposals import RetrievalBank
from vector_jepa_planner_frontier.stage_gates import (
    validate_p2_selection,
    validate_p5_advancement,
    validate_p8_selection,
)
from vector_jepa_planner_frontier.summarize import (
    candidate_mechanism_summary,
    load_result,
    result_path,
)
from vector_jepa_planner_frontier.train import head_step_limits, locked_training_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def _state_dict_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key].detach().cpu(), right[key].detach().cpu()) for key in left
    )


def _validate_training_budget(
    value: dict[str, Any],
    *,
    config: Any,
    method: Any,
    expected_heads: set[str],
    path: Path,
) -> None:
    summary = value.get("training_summary", {})
    if method.stage == "P5":
        if summary != {
            "steps": 0,
            "locked_steps": 0,
            "deterministic_assembly": True,
        }:
            raise ValueError(f"P5 assembly reports an invalid training budget: {path}")
        return
    locked_steps = locked_training_steps(config, method, expected_heads)
    active = set(expected_heads)
    if method.trainable_components is not None:
        active.intersection_update(method.trainable_components)
    random_untrained = (
        method.track == "F"
        and method.control.verifier_targets == "random_untrained"
        and expected_heads == {"verifier"}
    )
    if random_untrained:
        active.discard("verifier")
    limits = head_step_limits(config)
    expected_limits = {
        name: (locked_steps if method.track == "J" else limits[name]) for name in active
    }
    expected_steps = 0 if random_untrained else locked_steps
    if (
        int(summary.get("steps", -1)) != expected_steps
        or int(summary.get("locked_steps", -1)) != locked_steps
        or summary.get("module_step_limits") != expected_limits
        or summary.get("schedule") != "5pct_linear_warmup_cosine_to_10pct"
        or summary.get("random_untrained_control") is not random_untrained
    ):
        raise ValueError(
            f"component did not execute its locked training budget: {path}"
        )


def _mining_fold(task_hash: str) -> int:
    digest = hashlib.sha256(f"vector-jepa-mining-v1:{task_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3 + 1


def _validate_counterexample_chain(
    final_path: Path,
    *,
    config: Any,
    lock: dict[str, Any],
    method: Any,
    backbone_seed: int,
    planner_seed: int,
) -> None:
    """Authenticate all datasets and checkpoints behind a P6 final artifact."""

    expected_training_spec = training_spec_sha256(
        config,
        lock,
        method=method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )
    previous = resolve_path(
        config.paths.component_checkpoint_template.format(
            method=method.name,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
    )
    if not previous.is_file():
        raise FileNotFoundError(previous)
    expected_negative_source = (
        "matched_round_random_actions"
        if method.control.ranker_negatives == "random"
        else "planner_false_optimistic"
    )
    for round_index in range(1, config.training.counterexample_rounds + 1):
        dataset_path = resolve_path(
            config.paths.counterexample_dataset_template.format(
                method=method.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        round_path = resolve_path(
            config.paths.counterexample_round_template.format(
                method=method.name,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                round=round_index,
            )
        )
        if not dataset_path.is_file() or not round_path.is_file():
            raise FileNotFoundError(
                dataset_path if not dataset_path.is_file() else round_path
            )
        dataset = load_json(dataset_path)
        records = dataset.get("records")
        if not isinstance(records, list):
            raise ValueError(f"counterexample records are invalid: {dataset_path}")
        expected_dataset = {
            "schema": "vector-jepa-counterexamples-v1",
            "method": method.name,
            "backbone_seed": backbone_seed,
            "planner_seed": planner_seed,
            "round": round_index,
            "train_manifest_sha256": lock["train_manifest"]["sha256"],
            "source_checkpoint_sha256": sha256_file(previous),
            "diagnostic_limit": 0,
        }
        if any(
            dataset.get(key) != expected for key, expected in expected_dataset.items()
        ):
            raise ValueError(
                f"counterexample dataset provenance mismatch: {dataset_path}"
            )
        if (
            Path(str(dataset.get("source_checkpoint", ""))).resolve()
            != previous.resolve()
        ):
            raise ValueError(
                f"counterexample checkpoint chain is broken: {dataset_path}"
            )
        seen_tasks: set[str] = set()
        for record in records:
            task_hash = str(record.get("task_hash", ""))
            good = tuple(int(action) for action in record.get("good_actions", ()))
            negative = tuple(
                int(action) for action in record.get("false_optimistic_actions", ())
            )
            trigger = tuple(
                int(action) for action in record.get("mining_trigger_actions", ())
            )
            if (
                not task_hash
                or task_hash in seen_tasks
                or _mining_fold(task_hash) != round_index
                or record.get("negative_source") != expected_negative_source
                or record.get("outcome", {}).get("false_optimistic") is not True
                or len(good) != method.planner.horizon
                or len(negative) != method.planner.horizon
                or len(trigger) != method.planner.horizon
                or any(
                    action not in (1, 2, 3, 4) for action in good + negative + trigger
                )
                or good == negative
            ):
                raise ValueError(f"counterexample record is invalid: {dataset_path}")
            seen_tasks.add(task_hash)
        checkpoint = torch.load(round_path, map_location="cpu", weights_only=False)
        protocol = checkpoint.get("protocol", {})
        summary = checkpoint.get("counterexample_training_summary", {})
        if (
            checkpoint.get("experiment_family") != EXPERIMENT_FAMILY
            or int(checkpoint.get("format_version", -1)) != FORMAT_VERSION
            or checkpoint.get("stage") != "counterexample_training_round"
            or checkpoint.get("method_name") != method.name
            or int(checkpoint.get("backbone_seed", -1)) != backbone_seed
            or int(checkpoint.get("planner_seed", -1)) != planner_seed
            or int(checkpoint.get("counterexample_round", -1)) != round_index
            or checkpoint.get("analysis_spec_sha256")
            != analysis_spec_sha256(config, lock)
            or checkpoint.get("training_spec_sha256") != expected_training_spec
            or checkpoint.get("counterexample_dataset_sha256")
            != sha256_file(dataset_path)
            or Path(str(checkpoint.get("counterexample_dataset", ""))).resolve()
            != dataset_path.resolve()
            or int(summary.get("mined_count", -1)) != len(records)
            or int(summary.get("steps", -1))
            != (config.training.counterexample_round_steps if records else 0)
            or protocol.get("git_dirty") is not False
            or protocol.get("code_fingerprint") != lock["code_fingerprint"]
        ):
            raise ValueError(f"counterexample round provenance mismatch: {round_path}")
        previous = round_path
    if previous.resolve() != final_path.resolve():
        raise ValueError("final P6 checkpoint is not the authenticated third round")


def _validate_component(
    path: Path,
    *,
    config: Any,
    lock: dict[str, Any],
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    source_sha256: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_method = resolve_effective_method(
        config,
        lock,
        component_checkpoint_owner(config, method),
    )
    if value.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError(f"component checkpoint belongs to another study: {path}")
    if int(value.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported component checkpoint: {path}")
    expected_stage = (
        "counterexample_training_round"
        if uses_counterexample_rounds(checkpoint_method)
        else "component_calibration"
    )
    if value.get("stage") != expected_stage:
        raise ValueError(f"component checkpoint is not at its final stage: {path}")
    if expected_stage == "counterexample_training_round" and int(
        value.get("counterexample_round", -1)
    ) != int(config.training.counterexample_rounds):
        raise ValueError(f"counterexample checkpoint is not round three: {path}")
    if (
        value.get("method_name") != checkpoint_method.name
        or int(value.get("backbone_seed", -1)) != backbone_seed
        or int(value.get("planner_seed", -1)) != planner_seed
    ):
        raise ValueError(f"component checkpoint label mismatch: {path}")
    if value.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError(f"component analysis-spec mismatch: {path}")
    if value.get("training_spec_sha256") != training_spec_sha256(
        config,
        lock,
        method=checkpoint_method,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    ):
        raise ValueError(f"component training-spec mismatch: {path}")
    if value.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError(f"component uses a different backbone: {path}")
    expected_heads = required_head_names(checkpoint_method)
    checkpoint_heads = value.get("head_state_dicts", {})
    if (
        not isinstance(checkpoint_heads, dict)
        or set(checkpoint_heads) != expected_heads
    ):
        raise ValueError(f"component head set mismatch: {path}")
    _validate_training_budget(
        value,
        config=config,
        method=checkpoint_method,
        expected_heads=expected_heads,
        path=path,
    )
    metrics = value.get("validation_metrics", {})
    if (
        checkpoint_method.memory.hard_pruning
        and metrics.get("hard_pruning_eligible") is not True
    ):
        raise ValueError(f"hard-memory precision gate failed: {path}")
    if (
        checkpoint_method.track == "J"
        and metrics.get("jepa_stability_gate_passed") is not True
    ):
        raise ValueError(f"Track J representation-stability gate failed: {path}")
    if checkpoint_method.track == "J":
        validate_joint_training_metadata(value, config)
    if uses_counterexample_rounds(checkpoint_method):
        _validate_counterexample_chain(
            path,
            config=config,
            lock=lock,
            method=checkpoint_method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
    if checkpoint_method.stage == "P5":
        if value.get("assembly_stage") != "p5_deterministic_assembly":
            raise ValueError(f"P5 checkpoint is not a deterministic assembly: {path}")
        decision = validate_p5_advancement(config, lock)
        source_names = [str(decision["selected_p3_cell"])]
        radical = decision.get("selected_radical")
        if radical is not None:
            source_names.append(RADICAL_METHODS[str(radical)])
        records = value.get("initialization_parents")
        if not isinstance(records, list) or len(records) != len(source_names):
            raise ValueError(f"P5 assembly source count mismatch: {path}")
        expected_states: dict[str, dict[str, torch.Tensor]] = {}
        expected_ownership: dict[str, str] = {}
        for source_name, record in zip(source_names, records, strict=True):
            source_method = resolve_effective_method(
                config,
                lock,
                source_name,
            )
            source_path = component_checkpoint_path(
                config,
                source_method,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
            )
            if source_path is None:
                if required_head_names(source_method) or record != {
                    "method": source_name,
                    "path": None,
                    "sha256": None,
                    "head_names": [],
                }:
                    raise ValueError(f"invalid headless P5 source: {source_name}")
                continue
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_value = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_record = {
                "method": source_name,
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "head_names": sorted(source_value.get("head_state_dicts", {})),
            }
            if record != expected_record:
                raise ValueError(f"P5 assembly source mismatch: {source_name}")
            _validate_component(
                source_path,
                config=config,
                lock=lock,
                method=source_method,
                backbone_seed=backbone_seed,
                planner_seed=planner_seed,
                source_sha256=source_sha256,
            )
            for head_name, state in source_value["head_state_dicts"].items():
                if head_name in expected_heads and head_name not in expected_states:
                    expected_states[head_name] = state
                    expected_ownership[head_name] = source_name
        if value.get("head_ownership") != expected_ownership:
            raise ValueError(f"P5 assembly head ownership mismatch: {path}")
        if set(expected_states) != expected_heads or any(
            not _state_dict_equal(checkpoint_heads[name], expected_states[name])
            for name in expected_heads
        ):
            raise ValueError(f"P5 assembled head tensors changed unexpectedly: {path}")
    else:
        parent_path = parent_component_checkpoint_path(
            config,
            checkpoint_method,
            backbone_seed=backbone_seed,
            planner_seed=planner_seed,
        )
        if parent_path is not None:
            parent_record = value.get("initialization_parent", {})
            if (
                Path(str(parent_record.get("path", ""))) != parent_path
                or not parent_path.is_file()
                or parent_record.get("sha256") != sha256_file(parent_path)
            ):
                raise ValueError(f"component parent checkpoint mismatch: {path}")
    retrieval_record: dict[str, Any] | None = None
    retrieval_path_value = value.get("retrieval_bank_path")
    if checkpoint_method.proposal.retrieval_weight > 0.0:
        retrieval_path = Path(str(retrieval_path_value))
        if not retrieval_path.is_file():
            raise FileNotFoundError(retrieval_path)
        bank = RetrievalBank.load(retrieval_path)
        expected_fingerprint = metrics.get("retrieval_bank_fingerprint")
        if bank.fingerprint != expected_fingerprint:
            raise ValueError(f"retrieval bank differs from calibration: {path}")
        retrieval_record = {
            "path": str(retrieval_path),
            "sha256": sha256_file(retrieval_path),
            "fingerprint": bank.fingerprint,
        }
    elif retrieval_path_value is not None:
        raise ValueError(f"checkpoint has an unexpected retrieval bank: {path}")
    protocol = value.get("protocol", {})
    if protocol.get("git_dirty") is not False:
        raise ValueError(f"component was trained from a dirty worktree: {path}")
    if protocol.get("code_fingerprint") != lock["code_fingerprint"]:
        raise ValueError(f"component code fingerprint mismatch: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_owner": checkpoint_method.name,
        "retrieval_bank": retrieval_record,
    }


def _validate_validation_result(
    *,
    config: Any,
    lock: dict[str, Any],
    method: Any,
    backbone_seed: int,
    planner_seed: int,
    search_seed: int,
    action_selection: str,
) -> None:
    path = result_path(
        config,
        method=method.name,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
        search_seed=search_seed,
        split_role="validation",
        action_selection=action_selection,
    )
    result = load_result(
        path,
        analysis_hash=analysis_spec_sha256(config, lock),
        method=method.name,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
        search_seed=search_seed,
        split_role="validation",
        action_selection=action_selection,
        expected_count=int(lock["validation_manifest"]["count"]),
        expected_manifest_sha256=lock["validation_manifest"]["sha256"],
        expected_code_fingerprint=lock["code_fingerprint"],
        expected_method_sha256=effective_method_sha256(method),
    )
    candidate_mechanism_summary(
        result,
        method=method.name,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
        search_seed=search_seed,
        action_selection=action_selection,
    )


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    validate_p2_selection(config, lock)
    validate_p5_advancement(config, lock)
    p8_selection = validate_p8_selection(config, lock)
    require_clean_worktree(allow_dirty=False)
    outputs = (
        config.paths.confirmation_lock,
        config.paths.confirmation_mapping,
        config.paths.confirmation_schedule,
    )
    if any(resolve_path(path).exists() for path in outputs):
        raise FileExistsError("confirmation freeze artifacts are immutable")
    if resolve_path(config.paths.confirmation_opened).exists():
        raise RuntimeError("confirmatory data was already marked opened")
    power_path = resolve_path(config.paths.confirmation_power)
    power = load_json(power_path)
    analysis_hash = analysis_spec_sha256(config, lock)
    if power.get("analysis_spec_sha256") != analysis_hash:
        raise ValueError("power record belongs to another protocol")
    if power.get("claim_status") != "adequately_powered":
        raise RuntimeError("confirmation cannot freeze before the power gate passes")
    required_backbones = int(power["required_backbones"])
    if required_backbones > len(config.protocol.training_seeds):
        raise RuntimeError("configured backbone pool is smaller than required power")
    backbone_seeds = tuple(config.protocol.training_seeds[:required_backbones])
    comparison_count = int(power.get("comparison_count", -1))
    if comparison_count != int(p8_selection["comparison_count"]):
        raise ValueError("power multiplicity does not match the frozen P8 decision")
    track_f_winner = str(power.get("candidate", ""))
    if track_f_winner != p8_selection["selected_track_f"]:
        raise ValueError("power record does not identify the frozen Track F winner")
    p8_path = resolve_path(config.paths.p8_selection)
    power_p8 = power.get("p8_selection", {})
    if Path(str(power_p8.get("path", ""))) != p8_path or power_p8.get(
        "sha256"
    ) != sha256_file(p8_path):
        raise ValueError("power record references another P8 selection")
    method_names = list(p8_selection["confirmation_methods"])
    track_j_winner = p8_selection["selected_track_j"]
    configured = {method.name: method for method in config.methods}
    methods = [
        resolve_effective_method(config, lock, configured[name])
        for name in method_names
    ]
    if any(not method.confirmatory_eligible for method in methods):
        raise ValueError("confirmatory method family drifted from the protocol")
    primary_family = [
        {
            "hypothesis": "H1",
            "candidate": track_f_winner,
            "baseline": "b0_legacy_l2_cem",
            "subset": subset,
        }
        for subset in ("overall", "ood")
    ]
    if comparison_count == 4:
        primary_family.extend(
            {
                "hypothesis": "H2",
                "candidate": track_j_winner,
                "baseline": track_f_winner,
                "subset": subset,
            }
            for subset in ("overall", "ood")
        )
    if len(primary_family) != comparison_count:
        raise AssertionError("primary family count does not match the power record")

    source_records: dict[str, dict[str, Any]] = {}
    component_records: dict[str, dict[str, Any]] = {}
    for backbone_seed in backbone_seeds:
        source = checkpoint_path(config, seed=int(backbone_seed))
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = sha256_file(source)
        source_records[str(backbone_seed)] = {
            "path": str(source),
            "sha256": source_sha,
        }
        for method in methods:
            for planner_seed in planner_seed_values(config, method):
                for search_seed in config.protocol.search_seeds:
                    for action_selection in ("unmasked", "corrected_v1"):
                        _validate_validation_result(
                            config=config,
                            lock=lock,
                            method=method,
                            backbone_seed=int(backbone_seed),
                            planner_seed=int(planner_seed),
                            search_seed=int(search_seed),
                            action_selection=action_selection,
                        )
                component = component_checkpoint_path(
                    config,
                    method,
                    backbone_seed=int(backbone_seed),
                    planner_seed=int(planner_seed),
                )
                if component is not None:
                    if not component.is_file():
                        raise FileNotFoundError(component)
                    key = f"{method.name}:{backbone_seed}:{planner_seed}"
                    component_records[key] = _validate_component(
                        component,
                        config=config,
                        lock=lock,
                        method=method,
                        backbone_seed=int(backbone_seed),
                        planner_seed=int(planner_seed),
                        source_sha256=source_sha,
                    )

    runs: list[dict[str, Any]] = []
    for method in methods:
        for backbone_seed in backbone_seeds:
            for planner_seed in planner_seed_values(config, method):
                for search_seed in config.protocol.search_seeds:
                    for action_selection in ("unmasked", "corrected_v1"):
                        formal = result_path(
                            config,
                            method=method.name,
                            backbone_seed=int(backbone_seed),
                            planner_seed=int(planner_seed),
                            search_seed=int(search_seed),
                            split_role="confirmatory",
                            action_selection=action_selection,
                        )
                        if formal.exists():
                            raise RuntimeError(
                                "a named confirmatory result already exists before "
                                "freeze"
                            )
                        runs.append(
                            {
                                "method": method.name,
                                "backbone_seed": int(backbone_seed),
                                "planner_seed": int(planner_seed),
                                "search_seed": int(search_seed),
                                "action_selection": action_selection,
                                "formal_output": str(formal),
                            }
                        )
    rng = np.random.default_rng(config.protocol.run_order_seed + 2)
    by_backbone: dict[int, list[dict[str, Any]]] = {
        int(seed): [] for seed in backbone_seeds
    }
    for run in runs:
        by_backbone[int(run["backbone_seed"])].append(run)
    randomized_backbones = list(backbone_seeds)
    rng.shuffle(randomized_backbones)
    blocked_runs: list[dict[str, Any]] = []
    for block_index, backbone_seed in enumerate(randomized_backbones, start=1):
        block = by_backbone[int(backbone_seed)]
        rng.shuffle(block)
        for within_block_order, run in enumerate(block, start=1):
            blocked_runs.append(
                {
                    **run,
                    "block_id": f"B{block_index:03d}",
                    "within_block_order": within_block_order,
                }
            )
    blinded_root = resolve_path(config.paths.run_root) / "confirmatory_blinded"
    mapping_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    for order, run in enumerate(blocked_runs, start=1):
        run_id = f"R{order:06d}"
        opaque_output = blinded_root / f"{run_id}.json"
        if opaque_output.exists():
            raise RuntimeError("opaque confirmatory output already exists")
        mapping_rows.append(
            {
                "run_id": run_id,
                "order": order,
                "opaque_output": str(opaque_output),
                **run,
            }
        )
        schedule_rows.append(
            {
                "run_id": run_id,
                "order": order,
                "block_id": run["block_id"],
                "within_block_order": run["within_block_order"],
            }
        )

    mapping_path = resolve_path(config.paths.confirmation_mapping)
    schedule_path = resolve_path(config.paths.confirmation_schedule)
    atomic_json_dump(
        mapping_path,
        {
            "schema": "vector-jepa-confirmation-mapping-v1",
            "analysis_spec_sha256": analysis_hash,
            "runs": mapping_rows,
        },
    )
    os.chmod(mapping_path, 0o600)
    atomic_json_dump(
        schedule_path,
        {
            "schema": "vector-jepa-confirmation-schedule-v1",
            "analysis_spec_sha256": analysis_hash,
            "runs": schedule_rows,
        },
    )
    confirmation_lock = {
        "schema": "vector-jepa-confirmation-lock-v1",
        "status": "frozen_unopened",
        "analysis_spec_sha256": analysis_hash,
        "base_protocol_lock_sha256": sha256_file(
            resolve_path(config.paths.protocol_lock)
        ),
        "power_record": {
            "path": str(power_path),
            "sha256": sha256_file(power_path),
            "required_backbones": required_backbones,
            "comparison_count": comparison_count,
        },
        "p8_selection": {
            "path": str(p8_path),
            "sha256": sha256_file(p8_path),
        },
        "confirmatory_manifest": lock["confirmatory_manifest"],
        "methods": [method.name for method in methods],
        "track_f_winner": track_f_winner,
        "track_j_included": comparison_count == 4,
        "track_j_winner": track_j_winner,
        "primary_family": primary_family,
        "backbone_seeds": list(backbone_seeds),
        "planner_seeds": list(config.protocol.planner_seeds),
        "search_seeds": list(config.protocol.search_seeds),
        "source_checkpoints": source_records,
        "component_checkpoints": component_records,
        "mapping_sha256": sha256_file(mapping_path),
        "schedule_sha256": sha256_file(schedule_path),
        "run_count": len(mapping_rows),
    }
    atomic_json_dump(config.paths.confirmation_lock, confirmation_lock)


if __name__ == "__main__":
    main()
