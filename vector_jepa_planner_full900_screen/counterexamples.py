"""Run one locked hard- or random-negative ranker refinement round."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import torch

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import (
    atomic_json_dump,
    atomic_torch_save,
    hierarchical_seed,
    resolve_device,
    set_seed,
)
from vector_jepa_planner_frontier.counterexamples import mine_round, train_ranker
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

    source_sha = sha256_file(input_path)
    started = time.perf_counter()
    if dataset_path.exists():
        dataset = load_json(dataset_path)
        if dataset.get("source_checkpoint_sha256") != source_sha:
            raise ValueError(
                "existing counterexample dataset belongs to another checkpoint"
            )
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
