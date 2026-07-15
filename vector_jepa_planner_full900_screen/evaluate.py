"""Evaluate one planner on every task in the historical full-900 split."""

from __future__ import annotations

import argparse
import time

import torch

from final_closure.common import (
    read_jsonl,
    sha256_file,
    summarize_rows,
    validate_task_rows,
)
from vector_jepa_planner_frontier.common import (
    hierarchical_seed,
    prepare_formal_output,
    resolve_device,
    set_seed,
    validate_finite_tree,
)
from vector_jepa_planner_frontier.evaluate import build_controller, run_episode
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    component_checkpoint_path,
    load_config,
    load_json,
    metadata,
    method_by_name,
    require_clean_worktree,
    resolve_path,
    result_path,
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
    parser.add_argument(
        "--action-selection",
        choices=("corrected_v1", "unmasked"),
        required=True,
    )
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def _validate_budget(rows: list[dict[str, object]], *, legacy: bool) -> None:
    for row in rows:
        for trace in row["decision_traces"]:  # type: ignore[index]
            plan = int(trace["compute"]["plan_transitions"])
            if plan > 768:
                raise ValueError("planner exceeded the locked 1x transition budget")
            if legacy and plan != 768:
                raise ValueError("historical B0 no longer uses exactly 768 transitions")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    method = effective_method(config, lock, method_by_name(config, args.method))
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside historical seeds 42-51")
    expected_planner_seeds = (
        config.protocol.planner_seeds if method.component_checkpoint_required else (0,)
    )
    if args.planner_seed not in expected_planner_seeds:
        raise ValueError("planner seed is incompatible with this method")
    if args.allow_dirty_worktree:
        raise ValueError("formal full-900 evaluation never permits a dirty worktree")
    output = resolve_path(
        args.output
        or result_path(
            config,
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
            action_selection=args.action_selection,
        )
    )
    require_clean_worktree()
    rerun = prepare_formal_output(
        output, overwrite=args.overwrite, rerun_reason=args.rerun_reason
    )
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed(
            "full900-planner-evaluation",
            args.backbone_seed,
            args.planner_seed,
        ),
        deterministic=True,
    )
    component_path = component_checkpoint_path(
        config,
        method,
        backbone_seed=args.backbone_seed,
        planner_seed=args.planner_seed,
    )
    controller, provenance = build_controller(
        config,
        lock,
        method,
        seed=args.backbone_seed,
        planner_seed=args.planner_seed,
        search_seed=None,
        device=device,
        action_selection=args.action_selection,
        component_checkpoint=component_path,
    )
    manifest_path = resolve_path(config.paths.development_manifest)
    if sha256_file(manifest_path) != lock["development_manifest"]["sha256"]:
        raise ValueError("development full-900 manifest hash mismatch")
    entries = read_jsonl(manifest_path)
    if len(entries) != config.replication.task_count:
        raise ValueError("formal quick-screen evaluation must run all 900 tasks")
    started = time.perf_counter()
    rows = [
        run_episode(
            entry,
            controller,
            task_index=index,
            max_steps=config.protocol.max_steps,
        )
        for index, entry in enumerate(entries)
    ]
    validate_task_rows(rows, config.replication.task_count)
    _validate_budget(rows, legacy=method.name == "b0_legacy_l2_cem")
    payload = {
        "metadata": metadata(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=(
                args.planner_seed if method.component_checkpoint_required else None
            ),
            device=device,
        ),
        "stage": "full900_planner_evaluation",
        "split_role": "development",
        "action_selection": args.action_selection,
        "selection_status": "exploratory_development_not_fresh_confirmation",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "count": len(entries),
        },
        "provenance": provenance,
        "candidate_replay": {
            "enabled": False,
            "reason": (
                "quick screen retains full decision traces without duplicate replay"
            ),
        },
        "rerun": rerun,
        "summary": summarize_rows(rows, seen_max_size=config.protocol.seen_max_size),
        "resources": {
            "evaluation_wall_seconds": float(time.perf_counter() - started),
            "peak_accelerator_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "tasks": rows,
    }
    validate_finite_tree(payload)
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
