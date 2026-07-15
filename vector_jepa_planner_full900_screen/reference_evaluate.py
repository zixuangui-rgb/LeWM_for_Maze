"""Rerun the untouched final_closure LeWM controller under the new Q0 protocol."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from final_closure.common import (
    atomic_json_dump,
    baseline_config,
    load_checkpoint,
    read_jsonl,
    resolve_device,
    set_seed,
    sha256_file,
    summarize_rows,
)
from final_closure.common import (
    load_config as load_source_config,
)
from final_closure.evaluate import (
    LeWMCEMController,
    aggregate_compute,
    load_model,
    run_episode,
)
from vector_jepa_planner_frontier.compat import validate_source_contract
from vector_jepa_planner_full900_screen.common import (
    load_config,
    load_json,
    require_clean_worktree,
    resolve_path,
    validate_lock,
)


class ActionLoggingController:
    """Transparent wrapper that records the untouched controller's actions."""

    def __init__(self, controller: LeWMCEMController) -> None:
        self.controller = controller
        self.executed_actions: list[int] = []

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.executed_actions = []
        return self.controller.reset(*args, **kwargs)

    def choose(self, *args: Any, **kwargs: Any) -> tuple[int, dict[str, float]]:
        action, metrics = self.controller.choose(*args, **kwargs)
        self.executed_actions.append(int(action))
        return int(action), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    parser.add_argument(
        "--action-selection", choices=("corrected", "unmasked"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    validate_source_contract(config, lock)
    require_clean_worktree()
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError("Q0 reference output is immutable")
    source_config, source_lock = load_source_config(config.paths.source_config)
    baseline = baseline_config(source_config, "lewm_l2_cem_seqlen2")
    checkpoint_path = resolve_path(
        config.paths.checkpoint_template.format(name="lewm_l2_cem_seqlen2", seed=42)
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        config=source_config,
        lock=source_lock,
        name="lewm_l2_cem_seqlen2",
        seed=42,
        strict_provenance=False,
    )
    if checkpoint.get("formal_run") is not True:
        raise ValueError("Q0 reference requires the original formal checkpoint")
    device = resolve_device(args.device or config.device)
    set_seed(config.protocol.evaluation_seed, deterministic=True)
    model = load_model(baseline, checkpoint, device)
    controller = ActionLoggingController(
        LeWMCEMController(
            model,
            baseline["planner"],
            device=device,
            evaluation_seed=config.protocol.evaluation_seed,
            action_selection=args.action_selection,
        )
    )
    manifest_path = resolve_path(config.paths.development_manifest)
    if sha256_file(manifest_path) != lock["development_manifest"]["sha256"]:
        raise ValueError("Q0 reference manifest hash mismatch")
    entries = read_jsonl(manifest_path)
    if len(entries) != 900:
        raise ValueError("Q0 reference must run the complete full-900")
    started = time.perf_counter()
    rows = []
    for index, entry in enumerate(entries):
        row = run_episode(
            entry,
            controller,
            task_index=index,
            max_steps=config.protocol.max_steps,
        )
        row["executed_actions"] = list(controller.executed_actions)
        if len(row["executed_actions"]) != int(row["path_length"]):
            raise RuntimeError("Q0 action log and path length diverged")
        rows.append(row)
    payload = {
        "metadata": {
            "protocol_id": config.protocol_id,
            "quick_spec_sha256": lock["quick_spec_sha256"],
            "code_fingerprint": lock["code_fingerprint"],
            "role": "q0_untouched_final_closure_reference",
            "source_config_sha256": lock["source_baseline"]["config_sha256"],
            "source_lock_sha256": lock["source_baseline"]["lock_sha256"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_seed": 42,
            "evaluation_seed": config.protocol.evaluation_seed,
            "action_selection": args.action_selection,
            "device": str(device),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "results": {
            "navigation": summarize_rows(
                rows,
                seen_max_size=config.protocol.seen_max_size,
                max_steps=config.protocol.max_steps,
            ),
            "task_rows": rows,
            "compute": aggregate_compute(rows),
        },
    }
    if not np.isfinite(payload["metadata"]["elapsed_seconds"]):
        raise FloatingPointError("Q0 reference runtime is non-finite")
    atomic_json_dump(output, payload)


if __name__ == "__main__":
    main()
