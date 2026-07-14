"""Execute one blinded confirmatory run from its opaque schedule identifier."""

from __future__ import annotations

import argparse

from vector_jepa_planner_frontier.common import (
    load_json,
    load_study_config,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.confirmation import confirmation_row
from vector_jepa_planner_frontier.evaluate import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    row = confirmation_row(
        config,
        lock,
        run_id=args.run_id,
        require_opened=True,
    )
    run_evaluation(
        argparse.Namespace(
            config=args.config,
            method=row["method"],
            backbone_seed=int(row["backbone_seed"]),
            planner_seed=int(row["planner_seed"]),
            search_seed=int(row["search_seed"]),
            split_role="confirmatory",
            action_selection=row["action_selection"],
            output=row["opaque_output"],
            component_checkpoint=None,
            device=args.device,
            diagnostic_limit=0,
            allow_dirty_worktree=False,
            overwrite=args.overwrite,
            rerun_reason=args.rerun_reason,
            opaque_run_id=args.run_id,
        )
    )


if __name__ == "__main__":
    main()
