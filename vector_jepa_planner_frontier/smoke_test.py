"""Fast engineer-facing audit that needs no trained checkpoint or GPU."""

from __future__ import annotations

import argparse
import json

from vector_jepa_planner_frontier.audit_protocol import audit_config
from vector_jepa_planner_frontier.common import (
    load_study_config,
    planner_seed_values,
    resolve_path,
)
from vector_jepa_planner_frontier.run_plan import (
    blocked_oracle_jobs,
    evaluation_jobs,
    selected_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    return parser.parse_args()


def confirmatory_gate_record(config: object, *, config_path: str) -> dict[str, object]:
    """Validate the confirmatory gate both before and after P8 selection."""

    p8_selection = resolve_path(config.paths.p8_selection)
    if p8_selection.is_file():
        confirmatory = selected_methods(config, "confirmatory")
        jobs = evaluation_jobs(
            config,
            confirmatory,
            split_role="confirmatory",
            config_path=config_path,
        )
        expected = (
            sum(len(planner_seed_values(config, method)) for method in confirmatory)
            * len(config.protocol.training_seeds)
            * len(config.protocol.search_seeds)
            * 2
        )
        if len(jobs) != expected or len(jobs) not in (240, 400):
            raise RuntimeError("confirmatory run matrix is incomplete")
        gate = "frozen_p8_family_validated"
        job_count: int | None = len(jobs)
    else:
        try:
            selected_methods(config, "confirmatory")
        except RuntimeError as error:
            if "frozen P8 selection" not in str(error):
                raise
        else:
            raise RuntimeError("confirmatory scheduling opened before P8 selection")
        gate = "correctly_locked_pending_p8_selection"
        job_count = None
    return {
        "confirmatory_gate": gate,
        "confirmatory_job_count": job_count,
        "expected_after_p8": "240_if_K2_or_400_if_K4",
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    audit = audit_config(args.config)
    confirmation = confirmatory_gate_record(config, config_path=args.config)
    oracle_jobs = blocked_oracle_jobs(
        config,
        split_role="validation",
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "audit": audit,
                "oracle_job_count": len(oracle_jobs),
                **confirmation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
