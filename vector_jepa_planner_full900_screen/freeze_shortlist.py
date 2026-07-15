"""Freeze at most two seed-42 candidates for three-backbone replication."""

from __future__ import annotations

import argparse

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_full900_screen.analysis import (
    delta_sr,
    load_result,
    screen_planner_seed,
    sr,
    stratified_paired_bootstrap,
)
from vector_jepa_planner_full900_screen.common import (
    atomic_json_dump,
    load_config,
    load_json,
    resolve_path,
    result_path,
    validate_lock,
)
from vector_jepa_planner_full900_screen.methods import (
    direct_control_name,
    effective_method,
    validate_q1_selection,
    validate_shortlist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="vector_jepa_planner_full900_screen/configs/default.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_lock(config, lock)
    validate_q1_selection(config, lock)
    output = resolve_path(config.paths.p5_advancement)
    if output.exists():
        raise FileExistsError("shortlist decision is immutable")
    eligible_count = sum(role.advancement_eligible for role in config.method_roles)
    comparison_count = eligible_count * len(config.replication.action_selections) * 2
    if comparison_count != config.gates.bonferroni_comparison_count:
        raise ValueError("locked Bonferroni comparison family has drifted")
    adjusted_alpha = config.analysis.familywise_alpha / comparison_count
    input_sha256s: dict[str, str] = {}

    def load_and_record(method, action: str):
        planner_seed = screen_planner_seed(method)
        value = load_result(
            config,
            lock,
            method=method,
            backbone_seed=42,
            planner_seed=planner_seed,
            action_selection=action,
        )
        path = result_path(
            config,
            method=method.name,
            backbone_seed=42,
            planner_seed=planner_seed,
            action_selection=action,
        )
        input_sha256s[f"{method.name}:b42:p{planner_seed}:{action}"] = sha256_file(path)
        return value

    baseline = method_by_name(config, "b0_legacy_l2_cem")
    b0 = {
        action: load_and_record(baseline, action)
        for action in config.replication.action_selections
    }
    audits = []
    for role in config.method_roles:
        if not role.advancement_eligible:
            continue
        method = effective_method(config, lock, role.name)
        control_name = direct_control_name(config, lock, method.name)
        control = effective_method(config, lock, control_name)
        candidate_results = {
            action: load_and_record(method, action)
            for action in config.replication.action_selections
        }
        control_results = {
            action: load_and_record(control, action)
            for action in config.replication.action_selections
        }
        system = {
            action: stratified_paired_bootstrap(
                candidate_results[action],
                b0[action],
                samples=config.analysis.bootstrap_samples,
                seed=config.analysis.bootstrap_seed + index,
                alpha=adjusted_alpha,
            )
            for index, action in enumerate(config.replication.action_selections)
        }
        mechanism = {
            action: stratified_paired_bootstrap(
                candidate_results[action],
                control_results[action],
                samples=config.analysis.bootstrap_samples,
                seed=config.analysis.bootstrap_seed + 10 + index,
                alpha=adjusted_alpha,
            )
            for index, action in enumerate(config.replication.action_selections)
        }
        q1 = role.phase == "Q1"
        corrected_pass = (
            system["corrected_v1"]["delta"] >= config.gates.system_min_delta_sr
            and system["corrected_v1"]["ci_low"] > 0.0
            and system["unmasked"]["delta"] >= -config.gates.max_protocol_regression_sr
            and delta_sr(
                candidate_results["corrected_v1"],
                b0["corrected_v1"],
                "ood",
            )
            >= -config.gates.max_ood_regression_sr
            and (
                q1
                or (
                    mechanism["corrected_v1"]["delta"]
                    >= config.gates.mechanism_min_delta_sr
                    and mechanism["corrected_v1"]["ci_low"] > 0.0
                )
            )
        )
        unmasked_pass = (
            system["unmasked"]["delta"] >= config.gates.system_min_delta_sr
            and system["unmasked"]["ci_low"] > 0.0
            and system["corrected_v1"]["delta"]
            >= -config.gates.max_protocol_regression_sr
            and delta_sr(candidate_results["unmasked"], b0["unmasked"], "ood")
            >= -config.gates.max_ood_regression_sr
            and (
                q1
                or (
                    mechanism["unmasked"]["delta"]
                    >= config.gates.mechanism_min_delta_sr
                    and mechanism["unmasked"]["ci_low"] > 0.0
                )
            )
        )
        audits.append(
            {
                "method": method.name,
                "phase": role.phase,
                "direct_control": control.name,
                "system": system,
                "mechanism": mechanism,
                "corrected_ood_delta": delta_sr(
                    candidate_results["corrected_v1"],
                    b0["corrected_v1"],
                    "ood",
                ),
                "unmasked_ood_delta": delta_sr(
                    candidate_results["unmasked"], b0["unmasked"], "ood"
                ),
                "corrected_sr": sr(candidate_results["corrected_v1"]),
                "unmasked_sr": sr(candidate_results["unmasked"]),
                "corrected_pass": bool(corrected_pass),
                "unmasked_pass": bool(unmasked_pass),
            }
        )
    corrected = sorted(
        [row for row in audits if row["corrected_pass"]],
        key=lambda row: (
            -row["system"]["corrected_v1"]["delta"],
            -row["corrected_ood_delta"],
            row["method"],
        ),
    )
    unmasked = sorted(
        [row for row in audits if row["unmasked_pass"]],
        key=lambda row: (
            -row["system"]["unmasked"]["delta"],
            -row["unmasked_ood_delta"],
            row["method"],
        ),
    )
    shortlist: list[str] = []
    for pool in (corrected, unmasked):
        if pool and pool[0]["method"] not in shortlist:
            shortlist.append(pool[0]["method"])
    shortlist = shortlist[: config.gates.max_shortlist_size]
    payload = {
        "schema": "vector-jepa-full900-shortlist-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
        "q1_parent_sha256": sha256_file(resolve_path(config.paths.p2_selection)),
        "selection_data": "seed42_full900_development",
        "multiplicity": {
            "method": config.analysis.multiplicity_method,
            "familywise_alpha": config.analysis.familywise_alpha,
            "comparison_count": comparison_count,
            "per_comparison_alpha": adjusted_alpha,
        },
        "shortlist": shortlist,
        "corrected_leader": corrected[0]["method"] if corrected else None,
        "unmasked_leader": unmasked[0]["method"] if unmasked else None,
        "candidate_audits": audits,
        "input_sha256s": dict(sorted(input_sha256s.items())),
        "closed_without_shortlist": not shortlist,
    }
    atomic_json_dump(output, payload)
    validate_shortlist(config, lock)
    print(f"frozen shortlist: {shortlist}")


if __name__ == "__main__":
    main()
