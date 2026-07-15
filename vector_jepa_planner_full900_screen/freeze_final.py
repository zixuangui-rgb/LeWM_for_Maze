"""Freeze one final winner after paired full-900 runs on backbones 42-44."""

from __future__ import annotations

import argparse

from final_closure.common import sha256_file
from vector_jepa_planner_frontier.common import method_by_name
from vector_jepa_planner_full900_screen.analysis import (
    delta_sr,
    load_result,
    mean,
    screen_planner_seed,
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
    component_parity_audits,
    direct_control_name,
    effective_method,
    validate_final_selection,
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
    shortlist_record = validate_shortlist(config, lock)
    shortlist = list(shortlist_record.get("shortlist", []))
    component_parity = component_parity_audits(
        config,
        candidates=shortlist,
        backbone_seeds=config.replication.expansion_backbone_seeds,
        planner_seeds=(config.replication.screen_planner_seeds[0],),
    )
    output = resolve_path(config.paths.p7_selection)
    if output.exists():
        raise FileExistsError("final-winner decision is immutable")
    input_sha256s: dict[str, str] = {}

    def load_and_record(method, seed: int, action: str):
        planner_seed = screen_planner_seed(method)
        value = load_result(
            config,
            lock,
            method=method,
            backbone_seed=seed,
            planner_seed=planner_seed,
            action_selection=action,
        )
        path = result_path(
            config,
            method=method.name,
            backbone_seed=seed,
            planner_seed=planner_seed,
            action_selection=action,
        )
        input_sha256s[f"{method.name}:b{seed}:p{planner_seed}:{action}"] = sha256_file(
            path
        )
        return value

    baseline = method_by_name(config, "b0_legacy_l2_cem")
    audits = []
    for name in shortlist:
        method = effective_method(config, lock, name)
        control = effective_method(
            config, lock, direct_control_name(config, lock, method.name)
        )
        per_seed = []
        for seed in config.replication.expansion_backbone_seeds:
            row: dict[str, object] = {"backbone_seed": seed}
            for action in config.replication.action_selections:
                candidate = load_and_record(method, seed, action)
                b0 = load_and_record(baseline, seed, action)
                control_result = load_and_record(control, seed, action)
                row[f"{action}_system_delta"] = delta_sr(candidate, b0)
                row[f"{action}_mechanism_delta"] = delta_sr(candidate, control_result)
                row[f"{action}_ood_delta"] = delta_sr(candidate, b0, "ood")
            per_seed.append(row)
        corrected_deltas = [float(row["corrected_v1_system_delta"]) for row in per_seed]
        unmasked_deltas = [float(row["unmasked_system_delta"]) for row in per_seed]
        corrected_mechanism = [
            float(row["corrected_v1_mechanism_delta"]) for row in per_seed
        ]
        unmasked_mechanism = [
            float(row["unmasked_mechanism_delta"]) for row in per_seed
        ]
        q1 = method.stage == "P2"
        corrected_pass = (
            mean(corrected_deltas) >= config.gates.system_min_delta_sr
            and sum(value > 0.0 for value in corrected_deltas)
            >= config.gates.required_positive_backbones
            and mean(unmasked_deltas) >= -config.gates.max_protocol_regression_sr
            and mean(float(row["corrected_v1_ood_delta"]) for row in per_seed)
            >= -config.gates.max_ood_regression_sr
            and (q1 or mean(corrected_mechanism) >= config.gates.mechanism_min_delta_sr)
        )
        unmasked_pass = (
            mean(unmasked_deltas) >= config.gates.system_min_delta_sr
            and sum(value > 0.0 for value in unmasked_deltas)
            >= config.gates.required_positive_backbones
            and mean(corrected_deltas) >= -config.gates.max_protocol_regression_sr
            and mean(float(row["unmasked_ood_delta"]) for row in per_seed)
            >= -config.gates.max_ood_regression_sr
            and (q1 or mean(unmasked_mechanism) >= config.gates.mechanism_min_delta_sr)
        )
        audits.append(
            {
                "method": method.name,
                "direct_control": control.name,
                "per_backbone": per_seed,
                "mean_corrected_system_delta": mean(corrected_deltas),
                "mean_unmasked_system_delta": mean(unmasked_deltas),
                "mean_corrected_mechanism_delta": mean(corrected_mechanism),
                "mean_unmasked_mechanism_delta": mean(unmasked_mechanism),
                "corrected_pass": bool(corrected_pass),
                "unmasked_pass": bool(unmasked_pass),
            }
        )
    corrected = sorted(
        [row for row in audits if row["corrected_pass"]],
        key=lambda row: (-row["mean_corrected_system_delta"], row["method"]),
    )
    unmasked = sorted(
        [row for row in audits if row["unmasked_pass"]],
        key=lambda row: (-row["mean_unmasked_system_delta"], row["method"]),
    )
    winner = (
        corrected[0]["method"]
        if corrected
        else (unmasked[0]["method"] if unmasked else None)
    )
    payload = {
        "schema": "vector-jepa-full900-final-winner-v1",
        "protocol_id": config.protocol_id,
        "quick_spec_sha256": lock["quick_spec_sha256"],
        "shortlist_sha256": sha256_file(resolve_path(config.paths.p5_advancement)),
        "selection_data": "backbones42_44_full900_development",
        "primary_selection_track": "corrected_v1",
        "fallback_track": "unmasked",
        "winner": winner,
        "winner_direct_control": (
            direct_control_name(config, lock, winner) if winner is not None else None
        ),
        "candidate_audits": audits,
        "component_parity_audits": component_parity,
        "input_sha256s": dict(sorted(input_sha256s.items())),
        "closed_without_winner": winner is None,
    }
    atomic_json_dump(output, payload)
    validate_final_selection(config, lock)
    print(f"frozen final winner: {winner}")


if __name__ == "__main__":
    main()
