#!/usr/bin/env python3
"""Validate all final runs and generate paper tables, figures, and closure gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from final_closure import EXPERIMENT_FAMILY, FORMAT_VERSION
from final_closure.common import (
    analysis_spec_sha256,
    atomic_json_dump,
    crossed_paired_bootstrap,
    experiment_code_fingerprint,
    git_commit,
    load_checkpoint,
    load_config,
    load_json,
    mean_std,
    read_jsonl,
    require_clean_worktree,
    require_new_output,
    require_study_open,
    sha256_file,
    summarize_rows,
    validate_task_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def format_path(template: str, **values: Any) -> Path:
    return Path(template.format(**values))


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"result mismatch for {label}: {actual!r} != {expected!r}")


def result_checkpoint_path(metadata: dict[str, Any]) -> Path:
    value = (
        metadata.get("checkpoint")
        or metadata.get("planner_ckpt")
        or metadata.get("representation_ckpt")
    )
    return Path(str(value or ""))


def checkpoint_hash_is_valid(metadata: dict[str, Any]) -> None:
    path = result_checkpoint_path(metadata)
    if not path.exists():
        raise ValueError(f"result checkpoint is missing: {path}")
    assert_equal(
        sha256_file(path), metadata.get("checkpoint_sha256"), "checkpoint SHA256"
    )


def validate_baseline_result(
    data: dict[str, Any],
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    baseline: dict[str, Any],
    seed: int,
    split_role: str,
    action_selection: str,
) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    assert_equal(metadata.get("experiment_family"), EXPERIMENT_FAMILY, "family")
    assert_equal(int(metadata.get("format_version", -1)), FORMAT_VERSION, "format")
    assert_equal(metadata.get("protocol_id"), config["protocol_id"], "protocol")
    assert_equal(metadata.get("baseline_name"), baseline["name"], "baseline name")
    assert_equal(metadata.get("baseline_kind"), baseline["kind"], "baseline kind")
    assert_equal(int(metadata.get("training_seed", -1)), seed, "training seed")
    assert_equal(metadata.get("split_role"), split_role, "split role")
    assert_equal(metadata.get("action_selection"), action_selection, "action selection")
    assert_equal(int(metadata.get("max_steps", -1)), 128, "max steps")
    assert_equal(int(metadata.get("seed", -1)), 42, "evaluation seed")
    role_key = f"{split_role}_manifest"
    assert_equal(
        metadata.get("evaluated_manifest_sha256"),
        lock[role_key]["sha256"],
        "evaluated manifest",
    )
    assert_equal(
        int(metadata.get("task_count", -1)), lock[role_key]["count"], "task count"
    )
    assert_equal(
        metadata.get("analysis_spec_sha256"),
        analysis_spec_sha256(config, lock),
        "analysis spec",
    )
    assert_equal(metadata.get("git_dirty"), False, "evaluation dirty flag")
    assert_equal(metadata.get("training_git_dirty"), False, "training dirty flag")
    assert_equal(metadata.get("git_commit"), git_commit(), "evaluation Git commit")
    assert_equal(metadata.get("training_git_commit"), git_commit(), "training commit")
    expected_fingerprint = experiment_code_fingerprint()
    assert_equal(
        metadata.get("code_fingerprint"), expected_fingerprint, "evaluation code"
    )
    assert_equal(
        metadata.get("training_code_fingerprint"), expected_fingerprint, "training code"
    )
    expected_primary = split_role == "confirmatory" and action_selection == "unmasked"
    assert_equal(
        metadata.get("comparable_to_primary"), expected_primary, "primary-comparability"
    )
    assert_equal(
        metadata.get("oracle_action_assistance"),
        action_selection == "corrected",
        "oracle assistance flag",
    )
    assert_equal(metadata.get("formal_evaluation"), True, "formal evaluation flag")
    checkpoint_hash_is_valid(metadata)
    results = data.get("results", {})
    rows = validate_task_rows(results.get("task_rows"), int(lock[role_key]["count"]))
    navigation = results.get("navigation")
    if not isinstance(navigation, dict):
        raise ValueError("baseline result is missing navigation summary")
    if int(navigation.get("overall", {}).get("n", -1)) != len(rows):
        raise ValueError("baseline navigation count does not match task rows")
    return {
        "navigation": navigation,
        "task_rows": rows,
        "compute": results["compute"],
        "metadata": metadata,
    }


def validate_spatial_result(
    data: dict[str, Any],
    *,
    lock: dict[str, Any],
    method: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    source = lock["source_spatial_experiment"]
    assert_equal(
        metadata.get("experiment_family"), source["experiment_family"], "spatial family"
    )
    assert_equal(
        int(metadata.get("format_version", -1)),
        source["format_version"],
        "spatial format",
    )
    assert_equal(
        metadata.get("git_commit"), source["git_commit"], "spatial eval commit"
    )
    assert_equal(
        metadata.get("training_git_commit"),
        source["git_commit"],
        "spatial train commit",
    )
    assert_equal(
        metadata.get("code_fingerprint"),
        source["code_fingerprint"],
        "spatial eval code",
    )
    assert_equal(
        metadata.get("training_code_fingerprint"),
        source["code_fingerprint"],
        "spatial train code",
    )
    assert_equal(metadata.get("git_dirty"), False, "spatial evaluation dirty flag")
    assert_equal(
        metadata.get("training_git_dirty"), False, "spatial training dirty flag"
    )
    assert_equal(int(metadata.get("training_seed", -1)), seed, "spatial seed")
    assert_equal(metadata.get("split_role"), "confirmatory", "spatial split")
    assert_equal(
        metadata.get("action_selection"), "unmasked", "spatial action selection"
    )
    assert_equal(metadata.get("mode"), "learned", "spatial result mode")
    assert_equal(int(metadata.get("max_steps", -1)), 128, "spatial max steps")
    assert_equal(int(metadata.get("seed", -1)), 42, "spatial evaluation seed")
    assert_equal(
        metadata.get("evaluated_manifest_sha256"),
        lock["confirmatory_manifest"]["sha256"],
        "spatial manifest",
    )
    assert_equal(int(metadata.get("task_count", -1)), 900, "spatial task count")
    assert_equal(metadata.get("comparable_to_primary"), True, "spatial comparability")
    if not metadata.get("analysis_spec_sha256"):
        raise ValueError("spatial result is missing its analysis spec")
    assert_equal(
        metadata.get("training_analysis_spec_sha256"),
        metadata.get("analysis_spec_sha256"),
        "spatial train/evaluation analysis spec",
    )
    if not metadata.get("training_experiment_spec_sha256"):
        raise ValueError("spatial result is missing its training spec")
    assert_equal(int(metadata.get("max_per_size", -1)), 0, "spatial max-per-size")
    assert_equal(int(metadata.get("limit", -1)), 0, "spatial task limit")
    assert_equal(
        metadata.get("recompute_every_step"), False, "spatial field recomputation"
    )
    checkpoint_hash_is_valid(metadata)
    results = data.get("results", {})
    primary_key = str(method["primary_iterations"])
    if primary_key not in results:
        raise ValueError(f"spatial result lacks preregistered K={primary_key}")
    canonical_iteration_ids: list[str] | None = None
    for iteration, result in results.items():
        if not isinstance(result, dict):
            raise ValueError(f"spatial K={iteration} result is not an object")
        rows = validate_task_rows(result.get("task_rows"), 900)
        identifiers = sorted(str(row["task_id"]) for row in rows)
        if canonical_iteration_ids is None:
            canonical_iteration_ids = identifiers
        elif identifiers != canonical_iteration_ids:
            raise ValueError("spatial K curves do not evaluate identical task IDs")
        if result.get("navigation") != summarize_rows(rows, 21, 128):
            raise ValueError(f"spatial K={iteration} navigation summary is stale")
    primary = results[primary_key]
    return {
        "navigation": primary["navigation"],
        "task_rows": primary["task_rows"],
        "all_iterations": results,
        "metadata": metadata,
    }


def runtime_signature(metadata: dict[str, Any], field: str) -> tuple[Any, ...]:
    runtime = metadata.get(field, {})
    return (
        runtime.get("python"),
        runtime.get("torch"),
        runtime.get("numpy"),
        runtime.get("cuda_runtime"),
        runtime.get("cudnn"),
        runtime.get("cuda_device_name"),
    )


def validate_protocol_audit(
    data: dict[str, Any],
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    assert_equal(data.get("protocol_id"), config["protocol_id"], "audit protocol")
    assert_equal(data.get("status"), "passed", "audit status")
    assert_equal(
        data.get("analysis_spec_sha256"),
        analysis_spec_sha256(config, lock),
        "audit analysis spec",
    )
    metadata = data.get("metadata", {})
    assert_equal(metadata.get("formal_audit"), True, "formal audit flag")
    assert_equal(metadata.get("git_dirty"), False, "audit dirty flag")
    assert_equal(metadata.get("git_commit"), git_commit(), "audit Git commit")
    assert_equal(
        metadata.get("code_fingerprint"),
        experiment_code_fingerprint(),
        "audit code fingerprint",
    )
    manifests = data.get("manifests", {})
    assert_equal(manifests.get("entries_regenerated"), True, "entry regeneration")
    expected_hashes = {
        role: lock[role]["sha256"]
        for role in (
            "train_manifest",
            "development_manifest",
            "confirmatory_manifest",
        )
    }
    expected_counts = {
        role: int(lock[role]["count"])
        for role in (
            "train_manifest",
            "development_manifest",
            "confirmatory_manifest",
        )
    }
    assert_equal(manifests.get("hashes"), expected_hashes, "audit manifest hashes")
    assert_equal(manifests.get("counts"), expected_counts, "audit manifest counts")
    expected_overlap = {
        "topology_overlap": 0,
        "layout_overlap": 0,
        "task_overlap": 0,
    }
    for pair, overlap in manifests.get("overlaps", {}).items():
        assert_equal(overlap, expected_overlap, f"audit overlap {pair}")
    assert_equal(
        set(manifests.get("overlaps", {})),
        {
            "train_vs_development",
            "train_vs_confirmatory",
            "development_vs_confirmatory",
        },
        "audit overlap pairs",
    )
    assert_equal(
        int(manifests.get("confirmatory_step_cap_failures", -1)),
        int(lock["confirmatory_manifest"]["step_cap_failures"]),
        "audit step-cap failures",
    )
    source = data.get("source_spatial_experiment", {})
    assert_equal(
        source.get("git_commit"),
        lock["source_spatial_experiment"]["git_commit"],
        "audit spatial source commit",
    )
    assert_equal(
        source.get("code_fingerprint"),
        lock["source_spatial_experiment"]["code_fingerprint"],
        "audit spatial source fingerprint",
    )


def require_task_alignment(records: dict[str, list[dict[str, Any]]]) -> None:
    canonical: list[str] | None = None
    for name, runs in records.items():
        if len(runs) != 10:
            raise ValueError(f"{name} requires exactly ten runs")
        for run in runs:
            identifiers = sorted(str(row["task_id"]) for row in run["task_rows"])
            if canonical is None:
                canonical = identifiers
            elif identifiers != canonical:
                raise ValueError(f"{name} does not contain the canonical 900 tasks")


def validate_records_against_manifest(
    records: dict[str, list[dict[str, Any]]],
    entries: list[dict[str, Any]],
) -> None:
    manifest_by_id = {str(entry["task_hash"]): entry for entry in entries}
    expected_ids = sorted(manifest_by_id)
    if len(expected_ids) != len(entries):
        raise ValueError("manifest task hashes must be unique")
    field_pairs = {
        "maze_size": "maze_size",
        "topology_seed": "topology_seed",
        "start_cell": "start_cell",
        "goal_cell": "goal_cell",
        "optimal_length": "bfs_path_length",
    }
    for name, runs in records.items():
        for run in runs:
            rows = run["task_rows"]
            if sorted(str(row["task_id"]) for row in rows) != expected_ids:
                raise ValueError(f"{name} task rows do not match the locked manifest")
            for row in rows:
                entry = manifest_by_id[str(row["task_id"])]
                for row_field, manifest_field in field_pairs.items():
                    if int(row[row_field]) != int(entry[manifest_field]):
                        raise ValueError(
                            f"{name} task {row['task_id']} has mismatched {row_field}"
                        )
            recomputed = summarize_rows(rows, seen_max_size=21, max_steps=128)
            if run["navigation"] != recomputed:
                raise ValueError(f"{name} navigation summary is stale or inconsistent")


def summarize_method(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scope in ("overall", "seen", "ood"):
        summary[scope] = {
            "sr": mean_std(run["navigation"][scope]["sr"] for run in runs),
            "spl": mean_std(run["navigation"][scope]["spl"] for run in runs),
            "n_per_seed": int(runs[0]["navigation"][scope]["n"]),
        }
    sizes = sorted(runs[0]["navigation"]["by_size"], key=int)
    summary["by_size"] = {
        size: {
            "sr": mean_std(run["navigation"]["by_size"][size]["sr"] for run in runs),
            "spl": mean_std(run["navigation"]["by_size"][size]["spl"] for run in runs),
        }
        for size in sizes
    }
    path_bins = list(runs[0]["navigation"]["by_shortest_path"])
    summary["by_shortest_path"] = {
        path_bin: {
            "sr": mean_std(
                run["navigation"]["by_shortest_path"][path_bin]["sr"] for run in runs
            ),
            "spl": mean_std(
                run["navigation"]["by_shortest_path"][path_bin]["spl"] for run in runs
            ),
            "n_per_seed": int(runs[0]["navigation"]["by_shortest_path"][path_bin]["n"]),
        }
        for path_bin in path_bins
    }
    summary["failure_diagnostics"] = {
        "invalid_rate": mean_std(
            run["navigation"]["overall"]["invalid_rate"] for run in runs
        ),
        "loop_or_cycle_rate": mean_std(
            run["navigation"]["overall"]["loop_or_cycle_rate"] for run in runs
        ),
    }
    return summary


def _one_int(values: list[Any], label: str) -> int:
    converted = {int(value) for value in values if value is not None}
    if len(converted) != 1:
        raise ValueError(f"{label} must contain one non-missing value")
    return next(iter(converted))


def summarize_compute(
    name: str,
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = [run["metadata"] for run in runs]
    if metadata[0].get("baseline_name"):
        parameter_count = _one_int(
            [item.get("parameter_count") for item in metadata],
            f"{name} parameter count",
        )
        wallclock = mean_std(
            run["compute"]["wallclock_seconds"] / run["compute"]["task_count"]
            for run in runs
        )
        decisions = mean_std(
            run["compute"]["decision_count"] / run["compute"]["task_count"]
            for run in runs
        )
        result: dict[str, Any] = {
            "parameter_count": parameter_count,
            "wallclock_seconds_per_task": wallclock,
            "decisions_per_task": decisions,
            "wallclock_scope": "runtime-specific descriptive measurement",
        }
        if metadata[0]["baseline_kind"] == "bc":
            mac_maps = [run["compute"]["forward_macs_by_maze_size"] for run in runs]
            if any(value != mac_maps[0] for value in mac_maps[1:]):
                raise ValueError("BC forward MAC maps differ across seeds")
            result["policy_forward_macs_by_maze_size"] = mac_maps[0]
        else:
            result["cem_candidate_transition_predictions_per_task"] = mean_std(
                run["compute"]["auxiliary_totals"].get(
                    "cem_candidate_transition_predictions", 0.0
                )
                / run["compute"]["task_count"]
                for run in runs
            )
            result["fallback_transition_predictions_per_task"] = mean_std(
                run["compute"]["auxiliary_totals"].get(
                    "fallback_transition_predictions", 0.0
                )
                / run["compute"]["task_count"]
                for run in runs
            )
        return result
    method = next(item for item in config["spatial_methods"] if item["name"] == name)
    primary_k = str(method["primary_iterations"])
    planner_macs = _one_int(
        [item["planner_inference_conv_macs"]["25"][primary_k] for item in metadata],
        f"{name} planner MACs",
    )
    representation_macs = _one_int(
        [item["representation_inference_conv_macs"]["25"] for item in metadata],
        f"{name} representation MACs",
    )
    return {
        "parameter_count": _one_int(
            [item.get("total_inference_parameter_count") for item in metadata],
            f"{name} total parameter count",
        ),
        "planner_parameter_count": _one_int(
            [item.get("planner_parameter_count") for item in metadata],
            f"{name} planner parameter count",
        ),
        "representation_parameter_count": _one_int(
            [item.get("representation_planning_parameter_count") for item in metadata],
            f"{name} representation parameter count",
        ),
        "primary_iterations": int(primary_k),
        "conv_macs_at_size25": planner_macs + representation_macs,
        "planner_conv_macs_at_size25": planner_macs,
        "representation_conv_macs_at_size25": representation_macs,
    }


def interval_status(effect: dict[str, Any]) -> str:
    if float(effect["ci_low"]) > 0:
        return "positive_interval"
    if float(effect["ci_high"]) < 0:
        return "negative_interval"
    return "interval_overlaps_zero"


def paired_effects(
    records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    comparisons = analysis["comparisons"]
    simultaneous_alpha = float(analysis["familywise_alpha"]) / len(comparisons)
    samples = int(analysis["bootstrap_samples"])
    base_seed = int(analysis["bootstrap_seed"])
    output: dict[str, Any] = {}
    for index, comparison in enumerate(comparisons):
        candidate = comparison["candidate"]
        baseline = comparison["baseline"]
        candidate_rows = [run["task_rows"] for run in records[candidate]]
        baseline_rows = [run["task_rows"] for run in records[baseline]]
        success_simultaneous = crossed_paired_bootstrap(
            candidate_rows,
            baseline_rows,
            metric="success",
            samples=samples,
            alpha=simultaneous_alpha,
            seed=base_seed + index * 100,
            pair_seeds=False,
            task_strata_key="maze_size",
        )
        success_95 = crossed_paired_bootstrap(
            candidate_rows,
            baseline_rows,
            metric="success",
            samples=samples,
            alpha=0.05,
            seed=base_seed + index * 100 + 1,
            pair_seeds=False,
            task_strata_key="maze_size",
        )
        spl_95 = crossed_paired_bootstrap(
            candidate_rows,
            baseline_rows,
            metric="spl",
            samples=samples,
            alpha=0.05,
            seed=base_seed + index * 100 + 2,
            pair_seeds=False,
            task_strata_key="maze_size",
        )
        key = f"{candidate}__minus__{baseline}"
        output[key] = {
            **comparison,
            "claim_role": "secondary_fixed_addendum",
            "success_simultaneous": success_simultaneous,
            "success_95": success_95,
            "spl_95": spl_95,
            "success_interval_status": interval_status(success_simultaneous),
        }
    return output


def assistance_effects(
    primary: dict[str, list[dict[str, Any]]],
    corrected: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["analysis"]["bootstrap_seed"]) + 50_000
    output: dict[str, Any] = {}
    for index, baseline in enumerate(config["baselines"]):
        name = baseline["name"]
        primary_rows = [run["task_rows"] for run in primary[name]]
        corrected_rows = [run["task_rows"] for run in corrected[name]]
        assistance_rates = []
        for run in corrected[name]:
            decisions = sum(int(row["path_length"]) for row in run["task_rows"])
            assisted = sum(
                float(row.get("auxiliary", {}).get("assisted_action", 0.0))
                for row in run["task_rows"]
            )
            assistance_rates.append(assisted / max(decisions, 1))
        output[name] = {
            "corrected_minus_unmasked_sr": crossed_paired_bootstrap(
                corrected_rows,
                primary_rows,
                metric="success",
                samples=samples,
                alpha=0.05,
                seed=seed + index * 10,
                task_strata_key="maze_size",
            ),
            "corrected_minus_unmasked_spl": crossed_paired_bootstrap(
                corrected_rows,
                primary_rows,
                metric="spl",
                samples=samples,
                alpha=0.05,
                seed=seed + index * 10 + 1,
                task_strata_key="maze_size",
            ),
            "assisted_decision_rate": mean_std(assistance_rates),
            "interpretation": "oracle-assisted diagnostic; excluded from primary table",
        }
    return output


def spatial_k_curves(
    spatial_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for name, records in spatial_records.items():
        keys = sorted(
            set.intersection(*(set(record["all_iterations"]) for record in records)),
            key=int,
        )
        curves[name] = {}
        for key in keys:
            planner_macs = _one_int(
                [
                    record["metadata"]["planner_inference_conv_macs"]["25"][key]
                    for record in records
                ],
                f"{name} K={key} planner MACs",
            )
            representation_macs = _one_int(
                [
                    record["metadata"]["representation_inference_conv_macs"]["25"]
                    for record in records
                ],
                f"{name} representation MACs",
            )
            curves[name][key] = {
                "sr": mean_std(
                    record["all_iterations"][key]["navigation"]["overall"]["sr"]
                    for record in records
                ),
                "spl": mean_std(
                    record["all_iterations"][key]["navigation"]["overall"]["spl"]
                    for record in records
                ),
                "conv_macs_size25": planner_macs + representation_macs,
                "planner_conv_macs_size25": planner_macs,
                "representation_conv_macs_size25": representation_macs,
            }
    return curves


def development_alignment(
    development: dict[str, dict[str, list[dict[str, Any]]]],
    lock: dict[str, Any],
) -> dict[str, Any]:
    anchors = lock["legacy_development_anchors"]
    output: dict[str, Any] = {}
    for name, protocols in development.items():
        corrected_sr = mean_std(
            run["navigation"]["overall"]["sr"] for run in protocols["corrected"]
        )
        unmasked_sr = mean_std(
            run["navigation"]["overall"]["sr"] for run in protocols["unmasked"]
        )
        anchor_key = (
            "bc_deepcnn_sr" if name == "bc_deepcnn_fixed" else "lewm_l2_cem_seqlen2_sr"
        )
        entry: dict[str, Any] = {
            "unmasked_sr": unmasked_sr,
            "corrected_sr": corrected_sr,
            "legacy_corrected_sr_anchor": float(anchors[anchor_key]),
            "corrected_sr_minus_legacy_anchor": float(
                corrected_sr["mean"] - float(anchors[anchor_key])
            ),
            "used_for_model_selection": False,
        }
        if name == "lewm_l2_cem_seqlen2":
            corrected_spl = mean_std(
                run["navigation"]["overall"]["spl"] for run in protocols["corrected"]
            )
            entry.update(
                {
                    "corrected_spl": corrected_spl,
                    "legacy_corrected_spl_anchor": float(
                        anchors["lewm_l2_cem_seqlen2_spl"]
                    ),
                    "corrected_spl_minus_legacy_anchor": float(
                        corrected_spl["mean"]
                        - float(anchors["lewm_l2_cem_seqlen2_spl"])
                    ),
                }
            )
        output[name] = entry
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_tables(summary: dict[str, Any], table_dir: Path) -> list[Path]:
    primary_rows = []
    for name, method in summary["methods"].items():
        primary_rows.append(
            {
                "method": name,
                "sr_mean": method["overall"]["sr"]["mean"],
                "sr_sd": method["overall"]["sr"]["std"],
                "spl_mean": method["overall"]["spl"]["mean"],
                "spl_sd": method["overall"]["spl"]["std"],
                "seen_sr_mean": method["seen"]["sr"]["mean"],
                "ood_sr_mean": method["ood"]["sr"]["mean"],
                "invalid_rate_mean": method["failure_diagnostics"]["invalid_rate"][
                    "mean"
                ],
                "loop_rate_mean": method["failure_diagnostics"]["loop_or_cycle_rate"][
                    "mean"
                ],
                "seeds": method["overall"]["sr"]["n"],
            }
        )
    effect_rows = []
    for key, effect in summary["paired_effects"].items():
        effect_rows.append(
            {
                "comparison": key,
                "delta_sr": effect["success_simultaneous"]["delta"],
                "simultaneous_ci_low": effect["success_simultaneous"]["ci_low"],
                "simultaneous_ci_high": effect["success_simultaneous"]["ci_high"],
                "simultaneous_alpha": effect["success_simultaneous"]["alpha"],
                "delta_spl": effect["spl_95"]["delta"],
                "spl_95_ci_low": effect["spl_95"]["ci_low"],
                "spl_95_ci_high": effect["spl_95"]["ci_high"],
                "status": effect["success_interval_status"],
                "claim_role": effect["claim_role"],
            }
        )
    path_rows = []
    for name, method in summary["methods"].items():
        for path_bin, values in method["by_shortest_path"].items():
            path_rows.append(
                {
                    "method": name,
                    "shortest_path_bin": path_bin,
                    "n_per_seed": values["n_per_seed"],
                    "sr_mean": values["sr"]["mean"],
                    "sr_sd": values["sr"]["std"],
                    "spl_mean": values["spl"]["mean"],
                    "spl_sd": values["spl"]["std"],
                }
            )
    curve_rows = []
    for name, curve in summary["spatial_k_curves"].items():
        for iterations, values in curve.items():
            curve_rows.append(
                {
                    "method": name,
                    "iterations": int(iterations),
                    "sr_mean": values["sr"]["mean"],
                    "sr_sd": values["sr"]["std"],
                    "spl_mean": values["spl"]["mean"],
                    "spl_sd": values["spl"]["std"],
                    "conv_gmac_size25": values["conv_macs_size25"] / 1e9,
                }
            )
    compute_rows = []
    for name, method in summary["methods"].items():
        compute = method["compute"]
        compute_rows.append(
            {
                "method": name,
                "parameter_count": compute["parameter_count"],
                "primary_iterations": compute.get("primary_iterations", ""),
                "conv_gmac_size25": (
                    compute["conv_macs_at_size25"] / 1e9
                    if "conv_macs_at_size25" in compute
                    else ""
                ),
                "policy_forward_macs_by_size": json.dumps(
                    compute.get("policy_forward_macs_by_maze_size", {}),
                    sort_keys=True,
                ),
                "wallclock_seconds_per_task_mean": compute.get(
                    "wallclock_seconds_per_task", {}
                ).get("mean", ""),
                "decisions_per_task_mean": compute.get("decisions_per_task", {}).get(
                    "mean", ""
                ),
                "cem_predictions_per_task_mean": compute.get(
                    "cem_candidate_transition_predictions_per_task", {}
                ).get("mean", ""),
            }
        )
    outputs = [
        table_dir / "primary_results.csv",
        table_dir / "paired_effects.csv",
        table_dir / "path_length_generalization.csv",
        table_dir / "spatial_k_curves.csv",
        table_dir / "compute_summary.csv",
    ]
    for path, rows in zip(
        outputs,
        (primary_rows, effect_rows, path_rows, curve_rows, compute_rows),
        strict=True,
    ):
        write_csv(path, rows)
    return outputs


def stat_text(value: dict[str, Any]) -> str:
    return f"{value['mean']:.3f} +/- {value['std']:.3f}"


def effect_text(value: dict[str, Any]) -> str:
    return f"{value['delta']:.3f} [{value['ci_low']:.3f}, {value['ci_high']:.3f}]"


def compute_text(value: dict[str, Any]) -> str:
    if "conv_macs_at_size25" in value:
        return f"{value['conv_macs_at_size25'] / 1e9:.3f} conv GMAC at size 25"
    if "policy_forward_macs_by_maze_size" in value:
        macs = value["policy_forward_macs_by_maze_size"]["25"]
        return f"{macs / 1e9:.3f} GMAC/decision at size 25"
    predictions = value["cem_candidate_transition_predictions_per_task"]
    return f"{stat_text(predictions)} predictor transitions/task"


def paper_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Maze-JEPA Final Paper Closure Results",
        "",
        "## Analysis status",
        "",
        (
            "The original Spatial-JEPA confirmatory outcomes were observed before "
            "the two baselines in this addendum were run. Therefore, all new "
            "cross-family comparisons below are fixed secondary analyses, not newly "
            "preregistered confirmatory hypotheses. No confirmatory task was used "
            "for checkpoint selection, hyperparameter selection, or rerun decisions."
        ),
        (
            "These are system-level capability comparisons. Training supervision, "
            "examples seen, and optimization compute are intentionally not equalized, "
            "so differences do not isolate representation learning or sample "
            "efficiency."
        ),
        "",
        "## Primary unmasked results",
        "",
        "| Method | SR | SPL | Seen SR | OOD SR | Invalid rate | Loop/cycle rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in summary["methods"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                name,
                stat_text(method["overall"]["sr"]),
                stat_text(method["overall"]["spl"]),
                stat_text(method["seen"]["sr"]),
                stat_text(method["ood"]["sr"]),
                stat_text(method["failure_diagnostics"]["invalid_rate"]),
                stat_text(method["failure_diagnostics"]["loop_or_cycle_rate"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "Primary-table +/- values and figure error bars are sample standard "
                "deviations across the ten independent training runs."
            ),
            (
                "SR is the sole multiplicity-controlled endpoint. Its four fixed "
                "comparisons use Bonferroni simultaneous percentile intervals at "
                "familywise alpha 0.05. SPL intervals are secondary 95% descriptive "
                "intervals."
            ),
            (
                "Cross-method intervals independently resample each method's training "
                "seeds and jointly resample matched task hashes within maze-size "
                "strata. Action-protocol diagnostics pair both seed and task because "
                "they share checkpoints."
            ),
            "",
            "## Fixed paired comparisons",
            "",
            (
                "| Comparison | Delta SR (simultaneous CI) | Delta SPL (95% CI) "
                "| Interval status |"
            ),
            "|---|---:|---:|---|",
        ]
    )
    for effect in summary["paired_effects"].values():
        lines.append(
            f"| {effect['label']} | {effect_text(effect['success_simultaneous'])} | "
            f"{effect_text(effect['spl_95'])} | `{effect['success_interval_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Oracle-assistance diagnostic",
            "",
            (
                "`corrected` uses true wall validity and immediate-backtracking "
                "information. It is reported only to quantify how much the historical "
                "evaluation protocol helped each baseline."
            ),
            "",
            (
                "| Method | Corrected minus unmasked SR | Corrected minus unmasked "
                "SPL | Assisted decisions |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in summary["assistance_effects"].items():
        lines.append(
            f"| `{name}` | {effect_text(value['corrected_minus_unmasked_sr'])} | "
            f"{effect_text(value['corrected_minus_unmasked_spl'])} | "
            f"{stat_text(value['assisted_decision_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Development compatibility check",
            "",
            (
                "These rows compare the fixed reimplementations with historical "
                "development-set anchors. They are an implementation-alignment "
                "diagnostic and never select a model."
            ),
            "",
            (
                "| Method | Development unmasked SR | Development corrected SR "
                "| Legacy corrected SR | Difference |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in summary["development_alignment"].items():
        lines.append(
            f"| `{name}` | {stat_text(value['unmasked_sr'])} | "
            f"{stat_text(value['corrected_sr'])} | "
            f"{value['legacy_corrected_sr_anchor']:.3f} | "
            f"{value['corrected_sr_minus_legacy_anchor']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Compute accounting",
            "",
            "| Method | Parameters | Locked primary compute |",
            "|---|---:|---|",
        ]
    )
    for name, method in summary["methods"].items():
        compute = method["compute"]
        lines.append(
            f"| `{name}` | {compute['parameter_count']:,} | {compute_text(compute)} |"
        )
    lines.extend(
        [
            "",
            "## Generalization and computation artifacts",
            "",
            (
                "The machine-readable tables include per-size results, seen/OOD "
                "splits, shortest-path bins, failure diagnostics, and the complete "
                "R4/J1 iteration curves. Cross-family wall-clock values are "
                "descriptive because BC and CEM execute different primitive "
                "operations; they are not presented as hardware-independent FLOP "
                "equivalence."
            ),
            "",
            "## Closure decision",
            "",
            (
                "This experiment family is closed regardless of score direction. "
                "The allowed reasons to rerun are missing tasks, duplicate tasks, "
                "manifest/checkpoint/code hash mismatch, non-finite output, or "
                "interrupted execution. A low or surprising score is not a rerun "
                "criterion and does not reopen architecture search."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config, lock = load_config(args.config)
    require_study_open(config)
    require_clean_worktree(args.allow_dirty_worktree)
    summary_path = Path(config["paths"]["summary_json"])
    report_path = Path(config["paths"]["paper_report"])
    gate_path = Path(config["paths"]["closure_gate"])
    require_new_output(summary_path, args.overwrite)
    require_new_output(report_path, args.overwrite)
    if not args.skip_figures:
        require_new_output(gate_path, args.overwrite)
    table_dir = Path(config["paths"]["table_dir"])
    for name in (
        "primary_results.csv",
        "paired_effects.csv",
        "path_length_generalization.csv",
        "spatial_k_curves.csv",
        "compute_summary.csv",
    ):
        require_new_output(table_dir / name, args.overwrite)
    if not args.skip_figures:
        figure_dir = Path(config["paths"]["figure_dir"])
        for name in (
            "primary_results.png",
            "generalization_by_size.png",
            "generalization_by_path_length.png",
            "spatial_iteration_curve.png",
            "spatial_compute_curve.png",
        ):
            require_new_output(figure_dir / name, args.overwrite)
    seeds = [int(value) for value in config["seeds"]]
    confirmatory_entries = read_jsonl(config["paths"]["confirmatory_manifest"])
    development_entries = read_jsonl(config["paths"]["development_manifest"])
    audit_path = Path(config["paths"]["audit_output"])
    if not audit_path.exists():
        raise FileNotFoundError(f"missing required formal protocol audit: {audit_path}")
    validate_protocol_audit(load_json(audit_path), config=config, lock=lock)
    source_files: list[Path] = [audit_path]
    primary_records: dict[str, list[dict[str, Any]]] = {}
    corrected_records: dict[str, list[dict[str, Any]]] = {}
    development_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    spatial_records: dict[str, list[dict[str, Any]]] = {}
    source_analysis_specs: set[str] = set()
    for method in config["spatial_methods"]:
        name = method["name"]
        records = []
        checkpoint_hashes: set[str] = set()
        training_runtime_signatures: set[tuple[Any, ...]] = set()
        evaluation_runtime_signatures: set[tuple[Any, ...]] = set()
        for seed in seeds:
            path = format_path(
                config["paths"]["spatial_result_template"], name=name, seed=seed
            )
            if not path.exists():
                raise FileNotFoundError(f"missing locked spatial result: {path}")
            data = load_json(path)
            record = validate_spatial_result(data, lock=lock, method=method, seed=seed)
            records.append(record)
            source_files.append(path)
            source_checkpoint = result_checkpoint_path(data["metadata"])
            source_files.append(source_checkpoint)
            checkpoint_hashes.add(sha256_file(source_checkpoint))
            training_runtime_signatures.add(
                runtime_signature(data["metadata"], "training_runtime")
            )
            evaluation_runtime_signatures.add(
                runtime_signature(data["metadata"], "runtime")
            )
            source_analysis_specs.add(str(data["metadata"].get("analysis_spec_sha256")))
        if len(checkpoint_hashes) != len(seeds):
            raise ValueError(f"{name} reused a checkpoint across training seeds")
        if len(training_runtime_signatures) != 1:
            raise ValueError(f"{name} training runtimes differ across seeds")
        if len(evaluation_runtime_signatures) != 1:
            raise ValueError(f"{name} evaluation runtimes differ across seeds")
        spatial_records[name] = records
        primary_records[name] = records
    if len(source_analysis_specs) != 1 or "None" in source_analysis_specs:
        raise ValueError("imported spatial results do not share one analysis spec")
    for baseline in config["baselines"]:
        name = baseline["name"]
        primary_records[name] = []
        corrected_records[name] = []
        development_records[name] = {"unmasked": [], "corrected": []}
        checkpoint_hashes: set[str] = set()
        training_runtime_signatures: set[tuple[Any, ...]] = set()
        evaluation_runtime_signatures: set[tuple[Any, ...]] = set()
        parameter_counts: set[int] = set()
        for seed in seeds:
            checkpoint_path = format_path(
                config["paths"]["checkpoint_template"], name=name, seed=seed
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"missing baseline checkpoint: {checkpoint_path}"
                )
            checkpoint = load_checkpoint(
                checkpoint_path,
                config=config,
                lock=lock,
                name=name,
                seed=seed,
                strict_provenance=True,
            )
            if checkpoint.get("formal_run") is not True:
                raise ValueError(f"{name} seed={seed} checkpoint is diagnostic")
            assert_equal(
                checkpoint.get("training_config"),
                baseline["train"],
                "checkpoint frozen training config",
            )
            checkpoint_hashes.add(sha256_file(checkpoint_path))
            source_files.append(checkpoint_path)
            for split_role in ("development", "confirmatory"):
                template = config["paths"][f"{split_role}_result_template"]
                for action_selection in ("unmasked", "corrected"):
                    path = format_path(
                        template,
                        name=name,
                        seed=seed,
                        action_selection=action_selection,
                    )
                    if not path.exists():
                        raise FileNotFoundError(f"missing baseline result: {path}")
                    data = load_json(path)
                    record = validate_baseline_result(
                        data,
                        config=config,
                        lock=lock,
                        baseline=baseline,
                        seed=seed,
                        split_role=split_role,
                        action_selection=action_selection,
                    )
                    assert_equal(
                        data["metadata"].get("checkpoint_sha256"),
                        sha256_file(checkpoint_path),
                        "run-plan checkpoint path",
                    )
                    assert_equal(
                        int(data["metadata"].get("parameter_count", -1)),
                        int(checkpoint["parameter_count"]),
                        "checkpoint/result parameter count",
                    )
                    training_runtime_signatures.add(
                        runtime_signature(data["metadata"], "training_runtime")
                    )
                    evaluation_runtime_signatures.add(
                        runtime_signature(data["metadata"], "runtime")
                    )
                    parameter_counts.add(int(data["metadata"]["parameter_count"]))
                    source_files.append(path)
                    if split_role == "confirmatory":
                        target = (
                            primary_records
                            if action_selection == "unmasked"
                            else corrected_records
                        )
                        target[name].append(record)
                    else:
                        development_records[name][action_selection].append(record)
        if len(checkpoint_hashes) != len(seeds):
            raise ValueError(f"{name} reused a checkpoint across training seeds")
        if len(training_runtime_signatures) != 1:
            raise ValueError(f"{name} training runtimes differ across seeds")
        if len(evaluation_runtime_signatures) != 1:
            raise ValueError(f"{name} evaluation runtimes differ across result files")
        if len(parameter_counts) != 1:
            raise ValueError(f"{name} parameter counts differ across seeds")
    require_task_alignment(primary_records)
    require_task_alignment(corrected_records)
    validate_records_against_manifest(primary_records, confirmatory_entries)
    validate_records_against_manifest(corrected_records, confirmatory_entries)
    require_task_alignment(
        {name: protocols["unmasked"] for name, protocols in development_records.items()}
    )
    validate_records_against_manifest(
        {
            name: protocols["unmasked"]
            for name, protocols in development_records.items()
        },
        development_entries,
    )
    validate_records_against_manifest(
        {
            name: protocols["corrected"]
            for name, protocols in development_records.items()
        },
        development_entries,
    )
    require_task_alignment(
        {
            name: protocols["corrected"]
            for name, protocols in development_records.items()
        }
    )
    for protocols in development_records.values():
        first = sorted(row["task_id"] for row in protocols["unmasked"][0]["task_rows"])
        second = sorted(
            row["task_id"] for row in protocols["corrected"][0]["task_rows"]
        )
        if first != second:
            raise ValueError("development action protocols use different tasks")
    methods = {
        name: summarize_method(records) for name, records in primary_records.items()
    }
    for name, records in primary_records.items():
        methods[name]["compute"] = summarize_compute(name, records, config)
    summary: dict[str, Any] = {
        "protocol": {
            "protocol_id": config["protocol_id"],
            "study_role": config["study_role"],
            "analysis_spec_sha256": analysis_spec_sha256(config, lock),
            "source_spatial_analysis_spec_sha256": next(iter(source_analysis_specs)),
            "seeds": seeds,
            "task_count_per_seed": 900,
            "primary_action_selection": "unmasked",
            "new_cross_family_claim_role": "secondary_fixed_addendum",
            "confirmatory_model_selection": False,
            "score_triggered_reruns": False,
        },
        "methods": methods,
        "paired_effects": paired_effects(primary_records, config),
        "assistance_effects": assistance_effects(
            primary_records, corrected_records, config
        ),
        "development_alignment": development_alignment(development_records, lock),
        "spatial_k_curves": spatial_k_curves(spatial_records),
        "source_file_count": len(set(source_files)),
        "closure_status": "complete_after_artifact_generation",
    }
    table_paths = artifact_tables(summary, table_dir)
    figure_paths: list[Path] = []
    if not args.skip_figures:
        from final_closure.plot_results import create_figures

        figure_paths = create_figures(summary, figure_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(paper_report(summary), encoding="utf-8")
    artifacts = [*table_paths, *figure_paths, report_path]
    summary["artifacts"] = {str(path): sha256_file(path) for path in artifacts}
    atomic_json_dump(summary_path, summary)
    if not args.skip_figures:
        gate = {
            "protocol_id": config["protocol_id"],
            "status": "complete",
            "completion_is_score_independent": True,
            "required_training_seeds": seeds,
            "required_tasks_per_seed": 900,
            "required_primary_methods": list(primary_records),
            "source_files": {
                str(path): sha256_file(path) for path in sorted(set(source_files))
            },
            "summary_sha256": sha256_file(summary_path),
            "artifacts": summary["artifacts"],
            "rerun_allowed_only_for": [
                "interrupted_execution",
                "missing_or_duplicate_task",
                "manifest_checkpoint_or_code_hash_mismatch",
                "non_finite_output",
            ],
            "rerun_for_low_or_surprising_score": False,
            "next_architecture_search_authorized": False,
        }
        atomic_json_dump(gate_path, gate)
    print(f"paper closure summary written to {summary_path}")
    if not args.skip_figures:
        print(f"score-independent closure gate written to {gate_path}")


if __name__ == "__main__":
    main()
