#!/usr/bin/env python3
"""Summarize aligned multi-seed results without selecting K on the test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_jepa_planning.common import (
    experiment_code_fingerprint,
    git_commit,
    require_clean_worktree,
    require_new_output,
    sha256_file,
    strict_json_dump,
)
from spatial_jepa_planning.run_plan import (
    analysis_spec_sha256,
    merged,
    training_spec_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="spatial_jepa_planning/configs/default.json"
    )
    parser.add_argument("--output", default="spatial_jepa_planning_runs/summary.md")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def format_path(template: str, *, name: str, seed: int) -> Path:
    return Path(template.format(name=name, seed=seed))


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def runtime_signature(runtime: dict[str, Any], device: Any) -> tuple[Any, ...]:
    capability = runtime.get("cuda_device_capability")
    return (
        runtime.get("python"),
        runtime.get("numpy"),
        runtime.get("torch"),
        runtime.get("cuda_runtime"),
        runtime.get("cudnn"),
        runtime.get("platform"),
        runtime.get("cuda_device_name"),
        tuple(capability) if capability is not None else None,
        str(device or "").split(":")[0],
    )


def validate_metadata(
    data: dict[str, Any],
    *,
    expected_eval_hash: str,
    max_steps: int,
    expected_task_count: int,
    evaluation_seed: int,
    expected_mode: str,
    expected_action_selection: str,
    require_primary: bool,
    expected_analysis_spec_sha256: str,
    verify_checkpoint_file: bool = False,
) -> None:
    metadata = data.get("metadata", {})
    if metadata.get("eval_manifest_sha256") != expected_eval_hash:
        raise ValueError("result eval manifest hash does not match the protocol lock")
    if int(metadata.get("max_steps", -1)) != max_steps:
        raise ValueError("result max_steps does not match the protocol lock")
    if int(metadata.get("seed", -1)) != evaluation_seed:
        raise ValueError("result evaluation seed does not match the protocol lock")
    if int(metadata.get("task_count", -1)) != expected_task_count:
        raise ValueError("result task count does not match the protocol lock")
    if metadata.get("evaluated_manifest_sha256") != expected_eval_hash:
        raise ValueError("result was not evaluated on the confirmatory manifest")
    if metadata.get("split_role") != "confirmatory":
        raise ValueError("primary summary accepts only confirmatory split results")
    if metadata.get("analysis_spec_sha256") != expected_analysis_spec_sha256:
        raise ValueError("result analysis spec does not match the preregistration")
    if (
        int(metadata.get("max_per_size", -1)) != 0
        or int(metadata.get("limit", -1)) != 0
    ):
        raise ValueError("primary summary requires unsampled full-manifest evaluation")
    if metadata.get("mode") != expected_mode:
        raise ValueError(f"expected mode={expected_mode}, got {metadata.get('mode')}")
    expected_oracle_protocols = {
        "oracle_bfs": "oracle_bfs_shortest_path",
        "oracle_vi": "oracle_vi_with_oracle_validity",
    }
    if (
        expected_mode in expected_oracle_protocols
        and metadata.get("effective_action_protocol")
        != expected_oracle_protocols[expected_mode]
    ):
        raise ValueError(f"incorrect oracle action protocol for {expected_mode}")
    if (
        expected_mode == "learned"
        and metadata.get("action_selection") != expected_action_selection
    ):
        raise ValueError(
            f"expected action selection {expected_action_selection}, got "
            f"{metadata.get('action_selection')}"
        )
    if metadata.get("mode") == "learned" and metadata.get("recompute_every_step"):
        raise ValueError(
            "primary summary expects one static field computation per task"
        )
    if (
        metadata.get("mode") == "decoded_bfs"
        and metadata.get("decoded_action_selection") != "predicted"
    ):
        raise ValueError("decoded-map summary must not use oracle action correction")
    if metadata.get("git_dirty"):
        raise ValueError(
            "formal summary rejects results produced from a dirty worktree"
        )
    if not metadata.get("git_commit"):
        raise ValueError("formal result is missing its evaluation Git commit")
    if metadata.get("git_commit") != git_commit():
        raise ValueError("result was not evaluated from the current Git commit")
    current_code = experiment_code_fingerprint()
    if metadata.get("code_fingerprint") != current_code:
        raise ValueError("evaluation code fingerprint differs from current source")
    if expected_mode in {"learned", "decoded_bfs"}:
        if metadata.get("training_git_dirty") is not False:
            raise ValueError(
                "formal summary rejects checkpoints trained from a dirty worktree"
            )
        if not metadata.get("training_git_commit"):
            raise ValueError("checkpoint is missing its training Git commit")
        if metadata.get("training_git_commit") != metadata.get("git_commit"):
            raise ValueError(
                "checkpoint training and evaluation used different Git commits"
            )
        if (
            metadata.get("training_analysis_spec_sha256")
            != expected_analysis_spec_sha256
        ):
            raise ValueError(
                "checkpoint was not trained under the preregistered analysis spec"
            )
        if metadata.get("training_code_fingerprint") != current_code:
            raise ValueError("training and evaluation code fingerprints differ")
        if not metadata.get("checkpoint_sha256"):
            raise ValueError("result is missing the evaluated checkpoint hash")
        if verify_checkpoint_file:
            checkpoint_path = (
                metadata.get("planner_ckpt")
                if expected_mode == "learned"
                else metadata.get("representation_ckpt")
            )
            if not checkpoint_path or not Path(checkpoint_path).exists():
                raise ValueError(f"evaluated checkpoint is missing: {checkpoint_path}")
            if sha256_file(checkpoint_path) != metadata.get("checkpoint_sha256"):
                raise ValueError(
                    f"evaluated checkpoint hash changed: {checkpoint_path}"
                )
    if require_primary and not metadata.get("comparable_to_primary", False):
        raise ValueError("result is not marked comparable to the primary protocol")
    if not require_primary and metadata.get("comparable_to_primary", False):
        raise ValueError("diagnostic action protocol must not be marked primary")


def validate_task_rows(result: dict[str, Any], expected_task_count: int) -> None:
    rows = result.get("task_rows")
    if not isinstance(rows, list) or len(rows) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} task rows, got "
            f"{len(rows) if isinstance(rows, list) else 'non-list'}"
        )
    task_ids = [row.get("task_id") for row in rows if isinstance(row, dict)]
    if len(task_ids) != expected_task_count or any(item is None for item in task_ids):
        raise ValueError("every task row must be an object with a task_id")
    if len(set(task_ids)) != expected_task_count:
        raise ValueError("task rows must contain unique task IDs")


def crossed_paired_bootstrap(
    candidate_rows: list[list[dict[str, Any]]],
    baseline_rows: list[list[dict[str, Any]]],
    *,
    metric: str,
    samples: int,
    alpha: float = 0.05,
    seed: int = 20260710,
) -> dict[str, float]:
    if len(candidate_rows) != len(baseline_rows) or not candidate_rows:
        raise ValueError("paired bootstrap needs matched non-empty seed lists")
    paired_differences: list[np.ndarray] = []
    common_order: list[str] | None = None
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        candidate_by_id = {row["task_id"]: row for row in candidate}
        baseline_by_id = {row["task_id"]: row for row in baseline}
        if candidate_by_id.keys() != baseline_by_id.keys():
            raise ValueError("paired result files do not contain identical task IDs")
        ordered_ids = sorted(candidate_by_id)
        if common_order is None:
            common_order = ordered_ids
        elif ordered_ids != common_order:
            raise ValueError("all training seeds must evaluate identical task IDs")
        paired_differences.append(
            np.asarray(
                [
                    float(candidate_by_id[key][metric])
                    - float(baseline_by_id[key][metric])
                    for key in ordered_ids
                ],
                dtype=np.float64,
            )
        )
    difference_matrix = np.stack(paired_differences, axis=0)
    observed = float(difference_matrix.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected_seeds = rng.integers(
            0, difference_matrix.shape[0], difference_matrix.shape[0]
        )
        selected_tasks = rng.integers(
            0, difference_matrix.shape[1], difference_matrix.shape[1]
        )
        draws[sample_index] = float(
            difference_matrix[np.ix_(selected_seeds, selected_tasks)].mean()
        )
    return {
        "delta": observed,
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "alpha": float(alpha),
        "bootstrap_samples": samples,
    }


def primary_result(data: dict[str, Any], primary_iterations: int) -> dict[str, Any]:
    key = str(primary_iterations)
    if key not in data.get("results", {}):
        available = sorted(data.get("results", {}))
        raise ValueError(f"primary K={key} missing; available={available}")
    return data["results"][key]


def stat_text(stat: dict[str, Any], digits: int = 3) -> str:
    if stat.get("mean") is None:
        return "NA"
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def effect_text(effect: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{effect['delta']:.{digits}f} "
        f"[{effect['ci_low']:.{digits}f}, {effect['ci_high']:.{digits}f}]"
    )


def main() -> None:
    args = parse_args()
    require_clean_worktree(args.allow_dirty_worktree)
    config = load_json(args.config)
    lock = load_json(config["paths"]["protocol_lock"])
    eval_hash = str(lock["eval_manifest"]["sha256"])
    expected_task_count = int(lock["eval_manifest"]["count"])
    max_steps = int(config["protocol"]["max_steps"])
    evaluation_seed = int(config["evaluation"]["seed"])
    seeds = [int(seed) for seed in config["seeds"]]
    eval_sizes = [
        str(size)
        for size in sorted(
            int(value) for value in lock["eval_manifest"]["counts_by_size"]
        )
    ]
    planner_template = config["paths"]["planner_eval_template"]
    planner_diagnostic_template = config["paths"]["planner_diagnostic_eval_template"]
    representation_template = config["paths"]["representation_eval_template"]
    bootstrap_samples = int(config["evaluation"].get("bootstrap_samples", 20000))
    familywise_alpha = float(config["inference"]["familywise_alpha"])
    confirmatory_variants = [
        variant
        for variant in config["planners"]
        if variant.get("enabled", True)
        and variant.get("hypothesis", {}).get("role") == "confirmatory"
    ]
    simultaneous_alpha = familywise_alpha / len(confirmatory_variants)
    expected_analysis_spec = analysis_spec_sha256(config)

    oracle_bfs_path = Path(config["paths"]["oracle_bfs_output"])
    oracle_vi_path = Path(config["paths"]["oracle_vi_output"])
    if not oracle_bfs_path.exists() or not oracle_vi_path.exists():
        raise ValueError("oracle anchor results are required before summary")
    oracle_bfs = load_json(oracle_bfs_path)
    oracle_vi = load_json(oracle_vi_path)
    validate_metadata(
        oracle_bfs,
        expected_eval_hash=eval_hash,
        max_steps=max_steps,
        expected_task_count=expected_task_count,
        evaluation_seed=evaluation_seed,
        expected_mode="oracle_bfs",
        expected_action_selection="",
        require_primary=True,
        expected_analysis_spec_sha256=expected_analysis_spec,
        verify_checkpoint_file=True,
    )
    validate_metadata(
        oracle_vi,
        expected_eval_hash=eval_hash,
        max_steps=max_steps,
        expected_task_count=expected_task_count,
        evaluation_seed=evaluation_seed,
        expected_mode="oracle_vi",
        expected_action_selection="",
        require_primary=True,
        expected_analysis_spec_sha256=expected_analysis_spec,
        verify_checkpoint_file=True,
    )
    validate_task_rows(oracle_bfs["results"], expected_task_count)
    for result in oracle_vi["results"].values():
        validate_task_rows(result, expected_task_count)
    exact_bfs_sr = float(oracle_bfs["results"]["navigation"]["overall"]["sr"])
    exact_vi_sr = float(oracle_vi["results"]["256"]["navigation"]["overall"]["sr"])
    expected_oracle = float(lock["evaluation"]["expected_exact_oracle_sr"])
    if not np.isclose(exact_bfs_sr, expected_oracle) or not np.isclose(
        exact_vi_sr, expected_oracle
    ):
        raise ValueError(
            "current exact BFS/VI anchors do not reach the locked step-cap ceiling"
        )

    planner_summaries: dict[str, Any] = {}
    primary_rows: dict[str, list[list[dict[str, Any]]]] = {}
    k_curves: dict[str, dict[str, Any]] = {}
    action_protocol_summaries: dict[str, Any] = {}
    planner_source_rep_hashes: dict[str, dict[int, str | None]] = {}
    training_runtime_signatures: set[tuple[Any, ...]] = set()
    evaluation_runtime_signatures: set[tuple[Any, ...]] = set()
    enabled_planners = [
        variant for variant in config["planners"] if variant.get("enabled", True)
    ]
    for variant in enabled_planners:
        name = str(variant["name"])
        primary_k = int(variant["primary_iterations"])
        train_config = dict(config["representation_defaults"])
        train_config.update(config["planner_defaults"])
        train_config.update(variant.get("train", {}))
        train_config["input_mode"] = variant["input_mode"]
        representation_name = (
            str(variant["representation"])
            if variant["input_mode"] == "spatial_jepa"
            else None
        )
        loaded: list[dict[str, Any]] = []
        diagnostics: dict[str, list[dict[str, Any]]] = {
            action: [] for action in config["protocol"]["diagnostic_action_selections"]
        }
        planner_source_rep_hashes[name] = {}
        for seed in seeds:
            path = format_path(planner_template, name=name, seed=seed)
            if not path.exists():
                continue
            data = load_json(path)
            validate_metadata(
                data,
                expected_eval_hash=eval_hash,
                max_steps=max_steps,
                expected_task_count=expected_task_count,
                evaluation_seed=evaluation_seed,
                expected_mode="learned",
                expected_action_selection="unmasked",
                require_primary=True,
                expected_analysis_spec_sha256=expected_analysis_spec,
                verify_checkpoint_file=True,
            )
            for result in data.get("results", {}).values():
                validate_task_rows(result, expected_task_count)
            if int(data["metadata"].get("training_seed", -1)) != seed:
                raise ValueError(f"result/checkpoint seed label mismatch for {name}")
            expected_spec = training_spec_sha256(
                config,
                train_config,
                variant_name=name,
                seed=seed,
                representation_name=representation_name,
            )
            if data["metadata"].get("training_experiment_spec_sha256") != expected_spec:
                raise ValueError(f"training spec mismatch for {name} seed={seed}")
            runtime = data["metadata"].get("runtime", {})
            evaluation_runtime_signatures.add(
                runtime_signature(runtime, data["metadata"].get("device"))
            )
            training_runtime = data["metadata"].get("training_runtime", {})
            training_runtime_signatures.add(
                runtime_signature(
                    training_runtime,
                    data["metadata"].get("training_device"),
                )
            )
            loaded.append(data)
            planner_source_rep_hashes[name][seed] = data["metadata"].get(
                "source_representation_sha256"
            )
            for action_selection in diagnostics:
                diagnostic_path = Path(
                    planner_diagnostic_template.format(
                        name=name,
                        seed=seed,
                        action_selection=action_selection,
                    )
                )
                if not diagnostic_path.exists():
                    continue
                diagnostic = load_json(diagnostic_path)
                validate_metadata(
                    diagnostic,
                    expected_eval_hash=eval_hash,
                    max_steps=max_steps,
                    expected_task_count=expected_task_count,
                    evaluation_seed=evaluation_seed,
                    expected_mode="learned",
                    expected_action_selection=action_selection,
                    require_primary=False,
                    expected_analysis_spec_sha256=expected_analysis_spec,
                    verify_checkpoint_file=True,
                )
                if diagnostic["metadata"].get("checkpoint_sha256") != data[
                    "metadata"
                ].get("checkpoint_sha256"):
                    raise ValueError(
                        "action protocols use different checkpoints for "
                        f"{name} seed={seed}"
                    )
                for result in diagnostic.get("results", {}).values():
                    validate_task_rows(result, expected_task_count)
                diagnostics[action_selection].append(diagnostic)
        if not loaded:
            if not args.allow_incomplete:
                raise ValueError(f"missing all confirmatory results for {name}")
            planner_summaries[name] = {"status": "missing", "primary_k": primary_k}
            continue
        if len(loaded) != len(seeds):
            raise ValueError(f"partial seed set for {name}: {len(loaded)}/{len(seeds)}")
        for action_selection, diagnostic_runs in diagnostics.items():
            if len(diagnostic_runs) != len(seeds):
                raise ValueError(
                    f"partial {action_selection} diagnostic set for {name}: "
                    f"{len(diagnostic_runs)}/{len(seeds)}"
                )
        primary = [primary_result(data, primary_k) for data in loaded]
        planner_parameter_counts = {
            int(data["metadata"].get("planner_parameter_count", -1)) for data in loaded
        }
        representation_parameter_counts = {
            int(data["metadata"].get("representation_planning_parameter_count", -1))
            for data in loaded
        }
        total_parameter_counts = {
            int(data["metadata"].get("total_inference_parameter_count", -1))
            for data in loaded
        }
        planner_primary_macs = {
            int(
                data["metadata"]
                .get("planner_inference_conv_macs", {})
                .get("25", {})
                .get(str(primary_k), -1)
            )
            for data in loaded
        }
        representation_primary_macs = {
            int(
                data["metadata"]
                .get("representation_inference_conv_macs", {})
                .get("25", -1)
            )
            for data in loaded
        }
        if len(planner_parameter_counts) != 1 or -1 in planner_parameter_counts:
            raise ValueError(f"planner parameter counts are missing/mixed for {name}")
        if (
            len(representation_parameter_counts) != 1
            or -1 in representation_parameter_counts
        ):
            raise ValueError(
                f"representation parameter counts are missing/mixed for {name}"
            )
        if len(total_parameter_counts) != 1 or -1 in total_parameter_counts:
            raise ValueError(f"total parameter counts are missing/mixed for {name}")
        if len(planner_primary_macs) != 1 or -1 in planner_primary_macs:
            raise ValueError(f"planner MAC counts are missing/mixed for {name}")
        if len(representation_primary_macs) != 1 or -1 in representation_primary_macs:
            raise ValueError(f"representation MAC counts are missing/mixed for {name}")
        planner_macs = next(iter(planner_primary_macs))
        representation_macs = next(iter(representation_primary_macs))
        planner_parameters = next(iter(planner_parameter_counts))
        representation_parameters = next(iter(representation_parameter_counts))
        total_parameters = next(iter(total_parameter_counts))
        if total_parameters != planner_parameters + representation_parameters:
            raise ValueError(f"inconsistent total parameter count for {name}")
        if variant["input_mode"] == "raw" and (
            representation_parameters != 0 or representation_macs != 0
        ):
            raise ValueError("raw planner unexpectedly reports representation cost")
        if variant["input_mode"] == "spatial_jepa" and (
            representation_parameters <= 0 or representation_macs <= 0
        ):
            raise ValueError("Spatial-JEPA planner is missing representation cost")
        primary_rows[name] = [result["task_rows"] for result in primary]
        planner_summaries[name] = {
            "status": "complete",
            "primary_k": primary_k,
            "planner_parameter_count": planner_parameters,
            "representation_planning_parameter_count": representation_parameters,
            "total_inference_parameter_count": total_parameters,
            "size25_planner_conv_macs": planner_macs,
            "size25_representation_conv_macs": representation_macs,
            "size25_total_conv_macs": planner_macs + representation_macs,
            "sr": mean_std(
                [result["navigation"]["overall"]["sr"] for result in primary]
            ),
            "spl": mean_std(
                [result["navigation"]["overall"]["spl"] for result in primary]
            ),
            "seen_sr": mean_std(
                [result["navigation"]["seen"]["sr"] for result in primary]
            ),
            "ood_sr": mean_std(
                [result["navigation"]["ood"]["sr"] for result in primary]
            ),
            "eligible_sr": mean_std(
                [result["navigation"]["overall"]["eligible_sr"] for result in primary]
            ),
            "invalid_rate": mean_std(
                [result["navigation"]["overall"]["invalid_rate"] for result in primary]
            ),
            "loop_rate": mean_std(
                [
                    result["navigation"]["overall"]["loop_or_cycle_rate"]
                    for result in primary
                ]
            ),
            "path_sr": {
                label: mean_std(
                    [
                        result["navigation"]["by_shortest_path"][label]["sr"]
                        for result in primary
                    ]
                )
                for label in ("001-016", "017-032", "033-064", "065-128", "129+")
            },
            "size_sr": {
                size: mean_std(
                    [result["navigation"]["by_size"][size]["sr"] for result in primary]
                )
                for size in eval_sizes
            },
            "local_top1": mean_std(
                [result["field"]["overall"]["local_top1"] for result in primary]
            ),
            "local_margin": mean_std(
                [result["field"]["overall"]["local_margin"] for result in primary]
            ),
            "value_pearson": mean_std(
                [
                    result["field"]["overall"]["value_pearson"]
                    for result in primary
                    if result["field"]["overall"]["value_pearson"] is not None
                ]
            ),
        }
        action_protocol_summaries[name] = {
            "unmasked": {
                "sr": planner_summaries[name]["sr"],
            }
        }
        for action_selection, diagnostic_runs in diagnostics.items():
            diagnostic_primary = [
                primary_result(data, primary_k) for data in diagnostic_runs
            ]
            diagnostic_rows = [result["task_rows"] for result in diagnostic_primary]
            action_protocol_summaries[name][action_selection] = {
                "sr": mean_std(
                    [
                        result["navigation"]["overall"]["sr"]
                        for result in diagnostic_primary
                    ]
                ),
                "assistance_delta_sr": crossed_paired_bootstrap(
                    diagnostic_rows,
                    primary_rows[name],
                    metric="success",
                    samples=bootstrap_samples,
                    alpha=familywise_alpha,
                ),
            }
        all_keys = sorted(
            set.intersection(*(set(data["results"]) for data in loaded)), key=int
        )
        k_curves[name] = {
            key: {
                "sr": mean_std(
                    [
                        data["results"][key]["navigation"]["overall"]["sr"]
                        for data in loaded
                    ]
                ),
                "ood_sr": mean_std(
                    [data["results"][key]["navigation"]["ood"]["sr"] for data in loaded]
                ),
                "local_top1": mean_std(
                    [
                        data["results"][key]["field"]["overall"]["local_top1"]
                        for data in loaded
                    ]
                ),
            }
            for key in all_keys
        }

    comparisons: dict[str, Any] = {}
    for variant in enabled_planners:
        name = str(variant["name"])
        baseline = variant.get("comparison_baseline")
        if not baseline or name not in primary_rows or baseline not in primary_rows:
            continue
        hypothesis = variant.get("hypothesis", {})
        is_confirmatory = hypothesis.get("role") == "confirmatory"
        comparison_alpha = simultaneous_alpha if is_confirmatory else familywise_alpha
        sr_result = crossed_paired_bootstrap(
            primary_rows[name],
            primary_rows[baseline],
            metric="success",
            samples=bootstrap_samples,
            alpha=comparison_alpha,
        )
        status = "exploratory"
        if is_confirmatory and hypothesis.get("test") == "superiority":
            minimum_effect = float(hypothesis["minimum_effect_sr"])
            status = (
                "supported"
                if sr_result["ci_low"] >= minimum_effect
                else "not_supported"
            )
        elif is_confirmatory and hypothesis.get("test") == "noninferiority":
            margin = float(hypothesis["margin_sr"])
            status = "supported" if sr_result["ci_low"] > -margin else "not_supported"
        comparisons[name] = {
            "baseline": baseline,
            "hypothesis": hypothesis or {"role": "exploratory"},
            "conclusion": status,
            "sr": sr_result,
            "spl": crossed_paired_bootstrap(
                primary_rows[name],
                primary_rows[baseline],
                metric="spl",
                samples=bootstrap_samples,
                alpha=comparison_alpha,
            ),
        }

    representation_summaries: dict[str, Any] = {}
    representation_checkpoint_hashes: dict[str, dict[int, str]] = {}
    for variant in config["representations"]:
        if not variant.get("enabled", True):
            continue
        name = str(variant["name"])
        train_config = merged(config["representation_defaults"], variant)
        representation_checkpoint_hashes[name] = {}
        loaded = []
        for seed in seeds:
            path = format_path(representation_template, name=name, seed=seed)
            if path.exists():
                data = load_json(path)
                validate_metadata(
                    data,
                    expected_eval_hash=eval_hash,
                    max_steps=max_steps,
                    expected_task_count=expected_task_count,
                    evaluation_seed=evaluation_seed,
                    expected_mode="decoded_bfs",
                    expected_action_selection="",
                    require_primary=True,
                    expected_analysis_spec_sha256=expected_analysis_spec,
                    verify_checkpoint_file=True,
                )
                validate_task_rows(data.get("results", {}), expected_task_count)
                if int(data["metadata"].get("training_seed", -1)) != seed:
                    raise ValueError(f"representation seed label mismatch for {name}")
                expected_spec = training_spec_sha256(
                    config,
                    train_config,
                    variant_name=name,
                    seed=seed,
                    representation_name=None,
                )
                if (
                    data["metadata"].get("training_experiment_spec_sha256")
                    != expected_spec
                ):
                    raise ValueError(
                        f"representation training spec mismatch for {name} seed={seed}"
                    )
                representation_checkpoint_hashes[name][seed] = str(
                    data["metadata"]["checkpoint_sha256"]
                )
                runtime = data["metadata"].get("runtime", {})
                evaluation_runtime_signatures.add(
                    runtime_signature(runtime, data["metadata"].get("device"))
                )
                training_runtime = data["metadata"].get("training_runtime", {})
                training_runtime_signatures.add(
                    runtime_signature(
                        training_runtime,
                        data["metadata"].get("training_device"),
                    )
                )
                loaded.append(data["results"])
        if not loaded:
            if not args.allow_incomplete:
                raise ValueError(
                    f"missing all confirmatory representation results for {name}"
                )
            representation_summaries[name] = {"status": "missing"}
            continue
        if len(loaded) != len(seeds):
            raise ValueError(f"partial representation seed set for {name}")
        representation_summaries[name] = {
            "status": "complete",
            "sr": mean_std(
                [result["navigation"]["overall"]["sr"] for result in loaded]
            ),
            "spl": mean_std(
                [result["navigation"]["overall"]["spl"] for result in loaded]
            ),
            "wall_iou": mean_std([result["decoder"]["wall_iou"] for result in loaded]),
            "agent_accuracy": mean_std(
                [result["decoder"]["agent_accuracy_per_step"] for result in loaded]
            ),
            "goal_accuracy": mean_std(
                [result["decoder"]["goal_accuracy_per_step"] for result in loaded]
            ),
            "planning_channel_std": mean_std(
                [
                    result["decoder"]["planning_channel_std_per_step"]
                    for result in loaded
                ]
            ),
            "planning_token_norm_std": mean_std(
                [
                    result["decoder"]["planning_token_norm_std_per_step"]
                    for result in loaded
                ]
            ),
        }

    for variant in enabled_planners:
        if variant["input_mode"] != "spatial_jepa":
            continue
        planner_name = str(variant["name"])
        representation_name = str(variant["representation"])
        if planner_summaries.get(planner_name, {}).get("status") != "complete":
            continue
        if (
            representation_summaries.get(representation_name, {}).get("status")
            != "complete"
        ):
            continue
        for seed in seeds:
            if planner_source_rep_hashes.get(planner_name, {}).get(seed) != (
                representation_checkpoint_hashes.get(representation_name, {}).get(seed)
            ):
                raise ValueError(
                    f"planner {planner_name} seed={seed} did not use the evaluated "
                    f"{representation_name} checkpoint"
                )

    if len(training_runtime_signatures) != 1:
        raise ValueError(
            "formal training runs used inconsistent runtimes/devices: "
            f"{training_runtime_signatures}"
        )
    if len(evaluation_runtime_signatures) != 1:
        raise ValueError(
            "formal evaluation runs used inconsistent runtimes/devices: "
            f"{evaluation_runtime_signatures}"
        )

    summary = {
        "protocol": {
            "eval_manifest_sha256": eval_hash,
            "analysis_spec_sha256": expected_analysis_spec,
            "code_fingerprint": experiment_code_fingerprint(),
            "split_role": "confirmatory",
            "max_steps": max_steps,
            "seeds": seeds,
            "primary_action_selection": "unmasked",
            "primary_k_is_preregistered": True,
            "test_k_used_for_model_selection": False,
            "familywise_alpha": familywise_alpha,
            "simultaneous_ci_alpha": simultaneous_alpha,
            "multiplicity_control": "bonferroni_simultaneous_ci",
            "loop_or_cycle_definition": "maximum visits to any state >= 4",
            "training_runtime_signature": list(next(iter(training_runtime_signatures))),
            "evaluation_runtime_signature": list(
                next(iter(evaluation_runtime_signatures))
            ),
        },
        "reference_anchors": lock["reference_anchors"],
        "current_oracles": {
            "expected_sr": expected_oracle,
            "exact_bfs_sr": exact_bfs_sr,
            "oracle_vi_k256_sr": exact_vi_sr,
        },
        "representations": representation_summaries,
        "planners": planner_summaries,
        "k_curves": k_curves,
        "paired_comparisons": comparisons,
        "action_protocol_diagnostics": action_protocol_summaries,
        "claim_boundaries": [
            "Legacy BC and LeWM anchors were not evaluated on this confirmatory split.",
            "No result in this report establishes texture or cross-task "
            "generalization.",
            "Corrected and model-valid results are assistance diagnostics, "
            "not absolute ability.",
        ],
    }

    lines = [
        "# Spatial-JEPA Iterative Planning Results",
        "",
        "All primary rows use the untouched confirmatory 900-task manifest, "
        "`max_steps=128`, fully unmasked model actions, ten training seeds, and "
        "preregistered K. Test-set K is not selected by maximum SR.",
        "Legacy Set-B, BC, and LeWM numbers are context only and cannot support "
        "a superiority claim on this new split.",
        "",
        "## Current Oracle Checks",
        "",
        "| Anchor | SR@128 | Expected |",
        "|---|---:|---:|",
        f"| Exact BFS | {exact_bfs_sr:.6f} | {expected_oracle:.6f} |",
        f"| Oracle VI K=256 | {exact_vi_sr:.6f} | {expected_oracle:.6f} |",
        "",
        "## Planner Primary Results",
        "",
        "Size-25 Conv2d MACs include the complete static-field inference path. "
        "Representation cost is zero for raw-input controls.",
        "",
        "| Variant | K | Seeds | Planner params | Rep-path params | Total params | "
        "Planner GMACs | Rep GMACs | Total GMACs | SR | Eligible SR | SPL | "
        "OOD SR | Invalid | Loop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in enabled_planners:
        name = str(variant["name"])
        result = planner_summaries[name]
        if result["status"] != "complete":
            lines.append(
                f"| `{name}` | {result['primary_k']} | 0 | NA | NA | NA | NA | "
                "NA | NA | NA | NA | NA | NA | NA | NA |"
            )
            continue
        lines.append(
            f"| `{name}` | {result['primary_k']} | {result['sr']['n']} | "
            f"{result['planner_parameter_count']:,} | "
            f"{result['representation_planning_parameter_count']:,} | "
            f"{result['total_inference_parameter_count']:,} | "
            f"{result['size25_planner_conv_macs'] / 1e9:.3f} | "
            f"{result['size25_representation_conv_macs'] / 1e9:.3f} | "
            f"{result['size25_total_conv_macs'] / 1e9:.3f} | "
            f"{stat_text(result['sr'])} | {stat_text(result['eligible_sr'])} | "
            f"{stat_text(result['spl'])} | {stat_text(result['ood_sr'])} | "
            f"{stat_text(result['invalid_rate'])} | {stat_text(result['loop_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "Crossed paired bootstrap resamples training seeds and the shared "
            "task IDs as crossed factors. Three preregistered hypotheses use "
            "Bonferroni simultaneous confidence intervals.",
            "",
            "| Candidate | Baseline | Role | Preregistered criterion | Conclusion | "
            "Delta SR [CI] | Delta SPL [CI] |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for name, result in comparisons.items():
        sr = result["sr"]
        spl = result["spl"]
        hypothesis = result["hypothesis"]
        if hypothesis.get("test") == "superiority":
            criterion = f"SR CI low >= +{hypothesis['minimum_effect_sr']:.2f}"
        elif hypothesis.get("test") == "noninferiority":
            criterion = f"SR CI low > -{hypothesis['margin_sr']:.2f}"
        else:
            criterion = "exploratory"
        lines.append(
            f"| `{name}` | `{result['baseline']}` | "
            f"{hypothesis.get('role', 'exploratory')} | {criterion} | "
            f"{result['conclusion']} | "
            f"{sr['delta']:.3f} [{sr['ci_low']:.3f}, {sr['ci_high']:.3f}] | "
            f"{spl['delta']:.3f} [{spl['ci_low']:.3f}, {spl['ci_high']:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Action Assistance Diagnostics",
            "",
            "These rows quantify assistance from learned-valid masking or oracle "
            "valid/no-backtracking correction. They are not primary ability scores.",
            "",
            "| Variant | Unmasked SR | Model-valid SR | Model-valid Delta [CI] | "
            "Corrected SR | Corrected Delta [CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in enabled_planners:
        name = str(variant["name"])
        protocols = action_protocol_summaries.get(name)
        if not protocols:
            lines.append(f"| `{name}` | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| `{name}` | {stat_text(protocols['unmasked']['sr'])} | "
            f"{stat_text(protocols['model_valid']['sr'])} | "
            f"{effect_text(protocols['model_valid']['assistance_delta_sr'])} | "
            f"{stat_text(protocols['corrected']['sr'])} | "
            f"{effect_text(protocols['corrected']['assistance_delta_sr'])} |"
        )

    lines.extend(
        [
            "",
            "## Shortest-Path Stratification",
            "",
            "The `129+` bin is structurally censored by `max_steps=128` and must "
            "not be interpreted as an ordinary planning failure.",
            "",
            "| Variant | 1-16 | 17-32 | 33-64 | 65-128 | 129+ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in enabled_planners:
        name = str(variant["name"])
        result = planner_summaries[name]
        if result["status"] != "complete":
            lines.append(f"| `{name}` | NA | NA | NA | NA | NA |")
            continue
        path_sr = result["path_sr"]
        lines.append(
            f"| `{name}` | {stat_text(path_sr['001-016'])} | "
            f"{stat_text(path_sr['017-032'])} | {stat_text(path_sr['033-064'])} | "
            f"{stat_text(path_sr['065-128'])} | {stat_text(path_sr['129+'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Size SR",
            "",
            "| Variant | " + " | ".join(f"Size {size}" for size in eval_sizes) + " |",
            "|---|" + "---:|" * len(eval_sizes),
        ]
    )
    for variant in enabled_planners:
        name = str(variant["name"])
        result = planner_summaries[name]
        if result["status"] != "complete":
            lines.append(f"| `{name}` | " + " | ".join("NA" for _ in eval_sizes) + " |")
            continue
        lines.append(
            f"| `{name}` | "
            + " | ".join(stat_text(result["size_sr"][size]) for size in eval_sizes)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Learned-Field Diagnostics",
            "",
            "Local top-1 is tie-aware and masks oracle-invalid actions only for this "
            "diagnostic; it is not the unmasked navigation policy.",
            "",
            "| Variant | Local top-1 | Local margin | Value Pearson |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in enabled_planners:
        name = str(variant["name"])
        result = planner_summaries[name]
        if result["status"] != "complete":
            lines.append(f"| `{name}` | NA | NA | NA |")
            continue
        lines.append(
            f"| `{name}` | {stat_text(result['local_top1'])} | "
            f"{stat_text(result['local_margin'])} | "
            f"{stat_text(result['value_pearson'])} |"
        )

    lines.extend(
        [
            "",
            "## Decoded-Map BFS",
            "",
            "| Representation | Seeds | SR | SPL | Wall IoU | Agent acc/step | "
            "Goal acc/step | Channel std | Token-norm std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in representation_summaries.items():
        if result["status"] != "complete":
            lines.append(f"| `{name}` | 0 | NA | NA | NA | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| `{name}` | {result['sr']['n']} | {stat_text(result['sr'])} | "
                f"{stat_text(result['spl'])} | {stat_text(result['wall_iou'])} | "
                f"{stat_text(result['agent_accuracy'])} | "
                f"{stat_text(result['goal_accuracy'])} | "
                f"{stat_text(result['planning_channel_std'])} | "
                f"{stat_text(result['planning_token_norm_std'])} |"
            )

    output = Path(args.output)
    json_output = output.with_suffix(".json")
    require_new_output(output, args.overwrite)
    require_new_output(json_output, args.overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    strict_json_dump(json_output, summary)
    print(f"saved={output}")
    print(f"saved={json_output}")


if __name__ == "__main__":
    main()
