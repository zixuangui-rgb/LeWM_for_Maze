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

from spatial_jepa_planning.common import strict_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="spatial_jepa_planning/configs/default.json"
    )
    parser.add_argument("--output", default="spatial_jepa_planning_runs/summary.md")
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


def validate_metadata(
    data: dict[str, Any],
    *,
    expected_eval_hash: str,
    max_steps: int,
    expected_task_count: int,
    evaluation_seed: int,
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
    if (
        int(metadata.get("max_per_size", -1)) != 0
        or int(metadata.get("limit", -1)) != 0
    ):
        raise ValueError("primary summary requires unsampled full-manifest evaluation")
    if metadata.get("action_selection") != "corrected":
        raise ValueError(
            "primary summary accepts only corrected action-selection results"
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
    if not metadata.get("comparable_to_full900", False):
        raise ValueError("primary summary accepts only full-900 comparable results")


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


def hierarchical_paired_bootstrap(
    candidate_rows: list[list[dict[str, Any]]],
    baseline_rows: list[list[dict[str, Any]]],
    *,
    metric: str,
    samples: int,
    seed: int = 20260710,
) -> dict[str, float]:
    if len(candidate_rows) != len(baseline_rows) or not candidate_rows:
        raise ValueError("paired bootstrap needs matched non-empty seed lists")
    paired_differences: list[np.ndarray] = []
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        candidate_by_id = {row["task_id"]: row for row in candidate}
        baseline_by_id = {row["task_id"]: row for row in baseline}
        if candidate_by_id.keys() != baseline_by_id.keys():
            raise ValueError("paired result files do not contain identical task IDs")
        ordered_ids = sorted(candidate_by_id)
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
    observed = float(
        np.mean([differences.mean() for differences in paired_differences])
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected_seeds = rng.integers(
            0, len(paired_differences), len(paired_differences)
        )
        seed_means: list[float] = []
        for seed_index in selected_seeds:
            differences = paired_differences[int(seed_index)]
            task_indices = rng.integers(0, len(differences), len(differences))
            seed_means.append(float(differences[task_indices].mean()))
        draws[sample_index] = float(np.mean(seed_means))
    return {
        "delta": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
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


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    lock = load_json(config["paths"]["protocol_lock"])
    eval_hash = str(lock["eval_manifest"]["sha256"])
    expected_task_count = int(lock["eval_manifest"]["count"])
    max_steps = int(config["protocol"]["max_steps"])
    evaluation_seed = int(config["evaluation"]["seed"])
    seeds = [int(seed) for seed in config["seeds"]]
    planner_template = config["paths"]["planner_eval_template"]
    representation_template = config["paths"]["representation_eval_template"]
    bootstrap_samples = int(config["evaluation"].get("bootstrap_samples", 10000))

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
    )
    validate_metadata(
        oracle_vi,
        expected_eval_hash=eval_hash,
        max_steps=max_steps,
        expected_task_count=expected_task_count,
        evaluation_seed=evaluation_seed,
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
    enabled_planners = [
        variant for variant in config["planners"] if variant.get("enabled", True)
    ]
    for variant in enabled_planners:
        name = str(variant["name"])
        primary_k = int(variant["primary_iterations"])
        loaded: list[dict[str, Any]] = []
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
            )
            for result in data.get("results", {}).values():
                validate_task_rows(result, expected_task_count)
            if int(data["metadata"].get("training_seed", -1)) != seed:
                raise ValueError(f"result/checkpoint seed label mismatch for {name}")
            loaded.append(data)
        if not loaded:
            planner_summaries[name] = {"status": "missing", "primary_k": primary_k}
            continue
        if len(loaded) != len(seeds):
            raise ValueError(f"partial seed set for {name}: {len(loaded)}/{len(seeds)}")
        primary = [primary_result(data, primary_k) for data in loaded]
        primary_rows[name] = [result["task_rows"] for result in primary]
        planner_summaries[name] = {
            "status": "complete",
            "primary_k": primary_k,
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
        comparisons[name] = {
            "baseline": baseline,
            "sr": hierarchical_paired_bootstrap(
                primary_rows[name],
                primary_rows[baseline],
                metric="success",
                samples=bootstrap_samples,
            ),
            "spl": hierarchical_paired_bootstrap(
                primary_rows[name],
                primary_rows[baseline],
                metric="spl",
                samples=bootstrap_samples,
            ),
        }

    representation_summaries: dict[str, Any] = {}
    for variant in config["representations"]:
        if not variant.get("enabled", True):
            continue
        name = str(variant["name"])
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
                )
                validate_task_rows(data.get("results", {}), expected_task_count)
                if int(data["metadata"].get("training_seed", -1)) != seed:
                    raise ValueError(f"representation seed label mismatch for {name}")
                loaded.append(data["results"])
        if not loaded:
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
        }

    summary = {
        "protocol": {
            "eval_manifest_sha256": eval_hash,
            "max_steps": max_steps,
            "seeds": seeds,
            "primary_k_is_preregistered": True,
            "test_k_used_for_model_selection": False,
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
    }

    lines = [
        "# Spatial-JEPA Iterative Planning Results",
        "",
        "All primary rows use the locked full-900 manifest, `max_steps=128`, "
        "corrected action selection, and preregistered K. Test-set K is not "
        "selected by maximum SR.",
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
        "| Variant | K | Seeds | SR | SPL | Seen SR | OOD SR | Local top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in enabled_planners:
        name = str(variant["name"])
        result = planner_summaries[name]
        if result["status"] != "complete":
            lines.append(
                f"| `{name}` | {result['primary_k']} | 0 | NA | NA | NA | NA | NA |"
            )
            continue
        lines.append(
            f"| `{name}` | {result['primary_k']} | {result['sr']['n']} | "
            f"{stat_text(result['sr'])} | {stat_text(result['spl'])} | "
            f"{stat_text(result['seen_sr'])} | {stat_text(result['ood_sr'])} | "
            f"{stat_text(result['local_top1'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "Hierarchical bootstrap resamples training seeds first and fixed "
            "evaluation tasks second.",
            "",
            "| Candidate | Baseline | Delta SR [95% CI] | Delta SPL [95% CI] |",
            "|---|---|---:|---:|",
        ]
    )
    for name, result in comparisons.items():
        sr = result["sr"]
        spl = result["spl"]
        lines.append(
            f"| `{name}` | `{result['baseline']}` | "
            f"{sr['delta']:.3f} [{sr['ci_low']:.3f}, {sr['ci_high']:.3f}] | "
            f"{spl['delta']:.3f} [{spl['ci_low']:.3f}, {spl['ci_high']:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Decoded-Map BFS",
            "",
            "| Representation | Seeds | SR | SPL | Wall IoU | Agent acc/step | "
            "Goal acc/step |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in representation_summaries.items():
        if result["status"] != "complete":
            lines.append(f"| `{name}` | 0 | NA | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| `{name}` | {result['sr']['n']} | {stat_text(result['sr'])} | "
                f"{stat_text(result['spl'])} | {stat_text(result['wall_iou'])} | "
                f"{stat_text(result['agent_accuracy'])} | "
                f"{stat_text(result['goal_accuracy'])} |"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_output = output.with_suffix(".json")
    strict_json_dump(json_output, summary)
    print(f"saved={output}")
    print(f"saved={json_output}")


if __name__ == "__main__":
    main()
