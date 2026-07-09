#!/usr/bin/env python3
"""Summarize the planning-repair ablation matrix into one Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize planning-repair ablations.")
    parser.add_argument("--config", default="planning_repair/configs/default.json")
    parser.add_argument("--output", default="planning_repair_runs/ablation_summary.md")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_optional(path: str | Path) -> Any | None:
    path = Path(path)
    return read_json(path) if path.exists() else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if v != v:
        return "NA"
    return f"{v:.{digits}f}"


def enabled_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        variant
        for variant in cfg.get("variants", [])
        if bool(variant.get("enabled", True))
    ]


def variant_name(variant: dict[str, Any]) -> str:
    return str(variant["name"])


def variant_run_id(variant: dict[str, Any]) -> str:
    return str(variant.get("diagnostics_run_id", f"planning_repair_{variant_name(variant)}"))


def variant_output(variant: dict[str, Any], key: str, subdir: str) -> str:
    outputs = variant.get("outputs", {})
    if key in outputs:
        return str(outputs[key])
    return f"planning_repair_runs/{variant_name(variant)}/{subdir}/results.json"


def diagnostics_dir(cfg: dict[str, Any], run_id: str, baseline: bool = False) -> Path:
    diagnostics_cfg = dict(cfg.get("diagnostics", {}))
    if baseline:
        diagnostics_cfg.update(cfg.get("baseline_diagnostics", {}))
    out_dir = diagnostics_cfg.get("out_dir", "diagnostics_runs")
    return Path(out_dir) / run_id / "metrics"


def find_probe_metric(
    probes: dict[str, Any] | None,
    *,
    layer: str,
    task: str,
    probe_type: str = "mlp",
    metric: str,
) -> Any:
    if not probes:
        return None
    rows = probes.get("results", [])
    candidates = [
        row
        for row in rows
        if row.get("scope") == "unified_all_eval"
        and row.get("layer") == layer
        and row.get("task") == task
        and row.get("probe_type") == probe_type
    ]
    if not candidates:
        return None
    metrics = candidates[0].get("metrics", {})
    return metrics.get(metric)


def rollout_metric(rollout: dict[str, Any] | None, mode: str, metric: str, horizon: str = "10") -> Any:
    if not rollout:
        return None
    overall = rollout.get("summary", {}).get(mode, {}).get("overall", {})
    if horizon not in overall and overall:
        horizon = str(max(int(key) for key in overall))
    return overall.get(horizon, {}).get(metric)


def metric_alignment(alignment: dict[str, Any] | None, scorer: str, metric: str) -> Any:
    if not alignment:
        return None
    return alignment.get("summary", {}).get(scorer, {}).get("overall", {}).get(metric)


def failure_metric(failure: dict[str, Any] | None, metric: str) -> Any:
    if not failure:
        return None
    return failure.get("summary", {}).get("overall", {}).get(metric)


def failure_tag(failure: dict[str, Any] | None, tag: str) -> Any:
    if not failure:
        return None
    return failure.get("summary", {}).get("overall", {}).get("tag_rates", {}).get(tag)


def planning_sr(path: str | Path) -> Any:
    data = load_optional(path)
    if not data:
        return None
    return data.get("summary", {}).get("sr")


def prefix_rollout_error(path: str | Path, horizon: str = "5") -> Any:
    data = load_optional(path)
    if not data:
        return None
    overall = data.get("summary", {}).get("overall", {})
    if horizon not in overall and overall:
        horizon = str(max(int(key) for key in overall))
    return overall.get(horizon, {}).get("nn_bfs_error")


def best_p0_sr(path: str | Path) -> Any:
    data = load_optional(path)
    if not data:
        return None
    best = None
    for by_horizon in data.get("results", {}).values():
        for summary in by_horizon.values():
            sr = summary.get("sr")
            if sr is None:
                continue
            best = float(sr) if best is None else max(best, float(sr))
    return best


def collect_diagnostics(cfg: dict[str, Any], run_id: str, *, baseline: bool = False) -> dict[str, Any]:
    metrics_dir = diagnostics_dir(cfg, run_id, baseline=baseline)
    probes = load_optional(metrics_dir / "probe_metrics.json")
    alignment = load_optional(metrics_dir / "metric_alignment.json")
    rollout = load_optional(metrics_dir / "predictor_rollout.json")
    failure = load_optional(metrics_dir / "failure_taxonomy.json")
    return {
        "embedding_goal_y_rmse": find_probe_metric(
            probes,
            layer="embedding",
            task="goal_y_norm",
            metric="rmse",
        ),
        "embedding_valid_exact": find_probe_metric(
            probes,
            layer="embedding",
            task="valid_action",
            metric="exact_match",
        ),
        "embedding_optimal_top1": find_probe_metric(
            probes,
            layer="embedding",
            task="optimal_action",
            metric="top1_any_optimal",
        ),
        "latent_l2_local_top1": metric_alignment(alignment, "latent_l2", "local_top1"),
        "latent_l2_local_margin": metric_alignment(alignment, "latent_l2", "local_margin"),
        "closed_loop_h10_bfs_error": rollout_metric(
            rollout,
            "closed_loop",
            "nn_bfs_error",
            "10",
        ),
        "failure_sr": failure_metric(failure, "sr"),
        "metric_wrong_rate": failure_tag(failure, "metric_wrong"),
        "predictor_wrong_rate": failure_tag(failure, "predictor_wrong"),
        "loop_rate": failure_tag(failure, "loop_or_cycle"),
    }


def variant_losses(cfg: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    train = dict(cfg.get("train", {}))
    train.update(variant.get("train", {}))
    return {
        "valid": train.get("lambda_valid", 0.0),
        "action": train.get("lambda_action", 0.0),
        "bfs": train.get("lambda_bfs", 0.0),
        "reach": train.get("lambda_reach", 0.0),
        "prefix": train.get("lambda_prefix", 0.0),
    }


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    rows: list[dict[str, Any]] = []

    baseline_cfg = cfg.get("baseline_diagnostics", {})
    baseline_run_id = baseline_cfg.get("run_id", "planning_repair_baseline")
    rows.append(
        {
            "name": "baseline",
            "kind": "frozen baseline",
            "losses": {"valid": 0, "action": 0, "bfs": 0, "reach": 0, "prefix": 0},
            "diag": collect_diagnostics(cfg, baseline_run_id, baseline=True),
            "aux_sr": None,
            "prefix_sr": None,
        }
    )

    for variant in enabled_variants(cfg):
        rows.append(
            {
                "name": variant_name(variant),
                "kind": variant.get("description", ""),
                "losses": variant_losses(cfg, variant),
                "diag": collect_diagnostics(cfg, variant_run_id(variant)),
                "aux_sr": planning_sr(variant_output(variant, "aux_eval_output", "aux_action_head")),
                "prefix_rollout": prefix_rollout_error(
                    variant_output(variant, "prefix_rollout_output", "prefix_rollout")
                ),
                "prefix_sr": planning_sr(variant_output(variant, "prefix_eval_output", "prefix_planner")),
            }
        )

    p0_sr = best_p0_sr(cfg.get("paths", {}).get("p0_output", ""))
    lines = [
        "# Planning Repair Ablation Summary",
        "",
        f"Config: `{args.config}`",
        f"P0 best short-horizon baseline SR: `{fmt(p0_sr)}`",
        "",
        "## Matrix",
        "",
        "| Variant | Role | valid/action/bfs/reach/prefix | emb valid | emb opt-action | emb goal_y RMSE | L2 local top1 | L2 margin | old h10 drift | prefix h5 drift | failure SR | metric wrong | predictor wrong | loop | aux SR | prefix SR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        losses = row["losses"]
        diag = row["diag"]
        loss_text = "/".join(
            fmt(losses[key], 2) for key in ["valid", "action", "bfs", "reach", "prefix"]
        )
        lines.append(
            "| {name} | {kind} | {losses} | {valid} | {opt} | {goal} | {top1} | {margin} | {drift} | {prefix_drift} | {sr} | {metric_wrong} | {predictor_wrong} | {loop} | {aux_sr} | {prefix_sr} |".format(
                name=row["name"],
                kind=row["kind"],
                losses=loss_text,
                valid=fmt(diag["embedding_valid_exact"]),
                opt=fmt(diag["embedding_optimal_top1"]),
                goal=fmt(diag["embedding_goal_y_rmse"]),
                top1=fmt(diag["latent_l2_local_top1"]),
                margin=fmt(diag["latent_l2_local_margin"]),
                drift=fmt(diag["closed_loop_h10_bfs_error"]),
                prefix_drift=fmt(row.get("prefix_rollout")),
                sr=fmt(diag["failure_sr"]),
                metric_wrong=fmt(diag["metric_wrong_rate"]),
                predictor_wrong=fmt(diag["predictor_wrong_rate"]),
                loop=fmt(diag["loop_rate"]),
                aux_sr=fmt(row["aux_sr"]),
                prefix_sr=fmt(row["prefix_sr"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `continued_control` is the mandatory control for extra training, longer `seq_len`, and optimizer exposure. A repair claim should compare against it, not only against the frozen baseline.",
            "- If `p1_info_aux` improves `emb valid / goal_y RMSE` over `continued_control`, the projector information wall is being repaired.",
            "- If `p15_action_ranking` improves `emb opt-action`, `L2 local top1`, or `metric wrong` over `p1_info_aux`, the listwise action objective is doing causal work.",
            "- `old h10 drift` is the original one-step predictor diagnostic; `prefix h5 drift` is the direct multi-horizon prefix head.",
            "- If only `prefix h5 drift` improves, the claim is that prefix planning bypasses recursive drift. If `old h10 drift` also improves, then the shared backbone dynamics became more stable too.",
            "- If `p2_full` improves SR without the diagnostic improvements above, treat it as planner engineering rather than a proved JEPA representation repair.",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
