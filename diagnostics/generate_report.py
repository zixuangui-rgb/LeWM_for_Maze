#!/usr/bin/env python3
"""Generate a Chinese Markdown report from diagnostics outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from diagnostics.common import ensure_dir, read_json, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Maze-JEPA diagnostic report.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="diagnostics_runs")
    parser.add_argument("--title", default="Maze-JEPA Diagnostic Report")
    return parser.parse_args()


def load_optional(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if v != v:
        return "NA"
    return f"{v:.{digits}f}"


def metric_value(metrics: dict[str, Any], task: str) -> str:
    if task == "bfs_distance_norm":
        return fmt(metrics.get("rmse"))
    if task == "valid_action":
        return fmt(metrics.get("exact_match"))
    if task == "optimal_action":
        return fmt(metrics.get("top1_any_optimal", metrics.get("accuracy")))
    return fmt(metrics.get("accuracy"))


def probe_section(probes: dict[str, Any] | None) -> str:
    if not probes:
        return "## 1. Layer-wise Probes\n\n未找到 `probe_metrics.json`。\n"
    rows = probes.get("results", [])
    lines = [
        "## 1. Layer-wise Probes",
        "",
        "这部分回答：不同层里到底保留了哪些可用于导航的信息。Linear probe 表示信息是否容易线性读出；MLP probe 表示信息是否存在但需要非线性解码。",
        "",
        "| Scope | Layer | Task | Probe | Main metric | Seen | OOD |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    task_order = {
        "agent_x": 0,
        "agent_y": 1,
        "goal_x": 2,
        "goal_y": 3,
        "valid_action": 4,
        "bfs_distance_norm": 5,
        "optimal_action": 6,
    }
    filtered = [
        row
        for row in rows
        if row.get("scope") == "unified_all_eval" or str(row.get("scope", "")).startswith("per_size_sz21")
    ]
    filtered.sort(key=lambda r: (r.get("scope", ""), r.get("layer", ""), task_order.get(r.get("task", ""), 99), r.get("probe_type", "")))
    for row in filtered:
        metrics = row.get("metrics", {})
        groups = row.get("metrics_by_group", {})
        task = row.get("task", "")
        lines.append(
            "| {scope} | {layer} | {task} | {probe} | {main} | {seen} | {ood} |".format(
                scope=row.get("scope", ""),
                layer=row.get("layer", ""),
                task=task,
                probe=row.get("probe_type", ""),
                main=metric_value(metrics, task),
                seen=metric_value(groups.get("seen", {}), task) if groups else "NA",
                ood=metric_value(groups.get("ood", {}), task) if groups else "NA",
            )
        )
    lines.extend(
        [
            "",
            "解读方式：",
            "",
            "- 如果 `spatial_flat/spatial_pool` 很高但 `embedding` 明显下降，优先怀疑 projector 丢失空间/拓扑信息。",
            "- 如果位置 probe 高但 `optimal_action` 低，说明不是单纯“知道在哪”，而是缺少 geodesic distance 或局部动作排序结构。",
            "- 如果 unified probe 的 OOD 明显低于 seen，说明尺寸泛化本身是瓶颈。",
            "",
        ]
    )
    return "\n".join(lines)


def metric_alignment_section(alignment: dict[str, Any] | None) -> str:
    if not alignment:
        return "## 2. Metric Alignment\n\n未找到 `metric_alignment.json`。\n"
    lines = [
        "## 2. Metric Alignment",
        "",
        "这部分回答：L2 / DistanceHead / QRL 的分数是否真的和 BFS 距离、局部最优动作一致。",
        "",
        "| Scorer | Pearson | Spearman | Local top-1 | Local pairwise | Local margin | Seen top-1 | OOD top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(alignment.get("summary", {}).items()):
        overall = item.get("overall", {})
        seen = item.get("by_bucket", {}).get("seen", {})
        ood = item.get("by_bucket", {}).get("ood", {})
        lines.append(
            f"| {name} | {fmt(overall.get('pearson'))} | {fmt(overall.get('spearman'))} | "
            f"{fmt(overall.get('local_top1'))} | {fmt(overall.get('local_pairwise'))} | "
            f"{fmt(overall.get('local_margin'))} | {fmt(seen.get('local_top1'))} | {fmt(ood.get('local_top1'))} |"
        )
    lines.extend(
        [
            "",
            "解读方式：",
            "",
            "- `Pearson/Spearman` 高但 `Local top-1` 低：说明全局距离回归看起来不错，但局部动作排序仍然不能支持导航。",
            "- `Local margin` 小：说明好动作和坏动作的分数差距太小，planner 容易受 predictor 噪声影响。",
            "- OOD top-1 掉得多：说明 metric 的尺寸泛化不足。",
            "",
        ]
    )
    return "\n".join(lines)


def rollout_section(rollout: dict[str, Any] | None) -> str:
    if not rollout:
        return "## 3. Predictor Rollout\n\n未找到 `predictor_rollout.json`。\n"
    lines = [
        "## 3. Predictor Rollout",
        "",
        "这部分回答：predictor 想象出来的 latent 会不会随 horizon 变长而偏离真实状态流形。",
        "",
        "| Mode | Horizon | Latent MSE | Cosine | NN exact | NN BFS error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in sorted(rollout.get("summary", {}).items()):
        for horizon, metrics in sorted(item.get("overall", {}).items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {mode} | {horizon} | {fmt(metrics.get('latent_mse'))} | {fmt(metrics.get('cosine'))} | "
                f"{fmt(metrics.get('nn_exact'))} | {fmt(metrics.get('nn_bfs_error'))} |"
            )
    lines.extend(
        [
            "",
            "解读方式：",
            "",
            "- `teacher_forced` 差：一阶 predictor 本身就不准。",
            "- `teacher_forced` 可以但 `closed_loop` 快速变差：rollout 累积误差是瓶颈。",
            "- `NN exact` 低且 `NN BFS error` 高：预测 latent 已经离开真实可导航状态流形。",
            "",
        ]
    )
    return "\n".join(lines)


def failure_section(failure: dict[str, Any] | None) -> str:
    if not failure:
        return "## 4. Failure Taxonomy\n\n未找到 `failure_taxonomy.json`。\n"
    summary = failure.get("summary", {})
    overall = summary.get("overall", {})
    lines = [
        "## 4. Failure Taxonomy",
        "",
        "这部分回答：导航失败主要是哪类原因造成的。",
        "",
        f"- Overall SR: `{fmt(overall.get('sr'))}`",
        f"- Episodes: `{overall.get('n', 'NA')}`",
        "",
        "| Tag | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    counts = overall.get("tag_counts", {})
    rates = overall.get("tag_rates", {})
    for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {tag} | {count} | {fmt(rates.get(tag))} |")
    lines.extend(
        [
            "",
            "解读方式：",
            "",
            "- `metric_wrong` 高：优先改 metric / action-ranking objective。",
            "- `predictor_wrong` 高：优先改 predictor 训练或 predictor-aligned head。",
            "- `loop_or_cycle` 高：需要 planner 记忆、anti-loop 或更大的局部 margin。",
            "- `ood_size` 高：需要结构性尺寸泛化方案，例如 fully convolutional value/distance map。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    metrics = out / "metrics"
    probes = load_optional(metrics / "probe_metrics.json")
    alignment = load_optional(metrics / "metric_alignment.json")
    rollout = load_optional(metrics / "predictor_rollout.json")
    failure = load_optional(metrics / "failure_taxonomy.json")

    parts = [
        f"# {args.title}",
        "",
        f"Run id: `{args.run_id}`",
        "",
        "本报告由 `diagnostics/` 自动生成，目标是把 JEPA/LeWM 在 Maze 导航中的瓶颈拆成可复用、可横向比较的诊断指标。",
        "",
        probe_section(probes),
        metric_alignment_section(alignment),
        rollout_section(rollout),
        failure_section(failure),
        "## 5. Recommended Reading Order",
        "",
        "1. 先看 Layer-wise probes，判断信息在哪一层丢失。",
        "2. 再看 Metric alignment，判断 distance/score 是否真的能支持局部动作选择。",
        "3. 再看 Predictor rollout，判断 CEM/predictor-greedy 是否被 rollout drift 限制。",
        "4. 最后看 Failure taxonomy，把失败 episode 映射到下一步工程动作。",
        "",
    ]
    report_path = out / "diagnostic_report.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
