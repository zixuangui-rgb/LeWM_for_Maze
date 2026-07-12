#!/usr/bin/env python3
"""Generate deterministic publication-ready figures from the closure summary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from final_closure import FIGURE_FILENAMES
from final_closure.common import (
    RERUN_REASONS,
    load_config,
    load_json,
    prepare_rerun,
    require_new_output,
    require_study_open,
)

DISPLAY_NAMES = {
    "r4_raw_iterative_progressive": "R4 Raw Iterative",
    "j1_spatial_iterative_frozen": "J1 Spatial-JEPA",
    "bc_deepcnn_fixed": "BC DeepCNN",
    "lewm_l2_cem_seqlen2": "LeWM L2-CEM",
}
COLORS = {
    "r4_raw_iterative_progressive": "#2f6b4f",
    "j1_spatial_iterative_frozen": "#2f6fb0",
    "bc_deepcnn_fixed": "#b24a3b",
    "lewm_l2_cem_seqlen2": "#8a6a24",
}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "paper figures require the optional dependency: pip install -e '.[paper]'"
        ) from error
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.clf()


def create_figures(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["methods"])
    labels = [DISPLAY_NAMES.get(name, name) for name in methods]
    colors = [COLORS.get(name, "#555555") for name in methods]
    positions = np.arange(len(methods))

    primary_path = output_dir / "primary_results.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for axis, metric, label in zip(
        axes, ("sr", "spl"), ("Success rate", "SPL"), strict=True
    ):
        means = [
            summary["methods"][name]["overall"][metric]["mean"] for name in methods
        ]
        errors = [
            summary["methods"][name]["overall"][metric]["std"] for name in methods
        ]
        axis.bar(positions, means, yerr=errors, color=colors, capsize=3, width=0.72)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_ylabel(label)
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save(fig, primary_path)

    size_path = output_dir / "generalization_by_size.png"
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    for name in methods:
        values = summary["methods"][name]["by_size"]
        sizes = np.asarray(sorted(int(value) for value in values), dtype=np.int64)
        means = [values[str(size)]["sr"]["mean"] for size in sizes]
        errors = [values[str(size)]["sr"]["std"] for size in sizes]
        axis.errorbar(
            sizes,
            means,
            yerr=errors,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=COLORS.get(name, "#555555"),
            label=DISPLAY_NAMES.get(name, name),
        )
    axis.axvline(22, color="#666666", linestyle="--", linewidth=0.9)
    axis.text(22.25, 0.03, "OOD", color="#555555", fontsize=8)
    axis.set_xlabel("Maze size")
    axis.set_ylabel("Success rate")
    axis.set_xticks(
        sorted(int(value) for value in summary["methods"][methods[0]]["by_size"])
    )
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    _save(fig, size_path)

    path_path = output_dir / "generalization_by_path_length.png"
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    bins = list(summary["methods"][methods[0]]["by_shortest_path"])
    x_values = np.arange(len(bins))
    for name in methods:
        values = summary["methods"][name]["by_shortest_path"]
        means = [values[path_bin]["sr"]["mean"] for path_bin in bins]
        errors = [values[path_bin]["sr"]["std"] for path_bin in bins]
        axis.errorbar(
            x_values,
            means,
            yerr=errors,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=COLORS.get(name, "#555555"),
            label=DISPLAY_NAMES.get(name, name),
        )
    axis.set_xticks(x_values, bins)
    axis.set_xlabel("Oracle shortest-path length")
    axis.set_ylabel("Success rate")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    _save(fig, path_path)

    k_path = output_dir / "spatial_iteration_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    for name, curve in summary["spatial_k_curves"].items():
        iterations = np.asarray(sorted(int(value) for value in curve), dtype=np.int64)
        for axis, metric in zip(axes, ("sr", "spl"), strict=True):
            means = [curve[str(value)][metric]["mean"] for value in iterations]
            errors = [curve[str(value)][metric]["std"] for value in iterations]
            axis.errorbar(
                iterations,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.5,
                capsize=2,
                color=COLORS.get(name, "#555555"),
                label=DISPLAY_NAMES.get(name, name),
            )
    for axis, label in zip(axes, ("Success rate", "SPL"), strict=True):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Planning iterations K")
        axis.set_ylabel(label)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False)
    _save(fig, k_path)

    compute_path = output_dir / "spatial_compute_curve.png"
    fig, axis = plt.subplots(figsize=(5.8, 3.6))
    for name, curve in summary["spatial_k_curves"].items():
        ordered = [curve[str(value)] for value in sorted(int(key) for key in curve)]
        compute_gmac = [value["conv_macs_size25"] / 1e9 for value in ordered]
        means = [value["sr"]["mean"] for value in ordered]
        errors = [value["sr"]["std"] for value in ordered]
        axis.errorbar(
            compute_gmac,
            means,
            yerr=errors,
            marker="o",
            linewidth=1.5,
            capsize=2,
            color=COLORS.get(name, "#555555"),
            label=DISPLAY_NAMES.get(name, name),
        )
    axis.set_xlabel("Size-25 convolutional GMAC")
    axis.set_ylabel("Success rate")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False)
    _save(fig, compute_path)
    plt.close("all")
    outputs = [primary_path, size_path, path_path, k_path, compute_path]
    if tuple(path.name for path in outputs) != FIGURE_FILENAMES:
        raise RuntimeError("figure artifact order differs from the closure schema")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_closure/configs/default.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-reason", choices=RERUN_REASONS, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    require_study_open(config)
    output_dir = Path(config["paths"]["figure_dir"])
    expected = [output_dir / name for name in FIGURE_FILENAMES]
    prepare_rerun(
        expected,
        overwrite=args.overwrite,
        reason=args.rerun_reason,
    )
    for path in expected:
        require_new_output(path, args.overwrite)
    summary = load_json(config["paths"]["summary_json"])
    outputs = create_figures(summary, output_dir)
    print("generated figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
