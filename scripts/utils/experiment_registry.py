"""Experiment registry for systematic LeWM navigation feasibility study.

Provides `register_experiment()` to write one row into the global experiment
registry CSV. Every experiment must call this function to ensure all runs are
tracked consistently.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Registry CSV columns as defined in Prompt 0.
REGISTRY_COLUMNS: list[str] = [
    "experiment_id",
    "date",
    "phase",
    "method",
    "maze_size",
    "size_ood_setting",
    "train_seed_start",
    "train_num_levels",
    "test_seed_start",
    "test_num_levels",
    "train_topology_hash_file",
    "test_topology_hash_file",
    "checkpoint",
    "encoder_architecture",
    "latent_dim",
    "uses_spatial_latent",
    "uses_topology_supervision",
    "probe_target",
    "metric_head",
    "planner",
    "cem_horizon",
    "cem_candidates",
    "cem_iterations",
    "random_seed",
    "SR",
    "SPL",
    "mean_return",
    "first_action_acc",
    "neighbor_argmin_acc",
    "bfs_spearman",
    "agent_cell_acc",
    "goal_cell_acc",
    "occupancy_iou",
    "notes",
]

# Default path relative to project root.
DEFAULT_REGISTRY_PATH: str = "results/registry/experiment_registry.csv"


def _find_project_root() -> Path:
    """Locate the project root by searching for the hdwm package directory."""
    current = Path.cwd()
    # Walk upwards until we find the hdwm directory or hit the filesystem root.
    for parent in [current, *current.parents]:
        if (parent / "hdwm").is_dir() and (parent / "configs").is_dir():
            return parent
    # Fallback: assume we are already in the project root.
    return current


def _init_registry(path: Path) -> None:
    """Create the registry CSV with header if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REGISTRY_COLUMNS)


def register_experiment(
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Write one row into the global experiment registry.

    Args:
        config:
            Dictionary of experiment configuration values.  Keys should be a
            subset of ``REGISTRY_COLUMNS``.  Missing keys will be filled with
            the empty string.
        metrics:
            Dictionary of evaluation metrics.  Keys should be a subset of
            ``REGISTRY_COLUMNS`` (e.g. SR, SPL, mean_return, …).  These are
            merged into the config dict (metrics take precedence).
        output_path:
            Path to the registry CSV.  If ``None``, the default location under
            ``results/registry/`` relative to the project root is used.

    Returns:
        The absolute path to the registry CSV that was written to.
    """
    if output_path is None:
        root = _find_project_root()
        output_path = root / DEFAULT_REGISTRY_PATH
    else:
        output_path = Path(output_path)

    _init_registry(output_path)

    # Merge config and metrics; metrics override config for overlapping keys.
    row: dict[str, Any] = {col: "" for col in REGISTRY_COLUMNS}
    row.update(config)
    row.update(metrics)

    # Auto-fill date if empty.
    if not row.get("date"):
        row["date"] = datetime.now().strftime("%Y-%m-%d")

    # Ensure experiment_id is set; generate one if missing.
    if not row.get("experiment_id"):
        row["experiment_id"] = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Read existing rows to determine if we should append or overwrite.
    existing_rows: list[dict[str, str]] = []
    if output_path.exists():
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    # If an experiment with the same ID already exists, replace it; otherwise append.
    replaced = False
    for i, existing in enumerate(existing_rows):
        if existing.get("experiment_id") == row["experiment_id"]:
            existing_rows[i] = {k: str(v) for k, v in row.items()}
            replaced = True
            break

    if not replaced:
        existing_rows.append({k: str(v) for k, v in row.items()})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)

    return output_path.resolve()


def read_registry(output_path: str | Path | None = None) -> list[dict[str, str]]:
    """Read all rows from the experiment registry.

    Args:
        output_path:
            Path to the registry CSV.  If ``None``, uses the default location.

    Returns:
        List of dictionaries, one per registered experiment.
    """
    if output_path is None:
        root = _find_project_root()
        output_path = root / DEFAULT_REGISTRY_PATH
    else:
        output_path = Path(output_path)

    if not output_path.exists():
        return []

    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
