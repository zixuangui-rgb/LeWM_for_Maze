#!/usr/bin/env python3
"""Generate or byte-for-byte verify all preregistered AIR0 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from air_jepa.stage0_workspace.common import (
    DEFAULT_CONFIG,
    atomic_text_dump,
    resolve_path,
)
from air_jepa.stage0_workspace.schemas import Stage0Config
from diagnostics.common import bfs_distances_from
from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    create_env,
    read_jsonl,
)

SIZES = tuple(range(9, 26, 2))
PER_SIZE = 100
NAMESPACES = {
    "air_dev": 2_000_000,
    "air_select": 3_000_000,
    "air_final": 4_000_000,
}
MANIFEST_SCHEMA = "air-jepa-stage0-task-v1"


def _serialize(entries: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        for entry in entries
    )


def _regenerated_layout(entry: dict[str, Any]) -> str:
    return canonical_layout_hash(create_env(entry)._maze_mask)


def _prior_layouts(config: Stage0Config) -> set[str]:
    paths = (
        config.paths.train_manifest,
        config.paths.historical_development_manifest,
        config.paths.historical_confirmatory_manifest,
    )
    layouts: set[str] = set()
    for path in paths:
        for entry in read_jsonl(resolve_path(path)):
            regenerated = _regenerated_layout(entry)
            layouts.add(regenerated)
    return layouts


def _generate_split(
    role: str,
    namespace: int,
    forbidden_layouts: set[str],
    forbidden_tasks: set[str],
) -> list[dict[str, Any]]:
    accepted_layouts: set[str] = set()
    accepted_tasks: set[str] = set()
    entries: list[dict[str, Any]] = []
    for size in SIZES:
        accepted = 0
        candidate_index = 0
        while accepted < PER_SIZE:
            topology_seed = namespace + size * 10_000 + candidate_index
            env_seed = topology_seed * 100 + 29
            seed_entry = {
                "maze_size": size,
                "topology_seed": topology_seed,
                "env_seed": env_seed,
            }
            env = create_env(seed_entry)
            _, info = env.reset()
            start = int(info["state"])
            goal = int(env._goal_position)
            layout_hash = canonical_layout_hash(env._maze_mask)
            candidate_index += 1
            if layout_hash in forbidden_layouts or layout_hash in accepted_layouts:
                continue
            distances = bfs_distances_from(env._maze_mask, goal, size)
            path_length = int(distances[start])
            if start == goal or path_length <= 0:
                continue
            task_hash = canonical_task_hash(
                maze_size=size,
                layout_hash=layout_hash,
                start_cell=start,
                goal_cell=goal,
            )
            if task_hash in forbidden_tasks or task_hash in accepted_tasks:
                continue
            walls = int(env._maze_mask.sum())
            entries.append(
                {
                    **seed_entry,
                    "start_cell": start,
                    "goal_cell": goal,
                    "layout_hash": layout_hash,
                    "task_hash": task_hash,
                    "bfs_path_length": path_length,
                    "num_walls": walls,
                    "num_free": size * size - walls,
                    "manifest_schema": MANIFEST_SCHEMA,
                    "split_role": role,
                }
            )
            accepted_layouts.add(layout_hash)
            accepted_tasks.add(task_hash)
            accepted += 1
    forbidden_layouts.update(accepted_layouts)
    forbidden_tasks.update(accepted_tasks)
    return entries


def _stable_subset(
    entries: list[dict[str, Any]],
    per_size: dict[int, int],
    namespace: str,
) -> list[dict[str, Any]]:
    groups: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[int(entry["maze_size"])].append(entry)
    selected: list[dict[str, Any]] = []
    for size in sorted(per_size):
        ranked = sorted(
            groups[size],
            key=lambda entry: hashlib.sha256(
                f"{namespace}:{entry['task_hash']}".encode("ascii")
            ).hexdigest(),
        )
        count = int(per_size[size])
        if len(ranked) < count:
            raise ValueError(f"not enough size-{size} tasks for {namespace}")
        selected.extend(ranked[:count])
    return selected


def generate_all(config: Stage0Config) -> dict[str, list[dict[str, Any]]]:
    forbidden_layouts = _prior_layouts(config)
    forbidden_tasks: set[str] = set()
    generated: dict[str, list[dict[str, Any]]] = {}
    for role, namespace in NAMESPACES.items():
        generated[role] = _generate_split(
            role,
            namespace,
            forbidden_layouts,
            forbidden_tasks,
        )
    early_counts = {
        size: (
            config.evaluation.early_seen_per_size
            if size <= config.evaluation.seen_max_size
            else config.evaluation.early_ood_per_size
        )
        for size in SIZES
    }
    generated["air_early"] = _stable_subset(
        generated["air_dev"], early_counts, "air-dev-early210-v1"
    )
    train = read_jsonl(resolve_path(config.paths.train_manifest))
    generated["preflight"] = _stable_subset(
        train,
        {size: 20 for size in range(9, 22, 2)},
        "air-preflight-v1",
    )
    return generated


def _paths(config: Stage0Config) -> dict[str, Path]:
    return {
        "air_dev": resolve_path(config.paths.air_dev_manifest),
        "air_early": resolve_path(config.paths.air_early_manifest),
        "air_select": resolve_path(config.paths.air_select_manifest),
        "air_final": resolve_path(config.paths.air_final_manifest),
        "preflight": resolve_path(config.paths.preflight_manifest),
    }


def validate_generated(generated: dict[str, list[dict[str, Any]]]) -> None:
    for role in ("air_dev", "air_select", "air_final"):
        entries = generated[role]
        counts = Counter(int(entry["maze_size"]) for entry in entries)
        if counts != Counter({size: PER_SIZE for size in SIZES}):
            raise ValueError(f"invalid {role} size distribution: {counts}")
        for key in ("topology_seed", "layout_hash", "task_hash"):
            if len({entry[key] for entry in entries}) != len(entries):
                raise ValueError(f"duplicate {key} in {role}")
    dev_tasks = {entry["task_hash"] for entry in generated["air_dev"]}
    early_tasks = {entry["task_hash"] for entry in generated["air_early"]}
    if not early_tasks < dev_tasks or len(early_tasks) != 210:
        raise ValueError("AIR early210 must be a strict 210-task subset of AIR_dev")
    if len(generated["preflight"]) != 140:
        raise ValueError("AIR preflight must contain 140 train-role tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    from air_jepa.stage0_workspace.common import load_config

    args = parse_args()
    config = load_config(args.config)
    generated = generate_all(config)
    validate_generated(generated)
    for role, path in _paths(config).items():
        expected = _serialize(generated[role])
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"manifest does not reproduce byte-for-byte: {path}")
            print(f"verified={path} tasks={len(generated[role])}")
            continue
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite preregistered manifest: {path}"
            )
        atomic_text_dump(path, expected)
        print(f"saved={path} tasks={len(generated[role])}")


if __name__ == "__main__":
    main()
