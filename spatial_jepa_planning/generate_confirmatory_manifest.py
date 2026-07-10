#!/usr/bin/env python3
"""Generate or verify the preregistered, untouched confirmatory maze split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common import bfs_distances_from
from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    create_env,
    read_jsonl,
)

SIZES = tuple(range(9, 26, 2))
PER_SIZE = 100
TOPOLOGY_NAMESPACE = 1_000_000
OUTPUT = "data/splits/spatial_jepa_confirm_eval_manifest.jsonl"
PRIOR_MANIFESTS = (
    "data/splits/unisize_train_manifest.jsonl",
    "data/splits/unisize_eval_manifest.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def regenerated_layout_hash(entry: dict[str, Any]) -> str:
    return canonical_layout_hash(create_env(entry)._maze_mask)


def prior_layout_hashes(paths: tuple[str, ...] = PRIOR_MANIFESTS) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        for entry in read_jsonl(path):
            layout_hash = regenerated_layout_hash(entry)
            if layout_hash in hashes:
                raise ValueError(
                    f"duplicate canonical layout in prior manifests: {path}"
                )
            hashes.add(layout_hash)
    return hashes


def generate_entries() -> list[dict[str, Any]]:
    forbidden_layouts = prior_layout_hashes()
    accepted_layouts: set[str] = set()
    accepted_tasks: set[str] = set()
    entries: list[dict[str, Any]] = []
    for size in SIZES:
        accepted_for_size = 0
        candidate_index = 0
        while accepted_for_size < PER_SIZE:
            topology_seed = TOPOLOGY_NAMESPACE + size * 10_000 + candidate_index
            env_seed = topology_seed * 100 + 17
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
            if task_hash in accepted_tasks:
                continue
            num_walls = int(env._maze_mask.sum())
            entries.append(
                {
                    **seed_entry,
                    "start_cell": start,
                    "goal_cell": goal,
                    "layout_hash": layout_hash,
                    "task_hash": task_hash,
                    "bfs_path_length": path_length,
                    "num_walls": num_walls,
                    "num_free": size * size - num_walls,
                    "manifest_schema": "spatial-jepa-confirm-v1",
                }
            )
            accepted_layouts.add(layout_hash)
            accepted_tasks.add(task_hash)
            accepted_for_size += 1
    return entries


def serialized(entries: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        for entry in entries
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    expected = serialized(generate_entries())
    if args.check:
        if not output.exists():
            raise FileNotFoundError(output)
        actual = output.read_text(encoding="utf-8")
        if actual != expected:
            raise ValueError(
                f"confirmatory manifest does not reproduce exactly: {output}"
            )
        print(f"verified={output}")
        return
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite preregistered manifest: {output}; use --check"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"saved={output} tasks={len(expected.splitlines())}")


if __name__ == "__main__":
    main()
