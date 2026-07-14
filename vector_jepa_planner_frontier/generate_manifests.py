"""Generate deterministic, topology-disjoint validation and confirmatory splits."""

from __future__ import annotations

import argparse
import json
from typing import Any

from diagnostics.common import bfs_distances_from
from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    create_env,
    read_jsonl,
)
from vector_jepa_planner_frontier.common import resolve_path

ROLE_SPECS = {
    "validation": {
        "sizes": tuple(range(9, 22, 2)),
        "per_size": 100,
        "namespace": 2_000_000,
        "schema": "vector-jepa-frontier-validation-v1",
        "default_output": "data/splits/vector_jepa_frontier_validation.jsonl",
    },
    "confirmatory": {
        "sizes": tuple(range(9, 26, 2)),
        "per_size": 100,
        "namespace": 3_000_000,
        "schema": "vector-jepa-frontier-confirmatory-v1",
        "default_output": "data/splits/vector_jepa_frontier_confirmatory.jsonl",
    },
}
PRIOR_MANIFESTS = (
    "data/splits/unisize_train_manifest.jsonl",
    "data/splits/unisize_eval_manifest.jsonl",
    "data/splits/spatial_jepa_confirm_eval_manifest.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def regenerated_layout_hash(entry: dict[str, Any]) -> str:
    return canonical_layout_hash(create_env(entry)._maze_mask)


def prior_layout_hashes(paths: tuple[str, ...]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"required prior manifest is missing: {resolved}")
        for entry in read_jsonl(resolved):
            layout_hash = regenerated_layout_hash(entry)
            if layout_hash in hashes:
                raise ValueError(
                    f"duplicate canonical layout in prior manifests: {path}"
                )
            hashes.add(layout_hash)
    return hashes


def generate_entries(role: str) -> list[dict[str, Any]]:
    if role not in ROLE_SPECS:
        raise ValueError(f"unknown manifest role: {role}")
    spec = ROLE_SPECS[role]
    prior = list(PRIOR_MANIFESTS)
    if role == "confirmatory":
        prior.append("data/splits/vector_jepa_frontier_validation.jsonl")
    forbidden_layouts = prior_layout_hashes(tuple(prior))
    accepted_layouts: set[str] = set()
    accepted_tasks: set[str] = set()
    entries: list[dict[str, Any]] = []
    for size in spec["sizes"]:
        accepted_for_size = 0
        candidate_index = 0
        while accepted_for_size < spec["per_size"]:
            topology_seed = int(spec["namespace"] + size * 10_000 + candidate_index)
            env_seed = topology_seed * 100 + 29
            seed_entry = {
                "maze_size": int(size),
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
            distances = bfs_distances_from(env._maze_mask, goal, int(size))
            path_length = int(distances[start])
            if start == goal or path_length <= 0:
                continue
            task_hash = canonical_task_hash(
                maze_size=int(size),
                layout_hash=layout_hash,
                start_cell=start,
                goal_cell=goal,
            )
            if task_hash in accepted_tasks:
                continue
            wall_count = int(env._maze_mask.sum())
            entries.append(
                {
                    **seed_entry,
                    "start_cell": start,
                    "goal_cell": goal,
                    "layout_hash": layout_hash,
                    "task_hash": task_hash,
                    "bfs_path_length": path_length,
                    "num_walls": wall_count,
                    "num_free": int(size * size - wall_count),
                    "manifest_schema": spec["schema"],
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
    spec = ROLE_SPECS[args.role]
    output = resolve_path(args.output or spec["default_output"])
    expected = serialized(generate_entries(args.role))
    if args.check:
        if not output.exists():
            raise FileNotFoundError(output)
        if output.read_text(encoding="utf-8") != expected:
            raise ValueError(f"{args.role} manifest does not reproduce exactly")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite locked manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
