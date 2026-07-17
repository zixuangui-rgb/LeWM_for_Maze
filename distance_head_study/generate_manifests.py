"""Generate and byte-check every DistanceHead study manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from diagnostics.common import bfs_distances_from
from distance_head_study.common import read_jsonl, resolve_path
from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    create_env,
)

ROLE_SPECS: dict[str, dict[str, Any]] = {
    "cal": {
        "schema": "distance-head-calibration-v1",
        "namespace": None,
        "sizes": tuple(range(9, 22, 2)),
        "per_size": 20,
        "output": "distance_head_study/manifests/d_cal.jsonl",
    },
    "screen": {
        "schema": "distance-head-screen-v1",
        "namespace": 4_000_000,
        "sizes": tuple(range(9, 22, 2)),
        "per_size": 20,
        "output": "distance_head_study/manifests/d_screen.jsonl",
    },
    "select": {
        "schema": "distance-head-select-v1",
        "namespace": 5_000_000,
        "sizes": tuple(range(9, 22, 2)),
        "per_size": 30,
        "output": "distance_head_study/manifests/d_select.jsonl",
    },
    "confirm": {
        "schema": "distance-head-confirm-v1",
        "namespace": 6_000_000,
        "sizes": tuple(range(9, 26, 2)),
        "per_size": 100,
        "output": "distance_head_study/manifests/d_confirm.jsonl",
    },
    "stress": {
        "schema": "distance-head-stress-v1",
        "namespace": 7_000_000,
        "sizes": (27, 29, 31),
        "per_size": 50,
        "output": "distance_head_study/manifests/d_stress.jsonl",
    },
}

PRIOR_MANIFESTS = (
    "data/splits/unisize_train_manifest.jsonl",
    "data/splits/unisize_eval_manifest.jsonl",
    "data/splits/spatial_jepa_confirm_eval_manifest.jsonl",
    "data/splits/vector_jepa_frontier_validation.jsonl",
    "data/splits/vector_jepa_frontier_confirmatory.jsonl",
)


def _stable_calibration_order(entry: dict[str, Any]) -> str:
    key = f"distance-head-cal-v1:{entry['task_hash']}".encode("ascii")
    return hashlib.sha256(key).hexdigest()


def calibration_entries() -> list[dict[str, Any]]:
    """Select training topologies deterministically, not for generalization."""

    source = read_jsonl("data/splits/unisize_train_manifest.jsonl")
    spec = ROLE_SPECS["cal"]
    entries: list[dict[str, Any]] = []
    for size in spec["sizes"]:
        candidates = [row for row in source if int(row["maze_size"]) == size]
        candidates.sort(key=_stable_calibration_order)
        if len(candidates) < spec["per_size"]:
            raise ValueError(f"insufficient training topologies for D_cal size={size}")
        for row in candidates[: spec["per_size"]]:
            entries.append(
                {
                    **row,
                    "manifest_schema": spec["schema"],
                    "split_role": "cal",
                    "source_role": "train_topology_calibration_only",
                }
            )
    return entries


def _regenerated_layouts(paths: Iterable[str | Path]) -> set[str]:
    layouts: set[str] = set()
    for path in paths:
        resolved = resolve_path(path)
        if not resolved.exists():
            continue
        local_layouts: set[str] = set()
        for entry in read_jsonl(resolved):
            layout = canonical_layout_hash(create_env(entry)._maze_mask)
            if layout in local_layouts:
                raise ValueError(f"duplicate layout inside prior manifest {path}")
            local_layouts.add(layout)
            layouts.add(layout)
    return layouts


def generated_entries(role: str) -> list[dict[str, Any]]:
    if role not in ROLE_SPECS or role == "cal":
        raise ValueError(f"role does not use topology generation: {role}")
    spec = ROLE_SPECS[role]
    earlier_roles = list(ROLE_SPECS).index(role)
    earlier_paths = [
        ROLE_SPECS[name]["output"] for name in list(ROLE_SPECS)[:earlier_roles]
    ]
    forbidden = _regenerated_layouts((*PRIOR_MANIFESTS, *earlier_paths))
    accepted_layouts: set[str] = set()
    accepted_tasks: set[str] = set()
    entries: list[dict[str, Any]] = []
    for size in spec["sizes"]:
        accepted = 0
        candidate_index = 0
        while accepted < spec["per_size"]:
            topology_seed = int(spec["namespace"] + size * 10_000 + candidate_index)
            candidate_index += 1
            entry = {
                "maze_size": int(size),
                "topology_seed": topology_seed,
                "env_seed": topology_seed * 100 + 37,
            }
            env = create_env(entry)
            _, info = env.reset()
            start = int(info["state"])
            goal = int(env._goal_position)
            layout = canonical_layout_hash(env._maze_mask)
            if layout in forbidden or layout in accepted_layouts:
                continue
            distances = bfs_distances_from(env._maze_mask, goal, int(size))
            path_length = int(distances[start])
            if start == goal or path_length <= 0:
                continue
            task_hash = canonical_task_hash(
                maze_size=int(size),
                layout_hash=layout,
                start_cell=start,
                goal_cell=goal,
            )
            if task_hash in accepted_tasks:
                continue
            walls = int(env._maze_mask.sum())
            entries.append(
                {
                    **entry,
                    "start_cell": start,
                    "goal_cell": goal,
                    "layout_hash": layout,
                    "task_hash": task_hash,
                    "bfs_path_length": path_length,
                    "num_walls": walls,
                    "num_free": int(size * size - walls),
                    "manifest_schema": spec["schema"],
                    "split_role": role,
                    "source_role": "fresh_topology_holdout",
                }
            )
            accepted_layouts.add(layout)
            accepted_tasks.add(task_hash)
            accepted += 1
    return entries


def generate_entries(role: str) -> list[dict[str, Any]]:
    return calibration_entries() if role == "cal" else generated_entries(role)


def serialized(entries: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        for entry in entries
    )


def validate_entries(role: str, entries: list[dict[str, Any]]) -> None:
    spec = ROLE_SPECS[role]
    expected = {int(size): int(spec["per_size"]) for size in spec["sizes"]}
    counts = Counter(int(entry["maze_size"]) for entry in entries)
    if dict(counts) != expected:
        raise ValueError(f"{role} size counts differ: {dict(counts)} != {expected}")
    layouts = [str(entry["layout_hash"]) for entry in entries]
    tasks = [str(entry["task_hash"]) for entry in entries]
    if len(layouts) != len(set(layouts)) or len(tasks) != len(set(tasks)):
        raise ValueError(f"{role} contains duplicate layouts or tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(ROLE_SPECS) + ("all",), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def process(role: str, *, check: bool) -> None:
    entries = generate_entries(role)
    validate_entries(role, entries)
    output = resolve_path(ROLE_SPECS[role]["output"])
    expected = serialized(entries)
    if check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            raise ValueError(f"{role} manifest does not reproduce byte-for-byte")
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite manifest: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8")


def main() -> None:
    args = parse_args()
    roles = tuple(ROLE_SPECS) if args.role == "all" else (args.role,)
    for role in roles:
        process(role, check=bool(args.check))
        print(f"{role}: {'checked' if args.check else 'written'}")


if __name__ == "__main__":
    main()
