"""Protocol/package lock construction and fail-closed verification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from air_jepa.stage0_workspace import AIR_METHODS, ALL_METHODS, SYSTEM_SEEDS
from air_jepa.stage0_workspace.common import (
    PACKAGE_LOCK_SCHEMA,
    PROTOCOL_LOCK_SCHEMA,
    atomic_json_dump,
    canonical_json_sha256,
    code_fingerprint,
    load_config,
    package_files,
    read_json,
    read_jsonl,
    relative_path,
    resolve_path,
    sha256_file,
    signed_payload,
    verify_signature,
)
from air_jepa.stage0_workspace.data import LOCKED_TRAIN_COUNTS
from air_jepa.stage0_workspace.generate_manifests import SIZES
from air_jepa.stage0_workspace.schemas import Stage0Config
from spatial_jepa_planning.common import (
    canonical_layout_hash,
    canonical_task_hash,
    validate_manifest_entry,
)

ALLOWED_EVALUATION_ROLES = frozenset(
    {"preflight", "air_early", "air_dev", "historical"}
)
SEALED_ROLES = frozenset({"air_select", "air_final"})


def manifest_summary(path: str | Path, *, regenerate: bool = True) -> dict[str, Any]:
    resolved = resolve_path(path)
    entries = read_jsonl(resolved)
    if not entries:
        raise ValueError(f"empty manifest: {resolved}")
    task_ids = [str(entry.get("task_hash")) for entry in entries]
    layouts = [str(entry.get("layout_hash")) for entry in entries]
    topologies = [
        (int(entry["maze_size"]), int(entry["topology_seed"])) for entry in entries
    ]
    if len(set(task_ids)) != len(entries) or "None" in task_ids:
        raise ValueError(f"missing or duplicate task hashes: {resolved}")
    if len(set(layouts)) != len(entries) or "None" in layouts:
        raise ValueError(f"missing or duplicate layout hashes: {resolved}")
    if len(set(topologies)) != len(entries):
        raise ValueError(f"duplicate topology identities: {resolved}")
    if regenerate:
        for entry in entries:
            validate_manifest_entry(entry, check_bfs=True)
    return {
        "path": relative_path(resolved),
        "sha256": sha256_file(resolved),
        "rows": len(entries),
        "by_size": {
            str(size): count
            for size, count in sorted(
                Counter(int(entry["maze_size"]) for entry in entries).items()
            )
        },
        "task_set_sha256": canonical_json_sha256(sorted(task_ids)),
        "layout_set_sha256": canonical_json_sha256(sorted(layouts)),
    }


def _identity_sets(path: str | Path) -> dict[str, set[Any]]:
    entries = read_jsonl(path)
    canonical_layouts: set[str] = set()
    canonical_tasks: set[str] = set()
    for entry in entries:
        env = validate_manifest_entry(entry, check_bfs=True)
        layout = canonical_layout_hash(env._maze_mask)
        canonical_layouts.add(layout)
        canonical_tasks.add(
            canonical_task_hash(
                maze_size=int(entry["maze_size"]),
                layout_hash=layout,
                start_cell=int(entry["start_cell"]),
                goal_cell=int(entry["goal_cell"]),
            )
        )
    return {
        "topology": {
            (int(entry["maze_size"]), int(entry["topology_seed"])) for entry in entries
        },
        "layout": canonical_layouts,
        "task": canonical_tasks,
    }


def require_disjoint_manifests(paths: dict[str, str | Path]) -> dict[str, Any]:
    identities = {name: _identity_sets(path) for name, path in paths.items()}
    comparisons: dict[str, Any] = {}
    names = tuple(paths)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = {
                key: len(identities[left][key] & identities[right][key])
                for key in ("topology", "layout", "task")
            }
            if any(overlap.values()):
                raise ValueError(f"manifest leakage {left}<->{right}: {overlap}")
            comparisons[f"{left}__{right}"] = overlap
    return comparisons


def expected_matrix(config: Stage0Config) -> dict[str, Any]:
    k_values = list(config.evaluation.k_values)
    full_curve = ["j1_receding", *AIR_METHODS]
    unmasked = [
        {"method": method, "seed": seed, "k": k, "action_protocol": "unmasked"}
        for seed in SYSTEM_SEEDS
        for method in full_curve
        for k in k_values
    ]
    static = [
        {"method": "j0_static", "seed": seed, "k": 4, "action_protocol": "unmasked"}
        for seed in SYSTEM_SEEDS
    ] + [
        {"method": "j1_static", "seed": seed, "k": 128, "action_protocol": "unmasked"}
        for seed in SYSTEM_SEEDS
    ]
    corrected = [
        {
            "method": method,
            "seed": seed,
            "k": 4 if method == "j0_static" else 128,
            "action_protocol": "corrected",
        }
        for seed in SYSTEM_SEEDS
        for method in ALL_METHODS
    ]
    return {
        "train": [
            {"method": method, "seed": seed, "steps": config.training.steps}
            for seed in SYSTEM_SEEDS
            for method in AIR_METHODS
        ],
        "historical_bridges": [
            {
                "method": method,
                "seed": seed,
                "k": k,
                "action_protocol": "unmasked",
            }
            for seed in SYSTEM_SEEDS
            for method, k in (("j0_static", 4), ("j1_static", 128))
        ],
        "air_dev_unmasked": unmasked + static,
        "air_dev_corrected": corrected,
        "air_early_context": [
            {
                "method": method,
                "seed": 42,
                "k": k,
                "action_protocol": "unmasked",
                "intervention": "normal",
            }
            for method in ("j1_receding", "air0_direct")
            for k in (16, 128)
        ],
        "air_early_interventions": [
            {
                "method": "air0_jepa",
                "seed": seed,
                "k": k,
                "action_protocol": "unmasked",
                "intervention": intervention,
            }
            for seed in SYSTEM_SEEDS
            for k in (16, 128)
            for intervention in (
                "normal",
                "copy_current",
                "true_future",
                "future_permutation",
                "future_zero",
            )
        ],
        "air_early_diagnostics": [
            {
                "method": "air0_jepa",
                "seed": 42,
                "k": 128,
                "states_per_maze": 24,
            }
        ],
        "diagnostics": [
            {"method": "air0_jepa", "seed": seed, "k": 128, "states_per_maze": 24}
            for seed in SYSTEM_SEEDS
        ],
        "evaluator_oracle": [
            {
                "method": "oracle_bfs",
                "split_role": "air_dev",
                "action_protocol": "unmasked",
                "max_steps": config.evaluation.max_steps,
            }
        ],
    }


def build_protocol_payload(config: Stage0Config) -> dict[str, Any]:
    paths = {
        "train": config.paths.train_manifest,
        "historical_dev": config.paths.historical_development_manifest,
        "historical_confirm": config.paths.historical_confirmatory_manifest,
        "air_dev": config.paths.air_dev_manifest,
        "air_select": config.paths.air_select_manifest,
        "air_final": config.paths.air_final_manifest,
    }
    summaries = {
        name: manifest_summary(path, regenerate=True) for name, path in paths.items()
    }
    expected_eval_counts = {str(size): 100 for size in SIZES}
    if summaries["train"]["rows"] != 2800 or summaries["train"]["by_size"] != {
        str(size): count for size, count in LOCKED_TRAIN_COUNTS.items()
    }:
        raise ValueError("locked training manifest must have 400 tasks per size 9..21")
    for role in ("historical_dev", "historical_confirm"):
        if (
            summaries[role]["rows"] != 900
            or summaries[role]["by_size"] != expected_eval_counts
        ):
            raise ValueError(f"{role} must have 100 tasks per locked size")
    for role in ("air_dev", "air_select", "air_final"):
        if summaries[role]["rows"] != 900:
            raise ValueError(f"{role} must have 900 tasks")
        if summaries[role]["by_size"] != expected_eval_counts:
            raise ValueError(f"{role} must have 100 tasks per locked size")
    overlaps = require_disjoint_manifests(paths)
    early = manifest_summary(config.paths.air_early_manifest, regenerate=True)
    preflight = manifest_summary(config.paths.preflight_manifest, regenerate=True)
    early_ids = _identity_sets(config.paths.air_early_manifest)["task"]
    dev_ids = _identity_sets(config.paths.air_dev_manifest)["task"]
    if len(early_ids) != 210 or not early_ids < dev_ids:
        raise ValueError("air_early must be a strict 210-task subset of air_dev")
    train_ids = _identity_sets(config.paths.train_manifest)["task"]
    preflight_ids = _identity_sets(config.paths.preflight_manifest)["task"]
    if len(preflight_ids) != 140 or not preflight_ids < train_ids:
        raise ValueError("preflight must be a strict 140-task train subset")
    config_path = resolve_path(config.paths.protocol_lock).parent / "default.json"
    return signed_payload(
        {
            "schema": PROTOCOL_LOCK_SCHEMA,
            "experiment_id": config.experiment_id,
            "config_path": relative_path(config_path),
            "config_sha256": sha256_file(config_path),
            "manifests": {**summaries, "air_early": early, "preflight": preflight},
            "pairwise_holdout": overlaps,
            "allowed_evaluation_roles": sorted(ALLOWED_EVALUATION_ROLES),
            "sealed_roles": sorted(SEALED_ROLES),
            "matrix": expected_matrix(config),
        },
        "protocol_sha256",
    )


def verify_protocol_lock(config: Stage0Config) -> dict[str, Any]:
    lock = read_json(config.paths.protocol_lock)
    if not isinstance(lock, dict) or lock.get("schema") != PROTOCOL_LOCK_SCHEMA:
        raise ValueError("invalid AIR protocol lock schema")
    verify_signature(lock, "protocol_sha256")
    expected = build_protocol_payload(config)
    if lock != expected:
        raise ValueError("protocol lock differs from regenerated protocol")
    return lock


def build_package_payload() -> dict[str, Any]:
    files = {relative_path(path): sha256_file(path) for path in package_files()}
    return signed_payload(
        {
            "schema": PACKAGE_LOCK_SCHEMA,
            "code_fingerprint": code_fingerprint(),
            "files": files,
        },
        "package_sha256",
    )


def verify_package_lock(config: Stage0Config) -> dict[str, Any]:
    lock = read_json(config.paths.package_lock)
    if not isinstance(lock, dict) or lock.get("schema") != PACKAGE_LOCK_SCHEMA:
        raise ValueError("invalid AIR package lock schema")
    verify_signature(lock, "package_sha256")
    expected = build_package_payload()
    if lock != expected:
        raise ValueError("package contents changed after package lock creation")
    return lock


def write_protocol_lock(config: Stage0Config, *, check: bool) -> None:
    output = resolve_path(config.paths.protocol_lock)
    payload = build_protocol_payload(config)
    if check:
        if read_json(output) != payload:
            raise ValueError("protocol lock check failed")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol lock: {output}")
    atomic_json_dump(output, payload)


def write_package_lock(config: Stage0Config, *, check: bool) -> None:
    output = resolve_path(config.paths.package_lock)
    payload = build_package_payload()
    if check:
        if read_json(output) != payload:
            raise ValueError("package lock check failed")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite package lock: {output}")
    atomic_json_dump(output, payload)


def require_role_allowed(role: str) -> None:
    if role in SEALED_ROLES:
        raise PermissionError(
            f"{role} is sealed for AIR0-v1 and cannot be evaluated by this package"
        )
    if role not in ALLOWED_EVALUATION_ROLES:
        raise ValueError(f"unknown or unauthorized evaluation role: {role}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target", choices=("protocol", "package"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    if args.target == "protocol":
        write_protocol_lock(config, check=args.check)
    else:
        write_package_lock(config, check=args.check)
    print(f"verified={args.target}" if args.check else f"locked={args.target}")


if __name__ == "__main__":
    main()
