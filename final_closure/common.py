"""Protocol, provenance, statistics, and evaluation helpers for final closure."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from final_closure import EXPERIMENT_FAMILY, FORMAT_VERSION, PROTOCOL_ID
from spatial_jepa_planning.common import (
    ACTION_IDS,
    ACTION_TO_SLOT,
    bfs_distances_from,
    canonical_json_sha256,
    create_env,
    next_state,
    observe_state,
    read_jsonl,
    runtime_metadata,
    set_agent_state,
    sha256_file,
    strict_json_dump,
    summarize_rows,
    task_id,
    validate_manifest_entry,
    verify_holdout,
)

ROOT = Path(__file__).resolve().parents[1]
RERUN_REASONS = (
    "interrupted_execution",
    "missing_or_duplicate_task",
    "manifest_checkpoint_or_code_hash_mismatch",
    "non_finite_output",
)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"config protocol is not {PROTOCOL_ID}")
    lock = load_json(config["paths"]["protocol_lock"])
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol lock is not {PROTOCOL_ID}")
    expected_spec = lock.get("analysis_spec_sha256")
    actual_spec = analysis_spec_sha256(config, lock)
    if expected_spec != actual_spec:
        raise ValueError(
            "config no longer matches the independently locked analysis spec: "
            f"{actual_spec} != {expected_spec}"
        )
    return config, lock


def baseline_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in config["baselines"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one baseline named {name!r}")
    return matches[0]


def set_seed(seed: int, *, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS was requested but is unavailable: {requested}")
    return torch.device(name)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool:
    paths = [
        "final_closure",
        "hdwm",
        "diagnostics/common.py",
        "scripts/train/train_dim256.py",
        "scripts/train/train_canonical_lewm.py",
        "scripts/eval/eval_setb_distance_head_fixed.py",
        "spatial_jepa_planning/common.py",
        "data/splits",
        "pyproject.toml",
    ]
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def require_clean_worktree(allow_dirty: bool) -> None:
    if git_worktree_dirty() and not allow_dirty:
        raise RuntimeError(
            "formal final-closure runs require a clean experiment worktree; "
            "commit the package first"
        )


def require_new_output(path: str | Path, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite formal output {path}; overwrite is reserved "
            "for documented execution-failure reruns"
        )


def prepare_rerun(
    paths: Iterable[str | Path],
    *,
    overwrite: bool,
    reason: str,
) -> dict[str, Any] | None:
    outputs = [Path(path) for path in paths]
    existing = [path for path in outputs if path.exists()]
    if not overwrite:
        if reason:
            raise ValueError("--rerun-reason requires --overwrite")
        return None
    if reason not in RERUN_REASONS:
        raise ValueError("overwriting formal output requires an allowed rerun reason")
    if not existing:
        raise ValueError(
            "--overwrite was requested but no selected output exists; rerun the "
            "missing output without an overwrite flag"
        )
    return {
        "reason": reason,
        "superseded_outputs": {
            str(path): sha256_file(path) for path in sorted(existing)
        },
    }


def validate_rerun_record(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} rerun record must be an object")
    reason = value.get("reason")
    if reason not in RERUN_REASONS:
        raise ValueError(f"{label} has an invalid rerun reason")
    outputs = value.get("superseded_outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"{label} rerun record has no superseded outputs")
    for path, digest in outputs.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label} rerun output path is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} rerun SHA256 is invalid")
    return value


def require_study_open(config: dict[str, Any]) -> None:
    gate = Path(config["paths"]["closure_gate"])
    if gate.exists():
        raise RuntimeError(
            f"the final-closure study is immutable after {gate} is created; "
            "start a separately named protocol for future experiments"
        )


def experiment_code_fingerprint() -> str:
    files = list((ROOT / "final_closure").rglob("*.py"))
    files.extend((ROOT / "hdwm").rglob("*.py"))
    files.extend(
        [
            ROOT / "diagnostics/common.py",
            ROOT / "scripts/train/train_dim256.py",
            ROOT / "scripts/train/train_canonical_lewm.py",
            ROOT / "scripts/eval/eval_setb_distance_head_fixed.py",
            ROOT / "spatial_jepa_planning/common.py",
            ROOT / "pyproject.toml",
        ]
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def analysis_spec_sha256(config: dict[str, Any], lock: dict[str, Any]) -> str:
    payload = {
        "schema": "maze-jepa-final-analysis-v1",
        "protocol_id": config["protocol_id"],
        "study_role": config["study_role"],
        "seeds": config["seeds"],
        "manifests": {
            role: lock[role]["sha256"]
            for role in (
                "train_manifest",
                "development_manifest",
                "confirmatory_manifest",
            )
        },
        "protocol": config["protocol"],
        "baselines": config["baselines"],
        "spatial_methods": config["spatial_methods"],
        "analysis": config["analysis"],
        "source_spatial_experiment": lock["source_spatial_experiment"],
    }
    return canonical_json_sha256(payload)


def training_spec_sha256(
    config: dict[str, Any],
    lock: dict[str, Any],
    *,
    name: str,
    seed: int,
) -> str:
    payload = {
        "schema": "maze-jepa-final-training-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "baseline": baseline_config(config, name),
        "seed": int(seed),
        "train_manifest_sha256": lock["train_manifest"]["sha256"],
        "development_is_not_used_for_selection": True,
        "confirmatory_is_not_used_for_selection": True,
    }
    return canonical_json_sha256(payload)


def protocol_metadata(
    config: dict[str, Any],
    lock: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_role": config["study_role"],
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "code_fingerprint": experiment_code_fingerprint(),
        "runtime": runtime_metadata(),
        "device": str(device),
        "seed": int(seed),
        "max_steps": int(config["protocol"]["max_steps"]),
        "action_ids": list(ACTION_IDS),
        "train_manifest_sha256": lock["train_manifest"]["sha256"],
        "development_manifest_sha256": lock["development_manifest"]["sha256"],
        "confirmatory_manifest_sha256": lock["confirmatory_manifest"]["sha256"],
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
    }


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        strict_json_dump(temporary, payload)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    config: dict[str, Any],
    lock: dict[str, Any],
    name: str,
    seed: int,
    strict_provenance: bool,
) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError(f"not a final-closure checkpoint: {path}")
    if int(data.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported final-closure checkpoint format: {path}")
    if data.get("stage") != "baseline_training":
        raise ValueError("checkpoint stage is not baseline_training")
    if data.get("baseline_name") != name or int(data.get("training_seed", -1)) != seed:
        raise ValueError("checkpoint name/seed does not match its run label")
    expected_baseline = baseline_config(config, name)
    if data.get("baseline_kind") != expected_baseline["kind"]:
        raise ValueError("checkpoint baseline kind does not match the locked config")
    validate_rerun_record(data.get("rerun"), f"checkpoint {path}")
    expected_analysis = analysis_spec_sha256(config, lock)
    expected_training = training_spec_sha256(config, lock, name=name, seed=seed)
    if data.get("analysis_spec_sha256") != expected_analysis:
        raise ValueError("checkpoint was not trained under the locked analysis spec")
    if data.get("training_spec_sha256") != expected_training:
        raise ValueError("checkpoint was not trained under the locked training spec")
    protocol = data.get("protocol", {})
    for role in (
        "train_manifest",
        "development_manifest",
        "confirmatory_manifest",
    ):
        if protocol.get(f"{role}_sha256") != lock[role]["sha256"]:
            raise ValueError(f"checkpoint {role} hash mismatch")
    if strict_provenance:
        if protocol.get("git_dirty") is not False:
            raise ValueError("formal evaluation rejects a dirty-training checkpoint")
        if protocol.get("git_commit") != git_commit():
            raise ValueError("training and evaluation must use the same Git commit")
        if protocol.get("code_fingerprint") != experiment_code_fingerprint():
            raise ValueError("training and evaluation code fingerprints differ")
    return data


def pad_bc_observation(observation: np.ndarray, canvas_size: int) -> torch.Tensor:
    size = int(observation.shape[0])
    output_size = max(size, int(canvas_size))
    padded = np.zeros((output_size, output_size, observation.shape[-1]), np.float32)
    padded[:size, :size] = observation
    return torch.from_numpy(padded).permute(2, 0, 1).contiguous()


def corrected_actions(env: Any, state: int, previous: int | None) -> list[int]:
    moving: list[int] = []
    non_backtracking: list[int] = []
    for action in ACTION_IDS:
        candidate = next_state(env, state, action)
        if candidate == state:
            continue
        moving.append(action)
        if previous is None or candidate != previous:
            non_backtracking.append(action)
    return non_backtracking or moving


def count_by_size(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(entry["maze_size"]) for entry in entries)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def validate_task_rows(rows: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} task rows")
    if not all(isinstance(row, dict) and row.get("task_id") for row in rows):
        raise ValueError("every task row must contain task_id")
    identifiers = [str(row["task_id"]) for row in rows]
    if len(set(identifiers)) != expected_count:
        raise ValueError("task rows must contain unique task IDs")
    required = {"success", "spl", "maze_size", "optimal_length", "path_length"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"task row is missing fields: {sorted(missing)}")
        if not 0.0 <= float(row["spl"]) <= 1.0:
            raise ValueError("SPL must lie in [0, 1]")
        if not isinstance(row["success"], bool):
            raise ValueError("success must be a JSON boolean")
        optimal = int(row["optimal_length"])
        path_length = int(row["path_length"])
        if optimal < 0 or not 0 <= path_length <= 128:
            raise ValueError("invalid oracle or executed path length")
        if row["success"]:
            if path_length < optimal:
                raise ValueError("successful path cannot be shorter than oracle BFS")
            expected_spl = optimal / max(optimal, path_length, 1)
            if not math.isclose(float(row["spl"]), expected_spl, abs_tol=1e-12):
                raise ValueError("successful task has inconsistent SPL")
        elif not math.isclose(float(row["spl"]), 0.0, abs_tol=1e-12):
            raise ValueError("failed task must have zero SPL")
        if "invalid_actions" in row:
            invalid = int(row["invalid_actions"])
            if not 0 <= invalid <= path_length:
                raise ValueError("invalid action count exceeds executed path length")
        if "repeat_states" in row:
            repeats = int(row["repeat_states"])
            if not 0 <= repeats <= path_length:
                raise ValueError("repeat-state count exceeds executed path length")
        if "max_state_visits" in row:
            max_visits = int(row["max_state_visits"])
            if not 1 <= max_visits <= path_length + 1:
                raise ValueError(
                    "maximum state visits is inconsistent with path length"
                )
            if "loop_or_cycle" in row and bool(row["loop_or_cycle"]) != (
                max_visits >= 4
            ):
                raise ValueError("loop/cycle flag disagrees with maximum state visits")
        if "episode_seconds" in row:
            episode_seconds = float(row["episode_seconds"])
            if not math.isfinite(episode_seconds) or episode_seconds < 0.0:
                raise ValueError(
                    "episode wall-clock time must be finite and non-negative"
                )
        if "auxiliary" in row:
            auxiliary = row["auxiliary"]
            if not isinstance(auxiliary, dict):
                raise ValueError("task auxiliary metrics must be an object")
            for name, value in auxiliary.items():
                if not isinstance(name, str) or not math.isfinite(float(value)):
                    raise ValueError("task auxiliary metrics must be named and finite")
        if row["success"] and int(row.get("final_bfs_distance", 0)) != 0:
            raise ValueError("successful task must end at BFS distance zero")
    return rows


def crossed_paired_bootstrap(
    candidate_rows: list[list[dict[str, Any]]],
    baseline_rows: list[list[dict[str, Any]]],
    *,
    metric: str,
    samples: int,
    alpha: float,
    seed: int,
    pair_seeds: bool = True,
    task_strata_key: str | None = None,
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows) or not candidate_rows:
        raise ValueError("crossed bootstrap requires matched, non-empty seed lists")
    candidate_matrices: list[np.ndarray] = []
    baseline_matrices: list[np.ndarray] = []
    canonical_ids: list[str] | None = None
    canonical_strata: list[str] | None = None
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        candidate_by_id = {str(row["task_id"]): row for row in candidate}
        baseline_by_id = {str(row["task_id"]): row for row in baseline}
        identifiers = sorted(candidate_by_id)
        if identifiers != sorted(baseline_by_id):
            raise ValueError("paired methods do not contain identical task IDs")
        if canonical_ids is None:
            canonical_ids = identifiers
            if task_strata_key is not None:
                canonical_strata = [
                    str(candidate_by_id[key][task_strata_key]) for key in identifiers
                ]
        elif identifiers != canonical_ids:
            raise ValueError("all seeds must evaluate the same task IDs")
        if task_strata_key is not None:
            candidate_strata = [
                str(candidate_by_id[key][task_strata_key]) for key in identifiers
            ]
            baseline_strata = [
                str(baseline_by_id[key][task_strata_key]) for key in identifiers
            ]
            if (
                candidate_strata != canonical_strata
                or baseline_strata != canonical_strata
            ):
                raise ValueError("paired methods disagree on task strata")
        candidate_matrices.append(
            np.asarray(
                [float(candidate_by_id[key][metric]) for key in identifiers],
                dtype=np.float64,
            )
        )
        baseline_matrices.append(
            np.asarray(
                [float(baseline_by_id[key][metric]) for key in identifiers],
                dtype=np.float64,
            )
        )
    candidate_matrix = np.stack(candidate_matrices)
    baseline_matrix = np.stack(baseline_matrices)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    batch_size = min(256, samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        candidate_seeds = rng.integers(
            0,
            candidate_matrix.shape[0],
            (count, candidate_matrix.shape[0]),
        )
        baseline_seeds = (
            candidate_seeds
            if pair_seeds
            else rng.integers(
                0,
                baseline_matrix.shape[0],
                (count, baseline_matrix.shape[0]),
            )
        )
        if canonical_strata is None:
            selected_tasks = rng.integers(
                0,
                candidate_matrix.shape[1],
                (count, candidate_matrix.shape[1]),
            )
        else:
            strata = np.asarray(canonical_strata)
            stratum_indices = [
                np.flatnonzero(strata == value) for value in sorted(set(strata))
            ]
            selected_tasks = np.concatenate(
                [
                    rng.choice(indices, size=(count, len(indices)), replace=True)
                    for indices in stratum_indices
                ],
                axis=1,
            )
        candidate_seed_mean = candidate_matrix[candidate_seeds].mean(axis=1)
        baseline_seed_mean = baseline_matrix[baseline_seeds].mean(axis=1)
        sampled_difference = candidate_seed_mean - baseline_seed_mean
        draws[start : start + count] = np.take_along_axis(
            sampled_difference, selected_tasks, axis=1
        ).mean(axis=1)
    return {
        "delta": float(candidate_matrix.mean() - baseline_matrix.mean()),
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "alpha": float(alpha),
        "bootstrap_samples": int(samples),
        "seed_resampling": (
            "paired_across_methods" if pair_seeds else "independent_across_methods"
        ),
        "task_resampling": (
            f"paired_by_task_id_within_{task_strata_key}"
            if task_strata_key is not None
            else "paired_by_task_id"
        ),
    }


def mean_std(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean/std requires finite non-empty values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def estimate_forward_macs(
    model: nn.Module,
    example: torch.Tensor,
) -> int:
    """Count Conv2d and Linear multiply-accumulates for one forward pass."""

    total = 0

    def hook(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel = int(module.kernel_size[0] * module.kernel_size[1])
            per_output = kernel * int(module.in_channels // module.groups)
            total += int(output.numel() * per_output)
        elif isinstance(module, nn.Linear):
            total += int(output.numel() * module.in_features)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


def task_seed(evaluation_seed: int, task_index: int, step: int) -> int:
    return int((evaluation_seed * 10_000 + task_index) * 10_000 + step)


def environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


__all__ = [
    "ACTION_IDS",
    "ACTION_TO_SLOT",
    "RERUN_REASONS",
    "ROOT",
    "analysis_spec_sha256",
    "atomic_json_dump",
    "atomic_torch_save",
    "baseline_config",
    "bfs_distances_from",
    "corrected_actions",
    "count_by_size",
    "create_env",
    "crossed_paired_bootstrap",
    "environment_summary",
    "estimate_forward_macs",
    "experiment_code_fingerprint",
    "git_commit",
    "git_worktree_dirty",
    "load_checkpoint",
    "load_config",
    "load_json",
    "mean_std",
    "next_state",
    "observe_state",
    "pad_bc_observation",
    "prepare_rerun",
    "protocol_metadata",
    "read_jsonl",
    "require_clean_worktree",
    "require_new_output",
    "require_study_open",
    "resolve_device",
    "set_agent_state",
    "set_seed",
    "sha256_file",
    "strict_json_dump",
    "summarize_rows",
    "task_id",
    "task_seed",
    "training_spec_sha256",
    "validate_manifest_entry",
    "validate_rerun_record",
    "validate_task_rows",
    "verify_holdout",
]
