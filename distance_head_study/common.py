"""Shared provenance, hashing, runtime, and immutable-output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distance_head_study import EXPERIMENT_FAMILY, FORMAT_VERSION, PROTOCOL_ID
from distance_head_study.schemas import MethodCatalog, StudyConfig

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_hash_bindings(*mappings: dict[str, str]) -> dict[str, str]:
    """Flatten evidence maps while rejecting contradictory file bindings."""

    merged: dict[str, str] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError("evidence hash binding must be a mapping")
        for path, value in mapping.items():
            if not isinstance(path, str) or not path or not isinstance(value, str):
                raise ValueError("evidence hash binding is malformed")
            if path in merged and merged[path] != value:
                raise ValueError(f"conflicting evidence hashes for {path}")
            merged[path] = value
    return merged


def load_json(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(resolve_path(path), encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def load_study_config(path: str | Path) -> StudyConfig:
    return StudyConfig.model_validate(load_json(path))


def load_method_catalog(path: str | Path) -> MethodCatalog:
    return MethodCatalog.model_validate(load_json(path))


def _atomic_path(output: Path) -> tuple[int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    return handle, Path(name)


def atomic_json_dump(path: str | Path, value: Any) -> None:
    output = resolve_path(path)
    handle, temporary = _atomic_path(output)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(value),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text_dump(path: str | Path, value: str) -> None:
    output = resolve_path(path)
    handle, temporary = _atomic_path(output)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, value: dict[str, Any]) -> None:
    output = resolve_path(path)
    handle, temporary = _atomic_path(output)
    os.close(handle)
    try:
        torch.save(value, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool:
    guarded = (
        "distance_head_study",
        "hdwm",
        "final_closure",
        "diagnostics",
        "scripts/train/train_dim256.py",
        "spatial_jepa_planning",
        "vector_jepa_planner_frontier",
        "pyproject.toml",
        "uv.lock",
    )
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *guarded],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def require_clean_worktree(*, allow_dirty: bool = False) -> None:
    if git_worktree_dirty() and not allow_dirty:
        raise RuntimeError(
            "formal DistanceHead runs require a clean committed experiment worktree"
        )


def experiment_code_files() -> tuple[Path, ...]:
    """Enumerate the complete scientific implementation boundary."""

    files = [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "protocol_lock.json"
        and "manifests" not in path.parts
    ]
    dependency_roots = (
        ROOT / "diagnostics",
        ROOT / "final_closure",
        ROOT / "hdwm",
        ROOT / "spatial_jepa_planning",
        ROOT / "vector_jepa_planner_frontier",
    )
    for dependency_root in dependency_roots:
        files.extend(dependency_root.rglob("*.py"))
    dependencies = (
        ROOT / "scripts/train/train_dim256.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    files.extend(path for path in dependencies if path.exists())
    return tuple(sorted(set(files)))


def experiment_code_fingerprint() -> str:
    """Hash this package and every imported scientific implementation boundary."""

    digest = hashlib.sha256()
    for path in experiment_code_files():
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def set_seed(seed: int, *, deterministic: bool = True) -> None:
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


def hierarchical_seed(namespace: str, *values: int) -> int:
    payload = (
        namespace.encode("utf-8")
        + b":"
        + b":".join(str(int(value)).encode("ascii") for value in values)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def resolve_device(requested: str) -> torch.device:
    selected = requested.strip().lower()
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    if selected.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS requested but unavailable: {requested}")
    return torch.device(selected)


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def manifest_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    rows = read_jsonl(resolved)
    sizes: dict[str, int] = {}
    for row in rows:
        key = str(int(row["maze_size"]))
        sizes[key] = sizes.get(key, 0) + 1
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(resolved),
        "count": len(rows),
        "counts_by_size": dict(sorted(sizes.items(), key=lambda item: int(item[0]))),
    }


def prepare_immutable_file(path: str | Path) -> Path:
    output = resolve_path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def prepare_run_directory(path: str | Path, *, resume: bool = False) -> Path:
    output = resolve_path(path)
    if output.exists() and not resume:
        raise FileExistsError(f"formal run directory already exists: {output}")
    if resume and not output.exists():
        raise FileNotFoundError(f"cannot resume missing run directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    return output


def source_backbone_path(config: StudyConfig, backbone_seed: int) -> Path:
    template = (
        config.paths.legacy_backbone_template
        if backbone_seed in config.seeds.historical_backbones
        else config.paths.fresh_backbone_template
    )
    return resolve_path(
        template.format(
            checkpoint_root=config.paths.checkpoint_root,
            backbone_seed=int(backbone_seed),
        )
    )


def validate_backbone_protocol_binding(
    config: StudyConfig,
    payload: dict[str, Any],
    *,
    backbone_seed: int,
    protocol_lock: dict[str, Any],
) -> str:
    """Validate fresh-study backbones while preserving immutable legacy assets."""
    seed = int(backbone_seed)
    if seed in config.seeds.historical_backbones:
        from final_closure.common import (
            EXPERIMENT_FAMILY as SOURCE_EXPERIMENT_FAMILY,
        )
        from final_closure.common import (
            FORMAT_VERSION as SOURCE_FORMAT_VERSION,
        )
        from final_closure.common import (
            analysis_spec_sha256 as source_analysis_spec_sha256,
        )
        from final_closure.common import baseline_config, validate_rerun_record
        from final_closure.common import (
            training_spec_sha256 as source_training_spec_sha256,
        )

        source_config = load_json(config.paths.source_config)
        source_lock = load_json(config.paths.source_lock)
        baseline_name = "lewm_l2_cem_seqlen2"
        baseline = baseline_config(source_config, baseline_name)
        expected = {
            "experiment_family": SOURCE_EXPERIMENT_FAMILY,
            "format_version": SOURCE_FORMAT_VERSION,
            "stage": "baseline_training",
            "baseline_name": baseline_name,
            "baseline_kind": baseline["kind"],
            "training_seed": seed,
            "formal_run": True,
            "analysis_spec_sha256": source_analysis_spec_sha256(
                source_config, source_lock
            ),
            "training_spec_sha256": source_training_spec_sha256(
                source_config,
                source_lock,
                name=baseline_name,
                seed=seed,
            ),
            "training_config": baseline["train"],
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        protocol = payload.get("protocol")
        if not isinstance(protocol, dict):
            mismatches["protocol"] = {
                "expected": "source manifest bindings",
                "observed": protocol,
            }
        else:
            for role in (
                "train_manifest",
                "development_manifest",
                "confirmatory_manifest",
            ):
                expected_hash = source_lock[role]["sha256"]
                observed_hash = protocol.get(f"{role}_sha256")
                if observed_hash != expected_hash:
                    mismatches[f"protocol.{role}_sha256"] = {
                        "expected": expected_hash,
                        "observed": observed_hash,
                    }
        if "model_config" not in payload or "model_state_dict" not in payload:
            mismatches["model_payload"] = {
                "expected": "model_config and model_state_dict",
                "observed": sorted(
                    key
                    for key in ("model_config", "model_state_dict")
                    if key in payload
                ),
            }
        validate_rerun_record(payload.get("rerun"), "historical backbone")
        if mismatches:
            raise ValueError(
                "historical backbone source binding differs: "
                f"{canonical_json_sha256(mismatches)}"
            )
        return "historical"
    if seed not in config.seeds.ordered_confirmation_backbones:
        raise ValueError(f"backbone seed is outside every registered pool: {seed}")
    from final_closure.common import baseline_config

    source_config = load_json(config.paths.source_config)
    source_lock = load_json(config.paths.source_lock)
    baseline_name = "lewm_l2_cem_seqlen2"
    baseline = baseline_config(source_config, baseline_name)
    fresh_source_spec = canonical_json_sha256(
        {
            "schema": "distance-head-source-backbone-v1",
            "source_protocol_id": source_config["protocol_id"],
            "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
            "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
            "baseline": baseline,
            "fresh_seed": seed,
        }
    )
    expected = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "protocol_id": config.protocol_id,
        "stage": "fresh_source_backbone",
        "baseline_name": baseline_name,
        "baseline_kind": baseline["kind"],
        "formal_run": True,
        "training_seed": seed,
        "analysis_spec_sha256": protocol_lock["analysis_spec_sha256"],
        "protocol_lock_sha256": protocol_lock["protocol_lock_sha256"],
        "source_training_spec_sha256": fresh_source_spec,
        "source_analysis_spec_sha256": source_lock["analysis_spec_sha256"],
        "source_train_manifest_sha256": source_lock["train_manifest"]["sha256"],
        "training_config": baseline["train"],
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    training = payload.get("training")
    if not isinstance(training, dict) or int(
        training.get("optimizer_steps", -1)
    ) != int(baseline["train"]["steps"]):
        mismatches["training.optimizer_steps"] = {
            "expected": int(baseline["train"]["steps"]),
            "observed": (
                training.get("optimizer_steps") if isinstance(training, dict) else None
            ),
        }
    if "model_config" not in payload or "model_state_dict" not in payload:
        mismatches["model_payload"] = {
            "expected": "model_config and model_state_dict",
            "observed": sorted(
                key for key in ("model_config", "model_state_dict") if key in payload
            ),
        }
    if mismatches:
        raise ValueError(
            "fresh backbone protocol binding differs: "
            f"{canonical_json_sha256(mismatches)}"
        )
    return "fresh_confirmation"


def head_checkpoint_path(
    config: StudyConfig,
    *,
    method: str,
    backbone_seed: int,
    head_seed: int,
) -> Path:
    return resolve_path(
        config.paths.head_checkpoint_template.format(
            checkpoint_root=config.paths.checkpoint_root,
            method=method,
            backbone_seed=int(backbone_seed),
            head_seed=int(head_seed),
        )
    )


def historical_seed_registry(config: StudyConfig) -> set[int]:
    registry = load_json(config.paths.seed_registry)
    values = {int(seed) for seed in registry.get("historical_backbone_seeds", [])}
    source_paths = (
        ROOT / "final_closure/configs/default.json",
        ROOT / "spatial_jepa_planning/configs/default.json",
        ROOT / "vector_jepa_planner_frontier/configs/default.json",
        ROOT / "vector_jepa_planner_full900_screen/configs/default.json",
    )
    for path in source_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        candidates: list[Any] = []
        if isinstance(payload.get("seeds"), list):
            candidates.extend(payload["seeds"])
        protocol = payload.get("protocol", {})
        if isinstance(protocol, dict) and isinstance(
            protocol.get("training_seeds"), list
        ):
            candidates.extend(protocol["training_seeds"])
        replication = payload.get("replication", {})
        if isinstance(replication, dict):
            for key in (
                "screen_backbone_seeds",
                "expansion_backbone_seeds",
                "final_backbone_seeds",
            ):
                if isinstance(replication.get(key), list):
                    candidates.extend(replication[key])
        values.update(int(seed) for seed in candidates)
    return values


def validate_confirmation_seed_freshness(config: StudyConfig) -> dict[str, Any]:
    historical = historical_seed_registry(config)
    ordered = tuple(config.seeds.ordered_confirmation_backbones)
    overlap = historical & set(ordered)
    if overlap:
        raise ValueError(f"fresh confirmation seed collision: {sorted(overlap)}")
    if ordered != tuple(range(ordered[0], ordered[0] + len(ordered))):
        raise ValueError("ordered confirmation seeds must be a contiguous prefix")
    return {
        "historical_count": len(historical),
        "historical_sha256": canonical_json_sha256(sorted(historical)),
        "ordered_confirmation": list(ordered),
        "overlap": [],
    }


def protocol_metadata(
    config: StudyConfig,
    lock: dict[str, Any],
    *,
    stage: str,
    split_role: str,
    backbone_seed: int,
    head_seed: int,
    method_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "code_fingerprint": experiment_code_fingerprint(),
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "runtime": runtime_metadata(),
        "stage": stage,
        "split_role": split_role,
        "backbone_seed": int(backbone_seed),
        "head_seed": int(head_seed),
        "method_sha256": method_sha256,
        "device": str(device),
    }


def require_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in paths:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        records[resolved.relative_to(ROOT).as_posix()] = sha256_file(resolved)
    return records


__all__ = [
    "PACKAGE_ROOT",
    "ROOT",
    "atomic_json_dump",
    "atomic_text_dump",
    "atomic_torch_save",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "experiment_code_fingerprint",
    "git_commit",
    "git_worktree_dirty",
    "head_checkpoint_path",
    "hierarchical_seed",
    "historical_seed_registry",
    "load_json",
    "load_method_catalog",
    "load_study_config",
    "manifest_record",
    "prepare_immutable_file",
    "prepare_run_directory",
    "protocol_metadata",
    "read_jsonl",
    "require_clean_worktree",
    "require_hashes",
    "resolve_device",
    "resolve_path",
    "runtime_metadata",
    "set_seed",
    "sha256_file",
    "source_backbone_path",
    "validate_backbone_protocol_binding",
    "validate_confirmation_seed_freshness",
]
