"""Protocol, provenance, locking, and compute-accounting helpers."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from final_closure.common import RERUN_REASONS, prepare_rerun
from spatial_jepa_planning.common import (
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    strict_json_dump,
    verify_holdout,
)
from vector_jepa_planner_frontier import (
    EXPERIMENT_FAMILY,
    FORMAT_VERSION,
    PROTOCOL_ID,
)
from vector_jepa_planner_frontier.schemas import MethodConfig, StudyConfig

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_study_config(path: str | Path) -> StudyConfig:
    return StudyConfig.model_validate(load_json(path))


def method_by_name(config: StudyConfig, name: str) -> MethodConfig:
    matches = [method for method in config.methods if method.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one configured method named {name!r}")
    return matches[0]


def planner_seed_values(config: StudyConfig, method: MethodConfig) -> tuple[int, ...]:
    """Return true head seeds, or one non-replicated sentinel for headless methods."""

    return (
        config.protocol.planner_seeds if method.component_checkpoint_required else (0,)
    )


def uses_counterexample_rounds(method: MethodConfig) -> bool:
    return bool(
        method.stage == "P6"
        and method.scorer.counterexample_ranker_weight > 0.0
        and method.control.ranker_negatives in {"hard_three_rounds", "random"}
    )


def component_checkpoint_owner(
    config: StudyConfig, method: MethodConfig
) -> MethodConfig:
    """Return the method whose frozen component artifact an evaluation loads."""

    owner = method
    visited = {method.name}
    while owner.reuse_component_from is not None:
        source_name = owner.reuse_component_from
        if source_name in visited:
            raise ValueError("component checkpoint reuse contains a cycle")
        visited.add(source_name)
        owner = method_by_name(config, source_name)
    return owner


def component_checkpoint_path(
    config: StudyConfig,
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
) -> Path | None:
    """Resolve the calibrated or final three-round checkpoint for evaluation."""

    if not method.component_checkpoint_required:
        return None
    owner = component_checkpoint_owner(config, method)
    counterexample_rounds = uses_counterexample_rounds(owner)
    template = (
        config.paths.counterexample_round_template
        if counterexample_rounds
        else config.paths.component_checkpoint_template
    )
    values: dict[str, Any] = {
        "method": owner.name,
        "backbone_seed": int(backbone_seed),
        "planner_seed": int(planner_seed),
    }
    if counterexample_rounds:
        values["round"] = config.training.counterexample_rounds
    return resolve_path(template.format(**values))


def parent_component_checkpoint_path(
    config: StudyConfig,
    method: MethodConfig,
    *,
    backbone_seed: int,
    planner_seed: int,
) -> Path | None:
    if method.initialization_parent is None:
        return None
    parent = method_by_name(config, method.initialization_parent)
    path = component_checkpoint_path(
        config,
        parent,
        backbone_seed=backbone_seed,
        planner_seed=planner_seed,
    )
    if path is None:
        raise ValueError("a learned child cannot inherit a headless parent")
    return path


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


def hierarchical_seed(namespace: str, *values: int) -> int:
    payload = ":".join([namespace, *(str(int(value)) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


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
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS requested but unavailable: {requested}")
    return torch.device(name)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool:
    watched = (
        "vector_jepa_planner_frontier",
        "final_closure",
        "hdwm",
        "spatial_jepa_planning/common.py",
        "scripts/train/train_dim256.py",
        "data/splits",
        "pyproject.toml",
    )
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def require_clean_worktree(*, allow_dirty: bool) -> None:
    if git_worktree_dirty() and not allow_dirty:
        raise RuntimeError("formal runs require a clean committed experiment worktree")


def experiment_code_fingerprint() -> str:
    files = list(PACKAGE_ROOT.rglob("*.py"))
    files.extend((ROOT / "hdwm").rglob("*.py"))
    files.extend((ROOT / "final_closure").rglob("*.py"))
    files.extend(
        [
            ROOT / "spatial_jepa_planning/common.py",
            ROOT / "scripts/train/train_dim256.py",
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


def analysis_spec_sha256(config: StudyConfig, lock: dict[str, Any]) -> str:
    manifest_roles = (
        "train_manifest",
        "development_manifest",
        "validation_manifest",
        "confirmatory_manifest",
    )
    payload = {
        "schema": "vector-jepa-planner-frontier-analysis-v1",
        "protocol_id": config.protocol_id,
        "study_role": config.study_role,
        "protocol": config.protocol.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "analysis": config.analysis.model_dump(mode="json"),
        "methods": [method.model_dump(mode="json") for method in config.methods],
        "amendments": {
            name: lock[name]["sha256"]
            for name in (
                "amendments",
                "amendment_document",
                "amendment_before",
                "amendment_after",
            )
        },
        "manifests": {role: lock[role]["sha256"] for role in manifest_roles},
        "source_baseline": lock["source_baseline"],
    }
    return canonical_json_sha256(payload)


def validate_locked_artifacts(config: StudyConfig, lock: dict[str, Any]) -> None:
    if lock.get("status") != "locked":
        raise RuntimeError("formal operation requires a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    records = (
        lock.get("amendments", {}),
        lock.get("amendment_document", {}),
        lock.get("amendment_before", {}),
        lock.get("amendment_after", {}),
        lock.get("protocol_document", {}),
        lock.get("method_config", {}),
        lock.get("environment_lock", {}),
    )
    for record in records:
        path = record.get("path")
        expected = record.get("sha256")
        if not path or not expected or sha256_file(resolve_path(path)) != expected:
            raise ValueError(f"locked artifact hash mismatch: {path}")
    if lock.get("code_fingerprint") != experiment_code_fingerprint():
        raise ValueError("experiment code no longer matches the protocol lock")


def training_spec_sha256(
    config: StudyConfig,
    lock: dict[str, Any],
    *,
    method: MethodConfig,
    backbone_seed: int,
    planner_seed: int,
) -> str:
    payload = {
        "schema": "vector-jepa-planner-frontier-training-v1",
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "method": method.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "backbone_seed": int(backbone_seed),
        "planner_seed": int(planner_seed),
        "train_manifest_sha256": lock["train_manifest"]["sha256"],
        "validation_not_used_for_gradient_updates": True,
        "confirmatory_not_used_for_selection": True,
    }
    return canonical_json_sha256(payload)


@dataclass
class ComputeLedger:
    """Separate planning, assistance, and model-call accounting."""

    plan_transitions: int = 0
    assist_transitions: int = 0
    planner_forward_calls: int = 0
    assist_forward_calls: int = 0
    planner_max_batch: int = 0
    node_expansions: int = 0
    candidate_sequences: int = 0
    duplicate_candidates: int = 0
    verifier_forward_calls: int = 0
    reachability_forward_calls: int = 0
    ranker_forward_calls: int = 0
    proposal_forward_calls: int = 0
    join_forward_calls: int = 0
    dts_forward_calls: int = 0

    @property
    def total_transitions(self) -> int:
        return self.plan_transitions + self.assist_transitions

    def record_plan(self, *, transitions: int, batch_size: int, calls: int = 1) -> None:
        if transitions < 0 or batch_size < 0 or calls < 0:
            raise ValueError("compute counters cannot be negative")
        self.plan_transitions += int(transitions)
        self.planner_forward_calls += int(calls)
        self.planner_max_batch = max(self.planner_max_batch, int(batch_size))

    def record_assist(self, *, transitions: int, calls: int = 1) -> None:
        if transitions < 0 or calls < 0:
            raise ValueError("assistance compute cannot be negative")
        self.assist_transitions += int(transitions)
        self.assist_forward_calls += int(calls)

    def merge(self, other: ComputeLedger) -> None:
        self.plan_transitions += other.plan_transitions
        self.assist_transitions += other.assist_transitions
        self.planner_forward_calls += other.planner_forward_calls
        self.assist_forward_calls += other.assist_forward_calls
        self.planner_max_batch = max(self.planner_max_batch, other.planner_max_batch)
        self.node_expansions += other.node_expansions
        self.candidate_sequences += other.candidate_sequences
        self.duplicate_candidates += other.duplicate_candidates
        self.verifier_forward_calls += other.verifier_forward_calls
        self.reachability_forward_calls += other.reachability_forward_calls
        self.ranker_forward_calls += other.ranker_forward_calls
        self.proposal_forward_calls += other.proposal_forward_calls
        self.join_forward_calls += other.join_forward_calls
        self.dts_forward_calls += other.dts_forward_calls

    def to_dict(self) -> dict[str, int]:
        value = asdict(self)
        value["total_transitions"] = self.total_transitions
        return value


def validate_compute_ledger(value: dict[str, Any]) -> None:
    required = {
        "plan_transitions",
        "assist_transitions",
        "total_transitions",
        "planner_forward_calls",
        "assist_forward_calls",
        "planner_max_batch",
        "node_expansions",
        "candidate_sequences",
        "duplicate_candidates",
        "verifier_forward_calls",
        "reachability_forward_calls",
        "ranker_forward_calls",
        "proposal_forward_calls",
        "join_forward_calls",
        "dts_forward_calls",
    }
    if set(value) != required:
        raise ValueError("compute ledger fields do not match the formal schema")
    parsed = {key: int(item) for key, item in value.items()}
    if any(item < 0 for item in parsed.values()):
        raise ValueError("compute ledger values cannot be negative")
    if parsed["total_transitions"] != (
        parsed["plan_transitions"] + parsed["assist_transitions"]
    ):
        raise ValueError("total compute does not equal planning plus assistance")


def atomic_json_dump(path: str | Path, payload: Any) -> None:
    output = resolve_path(path)
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


def atomic_bytes_dump(path: str | Path, payload: bytes) -> None:
    """Publish an immutable binary/text artifact without a partial final file."""

    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text_dump(path: str | Path, payload: str) -> None:
    atomic_bytes_dump(path, payload.encode("utf-8"))


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    output = resolve_path(path)
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


def manifest_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    entries = read_jsonl(resolved)
    counts = Counter(int(entry["maze_size"]) for entry in entries)
    return {
        "path": str(Path(path)),
        "sha256": sha256_file(resolved),
        "count": len(entries),
        "counts_by_size": {str(size): count for size, count in sorted(counts.items())},
    }


def validate_manifest_isolation(config: StudyConfig) -> dict[str, dict[str, int]]:
    paths = {
        "train": config.paths.train_manifest,
        "development": config.paths.development_manifest,
        "validation": config.paths.validation_manifest,
        "confirmatory": config.paths.confirmatory_manifest,
    }
    entries = {name: read_jsonl(resolve_path(path)) for name, path in paths.items()}
    overlaps: dict[str, dict[str, int]] = {}
    names = tuple(entries)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result = verify_holdout(entries[left], entries[right])
            overlaps[f"{left}_vs_{right}"] = result
            if any(int(value) != 0 for value in result.values()):
                raise ValueError(
                    f"manifest leakage detected: {left} vs {right}: {result}"
                )
    return overlaps


def protocol_metadata(
    config: StudyConfig,
    lock: dict[str, Any],
    *,
    method: MethodConfig,
    seed: int,
    device: torch.device,
    planner_seed: int | None = None,
    search_seed: int | None = None,
) -> dict[str, Any]:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_role": config.study_role,
        "analysis_spec_sha256": analysis_spec_sha256(config, lock),
        "method": method.model_dump(mode="json"),
        "seed": int(seed),
        "backbone_seed": int(seed),
        "planner_seed": int(planner_seed) if planner_seed is not None else None,
        "search_seed": int(search_seed) if search_seed is not None else None,
        "device": str(device),
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "code_fingerprint": experiment_code_fingerprint(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def validate_finite_tree(value: Any, *, label: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_tree(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite_tree(item, label=f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"non-finite result at {label}")


def prepare_formal_output(
    path: str | Path,
    *,
    overwrite: bool,
    rerun_reason: str,
) -> dict[str, Any] | None:
    output = resolve_path(path)
    rerun = prepare_rerun([output], overwrite=overwrite, reason=rerun_reason)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite formal output: {output}")
    return rerun


def prepare_formal_outputs(
    paths: list[str | Path],
    *,
    overwrite: bool,
    rerun_reason: str,
) -> dict[str, Any] | None:
    """Apply one immutable-output/rerun decision to a multi-file artifact."""

    outputs = [resolve_path(path) for path in paths]
    rerun = prepare_rerun(outputs, overwrite=overwrite, reason=rerun_reason)
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite formal outputs: "
                + ", ".join(str(path) for path in existing)
            )
    return rerun


__all__ = [
    "ComputeLedger",
    "PACKAGE_ROOT",
    "ROOT",
    "RERUN_REASONS",
    "analysis_spec_sha256",
    "atomic_bytes_dump",
    "atomic_json_dump",
    "atomic_text_dump",
    "atomic_torch_save",
    "component_checkpoint_owner",
    "component_checkpoint_path",
    "experiment_code_fingerprint",
    "git_commit",
    "git_worktree_dirty",
    "hierarchical_seed",
    "load_json",
    "load_study_config",
    "manifest_record",
    "method_by_name",
    "parent_component_checkpoint_path",
    "planner_seed_values",
    "prepare_formal_output",
    "prepare_formal_outputs",
    "protocol_metadata",
    "require_clean_worktree",
    "resolve_device",
    "resolve_path",
    "set_seed",
    "training_spec_sha256",
    "uses_counterexample_rounds",
    "validate_compute_ledger",
    "validate_finite_tree",
    "validate_manifest_isolation",
    "validate_locked_artifacts",
]
