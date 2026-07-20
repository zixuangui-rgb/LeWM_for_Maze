"""Shared immutable-artifact helpers for the A1 quick-validation package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from distance_head_study.common import ROOT

PACKAGE_ROOT = ROOT / "a1_quick_validation"
DEFAULT_PROFILE = PACKAGE_ROOT / "configs/quick_profile.json"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/default.json"
DEFAULT_PACKAGE_LOCK = PACKAGE_ROOT / "configs/package_lock.json"


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(resolve_path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(path: str | Path, value: Any) -> Path:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def prepare_immutable(path: str | Path) -> Path:
    output = resolve_path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def package_files() -> tuple[Path, ...]:
    excluded = {
        PACKAGE_ROOT / "configs/package_lock.json",
    }
    return tuple(
        sorted(
            path
            for path in PACKAGE_ROOT.rglob("*")
            if path.is_file()
            and path not in excluded
            and "__pycache__" not in path.parts
        )
    )


def guarded_worktree_dirty() -> bool:
    guarded = (
        "a1_quick_validation",
        "distance_head_study",
        "hdwm",
        "final_closure",
        "diagnostics",
        "spatial_jepa_planning",
        "vector_jepa_planner_frontier",
        "scripts/runs/run_seqlen2_metric_heads.sh",
        "scripts/train/train_distance_head_simple_setb.py",
        "scripts/eval/eval_setb_distance_head_fixed.py",
        "scripts/eval/run_cem_setb_correct.py",
        "scripts/train/train_dim256.py",
        "results/FINAL_REPORT.md",
        "pyproject.toml",
        "uv.lock",
    )
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *guarded,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def require_clean_worktree() -> None:
    if guarded_worktree_dirty():
        raise RuntimeError(
            "formal quick-validation runs require committed, clean scientific code"
        )


def relative(path: str | Path) -> str:
    return resolve_path(path).relative_to(ROOT).as_posix()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_PACKAGE_LOCK",
    "DEFAULT_PROFILE",
    "PACKAGE_ROOT",
    "atomic_json_dump",
    "canonical_json_sha256",
    "load_json",
    "package_files",
    "prepare_immutable",
    "relative",
    "require_clean_worktree",
    "resolve_path",
    "sha256_file",
]
