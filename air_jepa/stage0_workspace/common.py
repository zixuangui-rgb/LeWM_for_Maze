"""Hashing, determinism, paths, and immutable artifacts for AIR Stage 0."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from air_jepa.stage0_workspace.schemas import Stage0Config

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "air_jepa" / "stage0_workspace"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.json"
PACKAGE_LOCK_SCHEMA = "air-jepa-stage0-package-lock-v1"
PROTOCOL_LOCK_SCHEMA = "air-jepa-stage0-protocol-lock-v1"
SOURCE_LOCK_SCHEMA = "air-jepa-stage0-source-lock-v1"


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative_path(path: str | Path) -> str:
    resolved = resolve_path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: str | Path) -> Any:
    with open(resolve_path(path), encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(resolve_path(path), encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("formal JSON artifacts cannot contain NaN or Infinity")
        return value
    return value


def atomic_json_dump(path: str | Path, payload: Any) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(
                json_safe(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_text_dump(path: str | Path, text: str) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_new_output(path: str | Path) -> Path:
    output = resolve_path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(resolve_path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> Stage0Config:
    return Stage0Config.model_validate(read_json(path))


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return torch.device(name)


def require_h800_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("formal AIR execution requires an NVIDIA H800 device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    if "H800" not in name.upper():
        raise RuntimeError(f"formal AIR execution requires H800; selected {name}")


def runtime_metadata(device: torch.device | None = None) -> dict[str, Any]:
    def package_version(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    output: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pydantic": package_version("pydantic"),
        "gymnasium": package_version("gymnasium"),
        "omegaconf": package_version("omegaconf"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }
    if torch.cuda.is_available():
        index = device.index if device and device.index is not None else 0
        output["cuda_device_name"] = torch.cuda.get_device_name(index)
        output["cuda_device_capability"] = list(torch.cuda.get_device_capability(index))
    else:
        output["cuda_device_name"] = None
        output["cuda_device_capability"] = None
    return output


def runtime_signature(runtime: dict[str, Any]) -> tuple[Any, ...]:
    return (
        runtime.get("python"),
        runtime.get("platform"),
        runtime.get("numpy"),
        runtime.get("torch"),
        runtime.get("pydantic"),
        runtime.get("gymnasium"),
        runtime.get("omegaconf"),
        runtime.get("cuda_runtime"),
        runtime.get("cudnn"),
        runtime.get("deterministic_algorithms"),
        runtime.get("cudnn_deterministic"),
        runtime.get("cudnn_benchmark"),
        runtime.get("cublas_workspace_config"),
        runtime.get("python_hash_seed"),
        runtime.get("cuda_device_order"),
        runtime.get("cuda_device_name"),
        tuple(runtime.get("cuda_device_capability") or ()),
    )


def formal_runtime_valid(runtime: dict[str, Any]) -> bool:
    package_versions = (
        runtime.get("pydantic"),
        runtime.get("gymnasium"),
        runtime.get("omegaconf"),
    )
    return bool(
        runtime.get("deterministic_algorithms") is True
        and runtime.get("cudnn_deterministic") is True
        and runtime.get("cudnn_benchmark") is False
        and runtime.get("cublas_workspace_config") == ":4096:8"
        and runtime.get("python_hash_seed") == "0"
        and runtime.get("cuda_device_order") == "PCI_BUS_ID"
        and "H800" in str(runtime.get("cuda_device_name", "")).upper()
        and all(isinstance(value, str) and value for value in package_versions)
    )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_dirty() -> bool:
    scopes = (
        "air_jepa",
        "spatial_jepa_planning",
        "diagnostics/common.py",
        "hdwm/envs/procgen_maze.py",
        "data/splits",
        "pyproject.toml",
    )
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *scopes],
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
            "formal AIR runs require a clean tracked worktree; commit the package "
            "or use --allow-dirty only for explicitly labelled smoke tests"
        )


def _module_source_candidates(module: str) -> list[Path]:
    parts = tuple(part for part in module.split(".") if part)
    if not parts or any(not part.isidentifier() for part in parts):
        return []
    candidates: list[Path] = []
    for depth in range(1, len(parts)):
        package_init = ROOT.joinpath(*parts[:depth], "__init__.py")
        if package_init.is_file():
            candidates.append(package_init)
    module_file = ROOT.joinpath(*parts).with_suffix(".py")
    package_file = ROOT.joinpath(*parts, "__init__.py")
    for candidate in (module_file, package_file):
        if candidate.is_file():
            candidates.append(candidate)
    return candidates


def _local_python_dependency_closure(entry_points: Iterable[Path]) -> set[Path]:
    """Resolve the static local import closure used by the formal AIR runtime."""

    pending = [path.resolve() for path in entry_points if path.is_file()]
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        resolved.add(path)
        relative = path.relative_to(ROOT.resolve())
        package_parts = list(relative.parent.parts)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base = package_parts[:keep]
            else:
                base = []
            module_parts = [*base, *(node.module or "").split(".")]
            module_parts = [part for part in module_parts if part]
            if module_parts:
                modules.add(".".join(module_parts))
            for alias in node.names:
                if alias.name != "*":
                    modules.add(".".join([*module_parts, alias.name]))
        for module in modules:
            for candidate in _module_source_candidates(module):
                resolved_candidate = candidate.resolve()
                if resolved_candidate not in resolved:
                    pending.append(resolved_candidate)
    return resolved


def package_files() -> list[Path]:
    files = {
        path
        for path in (ROOT / "air_jepa").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name not in {"package_lock.json"}
    }
    runtime_entry_points = (
        ROOT / "spatial_jepa_planning" / "__init__.py",
        ROOT / "spatial_jepa_planning" / "models.py",
        ROOT / "spatial_jepa_planning" / "common.py",
        ROOT / "spatial_jepa_planning" / "evaluate.py",
        ROOT / "spatial_jepa_planning" / "losses.py",
        ROOT / "diagnostics" / "common.py",
        ROOT / "hdwm" / "envs" / "procgen_maze.py",
    )
    python_entry_points = [path for path in files if path.suffix == ".py"]
    python_entry_points.extend(runtime_entry_points)
    files.update(_local_python_dependency_closure(python_entry_points))
    for dependency in (
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "tests" / "test_air_jepa_stage0.py",
    ):
        if dependency.exists():
            files.add(dependency)
    return sorted(files)


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in package_files():
        relative = relative_path(path).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def signed_payload(payload: dict[str, Any], signature_key: str) -> dict[str, Any]:
    if signature_key in payload:
        raise ValueError(f"payload already contains signature key {signature_key}")
    output = dict(payload)
    output[signature_key] = canonical_json_sha256(payload)
    return output


def verify_signature(payload: dict[str, Any], signature_key: str) -> None:
    signature = payload.get(signature_key)
    unsigned = {key: value for key, value in payload.items() if key != signature_key}
    if not isinstance(signature, str) or signature != canonical_json_sha256(unsigned):
        raise ValueError(f"invalid artifact signature: {signature_key}")


def format_template(template: str, **values: Any) -> Path:
    return resolve_path(template.format(**values))


def parse_ints(values: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(
            int(value.strip()) for value in values.split(",") if value.strip()
        )
    else:
        parsed = tuple(int(value) for value in values)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("expected one or more positive integers")
    return parsed


__all__ = [
    "DEFAULT_CONFIG",
    "PACKAGE_LOCK_SCHEMA",
    "PACKAGE_ROOT",
    "PROTOCOL_LOCK_SCHEMA",
    "ROOT",
    "SOURCE_LOCK_SCHEMA",
    "atomic_json_dump",
    "atomic_text_dump",
    "canonical_json_sha256",
    "code_fingerprint",
    "format_template",
    "formal_runtime_valid",
    "git_commit",
    "git_worktree_dirty",
    "json_safe",
    "load_config",
    "package_files",
    "parse_ints",
    "prepare_new_output",
    "read_json",
    "read_jsonl",
    "relative_path",
    "require_clean_worktree",
    "require_h800_device",
    "resolve_device",
    "resolve_path",
    "runtime_metadata",
    "runtime_signature",
    "set_seed",
    "sha256_file",
    "signed_payload",
    "state_dict_sha256",
    "verify_signature",
]
