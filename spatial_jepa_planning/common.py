"""Protocol, sampling, checkpoint, and metric helpers for the experiment suite."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.common import (  # noqa: E402
    ACTION_IDS,
    ACTION_TO_SLOT,
    bfs_distances_from,
    create_env,
    next_state,
    observe_state,
    read_jsonl,
    set_agent_state,
    verify_holdout,
    write_json,
)
from spatial_jepa_planning import EXPERIMENT_FAMILY, FORMAT_VERSION  # noqa: E402
from spatial_jepa_planning.models import (  # noqa: E402
    PlannerConfig,
    SpatialRepresentation,
    SpatialRepresentationConfig,
    build_planner,
)


def set_seed(seed: int, deterministic: bool = False) -> None:
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


@dataclass
class RNGStreams:
    entries: np.random.Generator
    map_states: np.random.Generator
    sequences: np.random.Generator
    iteration_schedule: np.random.Generator


def make_rng_streams(seed: int) -> RNGStreams:
    """Create independent streams so one ablation cannot shift another's data."""
    children = np.random.SeedSequence(int(seed)).spawn(4)
    return RNGStreams(*(np.random.default_rng(child) for child in children))


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA/CPU for portable run plans."""
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device {requested!r}, but this PyTorch build has no CUDA"
        )
    if name.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError(f"requested device {requested!r}, but MPS is not available")
    return torch.device(name)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_layout_hash(wall_mask: np.ndarray) -> str:
    """Hash maze geometry independently of topology seed or manifest version."""
    mask = np.ascontiguousarray(wall_mask, dtype=np.uint8)
    header = f"maze-layout-v1:{mask.shape[0]}:{mask.shape[1]}:".encode("ascii")
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    return hashlib.sha256(header + packed).hexdigest()


def canonical_task_hash(
    *,
    maze_size: int,
    layout_hash: str,
    start_cell: int,
    goal_cell: int,
) -> str:
    payload = {
        "goal_cell": int(goal_cell),
        "layout_hash": str(layout_hash),
        "maze_size": int(maze_size),
        "schema": "maze-task-v1",
        "start_cell": int(start_cell),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_sha256(data: Any) -> str:
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_code_fingerprint() -> str:
    roots = (ROOT / "spatial_jepa_planning", ROOT / "hdwm")
    files = [path for root in roots for path in root.rglob("*.py")]
    files.append(ROOT / "diagnostics/common.py")
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_worktree_dirty() -> bool:
    try:
        output = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "spatial_jepa_planning",
                "hdwm",
                "diagnostics/common.py",
                "data/splits",
                "pyproject.toml",
            ],
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
            "formal runs require a clean tracked experiment worktree; commit changes "
            "or use --allow-dirty-worktree only for labelled diagnostics"
        )


def require_new_output(path: str | Path, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing formal output: {path}; pass --overwrite "
            "only for an explicitly documented rerun"
        )


def runtime_metadata() -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        metadata["cuda_device_name"] = None
        metadata["cuda_device_capability"] = None
    return metadata


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


def parse_int_list(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected a non-empty list of positive integers")
    return result


def protocol_metadata(
    *,
    train_manifest: str | Path,
    eval_manifest: str | Path,
    development_manifest: str | Path | None = None,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    metadata = {
        "experiment_family": EXPERIMENT_FAMILY,
        "format_version": FORMAT_VERSION,
        "git_commit": git_commit(),
        "git_dirty": git_worktree_dirty(),
        "code_fingerprint": experiment_code_fingerprint(),
        "runtime": runtime_metadata(),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": sha256_file(eval_manifest),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "action_ids": list(ACTION_IDS),
        "seen_max_size": 21,
    }
    if development_manifest is not None:
        metadata.update(
            {
                "development_manifest": str(development_manifest),
                "development_manifest_sha256": sha256_file(development_manifest),
            }
        )
    return metadata


def validate_manifest_entry(entry: dict[str, Any], *, check_bfs: bool = True) -> Any:
    env = create_env(entry)
    size = int(entry["maze_size"])
    start = int(entry["start_cell"])
    goal = int(entry["goal_cell"])
    wall_flat = env._maze_mask.reshape(-1)
    if start < 0 or start >= size * size or wall_flat[start]:
        raise ValueError(f"manifest start_cell is invalid: size={size} start={start}")
    if goal < 0 or goal >= size * size or wall_flat[goal]:
        raise ValueError(f"manifest goal_cell is invalid: size={size} goal={goal}")
    if int(env._goal_position) != goal:
        raise ValueError(
            "manifest/environment goal mismatch: "
            f"size={size} topology_seed={entry['topology_seed']} "
            f"manifest={goal} generated={env._goal_position}"
        )
    if check_bfs:
        distances = bfs_distances_from(env._maze_mask, goal, size)
        actual = int(distances[start])
        expected = int(entry["bfs_path_length"])
        if actual != expected:
            raise ValueError(
                f"manifest BFS mismatch for task {entry.get('task_hash')}: "
                f"expected={expected}, actual={actual}"
            )
    if "num_walls" in entry and int(env._maze_mask.sum()) != int(entry["num_walls"]):
        raise ValueError("manifest num_walls does not match regenerated topology")
    return env


def validate_manifest_pair(
    train_manifest: str | Path,
    eval_manifest: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_entries = read_jsonl(train_manifest)
    eval_entries = read_jsonl(eval_manifest)
    overlap = verify_holdout(train_entries, eval_entries)
    return train_entries, eval_entries, overlap


@dataclass
class ManifestSampler:
    entries: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for entry in self.entries:
            self.by_size[int(entry["maze_size"])].append(entry)
        if not self.by_size:
            raise ValueError("manifest sampler requires at least one entry")
        self.sizes = tuple(sorted(self.by_size))

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        size: int | None = None,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_size = int(size) if size is not None else int(rng.choice(self.sizes))
        if selected_size not in self.by_size:
            raise ValueError(f"size {selected_size} is not present in manifest")
        group = self.by_size[selected_size]
        indices = rng.integers(0, len(group), size=batch_size)
        return [group[int(index)] for index in indices]


def valid_action_field(wall_mask: np.ndarray) -> np.ndarray:
    height, width = wall_mask.shape
    valid = np.zeros((4, height, width), dtype=np.float32)
    free = ~wall_mask
    valid[0, 1:, :] = free[:-1, :]
    valid[1, :-1, :] = free[1:, :]
    valid[2, :, 1:] = free[:, :-1]
    valid[3, :, :-1] = free[:, 1:]
    valid *= free[None, :, :]
    return valid


def build_map_targets(env: Any, device: torch.device) -> dict[str, torch.Tensor]:
    height = int(env.config.height)
    width = int(env.config.width)
    if height != width:
        raise ValueError("the current protocol expects square mazes")
    goal = int(env._goal_position)
    distances_flat = bfs_distances_from(env._maze_mask, goal, width)
    distance = distances_flat.reshape(height, width).astype(np.float32)
    free = ~env._maze_mask
    if bool((distance[free] < 0).any()):
        raise ValueError("all free cells must be connected in Procgen Maze")
    valid = valid_action_field(env._maze_mask)
    optimal = np.zeros_like(valid)
    for state in np.flatnonzero(free.reshape(-1)).tolist():
        if state == goal:
            continue
        current_distance = int(distances_flat[state])
        row, col = divmod(int(state), width)
        for action in ACTION_IDS:
            slot = ACTION_TO_SLOT[int(action)]
            candidate = next_state(env, int(state), int(action))
            if (
                candidate != state
                and int(distances_flat[candidate]) == current_distance - 1
            ):
                optimal[slot, row, col] = 1.0
    goal_mask = np.zeros((height, width), dtype=np.float32)
    goal_mask.flat[goal] = 1.0
    return {
        "distance": torch.as_tensor(distance, device=device),
        "free_mask": torch.as_tensor(free, dtype=torch.bool, device=device),
        "goal_mask": torch.as_tensor(goal_mask, device=device),
        "valid_action_mask": torch.as_tensor(valid, device=device),
        "optimal_action_mask": torch.as_tensor(optimal, device=device),
    }


def stack_targets(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot stack an empty target list")
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def sample_map_batch(
    entries: list[dict[str, Any]],
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observations: list[np.ndarray] = []
    targets: list[dict[str, torch.Tensor]] = []
    for entry in entries:
        env = validate_manifest_entry(entry, check_bfs=False)
        free = np.flatnonzero((~env._maze_mask).reshape(-1))
        candidates = free[free != int(env._goal_position)]
        state = int(rng.choice(candidates if len(candidates) else free))
        observations.append(observe_state(env, state))
        targets.append(build_map_targets(env, device))
    obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    return obs, stack_targets(targets)


def sample_sequence_batch(
    entries: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    device: torch.device,
    sequence_length: int,
    trajectories_per_map: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if trajectories_per_map <= 0:
        raise ValueError("trajectories_per_map must be positive")
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    valid_fields: list[torch.Tensor] = []
    for entry in entries:
        runtime_entry = dict(entry)
        runtime_entry["env_seed"] = int(rng.integers(0, 2**31 - 1))
        env = create_env(runtime_entry)
        if int(env._goal_position) != int(entry["goal_cell"]):
            raise ValueError("trajectory seed changed the locked topology goal")
        batch = env.sample_sequence(
            batch_size=trajectories_per_map,
            sequence_length=sequence_length,
        )
        observations.append(batch.observations.to(device=device, dtype=torch.float32))
        actions.append(batch.actions.to(device=device, dtype=torch.long))
        valid = torch.as_tensor(
            valid_action_field(env._maze_mask),
            dtype=torch.float32,
            device=device,
        )
        valid_fields.append(
            valid[None, None].expand(trajectories_per_map, sequence_length, -1, -1, -1)
        )
    return (
        torch.cat(observations, dim=0),
        torch.cat(actions, dim=0),
        torch.cat(valid_fields, dim=0),
    )


def planner_features(
    observations: torch.Tensor,
    input_mode: str,
    representation: SpatialRepresentation | None,
) -> torch.Tensor:
    if input_mode == "raw":
        return observations.permute(0, 3, 1, 2)
    if input_mode == "spatial_jepa":
        if representation is None:
            raise ValueError("spatial_jepa input requires a representation")
        return representation.planning_latent(observations)
    raise ValueError(f"unsupported input_mode: {input_mode}")


def configure_representation_training(
    representation: SpatialRepresentation,
    mode: str,
) -> list[torch.nn.Parameter]:
    for parameter in representation.parameters():
        parameter.requires_grad = False
    if mode == "frozen":
        return []
    if mode == "all":
        modules = [representation]
    elif mode == "last_block":
        if not representation.encoder.blocks:
            raise ValueError("last_block mode requires at least one encoder block")
        modules = [
            representation.encoder.blocks[-1],
            representation.planning_projector,
            representation.map_decoder,
        ]
    else:
        raise ValueError("encoder mode must be frozen, last_block, or all")
    parameters: list[torch.nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    return parameters


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def estimate_planner_conv_macs(
    planner: torch.nn.Module,
    *,
    input_channels: int,
    maze_size: int,
    iterations: int,
    device: torch.device,
) -> int:
    """Count Conv2d multiply-accumulates for one inference field."""
    total = 0

    def hook(module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        if not isinstance(module, torch.nn.Conv2d) or not isinstance(
            output, torch.Tensor
        ):
            return
        kernel_height, kernel_width = module.kernel_size
        output_elements = output.numel()
        operations_per_output = (
            module.in_channels // module.groups * kernel_height * kernel_width
        )
        total += int(output_elements * operations_per_output)

    handles = [
        module.register_forward_hook(hook)
        for module in planner.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    was_training = planner.training
    planner.eval()
    try:
        dummy = torch.zeros(
            (1, input_channels, maze_size, maze_size),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            planner(dummy, iterations=iterations, deep_supervision_every=0)
    finally:
        for handle in handles:
            handle.remove()
        planner.train(was_training)
    return total


def estimate_representation_planning_conv_macs(
    representation: SpatialRepresentation,
    *,
    maze_size: int,
    device: torch.device,
) -> int:
    """Count Conv2d MACs used to produce the planner's spatial features."""
    total = 0

    def hook(module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        if not isinstance(module, torch.nn.Conv2d) or not isinstance(
            output, torch.Tensor
        ):
            return
        kernel_height, kernel_width = module.kernel_size
        operations_per_output = (
            module.in_channels // module.groups * kernel_height * kernel_width
        )
        total += int(output.numel() * operations_per_output)

    planning_modules = (representation.encoder, representation.planning_projector)
    handles = [
        module.register_forward_hook(hook)
        for root in planning_modules
        for module in root.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    was_training = representation.training
    representation.eval()
    try:
        observation = torch.zeros(
            (1, maze_size, maze_size, representation.config.observation_channels),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            representation.planning_latent(observation)
    finally:
        for handle in handles:
            handle.remove()
        representation.train(was_training)
    return total


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_representation_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[SpatialRepresentation, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError(f"not a {EXPERIMENT_FAMILY} checkpoint: {path}")
    if int(data.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {data.get('format_version')} != required "
            f"{FORMAT_VERSION}"
        )
    if "representation_config" not in data or "representation_state_dict" not in data:
        raise ValueError("checkpoint does not contain a spatial representation")
    config = SpatialRepresentationConfig.from_dict(data["representation_config"])
    model = SpatialRepresentation(config).to(device)
    model.load_state_dict(data["representation_state_dict"], strict=True)
    return model, data


def load_planner_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, SpatialRepresentation | None, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("experiment_family") != EXPERIMENT_FAMILY:
        raise ValueError(f"not a {EXPERIMENT_FAMILY} checkpoint: {path}")
    if int(data.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {data.get('format_version')} != required "
            f"{FORMAT_VERSION}"
        )
    config = PlannerConfig.from_dict(data["planner_config"])
    planner = build_planner(config).to(device)
    planner.load_state_dict(data["planner_state_dict"], strict=True)
    representation: SpatialRepresentation | None = None
    if data.get("input_mode") == "spatial_jepa":
        rep_config = SpatialRepresentationConfig.from_dict(
            data["representation_config"]
        )
        representation = SpatialRepresentation(rep_config).to(device)
        representation.load_state_dict(data["representation_state_dict"], strict=True)
    planner.eval()
    for parameter in planner.parameters():
        parameter.requires_grad = False
    if representation is not None:
        representation.eval()
        for parameter in representation.parameters():
            parameter.requires_grad = False
    return planner, representation, data


def gradient_cosine(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float]:
    active = [parameter for parameter in parameters if parameter.requires_grad]
    if not active:
        return {"first_norm": 0.0, "second_norm": 0.0, "cosine": float("nan")}
    first = torch.autograd.grad(
        first_loss, active, retain_graph=True, allow_unused=True
    )
    second = torch.autograd.grad(
        second_loss, active, retain_graph=True, allow_unused=True
    )
    dot = first_loss.new_tensor(0.0)
    first_norm = first_loss.new_tensor(0.0)
    second_norm = first_loss.new_tensor(0.0)
    for left, right in zip(first, second, strict=True):
        if left is None or right is None:
            continue
        dot = dot + (left * right).sum()
        first_norm = first_norm + left.square().sum()
        second_norm = second_norm + right.square().sum()
    denominator = first_norm.sqrt() * second_norm.sqrt()
    cosine = dot / denominator.clamp_min(torch.finfo(dot.dtype).eps)
    return {
        "first_norm": float(first_norm.sqrt().detach().cpu()),
        "second_norm": float(second_norm.sqrt().detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }


def task_id(entry: dict[str, Any]) -> str:
    fallback = (
        f"sz{entry['maze_size']}_topo{entry['topology_seed']}_"
        f"start{entry['start_cell']}"
    )
    return str(entry.get("task_hash") or fallback)


def summarize_rows(
    rows: list[dict[str, Any]],
    seen_max_size: int = 21,
    max_steps: int = 128,
) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0, "sr": 0.0, "spl": 0.0}
        eligible = [
            row
            for row in group
            if int(row.get("optimal_length", max_steps + 1)) <= max_steps
        ]
        return {
            "n": len(group),
            "sr": float(np.mean([float(row["success"]) for row in group])),
            "spl": float(np.mean([float(row["spl"]) for row in group])),
            "step_cap_eligible_n": len(eligible),
            "step_cap_ceiling": len(eligible) / len(group),
            "eligible_sr": float(np.mean([float(row["success"]) for row in eligible]))
            if eligible
            else 0.0,
            "loop_or_cycle_rate": float(
                np.mean([float(row.get("loop_or_cycle", False)) for row in group])
            ),
            "invalid_rate": float(
                sum(int(row.get("invalid_actions", 0)) for row in group)
                / max(sum(int(row["path_length"]) for row in group), 1)
            ),
        }

    by_size: dict[str, Any] = {}
    for size in sorted({int(row["maze_size"]) for row in rows}):
        by_size[str(size)] = summarize(
            [row for row in rows if int(row["maze_size"]) == size]
        )
    path_bins = (
        ("001-016", 1, 16),
        ("017-032", 17, 32),
        ("033-064", 33, 64),
        ("065-128", 65, 128),
        ("129+", 129, math.inf),
    )
    return {
        "overall": summarize(rows),
        "seen": summarize(
            [row for row in rows if int(row["maze_size"]) <= seen_max_size]
        ),
        "ood": summarize(
            [row for row in rows if int(row["maze_size"]) > seen_max_size]
        ),
        "by_size": by_size,
        "by_shortest_path": {
            label: summarize(
                [row for row in rows if lower <= int(row["optimal_length"]) <= upper]
            )
            for label, lower, upper in path_bins
        },
    }


def strict_json_dump(path: str | Path, data: Any) -> None:
    write_json(path, data)


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def count_by_size(entries: list[dict[str, Any]]) -> dict[int, int]:
    return dict(sorted(Counter(int(entry["maze_size"]) for entry in entries).items()))


__all__ = [
    "ACTION_IDS",
    "ACTION_TO_SLOT",
    "ManifestSampler",
    "RNGStreams",
    "build_map_targets",
    "canonical_layout_hash",
    "canonical_json_sha256",
    "canonical_task_hash",
    "configure_representation_training",
    "count_by_size",
    "create_env",
    "finite_mean",
    "experiment_code_fingerprint",
    "estimate_planner_conv_macs",
    "estimate_representation_planning_conv_macs",
    "git_commit",
    "git_worktree_dirty",
    "gradient_cosine",
    "load_planner_checkpoint",
    "load_representation_checkpoint",
    "make_rng_streams",
    "next_state",
    "observe_state",
    "parameter_count",
    "parse_int_list",
    "planner_features",
    "protocol_metadata",
    "read_jsonl",
    "require_clean_worktree",
    "require_new_output",
    "resolve_device",
    "sample_map_batch",
    "sample_sequence_batch",
    "save_checkpoint",
    "set_agent_state",
    "set_seed",
    "sha256_file",
    "strict_json_dump",
    "summarize_rows",
    "task_id",
    "validate_manifest_entry",
    "validate_manifest_pair",
    "valid_action_field",
]
