"""Goal-consistent cache shards and deterministic matched training batches."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnostics.common import bfs_distances_from
from distance_head_study import MODEL_ACTION_VOCAB_SIZE
from distance_head_study.common import (
    atomic_json_dump,
    atomic_torch_save,
    canonical_json_sha256,
    hierarchical_seed,
    read_jsonl,
    resolve_path,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.schemas import SamplerKind, StudyConfig
from final_closure.models import deserialize_lewm_config
from scripts.train.train_dim256 import Unisize256
from spatial_jepa_planning.common import create_env, observe_state

CACHE_SCHEMA = "distance-head-goal-consistent-cache-v1"
_SHARD_TENSOR_FIELDS = (
    "cells",
    "observations",
    "latents",
    "all_pairs_bfs",
    "goal_distances",
    "next_indices",
    "valid_actions",
    "optimal_actions",
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json_sha256(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _topology_content_sha256(payload: dict[str, Any]) -> str:
    metadata = dict(payload["metadata"])
    metadata.pop("content_sha256", None)
    tensors = {}
    for name in _SHARD_TENSOR_FIELDS:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"cache shard field is not a tensor: {name}")
        tensors[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_sha256(value),
        }
    return canonical_json_sha256({"metadata": metadata, "tensors": tensors})


def load_backbone_checkpoint(
    path: str | Path, device: torch.device, *, freeze: bool = True
) -> tuple[Unisize256, dict[str, Any]]:
    resolved = resolve_path(path)
    payload = torch.load(resolved, map_location=device, weights_only=False)
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"not a LeWM backbone checkpoint: {resolved}")
    model_config = deserialize_lewm_config(payload["model_config"])
    model = Unisize256(model_config, max_size=31).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = not freeze
    return model, payload


def encode_observations(
    model: Unisize256,
    observations: torch.Tensor,
    *,
    maze_size: int,
    device: torch.device,
    gradients: bool,
) -> torch.Tensor:
    values = observations.to(device=device, dtype=torch.float32).unsqueeze(1)
    manager = torch.enable_grad() if gradients else torch.no_grad()
    with manager:
        encoded = model.encoder(values, int(maze_size))
        embedding, _ = model.embedding_projector(encoded)
    result = embedding.squeeze(1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("backbone produced non-finite cached latents")
    return result if gradients else result.detach()


def _uint8_observations(env: Any, cells: np.ndarray) -> torch.Tensor:
    observations = np.stack([observe_state(env, int(cell)) for cell in cells])
    if not np.isfinite(observations).all():
        raise FloatingPointError("environment produced a non-finite observation")
    rounded = np.rint(observations)
    if not np.array_equal(observations, rounded):
        raise ValueError("observation cache expects exact integer-valued pixels")
    if rounded.min() < 0 or rounded.max() > 255:
        raise ValueError("observation values cannot be represented as uint8")
    return torch.from_numpy(rounded.astype(np.uint8, copy=False))


def _all_pairs_bfs(mask: np.ndarray, cells: np.ndarray, width: int) -> np.ndarray:
    distances = np.empty((len(cells), len(cells)), dtype=np.int16)
    for index, source in enumerate(cells.tolist()):
        full = bfs_distances_from(mask, int(source), int(width))
        selected = np.asarray(full, dtype=np.int64)[cells]
        if (selected < 0).any() or selected.max(initial=0) > np.iinfo(np.int16).max:
            raise ValueError("cache topology is disconnected or exceeds int16 distance")
        distances[index] = selected.astype(np.int16)
    return distances


def build_topology_shard(
    entry: dict[str, Any],
    model: Unisize256,
    *,
    backbone_path: Path,
    device: torch.device,
    encode_batch_size: int = 512,
    analysis_spec_sha256: str | None = None,
    protocol_lock_sha256: str | None = None,
) -> dict[str, Any]:
    env = create_env(entry)
    _, info = env.reset()
    if int(info["state"]) != int(entry["start_cell"]):
        raise ValueError("manifest start cell does not regenerate")
    if int(env._goal_position) != int(entry["goal_cell"]):
        raise ValueError("manifest goal cell does not regenerate")
    size = int(entry["maze_size"])
    cells = np.flatnonzero((~env._maze_mask).reshape(-1)).astype(np.int64)
    cell_to_index = {int(cell): index for index, cell in enumerate(cells.tolist())}
    observations = _uint8_observations(env, cells)
    latent_chunks = []
    for offset in range(0, len(cells), encode_batch_size):
        latent_chunks.append(
            encode_observations(
                model,
                observations[offset : offset + encode_batch_size],
                maze_size=size,
                device=device,
                gradients=False,
            ).cpu()
        )
    latents = torch.cat(latent_chunks, dim=0).to(dtype=torch.float32)
    bfs = _all_pairs_bfs(env._maze_mask, cells, size)
    goal_index = cell_to_index[int(entry["goal_cell"])]
    goal_distances = bfs[:, goal_index].copy()
    next_indices = np.empty((len(cells), env.config.action_vocab_size), dtype=np.int32)
    if env.config.action_vocab_size != MODEL_ACTION_VOCAB_SIZE:
        raise ValueError("environment action vocabulary differs from the locked LeWM")
    for source_index, source in enumerate(cells.tolist()):
        for action in range(env.config.action_vocab_size):
            target = int(env._next_state(int(source), env._decode_action(action)))
            next_indices[source_index, action] = cell_to_index[target]
    valid_actions = next_indices != np.arange(len(cells), dtype=np.int32)[:, None]
    valid_actions[:, 0] = False
    next_distances = goal_distances[next_indices]
    best = np.where(valid_actions, next_distances, np.iinfo(np.int16).max).min(axis=1)
    optimal_actions = valid_actions & (next_distances == best[:, None])
    if bool((optimal_actions[goal_distances > 0].sum(axis=1) == 0).any()):
        raise ValueError("a non-goal state has no BFS-optimal moving action")
    metadata = {
        "schema": CACHE_SCHEMA,
        "task_hash": str(entry["task_hash"]),
        "layout_hash": str(entry["layout_hash"]),
        "maze_size": size,
        "topology_seed": int(entry["topology_seed"]),
        "goal_cell": int(entry["goal_cell"]),
        "goal_index": int(goal_index),
        "cell_count": int(len(cells)),
        "max_goal_distance": int(goal_distances.max(initial=0)),
        "backbone_path": backbone_path.as_posix(),
        "backbone_sha256": sha256_file(backbone_path),
        "observation_semantics": "every state rendered with manifest goal",
        "analysis_spec_sha256": analysis_spec_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
    }
    payload = {
        "metadata": metadata,
        "entry": entry,
        "cells": torch.from_numpy(cells),
        "observations": observations,
        "latents": latents,
        "all_pairs_bfs": torch.from_numpy(bfs),
        "goal_distances": torch.from_numpy(goal_distances),
        "next_indices": torch.from_numpy(next_indices),
        "valid_actions": torch.from_numpy(valid_actions),
        "optimal_actions": torch.from_numpy(optimal_actions),
    }
    metadata["content_sha256"] = _topology_content_sha256(payload)
    return payload


def _validate_topology_shard(
    payload: dict[str, Any],
    entry: dict[str, Any],
    *,
    backbone_path: Path,
    analysis_spec_sha256: str,
    protocol_lock_sha256: str,
) -> None:
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "task_hash": str(entry["task_hash"]),
        "layout_hash": str(entry["layout_hash"]),
        "maze_size": int(entry["maze_size"]),
        "topology_seed": int(entry["topology_seed"]),
        "goal_cell": int(entry["goal_cell"]),
        "backbone_path": backbone_path.as_posix(),
        "backbone_sha256": sha256_file(backbone_path),
        "analysis_spec_sha256": analysis_spec_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "observation_semantics": "every state rendered with manifest goal",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("partial cache shard provenance differs from this protocol")
    if payload.get("entry") != entry:
        raise ValueError("partial cache shard manifest entry differs")
    required = {
        "cells",
        "observations",
        "latents",
        "all_pairs_bfs",
        "goal_distances",
        "next_indices",
        "valid_actions",
        "optimal_actions",
    }
    if not required <= set(payload):
        raise ValueError("partial cache shard is missing tensors")
    cells = payload["cells"]
    cell_count = int(metadata.get("cell_count", -1))
    maze_size = int(metadata.get("maze_size", -1))
    if (
        not isinstance(cells, torch.Tensor)
        or cells.shape != (cell_count,)
        or payload["observations"].shape != (cell_count, maze_size, maze_size, 5)
        or payload["latents"].shape != (cell_count, 256)
        or payload["all_pairs_bfs"].shape != (cell_count, cell_count)
        or payload["goal_distances"].shape != (cell_count,)
        or payload["next_indices"].shape != (cell_count, MODEL_ACTION_VOCAB_SIZE)
        or payload["valid_actions"].shape != (cell_count, MODEL_ACTION_VOCAB_SIZE)
        or payload["optimal_actions"].shape != (cell_count, MODEL_ACTION_VOCAB_SIZE)
    ):
        raise ValueError("partial cache shard tensor shapes differ")
    expected_dtypes = {
        "cells": torch.int64,
        "observations": torch.uint8,
        "latents": torch.float32,
        "all_pairs_bfs": torch.int16,
        "goal_distances": torch.int16,
        "next_indices": torch.int32,
        "valid_actions": torch.bool,
        "optimal_actions": torch.bool,
    }
    if any(payload[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("partial cache shard tensor dtypes differ")
    if not torch.isfinite(payload["latents"]).all():
        raise ValueError("partial cache shard contains non-finite latents")
    bfs = payload["all_pairs_bfs"]
    if bool((bfs < 0).any()) or not torch.equal(bfs, bfs.T):
        raise ValueError("partial cache shard BFS matrix is invalid")
    if bool((torch.diagonal(bfs) != 0).any()):
        raise ValueError("partial cache shard BFS diagonal is nonzero")
    goal_index = int(metadata.get("goal_index", -1))
    if not 0 <= goal_index < cell_count or not torch.equal(
        payload["goal_distances"], bfs[:, goal_index]
    ):
        raise ValueError("partial cache shard goal distances are invalid")
    if int(cells[goal_index]) != int(entry["goal_cell"]):
        raise ValueError("partial cache shard goal index differs from manifest")
    if int(metadata.get("max_goal_distance", -1)) != int(
        payload["goal_distances"].max()
    ):
        raise ValueError("partial cache shard max goal distance is invalid")
    next_indices = payload["next_indices"].long()
    if bool(((next_indices < 0) | (next_indices >= cell_count)).any()):
        raise ValueError("partial cache shard next-state indices are invalid")
    valid = next_indices != torch.arange(cell_count)[:, None]
    valid[:, 0] = False
    if not torch.equal(payload["valid_actions"], valid):
        raise ValueError("partial cache shard valid-action labels are invalid")
    next_distances = (
        payload["goal_distances"]
        .index_select(0, next_indices.reshape(-1))
        .reshape(cell_count, MODEL_ACTION_VOCAB_SIZE)
    )
    best = (
        next_distances.masked_fill(~valid, torch.iinfo(torch.int16).max)
        .min(dim=1, keepdim=True)
        .values
    )
    if not torch.equal(payload["optimal_actions"], valid & (next_distances == best)):
        raise ValueError("partial cache shard optimal-action labels are invalid")
    content_signature = metadata.get("content_sha256")
    expected_signature = _topology_content_sha256(payload)
    if content_signature != expected_signature:
        raise ValueError("partial cache shard content hash mismatch")


def cache_index_path(
    config: StudyConfig, *, split_role: str, backbone_seed: int
) -> Path:
    return resolve_path(
        config.paths.cache_index_template.format(
            split_role=split_role, backbone_seed=int(backbone_seed)
        )
    )


def build_cache(
    config: StudyConfig,
    *,
    split_role: str,
    manifest_path: str | Path,
    backbone_seed: int,
    backbone_path: Path,
    device: torch.device,
    analysis_spec_sha256: str,
    protocol_lock_sha256: str,
    diagnostic_limit: int = 0,
    output_path: str | Path | None = None,
) -> Path:
    index_path = (
        resolve_path(output_path)
        if output_path is not None
        else cache_index_path(
            config, split_role=split_role, backbone_seed=backbone_seed
        )
    )
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite cache index: {index_path}")
    model, backbone_payload = load_backbone_checkpoint(
        backbone_path, device, freeze=True
    )
    validate_backbone_protocol_binding(
        config,
        backbone_payload,
        backbone_seed=backbone_seed,
        protocol_lock={
            "analysis_spec_sha256": analysis_spec_sha256,
            "protocol_lock_sha256": protocol_lock_sha256,
        },
    )
    entries = read_jsonl(manifest_path)
    if diagnostic_limit:
        entries = entries[:diagnostic_limit]
    records: list[dict[str, Any]] = []
    shard_root = index_path.parent / "shards"
    for position, entry in enumerate(entries):
        shard_path = shard_root / f"{position:05d}_{entry['task_hash'][:16]}.pt"
        if shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            _validate_topology_shard(
                payload,
                entry,
                backbone_path=backbone_path,
                analysis_spec_sha256=analysis_spec_sha256,
                protocol_lock_sha256=protocol_lock_sha256,
            )
        else:
            payload = build_topology_shard(
                entry,
                model,
                backbone_path=backbone_path,
                device=device,
                analysis_spec_sha256=analysis_spec_sha256,
                protocol_lock_sha256=protocol_lock_sha256,
            )
            atomic_torch_save(shard_path, payload)
        records.append(
            {
                "position": position,
                "task_hash": entry["task_hash"],
                "maze_size": int(entry["maze_size"]),
                "path": shard_path.relative_to(resolve_path(".")).as_posix(),
                "sha256": sha256_file(shard_path),
                "cell_count": payload["metadata"]["cell_count"],
            }
        )
    index = {
        "schema": CACHE_SCHEMA,
        "split_role": split_role,
        "manifest_path": resolve_path(manifest_path)
        .relative_to(resolve_path("."))
        .as_posix(),
        "manifest_sha256": sha256_file(resolve_path(manifest_path)),
        "backbone_seed": int(backbone_seed),
        "backbone_path": backbone_path.relative_to(resolve_path(".")).as_posix(),
        "backbone_sha256": sha256_file(backbone_path),
        "diagnostic_limit": int(diagnostic_limit),
        "analysis_spec_sha256": analysis_spec_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "records": records,
    }
    atomic_json_dump(index_path, index)
    return index_path


@dataclass(frozen=True)
class TrainingBatch:
    source: torch.Tensor
    goal: torch.Tensor
    raw_distance: torch.Tensor
    max_distance: torch.Tensor
    next_latents: torch.Tensor
    next_distances: torch.Tensor
    valid_actions: torch.Tensor
    optimal_actions: torch.Tensor
    history_latents: torch.Tensor
    history_actions: torch.Tensor
    path_latents: torch.Tensor
    path_distances: torch.Tensor
    triangle_latent: torch.Tensor
    triangle_source_distance: torch.Tensor
    maze_size: int
    topology_positions: torch.Tensor
    source_indices: torch.Tensor
    next_indices: torch.Tensor
    history_indices: torch.Tensor
    path_indices: torch.Tensor
    triangle_index: torch.Tensor

    def to(self, device: torch.device) -> TrainingBatch:
        values = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return TrainingBatch(
            **values,
            maze_size=self.maze_size,
        )

    def validate(self) -> None:
        batch = self.source.shape[0]
        if self.source.ndim != 2 or self.goal.shape != self.source.shape:
            raise ValueError("training source/goal latent shapes are inconsistent")
        if self.next_latents.shape[:2] != (batch, MODEL_ACTION_VOCAB_SIZE):
            raise ValueError("training next latents must cover five model actions")
        if self.next_distances.shape != (batch, MODEL_ACTION_VOCAB_SIZE):
            raise ValueError("training next distances must cover five actions")
        if self.valid_actions.shape != (batch, MODEL_ACTION_VOCAB_SIZE):
            raise ValueError("training valid-action mask is malformed")
        if self.history_latents.shape[1:] != (3, self.source.shape[-1]):
            raise ValueError("predictor history must have shape [batch,3,latent]")
        if self.history_actions.shape != (batch, 3):
            raise ValueError("predictor actions must have shape [batch,3]")
        if self.path_latents.shape[0] != batch or self.path_latents.shape[1] != 13:
            raise ValueError("shortest-path prefixes must cover steps 0..12")
        if self.path_distances.shape != (batch, 13):
            raise ValueError("shortest-path distance labels must cover steps 0..12")
        if self.triangle_latent.shape != self.source.shape:
            raise ValueError("triangle waypoint latents must match source shape")
        if self.triangle_source_distance.shape != (batch,):
            raise ValueError("triangle source-waypoint distances are malformed")
        if self.next_indices.shape != (batch, MODEL_ACTION_VOCAB_SIZE):
            raise ValueError("cached next-state indices are malformed")
        if self.history_indices.shape != (batch, 3):
            raise ValueError("cached history indices are malformed")
        if self.path_indices.shape != (batch, 13):
            raise ValueError("cached path indices are malformed")
        if self.triangle_index.shape != (batch,):
            raise ValueError("cached triangle waypoint indices are malformed")
        if not bool((self.optimal_actions <= self.valid_actions).all()):
            raise ValueError("optimal action mask contains an invalid action")


class ShardedGoalDataset:
    """Small LRU over immutable cache shards."""

    def __init__(self, index_path: str | Path, *, lru_size: int = 32) -> None:
        import json

        resolved = resolve_path(index_path)
        self.index_path = resolved
        with open(resolved, encoding="utf-8") as stream:
            self.index = json.load(stream)
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError("cache index schema mismatch")
        self.records = list(self.index["records"])
        if not self.records:
            raise ValueError("cache index contains no records")
        positions = [int(record["position"]) for record in self.records]
        if positions != list(range(len(self.records))):
            raise ValueError("cache index positions are not contiguous and ordered")
        task_hashes = [str(record["task_hash"]) for record in self.records]
        if len(task_hashes) != len(set(task_hashes)):
            raise ValueError("cache index contains duplicate tasks")
        self.by_size: dict[int, list[int]] = defaultdict(list)
        for position, record in enumerate(self.records):
            self.by_size[int(record["maze_size"])].append(position)
        self.sizes = tuple(sorted(self.by_size))
        self.lru_size = int(lru_size)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def get(self, position: int) -> dict[str, Any]:
        if position in self._cache:
            value = self._cache.pop(position)
            self._cache[position] = value
            return value
        record = self.records[position]
        path = resolve_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"cache shard hash mismatch: {path}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["metadata"]["task_hash"] != record["task_hash"]:
            raise ValueError("cache shard/index task mismatch")
        self._cache[position] = value
        while len(self._cache) > self.lru_size:
            self._cache.popitem(last=False)
        return value


def validate_cache_binding(
    dataset: ShardedGoalDataset,
    config: StudyConfig,
    *,
    split_role: str,
    backbone_seed: int,
    protocol_lock: dict[str, Any],
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Fail if a cache is not bound to the requested split and backbone."""

    index = dataset.index
    if index.get("split_role") != split_role:
        raise ValueError("cache split role differs from requested split")
    if int(index.get("backbone_seed", -1)) != int(backbone_seed):
        raise ValueError("cache backbone seed differs from requested backbone")
    if index.get("analysis_spec_sha256") != protocol_lock["analysis_spec_sha256"]:
        raise ValueError("cache analysis specification differs from protocol lock")
    if index.get("protocol_lock_sha256") != protocol_lock["protocol_lock_sha256"]:
        raise ValueError("cache protocol-lock hash differs")
    manifest_path = resolve_path(getattr(config.paths, f"{split_role}_manifest"))
    expected_manifest = protocol_lock["analysis_spec"]["manifests"][split_role]
    if index.get("manifest_sha256") != expected_manifest["sha256"]:
        raise ValueError("cache manifest hash differs from the locked manifest")
    if sha256_file(manifest_path) != expected_manifest["sha256"]:
        raise ValueError("on-disk manifest differs from the protocol lock")
    indexed_manifest = resolve_path(str(index.get("manifest_path", "")))
    if indexed_manifest != manifest_path:
        raise ValueError("cache index points at a different manifest path")
    backbone_path = source_backbone_path(config, backbone_seed)
    if index.get("backbone_sha256") != sha256_file(backbone_path):
        raise ValueError("cache backbone hash differs from the requested checkpoint")
    indexed_backbone = resolve_path(str(index.get("backbone_path", "")))
    if indexed_backbone != backbone_path:
        raise ValueError("cache index points at a different backbone path")
    diagnostic_limit = int(index.get("diagnostic_limit", 0))
    if diagnostic_limit and not allow_partial:
        raise ValueError("formal training/diagnostics cannot use a partial cache")
    expected_count = (
        min(diagnostic_limit, int(expected_manifest["count"]))
        if diagnostic_limit
        else int(expected_manifest["count"])
    )
    if len(dataset.records) != expected_count:
        raise ValueError("cache record count differs from its locked manifest binding")
    return {
        "index_path": dataset.index_path.as_posix(),
        "index_sha256": sha256_file(dataset.index_path),
        "split_role": split_role,
        "backbone_seed": int(backbone_seed),
        "analysis_spec_sha256": str(index["analysis_spec_sha256"]),
        "protocol_lock_sha256": str(index["protocol_lock_sha256"]),
        "manifest_sha256": str(index["manifest_sha256"]),
        "backbone_sha256": str(index["backbone_sha256"]),
        "record_count": len(dataset.records),
        "diagnostic_limit": diagnostic_limit,
    }


def validate_recorded_cache_binding(
    binding: dict[str, Any],
    *,
    split_role: str,
    backbone_seed: int,
    protocol_lock: dict[str, Any],
) -> Path:
    """Verify that a recorded training/diagnostic cache dependency is unchanged."""

    required = {
        "index_path",
        "index_sha256",
        "split_role",
        "backbone_seed",
        "analysis_spec_sha256",
        "protocol_lock_sha256",
    }
    if not required <= set(binding):
        raise ValueError("recorded cache binding is incomplete")
    path = resolve_path(str(binding["index_path"]))
    expected = (
        binding.get("split_role") == split_role
        and int(binding.get("backbone_seed", -1)) == int(backbone_seed)
        and binding.get("analysis_spec_sha256") == protocol_lock["analysis_spec_sha256"]
        and binding.get("protocol_lock_sha256") == protocol_lock["protocol_lock_sha256"]
    )
    if not expected:
        raise ValueError("recorded cache binding belongs to another run")
    if not path.exists() or sha256_file(path) != binding.get("index_sha256"):
        raise ValueError(f"recorded cache index changed or is missing: {path}")
    return path


def _sample_sources(
    shard: dict[str, Any],
    count: int,
    sampler: SamplerKind,
    rng: np.random.Generator,
) -> np.ndarray:
    distances = shard["goal_distances"].numpy().astype(np.int64, copy=False)
    candidates = np.flatnonzero(distances > 0)
    if not len(candidates):
        raise ValueError("cache shard has no non-goal source state")
    if sampler == SamplerKind.UNIFORM:
        return rng.choice(candidates, size=count, replace=len(candidates) < count)
    if sampler in (SamplerKind.DISTANCE_BALANCED, SamplerKind.FULL_HORIZON):
        if sampler == SamplerKind.DISTANCE_BALANCED:
            # Log-like bins estimate the global BFS-distance distribution evenly.
            boundaries = np.asarray((1, 3, 7, 11, 19, 31, 47), dtype=np.int64)
        else:
            # Planner-aligned strata isolate each preregistered rollout horizon
            # and a beyond-horizon tail: 1, 2-3, 4-5, 6-8, 9-12, >12.
            boundaries = np.asarray((1, 3, 5, 8, 12), dtype=np.int64)
        bins = np.digitize(distances[candidates], boundaries, right=True)
        occupied = np.unique(bins)
        selected = []
        for sample_index in range(count):
            desired = occupied[sample_index % len(occupied)]
            pool = candidates[bins == desired]
            selected.append(int(rng.choice(pool)))
        rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)
    if sampler == SamplerKind.DECISION_BALANCED:
        ties = shard["optimal_actions"].sum(dim=1).numpy() > 1
        groups = (candidates[ties[candidates]], candidates[~ties[candidates]])
        selected = []
        for sample_index in range(count):
            pool = groups[sample_index % 2]
            if not len(pool):
                pool = candidates
            selected.append(int(rng.choice(pool)))
        rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)
    if sampler == SamplerKind.HARD_CROSSFIT:
        raise ValueError("hard_crossfit requires a signed mining artifact")
    raise ValueError(f"unsupported sampler: {sampler}")


def _history_for_source(
    next_indices: np.ndarray,
    source: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    def predecessors(target: int) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        rows, actions = np.where(next_indices[:, 1:MODEL_ACTION_VOCAB_SIZE] == target)
        for row, action_offset in zip(rows.tolist(), actions.tolist(), strict=True):
            action = action_offset + 1
            if row != target:
                matches.append((int(row), int(action)))
        return matches

    first_options = predecessors(source)
    if not first_options:
        return np.asarray([source, source, source]), np.asarray([4, 4, 4])
    first, first_action = first_options[int(rng.integers(len(first_options)))]
    second_options = predecessors(first)
    if not second_options:
        return np.asarray([first, first, source]), np.asarray([4, 4, first_action])
    second, second_action = second_options[int(rng.integers(len(second_options)))]
    return (
        np.asarray([second, first, source], dtype=np.int64),
        np.asarray([4, second_action, first_action], dtype=np.int64),
    )


def sample_training_batch(
    dataset: ShardedGoalDataset,
    *,
    sampler: SamplerKind,
    effective_batch_size: int,
    pairs_per_topology: int,
    schedule_seed: int,
    backbone_seed: int,
    step: int,
) -> TrainingBatch:
    if effective_batch_size % pairs_per_topology:
        raise ValueError("batch size must divide by pairs-per-topology")
    rng = np.random.default_rng(
        hierarchical_seed("distance-head-sample", schedule_seed, backbone_seed, step)
    )
    size = int(dataset.sizes[step % len(dataset.sizes)])
    topology_count = effective_batch_size // pairs_per_topology
    pool = np.asarray(dataset.by_size[size], dtype=np.int64)
    positions = rng.choice(
        pool, size=topology_count, replace=len(pool) < topology_count
    )
    source_rows: list[torch.Tensor] = []
    goal_rows: list[torch.Tensor] = []
    raw_rows: list[torch.Tensor] = []
    max_rows: list[torch.Tensor] = []
    next_rows: list[torch.Tensor] = []
    next_distance_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    optimal_rows: list[torch.Tensor] = []
    history_rows: list[torch.Tensor] = []
    history_action_rows: list[torch.Tensor] = []
    path_rows: list[torch.Tensor] = []
    path_distance_rows: list[torch.Tensor] = []
    triangle_rows: list[torch.Tensor] = []
    triangle_distance_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    index_rows: list[torch.Tensor] = []
    next_index_rows: list[torch.Tensor] = []
    history_index_rows: list[torch.Tensor] = []
    path_index_rows: list[torch.Tensor] = []
    triangle_index_rows: list[torch.Tensor] = []
    for position in positions.tolist():
        shard = dataset.get(int(position))
        sources = _sample_sources(shard, pairs_per_topology, sampler, rng)
        latents = shard["latents"]
        next_indices = shard["next_indices"].numpy()
        histories = [
            _history_for_source(next_indices, int(source), rng) for source in sources
        ]
        history_indices = np.stack([item[0] for item in histories])
        history_actions = np.stack([item[1] for item in histories])
        source_tensor = torch.from_numpy(sources)
        selected_next = shard["next_indices"].index_select(0, source_tensor)
        source_rows.append(latents.index_select(0, source_tensor))
        goal_index = int(shard["metadata"]["goal_index"])
        goal_rows.append(latents[goal_index].expand(len(sources), -1))
        raw_rows.append(shard["goal_distances"].index_select(0, source_tensor).float())
        max_rows.append(
            torch.full((len(sources),), float(shard["metadata"]["max_goal_distance"]))
        )
        next_rows.append(
            latents.index_select(0, selected_next.reshape(-1)).reshape(
                len(sources), MODEL_ACTION_VOCAB_SIZE, -1
            )
        )
        next_index_rows.append(selected_next)
        next_distance_rows.append(
            shard["goal_distances"]
            .index_select(0, selected_next.reshape(-1))
            .reshape(len(sources), MODEL_ACTION_VOCAB_SIZE)
            .float()
        )
        valid_rows.append(shard["valid_actions"].index_select(0, source_tensor))
        optimal_rows.append(shard["optimal_actions"].index_select(0, source_tensor))
        history_rows.append(
            latents.index_select(
                0, torch.from_numpy(history_indices.reshape(-1))
            ).reshape(len(sources), 3, -1)
        )
        history_index_rows.append(torch.from_numpy(history_indices))
        history_action_rows.append(torch.from_numpy(history_actions))
        path_indices = np.empty((len(sources), 13), dtype=np.int64)
        for row_index, source_index in enumerate(sources.tolist()):
            cursor = int(source_index)
            path_indices[row_index, 0] = cursor
            for path_step in range(1, 13):
                # A shortest-path prefix terminates at the absorbing goal.  The
                # environment itself permits moving away from the goal, so the
                # generic local-action mask cannot define this supervision rule.
                if int(shard["goal_distances"][cursor]) > 0:
                    optimal = np.flatnonzero(shard["optimal_actions"][cursor].numpy())
                    if not len(optimal):
                        raise ValueError(
                            "non-goal state has no optimal action in cache shard"
                        )
                    cursor = int(next_indices[cursor, int(optimal[0])])
                path_indices[row_index, path_step] = cursor
        flat_path = torch.from_numpy(path_indices.reshape(-1))
        path_rows.append(
            latents.index_select(0, flat_path).reshape(len(sources), 13, -1)
        )
        path_index_rows.append(torch.from_numpy(path_indices))
        path_distance_rows.append(
            shard["goal_distances"]
            .index_select(0, flat_path)
            .reshape(len(sources), 13)
            .float()
        )
        goal_index = int(shard["metadata"]["goal_index"])
        triangle_indices = np.empty(len(sources), dtype=np.int64)
        triangle_distances = np.empty(len(sources), dtype=np.float32)
        all_pairs = shard["all_pairs_bfs"].numpy()
        all_indices = np.arange(latents.shape[0], dtype=np.int64)
        for row_index, source_index in enumerate(sources.tolist()):
            waypoint_pool = all_indices[
                (all_indices != int(source_index)) & (all_indices != goal_index)
            ]
            if not len(waypoint_pool):
                raise ValueError("topology has no nontrivial triangle waypoint")
            waypoint = int(rng.choice(waypoint_pool))
            triangle_indices[row_index] = waypoint
            triangle_distances[row_index] = float(
                all_pairs[int(source_index), waypoint]
            )
        triangle_index_tensor = torch.from_numpy(triangle_indices)
        triangle_rows.append(latents.index_select(0, triangle_index_tensor))
        triangle_distance_rows.append(torch.from_numpy(triangle_distances))
        triangle_index_rows.append(triangle_index_tensor)
        position_rows.append(
            torch.full((len(sources),), int(position), dtype=torch.long)
        )
        index_rows.append(source_tensor)
    batch = TrainingBatch(
        source=torch.cat(source_rows),
        goal=torch.cat(goal_rows),
        raw_distance=torch.cat(raw_rows),
        max_distance=torch.cat(max_rows),
        next_latents=torch.cat(next_rows),
        next_distances=torch.cat(next_distance_rows),
        valid_actions=torch.cat(valid_rows),
        optimal_actions=torch.cat(optimal_rows),
        history_latents=torch.cat(history_rows),
        history_actions=torch.cat(history_action_rows),
        path_latents=torch.cat(path_rows),
        path_distances=torch.cat(path_distance_rows),
        triangle_latent=torch.cat(triangle_rows),
        triangle_source_distance=torch.cat(triangle_distance_rows),
        maze_size=size,
        topology_positions=torch.cat(position_rows),
        source_indices=torch.cat(index_rows),
        next_indices=torch.cat(next_index_rows),
        history_indices=torch.cat(history_index_rows),
        path_indices=torch.cat(path_index_rows),
        triangle_index=torch.cat(triangle_index_rows),
    )
    batch.validate()
    if batch.source.shape[0] != effective_batch_size:
        raise RuntimeError("deterministic sampler returned the wrong batch size")
    return batch


def refresh_joint_latents(
    dataset: ShardedGoalDataset,
    batch: TrainingBatch,
    model: Unisize256,
    *,
    device: torch.device,
    gradients: bool = True,
) -> TrainingBatch:
    """Re-encode every supervised state in the current trainable latent space."""

    source_observations: list[torch.Tensor] = []
    goal_observations: list[torch.Tensor] = []
    history_observations: list[torch.Tensor] = []
    next_observations: list[torch.Tensor] = []
    path_observations: list[torch.Tensor] = []
    triangle_observations: list[torch.Tensor] = []
    for row, position in enumerate(batch.topology_positions.tolist()):
        shard = dataset.get(int(position))
        observations = shard["observations"]
        source_observations.append(observations[int(batch.source_indices[row])])
        goal_observations.append(observations[int(shard["metadata"]["goal_index"])])
        history_observations.append(
            observations.index_select(0, batch.history_indices[row])
        )
        next_observations.append(observations.index_select(0, batch.next_indices[row]))
        path_observations.append(observations.index_select(0, batch.path_indices[row]))
        triangle_observations.append(observations[int(batch.triangle_index[row])])
    source = encode_observations(
        model,
        torch.stack(source_observations),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    )
    goal = encode_observations(
        model,
        torch.stack(goal_observations),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    )
    history_shape = batch.history_latents.shape
    history = encode_observations(
        model,
        torch.stack(history_observations).reshape(
            -1, *history_observations[0].shape[1:]
        ),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    ).reshape(history_shape)
    next_shape = batch.next_latents.shape
    next_latents = encode_observations(
        model,
        torch.stack(next_observations).reshape(-1, *next_observations[0].shape[1:]),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    ).reshape(next_shape)
    path_shape = batch.path_latents.shape
    path_latents = encode_observations(
        model,
        torch.stack(path_observations).reshape(-1, *path_observations[0].shape[1:]),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    ).reshape(path_shape)
    triangle_latent = encode_observations(
        model,
        torch.stack(triangle_observations),
        maze_size=batch.maze_size,
        device=device,
        gradients=gradients,
    )
    values = dict(batch.__dict__)
    values.update(
        source=source,
        goal=goal,
        history_latents=history,
        next_latents=next_latents,
        path_latents=path_latents,
        triangle_latent=triangle_latent,
    )
    refreshed = TrainingBatch(**values)
    refreshed.validate()
    return refreshed


def true_candidate_distances(
    dataset: ShardedGoalDataset,
    batch: TrainingBatch,
    candidate_actions: torch.Tensor,
    *,
    context_indices: torch.Tensor,
    executed_action_count: int | None = None,
) -> torch.Tensor:
    """Apply candidate action sequences to the cached true transition graph."""

    if candidate_actions.ndim != 3:
        raise ValueError("candidate actions need shape [context,candidate,horizon]")
    contexts, candidates, _ = candidate_actions.shape
    action_count = (
        candidate_actions.shape[2]
        if executed_action_count is None
        else int(executed_action_count)
    )
    if not 0 <= action_count <= candidate_actions.shape[2]:
        raise ValueError("executed action count is outside the candidate horizon")
    if context_indices.shape != (contexts,):
        raise ValueError("candidate context index shape mismatch")
    result = torch.empty((contexts, candidates), dtype=torch.float32)
    for output_row, batch_index in enumerate(context_indices.tolist()):
        position = int(batch.topology_positions[batch_index])
        shard = dataset.get(position)
        transitions = shard["next_indices"].numpy()
        distances = shard["goal_distances"].numpy()
        for candidate_index in range(candidates):
            state = int(batch.source_indices[batch_index])
            for action in candidate_actions[
                output_row, candidate_index, :action_count
            ].tolist():
                state = int(transitions[state, int(action)])
            result[output_row, candidate_index] = float(distances[state])
    return result


def evenly_spaced_indices(batch_size: int, count: int) -> torch.Tensor:
    """Choose deterministic, topology-spread rows from a grouped training batch."""

    if batch_size < 1 or not 1 <= count <= batch_size:
        raise ValueError("context count must lie inside the nonempty batch")
    positions = torch.arange(count, dtype=torch.long)
    return torch.div(
        (positions * 2 + 1) * batch_size,
        2 * count,
        rounding_mode="floor",
    )


def slice_training_batch(batch: TrainingBatch, start: int, stop: int) -> TrainingBatch:
    if start < 0 or stop > batch.source.shape[0] or start >= stop:
        raise ValueError("invalid training microbatch bounds")
    values = {
        name: value[start:stop] if isinstance(value, torch.Tensor) else value
        for name, value in batch.__dict__.items()
    }
    sliced = TrainingBatch(**values)
    sliced.validate()
    return sliced


__all__ = [
    "CACHE_SCHEMA",
    "ShardedGoalDataset",
    "TrainingBatch",
    "build_cache",
    "build_topology_shard",
    "cache_index_path",
    "evenly_spaced_indices",
    "encode_observations",
    "load_backbone_checkpoint",
    "refresh_joint_latents",
    "sample_training_batch",
    "slice_training_batch",
    "true_candidate_distances",
    "validate_cache_binding",
    "validate_recorded_cache_binding",
]
