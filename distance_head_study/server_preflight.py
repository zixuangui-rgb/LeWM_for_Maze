"""Run a small real-checkpoint I/O, cache, predictor, and loss preflight."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from distance_head_study.common import (
    load_study_config,
    read_jsonl,
    require_clean_worktree,
    resolve_device,
    set_seed,
    sha256_file,
    source_backbone_path,
    validate_backbone_protocol_binding,
)
from distance_head_study.data import (
    CACHE_SCHEMA,
    ShardedGoalDataset,
    build_topology_shard,
    load_backbone_checkpoint,
    sample_training_batch,
)
from distance_head_study.losses import compute_objective_terms, weighted_total
from distance_head_study.methods import load_and_resolve_method
from distance_head_study.models import build_distance_head
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.schemas import SamplerKind
from distance_head_study.train_head import _predict_all_actions
from vector_jepa_planner_frontier.schemas import RolloutSemantics
from vector_jepa_planner_frontier.world_model import VectorContext, VectorWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--backbone-seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    device = resolve_device(args.device or config.device)
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else source_backbone_path(config, args.backbone_seed)
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    set_seed(12345, deterministic=True)
    model, payload = load_backbone_checkpoint(checkpoint, device, freeze=True)
    validate_backbone_protocol_binding(
        config,
        payload,
        backbone_seed=args.backbone_seed,
        protocol_lock=lock,
    )
    entry = min(
        read_jsonl(config.paths.train_manifest), key=lambda row: int(row["maze_size"])
    )
    shard = build_topology_shard(
        entry,
        model,
        backbone_path=checkpoint,
        device=device,
        encode_batch_size=128,
    )
    with tempfile.TemporaryDirectory(prefix="distance-head-preflight-") as temporary:
        root = Path(temporary)
        shard_path = root / "shard.pt"
        torch.save(shard, shard_path)
        index = {
            "schema": CACHE_SCHEMA,
            "records": [
                {
                    "position": 0,
                    "task_hash": entry["task_hash"],
                    "maze_size": int(entry["maze_size"]),
                    "path": shard_path.as_posix(),
                    "sha256": sha256_file(shard_path),
                }
            ],
        }
        index_path = root / "index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        dataset = ShardedGoalDataset(index_path)
        batch = sample_training_batch(
            dataset,
            sampler=SamplerKind.UNIFORM,
            effective_batch_size=16,
            pairs_per_topology=16,
            schedule_seed=1,
            backbone_seed=args.backbone_seed,
            step=0,
        ).to(device)
    predicted = _predict_all_actions(model, batch, gradients=False)
    method, _, _ = load_and_resolve_method(
        config.paths.method_catalog,
        "b_dh_cem",
        decision_root=config.paths.decision_root,
        protocol_lock=lock,
    )
    assert method.head is not None and method.objectives is not None
    head = build_distance_head(method.head).to(device)
    terms = compute_objective_terms(head, method, batch)
    weights = {name: float(getattr(method.objectives, name)) for name in terms}
    loss = weighted_total(terms, weights)
    loss.backward()
    world_model = VectorWorldModel(model, device=device, history_size=3)
    context = VectorContext(
        embeddings=batch.history_latents[:1],
        actions=batch.history_actions[:1],
        goal=batch.goal[:1, None],
        maze_size=batch.maze_size,
        remaining_steps=128,
    )
    candidate_actions = np.ones((4, config.planner.horizon), dtype=np.int64)
    rollout = world_model.rollout(
        context,
        candidate_actions,
        semantics=RolloutSemantics.LEGACY_WARMUP_V1,
    )
    report = {
        "status": "pass",
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": lock["analysis_spec_sha256"],
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_model_config_present": "model_config" in payload,
        "maze_size": batch.maze_size,
        "cache_state_count": int(shard["latents"].shape[0]),
        "predicted_all_actions_shape": list(predicted.shape),
        "head_loss": float(loss.detach()),
        "rollout_shape": list(rollout.states.shape),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
