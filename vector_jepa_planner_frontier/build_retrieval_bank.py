"""Build a train-topology-only latent action-chunk retrieval bank."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from final_closure.common import read_jsonl, sha256_file
from vector_jepa_planner_frontier.common import (
    analysis_spec_sha256,
    hierarchical_seed,
    load_json,
    load_study_config,
    method_by_name,
    planner_seed_values,
    require_clean_worktree,
    resolve_device,
    resolve_path,
    set_seed,
    training_spec_sha256,
    validate_locked_artifacts,
)
from vector_jepa_planner_frontier.compat import load_source_lewm
from vector_jepa_planner_frontier.data import PlannerBatchSampler, encode_planner_batch
from vector_jepa_planner_frontier.effective_methods import resolve_effective_method
from vector_jepa_planner_frontier.proposals import RetrievalBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="vector_jepa_planner_frontier/configs/default.json"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone-seed", type=int, required=True)
    parser.add_argument("--planner-seed", type=int, required=True)
    parser.add_argument("--input-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-dirty-worktree", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    lock = load_json(config.paths.protocol_lock)
    validate_locked_artifacts(config, lock)
    if lock.get("status") != "locked":
        raise RuntimeError("retrieval-bank creation requires a completed protocol lock")
    if lock.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
        raise ValueError("config no longer matches the locked analysis specification")
    method = resolve_effective_method(config, lock, method_by_name(config, args.method))
    if method.reuse_component_from is not None:
        raise ValueError("checkpoint reuse aliases cannot rebuild retrieval banks")
    if args.allow_dirty_worktree:
        raise ValueError("formal retrieval-bank creation requires a clean worktree")
    if args.backbone_seed not in config.protocol.training_seeds:
        raise ValueError("backbone seed lies outside the locked matrix")
    if args.planner_seed not in planner_seed_values(config, method):
        raise ValueError("planner seed lies outside this method's locked matrix")
    if method.proposal.retrieval_weight <= 0.0:
        raise ValueError("selected method does not use retrieval proposals")
    chunk_count = args.chunks or config.training.retrieval_bank_chunks
    if chunk_count <= 0:
        raise ValueError("retrieval bank size must be positive")
    if args.chunks is not None and args.chunks != config.training.retrieval_bank_chunks:
        raise ValueError("formal retrieval bank cannot override the locked size")
    output = resolve_path(
        args.output
        or config.paths.retrieval_bank_template.format(
            method=method.name,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        )
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retrieval bank: {output}")
    require_clean_worktree(allow_dirty=args.allow_dirty_worktree)
    device = resolve_device(args.device or config.device)
    set_seed(
        hierarchical_seed("retrieval-bank", args.backbone_seed, args.planner_seed),
        deterministic=True,
    )
    model, _, source_path = load_source_lewm(
        config, lock, seed=args.backbone_seed, device=device
    )
    if method.track == "J":
        checkpoint_path = resolve_path(
            args.input_checkpoint
            or config.paths.component_training_template.format(
                method=method.name,
                backbone_seed=args.backbone_seed,
                planner_seed=args.planner_seed,
            )
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "component_training":
            raise ValueError("joint retrieval bank requires a training checkpoint")
        if (
            checkpoint.get("method_name") != method.name
            or int(checkpoint.get("backbone_seed", -1)) != args.backbone_seed
            or int(checkpoint.get("planner_seed", -1)) != args.planner_seed
        ):
            raise ValueError("joint retrieval checkpoint method/seed mismatch")
        if checkpoint.get("analysis_spec_sha256") != analysis_spec_sha256(config, lock):
            raise ValueError("joint-training checkpoint analysis-spec mismatch")
        if checkpoint.get("training_spec_sha256") != training_spec_sha256(
            config,
            lock,
            method=method,
            backbone_seed=args.backbone_seed,
            planner_seed=args.planner_seed,
        ):
            raise ValueError("joint-training checkpoint training-spec mismatch")
        if checkpoint.get("source_checkpoint_sha256") != sha256_file(source_path):
            raise ValueError("joint-training and retrieval source checkpoints differ")
        checkpoint_protocol = checkpoint.get("protocol", {})
        if checkpoint_protocol.get("git_dirty") is not False:
            raise ValueError("retrieval bank rejects a dirty training checkpoint")
        if checkpoint_protocol.get("code_fingerprint") != lock["code_fingerprint"]:
            raise ValueError("joint-training checkpoint code fingerprint mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    train_path = resolve_path(config.paths.train_manifest)
    if sha256_file(train_path) != lock["train_manifest"]["sha256"]:
        raise ValueError("training manifest hash mismatch")
    sampler = PlannerBatchSampler(
        read_jsonl(train_path), horizon=method.planner.horizon
    )
    rng = np.random.default_rng(
        hierarchical_seed(
            "retrieval-bank-sampler", args.backbone_seed, args.planner_seed
        )
    )
    sources: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    task_hashes: list[str] = []
    while sum(item.shape[0] for item in sources) < chunk_count:
        count = min(
            config.training.proposal_batch_size,
            chunk_count - sum(item.shape[0] for item in sources),
        )
        batch = sampler.sample(rng, batch_size=count, device=device)
        latents = encode_planner_batch(model, batch, gradients=False)
        sources.append(latents["source"].cpu())
        goals.append(latents["goal"].cpu())
        action_chunks.append(batch.optimal_action_chunks.cpu())
        task_hashes.extend(batch.task_hashes)
    bank = RetrievalBank(
        source_latents=torch.cat(sources),
        goal_latents=torch.cat(goals),
        action_chunks=torch.cat(action_chunks),
        task_hashes=tuple(task_hashes),
        topology_role="train",
    )
    bank.save(output)


if __name__ == "__main__":
    main()
