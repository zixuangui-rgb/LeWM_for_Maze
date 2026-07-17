"""Immutable candidate banks shared by all trajectory-ranking methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from distance_head_study import ACTION_IDS, MODEL_ACTION_VOCAB_SIZE
from distance_head_study.common import (
    atomic_torch_save,
    canonical_json_sha256,
    hierarchical_seed,
    load_study_config,
    require_clean_worktree,
    resolve_path,
)
from distance_head_study.protocol import verify_protocol_lock
from distance_head_study.schemas import StudyConfig

CANDIDATE_SCHEMA = "distance-head-candidate-bank-v1"


def candidate_bank_path(
    config: StudyConfig, *, split_role: str, backbone_seed: int
) -> Path:
    return resolve_path(
        config.paths.candidate_bank_template.format(
            split_role=split_role,
            backbone_seed=int(backbone_seed),
        )
    )


def generate_candidate_bank(
    config: StudyConfig,
    *,
    split_role: str,
    backbone_seed: int,
    output: Path | None = None,
    analysis_spec_sha256: str | None = None,
    protocol_lock_sha256: str | None = None,
) -> Path:
    path = output or candidate_bank_path(
        config, split_role=split_role, backbone_seed=backbone_seed
    )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate bank: {path}")
    count = int(config.training.candidate_sets_per_backbone)
    candidates = int(config.training.trajectory_candidates)
    horizon = int(config.planner.horizon)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        hierarchical_seed(
            "distance-head-candidate-bank",
            config.seeds.sample_schedule_seed,
            backbone_seed,
        )
    )
    actions = torch.randint(
        low=min(ACTION_IDS),
        high=MODEL_ACTION_VOCAB_SIZE,
        size=(count, candidates, horizon),
        generator=generator,
        dtype=torch.int64,
    )
    unique = torch.tensor(
        [torch.unique(item, dim=0).shape[0] for item in actions], dtype=torch.int64
    )
    if int(unique.min()) < int(0.95 * candidates):
        raise RuntimeError("candidate bank has unexpectedly many duplicate sequences")
    metadata: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "protocol_id": config.protocol_id,
        "analysis_spec_sha256": analysis_spec_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "split_role": split_role,
        "backbone_seed": int(backbone_seed),
        "set_count": count,
        "candidate_count": candidates,
        "horizon": horizon,
        "allowed_actions": list(ACTION_IDS),
        "schedule_seed": int(config.seeds.sample_schedule_seed),
        "actions_sha256": canonical_json_sha256(actions),
    }
    atomic_torch_save(path, {"metadata": metadata, "actions": actions})
    return path


def load_candidate_bank(path: str | Path) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    actions = payload.get("actions")
    if not isinstance(metadata, dict) or metadata.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate bank metadata is invalid")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError("candidate bank actions are invalid")
    if canonical_json_sha256(actions) != metadata.get("actions_sha256"):
        raise ValueError("candidate bank content hash mismatch")
    if bool(((actions < min(ACTION_IDS)) | (actions > max(ACTION_IDS))).any()):
        raise ValueError("candidate bank contains an out-of-protocol action")
    return metadata, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_head_study/configs/default.json")
    parser.add_argument("--split-role", default="train")
    parser.add_argument("--backbone-seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    require_clean_worktree()
    lock = verify_protocol_lock(config)
    print(
        generate_candidate_bank(
            config,
            split_role=args.split_role,
            backbone_seed=args.backbone_seed,
            analysis_spec_sha256=lock["analysis_spec_sha256"],
            protocol_lock_sha256=lock["protocol_lock_sha256"],
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_SCHEMA",
    "candidate_bank_path",
    "generate_candidate_bank",
    "load_candidate_bank",
]
