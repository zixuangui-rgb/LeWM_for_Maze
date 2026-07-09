#!/usr/bin/env python3
"""Fast CPU smoke tests for the planning_repair package."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from planning_repair.common import (
    build_or_load_model,
    compute_maze_supervision,
    load_backbone_from_repair_ckpt,
    set_seed,
)
from planning_repair.heads import (
    ActionPrefixPredictor,
    EmbeddingAuxConfig,
    EmbeddingAuxHeads,
    PrefixPredictorConfig,
    load_aux_heads,
    load_prefix_predictor,
    soft_target_cross_entropy,
)


def main() -> None:
    set_seed(123)
    device = torch.device("cpu")
    model, cfg, _ = build_or_load_model(None, device, latent_dim=32)
    model.train()
    env = ProcgenMazeEnv(
        ProcgenMazeConfig(
            height=9,
            width=9,
            observation_channels=5,
            p_noise=0.0,
            p_noop=0.0,
            p_action_turn=0.0,
            p_action_stay=0.0,
            resample_maze_per_sequence=False,
            topology_seed=90001,
        ),
        seed=123,
    )
    batch = env.sample_sequence(batch_size=2, sequence_length=4)
    obs = batch.observations.to(dtype=torch.float32)
    out = model(obs, batch.actions, 9)
    assert out["embedding"].shape == (2, 4, 32)
    assert out["prediction"].shape == (2, 3, 32)

    labels = compute_maze_supervision(
        states=batch.states,
        env=env,
        size=9,
        device=device,
        budgets=(1, 3, 5),
    )
    assert labels["valid_action"].shape == (2, 4, 4)
    assert labels["reachability"].shape == (2, 4, 3)

    aux_cfg = EmbeddingAuxConfig(latent_dim=32, hidden_dim=64, reach_budgets=(1, 3, 5))
    aux = EmbeddingAuxHeads(aux_cfg)
    aux_out = aux(out["embedding"])
    assert aux_out["action_logits"].shape == (2, 4, 4)
    ce = soft_target_cross_entropy(aux_out["action_logits"], labels["optimal_action_mask"])
    assert ce.ndim == 0

    prefix_cfg = PrefixPredictorConfig(
        latent_dim=32,
        hidden_dim=64,
        action_vocab_size=int(cfg.action_vocab_size),
        max_horizon=3,
    )
    prefix = ActionPrefixPredictor(prefix_cfg)
    prefix_out = prefix(out["embedding"][:, 0], batch.actions[:, :3])
    assert prefix_out.shape == (2, 3, 32)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "smoke.pt"
        torch.save(
            {
                "model_config": cfg,
                "model_state_dict": model.state_dict(),
                "aux_config": aux_cfg.to_dict(),
                "aux_state_dict": aux.state_dict(),
                "prefix_config": prefix_cfg.to_dict(),
                "prefix_state_dict": prefix.state_dict(),
            },
            ckpt,
        )
        loaded_model, data = load_backbone_from_repair_ckpt(ckpt, device)
        loaded_aux = load_aux_heads(data, device)
        loaded_prefix = load_prefix_predictor(data, device)
        assert loaded_model is not None
        assert loaded_aux is not None
        assert loaded_prefix is not None

    print("planning_repair smoke test passed")


if __name__ == "__main__":
    main()
