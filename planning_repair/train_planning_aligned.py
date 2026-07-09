#!/usr/bin/env python3
"""Train planning-aligned Maze-JEPA repair variants.

The script intentionally keeps the backbone architecture compatible with the
existing diagnostics loader. Extra heads are saved separately in the same
checkpoint, while ``model_state_dict`` remains a strict ``Unisize256`` state.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.losses import SIGReg
from planning_repair.common import (
    build_or_load_model,
    compute_maze_supervision,
    load_manifest_pair,
    parse_int_list,
    set_seed,
)
from planning_repair.heads import (
    ActionPrefixPredictor,
    EmbeddingAuxConfig,
    EmbeddingAuxHeads,
    PrefixPredictorConfig,
    soft_target_cross_entropy,
)
from scripts.train.train_canonical_lewm import compute_position_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train planning-aligned LeWM variants.")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--eval-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--init-model-ckpt", default=None)
    parser.add_argument("--variant-name", default="planning_aligned")
    parser.add_argument(
        "--output",
        default="checkpoints/planning_repair/planning_aligned.pt",
    )
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=31)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Baseline LeWM losses.
    parser.add_argument("--lambda-sigreg", type=float, default=0.09)
    parser.add_argument("--lambda-encoded-abs", type=float, default=0.1)
    parser.add_argument("--lambda-encoded-rel", type=float, default=1.0)
    parser.add_argument("--lambda-encoded-goal", type=float, default=0.5)

    # Planning-aligned embedding losses.
    parser.add_argument("--lambda-emb-agent", type=float, default=0.25)
    parser.add_argument("--lambda-emb-goal", type=float, default=0.25)
    parser.add_argument("--lambda-valid", type=float, default=0.5)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-bfs", type=float, default=0.25)
    parser.add_argument("--lambda-reach", type=float, default=0.25)
    parser.add_argument("--reach-budgets", default="1,3,5,8,12")
    parser.add_argument("--aux-hidden-dim", type=int, default=256)
    parser.add_argument("--aux-dropout", type=float, default=0.0)

    # Optional Fast-LeWM-style prefix predictor.
    parser.add_argument("--lambda-prefix", type=float, default=0.0)
    parser.add_argument("--prefix-horizon", type=int, default=5)
    parser.add_argument("--prefix-hidden-dim", type=int, default=256)
    parser.add_argument("--prefix-layers", type=int, default=1)
    parser.add_argument("--prefix-dropout", type=float, default=0.0)
    parser.add_argument("--prefix-target-detach", action="store_true", default=True)
    parser.add_argument("--no-prefix-target-detach", dest="prefix_target_detach", action="store_false")
    return parser.parse_args()


def make_env(entry: dict[str, Any], rng: np.random.Generator) -> ProcgenMazeEnv:
    size = int(entry["maze_size"])
    config = ProcgenMazeConfig(
        height=size,
        width=size,
        observation_channels=5,
        p_noise=0.0,
        p_noop=0.0,
        p_action_turn=0.0,
        p_action_stay=0.0,
        resample_maze_per_sequence=False,
        topology_seed=int(entry["topology_seed"]),
    )
    return ProcgenMazeEnv(config, seed=int(rng.integers(2**31 - 1)))


def goal_position_loss(output: dict[str, torch.Tensor], obs: torch.Tensor, size: int) -> torch.Tensor:
    batch_size, seq_len = obs.shape[:2]
    goal_flat = obs[:, :, :, :, ProcgenMazeEnv.CH_GOAL].reshape(batch_size, seq_len, -1)
    goal_state = goal_flat.argmax(dim=-1)
    goal_x = (goal_state % size).float() / max(size - 1, 1)
    goal_y = (goal_state // size).float() / max(size - 1, 1)
    goal_target = torch.stack([goal_x, goal_y], dim=-1).to(obs.device)
    return F.mse_loss(output["goal_pos_pred"], goal_target)


def loss_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def main() -> None:
    args = parse_args()
    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if args.prefix_horizon >= args.seq_len:
        args.prefix_horizon = args.seq_len - 1

    budgets = parse_int_list(args.reach_budgets)
    device = torch.device(args.device)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_entries, eval_entries, overlap = load_manifest_pair(
        args.train_manifest,
        args.eval_manifest,
    )
    del eval_entries
    print("=" * 80)
    print("TRAIN PLANNING-ALIGNED MAZE-JEPA")
    print("=" * 80)
    print(f"train_entries={len(train_entries)} holdout={overlap}")
    print(f"output={args.output}")
    print(f"device={device}")

    model, model_config, model_meta = build_or_load_model(
        args.init_model_ckpt,
        device,
        latent_dim=args.latent_dim,
        max_size=args.max_size,
    )
    aux_config = EmbeddingAuxConfig(
        latent_dim=int(model_config.latent_dim),
        hidden_dim=args.aux_hidden_dim,
        action_slots=4,
        reach_budgets=budgets,
        dropout=args.aux_dropout,
    )
    aux_heads = EmbeddingAuxHeads(aux_config).to(device)

    prefix_predictor: ActionPrefixPredictor | None = None
    prefix_config: PrefixPredictorConfig | None = None
    if args.lambda_prefix > 0.0:
        prefix_config = PrefixPredictorConfig(
            latent_dim=int(model_config.latent_dim),
            hidden_dim=args.prefix_hidden_dim,
            action_vocab_size=int(model_config.action_vocab_size),
            max_horizon=args.prefix_horizon,
            num_layers=args.prefix_layers,
            dropout=args.prefix_dropout,
        )
        prefix_predictor = ActionPrefixPredictor(prefix_config).to(device)

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    params = list(model.parameters()) + list(aux_heads.parameters())
    if prefix_predictor is not None:
        params += list(prefix_predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    aux_heads.train()
    if prefix_predictor is not None:
        prefix_predictor.train()

    running: dict[str, list[float]] = {
        "total": [],
        "pred": [],
        "sigreg": [],
        "encoded_abs": [],
        "encoded_rel": [],
        "encoded_goal": [],
        "emb_agent": [],
        "emb_goal": [],
        "valid": [],
        "action": [],
        "bfs": [],
        "reach": [],
        "prefix": [],
    }
    t0 = time.time()

    for step in range(1, args.steps + 1):
        entry = train_entries[int(rng.integers(len(train_entries)))]
        size = int(entry["maze_size"])
        env = make_env(entry, rng)
        batch = env.sample_sequence(
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
        )
        obs = batch.observations.to(device=device, dtype=torch.float32)
        actions = batch.actions.to(device=device)
        output = model(obs, actions, size)

        pred_loss = F.mse_loss(output["prediction"], output["target"])
        sigreg_loss = sigreg(output["sigreg_embedding"].transpose(0, 1))

        x, y, dx, dy = compute_position_labels(
            batch.states,
            obs[:, :, :, :, ProcgenMazeEnv.CH_GOAL],
            size,
        )
        abs_target = torch.stack([x, y], dim=-1).to(device)
        rel_target = torch.stack([dx, dy], dim=-1).to(device)
        encoded_abs_loss = F.mse_loss(output["abs_pos_pred"], abs_target)
        encoded_rel_loss = F.mse_loss(output["rel_pos_pred"], rel_target)
        encoded_goal_loss = goal_position_loss(output, obs, size)

        labels = compute_maze_supervision(
            states=batch.states,
            env=env,
            size=size,
            device=device,
            budgets=budgets,
        )
        aux = aux_heads(output["embedding"])
        emb_agent_loss = F.mse_loss(aux["agent_xy"], labels["agent_xy"])
        emb_goal_loss = F.mse_loss(aux["goal_xy"], labels["goal_xy"])
        valid_loss = F.binary_cross_entropy_with_logits(
            aux["valid_action_logits"],
            labels["valid_action"],
        )
        action_loss = soft_target_cross_entropy(
            aux["action_logits"],
            labels["optimal_action_mask"],
        )
        bfs_loss = F.smooth_l1_loss(
            aux["bfs_distance_norm"],
            labels["bfs_distance_norm"],
        )
        reach_loss = F.binary_cross_entropy_with_logits(
            aux["reachability_logits"],
            labels["reachability"],
        )

        prefix_loss = output["embedding"].new_tensor(0.0)
        if prefix_predictor is not None and args.lambda_prefix > 0.0:
            horizon = min(args.prefix_horizon, actions.shape[1])
            prefix_pred = prefix_predictor(output["embedding"][:, 0], actions[:, :horizon])
            prefix_target = output["embedding"][:, 1 : horizon + 1]
            if args.prefix_target_detach:
                prefix_target = prefix_target.detach()
            prefix_loss = F.mse_loss(prefix_pred, prefix_target)

        total = (
            pred_loss
            + args.lambda_sigreg * sigreg_loss
            + args.lambda_encoded_abs * encoded_abs_loss
            + args.lambda_encoded_rel * encoded_rel_loss
            + args.lambda_encoded_goal * encoded_goal_loss
            + args.lambda_emb_agent * emb_agent_loss
            + args.lambda_emb_goal * emb_goal_loss
            + args.lambda_valid * valid_loss
            + args.lambda_action * action_loss
            + args.lambda_bfs * bfs_loss
            + args.lambda_reach * reach_loss
            + args.lambda_prefix * prefix_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        values = {
            "total": total,
            "pred": pred_loss,
            "sigreg": sigreg_loss,
            "encoded_abs": encoded_abs_loss,
            "encoded_rel": encoded_rel_loss,
            "encoded_goal": encoded_goal_loss,
            "emb_agent": emb_agent_loss,
            "emb_goal": emb_goal_loss,
            "valid": valid_loss,
            "action": action_loss,
            "bfs": bfs_loss,
            "reach": reach_loss,
            "prefix": prefix_loss,
        }
        for name, value in values.items():
            running[name].append(loss_value(value))

        if step % args.log_every == 0 or step == args.steps:
            elapsed = max(time.time() - t0, 1e-6)
            window = min(args.log_every, step)
            avg = {
                name: float(np.mean(vals[-window:])) if vals else 0.0
                for name, vals in running.items()
            }
            print(
                f"step={step:>6d}/{args.steps} "
                f"total={avg['total']:.4f} pred={avg['pred']:.4f} "
                f"sigreg={avg['sigreg']:.4f} emb_xy={avg['emb_agent']:.4f} "
                f"valid={avg['valid']:.4f} action={avg['action']:.4f} "
                f"bfs={avg['bfs']:.4f} reach={avg['reach']:.4f} "
                f"prefix={avg['prefix']:.4f} ips={window / elapsed:.1f}",
                flush=True,
            )
            t0 = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        name: float(np.mean(values[-min(len(values), args.log_every) :]))
        if values
        else 0.0
        for name, values in running.items()
    }
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "aux_state_dict": aux_heads.state_dict(),
        "aux_config": aux_config.to_dict(),
        "training_args": vars(args),
        "training_summary": summary,
        "model_metadata": model_meta,
        "holdout": overlap,
        "repair_protocol": {
            "variant_name": args.variant_name,
            "train_manifest": args.train_manifest,
            "eval_manifest": args.eval_manifest,
            "init_model_ckpt": args.init_model_ckpt,
            "same_backbone_class": "scripts.train.train_dim256.Unisize256",
            "strict_old_diagnostics_compatible": True,
            "active_components": {
                "embedding_xy": args.lambda_emb_agent > 0.0 or args.lambda_emb_goal > 0.0,
                "valid_action": args.lambda_valid > 0.0,
                "action_ranking": args.lambda_action > 0.0,
                "bfs_distance": args.lambda_bfs > 0.0,
                "reachability": args.lambda_reach > 0.0,
                "prefix_predictor": args.lambda_prefix > 0.0,
            },
        },
    }
    if prefix_predictor is not None and prefix_config is not None:
        checkpoint["prefix_state_dict"] = prefix_predictor.state_dict()
        checkpoint["prefix_config"] = prefix_config.to_dict()
    torch.save(checkpoint, output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
