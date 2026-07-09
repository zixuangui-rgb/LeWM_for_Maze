#!/usr/bin/env python3
"""Render navigation trajectories as GIFs for Set B metric-head evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.qrl_head import QRLHead
from hdwm.planning import _bfs_shortest_path
from scripts.eval.eval_setb_distance_head_fixed import (
    HISTORY_SIZE,
    MAX_STEPS,
    create_env,
    encode_obs,
    manifest_task,
    moving_actions,
    observe_state,
    set_agent_state,
)
from scripts.train.train_dim256 import Unisize256


class MetricHead(Protocol):
    def __call__(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor: ...


def load_model(path: Path, device: torch.device, metric_ckpt: dict[str, Any] | None = None) -> Unisize256:
    ckpt = metric_ckpt if metric_ckpt is not None and "model_state_dict" in metric_ckpt else None
    if ckpt is None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Unisize256(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_metric_head(data: dict[str, Any], head_type: str, device: torch.device) -> nn.Module:
    cfg = data.get("config", {})
    if head_type == "distance":
        head: nn.Module = DistanceHead(
            latent_dim=int(cfg.get("latent_dim", 256)),
            hidden_dims=cfg.get("hidden_dims", [512, 256, 128]),
            input_mode=cfg.get("input_mode", "concat"),
        ).to(device)
    elif head_type == "qrl":
        head = QRLHead(
            latent_dim=int(cfg.get("latent_dim", 256)),
            hidden_dims=cfg.get("hidden_dims", [256, 128]),
            temperature=float(cfg.get("temperature", 0.1)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        raise ValueError(f"unknown head_type: {head_type}")
    head.load_state_dict(data["head_state_dict"], strict=True)
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def true_bfs(env: Any, state: int, goal: int) -> int:
    dist = _bfs_shortest_path(env._maze_mask, state, goal, env.config.width)
    return -1 if dist is None else int(dist)


def run_metric_trajectory(
    model: Unisize256,
    head: MetricHead,
    env: Any,
    start: int,
    goal: int,
    maze_size: int,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)
    cur_obs = set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path = [cur]
    steps: list[dict[str, Any]] = []

    if method == "predictor_greedy":
        num_actions = env.config.action_vocab_size
        cur_emb = encode_obs(model, cur_obs, maze_size, device)
        ctx_emb = cur_emb.repeat(1, HISTORY_SIZE, 1)
        ctx_act = torch.full((1, HISTORY_SIZE), num_actions - 1, dtype=torch.long, device=device)
    elif method == "model_free_greedy":
        ctx_emb = None
        ctx_act = None
    else:
        raise ValueError(f"unknown method: {method}")

    for step_idx in range(MAX_STEPS):
        if cur == goal:
            break
        actions = moving_actions(env, cur, previous)
        if not actions:
            actions = list(range(1, env.config.action_vocab_size))

        scores: dict[int, float] = {}
        if method == "model_free_greedy":
            for action in actions:
                next_state = int(env._next_state(cur, env._decode_action(action)))
                next_obs = observe_state(env, next_state)
                next_emb = encode_obs(model, next_obs, maze_size, device)
                with torch.no_grad():
                    scores[action] = float(head(next_emb, goal_emb).item())
        else:
            assert ctx_emb is not None and ctx_act is not None
            num_actions = env.config.action_vocab_size
            ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
            ctx_act_rep = ctx_act[:, :-1].repeat(num_actions, 1)
            ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
            with torch.no_grad():
                pred_emb = model.predictor(ctx_emb_rep, ctx_act_rep)[:, -1, :]
                goal_rep = goal_emb.expand(num_actions, -1, -1).squeeze(1)
                pred_scores = head(pred_emb, goal_rep)
            scores = {action: float(pred_scores[action].item()) for action in actions}

        action = min(scores, key=scores.get)
        prev = cur
        obs, _, _, _, info = env.step(action)
        cur = int(info["state"])
        previous = prev
        path.append(cur)
        steps.append(
            {
                "step": step_idx + 1,
                "state": cur,
                "action": int(action),
                "score": float(scores[action]),
                "true_bfs": true_bfs(env, cur, goal),
                "candidate_scores": {str(key): value for key, value in scores.items()},
            }
        )

        if method == "predictor_greedy":
            assert ctx_emb is not None and ctx_act is not None
            new_emb = encode_obs(model, obs, maze_size, device)
            ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
            ctx_act = torch.cat(
                [ctx_act[:, 1:], torch.tensor([[action]], dtype=torch.long, device=device)],
                dim=1,
            )

    success = cur == goal
    return {
        "success": success,
        "path": path,
        "steps": steps,
        "path_length": len(path) - 1,
        "initial_bfs": true_bfs(env, start, goal),
        "final_bfs": true_bfs(env, cur, goal),
        "min_bfs": min([true_bfs(env, state, goal) for state in path]),
    }


def analyze_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    path = [int(state) for state in result["path"]]
    bfs_seq = [int(step["true_bfs"]) for step in result["steps"]]
    repeated_states = len(path) - len(set(path))
    first_bfs_increase = next(
        (idx + 1 for idx in range(1, len(bfs_seq)) if bfs_seq[idx] > bfs_seq[idx - 1]),
        None,
    )
    non_improving = sum(
        1 for idx in range(1, len(bfs_seq)) if bfs_seq[idx] >= bfs_seq[idx - 1]
    )
    revisit_ratio = repeated_states / max(len(path), 1)
    reason = "success"
    if not result["success"]:
        if repeated_states > len(path) // 3:
            reason = "looping_or_revisiting"
        elif result["min_bfs"] <= 2 and result["final_bfs"] > result["min_bfs"]:
            reason = "near_goal_then_drifted"
        elif result["final_bfs"] >= result["initial_bfs"]:
            reason = "no_net_progress"
        else:
            reason = "timeout_after_partial_progress"
    return {
        "reason": reason,
        "repeated_states": repeated_states,
        "revisit_ratio": float(revisit_ratio),
        "first_bfs_increase_step": first_bfs_increase,
        "non_improving_steps": non_improving,
    }


def draw_frame(
    env: Any,
    path_prefix: list[int],
    start: int,
    goal: int,
    title: str,
    cell_px: int,
) -> np.ndarray:
    height, width = env._maze_mask.shape
    top_px = 42
    image = Image.new("RGB", (width * cell_px, height * cell_px + top_px), (246, 246, 246))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width * cell_px, top_px], fill=(250, 250, 250))
    draw.text((6, 8), title, fill=(20, 20, 20))

    offset_y = top_px
    for row in range(height):
        for col in range(width):
            x0 = col * cell_px
            y0 = offset_y + row * cell_px
            color = (36, 36, 36) if env._maze_mask[row, col] else (238, 238, 238)
            draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=color)

    for idx, state in enumerate(path_prefix):
        row, col = divmod(state, width)
        x0 = col * cell_px
        y0 = offset_y + row * cell_px
        color = (60, 120, 216) if idx < len(path_prefix) - 1 else (245, 197, 66)
        margin = max(2, cell_px // 5)
        draw.ellipse(
            [x0 + margin, y0 + margin, x0 + cell_px - margin, y0 + cell_px - margin],
            fill=color,
        )

    for state, color in [(start, (58, 168, 82)), (goal, (218, 65, 55))]:
        row, col = divmod(state, width)
        x0 = col * cell_px
        y0 = offset_y + row * cell_px
        margin = max(2, cell_px // 7)
        draw.rectangle(
            [x0 + margin, y0 + margin, x0 + cell_px - margin, y0 + cell_px - margin],
            outline=color,
            width=max(2, cell_px // 8),
        )
    return np.asarray(image)


def save_gif(
    env: Any,
    result: dict[str, Any],
    start: int,
    goal: int,
    output: Path,
    fps: float,
) -> None:
    size = int(env.config.width)
    cell_px = max(10, min(28, 420 // size))
    frames: list[np.ndarray] = []
    path = [int(state) for state in result["path"]]
    steps = result["steps"]
    for idx, _state in enumerate(path):
        if idx == 0:
            detail = f"step 0 | bfs={result['initial_bfs']}"
        else:
            step = steps[idx - 1]
            detail = f"step {idx} | a={step['action']} | bfs={step['true_bfs']} | score={step['score']:.3f}"
        frames.append(draw_frame(env, path[: idx + 1], start, goal, detail, cell_px))
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=1.0 / fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--head-ckpt", required=True)
    parser.add_argument("--head-type", choices=["distance", "qrl"], required=True)
    parser.add_argument("--method", choices=["model_free_greedy", "predictor_greedy"], default="predictor_greedy")
    parser.add_argument("--case", choices=["success", "failure", "both"], default="both")
    parser.add_argument("--maze-size", type=int, default=25)
    parser.add_argument("--num-gifs", type=int, default=2)
    parser.add_argument("--output-dir", default="results/visualizations/navigation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries = [entry for entry in entries if int(entry["maze_size"]) == args.maze_size]
    if not entries:
        raise ValueError(f"no manifest entries for maze_size={args.maze_size}")

    head_data = torch.load(Path(args.head_ckpt), map_location=device, weights_only=False)
    model = load_model(Path(args.model_ckpt), device, head_data)
    head = load_metric_head(head_data, args.head_type, device)
    out_dir = Path(args.output_dir)
    wanted = ["success", "failure"] if args.case == "both" else [args.case]
    found = {key: 0 for key in wanted}
    analyses: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        if all(found[key] >= args.num_gifs for key in wanted):
            break
        env = create_env(entry)
        start, goal, opt = manifest_task(entry, env)
        result = run_metric_trajectory(
            model, head, env, start, goal, int(entry["maze_size"]), args.method, device
        )
        label = "success" if result["success"] else "failure"
        if label not in wanted or found[label] >= args.num_gifs:
            continue
        found[label] += 1
        stem = f"{args.head_type}_{args.method}_sz{args.maze_size}_{label}_{found[label]:02d}_idx{idx:03d}"
        gif_path = out_dir / f"{stem}.gif"
        save_gif(env, result, start, goal, gif_path, args.fps)

        analysis = {
            "gif": str(gif_path),
            "manifest_index_within_size": idx,
            "maze_size": args.maze_size,
            "topology_seed": entry.get("topology_seed"),
            "start_cell": start,
            "goal_cell": goal,
            "optimal_length": opt,
            "success": result["success"],
            "path_length": result["path_length"],
            "initial_bfs": result["initial_bfs"],
            "final_bfs": result["final_bfs"],
            "min_bfs": result["min_bfs"],
            **analyze_trajectory(result),
        }
        analyses.append(analysis)
        print(f"Saved {gif_path} | {analysis['reason']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.head_type}_{args.method}_sz{args.maze_size}_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(analyses, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
