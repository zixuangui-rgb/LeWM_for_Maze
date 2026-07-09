#!/usr/bin/env python3
"""Evaluate Set B DistanceHead navigation with corrected action selection.

Set B uses unisize_train_manifest for training sizes 9-21 and
unisize_eval_manifest for held-out topologies on sizes 9-25. This script
evaluates the fixed tasks from the eval manifest, grouped by maze size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.planning import _bfs_shortest_path, cem_plan
from scripts.train.train_dim256 import Unisize256


HISTORY_SIZE = 3
MAX_STEPS = 128


def create_env(entry: dict[str, Any]) -> ProcgenMazeEnv:
    sz = int(entry["maze_size"])
    return ProcgenMazeEnv(
        ProcgenMazeConfig(
            height=sz,
            width=sz,
            observation_channels=5,
            p_noise=0.0,
            p_noop=0.0,
            p_action_turn=0.0,
            p_action_stay=0.0,
            resample_maze_per_sequence=False,
            topology_seed=int(entry["topology_seed"]),
        ),
        seed=int(entry.get("env_seed", 42)),
    )


def moving_actions(
    env: ProcgenMazeEnv, state: int, previous_state: int | None = None
) -> list[int]:
    """Return actions that move the agent; avoid immediate backtracking when possible."""
    actions: list[int] = []
    non_backtracking: list[int] = []
    for action in range(1, env.config.action_vocab_size):
        next_state = int(env._next_state(state, env._decode_action(action)))
        if next_state != state:
            actions.append(action)
            if previous_state is None or next_state != previous_state:
                non_backtracking.append(action)
    return non_backtracking or actions


def observe_state(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    obs, _ = env._observe_with_noise(np.array([state]))
    return obs[0]


def set_agent_state(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    """Place the agent at an arbitrary non-wall state without reset's goal check."""
    if env._maze_mask.reshape(-1)[state]:
        raise ValueError("state must be an empty cell")
    env._state = state
    env._elapsed_steps = 0
    obs = observe_state(env, state)
    env._last_observation = obs
    env._last_noise_mask = np.zeros_like(env._maze_mask, dtype=bool)
    return obs


def encode_obs(
    model: Unisize256,
    obs: np.ndarray,
    maze_size: int,
    device: torch.device,
) -> torch.Tensor:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encoder(obs_t, maze_size)
        embedding, _ = model.embedding_projector(encoded)
    return embedding


def load_model_and_head(
    model_ckpt: Path,
    head_ckpt: Path,
    device: torch.device,
) -> tuple[Unisize256, DistanceHead]:
    head_data = torch.load(head_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in head_data:
        model = Unisize256(head_data["model_config"], max_size=31).to(device)
        model.load_state_dict(head_data["model_state_dict"], strict=True)
        model_source = str(head_ckpt)
    else:
        ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        model = Unisize256(ckpt["model_config"], max_size=31).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model_source = str(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    cfg = head_data.get("config", {})
    head = DistanceHead(
        latent_dim=int(cfg.get("latent_dim", 256)),
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        input_mode=cfg.get("input_mode", "concat"),
    ).to(device)
    head.load_state_dict(head_data["head_state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    print(f"Model source: {model_source}")
    return model, head


def manifest_task(entry: dict[str, Any], env: ProcgenMazeEnv) -> tuple[int, int, int]:
    start = int(entry["start_cell"])
    goal = int(entry["goal_cell"])
    opt = _bfs_shortest_path(env._maze_mask, start, goal, env.config.width)
    if opt is None:
        raise ValueError("manifest task start/goal are disconnected")
    return start, goal, int(opt)


def run_model_free_greedy(
    model: Unisize256,
    head: DistanceHead,
    env: ProcgenMazeEnv,
    start: int,
    goal: int,
    maze_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    del seed
    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        candidates = moving_actions(env, cur, previous)
        if not candidates:
            candidates = list(range(1, env.config.action_vocab_size))

        best_action = candidates[0]
        best_score = float("inf")
        for action in candidates:
            next_state = int(env._next_state(cur, env._decode_action(action)))
            obs = observe_state(env, next_state)
            next_emb = encode_obs(model, obs, maze_size, device)
            score = float(head(next_emb, goal_emb).item())
            if score < best_score:
                best_score = score
                best_action = action

        prev = cur
        _, _, _, _, info = env.step(best_action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and best_action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur

    final_bfs = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
    }


def run_predictor_greedy(
    model: Unisize256,
    head: DistanceHead,
    env: ProcgenMazeEnv,
    start: int,
    goal: int,
    maze_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    num_actions = env.config.action_vocab_size
    del seed
    start_obs = set_agent_state(env, start)
    start_emb = encode_obs(model, start_obs, maze_size, device)
    ctx_emb = start_emb.repeat(1, HISTORY_SIZE, 1)
    ctx_act = torch.full((1, HISTORY_SIZE), num_actions - 1, dtype=torch.long, device=device)

    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for _ in range(MAX_STEPS):
        if cur == goal:
            break
        ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
        ctx_act_rep = ctx_act[:, :-1].repeat(num_actions, 1)
        ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
        with torch.no_grad():
            pred_emb = model.predictor(ctx_emb_rep, ctx_act_rep)
            next_emb = pred_emb[:, -1, :]
            goal_rep = goal_emb.expand(num_actions, -1, -1).squeeze(1)
            scores = head(next_emb, goal_rep)

        valid_actions = moving_actions(env, cur, previous)
        if valid_actions:
            mask = torch.full_like(scores, float("inf"))
            mask[torch.tensor(valid_actions, dtype=torch.long, device=device)] = 0.0
            scores = scores + mask
        action = int(scores.argmin())

        prev = cur
        obs, _, _, _, info = env.step(action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur

        new_emb = encode_obs(model, obs, maze_size, device)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
        ctx_act = torch.cat(
            [ctx_act[:, 1:], torch.tensor([[action]], dtype=torch.long, device=device)],
            dim=1,
        )

    final_bfs = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
    }


def select_one_step_action(
    model: Unisize256,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    env: ProcgenMazeEnv,
    ctx_emb: torch.Tensor,
    ctx_act: torch.Tensor,
    goal_emb: torch.Tensor,
    cur: int,
    previous: int | None,
    device: torch.device,
) -> int:
    valid_actions = moving_actions(env, cur, previous)
    if not valid_actions:
        return 1
    num_actions = env.config.action_vocab_size
    ctx_emb_rep = ctx_emb.expand(num_actions, -1, -1)
    ctx_act_rep = ctx_act[:, :-1].repeat(num_actions, 1)
    ctx_act_rep[:, -1] = torch.arange(num_actions, device=device)
    with torch.no_grad():
        pred_emb = model.predictor(ctx_emb_rep, ctx_act_rep)
        next_emb = pred_emb[:, -1, :]
        goal_rep = goal_emb.expand(num_actions, -1, -1).squeeze(1)
        scores = score_fn(next_emb, goal_rep)
    mask = torch.full_like(scores, float("inf"))
    mask[torch.tensor(valid_actions, dtype=torch.long, device=device)] = 0.0
    return int((scores + mask).argmin())


def run_cem(
    model: Unisize256,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    env: ProcgenMazeEnv,
    start: int,
    goal: int,
    maze_size: int,
    device: torch.device,
    seed: int,
    horizon: int,
    num_candidates: int,
    cem_iters: int,
) -> dict[str, Any]:
    num_actions = env.config.action_vocab_size
    elites = max(num_candidates // 8, 8)
    start_obs = set_agent_state(env, start)
    start_emb = encode_obs(model, start_obs, maze_size, device)
    ctx_emb = start_emb.repeat(1, HISTORY_SIZE, 1)
    ctx_act = torch.full((1, HISTORY_SIZE), num_actions - 1, dtype=torch.long, device=device)

    goal_obs = observe_state(env, goal)
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    set_agent_state(env, start)
    cur = start
    previous: int | None = None
    path_length = 0
    stuck_steps = 0
    invalid_actions = 0
    last = cur

    for step in range(MAX_STEPS):
        if cur == goal:
            break
        best_seq, _, _ = cem_plan(
            model,
            ctx_emb,
            ctx_act,
            goal_emb,
            horizon=horizon,
            history_size=HISTORY_SIZE,
            num_candidates=num_candidates,
            num_elites=elites,
            cem_iters=cem_iters,
            momentum=0.1,
            num_actions=num_actions,
            device=device,
            seed=seed * 10000 + step,
            score_fn=score_fn,
            allowed_actions=np.arange(1, num_actions),
        )
        action = int(best_seq[0])
        if action not in moving_actions(env, cur, previous):
            action = select_one_step_action(
                model, score_fn, env, ctx_emb, ctx_act, goal_emb, cur, previous, device
            )

        prev = cur
        obs, _, _, _, info = env.step(action)
        cur = int(info["state"])
        previous = prev
        path_length += 1
        if cur == prev and action != 0:
            invalid_actions += 1
        if cur == last:
            stuck_steps += 1
        last = cur
        new_emb = encode_obs(model, obs, maze_size, device)
        ctx_emb = torch.cat([ctx_emb[:, 1:], new_emb], dim=1)
        ctx_act = torch.cat(
            [ctx_act[:, 1:], torch.tensor([[action]], dtype=torch.long, device=device)],
            dim=1,
        )

    final_bfs = _bfs_shortest_path(env._maze_mask, cur, goal, env.config.width)
    success = cur == goal
    return {
        "success": success,
        "path_length": path_length,
        "stuck_steps": stuck_steps,
        "invalid_actions": invalid_actions,
        "final_bfs_distance": 0 if success else (final_bfs if final_bfs is not None else -1),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    total_steps = max(sum(int(row["path_length"]) for row in rows), 1)
    return {
        "n": len(rows),
        "sr": float(len(successes) / len(rows)),
        "spl": float(np.mean([row["spl"] for row in rows])),
        "avg_path_success": float(np.mean([row["path_length"] for row in successes])) if successes else 0.0,
        "avg_final_bfs": float(np.mean([row["final_bfs_distance"] for row in failures])) if failures else 0.0,
        "stuck_rate": float(sum(row["stuck_steps"] for row in rows) / total_steps),
        "invalid_rate": float(sum(row["invalid_actions"] for row in rows) / total_steps),
        "num_success": len(successes),
        "num_failure": len(failures),
    }


def evaluate_method(
    name: str,
    entries: list[dict[str, Any]],
    model: Unisize256,
    head: DistanceHead,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    t0 = time.time()

    def dh_score(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return head(terminal, goal)

    def l2_score(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(terminal, goal, reduction="none").sum(dim=-1)

    def l2_dh_score(terminal: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return 0.5 * l2_score(terminal, goal) + 0.5 * dh_score(terminal, goal)

    for idx, entry in enumerate(entries):
        env = create_env(entry)
        start, goal, opt = manifest_task(entry, env)
        maze_size = int(entry["maze_size"])
        seed = args.seed * 10000 + idx
        if name == "model_free_greedy":
            row = run_model_free_greedy(model, head, env, start, goal, maze_size, device, seed)
        elif name == "predictor_greedy":
            row = run_predictor_greedy(model, head, env, start, goal, maze_size, device, seed)
        elif name == "cem_distance":
            row = run_cem(
                model,
                dh_score,
                env,
                start,
                goal,
                maze_size,
                device,
                seed,
                args.horizon,
                args.num_candidates,
                args.cem_iters,
            )
        elif name == "cem_l2":
            row = run_cem(
                model,
                l2_score,
                env,
                start,
                goal,
                maze_size,
                device,
                seed,
                args.horizon,
                args.num_candidates,
                args.cem_iters,
            )
        elif name == "cem_l2_distance":
            row = run_cem(
                model,
                l2_dh_score,
                env,
                start,
                goal,
                maze_size,
                device,
                seed,
                args.horizon,
                args.num_candidates,
                args.cem_iters,
            )
        else:
            raise ValueError(f"unknown method: {name}")

        row["op_len"] = opt
        row["maze_size"] = maze_size
        row["spl"] = opt / max(int(row["path_length"]), opt) if row["success"] else 0.0
        all_rows.append(row)
        by_size[maze_size].append(row)

        if (idx + 1) % args.progress_every == 0:
            current = summarize(all_rows)
            print(f"  [{name}] {idx + 1:>4d}/{len(entries)} SR={current['sr']:.4f}")

    result = summarize(all_rows)
    result["time"] = float(time.time() - t0)
    result["by_size"] = {str(size): summarize(rows) for size, rows in sorted(by_size.items())}
    return result


def filtered_entries(entries: list[dict[str, Any]], max_per_size: int, limit: int) -> list[dict[str, Any]]:
    if max_per_size > 0:
        counts: dict[int, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        for entry in entries:
            size = int(entry["maze_size"])
            if counts[size] >= max_per_size:
                continue
            counts[size] += 1
            selected.append(entry)
        entries = selected
    if limit > 0:
        entries = entries[:limit]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--head-ckpt", default="checkpoints/metric_heads/distance_head_set_b:_multi-size.pt")
    parser.add_argument("--output", default="results/set_b_multisize/distance_head_fixed_eval.json")
    parser.add_argument("--methods", default="model_free_greedy,predictor_greedy,cem_distance")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--cem-iters", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries = filtered_entries(entries, args.max_per_size, args.limit)
    sizes = sorted({int(entry["maze_size"]) for entry in entries})
    print(f"Entries: {len(entries)}, sizes={sizes}")
    print(f"Device: {device}")

    model, head = load_model_and_head(Path(args.model_ckpt), Path(args.head_ckpt), device)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    results: dict[str, Any] = {
        "manifest": args.manifest,
        "model_ckpt": args.model_ckpt,
        "head_ckpt": args.head_ckpt,
        "methods": {},
    }
    for method in methods:
        print(f"\n[{method}]")
        results["methods"][method] = evaluate_method(method, entries, model, head, device, args)
        r = results["methods"][method]
        print(
            f"  SR={r['sr']:.4f} SPL={r['spl']:.4f} "
            f"stuck={r['stuck_rate']:.4f} invalid={r['invalid_rate']:.4f} "
            f"S/F={r['num_success']}/{r['num_failure']} time={r['time']:.0f}s"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
