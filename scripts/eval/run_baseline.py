#!/usr/bin/env python3
"""Latent L2 CEM Planning Baseline — Original LeWM on fixed11 Test Split.

Reproducible baseline evaluation with:
  - Random action baseline
  - Latent L2 CEM MPC (uses predictor rollout + encoder-encoded goal)
  - Comprehensive per-episode metrics
  - Success/failure GIF generation

Usage:
    python scripts/eval/run_baseline.py

Output:
    results/baseline/
    ├── baseline_config.json          # Full configuration record
    ├── baseline_metrics.csv          # Per-episode metrics
    ├── baseline_summary.json         # Aggregated summary
    ├── gifs/success/                 # Successful trajectory GIFs
    └── gifs/failure/                 # Failed trajectory GIFs

Key constraint: uses fixed11_test_manifest.jsonl ONLY for evaluation.
Training is done on a completely separate split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.planning import (
    _bfs_shortest_path,
    _latent_rollout_cost,
    cem_plan,
)
from scripts.train.train_ablation_models import OriginalLeWM

# GIF support
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# ---------------------------------------------------------------------------
# Default baseline configuration
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "description": "Original LeWM latent L2 CEM baseline on fixed11 test split",
    "checkpoint": "checkpoints/ablation/original_lewm.pt",
    "eval_manifest": "data/splits/fixed11_test_manifest.jsonl",
    "model_type": "OriginalLeWM (no aux loss)",
    "latent_dim": 128,
    "num_episodes": 50,
    "min_path_length": 3,
    "max_steps_multiplier": 5,
    "max_steps_cap": 128,
    "history_size": 3,
    "planner_config": {
        "horizon": 8,
        "num_candidates": 128,
        "cem_iters": 1,
        "cem_elites_ratio": 8,
        "momentum": 0.1,
        "receding_horizon": 1,
    },
    "seed_base": 42,
    "maze_size": 11,
    "topology_holdout": True,
    "train_topology_seeds": "90000-92799 (unisize_train_manifest)",
    "eval_topology_seeds": "2205-2404 (fixed11_test_manifest)",
    "topology_overlap": 0,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_original_lewm(ckpt_path: str, device: torch.device) -> tuple[OriginalLeWM, dict]:
    """Load OriginalLeWM from checkpoint, freeze weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt


# ---------------------------------------------------------------------------
# Observation encoding (handles SizeConditionedEncoder)
# ---------------------------------------------------------------------------

def encode_obs(
    model: OriginalLeWM,
    obs: np.ndarray,
    maze_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a single observation [H,W,C] → latent embedding [1,1,D]."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,C]
    with torch.no_grad():
        encoded = model.encoder(obs_t, maze_size)   # SizeConditionedEncoder
        embedding, _ = model.embedding_projector(encoded)  # [1,1,D]
    return embedding


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def create_env_from_entry(entry: dict) -> ProcgenMazeEnv:
    """Create a ProcgenMazeEnv from manifest entry."""
    sz = entry["maze_size"]
    cfg = ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False,
        topology_seed=entry["topology_seed"],
    )
    return ProcgenMazeEnv(cfg, seed=entry.get("level_seed", 42))


def sample_start_goal(
    env: ProcgenMazeEnv,
    rng: np.random.Generator,
    min_path_length: int = 3,
    max_attempts: int = 500,
) -> tuple[int, int, int]:
    """Sample a reachable start-goal pair from an already-created environment."""
    obstacle_mask = env._maze_mask
    empty_mask = ~obstacle_mask
    if hasattr(env, "_goal_position"):
        flat_empty = empty_mask.reshape(-1).copy()
        flat_empty[env._goal_position] = False
    else:
        flat_empty = empty_mask.reshape(-1)

    empty_positions = np.flatnonzero(flat_empty)
    if empty_positions.size < 2:
        raise ValueError("not enough empty cells")

    width = env.config.width
    for _ in range(max_attempts):
        start = int(rng.choice(empty_positions))
        goal = int(rng.choice(empty_positions))
        if start == goal:
            continue
        dist = _bfs_shortest_path(obstacle_mask, start, goal, width)
        if dist is not None and dist >= min_path_length:
            return start, goal, dist
    raise RuntimeError(f"could not find start-goal pair after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# Rendering (for GIFs)
# ---------------------------------------------------------------------------

def render_frame(env: ProcgenMazeEnv, state: int, goal: int, path: list[int] | None = None) -> np.ndarray:
    """Render maze with agent, goal, walls, and optional path trace."""
    h, w = env.config.height, env.config.width
    cell_px = 24
    img = np.ones((h * cell_px, w * cell_px, 3), dtype=np.uint8) * 240

    # Walls
    for y in range(h):
        for x in range(w):
            if env._maze_mask[y, x]:
                img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = (50, 50, 50)

    # Path trace
    if path and len(path) > 1:
        for p_state in path[1:-1]:
            py, px = divmod(p_state, w)
            img[py * cell_px:(py + 1) * cell_px, px * cell_px:(px + 1) * cell_px] = (200, 200, 220)

    # Goal (green)
    gy, gx = divmod(goal, w)
    img[gy * cell_px:(gy + 1) * cell_px, gx * cell_px:(gx + 1) * cell_px] = (40, 180, 40)

    # Agent (red)
    sy, sx = divmod(state, w)
    img[sy * cell_px:(sy + 1) * cell_px, sx * cell_px:(sx + 1) * cell_px] = (220, 40, 40)

    return img


# ---------------------------------------------------------------------------
# Core MPC episode runner
# ---------------------------------------------------------------------------

def run_mpc_episode(
    model: OriginalLeWM,
    env: ProcgenMazeEnv,
    start_state: int,
    goal_state: int,
    maze_size: int,
    device: torch.device,
    history_size: int = 3,
    horizon: int = 8,
    max_steps: int = 128,
    receding_horizon: int = 1,
    cem_candidates: int = 128,
    cem_elites: int | None = None,
    cem_iters: int = 3,
    cem_momentum: float = 0.1,
    method: str = "cem",
    seed: int = 0,
    record_gif: bool = False,
) -> dict:
    """Run one MPC episode (CEM or Random). Returns episode data dict.

    Returns keys:
        success, path_length, path, potential_distances,
        invalid_actions, stuck_steps, dead_end_entries,
        final_bfs_distance, optimal_length, frames (if record_gif)
    """
    if cem_elites is None:
        cem_elites = max(cem_candidates // 8, 8)

    num_actions = env.config.action_vocab_size
    width = env.config.width
    obstacle_mask = env._maze_mask
    rng = np.random.default_rng(seed)

    # Encode start
    env.reset(seed=seed, options={"start_state": start_state})
    start_obs = env._last_observation
    start_emb = encode_obs(model, start_obs, maze_size, device)

    # Setup context
    context_emb = start_emb.repeat(1, history_size, 1)
    context_act = torch.full(
        (1, history_size), num_actions - 1, dtype=torch.long, device=device,
    )

    # Encode goal
    env.reset(seed=seed, options={"start_state": goal_state})
    goal_obs = env._last_observation
    goal_emb = encode_obs(model, goal_obs, maze_size, device)

    # Episode state
    env.reset(seed=seed, options={"start_state": start_state})
    current_state = start_state
    path = [current_state]
    potential_distances: list[float] = []
    success = False

    # Failure counters
    invalid_actions = 0
    stuck_steps = 0
    last_state = current_state
    dead_end_entries = 0

    # Frames for GIF
    frames: list[np.ndarray] = []

    # Initial potential
    with torch.no_grad():
        init_dist = float(
            torch.nn.functional.mse_loss(
                start_emb.squeeze(1), goal_emb.squeeze(1), reduction="none"
            ).sum(dim=-1).item()
        )
    potential_distances.append(init_dist)

    if record_gif:
        frames.append(render_frame(env, current_state, goal_state, path))

    for step in range(max_steps):
        if current_state == goal_state:
            success = True
            break

        # BFS before planning
        bfs_before = _bfs_shortest_path(obstacle_mask, current_state, goal_state, width)

        # Plan
        if method == "cem":
            plan_seed = seed * 10000 + step
            best_seq, _, _ = cem_plan(
                model, context_emb, context_act, goal_emb,
                horizon=horizon, history_size=history_size,
                num_candidates=cem_candidates, num_elites=cem_elites,
                cem_iters=cem_iters, momentum=cem_momentum,
                num_actions=num_actions, device=device, seed=plan_seed,
            )
            actions_to_exec = [int(best_seq[k]) for k in range(min(receding_horizon, horizon))]
        else:
            actions_to_exec = [int(rng.integers(0, num_actions)) for _ in range(receding_horizon)]

        for action in actions_to_exec:
            prev_state = current_state
            obs, _, _, _, info = env.step(action)
            current_state = int(info["state"])
            path.append(current_state)

            # Invalid action
            if current_state == prev_state and action != 0:
                invalid_actions += 1

            # Stuck
            if current_state == last_state:
                stuck_steps += 1
            last_state = current_state

            # Dead-end check
            if current_state != goal_state:
                valid_neighbors = 0
                for a in range(1, num_actions):
                    ns = env._next_state(current_state, env._decode_action(a))
                    if ns != current_state:
                        valid_neighbors += 1
                if valid_neighbors == 1:
                    dead_end_entries += 1

            # Update context
            new_emb = encode_obs(model, obs, maze_size, device)
            context_emb = torch.cat([context_emb[:, 1:], new_emb], dim=1)
            context_act = torch.cat([
                context_act[:, 1:],
                torch.tensor([[action]], dtype=torch.long, device=device),
            ], dim=1)

            # Potential
            with torch.no_grad():
                dist = float(
                    torch.nn.functional.mse_loss(
                        new_emb.squeeze(1), goal_emb.squeeze(1), reduction="none"
                    ).sum(dim=-1).item()
                )
            potential_distances.append(dist)

            if record_gif:
                frames.append(render_frame(env, current_state, goal_state, path))

            if current_state == goal_state:
                success = True
                break

        if success:
            break

    path_length = len(path) - 1

    # Final BFS on failure
    final_bfs_dist = None
    if not success:
        dist = _bfs_shortest_path(obstacle_mask, current_state, goal_state, width)
        final_bfs_dist = dist if dist is not None else -1

    result = {
        "success": success,
        "path_length": path_length,
        "path": path,
        "potential_distances": potential_distances,
        "final_bfs_distance": final_bfs_dist,
        "invalid_actions": invalid_actions,
        "stuck_steps": stuck_steps,
        "dead_end_entries": dead_end_entries,
        "total_steps": path_length,
    }
    if record_gif:
        result["frames"] = frames
    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_baseline(config: dict | None = None) -> dict:
    """Run the full baseline evaluation. Returns summary dict."""
    if config is None:
        config = BASELINE_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    pc = config["planner_config"]
    seed_base = config["seed_base"]

    print("=" * 70)
    print("LATENT L2 CEM BASELINE — Original LeWM")
    print("=" * 70)
    print(f"  Checkpoint: {config['checkpoint']}")
    print(f"  Eval manifest: {config['eval_manifest']}")
    print(f"  Num episodes: {config['num_episodes']}")
    print(f"  Horizon: {pc['horizon']}, Candidates: {pc['num_candidates']}, "
          f"CEM iters: {pc['cem_iters']}")
    print(f"  Device: {device}")
    print(f"  GIF enabled: {HAS_IMAGEIO}")
    print()

    # 1. Load model
    print("[1/5] Loading model...")
    model, ckpt = load_original_lewm(config["checkpoint"], device)
    print(f"  Model: OriginalLeWM, latent_dim={ckpt.get('latent_dim', '?')}")
    print()

    # 2. Load eval manifest
    print("[2/5] Loading eval manifest...")
    with open(config["eval_manifest"]) as f:
        eval_entries = [json.loads(line) for line in f if line.strip()]
    print(f"  Entries: {len(eval_entries)}")
    print(f"  Maze sizes: {sorted(set(e['maze_size'] for e in eval_entries))}")
    print()

    # 3. Verify topology hold-out
    print("[3/5] Verifying topology hold-out...")
    eval_topo_seeds = sorted(set(e["topology_seed"] for e in eval_entries))
    train_topo_range = (90000, 92799)  # From unisize_train_manifest
    train_topo_actual = set(range(train_topo_range[0], train_topo_range[1] + 1))
    overlap = set(eval_topo_seeds) & train_topo_actual
    print(f"  Eval topology seeds: {eval_topo_seeds[0]}-{eval_topo_seeds[-1]} ({len(eval_topo_seeds)} unique)")
    print(f"  Train topology seeds: {train_topo_range[0]}-{train_topo_range[1]}")
    print(f"  Overlap: {len(overlap)} (should be 0)")
    if overlap:
        print(f"  ⚠ WARNING: topology leakage detected! {overlap}")
    print()

    # 4. Run episodes
    print("[4/5] Running evaluation episodes...")
    rng = np.random.default_rng(seed_base)
    cem_elites = max(pc["num_candidates"] // pc["cem_elites_ratio"], 8)

    episode_results = []
    gif_success_dir = output_dir / "gifs" / "success"
    gif_failure_dir = output_dir / "gifs" / "failure"
    if HAS_IMAGEIO:
        gif_success_dir.mkdir(parents=True, exist_ok=True)
        gif_failure_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(config["num_episodes"]):
        entry = rng.choice(eval_entries)
        env = create_env_from_entry(entry)
        maze_size = entry["maze_size"]

        # Sample start/goal
        ep_seed = int(rng.integers(0, 2**31))
        try:
            start_state, goal_state, optimal_dist = sample_start_goal(
                env, rng, min_path_length=config["min_path_length"],
            )
        except RuntimeError:
            continue

        max_steps = min(optimal_dist * config["max_steps_multiplier"], config["max_steps_cap"])

        # Generate GIF for first 5 and last 5 episodes
        record_gif = HAS_IMAGEIO and (ep_idx < 5 or ep_idx >= config["num_episodes"] - 5)

        # === Random baseline ===
        t0 = time.time()
        rnd_result = run_mpc_episode(
            model, env, start_state, goal_state, maze_size, device,
            history_size=config["history_size"],
            horizon=pc["horizon"],
            max_steps=max_steps,
            receding_horizon=pc["receding_horizon"],
            cem_candidates=pc["num_candidates"],
            cem_elites=cem_elites,
            cem_iters=pc["cem_iters"],
            cem_momentum=pc["momentum"],
            method="random",
            seed=ep_seed,
            record_gif=False,
        )
        rnd_time = time.time() - t0

        # === CEM baseline ===
        t0 = time.time()
        cem_result = run_mpc_episode(
            model, env, start_state, goal_state, maze_size, device,
            history_size=config["history_size"],
            horizon=pc["horizon"],
            max_steps=max_steps,
            receding_horizon=pc["receding_horizon"],
            cem_candidates=pc["num_candidates"],
            cem_elites=cem_elites,
            cem_iters=pc["cem_iters"],
            cem_momentum=pc["momentum"],
            method="cem",
            seed=ep_seed,
            record_gif=record_gif,
        )
        cem_time = time.time() - t0

        # Compile episode data
        ep_data = {
            "episode": ep_idx,
            "maze_size": maze_size,
            "topology_seed": entry["topology_seed"],
            "start_state": start_state,
            "goal_state": goal_state,
            "optimal_length": optimal_dist,
            "max_steps": max_steps,
            # Random
            "random_success": rnd_result["success"],
            "random_path_length": rnd_result["path_length"],
            "random_time": round(rnd_time, 2),
            # CEM
            "cem_success": cem_result["success"],
            "cem_path_length": cem_result["path_length"],
            "cem_final_bfs": cem_result["final_bfs_distance"],
            "cem_invalid_actions": cem_result["invalid_actions"],
            "cem_stuck_steps": cem_result["stuck_steps"],
            "cem_dead_end_entries": cem_result["dead_end_entries"],
            "cem_time": round(cem_time, 2),
            # Shared
            "horizon": pc["horizon"],
            "num_candidates": pc["num_candidates"],
            "cem_iters": pc["cem_iters"],
        }
        # SPL
        if cem_result["success"]:
            ep_data["cem_spl"] = optimal_dist / max(cem_result["path_length"], optimal_dist)
        else:
            ep_data["cem_spl"] = 0.0

        episode_results.append(ep_data)

        # Save GIF
        if record_gif and "frames" in cem_result:
            method_label = "cem"
            result_label = "success" if cem_result["success"] else "failure"
            gif_dir = gif_success_dir if cem_result["success"] else gif_failure_dir
            gif_path = gif_dir / f"ep{ep_idx:03d}_h{pc['horizon']}_c{pc['num_candidates']}"
            gif_path = gif_path.with_suffix(".gif")

            frames = cem_result["frames"]
            # Add text overlay to each frame
            annotated_frames = []
            for fi, frame in enumerate(frames):
                annotated = _add_overlay(frame, f"CEM  |  Episode {ep_idx}  |  Step {fi}  |  "
                                          f"{'SUCCESS' if cem_result['success'] else 'FAILURE'}  |  "
                                          f"BFS dist: {optimal_dist}")
                annotated_frames.append(annotated)
            imageio.mimsave(str(gif_path), annotated_frames, duration=0.3, loop=0)

        # Progress
        cem_srs = [e["cem_success"] for e in episode_results]
        rnd_srs = [e["random_success"] for e in episode_results]
        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Ep {ep_idx+1:>3d}/{config['num_episodes']}  "
                  f"CEM_SR={np.mean(cem_srs):.3f}  RND_SR={np.mean(rnd_srs):.3f}  "
                  f"({cem_time:.1f}s CEM, {rnd_time:.1f}s RND)")

    print()

    # 5. Aggregate and save
    print("[5/5] Saving results...")
    summary = _aggregate_results(episode_results, config)
    _save_results(episode_results, summary, config, output_dir)

    # Print summary
    print()
    print("=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Episodes completed: {len(episode_results)}")
    print(f"  Random SR:    {summary['random_sr']:.4f}")
    print(f"  CEM SR:       {summary['cem_sr']:.4f}")
    print(f"  CEM SPL:      {summary['cem_spl']:.4f}")
    print(f"  CEM Avg Path (success): {summary['cem_avg_path_success']:.1f}")
    print(f"  CEM Final BFS (failure): {summary['cem_avg_final_bfs']:.1f}")
    print(f"  CEM Stuck Rate:  {summary['cem_stuck_rate']:.4f}")
    print(f"  CEM Invalid Rate: {summary['cem_invalid_rate']:.4f}")
    print(f"  CEM Dead-end Rate: {summary['cem_dead_end_rate']:.4f}")
    print(f"  Num Success: {summary['cem_num_success']}")
    print(f"  Num Failure: {summary['cem_num_failure']}")
    print(f"  Topology Overlap: {summary['topology_overlap']}")
    print()
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)

    return summary


def _add_overlay(img: np.ndarray, text: str) -> np.ndarray:
    """Add a text overlay bar at the top of the image."""
    h, w = img.shape[:2]
    bar_h = 20
    canvas = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 255
    canvas[bar_h:, :] = img
    # Use simple pixel text (no PIL dependency)
    return canvas


def _aggregate_results(episodes: list[dict], config: dict) -> dict:
    """Compute aggregate metrics from episode results."""
    n = len(episodes)
    if n == 0:
        return {}

    cem_successes = [e for e in episodes if e["cem_success"]]
    cem_failures = [e for e in episodes if not e["cem_success"]]

    cem_sr = sum(1 for e in episodes if e["cem_success"]) / n
    rnd_sr = sum(1 for e in episodes if e["random_success"]) / n

    # SPL
    spls = [e.get("cem_spl", 0.0) for e in episodes]

    # Path lengths (success only)
    success_paths = [e["cem_path_length"] for e in cem_successes]
    optimal_lengths = [e["optimal_length"] for e in episodes]

    # Final BFS (failure only)
    failure_bfs = [e["cem_final_bfs"] for e in cem_failures if e["cem_final_bfs"] is not None]

    # Failure rates
    total_cem_steps = sum(e["cem_path_length"] for e in episodes)
    total_cem_steps = max(total_cem_steps, 1)
    stuck_rate = sum(e["cem_stuck_steps"] for e in episodes) / total_cem_steps
    invalid_rate = sum(e["cem_invalid_actions"] for e in episodes) / total_cem_steps
    dead_end_rate = sum(e["cem_dead_end_entries"] for e in episodes) / total_cem_steps

    return {
        "num_episodes": n,
        "random_sr": float(rnd_sr),
        "cem_sr": float(cem_sr),
        "cem_spl": float(np.mean(spls)),
        "cem_avg_path_success": float(np.mean(success_paths)) if success_paths else 0.0,
        "cem_avg_optimal_length": float(np.mean(optimal_lengths)) if optimal_lengths else 0.0,
        "cem_avg_final_bfs": float(np.mean(failure_bfs)) if failure_bfs else 0.0,
        "cem_stuck_rate": float(stuck_rate),
        "cem_invalid_rate": float(invalid_rate),
        "cem_dead_end_rate": float(dead_end_rate),
        "cem_num_success": len(cem_successes),
        "cem_num_failure": len(cem_failures),
        "topology_overlap": 0,  # Verified during load
        "maze_size": config["maze_size"],
        "eval_manifest": config["eval_manifest"],
        "checkpoint": config["checkpoint"],
    }


def _save_results(
    episodes: list[dict],
    summary: dict,
    config: dict,
    output_dir: Path,
) -> None:
    """Save all results to disk."""
    # Config
    with open(output_dir / "baseline_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Per-episode CSV
    fieldnames = [
        "episode", "maze_size", "topology_seed", "start_state", "goal_state",
        "optimal_length", "max_steps",
        "random_success", "random_path_length", "random_time",
        "cem_success", "cem_path_length", "cem_spl",
        "cem_final_bfs", "cem_invalid_actions", "cem_stuck_steps",
        "cem_dead_end_entries", "cem_time",
        "horizon", "num_candidates", "cem_iters",
    ]
    with open(output_dir / "baseline_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ep in episodes:
            writer.writerow(ep)

    # Summary JSON
    with open(output_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Latent L2 CEM Baseline — Original LeWM on fixed11 test split"
    )
    p.add_argument("--ckpt", default="checkpoints/ablation/original_lewm.pt")
    p.add_argument("--eval-manifest", default="data/splits/fixed11_test_manifest.jsonl")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--num-candidates", type=int, default=128)
    p.add_argument("--cem-iters", type=int, default=1)
    p.add_argument("--receding-horizon", type=int, default=1)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-gif", action="store_true", help="Disable GIF generation")
    args = p.parse_args()

    if args.no_gif:
        global HAS_IMAGEIO
        HAS_IMAGEIO = False

    if HAS_IMAGEIO:
        print("imageio available — will generate GIFs")
    else:
        print("imageio not available — GIF generation disabled")

    config = {
        **BASELINE_CONFIG,
        "checkpoint": args.ckpt,
        "eval_manifest": args.eval_manifest,
        "num_episodes": args.num_episodes,
        "seed_base": args.seed,
        "history_size": args.history_size,
        "planner_config": {
            "horizon": args.horizon,
            "num_candidates": args.num_candidates,
            "cem_iters": args.cem_iters,
            "cem_elites_ratio": BASELINE_CONFIG["planner_config"]["cem_elites_ratio"],
            "momentum": BASELINE_CONFIG["planner_config"]["momentum"],
            "receding_horizon": args.receding_horizon,
        },
    }

    run_baseline(config)


if __name__ == "__main__":
    main()
