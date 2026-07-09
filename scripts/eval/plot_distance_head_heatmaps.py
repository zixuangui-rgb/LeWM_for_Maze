#!/usr/bin/env python3
"""Plot DistanceHead distance fields against BFS ground truth.

For each selected manifest task, this script renders:
  - maze layout with start/goal markers
  - true BFS distance to goal for every walkable cell
  - DistanceHead predicted distance to goal
  - DistanceHead error: prediction - BFS
  - latent L2 distance to goal, min-max normalized over walkable cells
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.planning import _bfs_shortest_path
from scripts.train.train_dim256 import Unisize256


def create_env(entry: dict[str, Any]) -> ProcgenMazeEnv:
    size = int(entry["maze_size"])
    return ProcgenMazeEnv(
        ProcgenMazeConfig(
            height=size,
            width=size,
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


def observe_state(env: ProcgenMazeEnv, state: int) -> np.ndarray:
    obs, _ = env._observe_with_noise(np.array([state]))
    return obs[0]


def load_model(model_ckpt: Path, device: torch.device) -> Unisize256:
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config")
    if model_config is None and "lewm_state_dict" in ckpt:
        lewm_path = ckpt.get("config", {}).get("lewm_checkpoint")
        if lewm_path is None:
            raise ValueError(f"{model_ckpt} does not contain model_config or lewm_checkpoint")
        base_ckpt = torch.load(lewm_path, map_location=device, weights_only=False)
        model_config = base_ckpt["model_config"]
    model = Unisize256(model_config, max_size=31).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("lewm_state_dict"))
    if state is None:
        raise ValueError(f"{model_ckpt} does not contain model weights")
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_head(head_ckpt: Path, device: torch.device) -> DistanceHead:
    data = torch.load(head_ckpt, map_location=device, weights_only=False)
    cfg = data.get("config", {})
    head = DistanceHead(
        latent_dim=int(cfg.get("latent_dim", 256)),
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        input_mode=cfg.get("input_mode", "concat"),
    ).to(device)
    head.load_state_dict(data["head_state_dict"])
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def encode_batch(
    model: Unisize256,
    observations: list[np.ndarray],
    maze_size: int,
    device: torch.device,
) -> torch.Tensor:
    obs = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    obs = obs.unsqueeze(1)
    with torch.no_grad():
        encoded = model.encoder(obs, maze_size)
        embedding, _ = model.embedding_projector(encoded)
    return embedding.squeeze(1)


def task_from_entry(entry: dict[str, Any], env: ProcgenMazeEnv) -> tuple[int, int]:
    if "start_cell" in entry and "goal_cell" in entry:
        return int(entry["start_cell"]), int(entry["goal_cell"])
    walkable = np.flatnonzero((~env._maze_mask).reshape(-1))
    return int(walkable[0]), int(walkable[-1])


def distance_fields(
    model: Unisize256,
    head: DistanceHead,
    entry: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    env = create_env(entry)
    size = int(entry["maze_size"])
    start, goal = task_from_entry(entry, env)
    walkable = np.flatnonzero((~env._maze_mask).reshape(-1)).tolist()

    observations = [observe_state(env, cell) for cell in walkable]
    latents = encode_batch(model, observations, size, device)
    goal_index = walkable.index(goal)
    goal_latent = latents[goal_index : goal_index + 1].expand(latents.shape[0], -1)

    with torch.no_grad():
        pred = head(latents, goal_latent).detach().cpu().numpy()
        l2 = F.mse_loss(latents, goal_latent, reduction="none").sum(dim=-1).detach().cpu().numpy()

    bfs = np.full(len(walkable), np.nan, dtype=np.float32)
    for idx, cell in enumerate(walkable):
        dist = _bfs_shortest_path(env._maze_mask, int(cell), goal, env.config.width)
        if dist is not None:
            bfs[idx] = float(dist)

    true_grid = np.full((size, size), np.nan, dtype=np.float32)
    pred_grid = np.full((size, size), np.nan, dtype=np.float32)
    err_grid = np.full((size, size), np.nan, dtype=np.float32)
    l2_grid = np.full((size, size), np.nan, dtype=np.float32)
    for idx, cell in enumerate(walkable):
        row, col = divmod(cell, size)
        true_grid[row, col] = bfs[idx]
        pred_grid[row, col] = pred[idx]
        err_grid[row, col] = pred[idx] - bfs[idx]
        l2_grid[row, col] = l2[idx]

    l2_min = np.nanmin(l2_grid)
    l2_max = np.nanmax(l2_grid)
    l2_norm = (l2_grid - l2_min) / max(float(l2_max - l2_min), 1e-8)

    valid = np.isfinite(bfs)
    dh_spear = float(spearmanr(pred[valid], bfs[valid]).statistic)
    l2_spear = float(spearmanr(l2[valid], bfs[valid]).statistic)
    dh_pear = float(pearsonr(pred[valid], bfs[valid]).statistic)
    l2_pear = float(pearsonr(l2[valid], bfs[valid]).statistic)
    mae = float(np.mean(np.abs(pred[valid] - bfs[valid])))

    maze = np.where(env._maze_mask, 1.0, 0.0)
    return {
        "entry": entry,
        "size": size,
        "start": start,
        "goal": goal,
        "maze": maze,
        "true": true_grid,
        "pred": pred_grid,
        "error": err_grid,
        "l2_norm": l2_norm,
        "metrics": {
            "dh_spearman": dh_spear,
            "l2_spearman": l2_spear,
            "dh_pearson": dh_pear,
            "l2_pearson": l2_pear,
            "dh_mae": mae,
        },
    }


def plot_fields(fields: dict[str, Any], output: Path) -> None:
    size = int(fields["size"])
    start = int(fields["start"])
    goal = int(fields["goal"])
    sy, sx = divmod(start, size)
    gy, gx = divmod(goal, size)
    metrics = fields["metrics"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), constrained_layout=True)
    titles = [
        "Maze",
        "True BFS",
        "DistanceHead",
        "DH - BFS",
        "Latent L2 norm",
    ]
    arrays = [
        fields["maze"],
        fields["true"],
        fields["pred"],
        fields["error"],
        fields["l2_norm"],
    ]
    cmaps = ["gray_r", "viridis", "viridis", "coolwarm", "magma"]

    for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
        image = ax.imshow(arr, cmap=cmap, origin="upper")
        ax.scatter([sx], [sy], c="lime", s=42, marker="o", edgecolors="black", linewidths=0.8)
        ax.scatter([gx], [gy], c="red", s=58, marker="*", edgecolors="black", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if title != "Maze":
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"size={size} topo={fields['entry']['topology_seed']} "
        f"DH rho={metrics['dh_spearman']:.3f}, L2 rho={metrics['l2_spearman']:.3f}, "
        f"DH MAE={metrics['dh_mae']:.2f}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def pick_entries(manifest: Path, sizes: list[int], index: int) -> list[dict[str, Any]]:
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    picked = []
    for size in sizes:
        candidates = [entry for entry in entries if int(entry["maze_size"]) == size]
        if not candidates:
            raise ValueError(f"no entry for size {size} in {manifest}")
        picked.append(candidates[min(index, len(candidates) - 1)])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--head-ckpt", default="checkpoints/metric_heads/distance_head_set_b:_multi-size.pt")
    parser.add_argument("--sizes", default="9,11,21,23,25")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="results/heatmaps/distance_head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model_ckpt), device)
    head = load_head(Path(args.head_ckpt), device)
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    entries = pick_entries(Path(args.manifest), sizes, args.index)

    summaries = []
    for entry in entries:
        fields = distance_fields(model, head, entry, device)
        size = int(fields["size"])
        output = Path(args.output_dir) / f"dh_heatmap_sz{size}_topo{entry['topology_seed']}.png"
        plot_fields(fields, output)
        summary = {
            "output": str(output),
            "maze_size": size,
            "topology_seed": int(entry["topology_seed"]),
            "start": int(fields["start"]),
            "goal": int(fields["goal"]),
            **fields["metrics"],
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = Path(args.output_dir) / "heatmap_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
