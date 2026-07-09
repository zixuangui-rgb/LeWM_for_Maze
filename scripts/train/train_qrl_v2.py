#!/usr/bin/env python3
"""Train a Set B QRL metric head on frozen or unfrozen LeWM latents.

QRL keeps the same planner interface as DistanceHead: lower Q(z, goal) means
closer to the goal. Compared with the pure regression DistanceHead baseline,
this trainer puts more weight on navigation-relevant order constraints:

- scale-aware BFS regression
- local valid-action ranking
- hard positive/negative contrastive ordering
- optional quasimetric triangle inequality

By default the LeWM backbone is frozen and only the QRL head is trained.
Pass --unfreeze-backbone to also train the encoder + embedding projector.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.metric_heads.qrl_head import QRLHead
from scripts.train.train_distance_head_v2 import (
    SetBMazeCache,
    Unisize256,
    load_backbone,
    sample_local_ranking_batch,
    sample_regression_batch,
    sample_triangle_batch,
)


@dataclass
class TrainConfig:
    model_ckpt: str = "checkpoints/unisize_dim256.pt"
    train_manifest: str = "data/splits/unisize_train_manifest.jsonl"
    val_manifest: str = "data/splits/unisize_eval_manifest.jsonl"
    output: str = "checkpoints/metric_heads/qrl_v2_setb.pt"
    latent_dim: int = 256
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    target_mode: str = "log_norm"
    steps: int = 30000
    batch_size: int = 512
    pairs_per_maze: int = 64
    local_batch_size: int = 512
    contrastive_batch_size: int = 512
    triangle_batch_size: int = 128
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-5
    regression_weight: float = 0.5
    ranking_weight: float = 2.0
    contrastive_weight: float = 1.0
    triangle_weight: float = 0.05
    ranking_margin: float = 0.03
    contrastive_margin: float = 0.05
    min_contrastive_gap: int = 2
    eval_every: int = 1000
    eval_batches: int = 8
    unfreeze_backbone: bool = False
    mazes_per_step: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class StepMazeCache:
    """Pre-encode a small set of mazes once per training step.

    When the backbone is unfrozen, encoding observations on-the-fly for every
    sampling call is prohibitively slow. This cache samples a fixed number of
    mazes at the start of each step, encodes their observations with gradients,
    and exposes the same interface as SetBMazeCache so existing samplers can
    be reused without modification.
    """

    def __init__(
        self,
        dataset: SetBMazeCache,
        model: Unisize256,
        rng: np.random.Generator,
        n_mazes: int = 4,
    ) -> None:
        self._indices = [dataset.sample_maze_idx(rng) for _ in range(n_mazes)]
        self._mazes = [dataset.get(idx) for idx in self._indices]
        self._latents = [dataset.encode_maze(idx, model, grad=True) for idx in self._indices]
        self.device = dataset.device

    def sample_maze_idx(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._indices)))

    def get(self, idx: int) -> dict[str, Any]:
        maze = dict(self._mazes[idx])
        maze["latents"] = self._latents[idx]
        return maze


def sample_contrastive_batch(
    dataset: SetBMazeCache | StepMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    neg_list: list[torch.Tensor] = []
    attempts = 0
    max_attempts = cfg.contrastive_batch_size * 128
    while len(anchor_list) < cfg.contrastive_batch_size and attempts < max_attempts:
        attempts += 1
        maze_idx = dataset.sample_maze_idx(rng)
        maze = dataset.get(maze_idx)
        latents = maze["latents"]
        if latents is None and model is not None:
            latents = dataset.encode_maze(maze_idx, model, grad=True)
        bfs = maze["bfs"]
        n = latents.shape[0]
        anchor = int(rng.integers(0, n))
        pos = int(rng.integers(0, n))
        if anchor == pos:
            continue
        pos_dist = int(bfs[anchor, pos])
        if pos_dist < 0:
            continue
        farther = np.flatnonzero(bfs[anchor] >= pos_dist + cfg.min_contrastive_gap)
        farther = farther[farther != anchor]
        if farther.size == 0:
            continue
        neg = int(rng.choice(farther))
        anchor_list.append(latents[anchor])
        pos_list.append(latents[pos])
        neg_list.append(latents[neg])
    if not anchor_list:
        raise RuntimeError("failed to sample contrastive batch")
    return (
        torch.stack(anchor_list).to(device),
        torch.stack(pos_list).to(device),
        torch.stack(neg_list).to(device),
    )


def evaluate_qrl(
    head: QRLHead,
    dataset: SetBMazeCache,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    model: Unisize256 | None = None,
) -> dict[str, float]:
    head.eval()
    reg_losses: list[float] = []
    rank_accs: list[float] = []
    contrast_accs: list[float] = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            z1, z2, target = sample_regression_batch(dataset, cfg, rng, device, model)
            pred = head(z1, z2)
            reg_losses.append(float(F.smooth_l1_loss(pred, target).item()))

            good, bad, goal = sample_local_ranking_batch(dataset, cfg, rng, device, model)
            good_score = head(good, goal)
            bad_score = head(bad, goal)
            rank_accs.append(float((good_score < bad_score).float().mean().item()))

            anchor, pos, neg = sample_contrastive_batch(dataset, cfg, rng, device, model)
            pos_score = head(anchor, pos)
            neg_score = head(anchor, neg)
            contrast_accs.append(float((pos_score < neg_score).float().mean().item()))
    return {
        "val_reg_loss": float(np.mean(reg_losses)),
        "val_rank_acc": float(np.mean(rank_accs)),
        "val_contrast_acc": float(np.mean(contrast_accs)),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", default="checkpoints/unisize_dim256.pt")
    parser.add_argument("--train-manifest", default="data/splits/unisize_train_manifest.jsonl")
    parser.add_argument("--val-manifest", default="data/splits/unisize_eval_manifest.jsonl")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--local-batch-size", type=int, default=512)
    parser.add_argument("--contrastive-batch-size", type=int, default=512)
    parser.add_argument("--triangle-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--target-mode", choices=["raw", "norm", "log", "log_norm"], default="log_norm")
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--triangle-weight", type=float, default=0.05)
    parser.add_argument("--ranking-margin", type=float, default=0.03)
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--output", default="checkpoints/metric_heads/qrl_v2_setb.pt")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Unfreeze LeWM encoder+projector and train end-to-end")
    parser.add_argument("--mazes-per-step", type=int, default=4, help="Number of mazes to encode per step in unfrozen mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainConfig(
        model_ckpt=args.model_ckpt,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        steps=args.steps,
        batch_size=args.batch_size,
        local_batch_size=args.local_batch_size,
        contrastive_batch_size=args.contrastive_batch_size,
        triangle_batch_size=args.triangle_batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        target_mode=args.target_mode,
        regression_weight=args.regression_weight,
        ranking_weight=args.ranking_weight,
        contrastive_weight=args.contrastive_weight,
        triangle_weight=args.triangle_weight,
        ranking_margin=args.ranking_margin,
        contrastive_margin=args.contrastive_margin,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        output=args.output,
        unfreeze_backbone=args.unfreeze_backbone,
        mazes_per_step=args.mazes_per_step,
        seed=args.seed,
        device=args.device,
    )


def load_entries(path: str) -> list[dict[str, object]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_holdout(
    train_entries: list[dict[str, object]],
    val_entries: list[dict[str, object]],
) -> None:
    train_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in train_entries}
    val_topo = {(entry["maze_size"], entry["topology_seed"]) for entry in val_entries}
    train_layout = {entry.get("layout_hash") for entry in train_entries if entry.get("layout_hash")}
    val_layout = {entry.get("layout_hash") for entry in val_entries if entry.get("layout_hash")}
    train_task = {entry.get("task_hash") for entry in train_entries if entry.get("task_hash")}
    val_task = {entry.get("task_hash") for entry in val_entries if entry.get("task_hash")}
    if train_topo & val_topo or train_layout & val_layout or train_task & val_task:
        raise ValueError("train/val leakage detected")


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 70)
    print("QRL V2 TRAINING")
    print("=" * 70)
    print(json.dumps(asdict(cfg), indent=2))

    train_entries = load_entries(cfg.train_manifest)
    val_entries_all = load_entries(cfg.val_manifest)
    val_entries = [entry for entry in val_entries_all if int(entry["maze_size"]) <= 21]
    verify_holdout(train_entries, val_entries)
    print(f"Train entries: {len(train_entries)}, val entries: {len(val_entries)}, overlap=0")

    model = load_backbone(cfg.model_ckpt, device, freeze=not cfg.unfreeze_backbone)
    if cfg.unfreeze_backbone:
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.embedding_projector.parameters():
            param.requires_grad = True
        n_backbone = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Unfrozen backbone params: {n_backbone:,}")

    train_ds = SetBMazeCache(model, train_entries, device, store_observations=cfg.unfreeze_backbone)
    val_ds = SetBMazeCache(model, val_entries, device, store_observations=cfg.unfreeze_backbone)
    train_ds.get(train_ds.sample_maze_idx(rng))
    val_ds.get(val_ds.sample_maze_idx(rng))

    head = QRLHead(latent_dim=cfg.latent_dim, hidden_dims=list(cfg.hidden_dims)).to(device)
    if cfg.unfreeze_backbone:
        backbone_params = list(model.encoder.parameters()) + list(model.embedding_projector.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.backbone_lr},
                {"params": head.parameters(), "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_rank_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    logs: list[dict[str, float]] = []
    train_losses: list[float] = []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        head.train()
        if cfg.unfreeze_backbone:
            step_ds = StepMazeCache(train_ds, model, rng, n_mazes=cfg.mazes_per_step)
            ds_for_training: SetBMazeCache | StepMazeCache = step_ds
        else:
            ds_for_training = train_ds

        z1, z2, target = sample_regression_batch(ds_for_training, cfg, rng, device)
        pred = head(z1, z2)
        reg_loss = F.smooth_l1_loss(pred, target)

        good, bad, goal = sample_local_ranking_batch(ds_for_training, cfg, rng, device)
        good_score = head(good, goal)
        bad_score = head(bad, goal)
        ranking_loss = F.relu(good_score - bad_score + cfg.ranking_margin).mean()

        anchor, pos, neg = sample_contrastive_batch(ds_for_training, cfg, rng, device)
        pos_score = head(anchor, pos)
        neg_score = head(anchor, neg)
        contrastive_loss = F.relu(pos_score - neg_score + cfg.contrastive_margin).mean()

        tri_loss = torch.tensor(0.0, device=device)
        if cfg.triangle_weight > 0 and cfg.triangle_batch_size > 0:
            za, zb, zc = sample_triangle_batch(ds_for_training, cfg, rng, device)
            tri_loss = head.triangle_loss(za, zb, zc)

        loss = (
            cfg.regression_weight * reg_loss
            + cfg.ranking_weight * ranking_loss
            + cfg.contrastive_weight * contrastive_loss
            + cfg.triangle_weight * tri_loss
        )
        opt.zero_grad()
        loss.backward()
        if cfg.unfreeze_backbone:
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        else:
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step()
        train_losses.append(float(loss.item()))

        if step % cfg.eval_every == 0:
            if cfg.unfreeze_backbone:
                model.eval()
            metrics = evaluate_qrl(head, val_ds, cfg, rng, device, model if cfg.unfreeze_backbone else None)
            if cfg.unfreeze_backbone:
                model.train()
            elapsed = time.time() - t0
            log_row = {
                "step": float(step),
                "train_loss": float(np.mean(train_losses[-cfg.eval_every :])),
                "reg_loss": float(reg_loss.item()),
                "ranking_loss": float(ranking_loss.item()),
                "contrastive_loss": float(contrastive_loss.item()),
                "triangle_loss": float(tri_loss.item()),
                "elapsed": float(elapsed),
                **metrics,
            }
            logs.append(log_row)
            print(
                f"Step {step:>6d}/{cfg.steps}: train={log_row['train_loss']:.4f} "
                f"reg={log_row['reg_loss']:.4f} rank={log_row['ranking_loss']:.4f} "
                f"con={log_row['contrastive_loss']:.4f} tri={log_row['triangle_loss']:.4f} "
                f"val_reg={metrics['val_reg_loss']:.4f} val_rank_acc={metrics['val_rank_acc']:.4f} "
                f"val_contrast_acc={metrics['val_contrast_acc']:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
            t0 = time.time()
            if metrics["val_rank_acc"] > best_rank_acc:
                best_rank_acc = metrics["val_rank_acc"]
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                if cfg.unfreeze_backbone:
                    best_state.update({f"backbone.{key}": value.detach().cpu().clone() for key, value in model.state_dict().items()})

    if best_state is not None:
        head.load_state_dict({key: value.to(device) for key, value in best_state.items() if not key.startswith("backbone.")})
        if cfg.unfreeze_backbone:
            model.load_state_dict({key.split("backbone.", 1)[1]: value.to(device) for key, value in best_state.items() if key.startswith("backbone.")})

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "head_type": "qrl_v2",
        "head_state_dict": head.state_dict(),
        "config": {
            **asdict(cfg),
            "hidden_dims": list(cfg.hidden_dims),
            "best_val_rank_acc": best_rank_acc,
        },
        "logs": logs,
        "final_train_loss": float(np.mean(train_losses[-min(1000, len(train_losses)) :])),
    }
    if cfg.unfreeze_backbone:
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["model_config"] = model.config
    torch.save(ckpt, output)
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(ckpt["config"], f, indent=2)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
