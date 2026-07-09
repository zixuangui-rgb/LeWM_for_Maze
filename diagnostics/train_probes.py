#!/usr/bin/env python3
"""Train and evaluate layer-wise diagnostic probes.

The probe suite is deliberately broad enough to avoid having to redo a second
round of diagnostics later. It tests whether each representation layer keeps:

- agent and goal coordinates;
- local wall/valid-action information;
- geodesic BFS distance-to-goal;
- optimal local action information.

For each task we train both a linear probe and an MLP probe. Linear probes tell
us whether the information is directly accessible; MLP probes tell us whether
the information exists but is encoded nonlinearly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diagnostics.common import add_common_args, ensure_dir, read_json, run_dir, write_json


CLASS_TASKS = ("agent_x", "agent_y", "goal_x", "goal_y", "optimal_action")
REG_TASKS = ("bfs_distance_norm",)
MULTILABEL_TASKS = ("valid_action",)
ALL_TASKS = CLASS_TASKS + REG_TASKS + MULTILABEL_TASKS
FIXED_DIM_LAYERS = {"spatial_pool", "encoded", "embedding"}


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train layer-wise diagnostic probes.")
    add_common_args(parser)
    parser.add_argument("--cache-path", default=None, help="Defaults to diagnostics_runs/<run-id>/feature_cache/features.pt")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-types", default="linear,mlp")
    return parser.parse_args()


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def make_probe(kind: str, in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    if kind == "linear":
        return LinearProbe(in_dim, out_dim)
    if kind == "mlp":
        return MLPProbe(in_dim, out_dim, hidden_dim)
    raise ValueError(f"unknown probe type: {kind}")


def output_dim(task: str, size: int | None, max_coord: int) -> int:
    if task in {"agent_x", "agent_y", "goal_x", "goal_y"}:
        return int(size if size is not None else max_coord)
    if task == "optimal_action":
        return 4
    if task == "valid_action":
        return 4
    if task == "bfs_distance_norm":
        return 1
    raise ValueError(f"unknown task: {task}")


def task_tensors(
    labels: dict[str, dict[str, torch.Tensor]],
    task: str,
    sizes: list[str],
) -> torch.Tensor:
    values = [labels[task][size] for size in sizes]
    y = torch.cat(values, dim=0)
    if task == "bfs_distance_norm":
        return y.float().view(-1, 1)
    if task == "valid_action":
        return y.float()
    return y.long()


def feature_tensors(features: dict[str, dict[str, torch.Tensor]], layer: str, sizes: list[str]) -> torch.Tensor:
    return torch.cat([features[layer][size].float() for size in sizes], dim=0)


def class_metrics(logits: torch.Tensor, y: torch.Tensor, optimal_mask: torch.Tensor | None = None) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    valid = y >= 0
    if valid.sum() == 0:
        return {"accuracy": float("nan"), "n": 0.0}
    pred = pred[valid]
    yv = y[valid]
    metrics = {"accuracy": float((pred == yv).float().mean().item()), "n": float(valid.sum().item())}
    if optimal_mask is not None:
        mask = optimal_mask[valid]
        top1_any = mask.gather(1, pred[:, None]).squeeze(1).gt(0)
        metrics["top1_any_optimal"] = float(top1_any.float().mean().item())
    return metrics


def multilabel_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return {
        "action_accuracy": float((pred == y).float().mean().item()),
        "exact_match": float((pred == y).all(dim=1).float().mean().item()),
        "n": float(y.shape[0]),
    }


def regression_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    err = pred.squeeze(1) - y.squeeze(1)
    ss_res = float((err**2).sum().item())
    centered = y.squeeze(1) - y.squeeze(1).mean()
    ss_tot = float((centered**2).sum().item())
    return {
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt((err**2).mean()).item()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "n": float(y.shape[0]),
    }


def train_one_probe(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    task: str,
    out_dim: int,
    probe_type: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    optimal_mask_eval: torch.Tensor | None = None,
) -> tuple[nn.Module, dict[str, float], dict[str, torch.Tensor]]:
    mean, std = standardize_fit(x_train)
    x_train = ((x_train - mean) / std).float()
    x_eval = ((x_eval - mean) / std).float()

    valid_train = torch.ones(x_train.shape[0], dtype=torch.bool)
    if task == "optimal_action":
        valid_train = y_train >= 0
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]

    model = make_probe(probe_type, x_train.shape[1], out_dim, hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            if task in CLASS_TASKS:
                loss = F.cross_entropy(logits, yb)
            elif task in MULTILABEL_TASKS:
                loss = F.binary_cross_entropy_with_logits(logits, yb.float())
            elif task in REG_TASKS:
                loss = F.smooth_l1_loss(logits, yb.float())
            else:
                raise ValueError(task)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        metrics = evaluate_model(model, x_eval, y_eval, task, device, optimal_mask_eval)
        score = select_score(task, metrics)
        if score > best_score or best_state is None:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_model(model, x_eval, y_eval, task, device, optimal_mask_eval)
    return model, metrics, {"mean": mean.cpu(), "std": std.cpu()}


def evaluate_model(
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    task: str,
    device: torch.device,
    optimal_mask_eval: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_eval.to(device)).cpu()
    if task in {"agent_x", "agent_y", "goal_x", "goal_y"}:
        return class_metrics(logits, y_eval)
    if task == "optimal_action":
        return class_metrics(logits, y_eval, optimal_mask_eval=optimal_mask_eval)
    if task == "valid_action":
        return multilabel_metrics(logits, y_eval.float())
    if task == "bfs_distance_norm":
        return regression_metrics(logits, y_eval.float())
    raise ValueError(task)


def select_score(task: str, metrics: dict[str, float]) -> float:
    if task == "bfs_distance_norm":
        rmse = metrics.get("rmse", float("inf"))
        return -rmse if math.isfinite(rmse) else -float("inf")
    if task == "valid_action":
        return metrics.get("exact_match", -float("inf"))
    if task == "optimal_action":
        return metrics.get("top1_any_optimal", metrics.get("accuracy", -float("inf")))
    return metrics.get("accuracy", -float("inf"))


def save_probe(
    path: Path,
    model: nn.Module,
    stats: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "standardization": stats,
            "metadata": metadata,
        },
        path,
    )


def run_scope(
    *,
    cache: dict[str, Any],
    scope: str,
    layer: str,
    task: str,
    probe_type: str,
    train_sizes: list[str],
    eval_sizes: list[str],
    out_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    ckpt_dir: Path,
    extra_eval_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    train_split = cache["splits"]["train"]
    eval_split = cache["splits"]["eval"]
    x_train = feature_tensors(train_split["features"], layer, train_sizes)
    y_train = task_tensors(train_split["labels"], task, train_sizes)
    x_eval = feature_tensors(eval_split["features"], layer, eval_sizes)
    y_eval = task_tensors(eval_split["labels"], task, eval_sizes)
    optimal_mask_eval = None
    if task == "optimal_action":
        optimal_mask_eval = task_tensors(eval_split["labels"], "optimal_action_mask", eval_sizes).float()

    model, metrics, stats = train_one_probe(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        task=task,
        out_dim=out_dim,
        probe_type=probe_type,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        optimal_mask_eval=optimal_mask_eval,
    )
    metadata = {
        "scope": scope,
        "layer": layer,
        "task": task,
        "probe_type": probe_type,
        "train_sizes": train_sizes,
        "eval_sizes": eval_sizes,
        "out_dim": out_dim,
        "metrics": metrics,
    }
    if extra_eval_groups:
        grouped: dict[str, dict[str, float]] = {}
        for group_name, group_sizes in extra_eval_groups.items():
            gx = feature_tensors(eval_split["features"], layer, group_sizes)
            gy = task_tensors(eval_split["labels"], task, group_sizes)
            gx = ((gx - stats["mean"]) / stats["std"]).float()
            gmask = None
            if task == "optimal_action":
                gmask = task_tensors(eval_split["labels"], "optimal_action_mask", group_sizes).float()
            grouped[group_name] = evaluate_model(model, gx, gy, task, device, gmask)
        metadata["metrics_by_group"] = grouped
    ckpt_name = f"{scope}_{layer}_{task}_{probe_type}.pt".replace("/", "_")
    save_probe(ckpt_dir / ckpt_name, model, stats, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    cache_path = Path(args.cache_path) if args.cache_path else out / "feature_cache" / "features.pt"
    ckpt_dir = ensure_dir(out / "probe_checkpoints")
    metrics_dir = ensure_dir(out / "metrics")
    device = torch.device(args.device)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    manifest_path = out / "feature_cache" / "manifest.json"
    cache_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    metadata = cache.get("metadata", {})
    max_coord = max(int(size) for size in cache_manifest.get("sizes", {}).get("eval", [])) if cache_manifest else 31

    available_layers = sorted(cache["splits"]["train"]["features"].keys())
    layers = available_layers if args.layers == "all" else [item.strip() for item in args.layers.split(",") if item.strip()]
    tasks = list(ALL_TASKS) if args.tasks == "all" else [item.strip() for item in args.tasks.split(",") if item.strip()]
    probe_types = [item.strip() for item in args.probe_types.split(",") if item.strip()]

    train_sizes_by_layer = {
        layer: sorted(cache["splits"]["train"]["features"][layer].keys(), key=int)
        for layer in layers
    }
    eval_sizes_by_layer = {
        layer: sorted(cache["splits"]["eval"]["features"][layer].keys(), key=int)
        for layer in layers
    }

    results: list[dict[str, Any]] = []
    print("=" * 80)
    print("TRAIN LAYER-WISE DIAGNOSTIC PROBES")
    print("=" * 80)

    for layer in layers:
        for task in tasks:
            for probe_type in probe_types:
                # Per-size probes: fair for spatial_flat and mirrors the symbolic BFS diagnostic.
                common_sizes = sorted(
                    set(train_sizes_by_layer[layer]) & set(eval_sizes_by_layer[layer]),
                    key=int,
                )
                for size_key in common_sizes:
                    out_dim = output_dim(task, int(size_key), max_coord)
                    print(f"[per_size] layer={layer} size={size_key} task={task} probe={probe_type}", flush=True)
                    row = run_scope(
                        cache=cache,
                        scope=f"per_size_sz{size_key}",
                        layer=layer,
                        task=task,
                        probe_type=probe_type,
                        train_sizes=[size_key],
                        eval_sizes=[size_key],
                        out_dim=out_dim,
                        args=args,
                        device=device,
                        ckpt_dir=ckpt_dir,
                    )
                    results.append(row)

                # Unified probes: only possible when feature dimension is stable across sizes.
                if layer in FIXED_DIM_LAYERS:
                    train_sizes = train_sizes_by_layer[layer]
                    eval_sizes = eval_sizes_by_layer[layer]
                    out_dim = output_dim(task, None, max_coord)
                    print(f"[unified] layer={layer} task={task} probe={probe_type}", flush=True)
                    extra_groups = {
                        "seen": [size for size in eval_sizes if int(size) <= int(metadata.get("seen_max_size", args.seen_max_size))],
                        "ood": [size for size in eval_sizes if int(size) > int(metadata.get("seen_max_size", args.seen_max_size))],
                    }
                    extra_groups.update({f"sz{size}": [size] for size in eval_sizes})
                    extra_groups = {name: sizes for name, sizes in extra_groups.items() if sizes}
                    row = run_scope(
                        cache=cache,
                        scope="unified_all_eval",
                        layer=layer,
                        task=task,
                        probe_type=probe_type,
                        train_sizes=train_sizes,
                        eval_sizes=eval_sizes,
                        out_dim=out_dim,
                        args=args,
                        device=device,
                        ckpt_dir=ckpt_dir,
                        extra_eval_groups=extra_groups,
                    )
                    results.append(row)

    output = {
        "cache_path": str(cache_path),
        "metadata": metadata,
        "probe_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "probe_types": probe_types,
            "tasks": tasks,
            "layers": layers,
        },
        "results": results,
    }
    write_json(metrics_dir / "probe_metrics.json", output)
    print(f"Saved: {metrics_dir / 'probe_metrics.json'}")
    print(f"Saved checkpoints under: {ckpt_dir}")


if __name__ == "__main__":
    main()
