#!/usr/bin/env python3
"""Evaluate trained Distance Head on held-out eval/test splits.

Computes:
    - MAE, MSE, RMSE
    - Pearson r, Spearman ρ
    - Per-bucket error (by true BFS distance)
    - Latent source comparison (encoded vs embedding)

Usage:
    python scripts/eval/eval_distance_head.py
    python scripts/eval/eval_distance_head.py --eval-manifest data/splits/fixed11_test_manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.metric_heads.distance_head import DistanceHead
from scripts.train.train_ablation_models import OriginalLeWM
from scripts.train.train_distance_head import (
    pre_extract_maze_latents, LatentPairDataset, DEFAULT_CONFIG,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    head: DistanceHead,
    model: OriginalLeWM,
    entries: list[dict],
    device: torch.device,
    config: dict,
    max_pairs_per_maze: int = 200,
) -> dict:
    """Evaluate distance head on a set of held-out mazes.

    For each maze, samples pairs and computes prediction vs label metrics.
    """
    head.eval()
    model.eval()

    all_preds = []
    all_labels = []
    all_maze_sizes = []
    bucket_errors = defaultdict(list)

    rng = np.random.default_rng(42)
    n_mazes = len(entries)

    print(f"  Evaluating on {n_mazes} mazes...")
    t0 = time.time()

    for idx in range(n_mazes):
        entry = entries[idx]
        latents, cells, bfs = pre_extract_maze_latents(
            model, entry, device, config["latent_source"],
        )
        n = len(cells)

        # Sample pairs
        n_pairs = min(max_pairs_per_maze, n * (n - 1))
        pairs_i = []
        pairs_j = []
        for _ in range(n_pairs):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            pairs_i.append(i)
            pairs_j.append(j)

        if len(pairs_i) < 2:
            continue

        z1 = latents[torch.tensor(pairs_i, device=device)]
        z2 = latents[torch.tensor(pairs_j, device=device)]

        with torch.no_grad():
            preds = head(z1, z2)  # [N_pairs]

        labels_list = []
        for i, j in zip(pairs_i, pairs_j):
            d = bfs[i, j]
            labels_list.append(float(d) if d >= 0 else float(config["max_distance"]))

        labels_t = torch.tensor(labels_list, dtype=torch.float32, device=device)

        all_preds.append(preds.cpu())
        all_labels.append(torch.tensor(labels_list))
        all_maze_sizes.append(entry["maze_size"])

        # Bucketed errors
        for p, l in zip(preds.cpu().numpy(), labels_list):
            bucket = int(l) // 5 * 5  # 0-4, 5-9, 10-14, ...
            bucket_errors[bucket].append(abs(p - l))

        if (idx + 1) % max(1, n_mazes // 5) == 0:
            print(f"    {idx+1}/{n_mazes} mazes ({time.time()-t0:.0f}s)")

    all_preds_np = torch.cat([p.float() for p in all_preds]).numpy()
    all_labels_np = np.concatenate(all_labels)

    # Overall metrics
    mae = float(np.mean(np.abs(all_preds_np - all_labels_np)))
    mse = float(np.mean((all_preds_np - all_labels_np) ** 2))
    rmse = float(np.sqrt(mse))
    pearson_r, pearson_p = pearsonr(all_preds_np, all_labels_np)
    spearman_rho, spearman_p = spearmanr(all_preds_np, all_labels_np)

    # Bucket summary
    bucket_summary = {}
    for bucket in sorted(bucket_errors.keys()):
        errs = bucket_errors[bucket]
        bucket_summary[int(bucket)] = {
            "count": len(errs),
            "mae": float(np.mean(errs)),
        }

    # Per-maze-size breakdown
    size_metrics = defaultdict(list)
    for preds_t, labels_lt, sz in zip(all_preds, all_labels, all_maze_sizes):
        preds_np = preds_t.float().numpy()
        labels_np = np.array(labels_lt, dtype=np.float32)
        if len(labels_np) > 1:
            sp, _ = spearmanr(preds_np, labels_np)
        else:
            sp = 0.0
        size_metrics[sz].append(sp)

    size_summary = {}
    for sz in sorted(size_metrics.keys()):
        vals = size_metrics[sz]
        size_summary[sz] = {
            "count": len(vals),
            "mean_spearman": float(np.mean(vals)),
        }

    return {
        "num_mazes": n_mazes,
        "num_pairs": len(all_labels_np),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "bucket_errors": bucket_summary,
        "size_breakdown": size_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate Distance Head")
    p.add_argument("--ckpt", default="checkpoints/metric_heads/distance_head.pt")
    p.add_argument("--lewm-ckpt", default="checkpoints/ablation/original_lewm.pt")
    p.add_argument("--eval-manifest", default="data/splits/fixed11_val_manifest.jsonl")
    p.add_argument("--output", default="results/phase4_metric_heads/distance_head_eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DISTANCE HEAD EVALUATION")
    print("=" * 70)

    # Load distance head
    print("[1] Loading distance head...")
    dc = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = dc["config"]
    head = DistanceHead(
        latent_dim=config["latent_dim"],
        hidden_dims=config["hidden_dims"],
        input_mode=config["input_mode"],
    ).to(device)
    head.load_state_dict(dc["head_state_dict"])
    print(f"  Config: latent_source={config['latent_source']}, input_mode={config['input_mode']}")
    print(f"  Hidden dims: {config['hidden_dims']}")
    print()

    # Load frozen LeWM
    print("[2] Loading frozen LeWM...")
    ckpt = torch.load(args.lewm_ckpt, map_location=device, weights_only=False)
    model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print()

    # Load eval manifest
    print("[3] Loading eval manifest...")
    with open(args.eval_manifest) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    print(f"  Entries: {len(entries)}")
    print()

    # Evaluate
    print("[4] Evaluating...")
    metrics = evaluate(head, model, entries, device, config)
    print()

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Pairs evaluated: {metrics['num_pairs']}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Pearson r:  {metrics['pearson_r']:.4f}  (p={metrics['pearson_p']:.4f})")
    print(f"  Spearman ρ: {metrics['spearman_rho']:.4f}  (p={metrics['spearman_p']:.4f})")
    print()
    print("  Bucket errors (MAE per BFS bucket):")
    for bucket in sorted(metrics['bucket_errors'].keys()):
        b = metrics['bucket_errors'][bucket]
        print(f"    bfs={bucket:>3d}-{bucket+4:>3d}: MAE={b['mae']:.2f} (n={b['count']})")

    # Save
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved: {out_dir / 'eval_metrics.json'}")
    print("=" * 70)

    # Also compute L2 baseline correlation for comparison
    print("\n[5] Computing L2 baseline correlation for comparison...")
    l2_preds = []
    l2_labels = []
    rng = np.random.default_rng(42)
    for idx in range(min(50, len(entries))):
        latents, cells, bfs = pre_extract_maze_latents(model, entries[idx], device, config["latent_source"])
        n = len(cells)
        for _ in range(100):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            z1 = latents[i:i+1]
            z2 = latents[j:j+1]
            d = bfs[i, j]
            if d < 0:
                continue
            l2 = float(F.mse_loss(z1, z2, reduction='none').sum(dim=-1).item())
            l2_preds.append(l2)
            l2_labels.append(float(d))

    l2_spearman, _ = spearmanr(l2_preds, l2_labels)
    l2_pearson, _ = pearsonr(l2_preds, l2_labels)
    print(f"  Latent L2: Spearman ρ={l2_spearman:.4f}, Pearson r={l2_pearson:.4f}")
    print(f"  Distance Head: Spearman ρ={metrics['spearman_rho']:.4f}, Pearson r={metrics['pearson_r']:.4f}")
    print(f"  Improvement: Δρ={metrics['spearman_rho'] - l2_spearman:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
