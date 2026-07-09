#!/usr/bin/env python3
"""Unified training for GCRL and QRL metric heads on frozen LeWM latents.

Usage:
    python scripts/train/train_metric_heads.py --head gcrl --steps 15000
    python scripts/train/train_metric_heads.py --head qrl --steps 15000
"""

import argparse, json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.metric_heads.gcrl_head import GCRLHead
from hdwm.metric_heads.qrl_head import QRLHead
from hdwm.metric_heads.distance_head import DistanceHead
from scripts.train.train_ablation_models import OriginalLeWM
from scripts.train.train_distance_head import LatentPairDataset

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_head(head_type, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {head_type.upper()} head...")
    print(f"  LR={config['lr']}, steps={config['steps']}, batch={config['batch_size']}")

    # Load LeWM
    ckpt = torch.load(config["lewm_ckpt"], map_location=device, weights_only=False)
    model = OriginalLeWM(ckpt["model_config"], max_size=31).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load data
    with open(config["train_manifest"]) as f:
        train_entries = [json.loads(l) for l in f if l.strip()]
    with open(config["val_manifest"]) as f:
        val_entries = [json.loads(l) for l in f if l.strip()]

    # Verify topology hold-out
    train_topo = set(e["topology_seed"] for e in train_entries)
    val_topo = set(e["topology_seed"] for e in val_entries)
    assert len(train_topo & val_topo) == 0, f"Topology leakage! Overlap: {len(train_topo & val_topo)}"
    print(f"  Topology hold-out verified: train={len(train_topo)}, val={len(val_topo)}, overlap=0")

    train_ds = LatentPairDataset(model, train_entries, device, config, is_train=True)
    val_ds = LatentPairDataset(model, val_entries, device, config, is_train=False)
    train_ds.get_maze(0); val_ds.get_maze(0)  # preload

    # Create head
    if head_type == "gcrl":
        head = GCRLHead(latent_dim=config["latent_dim"], hidden_dims=[256, 128],
                        horizons=[1, 2, 4, 8, 16, 32, 64]).to(device)
    elif head_type == "qrl":
        head = QRLHead(latent_dim=config["latent_dim"], hidden_dims=[256, 128]).to(device)
    else:
        raise ValueError(f"Unknown head: {head_type}")

    print(f"  Params: {sum(p.numel() for p in head.parameters()):,}")
    opt = optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=1e-5)
    rng = np.random.default_rng(42)

    train_losses = []
    t0 = time.time()

    for step in range(1, config["steps"] + 1):
        head.train()

        # Sample batch of latent pairs
        z1, z2, bfs_labels = train_ds.sample_batch(config["batch_size"], rng)

        if head_type == "gcrl":
            # Randomly pick a horizon bucket for each pair
            horizons = head.horizons
            h_indices = torch.randint(0, len(horizons), (z1.shape[0],), device=device)
            # Label: positive if bfs <= horizon
            h_vals = torch.tensor([horizons[i] for i in h_indices.tolist()],
                                  dtype=torch.float32, device=device)
            targets = (bfs_labels <= h_vals).float()
            logits = head(z1, z2, h_indices)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        elif head_type == "qrl":
            # MSE regression + triangle loss
            pred = head(z1, z2)
            mse_loss = F.mse_loss(pred, bfs_labels)

            # Triangle loss on random triplets (ensure same batch size)
            bsz = min(64, config["batch_size"] // 4)
            z_a, _, _ = train_ds.sample_batch(bsz, rng)
            z_b, _, _ = train_ds.sample_batch(bsz, rng)
            z_c, _, _ = train_ds.sample_batch(bsz, rng)
            # Ensure same size
            min_sz = min(z_a.shape[0], z_b.shape[0], z_c.shape[0])
            tri_loss = head.triangle_loss(z_a[:min_sz], z_b[:min_sz], z_c[:min_sz])

            loss = mse_loss + 0.1 * tri_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        train_losses.append(loss.item())

        if step % 2000 == 0:
            avg_loss = np.mean(train_losses[-2000:])
            elapsed = time.time() - t0

            # Quick val check
            head.eval()
            with torch.no_grad():
                z1v, z2v, bfs_v = val_ds.sample_batch(512, rng)
                if head_type == "gcrl":
                    hi = torch.randint(0, len(head.horizons), (z1v.shape[0],), device=device)
                    hv = torch.tensor([head.horizons[i] for i in hi.tolist()], device=device)
                    targets_v = (bfs_v <= hv).float()
                    logits_v = head(z1v, z2v, hi)
                    val_acc = ((torch.sigmoid(logits_v) > 0.5) == (targets_v > 0.5)).float().mean().item()
                    print(f"  Step {step:>5d}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")
                else:
                    pred_v = head(z1v, z2v)
                    val_mse = F.mse_loss(pred_v, bfs_v).item()
                    print(f"  Step {step:>5d}: loss={avg_loss:.4f}  val_mse={val_mse:.4f}  ({elapsed:.0f}s)")
            t0 = time.time()

    # Save
    out_dir = Path("checkpoints/metric_heads")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "head_type": head_type,
        "config": config,
        "final_loss": float(np.mean(train_losses[-1000:])),
    }, out_dir / f"{head_type}_head.pt")
    print(f"  Saved: {out_dir / f'{head_type}_head.pt'}")
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True, choices=["gcrl", "qrl"])
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    config = {
        "latent_dim": 256, "latent_source": "embedding",
        "lewm_ckpt": "checkpoints/ablation/original_lewm.pt",
        "train_manifest": "data/splits/fixed11_train_manifest.jsonl",
        "val_manifest": "data/splits/fixed11_val_manifest.jsonl",
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "max_distance": 121, "pairs_per_maze": 64,
    }
    train_head(args.head, config)


if __name__ == "__main__":
    main()
