#!/usr/bin/env python3
"""Evaluate a Set B QRL metric head with the corrected greedy planners."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdwm.metric_heads.qrl_head import QRLHead
from scripts.eval.eval_setb_distance_head_fixed import (
    create_env,
    manifest_task,
    run_model_free_greedy,
    run_predictor_greedy,
    summarize,
)
from scripts.train.train_dim256 import Unisize256


def load_model_and_qrl(
    model_ckpt: Path,
    qrl_ckpt: Path,
    device: torch.device,
) -> tuple[Unisize256, QRLHead]:
    data = torch.load(qrl_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in data:
        model = Unisize256(data["model_config"], max_size=31).to(device)
        model.load_state_dict(data["model_state_dict"], strict=True)
        model_source = str(qrl_ckpt)
    else:
        ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        model = Unisize256(ckpt["model_config"], max_size=31).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model_source = str(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    cfg = data.get("config", {})
    head = QRLHead(
        latent_dim=int(cfg.get("latent_dim", 256)),
        hidden_dims=cfg.get("hidden_dims", [256, 128]),
        temperature=float(cfg.get("temperature", 0.1)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    head.load_state_dict(data["head_state_dict"], strict=True)
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    print(f"Model source: {model_source}")
    return model, head


def evaluate_method(
    method: str,
    entries: list[dict[str, Any]],
    model: Unisize256,
    head: QRLHead,
    device: torch.device,
    progress_every: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    for idx, entry in enumerate(entries):
        env = create_env(entry)
        start, goal, opt = manifest_task(entry, env)
        maze_size = int(entry["maze_size"])
        run_seed = seed * 10000 + idx
        if method == "model_free_greedy":
            row = run_model_free_greedy(model, head, env, start, goal, maze_size, device, run_seed)
        elif method == "predictor_greedy":
            row = run_predictor_greedy(model, head, env, start, goal, maze_size, device, run_seed)
        else:
            raise ValueError(f"unknown method: {method}")
        row["op_len"] = opt
        row["maze_size"] = maze_size
        row["spl"] = opt / max(int(row["path_length"]), opt) if row["success"] else 0.0
        rows.append(row)
        by_size[maze_size].append(row)
        if (idx + 1) % progress_every == 0:
            print(f"  [{method}] {idx + 1:>4d}/{len(entries)} SR={summarize(rows)['sr']:.4f}")
    result = summarize(rows)
    result["time"] = float(time.time() - t0)
    result["by_size"] = {str(size): summarize(size_rows) for size, size_rows in sorted(by_size.items())}
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
    parser.add_argument("--qrl-ckpt", default="checkpoints/metric_heads/qrl_v2_setb.pt")
    parser.add_argument("--output", default="results/set_b_multisize/qrl_v2_eval.json")
    parser.add_argument("--methods", default="model_free_greedy,predictor_greedy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-size", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    entries = filtered_entries(entries, args.max_per_size, args.limit)
    print(f"Entries: {len(entries)}, sizes={sorted({int(entry['maze_size']) for entry in entries})}")
    print(f"Device: {device}")

    model, head = load_model_and_qrl(Path(args.model_ckpt), Path(args.qrl_ckpt), device)
    results: dict[str, Any] = {
        "manifest": args.manifest,
        "model_ckpt": args.model_ckpt,
        "qrl_ckpt": args.qrl_ckpt,
        "methods": {},
    }
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        print(f"\n[{method}]")
        results["methods"][method] = evaluate_method(
            method, entries, model, head, device, args.progress_every, args.seed
        )
        summary = results["methods"][method]
        print(
            f"  SR={summary['sr']:.4f} SPL={summary['spl']:.4f} "
            f"stuck={summary['stuck_rate']:.4f} invalid={summary['invalid_rate']:.4f} "
            f"S/F={summary['num_success']}/{summary['num_failure']} time={summary['time']:.0f}s"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
