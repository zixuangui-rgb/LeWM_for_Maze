#!/usr/bin/env python3
"""Diagnose predictor rollout degradation.

Planning with JEPA/LeWM depends on more than a good static embedding. The
predictor must keep imagined latents on the real maze manifold. This diagnostic
measures how quickly predicted latents drift away from true future latents.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diagnostics.common import (
    add_common_args,
    all_pairs_bfs,
    create_env,
    encode_observations,
    ensure_dir,
    free_cells,
    load_lewm,
    observe_state,
    read_jsonl,
    run_dir,
    select_entries,
    size_bucket,
    verify_holdout,
    write_json,
)


HISTORY_SIZE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictor rollout degradation.")
    add_common_args(parser)
    parser.add_argument("--max-eval-per-size", type=int, default=40)
    parser.add_argument("--episodes-per-entry", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--horizons", default="1,2,3,5,8,10")
    return parser.parse_args()


def predict_next(
    model: torch.nn.Module,
    ctx_emb: torch.Tensor,
    action: int,
    num_actions: int,
    device: torch.device,
) -> torch.Tensor:
    ctx_act = torch.full((1, HISTORY_SIZE - 1), num_actions - 1, dtype=torch.long, device=device)
    ctx_act[:, -1] = int(action)
    with torch.no_grad():
        pred = model.predictor(ctx_emb, ctx_act)
    return pred[:, -1, :]


def nearest_state_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    all_latents: torch.Tensor,
    true_cell_idx: int,
    bfs: np.ndarray,
) -> dict[str, float]:
    dists = F.mse_loss(all_latents, pred.expand_as(all_latents), reduction="none").sum(dim=1)
    nn_idx = int(dists.argmin().item())
    bfs_err = float(bfs[nn_idx, true_cell_idx]) if bfs[nn_idx, true_cell_idx] >= 0 else float("nan")
    return {
        "nn_exact": float(nn_idx == true_cell_idx),
        "nn_bfs_error": bfs_err,
        "nn_index": float(nn_idx),
    }


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in ["latent_mse", "cosine", "nn_exact", "nn_bfs_error"]:
        vals = np.asarray([row[key] for row in rows if np.isfinite(row[key])], dtype=np.float64)
        out[key] = float(vals.mean()) if vals.size else float("nan")
    out["n"] = float(len(rows))
    return out


def main() -> None:
    args = parse_args()
    out = ensure_dir(run_dir(args))
    metrics_dir = ensure_dir(out / "metrics")
    train_entries = read_jsonl(args.train_manifest)
    eval_entries_all = read_jsonl(args.eval_manifest)
    verify_holdout(train_entries, eval_entries_all)
    eval_entries = select_entries(eval_entries_all, args.max_eval_per_size, args.seed + 301)
    horizons = sorted({int(item.strip()) for item in args.horizons.split(",") if item.strip()})
    max_horizon = max(horizons)
    seq_len = max(args.sequence_length, max_horizon + 1)

    device = torch.device(args.device)
    model = load_lewm(args.model_ckpt, device)
    rng = np.random.default_rng(args.seed + 302)

    print("=" * 80)
    print("EVALUATE PREDICTOR ROLLOUT")
    print("=" * 80)
    print(f"entries={len(eval_entries)} horizons={horizons} seq_len={seq_len}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for entry_idx, entry in enumerate(eval_entries):
        env = create_env(entry)
        size = int(entry["maze_size"])
        cells = free_cells(env)
        cell_to_idx = {int(cell): i for i, cell in enumerate(cells.tolist())}
        all_obs = [observe_state(env, int(cell)) for cell in cells.tolist()]
        all_latents = encode_observations(model, all_obs, size, device).detach()
        bfs = all_pairs_bfs(env._maze_mask, cells, env.config.width)

        for ep in range(args.episodes_per_entry):
            batch = env.sample_sequence(batch_size=1, sequence_length=seq_len)
            obs = batch.observations[0].cpu().numpy()
            actions = batch.actions[0].cpu().numpy().astype(np.int64)
            states = batch.states[0].cpu().numpy().astype(np.int64)
            true_latents = encode_observations(model, [obs[t] for t in range(seq_len)], size, device).detach()

            # Closed-loop rollout: append predicted latents back into context.
            ctx_closed = true_latents[0:1].unsqueeze(0).repeat(1, HISTORY_SIZE, 1)
            # Teacher-forced rollout: context is refreshed with true latents.
            ctx_teacher = ctx_closed.clone()
            closed_preds: dict[int, torch.Tensor] = {}
            teacher_preds: dict[int, torch.Tensor] = {}

            for h in range(1, max_horizon + 1):
                action = int(actions[h - 1])
                pred_closed = predict_next(model, ctx_closed, action, env.config.action_vocab_size, device)
                pred_teacher = predict_next(model, ctx_teacher, action, env.config.action_vocab_size, device)
                closed_preds[h] = pred_closed.squeeze(0)
                teacher_preds[h] = pred_teacher.squeeze(0)
                ctx_closed = torch.cat([ctx_closed[:, 1:], pred_closed.unsqueeze(1)], dim=1)
                true_next = true_latents[h : h + 1].unsqueeze(0)
                ctx_teacher = torch.cat([ctx_teacher[:, 1:], true_next], dim=1)

            for mode, preds in [("closed_loop", closed_preds), ("teacher_forced", teacher_preds)]:
                for h in horizons:
                    pred = preds[h]
                    true = true_latents[h]
                    true_cell_idx = cell_to_idx.get(int(states[h]), -1)
                    if true_cell_idx < 0:
                        continue
                    nn = nearest_state_metrics(pred, true, all_latents, true_cell_idx, bfs)
                    rows.append(
                        {
                            "mode": mode,
                            "horizon": h,
                            "maze_size": size,
                            "bucket": size_bucket(size, args.seen_max_size),
                            "topology_seed": int(entry["topology_seed"]),
                            "episode": ep,
                            "latent_mse": float(F.mse_loss(pred, true).item()),
                            "cosine": float(F.cosine_similarity(pred.view(1, -1), true.view(1, -1)).item()),
                            "nn_exact": nn["nn_exact"],
                            "nn_bfs_error": nn["nn_bfs_error"],
                        }
                    )

        if (entry_idx + 1) % 20 == 0 or entry_idx + 1 == len(eval_entries):
            print(f"{entry_idx + 1:>4d}/{len(eval_entries)} entries elapsed={time.time() - t0:.1f}s", flush=True)

    grouped: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    by_size: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_bucket: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        mode = str(row["mode"])
        horizon = str(row["horizon"])
        grouped[mode][horizon].append(row)
        by_size[mode][str(row["maze_size"])][horizon].append(row)
        by_bucket[mode][str(row["bucket"])][horizon].append(row)

    summary: dict[str, Any] = {}
    for mode, by_h in grouped.items():
        summary[mode] = {
            "overall": {h: summarize_rows(group) for h, group in sorted(by_h.items(), key=lambda kv: int(kv[0]))},
            "by_bucket": {
                bucket: {h: summarize_rows(group) for h, group in sorted(by_h2.items(), key=lambda kv: int(kv[0]))}
                for bucket, by_h2 in sorted(by_bucket[mode].items())
            },
            "by_size": {
                size: {h: summarize_rows(group) for h, group in sorted(by_h2.items(), key=lambda kv: int(kv[0]))}
                for size, by_h2 in sorted(by_size[mode].items(), key=lambda kv: int(kv[0]))
            },
        }

    output = {
        "metadata": {
            "model_ckpt": args.model_ckpt,
            "eval_manifest": args.eval_manifest,
            "max_eval_per_size": args.max_eval_per_size,
            "episodes_per_entry": args.episodes_per_entry,
            "sequence_length": seq_len,
            "horizons": horizons,
            "seed": args.seed,
        },
        "summary": summary,
        "rows": rows,
    }
    write_json(metrics_dir / "predictor_rollout.json", output)
    print(f"Saved: {metrics_dir / 'predictor_rollout.json'}")


if __name__ == "__main__":
    main()
