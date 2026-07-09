#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-setb_seqlen_ablation_20260708}
S=2
GPU=${GPU:-0}

LEWM=checkpoints/backbones/unisize_dim256_${RUN_ID}_seqlen${S}.pt
DH=checkpoints/metric_heads/distance_head_simple_${RUN_ID}_seqlen${S}.pt
QRL=checkpoints/metric_heads/qrl_v2_frozen_${RUN_ID}_seqlen${S}.pt

mkdir -p logs/$RUN_ID results/$RUN_ID checkpoints/metric_heads

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/train_distance_head_simple_setb.py \
  --model-ckpt "$LEWM" \
  --train-manifest data/splits/unisize_train_manifest.jsonl \
  --eval-manifest data/splits/unisize_eval_manifest.jsonl \
  --output "$DH" \
  --target-mode log_norm \
  --loss smooth_l1 \
  --steps 30000 \
  --batch-size 512 \
  --pairs-per-maze 64 \
  --device cuda \
  2>&1 | tee logs/$RUN_ID/train_dh_simple_seqlen${S}.log

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval/eval_setb_distance_head_fixed.py \
  --model-ckpt "$LEWM" \
  --head-ckpt "$DH" \
  --manifest data/splits/unisize_eval_manifest.jsonl \
  --methods model_free_greedy,predictor_greedy \
  --output results/$RUN_ID/dh_simple_greedy_full900_seqlen${S}.json \
  --device cuda \
  --progress-every 100 \
  2>&1 | tee logs/$RUN_ID/dh_simple_greedy_full900_seqlen${S}.log

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train/train_qrl_v2.py \
  --model-ckpt "$LEWM" \
  --train-manifest data/splits/unisize_train_manifest.jsonl \
  --val-manifest data/splits/unisize_eval_manifest.jsonl \
  --output "$QRL" \
  --steps 30000 \
  --target-mode log_norm \
  --regression-weight 0.5 \
  --ranking-weight 2.0 \
  --contrastive-weight 1.0 \
  --triangle-weight 0.05 \
  --device cuda \
  2>&1 | tee logs/$RUN_ID/train_qrl_v2_frozen_seqlen${S}.log

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval/eval_setb_qrl.py \
  --model-ckpt "$LEWM" \
  --qrl-ckpt "$QRL" \
  --manifest data/splits/unisize_eval_manifest.jsonl \
  --methods model_free_greedy,predictor_greedy \
  --output results/$RUN_ID/qrl_v2_frozen_greedy_full900_seqlen${S}.json \
  --device cuda \
  --progress-every 100 \
  2>&1 | tee logs/$RUN_ID/qrl_v2_frozen_greedy_full900_seqlen${S}.log
