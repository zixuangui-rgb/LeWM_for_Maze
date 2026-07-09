# Set B World Model Navigation Report

Date: 2026-07-08
Main run id: `setb_seqlen_ablation_20260708`

## 1. Research Question

We re-ran the Set B navigation experiments after fixing the planner/evaluation
bugs. The goal was to understand whether the LeWM backbone sequence length
changes latent navigability, and whether BFS-supervised metric heads
(`DistanceHead`, `QRL`) can outperform direct latent L2 planning.

The evaluated LeWM backbones differ only in training sequence length:

| Backbone | Training sequence length | Intended interpretation |
| --- | ---: | --- |
| `seqlen8` | 8 | Original setting, longer temporal context |
| `seqlen4` | 4 | Matches the current planner history size 3 more closely |
| `seqlen2` | 2 | Pure one-step next-latent prediction setting |

All three use:

- Training manifest: `data/splits/unisize_train_manifest.jsonl`
- Eval manifest: `data/splits/unisize_eval_manifest.jsonl`
- Train entries: 2800, sizes 9-21
- Eval entries: 900, sizes 9-25
- Eval topologies/tasks are strictly held out from training
- LeWM latent dim: 256
- CNN channels: `(64, 128, 256)`
- Steps: 30000
- Batch size: 256

## 2. Important Planner Fixes

Old planner results with high `stuck`/`invalid` rates are not comparable and
should be treated as invalid. The corrected planner/evaluator now:

- excludes STAY/no-op actions from planning candidate actions;
- masks wall/no-move actions;
- avoids immediate backtracking when another moving action exists;
- falls back to a valid one-step action if CEM proposes an invalid first action;
- evaluates only fixed tasks from `unisize_eval_manifest.jsonl`.

The old file `results/set_b_multisize/cem_results.json` has very high invalid
rates and is kept only as historical context, not as a valid baseline.

## 3. LeWM Training Loss

Loss is averaged over the last 500 steps at each log point.

| Backbone | Step | total | pred | abs | rel | goal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seqlen2 | 500 | 0.2040 | 0.0471 | 0.0237 | 0.0971 | 0.0877 |
| seqlen2 | 30000 | 0.0155 | 0.0027 | 0.0011 | 0.0014 | 0.0014 |
| seqlen4 | 500 | 0.1755 | 0.0363 | 0.0228 | 0.0846 | 0.0762 |
| seqlen4 | 30000 | 0.0140 | 0.0021 | 0.0010 | 0.0016 | 0.0017 |
| seqlen8 | 500 | 0.1588 | 0.0311 | 0.0242 | 0.0764 | 0.0695 |
| seqlen8 | 30000 | 0.0125 | 0.0014 | 0.0008 | 0.0015 | 0.0014 |

Training loss improves normally in all three settings. `seqlen8` has the lowest
final prediction loss, but this does not directly translate to best navigation.

## 4. Symbolic BFS Probe

Implementation:

- Train per-size MLP probes on frozen LeWM spatial CNN features from the train
  manifest.
- Decode `agent_x`, `agent_y`, `goal_x`, `goal_y`.
- Run oracle BFS on the true occupancy map using decoded start/goal positions.
- Evaluate on held-out eval manifest topologies.
- Sizes 9-21 only, because probes are trained per seen size.
- 100 eval tasks per size.

| Size | seqlen2 SR | seqlen4 SR | seqlen8 SR | seqlen2 posOK | seqlen4 posOK | seqlen8 posOK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 11 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 |
| 13 | 0.98 | 0.96 | 0.97 | 0.97 | 0.96 | 0.95 |
| 15 | 1.00 | 0.96 | 0.96 | 0.95 | 0.94 | 0.93 |
| 17 | 0.93 | 0.98 | 0.97 | 0.91 | 0.97 | 0.96 |
| 19 | 0.90 | 0.94 | 0.90 | 0.77 | 0.85 | 0.81 |
| 21 | 0.79 | 0.85 | 0.79 | 0.71 | 0.75 | 0.73 |
| Mean | 0.943 | **0.956** | 0.940 | 0.901 | **0.924** | 0.910 |

`seqlen4` is best on the symbolic probe, especially on larger seen mazes.

## 5. Corrected Latent L2 Baseline

Implementation:

- Uses frozen LeWM projector embeddings directly.
- `model_free_greedy`: scores true next-state latents by L2 distance to goal.
- `predictor_greedy`: scores predictor-generated next latents by L2 distance.
- `cem_l2`: CEM plans action sequences in predictor latent space, scoring final
  latent by L2 distance to goal.
- Full 900 held-out eval tasks, sizes 9-25.

| Backbone | model_free SR/SPL | predictor SR/SPL | CEM SR/SPL |
| --- | ---: | ---: | ---: |
| seqlen2 | **0.648 / 0.555** | **0.609 / 0.506** | **0.644 / 0.512** |
| seqlen4 | 0.641 / 0.530 | 0.594 / 0.470 | 0.629 / 0.445 |
| seqlen8 | 0.632 / 0.518 | 0.589 / 0.460 | 0.617 / 0.441 |

Although `seqlen4` is best for symbolic decoding, `seqlen2` is best for latent
L2 navigation. This suggests that one-step training improves the local latent
geometry used by L2 planning more than it improves symbolic position decoding.

## 6. Simple DistanceHead

Implementation:

- Frozen LeWM backbone.
- Head input: `(z_current, z_goal)` projector embeddings.
- Head architecture: MLP `512 -> 256 -> 128 -> scalar`, `softplus` output.
- Supervision: BFS shortest-path distance for sampled reachable state pairs.
- Target transform: `log1p(dist) / log1p(max_dist_per_maze)`.
- Loss: `smooth_l1`.
- Steps: 30000.
- Batch size: 512.
- No ranking, no action CE, no backbone unfreezing.

Training summary:

| Backbone | first eval_seen | final eval_seen | final eval_ood | best eval_seen |
| --- | ---: | ---: | ---: | ---: |
| seqlen2 | 0.0093 | **0.0085** | **0.0142** | **0.00815** |
| seqlen4 | 0.0107 | 0.0097 | 0.0214 | 0.00920 |
| seqlen8 | 0.0107 | 0.0096 | 0.0172 | 0.00934 |

Navigation results:

| Backbone | model_free SR/SPL | predictor SR/SPL |
| --- | ---: | ---: |
| seqlen2 | **0.648 / 0.564** | **0.636 / 0.535** |
| seqlen4 | 0.643 / 0.552 | 0.610 / 0.483 |
| seqlen8 | 0.632 / 0.545 | 0.633 / 0.498 |

Simple DistanceHead does not clearly outperform latent L2. It slightly improves
SPL in some settings but does not produce the expected large gain from BFS
supervision.

## 7. QRL Frozen Metric Head

Implementation:

- Frozen LeWM backbone.
- QRL head is asymmetric: separate source/target projections, then an MLP.
- Losses:
  - BFS regression with `log_norm` targets, weight 0.5;
  - local valid-action ranking, weight 2.0;
  - contrastive ordering, weight 1.0;
  - triangle inequality regularization, weight 0.05.
- Steps: 30000.
- Batch size: 512.

Training summary:

| Backbone | final val_reg | final val_rank_acc | best val_rank_acc |
| --- | ---: | ---: | ---: |
| seqlen2 | **0.0106** | **0.6440** | **0.6538** |
| seqlen4 | 0.0125 | 0.6316 | 0.6353 |
| seqlen8 | 0.0124 | 0.6333 | 0.6372 |

Navigation results from completed eval logs:

| Backbone | model_free SR/SPL | predictor SR/SPL |
| --- | ---: | ---: |
| seqlen2 | 0.643 / **0.554** | **0.626 / 0.522** |
| seqlen4 | **0.649 / 0.542** | 0.610 / 0.476 |
| seqlen8 | 0.621 / 0.523 | 0.617 / 0.487 |

Note: the QRL eval command saved all three runs to the same default output path
(`results/set_b_multisize/qrl_v2_eval.json`), so per-size QRL JSON was
overwritten. The final SR/SPL values above come from the preserved log files.

## 8. BC and RL References

These baselines were run outside the seqlen ablation, on the same Set B
train/eval split. They are not affected by the metric-head planner bug. The
latest BC/RL numbers below use the full 900-task Set B evaluation unless noted.

RL setup for the strongest runs:

- Training manifest: `data/splits/unisize_train_manifest.jsonl`
- Eval manifest: `data/splits/unisize_eval_manifest.jsonl`
- Train sizes: 9-21, 400 held-in topologies/tasks per size
- Eval sizes: 9-25, 100 held-out topologies/tasks per size
- Offline replay enumerates ground-truth one-step transitions from maze states.
- Reward target is BFS dense reward:
  closer-to-goal bonus, away/wall/stay penalties, and terminal goal bonus.
- The most stable objective uses reward-derived action labels:
  `argmax_a r(s, g, a)`, optimized with action CE.
- TD bootstrapping is logged but disabled in the final stable runs
  (`td_loss_coef=0.0`), because pure TD targets were observed to inflate Q
  scale during long offline training.

| Method | Eval size/count | Overall SR | Notes |
| --- | --- | ---: | --- |
| BC CNN | 900 tasks, 100/size | 0.781 | Full Set B BC reference, same evaluator as RL |
| BC-policy init + offline reward-ranking RL | 900 tasks, 100/size | **0.799** | Full BC policy initialized, action CE from BFS reward, KL to BC teacher |
| BC-policy init + offline reward-ranking RL | 900 tasks, 100/size | 0.794 | Same without final KL-regularized run |
| BC-encoder + offline reward-ranking RL | 900 tasks, 100/size | 0.662 | Loads BC encoder only; trains new Q/action head |
| BC | 270 tasks, 30/size | 0.770 | Smaller eval sample |
| RL dense | 270 tasks, 30/size | 0.004 | Failed to learn robust policy |
| RL sparse | 270 tasks, 30/size | 0.015 | Failed to learn robust policy |
| symbolic BFS probe | sizes 9-21, 100/size | 0.956 mean SR | Best seqlen4, seen sizes only |

Per-size comparison between the full Set B BC reference and the first
BC-policy-initialized RL run:

| Size | BC CNN SR | BC-policy RL SR | Delta |
| ---: | ---: | ---: | ---: |
| 9 | 0.99 | 1.00 | +0.01 |
| 11 | 1.00 | 0.99 | -0.01 |
| 13 | 0.98 | 0.99 | +0.01 |
| 15 | 0.93 | 0.95 | +0.02 |
| 17 | 0.87 | 0.90 | +0.03 |
| 19 | 0.80 | 0.84 | +0.04 |
| 21 | 0.55 | 0.55 | 0.00 |
| 23 (OOD) | 0.58 | 0.61 | +0.03 |
| 25 (OOD) | 0.33 | 0.32 | -0.01 |
| Overall | 0.781 | 0.794 | +0.013 |

BC-policy-initialized offline RL recovers BC-level performance and gives a
small overall improvement, but the gain is not large enough to change the main
conclusion. It does not solve the hardest OOD size-25 cases. BC-encoder-only RL
is much weaker, showing that the BC action head contains important reusable
navigation structure, not just the convolutional features.

BC CNN and BC-policy-initialized RL remain substantially stronger than current
LeWM metric-planning variants on full Set B.

## 9. Main Conclusions

1. The corrected planner removes the earlier invalid/stuck artifact. Old CEM
   numbers with high invalid rates should not be used.

2. `seqlen4` is best for symbolic position decodability, but `seqlen2` is best
   for latent L2 and simple DistanceHead navigation. Lower LeWM training loss
   from `seqlen8` does not imply better navigation.

3. Simple BFS-regression DistanceHead learns a clean scalar target, but it does
   not reliably improve local action ordering beyond latent L2. This explains
   why lower regression loss does not yield much higher SR.

4. QRL improves model-free SR slightly for `seqlen4`, but the gain is small and
   predictor-greedy remains close to L2/DH. The bottleneck is likely alignment
   between learned metric scores and local action ranking under predictor
   rollouts.

5. Current best LeWM metric-planning SR is around 0.65 on full 900-task Set B,
   below BC CNN at 0.781 and BC-policy-initialized offline RL at 0.799.
   Reaching >0.8 likely needs either a stronger planner objective directly
   optimized for local action selection, or a better predictor/latent dynamics
   objective, not only scalar distance regression.

6. Ground-truth reward-ranking offline RL is not a missing ingredient by
   itself. When initialized from the full BC policy, it improves Set B SR only
   slightly (0.781 -> 0.799). When initialized from the BC encoder only, it
   reaches only 0.662. This suggests that the remaining Set B failure is mainly
   an OOD policy-generalization / inductive-bias issue, not lack of dense
   reward supervision.

## 10. Recommended Next Experiments

1. Use `seqlen2` as the main backbone for predictor-aligned planning, because
   it gives the best latent L2 and simple DH predictor-greedy SR.

2. Parameterize planner `history_size` and compare:
   - `seqlen2 + history_size=1`
   - `seqlen4 + history_size=3`
   - `seqlen8 + history_size=3`

3. Train an action-ranking head directly on valid local actions:
   `score(z_next_action, z_goal)` with CE over optimal BFS-improving actions.
   This more directly targets navigation than global pairwise distance
   regression.

4. If QRL is kept, save QRL eval JSON with unique output paths and include
   per-size breakdown in the next report.

5. Treat BC-policy-initialized offline RL as a diagnostic baseline rather than
   a primary path to better OOD generalization. It is useful because it shows
   that BFS reward-ranking can recover BC-level behavior, but further gains
   likely require different inductive bias, such as fully convolutional value
   maps, explicit coordinate/spatial auxiliary heads, or planning over learned
   local value fields.

## 11. Source Artifacts

Primary results:

- `results/setb_seqlen_ablation_20260708/summary_metrics.json`
- `results/setb_seqlen_ablation_20260708/latent_l2_all_full900_seqlen{2,4,8}.json`
- `results/setb_seqlen_ablation_20260708/dh_simple_greedy_full900_seqlen{2,4,8}.json`
- `results/setb_seqlen_ablation_20260708/symbolic_bfs_probe_seqlen{2,4,8}.json`
- `logs/setb_seqlen_ablation_20260708/qrl_v2_frozen_greedy_full900_seqlen{2,4,8}.log`

Reference RL/BC runs discussed in Section 8:

- `results/set_b_multisize/bc_cnn_policy.pt`
- `results/set_b_multisize/gcrl_her_bcpolicy_policy.pt`
- `results/set_b_multisize/gcrl_her_bcpolicy_results.json`
- `results/set_b_multisize/bc_cnn_eval900_via_gcrl_evaluator.json`

Experiment checkpoints were used during the run but are intentionally not
included in the cleaned repository:

- `checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen{2,4,8}.pt`
- `checkpoints/metric_heads/distance_head_simple_setb_seqlen_ablation_20260708_seqlen{2,4,8}.pt`
- `checkpoints/metric_heads/qrl_v2_frozen_setb_seqlen_ablation_20260708_seqlen{2,4,8}.pt`
