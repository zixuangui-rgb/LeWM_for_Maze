# Maze-JEPA Planning Repair Experiments

这个目录是下一轮 Maze-JEPA 实验包，目标不是再比较一个新的 scorer，而是系统修复诊断报告中的三堵墙：

1. 表征/动作墙：`embedding optimal_action probe ~= 0.33`，`Local top-1 ~= 0.60`。
2. projector 信息墙：位置、goal、局部墙结构从 `spatial -> embedding` 单调劣化。
3. rollout 漂移墙：teacher-forced 稳定，但 closed-loop `h=10 nn_bfs_error ~= 9.7`。

所有脚本都保持和旧代码兼容：

- backbone 仍然是 `scripts.train.train_dim256.Unisize256`；
- checkpoint 仍保存 `model_config` 和 `model_state_dict`，可直接给 `diagnostics/run_all.py` 使用；
- 新增 aux heads / prefix predictor 单独保存在同一个 checkpoint 里，不污染旧 diagnostics loader。

## 文件结构

```text
planning_repair/
  common.py                    # 数据、BFS 标签、旧 checkpoint 兼容加载
  heads.py                     # embedding aux heads + action-prefix predictor
  train_planning_aligned.py    # P1/P1.5/P2 训练入口
  eval_b2_receding.py          # P0: 短 horizon receding CEM
  eval_aux_action_head.py      # A1/A3: embedding action head model-free eval
  eval_prefix_planner.py       # P2: action-prefix planner eval
  run_plan.py                  # JSON 配置驱动的一键编排
  configs/default.json         # 默认实验配置
  DESIGN.md                    # 实验设计和文献依据
```

## 推荐执行顺序

运行前先激活项目实际使用的 conda/venv。`run_plan.py` 会用启动它的同一个 Python 解释器去调用后续脚本，所以不要用缺少 `torch/gymnasium/omegaconf` 的系统 Python 启动总控脚本。

同时确认 `planning_repair/configs/default.json` 中的 `baseline_ckpt` 指向服务器上实际存在的 backbone checkpoint。如果文件名不同，只需要改配置或命令行里的 `--init-model-ckpt / --model-ckpt`，不需要改代码。

### P0：先跑 quick win，不重训

验证缩短 horizon + 每步 replan 能否缓解 rollout 漂移。

```bash
python planning_repair/eval_b2_receding.py \
  --model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen4.pt \
  --horizons 3,5,8,12 \
  --scorers latent_l2 \
  --output planning_repair_runs/p0_receding/results.json \
  --device cuda
```

快速 smoke test：

```bash
python planning_repair/eval_b2_receding.py \
  --model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen4.pt \
  --horizons 3,5 \
  --limit 4 \
  --num-candidates 8 \
  --device cpu
```

### P1/P1.5/P2：训练 planning-aligned checkpoint

默认同时训练：

- embedding 层 agent/goal xy；
- valid-action mask；
- optimal-action listwise head；
- BFS distance norm；
- budget-conditioned reachability；
- action-prefix multi-horizon predictor。

```bash
python planning_repair/train_planning_aligned.py \
  --init-model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen4.pt \
  --output checkpoints/planning_repair/planning_aligned_seqlen8.pt \
  --steps 30000 \
  --batch-size 256 \
  --seq-len 8 \
  --lambda-prefix 0.2 \
  --prefix-horizon 5 \
  --device cuda
```

快速 smoke test：

```bash
python planning_repair/train_planning_aligned.py \
  --init-model-ckpt checkpoints/backbones/unisize_dim256_setb_seqlen_ablation_20260708_seqlen4.pt \
  --output /tmp/planning_repair_smoke.pt \
  --steps 2 \
  --batch-size 2 \
  --seq-len 4 \
  --lambda-prefix 0.1 \
  --device cpu
```

### 诊断新 checkpoint

```bash
python diagnostics/run_all.py \
  --model-ckpt checkpoints/planning_repair/planning_aligned_seqlen8.pt \
  --run-id planning_repair_seqlen8 \
  --device cuda
```

重点看：

- `embedding optimal_action` 是否从 `0.341` 上升；
- `embedding valid_action` 是否从 `0.406` 上升；
- `goal_x/goal_y RMSE` 是否下降；
- `Local top-1 / Local margin` 是否上升；
- closed-loop `nn_bfs_error` 是否下降；
- failure taxonomy 中 `metric_wrong / predictor_wrong / loop_or_cycle` 是否下降。

### A1/A3 专门评估：aux action head

```bash
python planning_repair/eval_aux_action_head.py \
  --model-ckpt checkpoints/planning_repair/planning_aligned_seqlen8.pt \
  --output planning_repair_runs/aux_action_head/results.json \
  --device cuda
```

这一步回答：embedding 层动作排序是否已经足以做 model-free greedy。

### P2 专门评估：action-prefix planner

```bash
python planning_repair/eval_prefix_planner.py \
  --model-ckpt checkpoints/planning_repair/planning_aligned_seqlen8.pt \
  --horizon 5 \
  --num-candidates 128 \
  --score-all-prefixes \
  --terminal-scorer latent_l2 \
  --output planning_repair_runs/prefix_planner/results.json \
  --device cuda
```

如果 aux BFS head 已明显变好，也可以试：

```bash
python planning_repair/eval_prefix_planner.py \
  --model-ckpt checkpoints/planning_repair/planning_aligned_seqlen8.pt \
  --terminal-scorer aux_bfs \
  --score-all-prefixes \
  --device cuda
```

## 一键运行

先检查配置：

```bash
python planning_repair/run_plan.py \
  --config planning_repair/configs/default.json \
  --stages p0,p1,diagnostics,aux_eval,prefix_eval \
  --dry-run
```

真正运行：

```bash
python planning_repair/run_plan.py \
  --config planning_repair/configs/default.json \
  --stages p0,p1,diagnostics,aux_eval,prefix_eval
```

## 科学对照要求

每个新 checkpoint 至少保存以下结果：

```text
diagnostics_runs/<run_id>/diagnostic_report.md
planning_repair_runs/p0_receding/results.json
planning_repair_runs/aux_action_head/results.json
planning_repair_runs/prefix_planner/results.json
```

对照基线固定为当前诊断报告中的 `seqlen4_full`：

| 指标 | 当前基线 |
|---|---:|
| embedding optimal_action | 0.341 |
| embedding valid_action | 0.406 |
| Local top-1 | 0.588-0.598 |
| closed-loop h=10 nn_bfs_error | 9.71 |
| failure_taxonomy SR | 0.612 |
| seen / OOD SR | 0.683 / 0.365 |

如果 SR 上升但上述诊断指标不动，不能声称修复了 JEPA 表征，只能说 planner 工程绕过了部分失败。
