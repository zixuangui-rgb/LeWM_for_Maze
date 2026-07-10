# Spatial-JEPA Iterative Planning

这是 Maze-JEPA diagnostics 和 `planning_repair` 之后的确认性实验包。它检验：

1. 增加可重复局部计算的 iterative system 是否稳定改善 raw-grid planning；
2. full-resolution Spatial-JEPA 在相同 planner family 下是否不劣于 raw input；
3. 保留 map information 的 staged adaptation 是否优于 frozen representation。

旧 Set-B eval900 已参与方法开发，现在只作为 development。最终结论使用新生成且
预注册的 900-task confirmatory topology hold-out。绝对能力以完全 `unmasked`
action selection 为准，`model_valid` 和 oracle `corrected` 仅用于诊断帮助量。

## 目录

```text
spatial_jepa_planning/
  models.py                         # Spatial-JEPA、FF/recurrent planner、oracle VI
  losses.py                         # tie-aware CE、Bellman、gap、map/collapse losses
  common.py                         # labels、RNG streams、hash、runtime、metrics
  train.py                          # representation / planner / joint training
  evaluate.py                       # learned、decoded BFS、exact BFS、oracle VI
  generate_confirmatory_manifest.py # 确认集确定性生成与逐字节验证
  audit_protocol.py                 # 4600-task 三路 split 与配置审计
  run_plan.py                       # 534-command formal matrix 编排
  summarize.py                      # 十 seed crossed bootstrap 与严格结论判定
  smoke_test.py                     # CPU integration smoke test
  configs/default.json              # 模型、K、seed、hypothesis 预注册
  configs/protocol_lock.json        # 三个 manifest 与评估上限锁
  DESIGN.md                         # 架构、loss、实验问题
  ALIGNMENT_PROTOCOL.md             # 正式比较和推断合同
  RUNBOOK.md                        # 服务器运行步骤
```

## 安装与预检

```bash
cd LeWM_for_Maze
pip install -e '.[dev]'

python spatial_jepa_planning/generate_confirmatory_manifest.py --check
python spatial_jepa_planning/audit_protocol.py
python spatial_jepa_planning/smoke_test.py
python -m pytest -q tests/test_spatial_jepa_planning.py
python spatial_jepa_planning/run_plan.py --stages full --dry-run
```

审计必须确认：train/development/confirmatory 为 2800/900/900；三层 overlap
全为 0；确认集与 generator 完全一致；10 个 training seeds；534 个唯一 output。

默认 `device=auto`，有 CUDA 时使用 CUDA，否则 CPU。多卡可显式传
`--device cuda:N`。正式矩阵要求所有 seed 使用一致的 PyTorch/CUDA/cuDNN 和
GPU 型号。

## 正式运行

在 clean Git commit、无旧输出的条件下：

```bash
python spatial_jepa_planning/run_plan.py --stages full \
  2>&1 | tee logs/spatial_jepa_confirm_v2.log
```

也可以按阶段执行：

```bash
# development representation gate
python spatial_jepa_planning/run_plan.py \
  --stages train_representations,eval_representations

# planner training and development diagnostics
python spatial_jepa_planning/run_plan.py \
  --stages train_planners,eval_planners_development

# 一次性 confirmatory evaluation
python spatial_jepa_planning/run_plan.py \
  --stages anchors,eval_representations_confirmatory,eval_planners,summary
```

如果 development 后修改任何代码或配置，必须提交并从头重训全部 formal
checkpoints，再运行确认集。不要把修改前后的 checkpoint 混用。

正式 CLI 会拒绝 dirty worktree、非确定性训练、未锁 spec、覆盖旧文件和非有限
数值。checkpoint/result 保存 training spec、analysis spec、训练 Git 状态、代码
fingerprint、manifest/checkpoint/source-representation SHA、完整 runtime、参数量和
inference MACs；summary 还会拒绝由 dirty worktree 训练出的 checkpoint。

## 实验矩阵

- R0：raw value-only feedforward；
- R1：raw all-state tie-aware action CE；
- R2：raw CE + value + valid + Bellman + action gap；
- R2D：同 loss 的 full-receptive-field dilated feedforward；
- R3：fixed K=64 recurrent；
- R4：progressive recurrent，primary K=128；
- J0：Spatial-JEPA + dilated feedforward；
- J1：Spatial-JEPA + frozen recurrent；
- J2：last block 0.1x LR + map-preservation recurrent；
- J3：joint JEPA/planner + gradient-conflict audit。

K train 为 `{4,8,16,32,64,128}`，test 另含 256。K curve 只回答 test-time
compute scaling，不能用于在确认集挑最好 K。报告分别给 planner、representation
planning path 与总计的参数量/size-25 GMACs，并逐 size 报告 SR。

## 结果与结论边界

结果位于：

```text
checkpoints/spatial_jepa_planning/<variant>_seed<seed>.pt
spatial_jepa_planning_runs/<variant>/seed<seed>/*.json
spatial_jepa_planning_runs/anchors/*.json
spatial_jepa_planning_runs/summary.{md,json}
```

确认集的 128-step oracle ceiling 为 `881/900 = 0.9788888889`。summary 默认
要求十 seed、全部 variants、三种 action protocol 和 900 个唯一 task rows 全齐，
并使用 seed×task crossed paired bootstrap 与三假设 Bonferroni simultaneous CI。

本实验不能声称超过旧 BC/latent-L2/原始 LeWM，因为它们没有在新确认集上按同一
evaluator 重跑；也不能推出纹理、背景或跨任务泛化。decoded-map BFS 证明地图
信息可恢复，不等于 learned planner 自己完成 BFS。

正式合同见 [ALIGNMENT_PROTOCOL.md](ALIGNMENT_PROTOCOL.md)，服务器操作见
[RUNBOOK.md](RUNBOOK.md)，架构与 loss 见 [DESIGN.md](DESIGN.md)。
