# AIR-JEPA Stage 0 服务器输入清单

这些文件来自已经完成的 Spatial-JEPA 实验，不由 AIR0 重新训练。它们是严格输入，
不是工程师可替换的“同类 checkpoint”。

## 1. 每个 seed 的五个文件

对 `seed = 42,43,44`，必须存在：

```text
checkpoints/spatial_jepa_planning/spatial_info_sigreg_seed{seed}.pt
checkpoints/spatial_jepa_planning/j0_spatial_feedforward_seed{seed}.pt
checkpoints/spatial_jepa_planning/j1_spatial_iterative_frozen_seed{seed}.pt
spatial_jepa_planning_runs/j0_spatial_feedforward/seed{seed}/confirmatory_unmasked.json
spatial_jepa_planning_runs/j1_spatial_iterative_frozen/seed{seed}/confirmatory_unmasked.json
```

共 15 个文件。`# CHECK-REQUIRED`：工程师必须确认这些是最初正式复现的 final-step
artifacts，而不是临时 checkpoint、best checkpoint、开发集结果或 corrected 结果。

## 2. 不要修改 locked config

若文件在其他磁盘，优先使用 bind mount 或 symlink 映射到上述精确路径。不要编辑
`configs/default.json`；编辑后 package/protocol lock 会失败，也会失去严格对照。

## 3. 自动检查内容

`lock_sources.py --write` 会验证：

- `experiment_family == spatial_jepa_planning`；
- `format_version == 2`；
- representation stage 与 planner stage 正确；
- `protocol.seed` 与文件 seed 一致；
- train/development/confirmatory manifest SHA256 与锁定数据完全一致；
- source checkpoint 必须记录 clean Git provenance 与 code fingerprint；
- `training_accounting.optimizer_steps == 30000`；
- J0 是 `feedforward_dilated`，J1 是 `iterative`；
- `input_mode == spatial_jepa` 且 `encoder_mode == frozen`；
- 两个 planner 内嵌的 representation state hash 与对应 representation checkpoint 相同。
- representation、J0、J1 以及三个 seeds 的 `analysis_spec_sha256` 与 source-code
  fingerprint 必须一致；
- J1 必须包含一致的 representation/planner/total 参数量，以及 size-21/25 下包含 K128
  且可验证为 affine 的 Conv-MAC 曲线。

随后 L0 会在历史 confirmatory 900 tasks 上重新执行 `j0-static` 与 `j1-static`，并
逐 task 比较 success、steps、invalid、loop、final distance 和 SPL。任何差异都会阻断
AIR 训练，防止新 evaluator 改变旧基线语义。

## 4. 缺失时如何处理

缺 checkpoint：从原实验服务器或归档恢复 exact file；不要重新训练一个“差不多”的。

缺旧 result JSON：用原 commit、原 evaluator 和原正式 checkpoint 恢复历史 artifact，
并保留恢复命令和 hash；不要用 AIR evaluator 先生成文件再冒充历史结果。

metadata 不符或 tensor hash 不同：停止并报告。这代表输入不是协议所指的模型，不能
由工程师自行放宽校验。
