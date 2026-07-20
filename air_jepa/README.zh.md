# AIR-JEPA 实验系列

[English](README.md) | [中文](README.zh.md)

`air_jepa/` 是激进 JEPA 推理架构的独立实验系列。每个阶段必须放在单独子目录中，
拥有自己的实验 ID、协议锁、代码、manifest、测试和结论边界；后续阶段不得覆盖前一
阶段的 artifact 或用新规则重解释旧结果。

## 阶段目录

| 目录 | 实验 ID | 研究问题 | 状态 |
|---|---|---|---|
| `stage0_workspace/` | `procgen-maze-air0-workspace-v1` | 共享循环 workspace 能否把 action-conditioned future、cost 与迭代推理接成可验证的数据流 | 代码与协议包 |

“代码与协议包”不代表已经得到实验结果。正式结果只存在于服务器运行后生成的
`air_jepa_runs/` 中，并且只有 `L3 FINAL_CLOSURE` 可以关闭一个阶段。

## 系列规则

1. 复用已经验证的 Procgen Maze 环境、topology hold-out 和 Spatial-JEPA source
   checkpoints，不复制或修改环境语义。
2. 每阶段先冻结问题、对照、数据角色、seed、预算、统计和门槛，再生成完整 job DAG。
3. quicklook 只加快读出，不允许改变仍在运行的矩阵。
4. `corrected`、true-future 等 assistance/oracle 结果永远不能进入绝对能力主表。
5. 新阶段必须使用新的实验 ID 和尚未打开的数据角色。

第一阶段的工程入口见
[`stage0_workspace/README.zh.md`](stage0_workspace/README.zh.md)。
