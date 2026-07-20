# AIR-JEPA Stage 0：Shared Workspace

[English](README.md) | [中文](README.zh.md)

实验 ID：`procgen-maze-air0-workspace-v1`

## 一句话说明

冻结原来已经训练好的 Spatial-JEPA encoder/projector，把它输出的整张空间 latent
map 交给一个参数共享、可重复运行 K 次的 reasoning workspace。workspace 同时维护
goal、四个 action 和四个 future token，预测四个动作后的 latent field，再用同一个
distributional cost head 给四个候选动作排序。

本阶段不是要证明最终架构已经完成，而是判断这条新数据流是否值得成为后续主干。
正式执行固定为 135 个科学 cells 加 8 个审计/编排门禁，共 143 jobs；runner 会重建
并逐字段验证 DAG。缺少签名的 4×H800、128-batch 配对或 compute-match L0 证据时，
L1/L2/L3 均拒绝生成。

## 唯一科学处理差异

`AIR0-direct` 与 `AIR0-jepa` 的网络结构、初始化、训练 map/state、K 序列、optimizer、
训练步数和评测完全一致。唯一差异是 loss 权重：

| Method | action | future latent | distributional cost |
|---|---:|---:|---:|
| `air0_direct` | 1.0 | 0.0 | 0.0 |
| `air0_jepa` | 1.0 | 1.0 | 0.5 |

因此这项对照估计的是“显式 future + cost 监督的增量作用”，不是参数量差异。

## 数据流

```text
Procgen observation [H,W,5]
        |
        v
frozen Spatial-JEPA encoder + planning projector
        |
        v
spatial latent Z [64,H,W]
        |
        +--> immutable recall at every iteration
        v
shared Reasoner Block repeated K times
  local 4-neighbor attention
  <-> goal/action/future workspace tokens
        |
        v
four action-conditioned future fields
        |
        v
129-bin cost distribution for each action
        |
        v
argmin expected cost, execute one action, observe again, replan
```

主路径不能读取 wall mask、valid-action mask 或 BFS。BFS 只产生训练标签和离线诊断。
撞墙动作按照环境真实语义成为 no-op successor，不会从候选集中消失。

## 必须回答的问题

1. `AIR0-jepa` 的 unmasked full-900 SR/OOD 是否接近强基线 `j1-receding`？
2. 相同结构下，future/cost 监督是否优于 `AIR0-direct`？
3. K 从 1 增至 128 时，OOD 和长路径是否系统改善？
4. true-future、copy、permutation 和 zero 干预能否证明模型真的使用 future？
5. 若失败，错误来自 future prediction 还是 future-to-energy ranking？

## 目录

| 路径 | 用途 |
|---|---|
| `docs/EXPERIMENT_PLAN.zh.md` | 完整、锁定的科学设计 |
| `docs/ENGINEER_RUNBOOK.zh.md` | 工程师唯一正式运行入口 |
| `docs/SERVER_INPUTS.zh.md` | 服务器必须已有的 source artifacts |
| `docs/COMPATIBILITY.zh.md` | 与原 Spatial-JEPA 代码和 checkpoint 的接口合同 |
| `docs/RESULT_SCHEMA.zh.md` | artifact、证据角色和统计输出解释 |
| `docs/VALIDATION.zh.md` | 分层验证与验收命令 |
| `docs/IMPLEMENTATION_AUDIT.zh.md` | 代码-计划对应、已验证项与服务器检查边界 |
| `docs/REPLACEMENT_PROTOCOL.zh.md` | 客观技术失败后的整次 attempt 封存与重启规则 |
| `configs/default.json` | 唯一正式配置；关键值由 Pydantic 二次硬锁 |
| `configs/protocol_lock.json` | manifest、数据角色和完整科学矩阵锁 |
| `configs/package_lock.json` | 代码、文档和依赖文件内容锁 |
| `manifests/` | AIR_dev、early210 及封存的后续 split |
| `models.py` | shared workspace、future decoder、energy head |
| `train.py` | 两个配对 AIR 方法的训练入口 |
| `evaluate.py` | AIR 与 J0/J1 bridge 的统一评测器 |
| `evaluate_oracle.py` | AIR_dev full-900 的 exact-BFS step-cap 上限 |
| `diagnose.py` | full-900 local future/energy 因果诊断 |
| `plan_jobs.py` / `run_jobs.py` | 分数无关的完整 DAG 与四 GPU 调度器 |
| `summarize.py` | L1/L2/L3 释放、完整性审计、compute 表与 crossed bootstrap |

## 工程师可以与不可以决定什么

可以决定：如何创建虚拟环境、用 `tmux` 还是作业系统保持进程、日志和大 artifact
放在哪个物理磁盘，以及如何修复明确的硬件/驱动/权限故障。若改变物理路径，只能在
配置规定的位置做 mount/symlink，不能修改 locked config。

不可以决定：删减 seed/K/task、改变 batch/loss/训练步数、把 OOM 通过减小 batch
“修好”、按分数追加训练、改用 best checkpoint、跳过历史 bridge、提前打开
`AIR_select/AIR_final`，或把 diagnostic 当主结果。

任何不确定的 source 文件、checkpoint metadata 或旧结果缺失都必须先停在 L0，按
[`docs/SERVER_INPUTS.zh.md`](docs/SERVER_INPUTS.zh.md) 核对，不能猜。
