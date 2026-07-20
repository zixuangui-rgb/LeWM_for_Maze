# 验证和审计范围

## L0：静态规范

- Ruff format/check；
- Python compileall；
- JSON 可解析；
- 不修改 `distance_head_study` 锁定源文件；
- 新 run root 已加入 `.gitignore`。

## L1：方法和配置

- quick config 可由原 `StudyConfig` 严格解析；
- splits/seeds/planner/training/analysis 与原 config 完全相同；
- 方法表恰好七个方法；
- 三个 reference resolved spec 和 method hash 与原实验相同；
- 四个新方法均为静态直接父节点，无 dynamic decision parent；
- declared diff 与实际 diff 一致；
- 所有新方法冻结 backbone，且不使用 test BFS。

## L2：协议与证据隔离

- 原 `distance_head_study/configs/protocol_lock.json` 仍可完整 regenerate；
- quick 内层 protocol lock 可 regenerate；
- package lock 覆盖新目录除自身外的代码、文档、配置和内层 lock；
- 修改任一 package 文件都会使正式入口失败；
- manifests/hash/topology hold-out 与原实验一致；
- outputs、decisions、checkpoints 使用独立命名空间。

## L3：桥接器

- cache 重绑定不改变 record、task hash 和 shard hash；
- source index 或 shard 改变后验证失败；
- native quick cache 也必须通过完整 binding/shard 验证；
- reference checkpoint 新旧 method hash 必须相同；
- candidate actions 必须逐元素相同；
- head state hash 写入前后相同；
- 不允许新方法使用 checkpoint import。

## L4：选择逻辑

- Bellman 的 OR gate、predicted 的 AND gate、reachability 四指标 gate 分别测试边界；
- horizon control 永不晋级；
- Q1 最多两个新方法；
- Q2 要求两个 head seed 方向不为负且平均增益达标；
- SPL/unmasked safety gate 生效；
- Q3 winner 必须在 full-900 前锁定；
- paired rows task 集不一致时失败；
- tie-break 不依赖文件系统顺序。

## L5：作业执行

- plan signature 和 package lock 绑定；
- job ID 唯一，worker 范围正确；
- command 必须经过 `a1_quick_validation.run`；
- `{device}` 只做单值替换，不进入 shell；
- completion seal 绑定 plan、commands 和 logs；
- phase matrix 外的 method/split/seed/protocol 被 gateway 拒绝。

## L6：旧实验回归

运行：

```bash
.venv/bin/python -m pytest tests distance_head_study/tests a1_quick_validation/tests -q
.venv/bin/python -m distance_head_study.audit_protocol
```

新增包不能使旧测试或原协议 audit 失败。

## L7：服务器真实产物门

本地仓库没有 H800 checkpoint/cache，因此以下项目必须在服务器 Q0 完成：

- seed42 backbone payload 验证；
- source cache 全 shard hash；
- source reference checkpoint dependency chain；
- source/quick candidate action tensor equality；
- formal CUDA forward/train/diagnose/evaluate I/O；
- 四 worker GPU 映射。

在 L7 完成前，只能声明代码和协议通过本地审计，不能声明真实 GPU 实验已运行。
