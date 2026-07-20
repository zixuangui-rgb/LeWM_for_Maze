# AIR-JEPA Stage 0 验证与验收

验证分为五层。后一层不能替代前一层。

## V1：静态质量

```bash
uv run ruff check air_jepa tests/test_air_jepa_stage0.py
uv run python -m compileall -q air_jepa
```

检查 import、未定义变量、格式和语法。

## V2：AIR 单元与性质测试

```bash
uv run python -m pytest -q tests/test_air_jepa_stage0.py
```

覆盖 config 硬锁、共享参数、K 读出、padding mask、tie loss、cost clipping、future
copy baseline、两方法反向传播、真实 successor/no-op、配对 RNG、progressive K、manifest
重建、sealed role、143-job DAG、BFS oracle、crossed bootstrap、source tensor identity、
distance calibration 重算、动作轨迹计数、累计 K 日志、
final checkpoint、release row/aggregate 复算、paired checkpoint 审计、compute accounting
和 deterministic local states。还会主动篡改 formal cell、job plan、L0 H800/pairing
audit 与 compute-match lock，确认 fail-closed 路径真实生效；并覆盖递归依赖 fingerprint、
L0-prefix/checkpoint 绑定、source lineage、progressive-K phase、正式 checkpoint role、
静态/闭环 inference-call 语义与原子 checkpoint 故障注入。

## V3：仓库回归

```bash
uv run python -m pytest tests distance_head_study/tests a1_quick_validation/tests -q
```

必须确认 AIR 新包没有破坏原 diagnostics、Spatial-JEPA、closure 和 planner frontier。

## V4：CPU 数值 smoke

```bash
uv run python -m air_jepa.stage0_workspace.smoke_test
```

使用真实 train manifest、随机冻结 SpatialRepresentation 和两个 matched AIR loss，执行
successor 编码、forward/backward、finite gradient 与 future permutation。它不产生研究
分数，也不能替代 source checkpoint 集成测试。

## V5：服务器正式 L0

```bash
uv run python -m air_jepa.stage0_workspace.lock_sources --check
uv run python -m air_jepa.stage0_workspace.audit_protocol
uv run python -m air_jepa.stage0_workspace.benchmark --device cuda:2
```

上述三条是门禁内容说明。正式执行时应由签名 job plan 和唯一 `run_jobs` 调度器创建
对应输出；不要在 DAG 已生成后手工重复运行，否则 immutable output 检查会按设计拒绝。

正式 L0 才能验证四张 H800、真实 source metadata/tensor hashes、历史 J0/J1 exact
behavioral parity、1000-step smoke train、K128 throughput 和显存。L0 通过前，任何 AIR
性能输出都不是正式结果。

## 验收不变量

- repository/package/protocol/source/job-plan signatures 全部匹配；
- formal artifacts 来自 clean worktree，同一 runtime 与四张同构 H800；
- direct/JEPA 每 seed 初始化 hash、sample stream hash 与 target moments 成对一致；
- 六个 AIR checkpoints 都是 step 30000；
- full/early row count 分别为 900/210，无缺失重复；
- primary 只用 unmasked normal K128；
- 143/143 jobs complete，L3 release 签名有效；
- `air_select`、`air_final` 无任何 result path。
