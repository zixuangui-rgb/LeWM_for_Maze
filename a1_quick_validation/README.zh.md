# A1 DistanceHead 快速验证实验

本目录是在 `distance_head_study` 已锁定实验之上建立的独立快速验证包。它只回答一个问题：在完全保持 Vector-JEPA backbone、训练预算和 CEM planner 不变时，围绕当前最优 `a1_log` 加入 Bellman、一阶 predicted-latent 排序或 reachability 监督，是否有值得继续投入的稳定信号。

这不是论文级确认实验。它采用一套预注册、可停机的漏斗：`Q1 D_screen(140) -> Q2 D_select(210) -> Q3 legacy full-900`。只有通过前一阶段机制门和性能门的方法才能进入下一阶段；Q3 也仍然只是单个历史 backbone 上的探索性 full-900 结果。

## 目录

- `docs/EXPERIMENT_DESIGN.zh.md`：完整科学设计、假设、门槛与结论边界。
- `docs/ENGINEER_RUNBOOK.zh.md`：从服务器预检到四卡执行的逐条命令。
- `docs/RESULT_SCHEMA.zh.md`：结果、诊断、decision 和 completion seal 的字段约定。
- `docs/VALIDATION.zh.md`：代码审计和测试范围。
- `configs/default.json`：与原 DistanceHead schema 完全兼容的运行配置。
- `configs/methods.json`：三个参考方法、四个 A1 直接父节点处理。
- `configs/quick_profile.json`：快速阶段矩阵和所有晋级阈值。
- `configs/protocol_lock.json`：旧科学核心生成的内层协议锁。
- `configs/package_lock.json`：覆盖本目录代码、文档和配置的外层锁。
- `run.py`：所有正式操作的唯一受锁入口。
- `plan_jobs.py` / `run_jobs.py`：四个 GPU worker 的不可变作业计划与执行器。

## 不变项

以下字段全部继承 `distance_head_study`，不能由命令行覆盖：

- topology hold-out manifests；
- backbone seed 42；
- head 训练 30,000 steps、batch 512、AdamW、final-step checkpoint；
- horizon 12、64 candidates、8 elites、1 CEM iteration、每步 replan；
- `corrected_v1` 和 `unmasked` 的既有语义；
- encoder、projector、predictor 的结构和参数；
- planner 结构、参数和预算；
- 测试 BFS 标签只用于离线诊断，不进入动作选择。

完整运行请严格遵循 [ENGINEER_RUNBOOK.zh.md](docs/ENGINEER_RUNBOOK.zh.md)。
