# AIR-JEPA Stage 0 技术失败与替代运行协议

## 1. 目的

本协议只处理客观技术无效，不处理分数低、收敛慢或研究者不满意。AIR0-v1 的 runner
有意不提供单 cell 自动重试：选择性重跑某个 seed、method 或评测会引入可选停止和
结果筛选风险。

## 2. 可以认定为 technical invalid 的情况

- GPU、节点、驱动、文件系统或调度器发生可证实故障；
- 进程非零退出、OOM、NaN/Inf、artifact 缺失或 hash 校验失败；
- L0 发现 source、hardware、runtime、bridge parity 或协议不满足锁定合同；
- 代码实现错误在看到任何 AIR 性能结果前被发现。

以下情况不是 technical invalid：SR/SPL 低、某个 seed 表现差、K scaling 不明显、
future intervention 不符合预期、训练 loss 看起来“不够漂亮”。

## 3. 失败后立即保留的证据

停止当前唯一 runner，不删除或改写任何文件。至少保存：

- `job_plan.json` 及其签名；
- 全部 `job_status/*.json`、`logs/*.log` 和已经产生的 artifacts；
- Git commit、package/protocol/source lock hashes；
- 故障时间、GPU/节点、driver/runtime、返回码和原始错误；
- 已经出现性能输出与否，以及哪些数据角色已经被打开。

## 4. 替代运行记录

研究负责人批准后，在 `air_jepa_runs/replacement_ledger/` 新建一个不可覆盖的 JSON，
至少包含：

```text
schema: air-jepa-stage0-replacement-v1
failed_attempt_id
failed_run_root
failed_job_id
failure_class
failure_started_at_utc
failure_detected_at_utc
objective_evidence_paths[]
objective_evidence_sha256{}
performance_seen_before_decision: true|false
approved_by
approved_at_utc
approval_record_path
approval_record_sha256
old_git_commit
new_git_commit
scientific_protocol_changed: false
disposition
```

`failure_class` 只能是 `hardware`、`scheduler`、`filesystem`、`oom`、`nonfinite`、
`artifact_integrity`、`source_incompatibility` 或 `pre_score_code_defect`。自由文本解释放在
`disposition`，不能替代原始 log/hash。

## 5. 重启规则

1. 将完整失败目录原样移动到唯一归档名，例如
   `air_jepa_runs/stage0_workspace_attempt1_technical_invalid/`；不得只移动失败 cell。
2. 原 commit 未变且仅为外部基础设施故障时，可在相同锁定代码上从 L0 重启完整 DAG。
3. 只要代码、模型结构、loss、预算、manifest 或分析规则变化，就不得沿用 AIR0-v1
   结论；必须新建实验 ID、重新锁定 package/protocol，并从 L0 开始。
4. 新 attempt 不复用旧 attempt 的训练 checkpoint、evaluation result、benchmark、
   release 或 job status；source artifacts 可以复用，但必须重新生成并验证 source lock。
5. 最终交付同时包含所有 replacement records 和失败 attempt 的只读归档位置。

## 6. 判定边界

若无法客观区分“基础设施失败”和“结果不理想”，默认不得重跑。若失败发生在性能
已经可见之后，审批记录必须明确说明为何替代与分数无关；否则该新 attempt 只能标记为
探索性 follow-up，不能替代原 attempt 的预注册结论。
