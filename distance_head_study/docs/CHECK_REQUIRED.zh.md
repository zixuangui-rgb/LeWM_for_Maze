# 服务器 CHECK-REQUIRED

以下内容无法在本地代码仓库中替工程师确认。未逐项确认，不得把 server run 标为 formal。

## Checkpoint provenance

- [ ] `checkpoints/final_closure/lewm_l2_cem_seqlen2_seed42.pt` 至少到 seed 44 存在。
- [ ] 每个 checkpoint 可严格加载 `model_config/model_state_dict`。
- [ ] seed、steps、architecture、manifest hash、SIGReg 配置与 `final_closure` lock 一致。
- [ ] checkpoint 不是 validation-selected 或继续训练后改名的版本。
- [ ] fresh seed `1001+` 的目录为空，避免误复用历史未知模型。
- [ ] 历史 Simple DistanceHead 原命令/log 如可获得，补充 provenance audit；旧 checkpoint
  仍不得进入新方法 ranking。

## Runtime

- [ ] 四张卡确为 NVIDIA H800，属于同一 runtime block。
- [ ] `nvidia-smi` 显示驱动/CUDA 正常且没有未知占用。
- [ ] PyTorch/CUDA/cuDNN 版本四卡一致。
- [ ] `torch.use_deterministic_algorithms(True)` 在目标版本不报错。
- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8` 生效。
- [ ] 每张卡先跑 diagnostic smoke，确认单卡 memory headroom。
- [ ] `server_preflight --backbone-seed 42 --device cuda` 输出 `status=pass`，并保存日志。
- [ ] `server_preflight` 未使用 `--checkpoint` 绕过默认 provenance：它必须同时通过 historical
  checkpoint 的 experiment family/version/stage/baseline/seed/formal flag、训练 spec、三组
  source manifest hash、rerun record 与 model payload 核验；fresh seed 还必须核对本研究
  protocol lock、source spec、训练配置和精确 optimizer steps。

## Filesystem

- [ ] `distance_head_study_runs` 位于持久存储。
- [ ] cache 与 checkpoint 路径有足够容量；预留至少 0.5-1 TB 再根据首个 cache 实测修正。
- [ ] 多 worker 不会同时写同一个 artifact。
- [ ] 文件系统支持 atomic rename；若对象存储不支持，先写本地再单 writer 上传。
- [ ] 日志、job state、cache、checkpoint 均定期备份。
- [ ] job JSONL 与同名 `.metadata.json` 一起保存；completion seal/output hash 可恢复核验。
- [ ] scheduler 只在对应 artifact validator 通过后标记成功；exit code 0 不能替代内容验收。
- [ ] 在最终 closure 前不清理 cache index、candidate bank、checkpoint、diagnostic、task rows
  或 summary；这些文件属于可重验的证据链，不只是临时中间件。

## Scheduler adaptation

- [ ] 若不用本地 `run_jobs.py`，Slurm/Kubernetes job 保留 JSONL 中的 command、deps、outputs。
- [ ] 同一 comparison block 的 method/seed GPU 映射轮换。
- [ ] confirmation 以 backbone seed 为 block，paired methods 尽量在同一 runtime window。
- [ ] 重试原因、host、GPU UUID、开始/结束时间写入外部 scheduler metadata。
- [ ] 不把 scheduler array index 当训练 seed。
- [ ] 本地 executor 的 interrupted job 只在原 PID 已停止后使用 `--retry-interrupted`；普通
  非零退出使用 `--retry-failed`，且保留所有 attempt 日志。
- [ ] 若外部 scheduler 替代 `run_jobs.py`，实现 cache/head/diagnostic/result/confirm-open
  等价语义 validator，而不只检查 outputs 是否存在。

## Scientific invariants

- [ ] 不改 `max_steps/horizon/candidates/CEM iters/action IDs`。
- [ ] 不用 `23/25` 做 development。
- [ ] 不提前打开 `D_select/D_confirm`。
- [ ] 不按好看的 head seed 继续。
- [ ] 不减少 confirmation backbone n。
- [ ] 不把 diagnostic/dirty/limited output 放入 decision。
- [ ] limited evaluator/diagnostics 仍经过 seed tier 与 sealed-split gate，不把 smoke 参数
  当作提前查看 holdout 的通道。
- [ ] `distance_head_study_runs/smoke/` 与正式输出分盘或至少分目录监控，不手动搬入正式路径。
- [ ] 不用 test BFS 做 learned action selection。
- [ ] Seed-3 DAG 产出的三组 legacy baseline-only full-900 rows 完整，且 power analysis
  没有读取任何 candidate/finalist effect。
- [ ] `screen_selection` 使用代码别名 `@main` 的完整固定集合；negative closure 使用
  `@negative_closure`，不手工删减 eligible methods。
- [ ] iCEM/Beam/Best-first 的 `plan_transitions <= 768`；Beam/Best-first 因完整 branch 不可
  分时可以少用余量，但不得用 no-op 补齐或超限，实际 compute 必须保留。
- [ ] protocol fingerprint audit 覆盖 `scripts/train/train_dim256.py`、`final_closure`、
  `hdwm`、`spatial_jepa_planning` 与 `vector_jepa_planner_frontier` 的 Python 依赖。

## 可按环境调整的部分

- Python executable/virtualenv 路径；
- CUDA device 编号；
- cache/output 的物理挂载点，但 config 中逻辑角色和 hash 必须保留；
- 同时运行的独立 job 数；
- Slurm partition、time limit、memory request；
- 日志与监控方式。

任何会改变样本、optimizer step、effective batch、loss、checkpoint、planner 或 evaluator
语义的调整都不是“环境适配”，必须新建 protocol version。
