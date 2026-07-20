# AIR0 与原 Spatial-JEPA 实验的兼容合同

## 1. 复用边界

AIR0 的显式兼容入口包括：

```text
hdwm/envs/procgen_maze.py
diagnostics/common.py
spatial_jepa_planning/models.py
spatial_jepa_planning/common.py
spatial_jepa_planning/evaluate.py::run_navigation
```

package fingerprint 不只覆盖这几个入口：它会从全部 AIR Python 文件和上述入口出发，
递归解析仓库内的静态 Python import，并同时纳入完整 `air_jepa/`、
`tests/test_air_jepa_stage0.py`、`pyproject.toml` 与 `uv.lock`。因此诸如
`hdwm/config.py`、action utilities、LeWM model 定义或其本地依赖发生变化，也会让
`package_lock --check` 失败。动态生成、运行时 monkey patch 或未进入该闭包的替代模块
禁止用于正式运行。

## 2. 张量接口

| 接口 | 形状 | dtype | 说明 |
|---|---|---|---|
| observation | `[B,H,W,5]` | float32 | 原 Procgen Maze channel-last 观测 |
| planning latent | `[B,64,H,W]` | float32 | `SpatialRepresentation.planning_latent()` |
| successor observation | `[B,4,H,W,5]` | float32 | 真实四动作一步转移，含 no-op |
| successor latent | `[B,4,64,H,W]` | float32 | 同一个 frozen planning encoder 编码 |
| action order | `[1,2,3,4]` | int | UP, DOWN, LEFT, RIGHT |
| cost logits | `[B,4,129]` | float32 | bins 0..128，超出值截断到 128 |

AIR 的 `valid_mask` 是 padding mask。当前 same-size batch 没有 padding，因此全为 true；
它不是 free-cell 或 valid-action oracle mask，墙体 token 仍保留在输入中。

## 3. 冻结的含义

“frozen Spatial-JEPA”同时表示结构和参数不变：

- checkpoint 按 seed 固定；
- encoder 与 planning projector 设为 `eval()`；
- 所有 source parameters `requires_grad=False`；
- current/successor 编码在 `torch.no_grad()` 中；
- AIR optimizer 只接收 workspace 模型参数；
- checkpoint 记录 representation trainable parameter count，必须为 0。

## 4. 基线语义

- `j0-static`：从 start observation 计算一次 feedforward policy field，锁定 depth K4；
- `j1-static`：从 start observation 计算一次 iterative policy field，锁定 K128；
- `j1-receding`：每环境 step 读取新 observation 并重新计算同一个 J1，评测七个 K；
- `AIR0-*`：每环境 step 重新编码和运行 workspace，选择 expected cost 最低动作。

所有方法的 primary action protocol 都是 `unmasked`。`corrected` 才读取真实转移来排除
no-op 和 immediate backtracking，并被标记为 assistance diagnostic。

## 5. BFS 使用边界

允许：生成 train candidate cost、tie-aware optimal action、评测指标、local diagnostic
标签和失败后分析。

禁止：primary action function、candidate filter、workspace input、energy input或搜索过程
读取 BFS/墙体/validity。代码把这些路径分开，单测同时检查 unmasked 始终返回四动作。

## 6. 对照成立条件

同一 seed 的 `AIR0-direct` 与 `AIR0-jepa` 必须同时满足：

- `initial_model_state_sha256` 相同；
- `paired_sample_stream_sha256` 相同；
- 正式训练前 128 batches 的 prefix hash 必须逐 seed 等于 L0 protocol audit 已签名的
  `sample_stream_sha256`；
- `rng_stream_seeds`、`k_counts` 和完整 30k progressive-K 序列 hash 相同，并且该
  序列必须由 seed 与锁定 phase 规则重新生成；
- source representation state hash 相同；
- future target channel moments 相同；
- optimizer、30k steps、batch8、cosine、float32/no AMP 相同。

所有 formal artifacts 还必须记录同一 H800 software/runtime signature，包括
`PYTHONHASHSEED=0`、`CUDA_DEVICE_ORDER=PCI_BUS_ID` 与
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，以及 Python、NumPy、PyTorch、Pydantic、
Gymnasium、OmegaConf、CUDA 和 cuDNN 版本。

正式 AIR checkpoint 采用临时文件、`fsync` 和原子 rename 落盘；目标文件已存在时
拒绝覆盖。evaluation 与 diagnostic 统一拒绝 smoke、非 final-step 或 tensor hash
不匹配的 checkpoint。

汇总前若任一条件不满足，该 pair 为 protocol violation，不能计算方法差异。
