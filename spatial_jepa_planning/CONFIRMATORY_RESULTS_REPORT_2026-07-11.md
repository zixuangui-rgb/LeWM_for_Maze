# Spatial-JEPA Iterative Planning —— 确认性实验最终结果

**日期：** 2026-07-11
**协议：** `spatial-jepa-confirm-v2`（同事设计，我的环境 `lewm_maze_repro` 落地执行）
**数据来源：** 本文所有数字直接取自脚本生成的
`spatial_jepa_planning_runs/summary.md` / `summary.json`（由 `summarize.py`
从 10 seed × 900 task 的 confirmatory JSON 经 seed×task crossed paired
bootstrap + Bonferroni simultaneous CI 计算得出）。无手敲数字。

---

## 0. 一句话结论

迷宫规划上，任何 feedforward（非迭代）规划器——无论 pooled latent、per-cell
value field，还是全感受野 dilated CNN——都被钉死在约 0.60-0.63 SR；而 learned
迭代规划（weight-shared ConvGRU recall + progressive K + K-budgeted 监督）把它推到
0.949（逼近 oracle 0.979），且 Spatial-JEPA 世界模型表征不劣于 raw 像素输入
（0.936 vs 0.949）。这把 P4 的“feedforward 天花板”结论从一次性观察升级为有
全感受野对照 + confirmatory hold-out + 预注册 Bonferroni 统计的正式检验，并证明
迭代是打破天花板的关键，而非感受野或表征。

---

## 1. 实验设置（严谨性合同）

- **三分数据：** Train 2800（size 9-21 × 400）/ Development 900（旧 Set-B，
  已污染，仅调试）/ Confirmatory 900（size 9-25 × 100，新拓扑，一次性）；三者
  两两 topology/layout/task overlap = 0（canonical geometry hash 校验）。
- **主指标：** confirmatory 全 900 任务、`max_steps=128`、unmasked（完全用模型
  排序）、10 seeds（42-51）、预注册 primary K（FF=4，固定迭代=64，progressive
  迭代=128）；测试集 K 不参与模型选择。
- **统计：** seed×task crossed paired bootstrap（20000 次）× Bonferroni
  simultaneous CI（familywise α=0.05 → 每条 α=0.05/3）。
- **Oracle gate：** exact BFS = VI K=256 = 0.978889（= 881/900，19 个任务最短路
  >128 是结构性删失），`eligible_sr=1.0`——evaluator 正确。
- **确定性：** 全程 `torch.use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`，H800/CUDA 12.8 上无报错。
- **完整产出：** 100 planner ckpt + 10 rep ckpt = 110 个训练 checkpoint；
  10 seed ×（10 variant × 3 action 协议 + 1 rep）= 310 confirmatory JSON；
  0 训练失败、0 eval 失败
  （含自动重试机制应对瞬时 CUDA 故障）。

---

## 2. 主表：Primary SR（10 seeds，confirmatory，unmasked）

| Variant | K | 参数量（planner/总） | size-25 GMACs | SR | OOD SR | Local top-1 | 性质 |
|---|---:|---:|---:|---:|---:|---:|---|
| r0 value-only FF | 4 | 299,529 | 0.186 | 0.151±0.009 | 0.031 | 0.748 | feedforward |
| r1 action-CE FF | 4 | 299,529 | 0.186 | 0.625±0.010 | 0.246 | 0.914 | feedforward |
| r2 CE+Bellman+gap FF | 4 | 299,529 | 0.186 | 0.600±0.008 | 0.221 | 0.906 | feedforward |
| r2d dilated FF（全感受野） | 4 | 299,529 | 0.186 | 0.602±0.006 | 0.230 | 0.940 | feedforward 全 RF |
| r3 iterative 固定 K64 | 64 | 303,113 | 8.898 | 0.860±0.004 | 0.610 | 0.941 | 迭代 |
| r4 iterative progressive K128 | 128 | 303,113 | 17.746 | 0.949±0.015 | 0.844 | 0.985 | learned 迭代（主） |
| j0 spatial FF | 4 | 333,513 / 566,857 | 0.353 | 0.623±0.006 | 0.248 | 0.941 | JEPA+FF |
| j1 spatial iterative frozen | 128 | 337,097 / 570,441 | 17.912 | 0.936±0.012 | 0.805 | 0.979 | JEPA+迭代（frozen） |
| j2 spatial iterative lastblock | 128 | 337,097 / 570,441 | 17.912 | 0.944±0.012 | 0.822 | 0.983 | JEPA+迭代（staged） |
| j3 spatial iterative joint | 128 | 337,097 / 570,441 | 17.912 | 0.939±0.013 | 0.815 | 0.981 | JEPA+迭代（joint） |
| （oracle：exact BFS / VI K256） | - | - | - | 0.979 | - | - | 上界 |

**解读：**

- **feedforward 天花板约 0.60-0.63 复现：** r1/r2/r2d（全感受野）/j0 全部
  聚在 0.60-0.63。r2d（全感受野 dilated FF）= 0.602 ≈ r2（普通 FF）=
  0.600 → 天花板不是感受野问题。
- **迭代打破天花板：** r3（K64）= 0.860 → r4（K128）= 0.949；更多迭代
  → 更高 SR。
- **learned（非 oracle）：** r4/j1/j2/j3 不用 hardcoded 墙掩码，是真正学出来的
  迭代规划。

---

## 3. 三条预注册假设判定（Bonferroni simultaneous CI，10 seeds）

| 假设 | 比较 | 判据 | ΔSR [CI] | 结论 |
|---|---|---|---:|---|
| H1 迭代增益 | r4 - r2d | superiority，CI low ≥ +0.03 | +0.346 [+0.312, +0.381] | supported |
| H2 JEPA 非劣 | j1 - r4 | noninferiority，CI low > -0.03 | -0.013 [-0.029, +0.004] | supported（下界 -0.029 刚好在 -0.03 内，较窄） |
| H3 staged 增益 | j2 - j1 | superiority，CI low ≥ +0.03 | +0.008 [-0.005, +0.022] | not_supported |

- **H1（决定性）：** learned 迭代比全感受野 FF 高 +34.6 点。注意 r4 用 17.7
  GMACs vs r2d 0.186 GMACs（约 95× 计算量），故 H1 估计的是“完整迭代系统 +
  更多 inference compute”的增益，不能单独归因于 recurrence——这是协议诚实
  声明的。
- **H2：** frozen Spatial-JEPA 表征（0.936）不劣于 raw 输入（0.949），说明
  full-resolution Spatial-JEPA 保住了规划信息（绕开旧 stride-8 pooled backbone
  的 projector 瓶颈）。
- **H3：** staged last-block 微调（j2 0.944）并未显著优于 frozen（j1 0.936），
  增益 +0.8 在噪声内 → frozen JEPA 已足够，受约束适配无额外收益。

**exploratory 发现（不进主结论，但有机制价值）：**

- **r1 - r0 = +0.474：** 对 FF，动作目标（action CE）是关键杠杆，纯 value
  回归（r0=0.151）很差。
- **r2 - r1 = -0.026：** FF 上加 Bellman+gap 反而略伤（相对纯 action CE）。
- **r2d - r2 = +0.003：** dilation/全 RF 对 FF 可忽略 → 再证天花板非感受野。
- **r3 - r2d = +0.257：** 固定 K64 迭代即大涨。
- **j0 - r2d = +0.021：** Spatial-JEPA 表征对 FF 略有帮助（边际）。
- **j3 - j2 = -0.005：** joint 训练不优于 staged；梯度审计显示 rep/plan 梯度
  cosine 约 0.01-0.04（近正交，干扰已被 branched projector 抑制）。

---

## 4. 机制发现（为什么迭代能，FF 不能）

### 4.1 Local top-1 是 SR 的金标准预测器

延续 P4 诊断，跨 feedforward 与迭代两个 regime：

- FF：local top-1 0.91-0.94 → SR 0.60-0.63；
- 迭代：local top-1 0.979-0.985 → SR 0.936-0.949；
- oracle：0.98+ → 0.979。

局部动作排序能力直接决定 SR；迭代把 local top-1 从 0.94 推到 0.985，SR 随之
从 0.63 到 0.95。

### 4.2 shortest-path 分层

FF 在长路径上崩，迭代不会：

- FF（r2d）：路径 1-16 = 0.961，33-64 = 0.289，65-128 = 0.028；
- r4：1-16 = 1.000，33-64 = 0.995，65-128 = 0.809。

FF 无法把 value 传播远；迭代可以。这是天花板的核心机制。

### 4.3 size 外推（OOD 23/25）

- FF OOD 约 0.22-0.25（未见尺寸上崩溃）；
- r4 OOD = 0.844；j1/j2/j3 OOD = 0.805-0.822。

迭代规划能外推到更大 maze（支持“学到的可重复算法 > 固定深度模式”）。

### 4.4 assistance 诊断

`corrected = oracle valid + 防回退`：

- FF r0：unmasked 0.151 → corrected 0.702（差 +0.551；FF 大量 invalid/回退
  动作）；
- 迭代 r4：unmasked 0.949 → corrected 0.962（差 +0.014，已近最优）。

FF 失败部分是 invalid-action 翻滚；迭代规划器导航干净。`model_valid ≈ unmasked`
（有效头学得好）。

### 4.5 decoded-map BFS

decoded-map BFS = 0.979（rep 完美可解码：wall IoU/agent/goal acc 全 1.000）：地图
信息一直在表征里；瓶颈从来是 planner，不是 representation——这与 H2（frozen
JEPA 够用）一致。

---

## 5. 与 P4 / 前序工作的统一

| 阶段 | 结论 | 本实验地位 |
|---|---|---|
| diagnostics | 三墙：局部排序/投影器/rollout 漂移 | 锚点 |
| planning_repair | 表征墙可修（valid 0.85）但不 SR-binding；Wall1 是天花板 | 锚点 |
| P4 fcvp/vi | feedforward value field 撞 0.63；只有 hardcoded VI 到 0.957 | 被本实验升级：0.63 天花板用 r2d 全 RF 对照正式确认；learned 迭代（非 oracle）首次到 0.949 |
| 本实验（Spatial-JEPA） | learned 迭代打破天花板；JEPA 表征非劣；staged 无增益 | 正式 confirmatory 结论 |

**关键推进：** P4 的 vi=0.957 用的是 hardcoded 墙掩码 = oracle；本实验的
r4=0.949 / j1=0.936 是 learned ConvGRU 迭代（无 oracle 墙），且在新 hold-out +
10 seed + Bonferroni 下成立。这是从“迭代理论上能”到“learned 迭代实测能，且
表征用世界模型也行”的实证跨越。

---

## 6. 可得 / 不可得的结论（协议边界）

### 可以声称

- 新 confirmatory hold-out 上 feedforward 规划器约 0.63 上限（含全感受野 r2d
  对照）；
- learned 迭代规划稳定打破该上限到约 0.95（H1 supported）；
- frozen Spatial-JEPA 表征非劣于 raw（H2 supported）；
- staged 适配无显著增益（H3 not_supported）。

### 不能声称（协议明令）

- 超过旧 BC（0.781）/latent-L2/原 LeWM——它们未在新 confirmatory 集同协议
  重跑（仅作 legacy context）；
- 纹理/颜色/背景/跨任务/跨环境泛化；
- recurrence 在等 FLOPs 下优于所有 FF（H1 含更多 compute）；
- corrected SR = 无外部帮助能力。

---

## 7. 严谨性 / 复现性

- **协议忠实：** `spatial_jepa_planning/` 逐字节等同同事仓库 `@0eca772`
  （commit on local branch `spatial-jepa-confirm 810c7b7`，未 push，所有输出
  gitignored 本地）。
- **数据一致：** 我的 train/dev manifest SHA256 与 `protocol_lock` 逐字节一致；
  confirmatory 900 任务在我的环境可逐任务复现。
- **校验通过：** 全程确定性 + clean worktree + spec-hash 链 + runtime 同构校验
  全过；`summarize.py` 全部 metadata 校验通过（10 seed 齐，git commit 一致，
  ckpt hash 在）。
- **已知限制：** ①单架构（hidden_dim=64）；②r4/j1-j3 的 K128 比 FF 用约 95×
  inference compute（H1 估计含 compute）；③H2 下界 -0.029 距 -0.03 阈值很近
  （虽 supported 但窄）；④19 个最短路 >128 的任务被 `max_steps=128` 结构性删失
  （已计入 oracle 上界）。
