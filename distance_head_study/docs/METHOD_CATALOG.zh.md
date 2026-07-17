# 方法目录与单因素对照

机器可读真值是 [`../configs/methods.json`](../configs/methods.json)。每个 derived method
必须声明 `parent`、`overrides` 和完全相同的 `declared_changes`；resolver 会比较完整
scientific diff，出现未声明改动即失败。

## P0 Baselines

| ID | 含义 | 是否训练新 head |
|---|---|---|
| `b_l2_cem` | pooled latent-L2 + exact receding CEM | 否 |
| `b_dh_cem` | protocol-repaired historical concat DistanceHead | 是 |
| `b_dh_model_free` | true-next greedy，复用 `b_dh_cem` | 否 |
| `b_dh_predictor_greedy` | 历史 corrected-v1 predictor-greedy parity，复用 `b_dh_cem` | 否 |

`b_dh_predictor_greedy` 保留历史 corrected-v1 的五类 predictor fallback 接口，再从
四个可执行移动动作中选分数最低者。它用于连接历史结果，不等同于诊断报告中的
`predicted_latent_local`：后者从同一个当前 latent 分别施加四个移动动作，得到严格
action-aligned 的一步预测，再用 `h=1` 的 head 查询做局部排序。

## P1 Oracles

`o_dyn_true_rollout`、`o_score_true_bfs`、`o_bfs1` 只做 attribution。后两者读取 test BFS，
schema 强制 `role=oracle`、`confirmatory_eligible=false`。

## 主线

| Block | Methods | 唯一改变 |
|---|---|---|
| A target | `a1_log` | legacy per-maze target -> global `log1p` |
| A sampling | `a2_distance_balanced`, `a3_full_horizon` | sampler |
| B local | `b1_listwise` | tie-aware local listwise |
| B structure | `b2_bellman`, `b3_multistep` | Bellman 或 multistep/triangle |
| B factorial | `b5_local_structural` | locked structural parent 上加 local |
| C predictor | `c1_predicted_listwise` | predicted-next ordering |
| C calibration | `c2_dual_calibration` | true/pred domain adapter |
| D trajectory | `d1_trm_short`, `d2_trm_full` | short vs executed-action horizons `1/3/5/8/11`；对应 rollout slots `2/4/6/9/12` |
| D control | `d3_trm_shuffle` | 仅打乱 trajectory labels |
| D reach | `d4_reachability` | multi-budget reachability output/loss |

`a_target_parent/a_sampling_parent/b_structural_winner/b_parent/c_parent` 都来自签名 decision
artifact。下游配置包含这些 decision hashes，不能静默换 parent。

## Negative-claim reserve

| Family | Methods |
|---|---|
| regression | `r_loss_mae` |
| non-scalar | `r_output_ordinal`, `r_output_distribution` |
| local alternatives | `r_pairwise`, `r_delta` |
| metric structure | `r_eikonal`, `r_quasimetric` |
| temporal metric | `r_successor_contrastive` |
| architecture | `r_arch_asymmetric`, `r_arch_hierarchical_budget` |
| uncertainty | `r_uncertainty` |
| joint predictor | `j0_cont_predictor` vs `j0_dist_predictor` |
| joint projector | `j1_cont_projector` vs `j1_dist_projector` |
| joint full | `j2_cont_full` vs `j2_dist_full` |
| RC-aux style | `j3_rcaux_reach` |
| cost use | `p_path_integrated`, `p_hybrid_l2`, `p_reachability`, `p_risk_loop` |
| search | `p_icem`, `p_beam`, `p_best_first` |

`r_successor_contrastive` 不再是 local-listwise 的别名：它比较同一 shortest-path 上多个
successor 的 temporal order，并加入任意 waypoint 对照。`r_arch_hierarchical_budget` 是
三个 short/medium/long expert 加 horizon-conditioned gate 的真实 mixture，不是普通 MLP
换名。对应 forward/backward、expert 权重和 horizon sensitivity 均有机制测试。
其 scalar head 训练查询覆盖实际调用的 local slot `1` 与 legacy rollout slots
`2/4/6/9/12`。这与 trajectory supervision 的 executed-action horizons
`1/3/5/8/11` 是两套明确对应、但不能混写的坐标。

Planner-only `p_*` 方法不训练 head。它们复用 `c_parent`、`d4_reachability` 或
`r_uncertainty` checkpoint；`train_head` 对这些方法会直接报错。Hybrid 对 candidate
population 内的 DistanceHead/L2 分量分别标准化再组合，不直接相加异单位数值。
三种 search reserve 共享每 decision `768` predictor transitions 硬上限；iCEM 精确用满，
Beam/Best-first 只扩展完整 branch，因此实际用量可略低但不得超限，结果 row 保存真实值。
Model-free true-next scorer 明确使用 `predicted_domain=false`，predictor/rollout scorer 使用
`predicted_domain=true`；有 domain adapter 时不能混用这两个 latent 来源。

`j3_rcaux_reach` 的 horizon-conditioned input 和 multitask output 改变了首层/输出结构，
因此不能伪装成 strict full-checkpoint continuation。它使用 `compatible_shared` 初始化：
只加载 shape 相同的 shared trunk 与 scalar primary keys，新增 horizon 输入权重和
reachability 层随机初始化；加载 key 列表、parent path/hash 都进入 training spec。它与
`j1_dist_projector` 的差异应解释为“RC-aux-style 组合处理”，不能把总差异只归因于某一个
新增层。

`j0/j1/j2` 的 continuation control 与 distance treatment 都更新 head，并共享相同
distance objective、标定权重、optimizer steps 和可训练 backbone scope。Control 只在
distance path 的 latent 处 stop-gradient；原 JEPA continuation 仍更新相同 backbone scope。
因此 treatment-control 差异可以归因于 distance gradient 是否进入 backbone，而不是“多训
了一个 head”或不同 loss 计算量。

## 可支持的归因

- `A1 vs B-DH`：target scale；
- `A2/A3 vs locked A parent`：sampling，不累计；
- B 的 2x2：local main effect、structural main effect、interaction；
- `C1 vs B parent`：predicted-latent alignment；
- `D2 vs D1/D3`：full-horizon trajectory supervision，而非额外训练或随机 label；
- joint treatment vs 同 scope continuation：distance gradient，而非继续训练；
- `p_* vs checkpoint owner`：planner 使用方式，representation/head 参数完全相同。

跨 parent、跨 budget、跨 action protocol 或跨 split 的数字不能做上述因果归因。
