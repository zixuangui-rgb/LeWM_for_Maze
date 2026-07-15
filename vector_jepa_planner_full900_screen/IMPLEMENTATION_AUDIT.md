# 实现审计

| 协议要求 | 代码位置 | 机器检查 |
|---|---|---|
| exact B0 | `reference_evaluate.py`、`evaluate.py` | `parity.py`核对规范路径、当前checkpoint SHA、manifest顺序、完整任务字段和逐动作序列 |
| 900任务不可缩减 | `evaluate.py` | count必须等于900 |
| backbone冻结 | `train.py`复用frontier Track F | schema拒绝Track J |
| 1x预算 | planner config、`evaluate.py` | 每decision检查不超过768 |
| corrected/unmasked配对 | `run_plan.py` | 每个eval block生成两项 |
| Q2A不累计 | `configs/default.json` | 每项只有一个新增接口 |
| Memory有效接入 | `methods.py` | 固定Best-first父控制 |
| Q2B匹配控制 | config、`audit_protocol.py` | 叶子字段diff白名单；DTS用同预算uniform-expansion控制 |
| 共享head不漂移 | `methods.py`、selection scripts | DTS三方法及双向/forward逐张量、训练摘要和校准指标exact parity |
| 双向预算有效 | config、planner | 48+48条半程rollout后预留192 transitions重排 |
| 反例不泄漏 | `counterexamples.py` | 只读取train manifest和固定fold；中断恢复时逐record复验并重查1x mining预算 |
| 动态Q1父方法 | `freeze_q1.py`、`methods.py` | decision SHA进入effective spec |
| 多重比较 | `freeze_shortlist.py` | 固定48项Bonferroni区间；极端尾部精确枚举、无Monte Carlo误差 |
| 已绑定decision防替换 | `methods.py` | Q0 parity、输入result与上游decision SHA链；首次写入前的权限边界见runbook |
| 结果/权重一致 | `analysis.py` | 每次加载结果都重算源backbone与组件checkpoint路径和SHA256 |
| 最多两个shortlist | `freeze_shortlist.py` | schema gate=2 |
| 最多一个winner | `freeze_final.py` | corrected优先固定规则 |
| planner seed非伪重复 | `summarize.py` | backbone内先平均 |
| 交叉配对CI | `summarize.py` | backbone采样+共同task panel采样 |
| 不按分数重跑 | run-plan immutable outputs | 无score overwrite入口 |
| 完整报告 | `summarize.py` | mean/SD、逐尺寸、失败率、assistance和compute均输出 |
| 环境复现 | lock/common | `pyproject.toml`与`uv.lock`分别锁定并进入代码指纹 |
| 永久闭环 | `summarize.py` | closure包含summary/report hash；支持崩溃后的同内容续写 |

## 有意复用

底层encoder/predictor adapter、planner算法、head、数据sampler、corrected-v1和episode
runner复用 `vector_jepa_planner_frontier`。这避免创建第二份数值实现。新目录负责不同
的schema、方法矩阵、阶段依赖、full-900 evaluator、统计和锁。

## 已知边界

1. 仓库不含服务器checkpoint，本地不能完成真实GPU full-900 parity。
2. full-900已经观察，所有选择和最终报告均为探索性。
3. candidate replay被有意关闭；不能从该包主结果推断candidate coverage诊断。
4. `vector_jepa_frontier_validation`用于calibration，不用于模型选择，但它不是新的盲
   确认集。
5. 运行器串行执行job；集群并行必须保持冻结schedule和每个输出唯一owner。
6. 新包从仓库根目录运行，不改旧`pyproject.toml`和旧protocol locks；新协议单独锁定
   `pyproject.toml`与`uv.lock`。

这些边界必须保留在最终报告中，不得静默删除。
