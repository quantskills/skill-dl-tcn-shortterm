# TCN V46 效用对齐与一次性独立外推结果

日期：2026-08-06
预注册提示词：[TCN V46：效用对齐合同与一次性独立外推测试](../prompts/tcn-v46-prospective-utility-aligned-independent-test.md)

## 结论

V46 已按预注册合同完成一次且仅一次的 2025-04-07 至 2025-12-23 独立外推。正式状态为：

`v46_student_not_generalized`

这不是因为旧 `top_precision` 继续作为门禁。V46 已明确把预测 Top10% 与实际 Top10% 的集合重合率降为诊断项；即使完全忽略该指标，V42 student 相对 true-target TCN 仍有四个整体门失败：RankIC 置信下界、正向期限数、Top 超额收益和 NDCG@Top。

同时，本轮没有否定 TCN 架构本身。冻结的 V42 student 在该独立窗口相对 LSTM 的 RankIC、Top 超额收益和 NDCG 点估计均更高，三项非劣置信下界门全部通过；父证据中的训练 model-step 速度比仍为 `4.8789x`。失败的是“V42 共识蒸馏相对 true-target TCN 的增益能跨时期稳定外推”这一机制假设，而不是“TCN 必须放弃”。

## 冻结边界

- V42 ordinary validation 最晚日期：2024-10-15；
- 既有 once-only sealed test 最晚日期：2025-03-27；
- embargo 结束：2025-04-03；
- V46 独立窗口：2025-04-07 至 2025-12-23；
- fold-4 训练/归一化统计最晚日期：2024-06-11；
- 177 个交易日，708 个日期—期限组，33,674 个有效标签；
- 横截面宽度为 37–50 只股票；
- 三模型 × seeds 7/17/27 × 1/2/3/5 日，9 个冻结检查点全部通过父 receipt SHA-256 和 strict state-dict 校验；
- 不训练、不微调、不校准、不搜索门槛、不选择 seed/horizon；
- receipt id：`364eebcfad46b6935c1a63f49788598ba24625455ea09ea593cc78614e3402a8`。

## 指标语义纠偏

V46 同时输出三个不能互换的指标：

| 指标 | 定义 | V46 是否门禁 |
|---|---|---|
| `top_membership_precision` | 预测 Top10% 与实际收益 Top10% 的集合重合率 | 否 |
| `top_positive_return_rate` | 预测 Top10% 中原始持有期收益大于 0 的比例 | 否 |
| `top_above_cross_section_mean_rate` | 预测 Top10% 中收益高于同期横截面均值的比例 | 否 |

旧字段 `top_precision` 被保留为 `top_membership_precision` 的历史兼容别名。它从来不是正收益命中率。小样本单元测试已证明：集合重合率为 0 时，正收益命中率仍可为 100%，因此不能用一个名称解释另一个业务问题。

## Student 相对 true-target TCN

| 指标 | V42 student - control TCN |
|---|---:|
| RankIC | `+0.000839` |
| RankIC 95% CI low | `-0.006556` |
| Top excess return | `-0.000668` |
| NDCG@Top | `-0.001701` |
| Top membership precision（诊断） | `-0.010334` |
| Top positive-return rate（诊断） | `-0.002990` |
| Top above-cross-section-mean rate（诊断） | `-0.002943` |
| Top turnover（诊断） | `-0.079830` |

跨 seed 的 RankIC delta：

- seed 7：`+0.000868`；
- seed 17：`-0.007732`；
- seed 27：`+0.009382`。

跨期限的 RankIC delta：

- 1 日：`-0.003684`；
- 2 日：`-0.000211`；
- 3 日：`+0.001012`；
- 5 日：`+0.006241`。

因此 student 的点估计仅略高于 control，但收益与 Top 区域排序更差，且增益集中在 seed 27 和 3/5 日，不能称为跨初始化、跨期限的稳定机制外推。

## Student 相对 LSTM

| 指标 | V42 student - LSTM |
|---|---:|
| RankIC | `+0.004030` |
| RankIC 95% CI low | `-0.001705`，通过 `>= -0.0100` |
| Top excess return | `+0.000497` |
| Top excess return 95% CI low | `-0.000034`，通过 `>= -0.0005` |
| NDCG@Top | `+0.001577` |
| NDCG@Top 95% CI low | `-0.003563`，通过 `>= -0.0100` |
| 父证据 TCN/LSTM model-step 速度比 | `4.8789x`，通过 `>= 3.0x` |

冻结模型的独立期点估计也显示 control TCN 的 RankIC、Top 超额收益和 NDCG 均高于 LSTM。因而当前证据更符合：TCN 模型族在该数据与算力约束下仍有预测—速度价值，但 V42 的共识蒸馏增益没有稳定超过简单 true-target TCN。

V46 记录的纯推理吞吐约为：

- control TCN：12.5k–13.9k samples/s；
- student TCN：13.4k–14.2k samples/s；
- LSTM：12.1k–12.2k samples/s。

原始“3–5x”目标指同预算训练 model-step，不应拿本次短批量 CPU 推理比率替换该证据。

## 当前阶段与下一步

1. **速度基础设施：完成。** 单 TCN 的训练 model-step 速度证据仍为 `4.8789x`，因果卷积、WeightNorm、感受野、memmap 和单前向约束没有回归。
2. **评测语义：完成纠偏。** membership、正收益命中和相对横截面均值命中已分开，membership 不再作为 V46 非补偿门。
3. **V42 机制外推：失败。** ordinary-validation 上的 `+0.0145` RankIC 增量在新时期收缩为 `+0.00084`，且经济效用与 NDCG 转负。
4. **TCN 对 LSTM：独立期非劣—速度门通过。** 这支持继续研究 TCN，而不是改用 LSTM；但不支持宣称 TCN 普遍优于 LSTM。
5. **发布状态：不通过。** `alpha_ready=false`、`deployment_authorized=false`、`trading_authorized=false`。

不得再用该 177 日窗口调 teacher weight、损失、架构、seed、期限权重或门槛。下一轮若继续，应获取 2026 年及以后的新开发数据，围绕“跨时期稳定性/分布漂移”形成新机制与新的前瞻窗口；V46 只能用于事后诊断，不能重新成为开发集。

## 权威证据

- 产物目录：`artifacts/tcn-v46-utility-aligned-independent-test/`
- 决策：`decision.json`
- 完整 receipt：`receipt.json`
- 三模型比较：`student-control-comparison.json`、`student-lstm-comparison.json`
- 配对分块 bootstrap：`paired-bootstrap.parquet`
- 逐组语义指标：`utility-metrics.parquet`
