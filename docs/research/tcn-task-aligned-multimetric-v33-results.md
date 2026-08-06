# TCN 是否必然优于 LSTM，以及 RankIC 是否适合本项目

## 结论

TCN 的预测效果不一定优于 LSTM。Bai et al. 2018 证明的是一个通用 TCN 在一组通用序列基准上优于当时的标准循环网络，并建议把卷积架构作为序列建模的自然起点；它不是金融分钟线任务上的普适优越性定理。[论文](https://arxiv.org/abs/1803.01271)与[作者代码库](https://github.com/locuslab/TCN)覆盖复制记忆、序列 MNIST、音乐、语言建模等任务，没有股票横截面预测。

Microsoft Qlib 的公开可比 benchmark 已提供直接反例。在 CSI300/Alpha158 表中，TCN 的 RankIC 为 `0.0421`、年化收益为 `0.0262`；LSTM 分别为 `0.0435` 和 `0.0381`。Alpha360 表中 LSTM 也同时高于 TCN 的 RankIC 和年化收益。Qlib 同时强调这些结果调参有限，不能解释为模型上限。[Qlib benchmark](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

RankIC 对本项目“同一日、同一期限、全股票池横截面排序”的主任务是合理的。v33 已证明 TCN 与 LSTM 的逐样本输出契约相同：二者都输出 1/2/3/5 日的连续评分，都对齐 `next-open-rank-v2`，186,264 条预测在 `(seed, fold, sample_id, horizon)` 上完全一一对应。模型分数的数值尺度不同不构成不公平，因为 Spearman RankIC 对严格单调变换不敏感。

但 RankIC 不是最终 Top10% 多头决策的充分指标。它给全排序位置近似等权，而实际组合只消费顶部股票，并受收益尾部、换手、成本、容量和执行约束影响。近期的 [LambdaRankIC](https://arxiv.org/html/2605.00501) 也把 RankIC 放在“全序排序是目标”的场景，同时用 decile portfolio 指标补充评价，而没有把单一 RankIC 当作完整经济结论。

## v33 真实数据结果

数据为 2021–2025 PandaData 分钟线派生的普通 validation，3 seeds × 5 walk-forward folds，未访问 test/sealed test。TCN 使用 v32 `grouped-label-mean` 最佳 checkpoint；LSTM 按历史固定协议重新训练/重放。LSTM 历史 RankIC 最大复现误差为 `0.0`，TCN checkpoint 最大复现误差为 `2.78e-17`。

| 指标 | LSTM | TCN | TCN - LSTM | 解释 |
|---|---:|---:|---:|---|
| mean RankIC | 0.115545 | 0.099791 | -0.015755 | LSTM 显著更好 |
| RankICIR | 0.3997 | 0.3553 | - | LSTM 更稳定 |
| Pearson IC | 0.101989 | 0.084878 | -0.017110 | LSTM 更好 |
| Top10% raw return | 0.002952 | 0.002979 | +0.000028 | TCN 仅高约 0.28 bp，置信区间跨零 |
| Top10% excess return | 0.002329 | 0.002356 | +0.000028 | 同上，不显著 |
| Top10% precision | 0.1252 | 0.1142 | -0.0110 | LSTM 显著更好 |
| NDCG@Top10% | 0.5719 | 0.5648 | -0.0071 | LSTM 显著更好 |
| long-short spread | 0.006836 | 0.005801 | -0.001035 | LSTM 更好，仅诊断 |
| Top10% turnover | 0.5700 | 0.6103 | +0.0403 | TCN 换手更高，未计成本 |

日期块 bootstrap 的 95% CI：

- RankIC delta：`[-0.021752, -0.009451]`
- Top return delta：`[-0.000722, +0.000769]`
- Top excess delta：`[-0.000692, +0.000792]`
- Top precision delta：`[-0.017396, -0.004583]`
- NDCG@Top delta：`[-0.013551, -0.000605]`

形式判定为 `task_aligned_metrics_mixed_v33`，因为 TCN 的 Top raw return 点估计略正而其他主指标为负。但统计解释更严格：TCN 的微小收益优势无法区别于零；LSTM 在 RankIC、Top precision 和 NDCG 上的优势有较稳定的负向区间。因此当前证据不支持“TCN 预测效果优于 LSTM”，也不支持因为一项未显著的 Top return 点估计就宣布 TCN 获胜。

## “最后模型输出不是一个东西”的核对

需要区分三层：

1. **张量/目标语义**：本项目两者是同一个东西，均为四期限横截面连续评分，对齐同一 rank target。RankIC 可公平比较。
2. **训练产物协议**：不是同一个东西。TCN 是冻结父干后的续训产物，LSTM 是从头训练；因此当前只能比较具体产物，不能推断架构一般优劣。
3. **研究分数与交易决策**：不是同一个东西。RankIC 是全池统计分数，Top10% 多头及含成本组合才是经济决策输出；必须并列评价。

## 本地知识库发现

Hermes 知识库的可微 RankIC 与 benchmark 条目支持相同判断：全市场 RankIC 可能在加入域约束或时间平滑后下降，但受约束的长-only指数增强结果反而改善；原因是全 RankIC 把容量分配给全排序和两端，而真实 long-only 决策只消费顶部。知识库还建议对 long-only 任务研究 top-tail loss，并要求把 raw-signal/decile 指标放回成本、换手、容量和组合约束中复核。

这些材料属于本地深读证据；它们用于形成假设，v33 的结论仍以本项目可复现真实数据为准。

## Agent Reach 覆盖审计

| 渠道 | attempted | effective | deep read | 说明 |
|---|---|---|---|---|
| Hermes 本地知识库 | 是 | 是 | 是 | 深读可微 RankIC、模型 benchmark、统一输出适配条目 |
| Bai 论文 / arXiv | 是 | 是 | 是 | 主论文深读 |
| GitHub 原生 Agent Reach | 是 | 否 | 否 | WSL 中缺少 `gh`，未伪装为成功 |
| GitHub Web fallback | 是 | 是 | 是 | 深读 `locuslab/TCN` 与 Microsoft Qlib 官方 benchmark |
| 知乎 | 是 | 是 | 否 | Exa 找到候选文章，页面深读被 403 阻止 |
| 微信公众号 | 是 | 是 | 否 | 找到华泰/Qlib 等候选，页面与浏览器深读超时；仅作线索，不作为核心结论依据 |

知乎和公众号的索引线索普遍也把 IC/RankIC 与 APY、Sharpe、TopK 组合并列报告，但因为本轮无法完成原文深读，没有把搜索摘要当作权威证据。

## 当前问题定位与下一轮唯一变量

速度已达到当前本机协议的目标：v32 观察到 TCN/LSTM `6.53x` model-step、`5.90x` end-to-end。既有梯度诊断也已完成当前方案的稳定化。v33 把剩余问题进一步缩小为：TCN 的全池排序、顶部命中和顶部次序仍弱于 LSTM，而且换手更高；当前最需要优化的是 **TCN 表征到 Top-tail 决策的传递**，不是继续微调 dataloader，也不是否定 TCN 路线。

下一轮应只引入一个变量：在保持 v32 TCN 架构、父 checkpoint、batch、学习率、epoch、split 不变的前提下，用预注册的 `SmoothL1 + 小权重 Top-tail pairwise/listwise loss` 与原 `SmoothL1` 做 TCN-only 配对。主门槛应同时要求 Top precision/NDCG 改善、RankIC 不发生不可接受退化、Top return 的区间改善，并保持当前速度下限。若该变量失败，再转向解冻更多 TCN 表征容量；不要同时改损失与结构。
