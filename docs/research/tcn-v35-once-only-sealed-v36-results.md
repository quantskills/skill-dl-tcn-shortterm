# TCN v35 一次性 sealed test v36 结果

## 最终结论

v35 候选没有通过一次性 sealed test。最终状态为 `sealed_rejected_tcn_candidate_v36`，`candidate_model=false`。

这不是工程运行失败：速度、数据身份、checkpoint 身份、因果切分、一次性消费和逐样本评测均正常。失败点是 v35 在 ordinary validation 上观察到的 Top-tail checkpoint-selection 增量没有在两个 sealed 时段复现。该 sealed 身份已经永久消费，禁止根据本结果继续调参后重试。

## 不可变身份与覆盖

- Freeze ID：`642a485b09a2c381e5c1dac0fbd6d1938edf44e21c54d738cf3c4ff17ce486a6`
- Sealed result receipt：`68205700cf0a21b807c6ea8c6e5aea351a9732cda0662cf89d27a1fe2fe02db9`
- Sealed manifest SHA-256：`04cd32ec7f69fa3260ffa59f079dbdf03a8e26a204be16e96ffe551a578b7649`
- 逐样本/期限预测：562,113
- 模型-日期-期限指标组：28,800
- 市场日期/期限 paired groups：800
- Bootstrap units：2 sealed segments × 4 horizons = 8
- Bootstrap draws：5000
- 成本假设：单边 10 bps

评测仍遵守时间资格：第一 sealed 段只用 ordinary folds 0–2，第二段用 folds 0–4。每个日期/期限先对所有 time-safe seed/fold paired delta 求均值，再对日期 block bootstrap，避免模型重复预测造成伪样本扩张。

## Candidate 相对 RankIC control

| 指标 | mean delta | 95% CI low | 95% CI high | 门禁 |
|---|---:|---:|---:|---|
| RankIC | -0.000092 | -0.000336 | +0.000164 | 通过容忍门 |
| Top precision | -0.000097 | -0.000500 | +0.000333 | 均值门失败 |
| NDCG@Top | -0.000112 | -0.000598 | +0.000363 | 均值门失败 |
| Top return | -0.000011 | -0.000038 | +0.000017 | 通过容忍门 |
| 成本后收益 | -0.000010 | -0.000038 | +0.000019 | 通过容忍门 |
| Top turnover | -0.000701 | -0.001894 | +0.000449 | 通过 |

Top precision 和 NDCG 的均值同时为负，且两者 CI low 都没有达到主 robust-tail 门，因此任务对齐效果门失败。绝对差异很小且大多数区间跨零，但预注册规则不允许把“没有显著变差”解释成候选通过。

## Candidate 相对 LSTM

| 指标 | mean delta |
|---|---:|
| RankIC | -0.005624 |
| Top precision | -0.006389 |
| NDCG@Top | -0.003410 |
| Top return | -0.000162 |
| 成本后收益 | -0.000163 |
| Top turnover | +0.000281 |

sealed 数据上 LSTM 在主要预测和 Top-tail 指标上继续领先。TCN 架构不能被宣称必然优于 LSTM。

## 速度与机制解释

- TCN/LSTM model-step：`6.1094x`
- TCN/LSTM end-to-end：`5.6770x`

因此速度优化目标仍然成立；v34/v35 的分量梯度 cosine 也已证明梯度冲突不是当前主要故障。最终失败集中在泛化：ordinary validation 上的 checkpoint-selection 小增量没有跨越到 sealed 时段，而不是 TCN 速度或训练基础设施再次退化。

## 后续边界

不得再读取本 sealed 结果来选择 epoch、loss 权重、网络容量、特征或门槛。若继续研究 TCN，必须回到 ordinary data，以新的可证伪假设开展，并为未来候选准备从未参与本轮判断的新时间段 holdout。当前不授权部署、实盘、券商连接或收益承诺。
