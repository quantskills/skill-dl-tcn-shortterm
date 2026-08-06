# TCN v35 候选冻结与 sealed readiness v36 结果

## 结论

v36 已把 v35 的 ordinary-validation 候选转成可审计、不可事后变更的 sealed-test readiness，未执行 sealed test。当前状态是 `awaiting_explicit_sealed_authorization_v36`，不是最终预测效果结论，也不是部署或交易授权。

本轮发现并修复了一个比普通“禁止读测试标签”更隐蔽的时间穿越风险：v35 的五个 ordinary folds 与原始数据的两个 sealed folds 并不一一对应。第一段 sealed 从 2023-12-13 开始，因此不能使用在该日期之后才完成验证和 checkpoint 选择的 ordinary folds 3/4。v36 以 `validation_end_date < sealed_test_start_date` 为硬条件建立评测计划。

## 冻结身份

- v35 receipt：`96f673676f73056efe28d67913e8e8ab1029b28d733af4cb241a8fd91eb85f73`
- v35 状态：`constrained_tail_ordinary_validation_candidate_v35`
- v33 LSTM receipt：`caba5202de0dfccd8afc8383005e96a14125c400b601a97bf035b33baa19c67f`
- sealed split SHA-256：`04cd32ec7f69fa3260ffa59f079dbdf03a8e26a204be16e96ffe551a578b7649`
- v36 freeze ID：`642a485b09a2c381e5c1dac0fbd6d1938edf44e21c54d738cf3c4ff17ce486a6`
- v36 receipt：`4f86b1bcbe1846c7ec8c0c5fdf311b58753bd7e4aba1a6fe3ee190964b2655ad`

readiness 复算了 v35 与 v33 receipt 身份、全部声明输出、源文件以及进入计划的 TCN/LSTM checkpoint 哈希；任一漂移均在 sealed loader 之前 fail closed。

## Canonical sealed 数据描述

只把原始 split manifest 的 `stage=test, sealed=true` 行视为 canonical test：

| sealed fold | 日期 | 交易日 | 样本 |
|---:|---|---:|---:|
| 0 | 2023-12-13 至 2024-05-16 | 100 | 1991 |
| 1 | 2024-10-30 至 2025-03-27 | 100 | 1982 |

合计 3973 个样本。fold 1 的 `sealed_holdout` 1991 行只是 fold 0 测试段的重复保护登记，已明确排除，避免双重消费和重复计数。

## Time-safe checkpoint 计划

| sealed fold | 可用 ordinary folds | seed/fold exposures | candidate 改选 exposures | 最晚 validation end |
|---:|---|---:|---:|---|
| 0 | 0, 1, 2 | 9 | 5 | 2023-09-04 |
| 1 | 0, 1, 2, 3, 4 | 15 | 7 | 2024-10-15 |
| 合计 | — | 24 | 12 | — |

授权后的置信区间必须按交易日 block bootstrap，并在同一日期/期限内先聚合 paired model-unit delta，不能把同一 sealed 标签被多个 seed/fold 模型重复预测当成独立市场样本。

## 当前边界

- `authorization_received=false`
- `sealed_test_accessed=false`
- `evaluation_executed=false`
- `consumed_marker_created=false`

因此当前仍只能说：v35 在 ordinary validation 上通过，速度与梯度机制已达标，任务对齐 checkpoint selection 有稳定 Top precision 增量；不能说它在 sealed 数据上通过，也不能说 TCN 一定优于 LSTM。

下一步只有一个：用户在新的明确消息中逐字给出 `授权执行 sealed test` 后，按冻结协议一次性执行。结果无论通过或失败，都不得用同一 sealed 数据继续调参后重试。
