# TCN 受约束 Top-tail checkpoint selection v35：真实结果

## 结论

v35 已在真实 2021–2025 股票分钟线普通 validation 上完成 3 seeds × 5 folds × epoch 0..8 的全轨迹实验，状态为 `constrained_tail_ordinary_validation_candidate_v35`。完整性、selection 机制、预测效果和速度四门全部通过；sealed test 未访问、未授权。

本轮只训练一条固定 8-epoch Top-tail TCN 轨迹。control 与 candidate 共享全部 135 个 epoch checkpoints，唯一差异是 checkpoint selection：control 最大化 RankIC，candidate 在每个单元 `RankIC >= unit max RankIC - 0.002` 的可行域内最大化 `0.5 × Top precision + 0.5 × NDCG@top`。

不可变 artifact：`artifacts/tcn-constrained-tail-checkpoint-selection-v35`；receipt：`96f673676f73056efe28d67913e8e8ab1029b28d733af4cb241a8fd91eb85f73`。154 个 receipt outputs、135 个 checkpoint 文件已逐项复核 SHA-256，无缺失或漂移；135 个 checkpoint SHA-256 均唯一。

## 机制与完整性

- 15/15 trajectory 均完整训练到 epoch 8，连同 frozen parent epoch 0 共 135 个状态。
- control RankIC selection 与训练 leaderboard 的 best epoch 最大误差为 `0`。
- 每 epoch 任务对齐 RankIC 与训练历史最大误差 `5.55e-17`；selected checkpoint 指标重放最大误差 `1.11e-16`。
- 7/15 单元改变 checkpoint，所有 candidate checkpoint 均满足 `best RankIC - 0.002` 可行域。
- changed units 中 4 个改选更早 epoch、3 个改选更晚 epoch；candidate 最晚只选择 epoch 2。这否定了“单纯延长训练即可改善”的强解释，支持的是“任务对齐 selection 在早期邻近 checkpoints 中有价值”。
- Top-tail/SmoothL1 梯度 cosine 中位数 `+0.5143`；父模型 state/prediction drift `0`。
- 每 epoch 全轨迹任务指标重放耗时 `166.56s`，已与模型训练吞吐分账。

## Candidate 相对 RankIC control

| 指标 | control | candidate | delta | 95% CI |
|---|---:|---:|---:|---:|
| RankIC | 0.099187 | 0.098918 | -0.000269 | [-0.000637, +0.000044] |
| Top10% raw return | 0.003042 | 0.003141 | +0.000100 | [-0.000015, +0.000274] |
| Top10% precision | 0.114687 | 0.116146 | +0.001458 | [+0.000729, +0.002292] |
| NDCG@Top10% | 0.564953 | 0.565518 | +0.000566 | [-0.000053, +0.001192] |
| quantile monotonicity | 0.125624 | 0.125515 | -0.000109 | [-0.001192, +0.000955] |
| Top10% turnover | 0.614346 | 0.614346 | 0.000000 | [-0.001582, +0.001477] |

Top precision 的改善区间完全高于零；NDCG 点估计为正且区间下限仅轻微低于零，满足预注册 secondary CI 门；RankIC 退化远小于 `-0.002` 容忍度且区间跨零；Top return 点估计改善约 1 bp、区间跨零；换手不变。因此 candidate 相对同轨迹 RankIC control 通过普通验证效果门。

## Candidate 相对固定 LSTM benchmark

| 指标 | LSTM | candidate TCN | TCN - LSTM | 95% CI |
|---|---:|---:|---:|---:|
| RankIC | 0.115545 | 0.098918 | -0.016628 | [-0.023174, -0.010223] |
| Top10% raw return | 0.002952 | 0.003141 | +0.000190 | [-0.000618, +0.000950] |
| Top10% precision | 0.125208 | 0.116146 | -0.009062 | [-0.015315, -0.003018] |
| NDCG@Top10% | 0.571931 | 0.565518 | -0.006413 | [-0.012738, +0.000007] |
| Top10% turnover | 0.570042 | 0.614346 | +0.044304 | [+0.031857, +0.056543] |

v35 的晋级是“candidate 优于同一 TCN 训练轨迹的 RankIC selection”，不是“TCN 已优于 LSTM”。TCN 的 Top return 正点估计仍不显著；LSTM 在 RankIC、Top precision 和 turnover 上继续明显更好。不得把 ordinary-validation candidate 状态外推为架构普适优越性或可交易 Alpha。

## 速度

- 固定 8-epoch Top-tail TCN/LSTM model-step：`6.1094x`。
- 固定 8-epoch Top-tail TCN/LSTM end-to-end：`5.6770x`。
- 两项均通过 `>=3x` 门槛。selection 评测的 166.56 秒单列，不计入模型训练吞吐，也没有冒充训练速度优化。

## Unknown 更新与下一步

v35 解决了“RankIC checkpoint selection 是否压制 Top-tail 决策效果”的未知量：答案是 **会丢失一部分可复现的 Top precision/NDCG，但问题发生在早期相邻 checkpoints，而不是训练不够久**。

候选已通过预注册 ordinary validation，继续在相同 validation 上调 selection 容忍度、权重或 loss 会增加研究者自由度与过拟合风险。合理下一步不再是继续普通验证调参，而是：

1. 冻结 v35 的完整代码、数据、checkpoint selection 和 receipt。
2. 由用户显式授权后，按既有 promotion/freeze 状态机进行一次性 sealed test；不得提前查看或反复消费。
3. sealed test 必须同时检查 Top precision/NDCG、RankIC、Top return、成本后收益与 turnover。v35 当前仍未授权 sealed、部署或交易。

若用户暂不授权 sealed test，则停止这一候选上的调参并保留 artifact；后续新的 listwise/Lambda-style loss 必须作为独立研究分支，不能用同一 sealed test 反复筛选。
