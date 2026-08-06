# TCN Top-tail 任务对齐优化 v34：真实结果

## 结论

v34 已按预注册提示词完成真实五年股票分钟线普通验证，覆盖 3 seeds × 5 folds、control/candidate 两个 TCN objective，并用 v33 LSTM 逐样本预测作上下文 benchmark。状态为 `stop_top_tail_no_gain_v34`；sealed test 未访问、未授权。

本轮证明了三件事：

1. Top-tail loss 确实生效，梯度没有冲突：每个训练 batch 约 826 个有效 pair，SmoothL1 与 Top-tail 的 prediction-space 梯度 cosine 中位数为 `+0.5142`，父模型 drift 与 RankIC 重放误差分别为 `0`、`2.78e-17`。
2. TCN 相对 LSTM 的速度优势仍满足原目标：model-step `6.2309x`、端到端 `5.7578x`。但 Top-tail loss 与逐 batch 梯度诊断使 candidate/control 吞吐比只有 `0.7377x`，没有通过新增 objective 开销门 `0.85x`。
3. 固定 `0.05` 权重的 Top-tail objective 只带来极小且不稳的头部收益/命中点估计，NDCG 未改善，RankIC 显著小幅下降，换手显著上升，因此不能晋级。

不可变 artifact：`artifacts/tcn-top-tail-alignment-v34`；receipt：`de87e54264b942da3b67c0e9da3b9f2aa302e41a6b7ac99f8c0ad9ec9185cb2c`。46 个 receipt outputs 已逐项复核 SHA-256，无缺失或漂移。

## 单变量协议

- control：冻结父模型 shape-residual TCN，date-grouped SmoothL1。
- candidate：同一个 TCN、同一父 checkpoint、同一 88 个可训练参数、同一数据/seed/fold/日期顺序/学习率/batch/预算，只把 objective 改为 `SmoothL1 + 0.05 × Top-tail pairwise logistic`。
- Top-tail：每个日期/期限取真实 top 10%，与所有非 top 成员配对，温度 `0.1`。
- checkpoint 仍按 mean daily RankIC 选择，避免本轮同时改变训练 loss 与选择规则。

## Candidate 相对 Control TCN

| 指标 | control | candidate | delta | 95% CI |
|---|---:|---:|---:|---:|
| RankIC | 0.099791 | 0.099181 | -0.000610 | [-0.001134, -0.000095] |
| Top10% raw return | 0.002979 | 0.003035 | +0.000056 | [-0.000056, +0.000180] |
| Top10% precision | 0.114167 | 0.114375 | +0.000208 | [-0.000938, +0.001354] |
| NDCG@Top10% | 0.564849 | 0.564822 | -0.000027 | [-0.000959, +0.000965] |
| quantile monotonicity | 0.127722 | 0.125763 | -0.001960 | [-0.003473, -0.000399] |
| Top10% turnover | 0.610338 | 0.613397 | +0.003059 | [+0.000738, +0.005485] |

Top return 与 precision 的点估计方向略正，但区间都跨零；NDCG 基本不变；RankIC、单调性和换手的退化区间没有跨零。效果不是“接近通过”，而是没有形成稳健的 top-tail 排序改善。

## Candidate 相对 LSTM

| 指标 | LSTM | candidate TCN | TCN - LSTM |
|---|---:|---:|---:|
| RankIC | 0.115545 | 0.099181 | -0.016364 |
| Top10% raw return | 0.002952 | 0.003035 | +0.000083，区间跨零 |
| Top10% precision | 0.125208 | 0.114375 | -0.010833 |
| NDCG@Top10% | 0.571931 | 0.564822 | -0.007109 |
| Top10% turnover | 0.570042 | 0.613397 | +0.043354 |

TCN 仍保留一个不显著的 top return 正点估计，但在 RankIC、top precision、NDCG 与 turnover 上继续落后。本轮没有改变“TCN 不一定优于 LSTM；RankIC 合理但不充分”的 v33 结论。

## 原因定位

- 不是感受野、因果 padding、归一化、memmap、父模型漂移或梯度冲突：这些门全部通过。
- 不是 TCN/LSTM 的总体速度目标丢失：即便加入新 loss，TCN 仍为 LSTM 的 `6.23x/5.76x`。
- Top-tail 分量与 SmoothL1 的平均梯度方向同向，但它没有给头部内部次序直接提供监督；它只要求真实 top 整体高于 non-top，因此 precision 可能微升而 NDCG 不升。
- 15 个 candidate 单元中 7 个选择 epoch 0、8 个选择 epoch 1，没有任何单元选择更晚 epoch。RankIC checkpoint selection 很快回退到父模型或首轮，而 Top-tail 的目标可能需要不同的验证选择规则。这是证据支持的下一未知量，不等于可以事后改规则重判 v34。
- 新 loss 的 Python 分组、排序、pair 构造与逐 batch 双分量诊断带来约 26% 训练吞吐损失；需把“objective 固有成本”和“研究期诊断成本”分开基准。

## 下一轮唯一变量建议

v35 不应立即扩大模型或调多个 loss 超参。应保持 v34 candidate 的数据、TCN、父 checkpoints、Top-tail loss、权重、温度、seed/fold 与训练预算完全不变，只比较 checkpoint selection：

- control：v34 的 RankIC checkpoint selection。
- candidate：预注册的 validation top-tail selection score，例如标准化后的 `top_precision + NDCG@top`，同时用 RankIC 不退化约束作可行域，而不是把多个指标随意加权后追最优。

必须保存每个 epoch 的任务对齐验证指标，证明更晚 checkpoint 是否真的改善 top precision/NDCG；若没有，则否定“选择规则抑制增益”的假设，再独立研究能区分头部内部顺序的 listwise/Lambda-style loss。速度方面先把 cosine 诊断降为不参与正式吞吐计时的审计 pass，仍保留 receipt，不得把删除诊断带来的速度提升冒充模型优化。
