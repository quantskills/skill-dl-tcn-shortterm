# TCN v43 梯度归一化全横截面共识蒸馏结果

## 结论

v43 已按[预注册提示词](../prompts/tcn-v43-gradient-normalized-listwise-consensus.md)在真实 2021–2025 股票分钟线 ordinary validation 上完成 seed 7 × 5 folds。正式状态为：

`stop_listwise_consensus_seed7_v43`

Phase B seeds 17/27 未授权、未运行；sealed test 未访问。

本轮证伪了“v42 点式 teacher target 混合丢失横截面排序，可由梯度归一化 teacher listwise surrogate 修复”的假设。listwise 机制和梯度定标均正确生效，速度仍超过原定目标，但 candidate 相对 v42 点式 student 的六项指标全部下降，且对冻结 teacher ensemble 的 score fidelity 也下降。不得继续搜索梯度比例或 temperature。

## 红灯诊断

`tasks/diagnose_tcn_v43_task_contract_gap.py` 连续两次稳定复现 v42 症状：

- RankIC：`+0.014508`
- Top return：`+0.000336`
- Top precision：`-0.003344`
- 六项改善：`5/6`
- Top precision 负向分布：3 个 horizons、3 个 folds

这证明问题不局限于换手率、单个 fold 或单个 horizon。

## 唯一变量

v43 保持真实 rank target 不变：

`L_true = SmoothL1(student_score, true_rank_target)`

在每个完整 `(signal_date, horizon)` 横截面上，以冻结 teacher consensus rank 构造 soft-rank correlation loss `L_teacher`。teacher prediction-gradient 经无梯度比例缩放，使其范数固定为真实目标梯度的 25%：

`L = L_true + 0.25 × stop_gradient(||g_true|| / ||g_teacher||) × L_teacher`

固定 temperature 为 `0.1`，不使用 top-k、收益符号、方向标签、换手率或 LSTM teacher。

## 完整性与机制证据

- teacher seeds/folds：冻结 `{7,17,27} × {0..4}`。
- teacher targets：复用 v42 train-only artifact 和 fingerprint。
- validation teacher cells exposed：`0`。
- 真实 training target override：`false`。
- 8 epochs × 5 folds 的 teacher/true prediction-gradient ratio 中位数：`0.2500`，全部位于预注册 `[0.20,0.30]`。
- candidate 是单个 16-channel TCN，推理前向次数为 1。
- `sealed_test_accessed=false`。

## Candidate 相对原始 TCN control

| 指标 | delta |
|---|---:|
| RankIC | `+0.010080` |
| Pearson IC | `+0.004983` |
| Top return | `-0.000166` |
| Top precision | `-0.006500` |
| NDCG@top | `-0.003910` |
| Quantile monotonicity | `+0.009212` |

RankIC 在 5/5 folds、4/4 horizons 为正，95% CI low 为 `+0.004803`。但只有 3/6 指标改善，Top return、Top precision 和 NDCG 均突破非补偿下限，因此相对原始 control 也没有通过全局门。

## Candidate 相对 v42 点式 student

| 指标 | delta |
|---|---:|
| RankIC | `-0.004519` |
| Pearson IC | `-0.005187` |
| Top return | `-0.000641` |
| Top precision | `-0.004875` |
| NDCG@top | `-0.010046` |
| Quantile monotonicity | `-0.002758` |

六项改善数量为 `0/6`。candidate 对冻结 teacher validation ensemble 的平均 score RankIC 为 `0.892638`，v42 点式 student 为 `0.897647`，fidelity delta `-0.005009`。因此失败不是“更像 teacher 但真实指标错配”，而是当前 listwise surrogate 连 teacher 本身也没有复制得更好。

## 速度

- 相对 v42 点式 student model-step retention：`0.8403`
- complete-cycle retention：`0.8606`
- 折算 TCN/LSTM model-step ratio：`4.0130×`

速度门通过；v43 的失败集中在预测与 teacher fidelity，不是基础设施或 TCN 速度回归。

## 假设判断与后续边界

1. “点式混合丢失 teacher 全局排序”：当前 listwise surrogate 下被否定。
2. “loss 没生效或尺度错误”：被否定，梯度比例审计精确通过。
3. “速度成本使 TCN 失去 3–5× 优势”：被否定，仍为 `4.01×`。
4. 下一独立未知量是 teacher 跨种子分歧是否能识别不可靠的 pseudo target。若继续，应使用 train-only teacher agreement 作为全局可靠性权重，保持真实 target 不变；不得沿 v43 搜索 ratio/temperature，也不得围绕 Top precision、top-k 或换手率做局部优化。

## 权威证据

- Artifact：`artifacts/tcn-v43-listwise-consensus-seed7`
- Receipt ID：`f939087e76b2ecfc9e04c698ba37f240dd88477e722465b5769eb53d79577339`
- 结论上限：ordinary-validation single-TCN research evidence；`alpha_ready=false`。
