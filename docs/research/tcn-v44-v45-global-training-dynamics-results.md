# TCN v44–v45 全局训练动力学优化结果

## 结论

本轮按两份预注册提示词完成了真实 2021–2025 股票分钟线、seed 7 × 5 folds 的 ordinary validation：

1. [V44 验证集无关训练轨迹平均](../prompts/tcn-v44-validation-independent-trajectory-average.md)
2. [V45 线性蒸馏退火](../prompts/tcn-v45-linear-distillation-annealing.md)

两轮都没有围绕 Top precision、换手率、top-k、单折或单期限调参。两轮正式状态分别为：

- `stop_trajectory_average_seed7_v44`
- `stop_linear_distillation_annealing_seed7_v45`

因此 seeds 17/27 的 Phase B 均未授权，sealed test 未访问。V42 固定 25% 共识 student 仍是当前 ordinary-validation 上最强的全局排序 TCN，但其既有 Top precision 非补偿门失败没有被推翻，也没有被 V44/V45 修复。

## 前置未知量：teacher reliability weighting 被证伪

V43 建议先检查 teacher 跨种子分歧能否识别不可靠 pseudo target。V44 使用 V42 train-only artifact 逐 fold/horizon 重算后发现：

- 分歧与 teacher 绝对误差的原始 Spearman 均值为 `-0.059393`，20/20 单元为负；
- 分歧与 teacher 共识绝对幅度的 Spearman 为 `-0.388860`；
- 控制共识幅度后，分歧与误差相关转为 `+0.021697`，200 个分层单元中 152 个为正；
- teacher 局部 score margin 控制幅度后与误差的相关约 `-0.001204`。

原始“高分歧反而更准确”是共识幅度混杂；控制后虽恢复正确方向，但效应太弱。基于 agreement 或 margin 做逐样本 teacher 权重没有被授权，避免把混杂变量做成局部优化。

## V44：轨迹平均机制成功，任务结果失败

V42 的 15 个 seed-fold 训练轨迹显示：训练损失从 epoch 1 的 `0.111713` 下降到 epoch 8 的 `0.101023`，同期平均 validation RankIC 从 `0.089573` 降到 `0.084088`。V44 固定使用 epoch 2–8 等权参数平均，最终模型不按 validation 选平均 checkpoint。

机制和基础设施全部通过：

- raw replay 与 V42 预测最大误差：`0.0`；
- raw/average 原始训练 state 最大漂移：`0.0`；
- 在线平均与离线算术平均最大误差：`8.51e-8`；
- 5 折平均更新数：严格 `7`；
- averaged/raw-final 参数最大距离：`0.290363`；
- validation epoch 波动比例：`0.297033`；
- teacher fidelity delta：`+0.026889`；
- 折算 TCN/LSTM model-step 速度：`4.9823x`；
- 单模型、单前向、参数量不变。

相对原始 true-target TCN：

| 指标 | V44 delta |
|---|---:|
| RankIC | `+0.008073` |
| Pearson IC | `+0.004318` |
| Top return | `+0.000138` |
| Top precision | `-0.007375` |
| NDCG | `+0.002894` |
| Quantile monotonicity | `+0.002447` |

相对 V42 pointwise student 六项全部退化：RankIC `-0.006526`、Pearson `-0.005852`、Top return `-0.000337`、Top precision `-0.005750`、NDCG `-0.003243`、monotonicity `-0.009523`。

V44 证明后半程参数不是围绕更优解做无偏抖动；平均虽然大幅降低方差并更忠实地复制 teacher，却更稳定地收敛到较差的真实任务折中。不得调平均起点或终点重跑。

## V45：退火回到真实标签仍未超过 V42

使用 V44 保存的 raw epoch checkpoints 做零重训探针：epoch 与 teacher fidelity Spearman 为 `+0.547619`，epoch 与真实 validation RankIC 为 `-0.571429`，teacher fidelity 与真实 RankIC 为 `-0.476190`。V45 因而只测试唯一的线性 schedule：teacher 权重从 epoch 1 的 `0.25` 逐 epoch降到 epoch 8 的 `0`。

机制与速度通过：

- 40 个 fold-epoch schedule 单元最大公式误差：`2.78e-17`；
- epoch 8 teacher 权重：严格 `0.0`；
- validation teacher cells exposed：`0`；
- teacher fidelity 相对 V42：`-0.007631`，证明退火确实降低了 teacher 跟随；
- 折算 TCN/LSTM model-step 速度：`4.9749x`；
- 5/5 folds、4/4 horizons 相对原始 control 的 RankIC 增量为正。

相对原始 true-target TCN：

| 指标 | V45 delta |
|---|---:|
| RankIC | `+0.012049` |
| Pearson IC | `+0.007703` |
| Top return | `-0.000062` |
| Top precision | `-0.006875` |
| NDCG | `+0.003087` |
| Quantile monotonicity | `+0.006674` |

相对 V42 pointwise student仍为 0/6 改善：RankIC `-0.002551`、Pearson `-0.002467`、Top return `-0.000537`、Top precision `-0.005250`、NDCG `-0.003049`、monotonicity `-0.005295`。

退火比 V44 恢复了部分 RankIC，但没有改善 V42 的任务边界。不得继续搜索指数、余弦、分段或反向 schedule。

## 当前阶段判断

- **速度基础设施已完成。** V44/V45 的单 TCN 仍约为同预算 LSTM 的 `4.98x` model-step 速度，超过最初 3–5x 目标的下界。
- **因果性、WeightNorm、感受野、memmap、训练覆盖和梯度机制均有审计证据。** 本轮没有发现基础设施回归。
- **全局排序效果已获得稳定改善，但正式模型门仍未通过。** V42 在多种子上广泛改善 RankIC/Pearson/Top return/NDCG/monotonicity，却损害正收益命中保护；V44/V45 没有解除该冲突。
- **当前不能落地为 Alpha/部署模型。** 没有新的 sealed 授权，也没有 Phase B 或发布证据。

下一步不应继续在同一 ordinary validation 上调 teacher weight、平均窗口或退火曲线。需要先做新的规格决策：继续坚持“横截面排序为唯一任务、正收益命中只作保护诊断”，还是把正收益命中升级为正式的多任务目标。后者会改变当前 ADR 和模型输出契约，必须先修改 spec，再设计独立数据上的前瞻实验，不能以局部补丁形式加入现有 V42。

## 权威运行证据

- V44 artifact：`artifacts/tcn-v44-trajectory-average-seed7`
- V44 receipt：`a59cf50731d7dffeb2e541c418c394330cf887ecf63b576cbd317d23dd02acb3`
- V45 artifact：`artifacts/tcn-v45-distillation-annealing-seed7`
- V45 receipt：`425571d684a8e0da8aab2452cc7eb2e10504157cd20f544cdd2f166a65b1de79`
- 两轮均记录 `sealed_test_accessed=false`、`alpha_ready=false`。
