# TCN 动态 skip 独立学习率 v22 真实实验结果

## 结论

v22 已在冻结的五年真实股票分钟线、3 个种子、5 折 ordinary validation 上完成 15 个 TCN 训练单元。正式状态为：

`stop_dynamic_skip_lr_unstable_v22`

本轮明确排除了“动态 skip 分支只是因为共同学习率过低而没有学起来”这一简单解释。把 88 个动态参数的 Adam 学习率从 `0.003` 提高到 `0.01` 后，动态权重变化显著增强，但预测效果反而下降：相对静态 TCN 的配对平均 RankIC 为 `-0.000122`，相对共同学习率父候选为 `-0.003785`。这不是 TCN 速度问题；model-step 和端到端速度仍分别为 LSTM 的 `4.044×` 和 `3.894×`。

## 实验身份

- artifact：`artifacts/tcn-dynamic-skip-learning-rate-multiseed-v22`
- receipt：`681d2107dc95f164b4c62dea47dfbde57d76a6f27f72ccf69a847fe2971d8cd7`
- schema：`tcn-dynamic-skip-learning-rate-v22/v1`
- 当前候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-dslr1e2`
- 静态控制：`horizon-skip-c16-chomp-smooth`
- 共同学习率父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- 种子/折：`7,17,27 × 0..4`
- 参数：主干 `6260 @ 0.003`，动态 skip `88 @ 0.01`，总计 `6348`
- sealed test：未访问、未授权

父证据来自 v20 seed 7 和 v21 seeds 17/27。运行器先逐文件复核父 receipt 的输出 SHA-256、源数据 SHA-256、selection 状态和 sealed 标志，然后才开始训练。固定 LSTM 也由两个父 artifact 的完全相同 seed/fold 单元组成，没有重新训练或替换 benchmark。

## 预注册门槛结果

| 门槛 | 观测 | 结果 |
|---|---:|---|
| 候选 mean RankIC ≥ 0.100 | 0.095111 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对静态控制 mean delta ≥ +0.003 | -0.000122 | 失败 |
| 相对父候选 mean delta ≥ +0.001 | -0.003785 | 失败 |
| 每 seed mean delta > 0 | seed 17 为 -0.004252 | 失败 |
| 每 seed 至少 3/5 折不退化 | seed 17 为 1/5；seed 27 为 2/5 | 失败 |
| 1d delta ≥ 0 | -0.005789 | 失败 |
| 2d/3d/5d 保护门槛 | +0.003649 / +0.000949 / +0.000701 | 通过 |
| median samples/s ≥ 5000 | 5307.649 | 通过 |
| 动态 output weight L2 > 1e-12 | 最小 0.556051 | 通过 |
| block variation ≥ 1e-6 | 最小 0.003827 | 通过 |
| 相对父版本 variation ratio ≥ 1.5× | 配对中位数 4.595873× | 通过 |
| simplex error ≤ 1e-6 | 最大 2.384186e-7 | 通过 |
| TCN/LSTM 两种速度均 ≥ 3× | 4.044× / 3.894× | 通过 |

正式 blockers 为：

`mean_rankic_below_gate,mean_rankic_delta_below_gate,parent_mean_rankic_delta_below_gate,per_seed_mean_delta_not_positive,per_seed_fold_stability_below_gate,horizon_1d_degradation_below_gate`

## 逐种子结果

| seed | 静态 TCN | 共同 LR 父候选 | v22 候选 | vs 静态 | vs 父候选 | 不退化折 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.087731 | 0.096378 | 0.091165 | +0.003434 | -0.005213 | 4/5 |
| 17 | 0.098986 | 0.099986 | 0.094734 | -0.004252 | -0.005252 | 1/5 |
| 27 | 0.098982 | 0.100323 | 0.099433 | +0.000451 | -0.000890 | 2/5 |

v22 在每个 seed 上都低于共同学习率父候选。最严重的是 seed 17 fold 4，相对静态控制下降 `-0.016419`、相对父候选下降 `-0.020165`。因此失败不是由一个被其他种子抵消的微小异常造成。

## 动态机制诊断

独立学习率达到了“让动态分支更快学习”的机械目标，但没有转化成更好的预测：

- 父候选 output-weight L2：最小/中位数/最大为 `0.151774 / 0.216710 / 0.410532`。
- v22 output-weight L2：`0.556051 / 0.708846 / 1.130329`。
- 父候选 block variation 中位数：`0.001361`。
- v22 block variation 中位数：`0.007096`。
- 15 个配对单元的 variation ratio 全部大于 1，范围约 `2.54×–14.17×`，配对中位数 `4.60×`。
- variation 与相对父候选 RankIC delta 的描述性相关系数约为 `-0.357`；该相关只用于诊断，不是因果检验。

这组证据更符合“`0.01` 对零初始化输出层和短早停预算来说过于激进，动态尺度选择出现过冲/高方差”的解释。尤其是 1 日期限由 v21 的正增益转为 `-0.005789`，说明过强的样本条件化首先伤害最短期限，而不是模型完全没有利用动态分支。

## TCN 与 LSTM

- TCN mean RankIC：`0.095111`
- 固定 LSTM mean RankIC：`0.115545`
- 配对差：`-0.020435`
- TCN/LSTM model-step speed ratio：`4.044184×`
- TCN/LSTM end-to-end speed ratio：`3.894059×`
- TCN/LSTM 参数：`6348 / 6124`

所以当前状态是：TCN 的速度目标已经稳定满足，但预测效果仍未达到 LSTM；本轮优化甚至低于共同学习率的 TCN 父版本。不能将问题归因于样本量不足、动态参数没有更新或基础设施吞吐不足。

## 下一步边界

v22 不允许事后降低门槛或覆盖本轮结论。若继续，应该另立 v23，并把假设从“动态分支学得不够快”改为“动态分支需要受控的更新轨迹”。最小的新实验应优先考虑固定的两阶段 warm-up 或有上限的 scorer 学习率调度，并显式约束动态 variation 落在父版本与 v22 之间；不应再次扩大动态学习率，也不应改回非 TCN 主模型。
