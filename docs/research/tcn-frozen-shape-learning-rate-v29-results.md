# TCN 冻结 shape 分支学习率 v29 真实实验结果

## 结论

v29 已在冻结的 2021–2025 真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 shape-only TCN 单元。正式状态为：

`stop_frozen_shape_lr_no_gain_v29`

v29 唯一修改是将冻结 shape 分支的 Adam 学习率从 v28 的 `0.003` 降到 `0.001`；模型、6348 参数 parent checkpoint、数据、loss、shape scale、epoch、patience 和 checkpoint 选择全部冻结。完整性和速度门槛全部通过，但 mean RankIC 从 v28 的 `0.099543` 降到 `0.099225`，paired delta 为 `-0.000318`。三个 seed 相对 v28 都退化，15 个单元中只有 4 个提升、3 个持平、8 个下降。

因此“`0.003` 过冲，降低至 `0.001` 会改善预测”的假设被真实数据否定。低学习率确实把非零 shape output L2 中位数从 v28 的 `0.459924` 降到 `0.274953`，但更新幅度下降没有转化为效果提升。当前证据更支持：`0.001` 在固定的 8 epoch / patience 2 协议下过于保守，或 pointwise Smooth L1 对 RankIC 的 shape-only 可优化信号已接近上限；不能宣称 TCN 预测效果超过 LSTM。

## 实验身份与完整性

- artifact：`artifacts/tcn-frozen-shape-learning-rate-multiseed-v29`
- receipt：`1dfe144cfc40f850fbf4e44ccb41ff6fa5a6813841670c9603b062e6868e4b5c`
- schema：`tcn-frozen-shape-learning-rate-v29/v1`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-lr001`
- v28：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-decoupled-selection`
- seeds/folds：`7,17,27 × 0..4`
- 数据：23,821 个 5 分钟窗口，8 特征，480 bars 回看
- 总参数：`6436`；冻结参数：`6348`；可训练 shape 参数：`88`
- optimizer：`shape-residual-only-lr-0.001`
- checkpoint selection：`best-any-strict-improvement+patience-material-0.0005`
- sealed test：未访问、未授权

完整性观察：

- parent prediction 最大误差：`0`；
- raw-only/baseline RankIC 与历史 parent 最大误差：`0`；
- frozen parent state drift：`0`；
- simplex error：`2.384186e-7`；
- v29/v28 paired coverage：`15/15`；
- checkpoints：`15`；receipt 内 29 个输出哈希全部复算一致。

这些证据把结果差异限定在预注册的 shape-only 学习率变化，而不是模型、数据、parent、选择 infra 或 artifact 漂移。

## 预注册门槛结果

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.0996 | 0.099225 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对 parent mean delta ≥ +0.00075 | +0.000329 | 失败 |
| 三个 seed 相对 parent均 ≥ 0 | +0.000053 / +0.000346 / +0.000589 | 通过 |
| 每 seed 5/5 folds 不退化 parent | 5/5 / 5/5 / 5/5 | 通过 |
| 相对 v28 mean delta ≥ +0.00015 | -0.000318 | 失败 |
| 三个 seed 相对 v28 均 ≥ 0 | -0.000505 / -0.000227 / -0.000222 | 失败 |
| 相对静态 mean delta ≥ +0.002 | +0.003992 | 通过 |
| 相对 v26 mean delta ≥ +0.0015 | +0.001898 | 通过 |
| 相对 v25 mean delta ≥ +0.003 | +0.004698 | 通过 |
| 四期限相对 parent delta ≥ -0.001 | 全部为正 | 通过 |
| shape effect 单元 ≥ 8/15 | 10/15 | 通过 |
| median samples/s ≥ 4500 | 12182.644 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 10.446× / 8.801× | 通过 |

正式 blockers：

`mean_rankic_below_gate,parent_mean_rankic_delta_below_gate,v28_mean_rankic_delta_below_gate,per_seed_v28_delta_negative`

## 逐种子与期限结果

| seed | parent | v29 | vs parent | vs static | vs v26 | vs v28 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.096378 | 0.096432 | +0.000053 | +0.008701 | +0.004149 | -0.000505 |
| 17 | 0.099986 | 0.100331 | +0.000346 | +0.001345 | +0.002423 | -0.000227 |
| 27 | 0.100323 | 0.100912 | +0.000589 | +0.001930 | -0.000879 | -0.000222 |

| horizon | parent | v29 | delta |
|---:|---:|---:|---:|
| 1d | 0.074976 | 0.075256 | +0.000280 |
| 2d | 0.096801 | 0.097106 | +0.000306 |
| 3d | 0.104829 | 0.105345 | +0.000516 |
| 5d | 0.118978 | 0.119194 | +0.000216 |

所有期限仍略优于 raw parent，但幅度都小于 v28。逐单元相对 v28 的 delta 中位数为 `-0.0000418`，最差 `-0.001470`，最好 `+0.000369`；分布并非由单个异常 fold 驱动。

## 学习率机制判断

v29 best epoch 分布为 `epoch0=5, epoch1=2, epoch2=6, epoch3=1, epoch5=1`，shape 生效单元 `10/15`，比 v28 的 `9/15` 多一个；但相对 v28 只有 `4` 个单元提升、`3` 个持平、`8` 个下降。

更新幅度证据：

| 指标 | v28 lr=0.003 | v29 lr=0.001 | 变化 |
|---|---:|---:|---:|
| effect 单元数 | 9 | 10 | +1 |
| 非零 shape output L2 中位数 | 0.459924 | 0.274953 | -40.2% |
| 非零 shape output L2 均值 | 0.552775 | 0.299288 | -45.9% |
| 全单元 shape L2 中位数 | 0.399132 | 0.153098 | -61.6% |

学习率降低确实按预期抑制了 shape 权重，但更小权重不是更好泛化的充分条件。尤其 seed 27/fold 0 已训练到 epoch 5 仍比 v28 低 `-0.000209`，所以结果不能只归因于 patience 太短。更准确的判断是：在当前 Smooth L1 目标下，`0.003` 比 `0.001` 更接近有效步长区域；继续向更低学习率搜索没有证据基础。

## LSTM 与速度

- v29 TCN mean RankIC：`0.099225`
- 冻结 LSTM mean RankIC：`0.115545`
- 差值：`-0.016320`
- model-step speed ratio：`10.446×`
- end-to-end speed ratio：`8.801×`
- median samples/s：`12182.644`

TCN 的速度目标已明显完成；当前瓶颈仍是 ordinary-validation 预测效果。速度来自冻结 parent、只训练 88 个 shape 参数的协议，不能外推为所有 TCN 训练任务的固定倍数。

## 下一步边界

v29 否定了把 shape-only 学习率从 `0.003` 降至 `0.001` 的方向，下一轮不得继续盲目降低学习率或放宽 gate。若继续优化，最小可证伪方向应转向 **保持 v28 的 `0.003` 与全部 infra 不变，只修改 shape 分支的训练目标，使其与横截面 RankIC 对齐**；例如预注册一个很小权重的 differentiable rank loss，并以 v28 为 paired historical control。该方向必须先验证 loss 只作用于 88 个 shape 参数、无跨日/未来信息、不会改变 parent path，再做相同 15 单元 ordinary validation。sealed test 仍不得访问。
