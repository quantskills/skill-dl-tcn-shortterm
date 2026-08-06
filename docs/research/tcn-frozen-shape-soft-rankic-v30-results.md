# TCN 冻结 shape 分支 soft-RankIC v30 真实实验结果

## 结论

v30 已在冻结的 2021–2025 真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 date-grouped Smooth L1 control 和 15 个 soft-RankIC candidate，共 30 个 TCN 单元。正式状态为：

`stop_shape_rank_no_gain_v30`

本轮通过 contemporaneous control 把 batching 与 loss 分开后，结论非常明确：

- date-grouped Smooth L1 control 的 mean RankIC 为 `0.099791`，相对 v28 提高 `+0.000248`；
- 加入 `0.05 × soft-RankIC` 后，candidate mean RankIC 降到 `0.099422`；
- soft-rank 相对相同 batching control 的 paired delta 为 `-0.000369`，三个 seed 全部退化，15 单元中只有 3 个改善、3 个持平、9 个下降。

因此“当前 pointwise loss 错配可由这个 soft-RankIC surrogate 修复”被真实数据否定；v12 的负面结果在冻结 88 参数 shape 分支和更小权重下再次出现。与此同时，控制组证明 date-grouped batching 本身有正向总体信号，但 seed 27 相对 v28 仍为 `-0.000167`，尚不满足稳定性门槛，不能升级为候选模型或访问 sealed test。

## 实验身份与完整性

- artifact：`artifacts/tcn-frozen-shape-soft-rankic-multiseed-v30`
- receipt：`f45c89ecdb36c997fc320df6cb2ebe5c71a39a141bd2b3cc767d1627701aeea3`
- schema：`tcn-frozen-shape-soft-rankic-v30/v1`
- grouped control：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-grouped-smooth`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-soft-rank-w005-tau01`
- seeds/folds：`7,17,27 × 0..4`
- 数据：23,821 个 5 分钟窗口，8 特征，480 bars 回看
- 每模型总/冻结/可训练参数：`6436/6348/88`
- optimizer：`shape-residual-only-lr-0.003`
- candidate loss：`smooth-l1+0.05-soft-rankic-tau-0.1`
- control loss：`date-grouped-smooth-l1`
- batching：两者均为 `date-grouped`
- checkpoint selection：`best-any-strict-improvement+patience-material-0.0005`
- sealed test：未访问、未授权

完整性观察：

- candidate/control 逐单元 parent checkpoint SHA：全部相同；
- parent prediction 最大误差：`0`；
- raw-only/baseline RankIC 与历史 parent 最大误差：`0`；
- frozen parent state drift：`0`；
- simplex error：`2.384186e-7`；
- candidate/control/v28 paired coverage：`15/15`；
- checkpoints：`30`；receipt 内 44 个输出哈希全部复算一致。

这些证据证明 candidate-control 差异来自预注册的 soft-rank objective，而不是 parent、数据、batching、optimizer 或 checkpoint 选择漂移。

## Candidate 预注册门槛

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.0996 | 0.099422 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对 parent mean delta ≥ +0.00075 | +0.000526 | 失败 |
| 每 seed 5/5 folds 不退化 parent | 5/5 / 5/5 / 5/5 | 通过 |
| 相对 grouped control delta ≥ +0.00015 | -0.000369 | 失败 |
| 三个 seed 相对 grouped control 均 ≥ 0 | -0.000634 / -0.000375 / -0.000097 | 失败 |
| 相对 v28 delta ≥ +0.00015 | -0.000121 | 失败 |
| 三个 seed 相对 v28 均 ≥ 0 | +0.000053 / -0.000152 / -0.000264 | 失败 |
| 相对 static delta ≥ +0.002 | +0.004189 | 通过 |
| 相对 v26 delta ≥ +0.0015 | +0.002094 | 通过 |
| 相对 v25 delta ≥ +0.003 | +0.004895 | 通过 |
| 四期限相对 parent delta ≥ -0.001 | 全部为正 | 通过 |
| shape effect 单元 ≥ 8/15 | 7/15 | 失败 |
| median samples/s ≥ 4500 | 7739.040 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 6.156× / 5.695× | 通过 |

正式 blockers：

`mean_rankic_below_gate,parent_mean_rankic_delta_below_gate,shape_effect_unit_count_below_gate,v28_mean_rankic_delta_below_gate,per_seed_v28_delta_negative,grouped_control_mean_rankic_delta_below_gate,per_seed_grouped_control_delta_negative`

## Batching 与 rank loss 的独立贡献

| 组件 | Mean RankIC | 相对 v28 | 相对 parent |
|---|---:|---:|---:|
| v28 random-batch Smooth L1 | 0.099543 | — | +0.000647 |
| v30 date-grouped Smooth L1 | 0.099791 | +0.000248 | +0.000895 |
| v30 date-grouped soft-RankIC | 0.099422 | -0.000121 | +0.000526 |

所以本轮的总差分可精确分解为：

- batching contribution：`+0.000248`；
- rank-loss contribution：`-0.000369`；
- net candidate vs v28：`-0.000121`。

逐单元分布：

| Paired comparison | 改善 | 持平 | 退化 |
|---|---:|---:|---:|
| candidate vs grouped control | 3 | 3 | 9 |
| candidate vs v28 | 3 | 6 | 6 |
| grouped control vs v28 | 9 | 3 | 3 |

soft-rank 的负贡献不是单个异常 fold：三个 seed 均低于相同 batching control。它还使保存非零 shape 的单元从 control 的 `11/15` 降至 `7/15`，说明 rank surrogate 没有增强 shape 利用，反而使更多单元回退 epoch 0。

## 逐种子与期限结果

| seed | parent | grouped control | candidate | candidate vs control | candidate vs v28 |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.096378 | 0.097624 | 0.096990 | -0.000634 | +0.000053 |
| 17 | 0.099986 | 0.100781 | 0.100406 | -0.000375 | -0.000152 |
| 27 | 0.100323 | 0.100968 | 0.100871 | -0.000097 | -0.000264 |

| horizon | parent | candidate | delta |
|---:|---:|---:|---:|
| 1d | 0.074976 | 0.076011 | +0.001035 |
| 2d | 0.096801 | 0.097149 | +0.000348 |
| 3d | 0.104829 | 0.105036 | +0.000207 |
| 5d | 0.118978 | 0.119493 | +0.000515 |

candidate 的四期限均保护了 parent，但这不足以抵消其对 control 和 v28 的总体退化。

## Grouped control 的边界

grouped Smooth L1 是目前 mean RankIC 最高的冻结 shape 变体：

- mean RankIC：`0.099791`；
- vs parent：`+0.000895`；
- vs v28：`+0.000248`；
- vs static：`+0.004558`；
- vs v26：`+0.002463`；
- shape effect：`11/15`；
- model-step/end-to-end speed：`10.183× / 8.570×`。

但其相对 v28 的 seed delta 为 `+0.000687 / +0.000223 / -0.000167`，seed 27 未通过非退化门槛。它是值得保留的机制发现，不是已确认候选；不能把总体均值优势替代预注册的跨 seed 稳定性。

## LSTM 与速度

soft-RankIC candidate：

- mean RankIC：`0.099422`；
- 冻结 LSTM mean RankIC：`0.115545`；
- 差值：`-0.016123`；
- model-step speed ratio：`6.156×`；
- end-to-end speed ratio：`5.695×`。

grouped Smooth L1 control：

- mean RankIC：`0.099791`；
- 与 LSTM 差值：`-0.015755`；
- model-step speed ratio：`10.183×`；
- end-to-end speed ratio：`8.570×`。

soft-rank 的成对计算带来约 40% 的 model-step 吞吐损失，但仍保留超过 3× 的冻结 shape 训练速度。预测效果仍显著低于 LSTM。

## 假设判断与下一步边界

1. “soft-RankIC 修复目标错配”：否定；
2. “改善来自 date-grouped batching”：得到总体支持，但 seed 27 不稳定；
3. “rank surrogate 干扰 Smooth L1”：得到三个 seed 一致支持；
4. “shape 信息上限”：尚不能完全确认，因为 grouped control 仍产生新的总体增量。

下一轮不得继续扫描 soft-rank weight 或 temperature。若继续 v31，最小可证伪方向应只保留 grouped Smooth L1，研究其 seed 27 不稳定来源；优先做 batching 轨迹诊断或预注册的梯度累积稳定化，而不是再改模型或 loss。任何新干预必须以本轮 grouped control 为直接 paired control，并继续禁止 sealed test。
