# TCN 冻结父模型解耦 checkpoint 选择 v28 真实实验结果

## 结论

v28 已在冻结的 2021–2025 真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 shape-only TCN 单元。正式状态为：

`stop_decoupled_checkpoint_no_gain_v28`

v28 只将 checkpoint 保存阈值从 patience 的 `0.0005` 中解耦为 `0.0`，模型、父 checkpoint、训练数据、RNG、优化器和 early-stopping 轨迹全部保持 v27 不变。v28 与 v27 的 56 个 `(seed,fold,epoch)` mean RankIC 逐项最大误差为 `0`，coverage 完全一致；selected best 与 v28 epoch-history 最大值的误差也为 `0`。

解耦后，mean RankIC 从 v27 的 `0.099471` 提高到 `0.099543`，通过 `0.0995` 绝对门槛；shape 生效单元从 `7/15` 增至 `9/15`，也通过使用门槛。说明 v27 的选择 infra 确实丢失了已经训练出来的增量。

但恢复幅度不足预注册增量门槛：相对 frozen parent 为 `+0.000647 < +0.0007`，相对 v27 为 `+0.000072 < +0.00015`。不得因结果接近而降低阈值，因此停止于 ordinary validation，不访问 sealed test。

## 实验身份与完整性

- artifact：`artifacts/tcn-decoupled-checkpoint-selection-multiseed-v28`
- receipt：`70009fbb81c7d0846732a89f3ca09114f7a17c93780e2d749c3505c83e9bcc28`
- schema：`tcn-decoupled-checkpoint-selection-v28/v1`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-decoupled-selection`
- parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v27：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025`
- seeds/folds：`7,17,27 × 0..4`
- 总参数：`6436`；冻结参数：`6348`；可训练 shape 参数：`88`
- optimizer：`shape-residual-only-lr-0.003`
- checkpoint selection：`best-any-strict-improvement+patience-material-0.0005`
- sealed test：未访问、未授权

完整性观察：

- v28/v27 trajectory coverage：完全一致，56 行；
- trajectory RankIC 最大误差：`0`；
- selected best 与 observed epoch max 误差：`0`；
- parent prediction 最大误差：`0`；
- raw-only RankIC 与历史 parent 最大误差：`0`；
- frozen state drift：`0`；
- simplex error：`2.384186e-7`。

这些证据证明 v28 的增量完全来自 checkpoint 选择，不是重新训练差异、父路径漂移或数据变化。

## 预注册门槛结果

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.0995 | 0.099543 | 通过 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对 parent mean delta ≥ +0.0007 | +0.000647 | 失败 |
| 三个 seed 相对 parent 均 ≥ 0 | +0.000558 / +0.000573 / +0.000811 | 通过 |
| 每 seed 5/5 folds 不退化 | 5/5 / 5/5 / 5/5 | 通过 |
| 相对 v27 mean delta ≥ +0.00015 | +0.000072 | 失败 |
| 三个 seed 相对 v27 均 ≥ 0 | +0.000096 / +0.000068 / +0.000053 | 通过 |
| 相对静态 mean delta ≥ +0.002 | +0.004310 | 通过 |
| 相对 v26 mean delta ≥ +0.0015 | +0.002216 | 通过 |
| 相对 v25 mean delta ≥ +0.003 | +0.005016 | 通过 |
| 四期限相对 parent delta ≥ -0.001 | 全部满足 | 通过 |
| trajectory coverage/误差 | 完全一致 / 0 | 通过 |
| selected best 误差 ≤ 1e-12 | 0 | 通过 |
| shape effect 单元 ≥ 8/15 | 9/15 | 通过 |
| median samples/s ≥ 4500 | 11405.123 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 9.790× / 8.248× | 通过 |

正式 blockers：

`parent_mean_rankic_delta_below_gate,v27_mean_rankic_delta_below_gate`

## 解耦实际恢复了什么

v28 相对 v27 改变了三个最终 best epoch：

| seed/fold | v27 best | v28 best | v28 vs parent | v28 vs v27 |
|---|---:|---:|---:|---:|
| 7/4 | epoch 2 | epoch 4 | +0.001149 | +0.000479 |
| 17/3 | epoch 0 | epoch 1 | +0.000338 | +0.000338 |
| 27/1 | epoch 0 | epoch 1 | +0.000267 | +0.000267 |

其中 seed 17/fold 3 和 seed 27/fold 1 是新增的非零 shape 单元，使 effect coverage 从 7 增至 9。全体 leaderboard 共记录 4 次“保存 checkpoint 但不重置 patience”的事件，其中一次后来被更高 checkpoint 覆盖。

best epoch 分布由 v27 的 `epoch0=8, epoch1=3, epoch2=4` 变为：

- epoch 0：`6/15`；
- epoch 1：`5/15`；
- epoch 2：`3/15`；
- epoch 4：`1/15`。

## 逐种子与期限结果

| seed | parent | v28 | vs parent | vs static | vs v26 | vs v27 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.096378 | 0.096937 | +0.000558 | +0.009206 | +0.004654 | +0.000096 |
| 17 | 0.099986 | 0.100558 | +0.000573 | +0.001572 | +0.002650 | +0.000068 |
| 27 | 0.100323 | 0.101135 | +0.000811 | +0.002153 | -0.000657 | +0.000053 |

| horizon | parent | v28 | delta |
|---:|---:|---:|---:|
| 1d | 0.074976 | 0.075777 | +0.000801 |
| 2d | 0.096801 | 0.097461 | +0.000660 |
| 3d | 0.104829 | 0.105483 | +0.000654 |
| 5d | 0.118978 | 0.119452 | +0.000474 |

所有 seed 与 horizon 均为非负增量，但 seed 27 仍低于联合训练偶然较强的 v26。

## LSTM 与速度

- v28 TCN mean RankIC：`0.099543`
- 冻结 LSTM mean RankIC：`0.115545`
- 差值：`-0.016002`
- model-step speed ratio：`9.790×`
- end-to-end speed ratio：`8.248×`
- median samples/s：`11405.123`

速度仍只是“冻结 6348 个参数、训练 88 个参数”的本机协议结果；预测效果仍低于 LSTM，不能宣称效果优势或候选模型资格。

## 假设判断与下一步边界

“checkpoint/patience 耦合丢失了已训练增量”得到确认，但恢复量只有 `+0.000072`，说明它不是剩余效果差距的主要来源。v28 已把选择 infra 修正完毕，下一轮不应继续修改 checkpoint 门槛或事后放宽 gate。

剩余问题转向 shape-only 优化本身：当前 Adam `0.003` 在 1–2 个 epoch 内把 shape output L2 推到约 `0.4–1.0`，9 个单元受益、6 个单元仍回退 parent，存在步长偏大或 pointwise Smooth L1 与 RankIC 目标不完全对齐的可能。

若继续 v29，优先做单变量、冻结 parent 的 shape-only 学习率 bracket：保留 `0.003` 作为历史 control，只新增预注册 `0.001` candidate；训练选择沿用 v28，不改模型、数据、shape scale、loss 或 epoch。只有更低步长在三个 seed、四期限和相对 v28 paired gate 上稳定提升，才进一步考虑 rank-aligned loss；否则停止学习率方向。不得覆盖 v28 或访问 sealed test。
