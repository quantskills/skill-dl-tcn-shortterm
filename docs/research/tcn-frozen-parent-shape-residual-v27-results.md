# TCN 冻结父模型 shape residual v27 真实实验结果

## 结论

v27 已在冻结的 2021–2025 真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 shape-only TCN 单元。正式状态为：

`stop_frozen_parent_shape_residual_no_gain_v27`

冻结 v21 的 trunk、raw scorer、skip logits 和 heads，仅训练 88 个 shape residual 参数后，candidate mean RankIC 为 `0.099471`，相对 v21 frozen parent 提升 `+0.000575`。这把 v26 联合训练相对 parent 的 `-0.001568` 扭转为正，且三个 seed 均不退化，证明联合训练中的父路径漂移确实是 v26 退化的主要原因之一。

不过 v27 仍未通过预注册门槛：mean RankIC 距 `0.0995` 门槛少 `0.000029`；只有 `7/15` 单元保存了非零 shape checkpoint，低于 `8/15` 的使用门槛。不得因结果接近而事后降低阈值，因此停止于 ordinary validation，不访问 sealed test。

## 实验身份与完整性

- artifact：`artifacts/tcn-frozen-parent-shape-residual-multiseed-v27`
- receipt：`24fa584916019bade27895dc743c73fe4cfb19eb5839ad4336acaef3bdbb7915`
- schema：`tcn-frozen-parent-shape-residual-v27/v1`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025`
- parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- seeds/folds：`7,17,27 × 0..4`
- 总参数：`6436`；冻结参数：`6348`；可训练 shape 参数：`88`
- optimizer：`shape-residual-only-lr-0.003`
- shape scale：`0.25`
- sealed test：未访问、未授权

十五个父 checkpoint 均先通过 receipt 输出哈希验证。加载后 full prediction 与 raw-only parent prediction 的最大绝对误差为 `0`；训练后所有非 shape state 相对源 checkpoint 的最大漂移也为 `0`。epoch-0 raw-only RankIC 与历史 v21 parent 在十五单元上的最大误差为 `0`。因此本轮增量不含父模型漂移。

## 预注册门槛结果

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.0995 | 0.099471 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对 parent mean delta ≥ +0.0005 | +0.000575 | 通过 |
| 三个 seed 相对 parent 均 ≥ 0 | +0.000463 / +0.000505 / +0.000758 | 通过 |
| 每 seed 至少 3/5 folds 不退化 | 5/5 / 5/5 / 5/5 | 通过 |
| 相对静态 mean delta ≥ +0.002 | +0.004238 | 通过 |
| 相对 v26 mean delta ≥ +0.0015 | +0.002143 | 通过 |
| 相对 v25 mean delta ≥ +0.003 | +0.004944 | 通过 |
| 四期限相对 parent delta ≥ -0.001 | 全部满足 | 通过 |
| parent RankIC/prediction 误差 ≤ 1e-7 | 0 / 0 | 通过 |
| frozen state drift = 0 | 0 | 通过 |
| 非零 shape effect 单元 ≥ 8/15 | 7/15 | 失败 |
| simplex error ≤ 1e-6 | 2.384186e-7 | 通过 |
| median samples/s ≥ 4500 | 11727.870 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 10.322× / 8.647× | 通过 |

正式 blockers：

`mean_rankic_below_gate,shape_effect_unit_count_below_gate`

这里的速度只表示“冻结主干、仅训练 88 个参数”的本机 CPU 协议吞吐，不应外推成完整 TCN 从零训练相对 LSTM 的通用速度倍数。

## 逐种子结果

| seed | parent | v27 | vs parent | vs static | vs v25 | vs v26 | 不退化折 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.096378 | 0.096841 | +0.000463 | +0.009110 | +0.007599 | +0.004558 | 5/5 |
| 17 | 0.099986 | 0.100490 | +0.000505 | +0.001504 | +0.005825 | +0.002582 | 5/5 |
| 27 | 0.100323 | 0.101082 | +0.000758 | +0.002099 | +0.001407 | -0.000710 | 5/5 |

seed 27 的 v26 本来已因联合训练偶然超过 parent，因此严格冻结后的 v27 低于 v26；但 v27 在三个 seed 上都不低于各自 parent，稳定性边界明显更强。

## epoch-0 回退与 shape 使用

- best epoch 0：`8/15`，保持 parent 原样；
- best epoch 1：`3/15`；
- best epoch 2：`4/15`；
- 接受 shape 的 7 个单元平均 parent 增量：`+0.001233`；
- 十五单元合并 parent 增量：`+0.000575`。

被接受的 shape 单元为 seed/fold：`7/0, 7/4, 17/0, 17/1, 27/0, 27/2, 27/4`。其余八个单元通过 epoch 0 回退保持 parent，不产生伪退化。

## 期限与 LSTM

| horizon | parent | v27 | delta |
|---:|---:|---:|---:|
| 1d | 0.074976 | 0.075919 | +0.000943 |
| 2d | 0.096801 | 0.097339 | +0.000538 |
| 3d | 0.104829 | 0.105448 | +0.000619 |
| 5d | 0.118978 | 0.119178 | +0.000200 |

四个 horizon 均为正增量，shape 的收益并非靠牺牲某一期限换取。

- v27 TCN mean RankIC：`0.099471`
- 冻结 LSTM mean RankIC：`0.115545`
- 差值：`-0.016074`
- model-step speed ratio：`10.322×`
- end-to-end speed ratio：`8.647×`

v27 改善了 TCN，但普通验证效果仍低于 LSTM，不能宣称预测效果优势。

## 假设判断与下一步边界

“v26 主要被联合训练的 parent drift 拖累”得到确认：严格冻结把 parent delta 从负值恢复为正值，且 integrity 误差和 state drift 都为零。但“shape residual 足以稳定显著提升全部 fold”只得到部分支持：只有 7 个单元达到当前 checkpoint 接受阈值。

训练历史还显示当前 `min_delta=0.0005` 同时承担两种职责：决定是否保存 best checkpoint，以及决定 patience 是否重置。十五单元中，若只看已运行 epoch 的最高分，有两个 epoch-0 回退单元出现 `+0.000338`、`+0.000267` 的小幅正增量；另有一个已接受单元后来出现更高分，但相对当前 best 的增量小于 `0.0005`，因此没有保存。v27 的规则已预注册，不能事后改选。

若继续 v28，应保持模型、parent checkpoint、数据、shape scale、loss 和学习率不变，只把 checkpoint 选择与 patience anchor 解耦：best state 保存任何严格更高的 ordinary-validation score，而 `0.0005` 仅用于重置 patience。必须重新预注册并生成新 artifact，不能覆盖 v27。这样是在修正训练选择 infra，而不是追加模型搜索。
