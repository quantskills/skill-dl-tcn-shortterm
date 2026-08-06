# TCN 动态多尺度 v21 多种子确认结果

## 结论

v21 在冻结的真实 PandaData 五年分钟线、seeds 17/27、五折 ordinary validation 上得到有效负结果：

`stop_dynamic_multiscale_unstable_v21`

动态多尺度候选在两个新 seed 的平均 RankIC 都略高于同 seed 静态 `HorizonSkipTCN`，但新 seeds 的配对平均增量只有 `+0.001170`，没有达到预注册的 `+0.003`；seed 27 也只有 2/5 折不退化。因此 v20 seed 7 的 `+0.008647` 增益幅度未稳定复现。

该停止不是速度、容量、因果性、动态机制或绝对 RankIC 门禁导致的。当前结果不授权 sealed test、部署或实盘。

## 实验身份

- artifact：`artifacts/tcn-dynamic-multiscale-multiseed-v21`
- receipt：`797f395b7bc853e0d818d0fff99650aaa8f16be7fbb6822eeb22ae0da7a5cde1`
- schema：`tcn-dynamic-multiscale-multiseed-v21/v1`
- 父 v20 receipt：`9771fb41fa3dfd9a3a01ab6fe50ddafd0c69fd7d43b08a29bc05ba7d9710a76b`
- 固定 LSTM receipt：`c0d7a8d4e976f6aade52cd6ffd40d143a7d6bb54c8de6a48025709960ed6eada`
- receipt 输出：32 个文件，SHA-256 全部复核一致
- checkpoints：20 个
- sealed test：未访问

第一次执行因 15 分钟基础设施上限在固定 LSTM 重复训练阶段终止，未生成 selection 或 receipt，不构成一次模型试验。正式执行复用了 v14 在完全相同数据、seeds 17/27、folds、LSTM 超参数和当前环境下已经完成的 10 个真实 LSTM measurements；runner 验证了其 receipt、全部输出哈希、源 SHA、环境、超参数和单元覆盖。

## 新 seeds 正式判定

| seed | 静态 control mean RankIC | 动态候选 mean RankIC | 配对增量 | 不退化折数 |
|---:|---:|---:|---:|---:|
| 17 | 0.098986 | 0.099986 | +0.000999 | 3/5 |
| 27 | 0.098982 | 0.100323 | +0.001341 | 2/5 |
| 聚合 | 0.098984 | 0.100155 | +0.001170 | 5/10 |

候选 10 个单元 RankIC 全部为正，聚合 mean RankIC `0.10015451483060175` 高于 `0.09` 门槛。失败只来自：

1. `mean_rankic_delta_below_gate`：`+0.001170 < +0.003`。
2. `per_seed_fold_stability_below_gate`：seed 27 只有 2/5 folds 不退化，低于 3/5。

两个 seed 的平均 delta 都大于 0，因此没有触发 `per_seed_mean_delta_not_positive`。

## 逐期限结果

| 期限 | control RankIC | candidate RankIC | delta | 门槛 | 结果 |
|---:|---:|---:|---:|---:|---|
| 1 日 | 0.079758 | 0.081490 | +0.001731 | ≥0 | 通过 |
| 2 日 | 0.094392 | 0.095290 | +0.000898 | ≥-0.003 | 通过 |
| 3 日 | 0.103965 | 0.106810 | +0.002845 | ≥-0.005 | 通过 |
| 5 日 | 0.117821 | 0.117028 | -0.000793 | ≥-0.005 | 通过 |

动态多尺度对 1/2/3 日仍为正增益，5 日出现很小退化但在预注册保护范围内。期限结构不是 v21 停止原因。

## 动态机制与容量

- control/candidate 参数量：`6260 / 6348`
- 动态容量：`88`
- 动态输出 weight L2 最小值：`0.151774`
- block 权重样本变异最小值：`0.000793551`
- simplex error 最大值：`2.384186e-7`

10 个 seed/fold 单元的动态 block 权重变异均明显高于 `1e-6`，输出权重非零，simplex 误差低于 `1e-6`。因此候选确实在按样本选择 dilation 尺度；问题是这种选择带来的增益较小且折间不够稳定。

## 吞吐与 LSTM

- 候选 median samples/s：`5281.796`
- model-step 相对 LSTM：`4.001137x`
- 端到端相对 LSTM：`3.854201x`
- TCN mean RankIC：`0.1001545`
- LSTM mean RankIC：`0.1175205`
- 配对差：`-0.0173659`

速度门禁全部通过，并且新 seeds 的 TCN 绝对 RankIC 高于 seed 7 的 `0.0963785`；但同 seed 静态 control 同样提高到约 `0.098984`，所以动态多尺度本身的增量被压缩。LSTM 预测效果仍明显领先。

## 三 seed 描述性汇总

| seed | candidate mean RankIC | control mean RankIC | delta | 不退化折数 |
|---:|---:|---:|---:|---:|
| 7 | 0.096378 | 0.087731 | +0.008647 | 5/5 |
| 17 | 0.099986 | 0.098986 | +0.000999 | 3/5 |
| 27 | 0.100323 | 0.098982 | +0.001341 | 2/5 |

三 seed 简单平均增量约 `+0.003662`，四期限三 seed 汇总 delta 也全部为正。但该描述性结果不能覆盖正式失败，因为 seed 7 是模型选择所使用的父 seed；v21 的确认判定按预注册只使用新的 seeds 17/27。

## 诊断与下一方向

v21 排除了以下原因：

- TCN 速度不足；
- 动态参数未更新；
- block 权重没有样本差异；
- simplex 数值错误；
- 单一预测期限严重退化；
- 某个新 seed 的平均方向完全为负。

剩余问题是动态多尺度增益的幅度和折稳定性。新 seeds 的 block 权重变异范围约 `0.00079–0.00183`，整体低于 seed 7 的最高值 `0.00454`；同时 scorer 输出层零初始化意味着第一步隐藏层没有梯度，88 个动态参数与 6260 个基础参数共用学习率和短早停预算，可能造成动态路径在不同初始化下学习强度不一致。

后续若继续研究，应另立预注册实验，优先测试动态 skip 参数的独立、受限学习率或两阶段 warm-up，并同时以 seeds 7/17/27 做开发判定；不得事后把 v21 的 `+0.003` 门槛降到 `+0.001`。当前 v21 到此停止。
