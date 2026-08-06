# TCN 有符号时间适配器 v14 多种子确认结果

日期：2026-08-04

状态：`stop_signed_candidate_unstable_v14`

候选：`signed-context-c16-chomp-smooth`

Receipt：`c0d7a8d4e976f6aade52cd6ffd40d143a7d6bb54c8de6a48025709960ed6eada`

## 结论

v13 的 seed-7 增益没有在预注册的新 seeds 17/27 上复现。有符号时间适配器在确认集的 mean RankIC 为 `0.087236`，低于 `0.09` 门槛，也低于参数匹配 simplex TCN 的 `0.093183`，平均差 `-0.005947`。seed 17 只有 1/5 folds 不退化，seed 27 为 3/5；聚合 3 日 RankIC 下降 `-0.013606`。因此候选停止在 ordinary validation，不授权 sealed test。

失败不是 TCN infra 或速度问题。确认集上候选相对 LSTM 的 model-step 速度为 `4.571x`，端到端速度为 `4.335x`，均超过预注册的 `3x` 门槛；候选 median samples/s 为 `5820.35`。当前已验证的事实是：TCN 达到了原定 3–5x 速度目标，但这个无约束 signed adapter 没有形成可复现的预测增益。

## 预注册门槛

| 门槛 | 观察值 | 结果 |
|---|---:|---:|
| 候选 mean RankIC `>=0.09` | 0.087236 | 失败 |
| 候选正 RankIC 单元 `10/10` | 10/10 | 通过 |
| 候选相对控制平均增益 `>0` | -0.005947 | 失败 |
| 每个 seed 至少 3/5 folds 不退化 | seed 17：1/5；seed 27：3/5 | 失败 |
| 3 日 delta `>=-0.005` | -0.013606 | 失败 |
| 5 日 delta `>=-0.005` | -0.000553 | 通过 |
| median samples/s `>=5000` | 5820.35 | 通过 |
| model-step TCN/LSTM `>=3x` | 4.571x | 通过 |
| 端到端 TCN/LSTM `>=3x` | 4.335x | 通过 |

对应 blockers：`mean_rankic_below_gate`、`control_mean_rankic_degradation`、`per_seed_fold_stability_below_gate`、`horizon_3d_degradation_below_gate`。

## 确认集逐 seed 结果

| Seed | Simplex control | Signed candidate | Delta | 不退化 folds | 候选正 folds |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.097060 | 0.086175 | -0.010886 | 1/5 | 5/5 |
| 27 | 0.089305 | 0.088297 | -0.001008 | 3/5 | 5/5 |

逐折 delta：

- seed 17：`-0.028581`、`+0.012032`、`-0.009264`、`-0.018199`、`-0.010415`。
- seed 27：`-0.011101`、`+0.004427`、`+0.003320`、`-0.005568`、`+0.003882`。

seed 17 是主要失败来源，但 seed 27 的平均增益同样没有转正，所以不能把失败解释为单一异常 seed。

## 确认集逐 horizon 结果

| Horizon | Simplex control | Signed candidate | Delta |
|---|---:|---:|---:|
| 1 日 | 0.066252 | 0.067911 | +0.001659 |
| 2 日 | 0.091554 | 0.080267 | -0.011287 |
| 3 日 | 0.102691 | 0.089085 | -0.013606 |
| 5 日 | 0.112234 | 0.111681 | -0.000553 |

v13 中最明显的 1/2 日收益没有跨 seed 保持：1 日只剩轻微正增益，2 日转为明显负增益，3 日负迁移扩大。问题集中在 horizon-specific 时间滤波的优化稳定性，而不是感受野覆盖或 causal padding。

## LSTM 公平比较

- Signed TCN mean RankIC：`0.0872361`。
- LSTM mean RankIC：`0.1175205`。
- 配对平均差：`-0.0302844`。
- TCN/LSTM model-step：`4.5706x`。
- TCN/LSTM 端到端：`4.3352x`。
- TCN 参数量：6524；LSTM 参数量：6124。

速度目标已经完成，预测效果仍未追平 LSTM。本轮没有任何证据支持放弃 TCN；证据只否定了“当前无约束 signed adapter 在 seed 7 上的增益可以直接推广”这一更窄的假设。

## seeds 7/17/27 描述性合并

确认决策只使用新 seeds 17/27。以下三 seed 汇总仅用于理解稳定性，不回填确认门槛：

| 指标 | 三 seed 结果 |
|---|---:|
| Simplex mean RankIC | 0.091284 |
| Signed mean RankIC | 0.088750 |
| Signed - simplex | -0.002535 |
| Signed 不退化 folds | 7/15 |
| Signed 正 RankIC folds | 15/15 |
| Signed 最差 fold | 0.011317 |
| Signed - LSTM | -0.026796 |
| TCN/LSTM model-step | 4.372x |
| TCN/LSTM 端到端 | 4.192x |

三 seed 的 horizon delta 为：1 日 `+0.006672`、2 日 `-0.005551`、3 日 `-0.009722`、5 日 `-0.001537`。signed 候选的绝对 RankIC 在三个 seed 中均为正，seed 均值范围为 `0.086175..0.091777`；“不稳定”主要指它相对 simplex 的结构增益不可复现，而不是模型完全失去预测信号。

## 原因定位

1. **不是样本吞吐或模型规模。** 两条 TCN 均为 6524 参数，确认集速度通过，数据读取也没有成为 3x 门槛的阻塞项。
2. **不是已知的 TCN 基础坑。** 模型继续使用 strict causal chomp、WeightNorm、覆盖 480 步输入的 dilated receptive field、memmap 数据和相同 expanding folds；v14 没有改变这些基础条件。
3. **无约束时间权重的收益依赖初始化。** 候选与 simplex 初始函数相同，但解除 simplex 后，各 seed/fold 会学习不同符号与幅度的日内滤波；三 seed 配对 delta 的标准差为 `0.012317`，大于平均效应绝对值 `0.002535`。
4. **中周期 horizon 出现负迁移。** 2/3 日是确认集的主要损失来源，尤其 seed 17 的 3 日 delta 为 `-0.029689`；这与单纯增加感受野无关，而更像共享 trunk、四个 horizon 和无约束适配器之间的优化耦合。
5. **新 seeds 的 signed 候选更早停止。** leaderboard 显示 candidate 在 seeds 17/27 平均完成 5.4/5.2 epochs，而 control 为 6.4/6.0；fold 0 上控制跑满 8 epochs，candidate 分别只完成 4/3 epochs并明显退化。这是后续需要用适配器学习率、残差参数化、范数约束或 warm-up 做受控探针的线索，不是当前证据能单独证明的根因。

## 证据完整性

- v14 runner 正常结束，29/29 个 receipt 输出 SHA-256 复算一致。
- Receipt schema：`tcn-signed-multiseed-confirmation-v14/v1`。
- `sealed_test_accessed=false`；没有调用 PandaData、下载数据、访问 test/sealed、部署或执行外部写入。
- 运行前 Ruff、完整 mypy、统一测试入口和 production wheel/sdist 均通过。
- 审计发现原始 v14 `tcn-epoch-history.parquet` 缺少 `seed`/`model_seed` 列；leaderboard、checkpoint 文件名、selection 和最终指标均带 seed，因此不影响确认决策，但限制了逐 epoch 轨迹的直接归属。现有 artifact 保持不可变，源码已补齐这两个 provenance 字段并增加回归断言。

## 下一步边界

当前 signed 候选不得进入 sealed test，也不应继续追跑随机 seed。下一轮若继续优化 TCN，应把 v14 当作新基线，开展一个独立、预注册的稳定化实验：保留 TCN trunk 与双尺度 readout，但将无约束权重改为围绕 simplex 的零初始化 signed residual，并对 residual 使用独立较低学习率和显式范数约束；同时把 1/2 日与 3/5 日的负迁移作为独立门槛。simplex TCN 继续作为结构控制，LSTM 继续只作 benchmark。
