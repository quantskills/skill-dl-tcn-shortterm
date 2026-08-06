# TCN 动态 skip 受控 warm-up v23 真实实验结果

## 结论

v23 已在冻结的五年真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 TCN 单元。正式状态为：

`stop_dynamic_skip_warmup_unstable_v23`

两轮线性 warm-up 的工程行为完全正确，但没有改善预测效果。动态 skip 实际学习率逐 epoch 为 `0.003 → 0.004 → 0.005`，主干始终为 `0.003`；相对 v22 高学习率过冲边界，动态 variation 已回落到 `0.230819×`。然而候选 mean RankIC 只有 `0.095665`，相对静态 TCN 为 `+0.000432`，相对共同学习率父候选为 `-0.003230`。

因此 v23 否定了“只要先 warm-up、再把动态 LR 温和提高到 0.005，就能兼顾动态幅度与预测稳定性”的具体假设。当前问题仍不是速度、样本量、参数未更新或调度未执行。

## 实验身份

- artifact：`artifacts/tcn-dynamic-skip-warmup-multiseed-v23`
- receipt：`eda3849bf09e09f5e8dd324d5220e25dfec4fcf41c751ed07353ac89b1c6eada`
- schema：`tcn-dynamic-skip-warmup-v23/v1`
- 当前候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-warm2-lr5e3`
- 静态控制：`horizon-skip-c16-chomp-smooth`
- 共同学习率父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v22 高 LR 边界：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-dslr1e2`
- 参数：主干 `6260`，动态 skip `88`，合计 `6348`
- sealed test：未访问、未授权

运行器逐文件验证 v20、v21、v22 三个父 artifact 的 receipt、selection、输出 SHA-256、源数据 SHA-256 和 sealed 标志。历史静态 TCN、共同 LR TCN、高 LR 动态诊断及固定 LSTM 均未重新训练。

## 调度审计

epoch history 覆盖 15 个 seed/fold 单元，所有记录与预注册轨迹逐行一致：

| epoch | base LR | dynamic skip LR |
|---:|---:|---:|
| 1 | 0.003 | 0.003 |
| 2 | 0.003 | 0.004 |
| 3–8 | 0.003 | 0.005 |

优化器身份为：

`base-lr-0.003+dynamic-skip-linear-warmup-2-lr-0.003-to-0.005`

参数组完整且互斥，动态组精确为 88 个参数。调度执行错误不是实验失败原因。

## 预注册门槛结果

| 门槛 | 观测 | 结果 |
|---|---:|---|
| 候选 mean RankIC ≥ 0.099 | 0.095665 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对静态控制 mean delta ≥ +0.003 | +0.000432 | 失败 |
| 相对共同 LR 父候选 ≥ +0.0005 | -0.003230 | 失败 |
| 每 seed 相对静态控制 mean delta > 0 | seeds 7/17 为负 | 失败 |
| 每 seed 至少 3/5 折不退化 | 3/5、3/5、3/5 | 通过 |
| 每 seed 相对父候选 mean delta > 0 | seeds 7/17 为负 | 失败 |
| 1d/2d/3d/5d 保护门槛 | +0.000414/+0.002973/+0.001483/-0.003141 | 通过 |
| median samples/s ≥ 5000 | 5062.532 | 通过 |
| 动态 output weight L2 > 1e-12 | 最小 0.178099 | 通过 |
| block variation ≥ 1e-6 | 最小 0.001039 | 通过 |
| parent variation ratio ∈ [1.2,3.0] | 1.153140 | 失败 |
| current/high-LR variation ratio ≤ 0.75 | 0.230819 | 通过 |
| simplex error ≤ 1e-6 | 2.384186e-7 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 3.770× / 3.620× | 通过 |

正式 blockers：

`mean_rankic_below_gate,mean_rankic_delta_below_gate,parent_mean_rankic_delta_below_gate,per_seed_mean_delta_not_positive,parent_variation_ratio_below_gate,per_seed_parent_mean_delta_not_positive`

## 逐种子结果

| seed | 静态 TCN | 共同 LR 父候选 | v23 | vs 静态 | vs 父候选 | 不退化折 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.087731 | 0.096378 | 0.087718 | -0.000014 | -0.008661 | 3/5 |
| 17 | 0.098986 | 0.099986 | 0.098569 | -0.000417 | -0.001417 | 3/5 |
| 27 | 0.098982 | 0.100323 | 0.100710 | +0.001727 | +0.000386 | 3/5 |

seed 27 有小幅正向结果，但不足以抵消 seed 7 和 seed 17 的退化。最明显的单元是 seed 7 fold 1：共同 LR 父候选为 `0.096407`，v23 只有 `0.074891`，下降 `-0.021516`。

## 动态机制

| 诊断 | 共同 LR 父候选 | v23 warm-up | v22 高 LR |
|---|---:|---:|---:|
| variation 最小 | 0.000794 | 0.001039 | 0.003827 |
| variation 中位数 | 0.001361 | 0.001947 | 0.007096 |
| variation 最大 | 0.004541 | 0.005953 | 0.015309 |

简单中位数显示 v23 位于 v21/v22 之间；但按同 seed/fold 配对后的 parent ratio 中位数只有 `1.153140`，略低于 `1.2` 下界。15 个单元中，部分最佳 checkpoint 停在 epoch 1 或 epoch 2，导致其动态状态和父版本非常接近；例如 seed 7 fold 4 的父/v23 variation 完全相同。

更重要的是，v23 variation 与相对父候选 RankIC delta 的描述性相关约为 `-0.491`。这不是因果证明，但结合 v22 的负结果，证据不再支持“提高动态分支更新幅度”作为主要优化方向。

## TCN 与 LSTM

- v23 TCN mean RankIC：`0.095665`
- 固定 LSTM mean RankIC：`0.115545`
- 配对差：`-0.019880`
- model-step speed ratio：`3.769957×`
- end-to-end speed ratio：`3.620211×`
- TCN median samples/s：`5062.532`

TCN 速度目标继续满足，LSTM 的预测效果仍明显领先。

## 下一步边界

v21、v22、v23 共同说明：共同 LR `0.003` 仍是目前最好的动态 skip 优化轨迹；固定 `0.01` 和 warm-up 到 `0.005` 都降低了 RankIC。后续不应继续搜索更高动态 LR，也不应事后把 `1.2` variation 下界降到 `1.153`。

若继续，应恢复共同 LR `0.003`，把新假设转向动态表征的方差约束或跨期限结构，例如对动态 logits 使用显式 identity/shrinkage 正则、限制最短期限的动态偏移，或用 checkpoint/EMA 稳定化；每次仍只能预注册一个干预并用完整三种子证据验证。主模型保持 TCN。
