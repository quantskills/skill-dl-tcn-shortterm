# TCN 动态 skip 原始路径 + shape residual v26 真实实验结果

## 结论

v26 已在冻结的五年真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 TCN 单元。正式状态为：

`stop_dynamic_skip_raw_shape_residual_unstable_v26`

保留 v21 raw scorer 并增加 scale `0.25` 的零初始化 shape residual，明显优于 v25：mean RankIC 从 `0.094527` 提高到 `0.097328`，增量 `+0.002800`；seed 17 从 v25 的五折全部退化恢复为相对静态 `3/5` folds 不退化。候选相对静态 TCN 为 `+0.002094`，但仍比 v21 raw parent 低 `-0.001568`，没有达到晋级门槛。

因此 v26 部分确认“不能替换 raw token，shape 只能作为残差”的判断；但仅保证初始化等价并不能保证联合训练后 raw 路径仍保持 parent 性能。shape residual 确实改变了动态权重，却没有在 seeds 7/17 上稳定转化为 parent 增量。

## 实验身份

- artifact：`artifacts/tcn-dynamic-skip-raw-shape-residual-multiseed-v26`
- receipt：`bab0ce70db0e9a7bf6d31045f6a13b24ed8da67153e2fb4a7fa69d727b322599`
- schema：`tcn-dynamic-skip-raw-shape-residual-v26/v1`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- static control：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25 ablation：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- seeds/folds：`7,17,27 × 0..4`
- 参数：静态 `6260`、raw parent `6348`、v25 `6352`、v26 `6436`；raw/shape 动态参数各 `88`
- shape residual scale：`0.25`
- 优化器：全部参数共同 Adam `0.003`
- sealed test：未访问、未授权

实现测试证明：同 seed 下，在 shape logits 为零时，v26 的 raw logits、动态权重和最终输出与 v21 结构逐元素严格相同；新增模块在 raw hidden/output 创建完成后才实例化，因此不会改变 raw 分支初始化顺序。

## 预注册门槛结果

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.100 | 0.097328 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对静态 mean delta ≥ +0.003 | +0.002094 | 失败 |
| 相对 v21 parent mean delta ≥ +0.001 | -0.001568 | 失败 |
| 相对 v25 mean delta ≥ +0.003 | +0.002800 | 失败 |
| 每 seed 相对静态 mean delta > 0 | seed 17 为负 | 失败 |
| 每 seed 相对 parent mean delta > 0 | seeds 7/17 为负 | 失败 |
| 每 seed 相对 v25 mean delta > 0 | 三个 seed 均为正 | 通过 |
| 每 seed 至少 3/5 folds 相对静态不退化 | 4/5、3/5、4/5 | 通过 |
| 四期限保护门槛 | 全部满足 | 通过 |
| median samples/s ≥ 4500 | 4962.948 | 通过 |
| raw output weight L2 > 1e-12 | 最小 0.133748 | 通过 |
| shape output weight L2 > 1e-12 | 最小 0.142357 | 通过 |
| 每单元 shape weight effect ≥ 1e-6 | 最小 0.003275 | 通过 |
| block variation ≥ 1e-6 | 最小 0.000846 | 通过 |
| simplex error ≤ 1e-6 | 最大 2.384186e-7 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 3.755× / 3.638× | 通过 |

正式 blockers：

`mean_rankic_below_gate,mean_rankic_delta_below_gate,parent_mean_rankic_delta_below_gate,ablation_mean_rankic_delta_below_gate,per_seed_mean_delta_not_positive,per_seed_parent_mean_delta_not_positive`

## 逐种子结果

| seed | 静态 TCN | v21 parent | v25 | v26 | vs 静态 | vs parent | vs v25 | 静态不退化折 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.087731 | 0.096378 | 0.089242 | 0.092283 | +0.004552 | -0.004095 | +0.003041 | 4/5 |
| 17 | 0.098986 | 0.099986 | 0.094665 | 0.097909 | -0.001078 | -0.002077 | +0.003243 | 3/5 |
| 27 | 0.098982 | 0.100323 | 0.099675 | 0.101791 | +0.002809 | +0.001468 | +0.002117 | 4/5 |

v26 在三个 seed 上都超过 v25，并在 seed 27 上超过 v21 parent。相对 parent，15 个 fold 中有 `9/15` 为正：seed 7 为 `1/5`、seed 17 为 `3/5`、seed 27 为 `5/5`。总体 parent 回归主要由 seed 7 fold 1 的 `-0.012541` 和 seed 17 fold 4 的 `-0.018535` 两个大幅退化单元主导。

## 期限、机制与速度

| horizon | 相对静态 RankIC delta | 结果 |
|---:|---:|---|
| 1d | +0.005728 | 通过 |
| 2d | +0.001803 | 通过 |
| 3d | +0.002047 | 通过 |
| 5d | -0.001201 | 通过 |

shape residual 对最终动态权重的最大差值在 15 个单元中为 `0.003275–0.015228`，均明显高于使用门槛；shape output L2 均非零，说明辅助分支不是死分支。其 effect 均值为 `0.006580`，但 effect 与相对 parent RankIC delta 的 Pearson 相关仅 `-0.142`，没有“影响越大、效果越好”的证据。

block variation 均值 `0.002410`，高于 v21 parent 的约 `0.001771`，但远低于 v24/v25 的约 `0.00446/0.00468`。这与效果恢复方向一致，却仍不足以证明 variation 是目标本身。

- v26 TCN mean RankIC：`0.097328`
- 固定 LSTM mean RankIC：`0.115545`
- 配对差：`-0.018218`
- model-step speed ratio：`3.755224×`
- end-to-end speed ratio：`3.637919×`
- median samples/s：`4962.948`

速度、吞吐和四期限保护均通过；失败集中在跨 seed 的效果稳定性。

## 根因判断与下一步边界

v26 只在初始化时严格等价于 v21，之后 trunk、raw scorer、heads 和 shape residual 被共同训练。新增分支改变了后续梯度流和早停轨迹，raw 路径可以偏离父 checkpoint；因此“零初始化 residual”没有构成实际的 parent 性能下界。seed 27 全折超过 parent、seeds 7/17 被少数大幅退化 fold 拉低，也表明 shape 信息并非完全无效，问题更接近联合优化中的 co-adaptation，而不是容量或速度不足。

如果继续，v27 应从每个 seed/fold 的冻结 v21 checkpoint 初始化，冻结 trunk、raw scorer、skip logits 和 heads，只训练 shape residual 的 `88` 个参数；训练前 raw-only 预测必须与父 checkpoint 完全一致，并保存 shape-disabled RankIC counterfactual。这样才能直接回答 shape residual 是否提供 parent 之外的增量，同时消除 raw 路径漂移。仍不得访问 sealed test。
