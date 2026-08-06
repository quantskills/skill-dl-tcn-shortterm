# TCN 动态 skip 形状—幅度解耦 v25 真实实验结果

## 结论

v25 已在冻结的五年真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 TCN 单元。正式状态为：

`stop_dynamic_skip_shape_amplitude_unstable_v25`

把无仿射 LayerNorm 通道形状与独立 `log1p(RMS)` 幅度标量共同交给动态 scorer，没有恢复 v21 原始 token 的效果。candidate mean RankIC 为 `0.094527`，相对静态 TCN 为 `-0.000706`，相对 v21 raw-token parent 为 `-0.004368`，相对 v24 LayerNorm ablation 仍为 `-0.001218`。因此“一个 RMS 标量足以补回 LayerNorm 删除的有效信息”被当前三种子证据否定。

TCN/LSTM model-step 与端到端速度仍为 `3.581×/3.464×`，通过相对速度门槛；绝对 median throughput 为 `4817.171 samples/s`，低于预注册的 `5000`。本轮主要问题仍是预测效果，另有轻微吞吐退化。

## 实验身份

- artifact：`artifacts/tcn-dynamic-skip-shape-amplitude-multiseed-v25`
- receipt：`ca5fa451befddb8cd41de9a4e51d8429280042e5a5847762943c1c9bfd09c43a`
- schema：`tcn-dynamic-skip-shape-amplitude-v25/v1`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- static control：`horizon-skip-c16-chomp-smooth`
- raw-token parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- LayerNorm ablation：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-token-ln`
- seeds/folds：`7,17,27 × 0..4`
- 参数：静态 `6260`、v21/v24 `6348`、v25 `6352`；动态 scorer `92`，其中新增幅度投影权重 `4`
- 优化器：所有参数共同 Adam `0.003`，身份 `all-lr-0.003`
- sealed test：未访问、未授权

runner 在训练前验证 v20/v21/v24 父 receipt、selection status、27 个父输出哈希、源数据 SHA-256 与 sealed 标志，并复用冻结的静态、raw parent、LayerNorm ablation 和 LSTM 证据。

第一次正式启动完成训练后在决策阶段 fail-closed，因为 sweep leaderboard 未转发模型已有的 `dynamic_skip_normalization_parameter_count` 元数据；该次没有创建正式或临时 artifact。随后增加直接运行 tiny sweep 的回归测试，确认红灯后补齐公开审计列，保持配置、数据、模型和门槛不变，再执行本次正式实验。

## 预注册门槛结果

| 门槛 | 观察值 | 结果 |
|---|---:|---|
| candidate mean RankIC ≥ 0.100 | 0.094527 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对静态 mean delta ≥ +0.003 | -0.000706 | 失败 |
| 相对 raw parent mean delta ≥ +0.001 | -0.004368 | 失败 |
| 相对 v24 ablation mean delta ≥ +0.003 | -0.001218 | 失败 |
| 每 seed 相对静态 mean delta > 0 | seed 17 为负 | 失败 |
| 每 seed 相对 parent mean delta > 0 | 三个 seed 均为负 | 失败 |
| 每 seed 相对 ablation mean delta > 0 | seed 17 为负 | 失败 |
| 每 seed 至少 3/5 folds 不退化 | seed 17=0/5，seed 27=2/5 | 失败 |
| 四期限保护门槛 | 全部满足 | 通过 |
| median samples/s ≥ 5000 | 4817.171 | 失败 |
| output weight L2 > 1e-12 | 最小 0.158620 | 通过 |
| amplitude projection weight L2 > 1e-12 | 最小 0.136931 | 通过 |
| block variation ≥ 1e-6 | 最小 0.002534 | 通过 |
| simplex error ≤ 1e-6 | 最大 2.384186e-7 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 3.581× / 3.464× | 通过 |

正式 blockers：

`mean_rankic_below_gate,mean_rankic_delta_below_gate,parent_mean_rankic_delta_below_gate,ablation_mean_rankic_delta_below_gate,per_seed_mean_delta_not_positive,per_seed_parent_mean_delta_not_positive,per_seed_ablation_mean_delta_not_positive,per_seed_fold_stability_below_gate,throughput_below_gate`

## 逐种子结果

| seed | 静态 TCN | raw parent | v24 | v25 | vs 静态 | vs parent | vs v24 | 不退化折 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.087731 | 0.096378 | 0.088658 | 0.089242 | +0.001511 | -0.007137 | +0.000584 | 4/5 |
| 17 | 0.098986 | 0.099986 | 0.099586 | 0.094665 | -0.004321 | -0.005320 | -0.004920 | 0/5 |
| 27 | 0.098982 | 0.100323 | 0.098993 | 0.099675 | +0.000693 | -0.000649 | +0.000682 | 2/5 |

幅度标量在 seeds 7/27 上略高于 v24，却在 seed 17 的五个 fold 全部退化，导致总体低于静态控制、v24 和 raw parent。它没有解决跨 seed 初始化/优化敏感性。

## 逐期限与动态机制

| horizon | 相对静态 RankIC delta | 保护门槛 | 结果 |
|---:|---:|---:|---|
| 1d | +0.000817 | ≥ 0 | 通过 |
| 2d | +0.001766 | ≥ -0.003 | 通过 |
| 3d | -0.000855 | ≥ -0.005 | 通过 |
| 5d | -0.004551 | ≥ -0.005 | 通过 |

v25 block-weight variation 均值为 `0.004680`，约为 v21 parent `0.001771` 的 `2.64×`；variation CV 从 parent 的 `0.578040` 降到 `0.397115`。与 v24 一样，相对离散度下降没有转化为 RankIC 提升，进一步说明 variation CV 不应继续作为优化目标。

幅度投影权重和动态输出权重均为非零，表明该路径存在于实际 scorer 中；但本协议没有保存初始化权重差分或幅度置零 counterfactual，因此该 L2 只能证明路径非零，不能单独证明它带来了正向因果贡献。

## TCN 与 LSTM

- v25 TCN mean RankIC：`0.094527`
- 固定 LSTM mean RankIC：`0.115545`
- 配对差：`-0.021018`
- model-step speed ratio：`3.580669×`
- end-to-end speed ratio：`3.464187×`
- median samples/s：`4817.171`

速度优势继续成立，但效果差距扩大。

## 根因判断与下一步边界

v25 的信息分解并不充分。LayerNorm 删除了每个 token 的通道均值和标准差两个自由度，而单一 RMS 把均值与方差混合成一个标量，不能重建原始 token；`log1p` 还进一步压缩了幅度差异。另外，把隐藏层输入从 16 改为 17 会改变全部隐藏权重的 fan-in 初始化，使本轮同时暴露于信息不足和初始化扰动，seed 17 的一致退化与该风险相符。

下一轮不应继续用归一化特征替换 raw token。更安全的 v26 边界是：完整保留 v21 的 16 维 raw scorer 路径及其初始化顺序，另加一个零初始化、容量预注册的 shape residual 分支；只有辅助残差为零时必须严格退化为 v21 结构。这样才能检验“形状信息是否提供 raw path 之外的增量”，而不是再次删除已知有效信息。仍需三种子五折验证，主模型保持 TCN。
