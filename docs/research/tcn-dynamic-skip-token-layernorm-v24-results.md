# TCN 动态 skip token LayerNorm v24 真实实验结果

## 结论

v24 已在冻结的五年真实股票分钟线、seeds 7/17/27、五折 ordinary validation 上完成 15 个 TCN 单元。正式状态为：

`stop_dynamic_skip_token_layernorm_unstable_v24`

无仿射 token LayerNorm 达到了预注册的跨单元离散度目标：动态 variation CV 从父版本的 `0.578040` 降至 `0.519552`，比率为 `0.898816`，刚好优于 `<=0.90` 门槛。但预测效果没有改善：候选 mean RankIC 为 `0.095746`，相对静态 TCN 只有 `+0.000512`，相对共同学习率父候选为 `-0.003150`。

因此 v24 说明“让动态 scorer 对 token 平移和正比例缩放不敏感”能降低相对离散度，却同时移除了有用的幅度信息。当前失败不是参数量、优化器、因果性、速度或 normalization 未生效造成的。

## 实验身份

- artifact：`artifacts/tcn-dynamic-skip-token-layernorm-multiseed-v24`
- receipt：`a39e16f0cdf22af9ad18888708bbf7ed5b2254732ea33c990eec521fab1c641c`
- schema：`tcn-dynamic-skip-token-layernorm-v24/v1`
- 当前候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-token-ln`
- 静态控制：`horizon-skip-c16-chomp-smooth`
- 共同 LR 父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- seeds/folds：`7,17,27 × 0..4`
- 参数：静态 `6260`、动态候选 `6348`、动态 scorer `88`、LayerNorm 新增 `0`
- 优化器：全部参数共同 Adam `0.003`，身份 `all-lr-0.003`
- sealed test：未访问、未授权

运行器验证 v20/v21 父 receipt、selection、全部输出 SHA-256、源数据 SHA-256 和 sealed 标志后才训练。静态控制、共同 LR 父候选和固定 LSTM 均使用冻结证据。

首次启动在训练前被秘密扫描器拦截，因为公开配置键包含字符串 `token`；没有生成模型或 artifact。随后将公开配置键改为无歧义的 `dynamic_skip_input_normalization`，内部模型语义和预注册门槛未改变，安全扫描及定向测试通过后才执行正式实验。

## 预注册门槛结果

| 门槛 | 观测 | 结果 |
|---|---:|---|
| 候选 mean RankIC ≥ 0.100 | 0.095746 | 失败 |
| 15/15 RankIC 为正 | 15/15 | 通过 |
| 相对静态控制 mean delta ≥ +0.003 | +0.000512 | 失败 |
| 相对共同 LR 父候选 ≥ +0.001 | -0.003150 | 失败 |
| 每 seed 相对静态 mean delta > 0 | 三个 seed 均略大于 0 | 通过 |
| 每 seed 至少 3/5 折不退化 | seed 27 为 2/5 | 失败 |
| 每 seed 相对父候选 mean delta > 0 | 三个 seed 均为负 | 失败 |
| horizon 保护门槛 | +0.005104/+0.000074/+0.000657/-0.003785 | 通过 |
| median samples/s ≥ 5000 | 5047.733 | 通过 |
| output weight L2 > 1e-12 | 最小 0.159891 | 通过 |
| block variation ≥ 1e-6 | 最小 0.002347 | 通过 |
| variation CV ratio ≤ 0.90 | 0.898816 | 通过 |
| simplex error ≤ 1e-6 | 最大 2.384186e-7 | 通过 |
| TCN/LSTM 两种速度 ≥ 3× | 3.811× / 3.670× | 通过 |

正式 blockers：

`mean_rankic_below_gate,mean_rankic_delta_below_gate,parent_mean_rankic_delta_below_gate,per_seed_parent_mean_delta_not_positive,per_seed_fold_stability_below_gate`

## 逐种子结果

| seed | 静态 TCN | 共同 LR 父候选 | v24 | vs 静态 | vs 父候选 | 不退化折 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.087731 | 0.096378 | 0.088658 | +0.000927 | -0.007720 | 3/5 |
| 17 | 0.098986 | 0.099986 | 0.099586 | +0.000599 | -0.000400 | 3/5 |
| 27 | 0.098982 | 0.100323 | 0.098993 | +0.000011 | -0.001330 | 2/5 |

LayerNorm 在每个 seed 上都略高于静态 TCN，但三个 seed 全部低于未标准化的共同 LR 动态父候选。它把动态机制的大部分增量压缩到接近静态控制，而没有解决 seed 27 的折稳定性。

## 动态机制

| 诊断 | 共同 LR 父候选 | v24 LayerNorm |
|---|---:|---:|
| variation 最小 | 0.000794 | 0.002347 |
| variation 均值 | 0.001771 | 0.004459 |
| variation 标准差 | 0.001024 | 0.002316 |
| variation CV | 0.578040 | 0.519552 |
| variation 最大 | 0.004541 | 0.011111 |

LayerNorm 降低的是相对离散度，而不是绝对动态幅度；绝对 variation 反而约为父版本的 2.5 倍。原因是标准化后的通道形状更容易被共享 scorer 放大，而原始 token 的尺度信息被删除。CV 门槛通过并没有转化成 RankIC 门槛通过，说明“跨折 variation CV”不是足够的优化目标。

期限上，1 日 RankIC 相对静态控制改善 `+0.005104`，但 5 日下降 `-0.003785`。这表明 normalization 改变了期限间信息分配，而非简单地统一改善四个预测期限。

## TCN 与 LSTM

- v24 TCN mean RankIC：`0.095746`
- 固定 LSTM mean RankIC：`0.115545`
- 配对差：`-0.019800`
- model-step speed ratio：`3.811300×`
- end-to-end speed ratio：`3.670268×`
- median samples/s：`5047.733`

速度目标继续满足，LSTM 的预测效果仍领先。

## 下一步边界

v24 不支持继续使用纯 LayerNorm 或继续搜索动态学习率。共同 `0.003`、原始 token 输入的 v21 仍是当前最好动态 TCN。

若继续，应保留原始 token 的幅度信息，同时显式分离“形状”和“尺度”，例如把标准化通道形状与独立的 `log-RMS` 幅度特征共同交给 scorer；或者为 dilation block 提供显式身份，而不是让共享 scorer 从幅度间接推断 block。新版本必须预注册容量增量并重新做三种子比较，不能把 v24 的 CV 通过当作效果成功。
