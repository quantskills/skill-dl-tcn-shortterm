# TCN 动态跳连独立学习率 v22：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。目标不是替换 TCN，而是在已经满足 3–5× LSTM 速度要求、且梯度冲突问题已经隔离后，验证当前预测效果瓶颈是否来自 **动态 dilation-block skip 分支学习不足**。

## 已知事实

- 真实数据是五年股票分钟线派生出的 point-in-time 窗口、标签和 5 折 expanding validation；禁止读取 sealed test。
- TCN 使用因果卷积、chomp 移除未来 padding、weight normalization、memmap 窗口；8 个 dilation block 为 `1,2,4,8,16,32,64,128`，感受野覆盖 480 根输入分钟 bar。
- v20 的 `DynamicHorizonSkipTCN` 在 seed 7 相对静态 `HorizonSkipTCN` 的 mean RankIC 增量为 `+0.008647...`，但 v21 在新种子 17/27 上仅为 `+0.001170...`，未达到 `+0.003` 稳定性门槛。
- v21 的动态分支确实非零且 sample-conditioned，但最小 block-weight variation 只有 `0.000793551...`。这支持“动态分支存在，但在共同 `0.003` 学习率下学习幅度不足”的可检验假设，并不证明假设为真。
- v20/v21 的 TCN 相对固定 LSTM 的 model-step/end-to-end 速度均超过 3×；本轮不得牺牲该门槛。

## 唯一允许的干预

保持模型、数据、划分、损失、batch、epoch、patience、初始化种子和 TCN 主干学习率不变，只做一个变化：

- TCN 主干 6,260 个参数：Adam，学习率 `0.003`。
- `DynamicHorizonSkipTCN.dynamic_skip_parameters()` 返回的 88 个动态参数：Adam，学习率 `0.01`。

不得改变动态 skip hidden size、scale、网络容量、dilation、dropout、label、loss、early stopping、折数或种子。不得训练或引入非 TCN 主模型。LSTM 只作为冻结 benchmark。

## TDD 与公开契约

先写失败测试，再写实现。建立以下可审计契约：

1. `TCNTuningTrial.dynamic_skip_learning_rate` 可由真实配置显式解析。
2. `build_tcn_optimizer` 产生完整且互斥的两组参数；精确计数为 base `6260`、dynamic skip `88`、total `6348`，组身份为 `base-lr-0.003+dynamic-skip-lr-0.01`。
3. 独立学习率只允许用于 `dynamic_horizon_skip`，必须有限、正数且不超过主干学习率 10×；它不得和 adapter/residual/dynamic-attention 特殊组同时启用。
4. leaderboard 写入独立学习率、参数组身份及动态组计数。
5. 建立 v22 多种子决策器，对当前候选、历史静态控制、历史共同学习率父候选、动态机制和固定 LSTM 证据进行精确 coverage 与 hash 审计。

## 真实实验协议

- 当前候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-dslr1e2`。
- 历史静态控制：`horizon-skip-c16-chomp-smooth`。
- 历史共同学习率父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`。
- 种子：`7,17,27`；每个种子 5 折；总计只训练 15 个 v22 TCN 单元。
- 从 v20 读取 seed 7 的静态控制、共同学习率候选和固定 LSTM；从 v21 读取 seeds 17/27 的对应证据。所有父 receipt、输出 hash、source hash 和 sealed 标志必须验证。
- `max_epochs=8`、`patience=2`、`min_delta=0.002`、`batch_size=128`、`float32`、CPU 8 threads。
- 所有产物写入临时目录，完成后原子移动；拒绝覆盖已有 artifact；产物包含 resolved config、epoch history、leaderboard、动态权重诊断、历史控制、seed/horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。

## 预注册验收门槛

- 当前候选 15 单元 mean RankIC `>= 0.10`，且 `15/15` 为正。
- 相对历史静态控制的 paired mean RankIC delta `>= 0.003`。
- 每个 seed 相对静态控制的 mean delta `> 0`，且至少 `3/5` 折不退化。
- 相对历史共同学习率父候选的 paired mean RankIC delta `>= 0.001`，证明本轮 optimizer 干预有实质价值。
- 相对静态控制的 horizon delta：1d `>=0`、2d `>=-0.003`、3d `>=-0.005`、5d `>=-0.005`。
- median samples/s `>=5000`。
- 动态 output weight L2 `>1e-12`，所有单元 block-weight variation `>=1e-6`，simplex error `<=1e-6`。
- 当前 block-weight variation 的 paired median 相对父候选至少 `1.5×`。
- 相对固定 LSTM 的 model-step 和 end-to-end speed ratio 均 `>=3.0×`。

若效果门槛失败，状态为 `stop_dynamic_skip_lr_unstable_v22`；若效果通过但速度失败，状态为 `stop_dynamic_skip_lr_speed_v22`；全部通过才是 `dynamic_skip_lr_multiseed_confirmed_v22`。无论结果好坏，都如实记录，不得继续调参后覆盖本轮结论，不得访问 sealed test。

## 最终验证与汇报

运行定向测试后执行完整 Ruff、mypy、pytest 和 production build/preflight。报告必须清楚区分：

- TCN 速度门槛是否仍满足；
- 预测效果相对静态 TCN 是否稳定；
- 独立学习率相对共同学习率是否产生正向、可复现的增益；
- 动态权重变化是否按预期增大；
- 当前结论是“确认”“否定”还是“证据不足”，以及下一步只能基于新假设另立版本。
