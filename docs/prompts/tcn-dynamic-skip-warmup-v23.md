# TCN 动态 skip 受控 warm-up v23：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。必须继续以 TCN 为主模型，不得把失败解释为应替换 TCN。当前任务是验证动态 dilation-block skip 分支是否需要一条受控的优化轨迹，而不是继续使用过弱或过强的固定学习率。

## 已知证据

- 真实输入是冻结的 2021–2025 股票分钟线派生 point-in-time 窗口、标签和五折 expanding ordinary validation；禁止访问 sealed test。
- v21 共同学习率 `0.003` 的动态 skip 候选在 seeds 7/17/27 的描述性聚合 mean RankIC 约为 `0.098896`，相对静态 TCN 约 `+0.003662`；但新种子稳定性不足。
- v22 将 88 个动态参数的固定学习率提高到 `0.01` 后，动态 variation 配对中位数放大到父版本的 `4.595873×`，但 mean RankIC 降至 `0.095111`：相对静态 TCN `-0.000122`、相对共同学习率父候选 `-0.003785`，1 日 RankIC 退化 `-0.005789`。
- v22 的 model-step/end-to-end 速度仍为固定 LSTM 的 `4.044×/3.894×`。速度、因果卷积、chomp、weight normalization、memmap、参数是否更新和 simplex 数值都不是当前失败原因。

因此新假设是：动态分支需要在输出层脱离零初始化、主干开始形成稳定表征后再温和加速；目标不是最大化动态变化，而是把变化控制在 v21 与 v22 之间。

## 唯一允许的干预

保持数据、模型结构、参数量、初始化种子、损失、batch、dilation、early stopping、训练预算和 TCN 主干学习率全部不变。只将动态 skip 参数的优化轨迹改成：

- 主干 6,260 参数：Adam，所有 epoch 恒定 `0.003`。
- 动态 skip 88 参数：
  - epoch 1：`0.003`
  - epoch 2：`0.004`
  - epoch 3 及以后：封顶 `0.005`
- `max_epochs=8`、`patience=2`、`min_delta=0.002`。

这是一条预注册的两轮线性 warm-up；不得在看见任意 fold 结果后修改 target、warm-up 长度或门槛。不得同时加入梯度裁剪、权重衰减、新损失、新层、dropout 或其他干预。

## TDD 与公开契约

按红灯→绿灯垂直切片实现：

1. `TCNTuningTrial.dynamic_skip_warmup_epochs` 必须从真实 JSON 显式解析，默认 `0` 以保持历史协议兼容。
2. `dynamic_skip_learning_rate_for_epoch(trial, epoch)` 对 v23 精确返回 `0.003, 0.004, 0.005...`；epoch 必须从 1 开始。
3. `apply_tcn_epoch_learning_rates(bundle, trial, epoch)` 只更新名为 `dynamic_skip` 的参数组，base 组始终为 `0.003`；参数覆盖仍精确为 `6260+88=6348`。
4. warm-up 只能用于 `dynamic_horizon_skip`，必须配置动态 target LR，target 必须大于 base 且不超过 base 的 10 倍，warm-up 必须是非负整数且小于 `max_epochs`。
5. epoch history 写入实际应用的动态学习率；leaderboard 写入 warm-up 长度、target 和调度身份。
6. v23 决策器必须同时读取历史静态控制、v21 共同学习率父候选、v22 高学习率诊断和固定 LSTM，对 seed/fold coverage、参数组、receipt、输出 SHA-256、源数据 SHA-256 与 sealed 标志 fail-closed。

## 真实实验协议

- 候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-warm2-lr5e3`
- 静态控制：`horizon-skip-c16-chomp-smooth`
- 共同学习率父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- 高学习率边界：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-dslr1e2`
- seeds：`7,17,27`；folds：`0..4`；总共只训练 15 个 v23 TCN 单元。
- 从 v20 读取 seed 7 的静态/共同 LR/LSTM 证据，从 v21 读取 seeds 17/27 的同类证据，从 v22 读取高 LR 动态诊断。不得重新训练历史控制或 LSTM。
- CPU 8 threads、float32、batch 128、smooth L1、chomp、channels 16、kernel 3、dilations `1,2,4,8,16,32,64,128`、dynamic hidden 4、scale 1.0。
- 临时目录完成后原子移动，拒绝覆盖 artifact；输出 resolved config、epoch history、leaderboard、当前/父/高 LR 动态诊断、历史控制、seed/horizon summary、固定 LSTM comparison、15 个 checkpoint、selection 和 receipt。

## 预注册门槛

- 当前候选 15 单元 mean RankIC `>=0.099`，且 `15/15` 为正。
- 相对静态控制 paired mean RankIC delta `>=0.003`。
- 每个 seed 相对静态控制 mean delta `>0`，且至少 `3/5` folds 不退化。
- 相对共同 LR 父候选 paired mean RankIC delta `>=0.0005`，且每个 seed 的父版本 delta 都 `>0`。
- 相对静态控制的 horizon delta：1d `>=0`、2d `>=-0.003`、3d `>=-0.005`、5d `>=-0.005`。
- median samples/s `>=5000`。
- 动态 output weight L2 `>1e-12`，block variation 最小值 `>=1e-6`，simplex error 最大值 `<=1e-6`。
- 当前/共同 LR 父版本的 paired median variation ratio 必须位于 `[1.2,3.0]`。
- 当前/v22 高 LR 的 paired median variation ratio 必须 `<=0.75`，证明轨迹从过冲边界明显回落。
- 相对固定 LSTM model-step 和 end-to-end speed ratio 均 `>=3.0×`。

效果失败状态为 `stop_dynamic_skip_warmup_unstable_v23`；效果通过但速度失败为 `stop_dynamic_skip_warmup_speed_v23`；全部通过才是 `dynamic_skip_warmup_multiseed_confirmed_v23`。任何失败都不得用事后降门槛、删除 fold 或覆盖 artifact 的方式改写。

## 最终验收与报告

运行定向测试后执行全仓 Ruff、mypy、完整 pytest、`tasks/preflight.py`、`tasks/test.py` 和 production wheel/sdist build。报告必须分别回答：

- 调度是否逐 epoch 正确执行并可审计；
- 动态变化是否处于 v21 与 v22 之间；
- 相对静态 TCN 和共同 LR TCN 是否跨种子改善；
- TCN 的 3× 速度优势是否保持；
- 当前证据确认、否定或仍不足以支持“受控动态轨迹”假设。

不得访问 sealed test、部署、连接券商、执行交易或承诺收益。
