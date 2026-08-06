# TCN 动态 skip 形状—幅度解耦 v25：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，不得把 LSTM、GRU 或其他模型替换为主模型；LSTM 只作为冻结 benchmark。必须使用冻结的真实股票分钟线 ordinary-validation 证据，禁止访问 sealed test、连接券商、执行交易、部署或承诺收益。

## 当前事实

- 当前主线是 Bai 风格因果扩张 TCN：5 分钟 bar、480 bar 回看、dilations `1,2,4,8,16,32,64,128`、chomp 因果卷积、weight normalization、memmap、五折 expanding walk-forward ordinary validation。
- v21 原始 token、共同 Adam `0.003` 的 `DynamicHorizonSkipTCN` 是当前最佳动态 TCN：三种子合并 mean RankIC 约 `0.098896`，相对静态 TCN 约 `+0.003662`，但跨 fold 稳定性不足。
- v22 动态参数高学习率和 v23 warm-up 均失败，不能继续搜索动态学习率。
- v24 对 scorer token 使用无仿射 LayerNorm：variation CV 比率降至 `0.898816`，但 mean RankIC 降至 `0.095746`，相对 v21 为 `-0.003150`。LayerNorm 删除了有效幅度信息；速度仍达到 LSTM 的 `3.81×/3.67×`。
- 因此本轮只检验一个假设：保留标准化通道形状，并把原始 token 的幅度压缩为独立标量交给 scorer，可兼顾尺度稳健性和预测信息。

## 唯一允许的模型干预

对每个 `[batch, block, channel]` 原始最后有效 token `x` 构造 scorer 输入：

1. `shape = LayerNorm(channels, elementwise_affine=False)(x)`；
2. `amplitude = log1p(sqrt(mean(x², channel)))`，保留最后一维；
3. `scorer_input = concat(shape, amplitude)`，形状为 `[batch, block, channels + 1]`；
4. 仅把 `dynamic_skip_hidden` 的输入宽度从 `channels` 改为 `channels + 1`，其余 `tanh → dynamic_skip_output → bounded logits → block softmax` 不变；
5. 最终 horizon readout 必须继续混合原始 token `x`，不得混合标准化 token。

不得同时加入 block embedding、额外 attention、特殊学习率、warm-up、scheduler、weight decay、新损失或数据变更。本轮相对 v21 仅新增 `dynamic_skip_hidden × 1 = 4` 个幅度投影权重：静态控制 `6260` 参数、v21/v24 `6348`、v25 `6352`，其中动态 scorer `92`。

## TDD 与公开行为边界

按垂直红灯—绿灯切片实现，测试公开接口而非私有实现：

1. `DynamicHorizonSkipTCN.dynamic_skip_scorer_inputs(tokens)` 返回 scorer 真正消费的输入。`shape_log_rms` 模式返回 `channels+1` 维；前 `channels` 维对正比例缩放与通道平移保持形状不变，最后一维随原始幅度单调变化。
2. 零初始化动态输出层时，v25 与相同 seed 的静态 `HorizonSkipTCN` 输出严格相同；最终 readout 仍使用原始 token。
3. 默认 `none` 与历史行为、参数量、receipt 完全兼容；`layer_norm` 仍为 v24 的零参数模式。
4. 公开 JSON 键沿用 `dynamic_skip_input_normalization`，新增允许值 `shape_log_rms`，解析至 `TCNTuningTrial.dynamic_skip_token_normalization`；未知值或用于非 `dynamic_horizon_skip` 必须 fail-closed。
5. receipt/leaderboard 记录：输入模式、`log1p_rms` 幅度特征、scorer 输入宽度、normalization 参数数、动态参数数、幅度投影权重 L2、共同优化器身份。
6. 幅度计算只能读取当前因果 trunk 的最后有效 token，不得读取未来 bar、标签、市场横截面或 sealed split。

## 冻结训练协议

- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- static control：`horizon-skip-c16-chomp-smooth`
- raw-token parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- LayerNorm ablation：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-token-ln`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 v25 TCN 单元。
- channels 16、kernel 3、8 个 dilation、dropout 0、batch 128、smooth L1、float32、8 threads、max epochs 8、patience 2、min delta 0.002。
- 所有参数使用同一 Adam `0.003`；不得创建特殊参数组。
- 从 v20/v21 冻结产物读取静态控制、raw parent 和 LSTM；从 v24 冻结产物读取 LayerNorm ablation。必须验证父 receipt ID、selection status、输出 SHA-256、源数据 SHA-256、精确 seed/fold coverage 和 `sealed_test_accessed=false`，不得重训历史对照。
- 输出必须先写临时目录，完成后原子改名；拒绝覆盖已有 artifact。

## 预注册效果与速度门槛

- 15 单元 candidate mean RankIC `>=0.100`，且 `15/15` 为正。
- 相对静态控制 paired mean delta `>=+0.003`；每个 seed mean delta `>0`，且每个 seed 至少 `3/5` folds 不退化。
- 相对 raw-token v21 parent paired mean delta `>=+0.001`，且每个 seed mean delta 均 `>0`。
- 相对 v24 LayerNorm ablation paired mean delta `>=+0.003`，且每个 seed mean delta均 `>0`。
- horizon 相对静态控制保护门槛：1d `>=0`、2d `>=-0.003`、3d `>=-0.005`、5d `>=-0.005`。
- median samples/s `>=5000`；TCN/LSTM model-step 与 end-to-end speed ratio 均 `>=3.0×`。
- output weight L2 `>1e-12`；幅度投影权重 L2 `>1e-12`；block-weight variation 最小值 `>=1e-6`；simplex error 最大值 `<=1e-6`。
- 参数量必须精确为控制 `6260`、candidate `6352`、动态参数 `92`、相对 raw parent 新增 `4`；normalization 新增参数为 `0`；优化器身份必须为 `all-lr-0.003`。

效果失败状态：`stop_dynamic_skip_shape_amplitude_unstable_v25`；效果通过但速度失败：`stop_dynamic_skip_shape_amplitude_speed_v25`；全部通过：`dynamic_skip_shape_amplitude_multiseed_confirmed_v25`。不得事后降低阈值、删 seed/fold、修改候选或覆盖 artifact。

## 实验、审计与报告

实现专用 v25 决策器与 runner，生成 resolved config、epoch history、candidate leaderboard、历史控制、candidate/parent 动态诊断、seed summary、horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。receipt 必须包含 code identity、环境、父证据身份、所有输出 SHA-256 与 `sealed_test_accessed=false`。

真实实验后完成 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py` 和 production wheel/sdist build。报告必须明确回答：

- shape 分支是否保持尺度/平移不变，幅度分支是否恢复尺度敏感性；
- v25 是否同时超过静态控制、v21 raw parent 和 v24 LayerNorm ablation；
- 三个 seed 与四个 horizon 的稳定性是否改善；
- 4 个新增参数是否被 scorer 使用；
- 速度是否继续达到 `3×`；
- 假设被确认、否定还是证据不足，以及下一步是否有权继续。
