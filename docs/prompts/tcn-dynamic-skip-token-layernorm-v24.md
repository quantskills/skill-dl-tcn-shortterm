# TCN 动态 skip token LayerNorm v24：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。必须继续优化 TCN，不得更换为 LSTM 或其他主模型。当前目标是降低不同 dilation block 激活尺度漂移对动态 skip scorer 的干扰，并验证这种尺度不变性是否能提高跨 seed/fold 的预测稳定性。

## 已知事实

- 数据是冻结的 2021–2025 股票分钟线派生 point-in-time 窗口、标签和五折 expanding ordinary validation；禁止访问 sealed test。
- 模型使用因果卷积、chomp、weight normalization、memmap、8 个 dilation block `1,2,4,8,16,32,64,128`，感受野覆盖 480 根输入 bar。
- v21 共同学习率 `0.003` 的动态 skip 是当前最好版本：三种子描述性 mean RankIC 约 `0.098896`，相对静态 TCN 约 `+0.003662`，但 seeds 17/27 的折稳定性不足。
- v22 固定动态 LR `0.01` 和 v23 warm-up 到 `0.005` 均降低预测效果；所以本轮恢复所有参数共同 Adam `0.003`，不再改变学习率。
- v21 的 15 个 parent 单元中，block-weight variation 的 CV 约 `0.5780`、max/min 约 `5.72×`。variation 与 RankIC delta 的关系混合，不能简单把动态偏移整体压小。

新假设是：不同 dilation block 的最后有效 token 具有不同、随 seed/fold 漂移的通道尺度，动态 scorer 直接读取原始 token 会把激活幅度误当成尺度选择信号。对每个样本、每个 block 的通道向量做无仿射 LayerNorm，可让 scorer 关注通道形状而不是幅度，同时不增加参数、不破坏因果性。

## 唯一允许的干预

在 `DynamicHorizonSkipTCN` 的动态 scorer 输入处，对 `[batch, block, channel]` token 的最后一维应用：

`LayerNorm(channels, elementwise_affine=False)`

随后再进入原有 `dynamic_skip_hidden → tanh → dynamic_skip_output`。静态 `skip_logits`、TCN trunk、heads 和输出计算全部不变。

必须保持：

- 总参数 `6348`，其中动态 scorer `88`；LayerNorm 新增参数为 `0`。
- 所有参数共同 Adam `0.003`，不得创建特殊 LR 组、warm-up、weight decay 或 scheduler。
- channels 16、kernel 3、8 个 dilation、dropout 0、batch 128、smooth L1、max epochs 8、patience 2、min delta 0.002。
- 数据、seeds、folds、标签、损失、初始化和验证协议不变。

## TDD 与公开契约

按红灯→绿灯垂直切片实现：

1. JSON 使用不与凭据扫描器冲突的公开键 `dynamic_skip_input_normalization`；它解析到 `TCNTuningTrial.dynamic_skip_token_normalization`，允许 `none` 和 `layer_norm`，默认 `none` 保持历史证据兼容。
2. `DynamicHorizonSkipTCN.normalize_dynamic_skip_tokens()` 是公开行为边界。`layer_norm` 模式应对每个 sample/block token 的通道平移与正比例缩放保持动态权重不变；`none` 保持旧行为。
3. LayerNorm 必须无仿射参数，候选仍精确为 6348 参数；零初始化动态输出层时，候选仍与静态 `HorizonSkipTCN` 输出完全相同。
4. normalization 只能用于 `dynamic_horizon_skip`；未知值或用于其他模型必须 fail-closed。
5. leaderboard/receipt 写入 `dynamic_skip_token_normalization=layer_norm`、`optimizer_group_identity=all-lr-0.003` 和零新增参数证据。
6. v24 决策器必须精确验证当前候选、历史静态控制、共同 LR 父候选、固定 LSTM、seed/fold coverage、参数量、normalization identity、receipt、输出 SHA-256、源 SHA-256 与 sealed 标志。

## 真实实验协议

- 候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-token-ln`
- 静态控制：`horizon-skip-c16-chomp-smooth`
- 共同 LR 父候选：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 v24 TCN 单元。
- 从 v20 读取 seed 7 的静态/父候选/LSTM，从 v21 读取 seeds 17/27 的同类证据；不得重新训练历史控制或 LSTM。
- 临时目录完成后原子移动，拒绝覆盖；产物必须包含 resolved config、epoch history、leaderboard、当前/父动态诊断、历史控制、seed/horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。

## 预注册门槛

- 候选 15 单元 mean RankIC `>=0.100`，且 `15/15` 为正。
- 相对静态控制 paired mean RankIC delta `>=0.003`。
- 每个 seed 相对静态控制 mean delta `>0`，且至少 `3/5` folds 不退化。
- 相对共同 LR 父候选 paired mean RankIC delta `>=0.001`，且每个 seed 的父版本 delta 都 `>0`。
- horizon delta 相对静态控制：1d `>=0`、2d `>=-0.003`、3d `>=-0.005`、5d `>=-0.005`。
- median samples/s `>=5000`。
- output weight L2 `>1e-12`、block variation 最小 `>=1e-6`、simplex error 最大 `<=1e-6`。
- 当前 variation CV / 父版本 variation CV `<=0.90`，证明跨单元尺度离散度至少下降 10%。
- LayerNorm 不新增参数，候选/控制参数必须为 `6348/6260`。
- 相对固定 LSTM model-step 和 end-to-end speed ratio 均 `>=3.0×`。

效果失败状态为 `stop_dynamic_skip_token_layernorm_unstable_v24`；效果通过但速度失败为 `stop_dynamic_skip_token_layernorm_speed_v24`；全部通过才是 `dynamic_skip_token_layernorm_multiseed_confirmed_v24`。不得事后调低门槛、删 fold、修改 normalization 或覆盖 artifact。

## 最终验收与报告

运行定向测试后执行全仓 Ruff、mypy、完整 pytest、`tasks/preflight.py`、`tasks/test.py` 和 production wheel/sdist build。报告必须回答：

- normalization 是否无参数、保持零初始化等价和因果性；
- variation CV 是否下降；
- 相对静态和共同 LR 动态 TCN 是否跨种子改善；
- 速度是否仍满足 3×；
- LayerNorm 假设得到确认、否定还是证据不足。

不得访问 sealed test、部署、连接券商、执行交易或承诺收益。
