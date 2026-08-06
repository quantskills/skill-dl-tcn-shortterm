# TCN 动态 skip 原始路径 + shape residual v26：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，LSTM 只能作为冻结 benchmark。使用冻结的真实股票分钟线 ordinary validation；禁止访问 sealed test、连接券商、交易、部署或承诺收益。

## 已知事实

- 当前数据协议为 2021–2025 真实股票分钟线、5 分钟 bar、480 bar 回看、五折 expanding walk-forward ordinary validation；数据、标签、PIT 标的池、purge/embargo 和特征标准化均冻结。
- TCN 使用 chomp 因果卷积、weight normalization、dilations `1,2,4,8,16,32,64,128` 和 memmap；不得改动这些边界。
- v21 原始 token、共同 Adam `0.003` 的动态 skip 是当前最佳动态 TCN，三种子合并 mean RankIC 约 `0.098896`。
- v24 用 LayerNorm 形状替换原始 token 后降至 `0.095746`；v25 用 LayerNorm 形状加单一 `log1p(RMS)` 幅度仍降至 `0.094527`。二者证明不能删除 raw token；variation CV 下降也没有带来效果提升。
- v25 还改变了隐藏层 fan-in 初始化，seed 17 五折全部退化，因此下一轮必须保留 v21 原始 scorer 的参数形状、创建顺序和初始化。

## 唯一允许的模型干预

实现 `ShapeResidualDynamicHorizonSkipTCN`，继承并完整保留 v21 `DynamicHorizonSkipTCN` 的 raw 分支：

`raw token → Linear(16,4) → tanh → Linear(4,4)`

在 raw 分支全部创建完成之后，新增独立 shape residual：

`LayerNorm(16, elementwise_affine=False) → Linear(16,4) → tanh → Linear(4,4)`

shape 输出层权重和 bias 必须全零初始化。最终动态 logits 为：

`raw_logits + 0.25 × tanh(shape_logits)`

然后沿用 v21 外层：

`skip_logits + 1.0 × tanh(combined_dynamic_logits) → block softmax`

必须满足：

- 当 shape residual logits 为零时，动态权重和最终模型输出与同 seed 的 v21 raw 模型逐元素严格相同。
- raw 分支的参数形状、初始化顺序和随机数消费保持 v21 一致；新增模块只能在 raw hidden/output 创建后实例化。
- 最终 horizon readout 始终混合原始 token，不得混合 LayerNorm token。
- shape scale 固定 `0.25`；不得搜索该值。
- 静态控制 `6260`、v21 raw `6348`、v26 `6436`；总动态参数 `176`，其中 raw `88`、shape residual `88`，LayerNorm 参数 `0`。
- 所有参数共同 Adam `0.003`；不得增加特殊学习率、warm-up、scheduler、weight decay、新损失、block identity 或数据变更。

## TDD 与公开行为边界

以公开接口做垂直红灯—绿灯切片：

1. `raw_dynamic_skip_logits(tokens)` 返回 raw 分支 logits；同 seed 的 v21 与 v26 必须完全相同。
2. `shape_residual_inputs(tokens)` 返回无仿射 LayerNorm 形状；对每 sample/block 的正比例缩放与通道平移保持不变。
3. `shape_residual_logits(tokens)` 在初始化时严格为零；显式改变 shape 输出层后，最终动态权重必须相对 raw-only 权重发生变化。
4. `dynamic_skip_weights_without_shape_residual(block_sequences)` 是 raw-only counterfactual；初始化时与最终权重严格相同，训练后用于量化 shape 分支实际影响。
5. 配置新增 `dynamic_skip_shape_residual=true` 和 `dynamic_skip_shape_residual_scale=0.25`；默认关闭，历史 `none/layer_norm/shape_log_rms` 行为不变。shape residual 只允许 `model_kind=dynamic_horizon_skip` 且 raw 输入模式 `none`，否则 fail-closed。
6. leaderboard/receipt 写入 shape residual identity、scale、raw/shape/总动态参数数、shape normalization 参数数、shape output L2 和共同优化器身份。
7. tiny sweep 测试必须直接验证上述审计列，避免再次出现训练结束后才发现 leaderboard 缺列。

## 冻结训练协议

- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- static control：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25 failed ablation：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 v26 TCN 单元。
- channels 16、kernel 3、8 dilations、dropout 0、batch 128、smooth L1、float32、8 threads、max epochs 8、patience 2、min delta 0.002。
- 从 v20/v21 冻结产物读取静态、raw parent 和 LSTM；从 v25 读取失败消融。必须验证 receipt ID、selection status、输出哈希、源数据哈希、seed/fold coverage 和 `sealed_test_accessed=false`，不得重训历史证据。
- 临时目录写完后原子改名；拒绝覆盖既有 artifact。

## 预注册门槛

- candidate 15 单元 mean RankIC `>=0.100`，且 `15/15` 为正。
- 相对静态 paired mean delta `>=+0.003`；每 seed delta `>0`，每 seed 至少 `3/5` folds 不退化。
- 相对 v21 raw parent paired mean delta `>=+0.001`，且每 seed delta 均 `>0`。
- 相对 v25 failed ablation paired mean delta `>=+0.003`，且每 seed delta 均 `>0`。
- horizon 相对静态保护门槛：1d `>=0`、2d `>=-0.003`、3d `>=-0.005`、5d `>=-0.005`。
- median samples/s `>=4500`；TCN/LSTM model-step 与 end-to-end speed ratio 均 `>=3.0×`。
- raw output weight L2、shape output weight L2 均 `>1e-12`。
- 每个 seed/fold 的 shape residual weight effect max `>=1e-6`，block variation `>=1e-6`，simplex error `<=1e-6`。
- 参数精确为控制 `6260`、candidate `6436`、总动态 `176`、raw `88`、shape `88`；shape normalization 参数 `0`；优化器 `all-lr-0.003`。

效果失败状态：`stop_dynamic_skip_raw_shape_residual_unstable_v26`；效果通过但速度失败：`stop_dynamic_skip_raw_shape_residual_speed_v26`；全部通过：`dynamic_skip_raw_shape_residual_multiseed_confirmed_v26`。不得事后降低阈值、删 fold、改 scale 或覆盖 artifact。

## 实验产物和最终验收

实现 v26 专用决策器与 runner，输出 resolved config、epoch history、candidate leaderboard、历史对照、candidate diagnostics、seed/horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。diagnostics 必须同时记录最终动态权重、raw-only counterfactual 和两者最大差值。

真实实验后运行 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py` 和 production wheel/sdist build；重算 receipt ID，验证全部输出 SHA-256 和 checkpoint 数量。

报告必须明确回答：

- raw 路径初始化是否与 v21 严格相同；
- shape residual 是否真实改变动态权重；
- v26 是否超过静态、v21 raw parent 和 v25；
- 三个 seed、四个 horizon 与吞吐是否稳定；
- 假设被确认、否定还是证据不足；
- 未通过门槛时必须停止，不得访问 sealed test。
