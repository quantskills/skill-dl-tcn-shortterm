# TCN 冻结父模型 shape residual v27：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，LSTM 只能作为冻结 benchmark。使用冻结的 2021–2025 真实股票分钟线 ordinary validation；禁止访问 sealed test、连接券商、交易、部署、承诺收益或事后修改门槛。

## 已知事实与待证伪假设

- 数据、标签、PIT 标的池、5 分钟 bar、480 bar 回看、五折 expanding walk-forward、purge/embargo、特征标准化和 LSTM benchmark 全部冻结。
- TCN 必须继续使用 chomp 因果卷积、weight normalization、dilations `1,2,4,8,16,32,64,128`、channels `16` 和 memmap；不得改变感受野、数据加载、损失或预测任务。
- v21 raw-token 动态 skip 是当前父模型，三种子十五单元 mean RankIC 约 `0.098896`。
- v26 在初始化时严格保持 v21 raw 路径，新增 shape residual 后 mean RankIC 为 `0.097328`；相对静态 `+0.002094`、相对 v21 `-0.001568`、相对 v25 `+0.002800`，TCN/LSTM model-step 与 end-to-end 分别约 `3.755×/3.638×`。
- v26 的 shape 分支在十五单元全部生效，但 trunk、raw scorer、skip logits 和 heads 与 shape 分支共同训练，无法排除父路径漂移是退化主因。
- v27 的唯一问题是：在逐 seed/fold 精确冻结 v21 checkpoint 后，仅训练 88 个 shape residual 参数，能否在不破坏父模型的情况下取得可复现增量。

## 唯一允许的模型与训练干预

沿用 v26 `ShapeResidualDynamicHorizonSkipTCN`：

- raw parent：`raw token → Linear(16,4) → tanh → Linear(4,4)`；
- shape residual：`LayerNorm(16, elementwise_affine=False) → Linear(16,4) → tanh → Linear(4,4)`；
- combined logits：`raw_logits + 0.25 × tanh(shape_logits)`；
- final weights：`softmax(skip_logits + tanh(combined_logits), block_dim)`；
- readout 始终混合 raw token。

对每个 seed/fold：

1. seed 7 从 v20 checkpoint 加载对应 fold 的 v21 raw parent；seeds 17/27 从 v21 confirmation artifact 加载对应 checkpoint。
2. 加载必须 fail-closed：父 checkpoint 的 key、shape、dtype 必须与 candidate 的全部非 shape state 完全一致；只允许缺少四个 shape tensor。
3. 父 checkpoint 加载后，冻结 trunk、raw scorer、static skip logits 和四个 horizon heads；只有 `dynamic_skip_shape_hidden` 与 `dynamic_skip_shape_output` 的 88 个参数 `requires_grad=True`。
4. 优化器只接收这 88 个参数，使用 Adam `0.003`、weight decay `0`；optimizer identity 固定为 `shape-residual-only-lr-0.003`。
5. shape 输出层保持全零初始化。训练前 full prediction、raw-only prediction 和独立 v21 parent prediction必须逐元素严格相同。
6. 在 epoch 1 前评估 epoch 0 父模型，并将其作为合法的 best checkpoint。后续 epoch 只有在 ordinary-validation mean RankIC 超过当前 best 至少 `0.0005` 时才能替换；若没有提升，最终 checkpoint 必须保持 epoch 0 父模型等价状态。
7. max epochs `8`、patience `2`、batch `128`、smooth L1、float32、8 threads；不得添加 scheduler、warm-up、特殊损失、梯度技巧、数据变化或 shape scale 搜索。

## TDD 与审计接缝

通过公开接口逐个完成红灯—绿灯：

1. `load_frozen_raw_parent(parent_state)`：严格加载父 state，拒绝缺 key、多 key、shape/dtype 漂移；返回/暴露可审计冻结身份。
2. `shape_residual_parameters()`：恰好 88 个且是全部可训练参数；任何非 shape 参数仍可训练都必须失败。
3. `forward_without_shape_residual(inputs)`：加载后与独立父模型以及初始化 full forward 逐元素一致。
4. sweep 的 frozen-parent 模式必须在 epoch 0 建立 baseline；用一个会恶化的训练 fixture 验证最终仍选择 epoch 0。
5. 训练后比较所有非 shape state 与源 checkpoint，最大漂移必须严格为 `0`；shape state 可以变化。
6. leaderboard/receipt 必须记录 `parent_checkpoint_sha256`、`parent_prediction_max_abs_error`、`frozen_parent_state_drift_max`、`trainable_parameter_count=88`、`frozen_parameter_count=6348`、optimizer identity、`baseline_epoch=0` 和 best epoch。
7. diagnostics 同时保存 full、raw-only 权重差，以及 full/raw-only RankIC；raw-only RankIC 必须与历史 parent RankIC 在数值容差内一致。

## 冻结实验协议

- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025`
- static control：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- v26：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 shape-only 单元，不重训历史模型。
- candidate 总参数 `6436`、冻结参数 `6348`、可训练参数 `88`、shape normalization 参数 `0`。
- 必须验证 v20/v21/v25/v26 receipt ID、selection status、输出哈希、checkpoint SHA-256、源数据哈希、seed/fold coverage 以及 `sealed_test_accessed=false`。
- 临时目录完成后原子改名，拒绝覆盖已有 artifact。

## 预注册门槛

- candidate 15 单元 mean RankIC `>=0.0995`，且 `15/15` 为正。
- 相对 v21 frozen parent paired mean delta `>=+0.0005`；三个 seed 的 mean delta 均 `>=0`；每 seed 至少 `3/5` folds 不退化。
- 相对静态 paired mean delta `>=+0.002`。
- 相对 v26 paired mean delta `>=+0.0015`；相对 v25 paired mean delta `>=+0.003`。
- 四期限相对 parent 保护门槛：1d/2d/3d/5d 均 `>=-0.001`。
- epoch-0 raw-only RankIC 与历史 parent 的单元最大绝对误差 `<=1e-7`。
- 父 prediction 最大绝对误差 `<=1e-7`；冻结 state drift 必须严格 `0`。
- 可训练参数恰好 `88`、冻结参数 `6348`、candidate 总参数 `6436`；optimizer identity 精确匹配。
- shape output L2 `>1e-12` 且 shape weight effect `>=1e-6` 的训练后单元不少于 `8/15`；允许未改善单元保留 epoch 0 和零 shape effect。
- dynamic simplex error `<=1e-6`。
- 由于仅训练 88 个参数改变了计算协议，吞吐只做非退化保护：median samples/s `>=4500`；冻结 LSTM 对比的 model-step 与 end-to-end speed ratio 均 `>=3.0×`。

效果失败状态：`stop_frozen_parent_shape_residual_no_gain_v27`；冻结/等价性失败：`stop_frozen_parent_integrity_v27`；效果通过但速度失败：`stop_frozen_parent_shape_residual_speed_v27`；全部通过：`frozen_parent_shape_residual_confirmed_v27`。不得因为结果接近而降低阈值、删除 fold、改 seed、改 scale 或覆盖 artifact。

## 产物与最终验收

实现 v27 专用决策器与 runner，输出 resolved config、epoch history（含 epoch 0）、candidate leaderboard、父 checkpoint manifest、历史 controls、full/raw-only diagnostics、seed/horizon summary、LSTM comparison、15 个 candidate checkpoint、selection 和 receipt。

真实实验后运行 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py`、production wheel/sdist build 和 `git diff --check`；重算 receipt ID，验证全部输出 SHA-256 和 checkpoint 数量。

报告必须明确回答：父 checkpoint 是否逐单元严格保持；shape-only 是否超过父模型；改善来自多少 seed/fold 和哪些 horizon；epoch 0 回退触发多少次；速度是否保持；联合训练漂移假设被确认、否定还是证据不足。任何门槛不通过都必须停止，不得访问 sealed test。
