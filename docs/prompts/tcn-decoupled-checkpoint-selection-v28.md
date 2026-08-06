# TCN 冻结父模型解耦 checkpoint 选择 v28：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，LSTM 只能作为冻结 benchmark。使用冻结的 2021–2025 真实股票分钟线 ordinary validation；禁止访问 sealed test、连接券商、交易、部署、承诺收益或事后修改门槛。

## 已知事实与唯一问题

- 数据、标签、PIT 标的池、5 分钟 bar、480 bar 回看、五折 expanding walk-forward、purge/embargo、特征标准化和 LSTM benchmark 全部冻结。
- TCN 继续使用 chomp 因果卷积、weight normalization、dilations `1,2,4,8,16,32,64,128`、channels `16` 和 memmap；不得改变感受野、模型结构、特征、损失或预测任务。
- v27 从逐 seed/fold 的 v21 checkpoint 初始化，冻结 6348 个 parent 参数，只训练 88 个 shape residual 参数；parent prediction 误差与 frozen state drift 均为 `0`。
- v27 mean RankIC 为 `0.099471`，相对 parent `+0.000575`、相对静态 `+0.004238`、相对 v26 `+0.002143`；7/15 单元保存 shape，8/15 回退 epoch 0。
- v27 的 `min_delta=0.0005` 同时决定“是否保存 checkpoint”和“是否重置 patience”。真实 epoch history 显示两个回退单元曾出现 `+0.000338/+0.000267`，另有一个已接受单元出现更高分，但相对当前 best 的增量不足 `0.0005`，因此没有保存。
- v28 唯一问题是：在不改变任何训练轨迹、模型或超参数的条件下，解耦 checkpoint 选择与 patience anchor，能否恢复这些已经观测到但被丢弃的普通验证增量。

## 唯一允许的 infra 干预

保留 v27 的全部模型与训练协议，仅把验证选择改为双阈值状态机：

- `checkpoint_min_delta = 0.0`：只要当前 ordinary-validation mean RankIC 严格高于已保存 best score，就保存 checkpoint；
- `patience_min_delta = 0.0005`：只有当前 score 严格高于独立 patience anchor `0.0005`，才更新 anchor 并把 `epochs_without_material_improvement` 清零；否则计数加一；
- epoch 0 frozen parent 同时初始化 best score 和 patience anchor；
- early stopping 仍在计数达到 `patience=2` 时触发；
- 保存 checkpoint 不得改变 optimizer、model state、数据顺序、RNG 或 patience anchor；
- 默认 `checkpoint_min_delta=None` 时必须解析为原有行为，即 checkpoint 与 patience 都使用 `min_delta`，历史调用行为不变。

严禁修改：parent checkpoint、shape residual 结构、shape scale `0.25`、Adam `0.003`、batch `128`、smooth L1、weight decay `0`、max epochs `8`、patience `2`、线程 `8`、seed、fold、数据、标签、dataloader、损失、scheduler 或 warm-up。

## TDD 与公开审计接缝

1. 提供公开不可变 `ValidationSelectionState` 与 `advance_validation_selection(...)`：输入当前 score、best、patience anchor、计数及两个 delta，返回 checkpoint 是否改善、patience 是否实质改善和下一状态。
2. 固定序列测试：baseline `0.1000`，随后 `0.1003` 必须保存 checkpoint 但不重置 patience；`0.1006` 必须再次保存并重置 patience；更低 score 两者都不得触发。
3. 历史默认测试：当两个 delta 都为 `0.0005` 时，`0.1003` 不保存也不重置，保持 v27 以前语义。
4. fail-closed：delta 非有限、为负、checkpoint delta 大于 patience delta、score 非有限时必须拒绝或明确不改善，不得污染 best state。
5. sweep 新增可选 `checkpoint_min_delta`；epoch history 必须分别记录 `checkpoint_improved`、`patience_improved` 和 `epochs_without_material_improvement`。
6. leaderboard/receipt 必须记录 `checkpoint_min_delta=0`、`patience_min_delta=0.0005`、`checkpoint_selection_identity=best-any-strict-improvement+patience-material-0.0005`、best epoch、epoch 0 baseline 和冻结完整性字段。
7. v28 runner 必须将 v28 与 v27 的 `(seed,fold,epoch)` score 轨迹做差分：coverage 必须完全相同、mean RankIC 最大绝对误差 `<=1e-12`；否则状态必须为 integrity failure。
8. 每个 v28 leaderboard best score 必须等于其 epoch history 中含 epoch 0 的最大 score，最大选择误差 `<=1e-12`。

## 冻结实验协议

- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-decoupled-selection`
- static control：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- v26：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- v27：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 v28 shape-only 单元，不重训历史模型。
- candidate 总参数 `6436`、冻结参数 `6348`、可训练参数 `88`、shape normalization 参数 `0`。
- seed 7 从 v20、seeds 17/27 从 v21 加载精确 parent checkpoint；必须验证 v20/v21/v25/v26/v27 receipt、selection、输出哈希、checkpoint SHA-256、源数据哈希、coverage 和 `sealed_test_accessed=false`。
- 临时目录完成后原子改名，拒绝覆盖已有 artifact。

## 预注册门槛

- candidate 15 单元 mean RankIC `>=0.0995`，且 `15/15` 为正。
- 相对 frozen parent paired mean delta `>=+0.0007`；三个 seed mean delta 均 `>=0`；每 seed `5/5` folds 不退化。
- 相对 v27 paired mean delta `>=+0.00015`，且三个 seed mean delta 均 `>=0`。
- 相对静态 paired mean delta `>=+0.002`；相对 v26 `>=+0.0015`；相对 v25 `>=+0.003`。
- 四期限相对 parent 保护门槛：1d/2d/3d/5d 均 `>=-0.001`。
- v28/v27 epoch score trajectory coverage 完全一致，最大绝对误差 `<=1e-12`。
- selected best 与 epoch-history max score 最大绝对误差 `<=1e-12`。
- parent RankIC/prediction 最大绝对误差 `<=1e-7`；frozen state drift 必须严格为 `0`。
- 可训练参数 `88`、冻结参数 `6348`、总参数 `6436`；optimizer 与 selection identity 精确匹配。
- shape output L2 `>1e-12` 且 weight effect `>=1e-6` 的单元不少于 `8/15`。
- simplex error `<=1e-6`；median samples/s `>=4500`；冻结 LSTM 对比 model-step 与 end-to-end speed ratio 均 `>=3.0×`。

选择/冻结完整性失败：`stop_decoupled_checkpoint_integrity_v28`；效果失败：`stop_decoupled_checkpoint_no_gain_v28`；效果通过但速度失败：`stop_decoupled_checkpoint_speed_v28`；全部通过：`decoupled_checkpoint_selection_confirmed_v28`。不得因结果接近而降低阈值、删除 fold、改 seed 或覆盖 artifact。

## 产物与最终验收

实现 v28 专用决策器与 runner，输出 resolved config、epoch history（含 epoch 0 和双改善标记）、candidate leaderboard、父 checkpoint manifest、v27 trajectory comparison、历史 controls、full/raw-only diagnostics、seed/horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。

真实实验后运行 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py`、production wheel/sdist build 和 `git diff --check`；重算 receipt ID，验证全部输出 SHA-256 和 checkpoint 数量。

报告必须明确回答：训练 score 轨迹是否与 v27 完全一致；解耦后新增保存多少 checkpoint；v28 是否超过 parent/v27；三个 seed、四个 horizon 与速度是否稳定；被恢复的是选择 infra 丢失的既有增量还是训练变化。任何门槛不通过都必须停止，不得访问 sealed test。
