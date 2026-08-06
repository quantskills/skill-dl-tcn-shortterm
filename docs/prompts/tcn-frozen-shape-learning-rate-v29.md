# TCN 冻结 shape 分支学习率 v29：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，LSTM 只能作为冻结 benchmark。使用冻结的 2021–2025 真实股票分钟线 ordinary validation；禁止访问 sealed test、连接券商、交易、部署、承诺收益或事后修改门槛。

## 已知事实与当前唯一问题

- 数据、标签、PIT 标的池、5 分钟 bar、480 bar 回看、五折 expanding walk-forward、purge/embargo、特征标准化和 LSTM benchmark 全部冻结。
- TCN 继续使用 chomp 因果卷积、weight normalization、dilations `1,2,4,8,16,32,64,128`、channels `16` 和 memmap；不得改变感受野、模型结构、特征、损失或预测任务。
- v28 从逐 seed/fold 的 v20/v21 checkpoint 初始化，冻结 6348 个 parent 参数，只训练 88 个 shape residual 参数；parent prediction 误差、raw-only RankIC 误差和 frozen state drift 均为 `0`。
- v28 已将 checkpoint 保存阈值与 patience 解耦，56 个训练 score 与 v27 完全一致，并把 mean RankIC 提高到 `0.099543`；相对 parent `+0.000647`，相对 v27 `+0.000072`，9/15 单元保留 shape。
- v28 使用 Adam `0.003`。生效单元的 shape output L2 很快到达约 `0.4–1.0`，仍有 6/15 单元回退 epoch 0；这提示 shape-only 小参数分支可能存在学习率过大、早期越过较优 RankIC 区域的问题，但尚未得到因果验证。
- v29 唯一问题是：在训练、模型与选择协议全部不变时，仅将 shape-only Adam 学习率从 `0.003` 降到 `0.001`，能否在逐单元 paired ordinary validation 上稳定超过 v28。

## 预注册假设与证伪条件

按优先级检验以下互斥解释：

1. **步长过大**：`0.003` 对 88 个 shape 参数产生过冲；`0.001` 应降低更新幅度、增加有效 checkpoint 覆盖并提升相对 v28 的 paired RankIC。
2. **低学习率欠拟合**：`0.001` 在 `max_epochs=8`、`patience=2` 下不能及时产生可保存增量，effect 单元减少或 mean RankIC 退化。
3. **学习率无关**：v29 与 v28 仅有噪声级差异，无法通过 `+0.00015` paired gate。
4. **shape residual 信息上限**：即使优化轨迹更平稳，也无法缩小与 LSTM 的效果差距；此时停止学习率方向，下一轮才可预注册 rank-aligned objective，不能在 v29 中临时修改 loss。

只允许由本轮冻结证据判定上述解释。不得因为结果接近而更换解释、门槛、seed、fold 或 epoch。

## 唯一允许的训练干预

- candidate 学习率：`0.001`；历史 v28 control 学习率：`0.003`。
- optimizer 必须是 Adam，且只能包含 88 个 `dynamic_skip_shape_residual` 参数；identity 必须精确为 `shape-residual-only-lr-0.001`。
- checkpoint 选择继续使用 v28 双阈值：`checkpoint_min_delta=0.0`、`patience_min_delta=0.0005`、`patience=2`，identity 为 `best-any-strict-improvement+patience-material-0.0005`。

严禁修改：parent checkpoint、shape residual 结构、shape scale `0.25`、batch `128`、smooth L1、weight decay `0`、max epochs `8`、线程 `8`、seed、fold、数据、标签、dataloader、损失、scheduler、warm-up、模型精度或 checkpoint 门槛。

## TDD 与公开审计接缝

1. 先写失败测试，证明 frozen candidate 的 optimizer 只包含 shape residual 参数且参数量精确为 `88`，学习率精确为 `0.001`，identity 精确为 `shape-residual-only-lr-0.001`。
2. 新增公开不可变 v29 decision：输入 candidate、冻结历史 controls、diagnostics 和 LSTM comparison；先复用 v27/v28 的 parent 完整性、效果与速度检查，再单独做 v29 对 v28 的 paired 检查。
3. current leaderboard 必须精确包含 15 个 `(seed,fold)`，v28 历史 control 也必须精确包含同样 15 个单元；重复、缺失、非有限值或 trial identity 漂移必须 fail-closed。
4. leaderboard 必须记录学习率、optimizer identity、checkpoint/patience 阈值、selection identity、best/baseline epoch、总/冻结/可训练参数、parent checkpoint SHA、parent prediction error 和 frozen state drift。
5. v29/v28 比较必须按 `(seed,fold)` 一对一连接并输出 candidate、v28、paired delta；不得以非配对总体均值代替。
6. 提供通过、optimizer identity 漂移、相对 v28 无增益三类 decision 测试；历史配置及 v28 默认行为不得回归。
7. shape output L2、shape weight effect、effect 单元数和 best epoch 分布作为机制证据输出；L2 相对 v28 的变化只做诊断，不作为事后 promotion gate。

## 冻结实验协议

- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-lr001`
- v28 historical control：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-decoupled-selection`
- static control：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- v26：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- seeds：`7,17,27`；folds：`0..4`；只训练 15 个 v29 shape-only 单元，不重训任何历史模型。
- candidate 总参数 `6436`、冻结参数 `6348`、可训练参数 `88`、shape normalization 参数 `0`。
- seed 7 从 v20、seeds 17/27 从 v21 加载精确 parent checkpoint；必须验证 v20/v21/v25/v26/v28 receipt、selection、输出哈希、checkpoint SHA-256、源数据哈希、coverage 和 `sealed_test_accessed=false`。
- 使用 float32、torch threads `8`、num workers `0`；临时目录写完后原子改名，拒绝覆盖已有 artifact。

## 预注册门槛

- candidate 15 单元 mean RankIC `>=0.0996`，且 `15/15` 为正。
- 相对 frozen parent paired mean delta `>=+0.00075`；三个 seed mean delta 均 `>=0`；每 seed `5/5` folds 不退化。
- 相对 v28 paired mean delta `>=+0.00015`，且三个 seed mean delta均 `>=0`。
- 相对静态 paired mean delta `>=+0.002`；相对 v26 `>=+0.0015`；相对 v25 `>=+0.003`。
- 四期限相对 parent 保护门槛：1d/2d/3d/5d 均 `>=-0.001`。
- parent RankIC/prediction 最大绝对误差 `<=1e-7`；frozen state drift 必须严格为 `0`。
- 可训练参数 `88`、冻结参数 `6348`、总参数 `6436`；optimizer identity、学习率、shape scale 和 selection identity 必须精确匹配。
- shape output L2 `>1e-12` 且 weight effect `>=1e-6` 的单元不少于 `8/15`。
- simplex error `<=1e-6`；median samples/s `>=4500`；冻结 LSTM 对比 model-step 与 end-to-end speed ratio 均 `>=3.0×`。

完整性失败：`stop_frozen_shape_lr_integrity_v29`；效果失败：`stop_frozen_shape_lr_no_gain_v29`；效果通过但速度失败：`stop_frozen_shape_lr_speed_v29`；全部通过：`frozen_shape_lr001_confirmed_v29`。不得因结果接近而降低阈值、删除 fold、改 seed 或覆盖 artifact。

## 产物与最终验收

实现 v29 专用决策器与 runner，输出 resolved config、epoch history（含 epoch 0 和双改善标记）、candidate leaderboard、父 checkpoint manifest、v28 paired comparison、历史 controls、full/raw-only diagnostics、seed/horizon summary、LSTM comparison、15 个 checkpoint、selection 和 receipt。

真实实验后运行 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py`、production wheel/sdist build 和 `git diff --check`；重算 receipt ID，验证全部输出 SHA-256 和 checkpoint 数量。

报告必须明确回答：`0.001` 是否逐单元超过 v28；三个 seed、四期限与 effect coverage 是否稳定；shape 更新幅度是否下降；父路径、选择规则与数据是否完全冻结；当前结果支持过冲、欠拟合、学习率无关还是 shape 信息上限；TCN 与 LSTM 当前速度和效果差距各是多少。任何门槛不通过都必须停止，不得访问 sealed test。
