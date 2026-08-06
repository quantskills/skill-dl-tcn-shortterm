# TCN 冻结 shape 分支 soft-RankIC v30：完整执行提示词

你是 `skill-dl-tcn-shortterm` 项目的实现、实验与验收代理。继续优化 TCN 主模型，LSTM 只能作为冻结 benchmark。使用冻结的 2021–2025 真实股票分钟线 ordinary validation；禁止访问 sealed test、连接券商、交易、部署、承诺收益或事后修改门槛。

## 已知事实与当前唯一问题

- 数据、横截面 rank target、PIT 标的池、5 分钟 bar、480 bar 回看、五折 expanding walk-forward、purge/embargo、特征标准化和 LSTM benchmark 全部冻结。
- TCN 继续使用 chomp 因果卷积、weight normalization、dilations `1,2,4,8,16,32,64,128`、channels `16` 和 memmap；不得改变感受野、模型结构、特征或预测任务。
- v28 冻结 6348 个 parent 参数，只训练 88 个 shape residual 参数，Adam `0.003`、Smooth L1，mean RankIC `0.099543`，相对 parent `+0.000647`，9/15 shape 单元生效。
- v29 仅把学习率降到 `0.001`，shape output L2 明显下降，但 mean RankIC 降至 `0.099225`，三个 seed 均低于 v28。因此“`0.003` 明显过冲”被证伪，不再降低学习率。
- v12 曾在不同的 temporal-context 全参数模型上使用 weight `0.2`、temperature `0.1` 的 soft-RankIC，效果和速度都退化。本轮不是重复扫描：只在冻结 parent 的 88 参数 shape 分支上做一个预注册小权重 `0.05` 探针，并设置相同 batching 的 contemporaneous control，检验目标函数而非模型结构。
- v30 唯一问题是：在 parent、学习率和 infra 不变时，给 shape-only Smooth L1 增加小权重、按信号日和 horizon 隔离的 differentiable RankIC surrogate，能否稳定超过相同 date-grouped batching 的 Smooth L1 control 和历史 v28。

## 预注册假设与证伪条件

1. **目标错配**：pointwise Smooth L1 没有直接优化同日横截面顺序；soft-RankIC candidate 应同时超过 grouped-SmoothL1 control 与 v28。
2. **batching 效应**：date-grouped batching 本身改变优化轨迹；若 grouped control 改善而 candidate 没有额外增益，不能把结果归因于 rank loss。
3. **rank 梯度干扰**：即使权重降到 `0.05`，surrogate 仍干扰 Smooth L1；candidate 将低于 grouped control 或产生不稳定 seed。
4. **shape 信息/容量上限**：两种 grouped 训练都无法超过 v28；88 参数 shape 分支或当前八特征无法提供更多稳定横截面信息。

只能由冻结的 paired evidence 判定上述解释。不得因为结果接近而追加权重/temperature 搜索、改 seed、改 fold 或放宽门槛。

## 唯一允许的目标函数干预

训练两个逐 seed/fold、同 parent、同 RNG 协议的 trial：

1. grouped control：`grouped_smooth_l1`，只使用 masked Smooth L1；
2. candidate：`soft_rankic`，总损失为 `masked Smooth L1 + 0.05 × soft-RankIC loss`，temperature 固定 `0.1`。

两者都必须使用同一个 deterministic `DateGroupedBatchSampler`、batch cap `128`、seeded date order。一个物理 batch 可以打包多个信号日，但 objective 必须按 `(signal_date,horizon)` 分组，禁止跨日或跨 horizon 构造比较；不足两个有效股票的组不得产生 rank loss。目标只能消费训练 split 已有的 rank target，不得读取未来特征、validation label 或 sealed data。

两者均恢复 v28 的 Adam `0.003`；optimizer 只能包含 88 个 shape residual 参数，identity 必须精确为 `shape-residual-only-lr-0.003`。checkpoint 选择继续使用 `checkpoint_min_delta=0.0`、`patience_min_delta=0.0005`、`patience=2`。

严禁修改：parent checkpoint、shape residual 结构、shape scale `0.25`、batch cap、weight decay `0`、max epochs `8`、线程 `8`、数据、标签、特征、dataloader workers、模型精度、scheduler、warm-up 或 checkpoint 门槛。

## TDD 与公开审计接缝

1. `validate_tcn_tuning_plan` 只允许 `grouped_smooth_l1/soft_rankic` 用于 frozen-parent shape-residual dynamic-horizon-skip；相同策略用于非冻结 dynamic TCN 必须 fail-closed；历史 temporal-context soft-RankIC 仍可解析。
2. public loss 测试必须用已知两日、四 horizon 样本证明：正确横截面比反向横截面 loss 更低；不同 signal date/horizon 不发生比较；缺少有效横截面的组稳定回退；梯度有限且非零。
3. `run_tcn_validation_sweep` 的 tiny integration 必须证明两个 trial 都使用 `date-grouped` batching、精确 loss identity、相同 parent、相同 `0.003` shape-only optimizer，并保持 parent prediction error/state drift 为 `0`。
4. leaderboard/receipt 必须记录 strategy、loss identity、batching identity、soft-rank weight/temperature、optimizer/selection identity、参数量、best/baseline epoch 和 parent checkpoint SHA。
5. v30 decision 必须分别建立 candidate vs grouped control、candidate vs v28、candidate vs parent 的 15 单元 paired comparison；缺失、重复、非有限值、identity 漂移必须 fail-closed。
6. 先写通过、candidate loss identity 漂移、candidate 相对 grouped control 无增益三类失败测试，再实现最小代码使其通过；旧 v12/v27/v28/v29 测试不得回归。

## 冻结实验协议

- grouped control：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-grouped-smooth`
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-soft-rank-w005-tau01`
- v28 historical control：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-frozen-parent-shape-r025-decoupled-selection`
- static：`horizon-skip-c16-chomp-smooth`
- raw parent：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- v25：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-shape-log-rms`
- v26：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1-raw-shape-r025`
- seeds：`7,17,27`；folds：`0..4`；训练 15 个 grouped control 和 15 个 candidate，共 30 个新单元。
- 每个模型总参数 `6436`、冻结参数 `6348`、可训练参数 `88`、shape normalization 参数 `0`。
- seed 7 从 v20、seeds 17/27 从 v21 加载精确 parent checkpoint；必须验证 v20/v21/v25/v26/v28 receipt、selection、输出哈希、checkpoint SHA、源数据哈希和 `sealed_test_accessed=false`。
- 使用 float32、torch threads `8`、num workers `0`；临时目录写完后原子改名，拒绝覆盖已有 artifact。

## 预注册门槛

- candidate 15 单元 mean RankIC `>=0.0996`，且 `15/15` 为正。
- 相对 frozen parent paired mean delta `>=+0.00075`；三个 seed delta 均 `>=0`；每 seed `5/5` folds 不退化 parent。
- 相对 grouped control paired mean delta `>=+0.00015`，且三个 seed delta 均 `>=0`。
- 相对 v28 paired mean delta `>=+0.00015`，且三个 seed delta 均 `>=0`。
- 相对 static paired mean delta `>=+0.002`；相对 v26 `>=+0.0015`；相对 v25 `>=+0.003`。
- 四期限相对 parent：1d/2d/3d/5d 均 `>=-0.001`。
- candidate/control parent RankIC/prediction 最大绝对误差 `<=1e-7`；frozen state drift 严格为 `0`；两者 parent checkpoint SHA 必须逐单元相同。
- candidate identity 必须为 `soft_rankic`、`smooth-l1+0.05-soft-rankic-tau-0.1`、`date-grouped`、weight `0.05`、temperature `0.1`；control identity 必须为 `grouped_smooth_l1`、`date-grouped-smooth-l1`、`date-grouped`，且不携带 active soft-rank 参数。
- candidate/control optimizer 均为 `shape-residual-only-lr-0.003`；总/冻结/可训练参数精确为 `6436/6348/88`；selection identity 精确匹配。
- candidate shape output L2 `>1e-12` 且 weight effect `>=1e-6` 的单元不少于 `8/15`；simplex error `<=1e-6`。
- candidate median samples/s `>=4500`；冻结 LSTM 对比 model-step 与 end-to-end speed ratio 均 `>=3.0×`。

完整性失败：`stop_shape_rank_integrity_v30`；效果失败：`stop_shape_rank_no_gain_v30`；效果通过但速度失败：`stop_shape_rank_speed_v30`；全部通过：`shape_rank_objective_confirmed_v30`。任何失败都停在 ordinary validation，不得访问 sealed test。

## 产物与最终验收

实现 v30 专用决策器与 runner，输出 resolved config、两个 trial 的 epoch history/leaderboard、父 checkpoint manifest、candidate-vs-control-vs-v28 paired comparison、两个 trial 的 full/raw-only diagnostics、历史 controls、seed/horizon summary、LSTM comparison、30 个 checkpoints、selection 和 receipt。

真实实验后运行 focused tests、Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py`、production wheel/sdist build 和 `git diff --check`；重算 receipt ID，验证全部输出 SHA-256 和 checkpoint 数量。

报告必须明确回答：candidate 是否同时超过 grouped control 与 v28；batching 与 loss 各贡献多少；三个 seed、四 horizon 与 shape coverage 是否稳定；rank loss 对速度的代价；父模型和数据是否完全冻结；当前证据支持目标错配、batching 效应、rank 梯度干扰还是 shape 上限。不得以本轮接近为理由追加超参数扫描。
