# TCN v31：日期分组 batch 顺序稳定化诊断与单变量优化

你正在维护 `skill-dl-tcn-shortterm`。本轮不是更换 TCN、增加模型容量或继续扫描 loss，而是修复 v30 暴露出的训练基础设施不稳定：`DateGroupedBatchSampler` 在每个 epoch 都用同一个 seed 重新生成完全相同的日期顺序，使日期组合、Adam 状态和少量 shape-residual 参数形成固定轨迹耦合。v30 的 grouped SmoothL1 相对 v28 总体仅 `+0.000248`，seed 7/17 分别为正，但 seed 27 为 `-0.000167`，其中 fold 4 为 `-0.000643`。本轮必须先证伪，再只改一个变量。

## 目标

实现并验证 `epoch_seeded` 日期顺序策略：同一 `(seed, fold, epoch)` 可精确重放，不同 epoch 使用不同的确定性日期排列；与 `fixed_once` 同期配对对照，保持 TCN 架构、父 checkpoint、SmoothL1、date-grouped batching、batch cap、学习率、epoch/patience、数据和验证协议完全一致。

## 已冻结事实

- 真实数据：本地 Pandadata 近五年 A 股 5 分钟线派生的 23,821 个样本，8 个因果特征，窗口 480 bars，memmap 输入。
- 任务：按信号日做横截面 1/2/3/5 日收益排名预测。
- TCN：strict causal chomp、WeightNorm、channels=16、kernel=3、dilations=`1,2,4,8,16,32,64,128`、感受野 511，覆盖 480 bars。
- 当前模型：v28/v30 的 frozen-parent shape-residual dynamic-horizon-skip；总参数 6,436，可训练参数 88，父参数 6,348 必须逐位不漂移。
- 当前 loss：仅 masked SmoothL1；禁止 soft-RankIC、pairwise loss、PCGrad 或新辅助 loss。
- 当前优化器：Adam、lr=0.003、weight_decay=0。
- 当前 batch：按 `signal_date` 完整分组并在 batch cap=128 内打包，禁止跨日期构造排名目标；`num_workers=0`、float32。
- 当前验证：3 seeds `(7,17,27)` × 5 ordinary-validation folds；best checkpoint 使用任意严格提升，patience 使用 material delta=0.0005；最多 8 epochs，patience=2。
- 已完成速度门：TCN/LSTM model-step 约 4.001×，end-to-end 约 3.854×；v31 不得跌破 3×。
- 禁止 sealed test；不得联网、不得读取 broker、不得修改数据窗口或 split。

## 待裁决假设（按优先级）

1. `fixed_once` 让每个 epoch 的日期排列完全相同，日期组合和 Adam 状态发生固定耦合，是 seed 27 轨迹不稳定的可控来源。
2. 物理 batch 大小/有效标签数波动造成梯度尺度变化；若顺序变化后梯度范数 CV 没下降，则本轮机制未被证实。
3. seed 27 的差异主要来自 checkpoint/early-stopping 抖动；若最佳 epoch 改变但 bootstrap 效果不稳定，应停止优化。
4. `-0.000167` 属于验证噪声；若日期级 paired block-bootstrap 95% CI 跨零，则不得宣称预测提升。
5. seed 27 父模型基线较高、shape 分支提升空间较小；该项仅记录为不可控解释，不能据此改架构。

## 唯一允许的行为变量

为 `DateGroupedBatchSampler` 和 `TCNTuningTrial` 增加显式 `date_batch_order`：

- `fixed_once`：完全保持 v30 行为，每个 epoch 使用相同日期排列；作为同期 control。
- `epoch_seeded`：epoch 0 使用基准排列，此后用 `(base_seed, epoch)` 派生确定性随机流；相同 epoch 可重放、不同 epoch 的顺序 fingerprint 必须不同。

禁止同时引入梯度累积、梯度裁剪、学习率变化、batch size 变化、loss 重加权、模型变化、dropout 或 weight decay。

## 实现要求

1. sampler 提供 `set_epoch(epoch)`；负 epoch fail-closed。
2. `fixed_once` 对 `set_epoch` 不敏感；`epoch_seeded` 对相同 epoch 精确重放。
3. 训练循环必须在创建 epoch iterator 前设置 epoch；control 和 candidate 除 `date_batch_order` 外逐字段相同并共享同一父 checkpoint。
4. leaderboard 写入 `date_batch_order`；epoch history 写入：
   - `date_order_epoch`
   - `date_order_fingerprint`
   - `optimizer_step_count`
   - `gradient_norm_mean/std/cv/max`
   - `batch_size_mean/std/cv/min/max`
5. 梯度诊断只能观察，不得改变梯度或 optimizer 行为。
6. 增加日期级 paired RankIC evidence，键为 `(seed,fold,horizon,signal_date)`，同一 candidate/control 必须一对一。
7. 对日期有序的 paired delta 做确定性 circular block bootstrap，固定 2,000 draws；报告全局、每 seed 和 seed 27 的 mean、CI low/high。
8. 所有 artifact 原子写入新目录，拒绝覆盖；receipt 绑定数据、父 checkpoint、配置、代码、输出哈希，并明确 `sealed_test_accessed=false`。

## 真实配对实验

- control：`grouped_smooth_l1 + fixed_once`
- candidate：`grouped_smooth_l1 + epoch_seeded`
- 两者都跑 seeds 7/17/27 × folds 0..4。
- 两者必须使用相同模型 seed、父 checkpoint、训练/验证样本、初始 shape 参数、optimizer、epoch/patience 和 batch cap。
- LSTM 只复用冻结 benchmark 测量，不重新调参。

## 预注册门槛

### 完整性门

- 30 个 current units 齐全，无重复。
- candidate/control 父 checkpoint SHA-256 逐 unit 相同。
- frozen parent state drift=0；parent prediction max error=0。
- 两者 loss identity 都是 `date-grouped-smooth-l1`，batching identity 都是 `date-grouped`。
- control 只有一个跨 epoch order fingerprint；candidate 在训练 epoch 数大于 1 时必须有多个 fingerprint，且重放测试通过。
- 没有 sealed 访问。

### 机制门

- control 每个 unit 的跨 epoch order fingerprint 数必须恰为 1。
- candidate 每个 unit 的 order fingerprint 数必须等于实际训练 epoch 数，证明每个 epoch 的日期暴露顺序唯一且可重放。
- candidate/control 的梯度范数 CV 必须有限并作为观察证据报告，但不预注册方向门槛：重排消除的是固定顺序偏差，并不数学保证单 epoch 梯度离散度下降。
- 若日期顺序机制未实际生效，则状态必须是 `stop_epoch_seeded_mechanism_not_confirmed_v31`。

### 效果门

- candidate 相对同期 control 的 15-unit 配对 mean RankIC delta `>= +0.00015`。
- 每个 seed 的配对 mean delta `>= 0`，seed 27 `>= +0.00015`。
- 非退化 units 至少 12/15（delta `>= -0.00010`）。
- 日期级 paired block-bootstrap 全局 95% CI low `> 0`；seed 27 CI low `>= 0`。
- candidate mean RankIC 不低于 v30 grouped control 的 `0.099791`。
- 若 bootstrap CI 跨零，即使点估计为正，也只能报告“未证实”。

### 速度门

- model-step TCN/LSTM `>=3.0×`。
- end-to-end TCN/LSTM `>=3.0×`。
- candidate median samples/s 不低于 control 的 90%。

### 决策状态

- 完整性失败：`stop_epoch_seeded_integrity_v31`
- 机制未证实：`stop_epoch_seeded_mechanism_not_confirmed_v31`
- 效果未证实：`stop_epoch_seeded_no_gain_v31`
- 速度失败：`stop_epoch_seeded_speed_v31`
- 全部门通过：`epoch_seeded_grouped_batch_confirmed_v31`

无论结果如何，`sealed_test_authorized=false`；本轮只决定是否保留 `epoch_seeded` 作为下一版 grouped TCN 的工程默认值。

## 必需产物

- `config.resolved.json`
- `tcn-epoch-history.parquet`
- `tcn-leaderboard.parquet`
- `batch-order-diagnostics.parquet`
- `paired-rankic-by-date.parquet`
- `bootstrap-summary.parquet`
- `paired-unit-comparison.parquet`
- `shape-diagnostics.parquet`
- `parent-checkpoint-manifest.parquet`
- `seed-summary.parquet`
- `comparison.json`
- `selection.json`
- `report.md`
- `receipt.json`
- `checkpoints/*.pt`

## 验证命令

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tcn_grouped_batch_order_stability.py -q
.venv\Scripts\python.exe tasks/run_tcn_grouped_batch_order_stability.py --run-dir <real-run-dir> --split-manifest <ordinary-split> --config config/pandadata-tcn-grouped-batch-order-stability-v31.example.json --output-dir artifacts/tcn-grouped-batch-order-stability-v31
.venv\Scripts\python.exe tasks/preflight.py
.venv\Scripts\python.exe tasks/test.py
```

先用测试复现 `fixed_once` 的跨 epoch 重复顺序，再实现 `epoch_seeded`；真实结果不满足门槛时保留诊断和 artifact，但不得把失败包装成成功。
