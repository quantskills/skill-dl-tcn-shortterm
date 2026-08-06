# TCN v32：等日期、等 horizon 的 SmoothL1 单变量优化

你正在维护 `skill-dl-tcn-shortterm`。本轮必须继续使用 TCN，不得换成 LSTM、Transformer 或树模型。v31 已证明日期顺序每 epoch 重排机制能够正确工作，但真实 RankIC 无增益，因此保留 v30/v31 control 的 `fixed_once`。本轮只解决一个新的、可计算的权重错配：训练损失按有效股票标签逐项平均，而验证 RankIC 按 `(signal_date, horizon)` 等权平均。股票数更多、有效标签更多的日期会在训练中得到更大权重，但不会在验证中得到更大权重。

## 目标

实现 `date_horizon_mean` SmoothL1 reduction：先在每个 `(signal_date, horizon)` 内对有效股票的 SmoothL1 求均值，再对所有有效日期/期限组等权求均值。用 `label_mean` 作为同期 control，完成真实五年分钟线、3 seeds × 5 folds 配对验证和日期块 bootstrap。

## 已知证据

- v31 control 逐 unit 精确重现 v30 grouped control，mean RankIC `0.0997909483`。
- v31 `epoch_seeded` mean RankIC `0.0997345226`，相对 control `-0.0000564`；全局 bootstrap 95% CI `[-0.000247,+0.000135]`。
- v31 中 10/15 units 完全不变，12/15 candidate units 选择 epoch 0 或 1；瓶颈更像 shape 分支早期学习信号，而不是日期顺序。
- 现有 `masked_smooth_l1` 对全部有效标签求均值，因此日期股票数与 horizon 有效率会隐式决定训练权重。

## 冻结合同

- 数据：近五年 Pandadata A 股 5 分钟线，23,821 个样本、8 个因果特征、480 bars，memmap 读取。
- 任务：每个信号日的 1/2/3/5 日横截面收益排名。
- TCN：strict causal chomp、WeightNorm、channels=16、kernel=3、dilations=`1,2,4,8,16,32,64,128`、感受野 511。
- 模型：frozen-parent shape-residual dynamic-horizon-skip，总参数 6,436，可训练 shape 参数 88，父参数 6,348。
- optimizer：Adam、lr=0.003、weight_decay=0。
- batching：date-grouped、batch cap=128、`fixed_once`、num_workers=0、float32。
- 训练：最多 8 epochs，patience=2；checkpoint 任意严格提升，patience material delta=0.0005。
- 验证：seeds `(7,17,27)` × folds `0..4`，ordinary validation only。
- 禁止 sealed test、未来 padding、BatchNorm、未来特征、validation label 进入训练、外部写入和部署。

## 唯一允许变量

在 `TCNTuningTrial` 增加显式 `grouped_smooth_l1_reduction`：

- control：`label_mean`，完全复现 v31 fixed-once control；loss identity 保持 `date-grouped-smooth-l1`。
- candidate：`date_horizon_mean`；loss identity 为 `date-horizon-equal-smooth-l1`。

两者除 `trial_id` 和 reduction 外，所有字段必须逐字段相同。禁止同时改变 sampler、batch size、学习率、模型容量、dropout、权重衰减、梯度裁剪、梯度累积、残差幅度或 checkpoint 规则。

## Loss 合同

给定预测 `prediction[N,4]`、标签 `target[N,4]`、布尔 mask `mask[N,4]` 和长度 N 的 `signal_dates`：

1. 对 mask 为真的标签使用 PyTorch SmoothL1 默认 beta=1；无效标签不得进入 loss。
2. 按 `(signal_date,horizon_column)` 分组。
3. 每个非空组先对股票求均值。
4. 所有非空组再等权求均值。
5. 空 batch、无有效组、非有限有效标签、维度不匹配必须 fail-closed。
6. 单日期、单 horizon 且所有 label 有效时，数值必须与现有 masked SmoothL1 相同。
7. 重复一个日期内的零损失股票不得改变另一个日期组的权重。

## 诊断要求

每个训练 epoch 收据化：

- `loss_group_count_mean/min/max`
- `valid_label_count_mean/min/max`
- `labels_per_loss_group_mean`
- `gradient_norm_mean/std/cv/max`
- `optimizer_step_count`
- 物理 batch size 统计
- order fingerprint（control/candidate 都必须固定为一个）

leaderboard 收据化：

- `grouped_smooth_l1_reduction`
- `loss_identity`
- `median_epoch_gradient_norm_cv`
- `median_labels_per_loss_group`
- `date_order_fingerprint_count`
- best epoch、completed epochs、shape branch effect、父参数漂移和吞吐。

## 真实配对实验

- control：`fixed_once + grouped_smooth_l1 + label_mean`
- candidate：`fixed_once + grouped_smooth_l1 + date_horizon_mean`
- 两者共享同一 `(seed,fold)` 的父 checkpoint、model seed、日期顺序、训练/验证样本、初始化、optimizer 和早停规则。
- 共 30 个 current units。
- 复用冻结 LSTM benchmark，仅用于速度比，不重新调参。
- 最佳 checkpoint 重新生成日期级 RankIC，并按 `(seed,fold,horizon,signal_date)` 一对一配对。
- circular block bootstrap：2,000 draws，bootstrap seed=32；报告全局和每 seed 95% CI。

## 预注册门槛

### 完整性门

- control/candidate 各 15 units，覆盖完整且无重复。
- control 逐 unit 重现 v31 fixed-once control，最大绝对 RankIC 误差 `<=1e-12`。
- 共享父 checkpoint SHA-256；父状态漂移=0；父预测误差=0。
- 两者均为 grouped SmoothL1、date-grouped、fixed_once；只允许 reduction 不同。
- control order fingerprint count=1，candidate count=1。
- candidate 的 loss group、valid label 和 labels/group 诊断全部有限且大于零。
- sealed test 未访问。

### 机制门

- 最小不平衡 fixture 必须证明 label-mean=`0.375`、date-horizon-mean=`0.75`。
- 真实 candidate batch 的 median labels/group 必须大于 1，证明等权 reduction 确实作用于多股票组。
- candidate 的 loss identity 必须为 `date-horizon-equal-smooth-l1`，control 必须为 `date-grouped-smooth-l1`。
- 梯度 CV 仅报告，不设方向门：等组权重可能增加小组噪声，真实效果由 RankIC 和 bootstrap 裁决。

### 效果门

- 15-unit paired mean RankIC delta `>=+0.00015`。
- 每个 seed mean delta `>=0`。
- 至少 12/15 units 的 delta `>=-0.00010`。
- candidate mean RankIC `>=0.0999409483`（v31 control + 0.00015）。
- 全局日期块 bootstrap 95% CI low `>0`。
- seed 27 bootstrap CI low `>=0`。
- candidate best_epoch>0 的 trained-effect units 不少于 control。

### 速度门

- model-step TCN/LSTM `>=3.0x`。
- end-to-end TCN/LSTM `>=3.0x`。
- candidate/control median samples/s `>=0.90x`。

### 决策状态

- `stop_date_horizon_equal_integrity_v32`
- `stop_date_horizon_equal_mechanism_v32`
- `stop_date_horizon_equal_no_gain_v32`
- `stop_date_horizon_equal_speed_v32`
- `date_horizon_equal_smooth_l1_confirmed_v32`

任何状态下 `sealed_test_authorized=false`。只有全部门通过，才允许把 `date_horizon_mean` 作为下一轮 TCN grouped objective 的默认候选；否则继续保留 `label_mean`。

## 必需产物

- `config.resolved.json`
- `tcn-epoch-history.parquet`
- `tcn-leaderboard.parquet`
- `loss-reduction-diagnostics.parquet`
- `daily-rankic-long.parquet`
- `paired-rankic-by-date.parquet`
- `bootstrap-summary.parquet`
- `paired-unit-comparison.parquet`
- `shape-diagnostics.parquet`
- `parent-checkpoint-manifest.parquet`
- `seed-summary.parquet`
- `lstm-measurements.parquet`
- `comparison.json`
- `selection.json`
- `report.md`
- `receipt.json`
- `checkpoints/*.pt`

## 执行命令

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tcn_date_horizon_equal_smooth_l1.py -q
.venv\Scripts\python.exe tasks/run_tcn_date_horizon_equal_smooth_l1.py --run-dir artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be --split-manifest artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet --config config/pandadata-tcn-date-horizon-equal-smooth-l1-v32.example.json --output-dir artifacts/tcn-date-horizon-equal-smooth-l1-v32
.venv\Scripts\python.exe tasks/preflight.py
.venv\Scripts\python.exe tasks/test.py
.venv\Scripts\python.exe -m mypy src tests tasks
.venv\Scripts\python.exe -m ruff check src tests tasks
.venv\Scripts\python.exe -m build --no-isolation
```

先让最小不平衡 fixture 变绿，再跑真实实验。不得根据真实 validation 结果临时修改门槛，不得在 v32 内追加第二个变量。
