# TCN Top-tail 任务对齐优化 v34：完整执行提示词

你正在维护 `skill-dl-tcn-shortterm`。本轮只能优化 TCN，不得以 LSTM、Transformer、集成模型或数据泄漏替代 TCN。目标是在已经满足速度门、父模型冻结门和普通验证完整性门的前提下，修复 TCN 在真实五年股票分钟线普通验证集上的前 10% 选股精度与 NDCG 落后问题。

## 已知事实

- 数据是本地 PandaData 补齐后的近五年股票分钟线运行时，使用 `memmap` 特征窗；不得重新取数、扩 universe 或改变样本定义。
- v32 证据显示 TCN/LSTM 的 model-step 约为 `6.53x`，端到端约为 `5.90x`；本轮必须保留至少 `3x` 的既有门槛。
- v33 在完全相同预测契约下得到：TCN RankIC `0.099791`，LSTM `0.115545`；TCN top return 略高但置信区间跨零，top precision 与 NDCG 明显落后，turnover 更高。
- v32/v33 已证明当前问题不是感受野、因果 padding、BatchNorm 或大数据加载器错误：架构使用 chomp 因果 TCN、weight-normalized 路径、足够感受野和 memmap 数据。
- 当前 TCN 候选是冻结原始父模型后的 shape-residual 分支，只有 88 个可训练参数；不得解冻父模型。
- sealed test 未授权，本轮只能使用 train/validation/purged，任何 sealed 行都必须 fail closed。

## 单一假设

当前训练目标主要优化全截面逐标签 SmoothL1，与实际只使用前 10% 股票的决策不完全一致。将候选损失改为 `SmoothL1 + 0.05 × Top-tail pairwise logistic`，可能改善真实 top precision / NDCG，而不破坏 RankIC、速度和冻结父模型。

## 唯一允许变量

- control：`grouped_smooth_l1`，逐有效标签均值。
- candidate：`top_tail`。
- candidate 的 Top-tail 定义必须固定为：每个 `(signal_date, horizon)` 中按真实 `rank_target` 取 `ceil(N × 0.10)`，至少 1 个；把每个真实 top 成员与每个非 top 成员配对；损失为 `softplus(-(score_top - score_non_top) / 0.10)`；先在组内对 pair 求均值，再跨日期/期限组求均值；总损失为 SmoothL1 加 `0.05` 倍 Top-tail 损失。

除上述 objective 外，control 与 candidate 的数据 SHA-256、父 checkpoints、模型类、channels、kernel、dilations、dropout、学习率、batch size、优化器、date order、seeds、folds、epoch、patience、checkpoint selection、torch threads 和 precision 必须完全相同。

## 固定训练协议

- seeds：`7, 17, 27`。
- folds：`0..4`，共 15 个配对单元。
- 架构：`dynamic_horizon_skip` + chomp causal path + frozen raw parent + shape residual scale `0.25`。
- channels `16`；kernel `3`；dilations `1,2,4,8,16,32,64,128`；dropout `0`。
- learning rate `0.003`；batch size `128`；date order `fixed_once`；float32；CPU threads `8`；workers `0`。
- max epochs `8`；patience `2`；patience min delta `0.0005`；checkpoint min delta `0.0`。
- checkpoint 仍按普通验证 mean daily RankIC 选择。这一轮不得同时改变 checkpoint metric；若 Top-tail 改进被 RankIC checkpoint selection 抑制，作为下一轮独立未知量处理。

## 梯度与机制诊断

- 对 candidate 每个训练 batch，用 prediction-space 梯度分别计算 SmoothL1 与 Top-tail 分量的 cosine；只记录，不得本轮再加入 PCGrad、梯度投影或动态权重。
- 必须记录有效 `(date,horizon)` 组数、pair 数、有效标签数、每 epoch 梯度 cosine 中位数、总梯度 norm/CV、训练吞吐、冻结父模型 drift。
- 若全部 batch 无有效 Top-tail pair、父模型 drift 非零、参数不符或任何 sealed 数据被访问，实验立即失败。

## 普通验证评测

必须输出逐样本分数，使用与 v33 相同的 prediction/target/evaluation contracts，同时评价 control TCN、candidate TCN；并重放 v33 LSTM 结果作为上下文 benchmark，不重新调 LSTM。

至少报告：RankIC、Pearson IC、top 10% raw return、top excess return、long-short spread、top precision、NDCG@top、quantile monotonicity、top turnover，以及按日期块、seed/fold/horizon 分层的配对 bootstrap 95% CI。

## 预注册门槛

完整性和机制门：

- 15 个 control/candidate 配对单元完整，预测样本逐键一致，数据与父 checkpoint 哈希一致。
- candidate loss identity、`0.05/0.10/0.10` 参数和 Top-tail pair 均真实生效。
- 88 个 trainable 参数，父模型 state drift `0`，parent prediction max abs error `0`。
- candidate 的 prediction-space 分量梯度 cosine 中位数不得低于 `-0.25`。

效果门（candidate 相对 control）：

- mean top precision delta `> 0`。
- mean NDCG@top delta `> 0`。
- 两者至少一个的配对 block-bootstrap 95% CI low `> 0`，另一个 CI low 不得低于 `-0.002`。
- mean RankIC delta 不得低于 `-0.002`。
- top return CI low 不得低于 `-0.0005`。
- top turnover delta 不得高于 `+0.02`。

速度门：

- candidate/control 训练吞吐比不得低于 `0.85`。
- 继承并核验的 TCN/LSTM model-step 与端到端比均不得低于 `3.0x`。

只有完整性、机制、效果和速度四门全部通过，candidate 才可标记为普通验证候选；仍不得打开 sealed test。若未通过，准确报告哪一门失败并保留 TCN 路线，不得宣称 TCN 无效或改用 LSTM。

## 必需产物

- `tcn-epoch-history.parquet`
- `tcn-leaderboard.parquet`
- `predictions.parquet`
- `task-aligned-metrics.parquet`
- `task-aligned-summary.parquet`
- `control-candidate-comparison.json`
- `candidate-lstm-context-comparison.json`
- `bootstrap-summary.parquet`
- `gradient-conflict-diagnostics.parquet`
- `parent-checkpoint-manifest.parquet`
- `selection.json`
- `config.resolved.json`
- `report.md`
- 全部 best checkpoints、SHA-256 outputs 清单及不可变 receipt；receipt 必须声明 `sealed_test_accessed: false`。

## 执行顺序

先写会失败的 loss/parser/contract 测试；实现最小代码使其通过；运行目标测试、完整 pytest、Ruff、Mypy、预检；再运行真实五年 3-seed × 5-fold 普通验证。根据真实 artifact 更新研究结果和 work item，不得手写或推测实验数字。
