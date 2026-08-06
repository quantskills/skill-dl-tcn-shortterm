# TCN 任务对齐多指标评测 v33：完整实施提示词

你是 `skill-dl-tcn-shortterm` 项目的实现与验证代理。请在不访问 sealed test、不改动模型结构、不改动训练损失、不连接券商、不部署、不写外部系统的前提下，完成一次真实五年分钟线普通验证集上的 TCN/LSTM 任务对齐诊断。

## 当前已知事实

1. 项目主任务是沪深 A 股的多标的横截面收益排序。输入截至信号日收盘，输出未来 1、2、3、5 个交易日的四个连续评分，执行时点是下一交易日开盘。
2. 当前 v32 TCN 和 LSTM 的模型输出语义相同：都是形状为 `[batch, 4]` 的横截面评分，并用同一 `next-open-rank-v2` 标签训练和评估。二者不是预测不同任务。
3. 二者的训练协议并不相同：TCN 是已有父模型上的冻结父干形状残差续训；LSTM 是从头固定 8 epoch 训练。当前实验只能比较这两个具体训练产物，不能证明架构的一般优劣。
4. v32 的 TCN 相对 LSTM 已观察到约 `6.53x` model-step 和 `5.90x` end-to-end 速度比，但验证 RankIC 为约 `0.09979`，低于 LSTM 的约 `0.11555`。
5. RankIC 与本项目的“全股票池横截面排序”主任务相符，但项目实际经济决策是 Top10% 多头组合。只用 RankIC 不能判定 Top-tail 选股、原始收益、换手和最终组合表现。
6. Bai et al. 2018 的结论来自通用序列基准，不是“TCN 在金融预测上必然优于 LSTM”的定理。Microsoft Qlib 的公开同协议结果中存在 LSTM 同时优于 TCN 的 RankIC 与组合收益反例。

## 要回答的问题

1. v32 的 TCN 与固定 LSTM 基准在完全相同样本、标签和评分契约下，RankIC 差距是否可以复现？
2. RankIC 的模型排序是否与 Top10% 多头原始收益、Top10% 超额收益、Top-tail 命中率、NDCG、分位单调性和换手诊断一致？
3. 如果模型赢家发生反转，问题是“评测目标与决策目标不对齐”；如果 LSTM 在全排序与 Top-tail 指标上都获胜，问题更可能仍在 TCN 表征/训练协议；如果 TCN 在 Top-tail 获胜而 RankIC 落后，则下一轮应优化 Top-tail 对齐目标，而不是继续追全池 RankIC。

## 可证伪假设（按优先级）

1. `H1`：两模型输出和目标语义一致，但历史比较没有把契约 ID 与逐样本覆盖显式固化，存在误比较风险。
2. `H2`：RankIC 对主排序任务合理，但不足以代表 Top10% 多头决策；全池 RankIC 与 Top-tail 指标可能给出相反赢家。
3. `H3`：TCN 的实际瓶颈不是速度或梯度，而是有用表征向顶部股票选择的传递不足。
4. `H4`：两模型都在同一普通验证集上选择 checkpoint 并报告效果，存在选择偏差；本轮只诊断，不把结果升级为 sealed-test 或 Alpha 结论。
5. `H5`：“TCN 预测效果一定优于 LSTM”为假；架构优劣依赖数据、任务、预算和训练协议。

## 本轮唯一干预

只增加“统一逐样本预测契约 + 多指标评测契约”。不得同时修改 TCN/LSTM 结构、损失函数、优化器、学习率、batch size、epoch、数据窗口、标签或 split。这样任何结论都只能归因于评测视角的改变，而不是模型变更。

## 数据与父产物

- 真实数据：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be`
- 普通验证 split：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- TCN 父产物：`artifacts/tcn-date-horizon-equal-smooth-l1-v32`
- TCN 使用 v32 的 `grouped-label-mean` control checkpoint，而不是已经失败的 date-horizon-equal candidate。
- seeds 必须为 `7/17/27`，folds 必须为 `0..4`；只允许 `train/validation/purged`，任何 `sealed=True` 或 test stage 必须 fail closed。
- 所有输入必须核对 SHA-256；输出目录存在时拒绝覆盖。

## 模型重放协议

### TCN

- 从 v32 的 15 个 `grouped-label-mean` 最佳 checkpoint 只读重放。
- 架构、归一化、fold、seed、validation positions 与 v32 保持一致。
- `training_contract_id = frozen-parent-shape-residual-continuation-v32`。

### LSTM

- `hidden_size=34`、`Adam(lr=0.003)`、`batch_size=128`、`epochs=8`、float32、CPU、8 torch threads、num_workers=0。
- 每个 `(seed, fold)` 使用 `model_seed = seed + fold * 100`，从头训练。
- 每 epoch 在同一普通 validation 上算 mean daily RankIC，保存最佳 checkpoint；重放最佳 checkpoint 生成逐样本分数。
- 最佳 RankIC 必须与 v32 `lstm-measurements.parquet` 在可复现容差内吻合，否则停止并标记历史基准重放失败。
- `training_contract_id = from-scratch-masked-smooth-l1-8epoch-v32`。

## 统一逐样本预测契约

每个模型必须输出以下列：

`model, seed, fold, sample_id, instrument_id, signal_date, horizon, score, rank_target, raw_return, stage, sealed, prediction_contract_id, target_contract_id, evaluation_contract_id, training_contract_id`

固定 ID：

- `prediction_contract_id = cross-sectional-score-v1`
- `target_contract_id = next-open-rank-v2`
- `evaluation_contract_id = ordinary-validation-multimetric-v33`
- `stage = validation`
- `sealed = false`

强制校验：

1. 两模型在 `(seed, fold, sample_id, horizon)` 上完全一一对应。
2. `instrument_id/signal_date/rank_target/raw_return` 完全一致。
3. 不允许缺样本、重复键、非有限值、不同 target/evaluation contract。
4. 允许两个模型的训练协议不同，但必须显式记录，且禁止据此宣称架构公平对比已完成。

## 评测指标

先在每个 `(model, seed, fold, signal_date, horizon)` 横截面计算，再聚合：

1. `RankIC`：score 与 rank_target 的 Spearman 相关，衡量全池相对排序。
2. `Pearson IC`：score 与 raw_return 的 Pearson 相关，补充评分幅度信息。
3. `RankIC std / RankICIR`：衡量跨日期稳定性。
4. `Top10% mean raw return`：预测 Top10% 的下一开盘口径原始持有期收益。
5. `Top10% excess return`：Top10% 收益减当日横截面均值。
6. `Top-bottom spread`：仅诊断，不当作可执行 A 股多空组合。
7. `Top10% precision`：预测 Top10% 与实际收益 Top10% 的交集比例。
8. `NDCG@Top10%`：顶部排序质量。
9. `quantile monotonicity`：十分位收益单调性。
10. `Top10% membership turnover`：相邻信号日 Top 集合变化，属于未计成本诊断。

对 `RankIC/Top return/Top excess/Top precision/NDCG` 做按 `(seed,fold,horizon)` 保留时间顺序的配对日期块 bootstrap，输出 95% CI。

## 判定规则

- `task_aligned_metrics_agree_tcn_v33`：TCN 在 RankIC、Top return、Top precision 三个主诊断上均高于 LSTM。
- `task_aligned_metrics_agree_lstm_v33`：TCN 在三个主诊断上均低于 LSTM。
- `task_aligned_metrics_mixed_v33`：三者方向不一致；禁止用单一 RankIC 宣称总赢家。
- `stop_v33_contract_mismatch`：输出/标签/样本/阶段契约任一不一致。
- `stop_v33_lstm_replay_mismatch`：历史 LSTM RankIC 无法复现。

无论结果如何：不得访问 sealed test，不得扩大 Top50，不得部署，不得宣布 TCN 必然优于 LSTM，不得宣布已获得可交易 Alpha。

## 必须生成的产物

在不可覆盖的新目录中生成：

- `predictions.parquet`
- `daily-metrics.parquet`
- `model-summary.parquet`
- `paired-bootstrap.parquet`
- `lstm-epoch-history.parquet`
- `lstm-checkpoint-summary.parquet`
- `contract-audit.json`
- `comparison.json`
- `selection.json`
- `config.resolved.json`
- `receipt.json`
- `report.md`

receipt 必须记录输入哈希、父 receipt、代码 identity、环境、输出哈希、`sealed_test_accessed=false`。

## 验证顺序

1. 先写失败测试，证明旧比较会容忍契约漂移，并构造“全池 RankIC 更高但 Top-tail 更差”的合成反例。
2. 实现 fail-closed 契约验证和多指标 evaluator，使测试转绿。
3. 用真实五年分钟数据运行完整 3 seeds × 5 folds 重放。
4. 检查历史 LSTM RankIC 重放误差、两模型逐样本覆盖、输出契约和 bootstrap。
5. 运行 focused pytest、`python -m mypy`、`python -m ruff check .`、`python tasks/preflight.py`、`python tasks/test.py`、`python -m build`。
6. 最终报告必须把“观察事实、推断、仍未知”分开，并明确下一轮只能根据 v33 的赢家结构选择一个变量：Top-tail 目标、TCN 表征，或公平训练协议。
