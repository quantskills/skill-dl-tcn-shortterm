# TCN 受约束 Top-tail checkpoint selection v35：完整执行提示词

你正在维护 `skill-dl-tcn-shortterm`。本轮必须继续优化 TCN，不得更换成 LSTM、Transformer、集成模型，不得解冻父模型，不得扩 universe，也不得访问 sealed test。LSTM 只作为固定 benchmark 重放。

## 已知事实

- v34 在真实 2021–2025 股票分钟线普通 validation 上证明 Top-tail loss 已真实生效：每个 batch 约 826 pairs，SmoothL1 与 Top-tail prediction-space 梯度 cosine 中位数 `+0.5142`，父模型 drift `0`，不存在梯度冲突复发。
- v34 TCN/LSTM model-step 与端到端速度为 `6.23x/5.76x`，TCN 总体速度目标仍满足；Top-tail objective 和逐 batch 诊断使其比 grouped SmoothL1 control 慢约 26%。
- v34 candidate 相对 control 的 Top precision 仅 `+0.000208`、Top return `+0.000056`，置信区间均跨零；NDCG `-0.000027`、RankIC `-0.000610`、turnover `+0.003059`。
- v34 的 15 个 candidate 单元中，7 个选择 epoch 0、8 个选择 epoch 1，没有任何单元选择更晚 epoch。checkpoint 仍按 RankIC 选择，可能在 Top-tail 信号充分形成前回退到父模型或首轮。
- 该现象只能形成“checkpoint selection 可能抑制 Top-tail”的假设，不能事后改 v34 的判定。

## 单一假设

保持 v34 Top-tail 训练 objective 完全不变，固定跑满 8 epochs 后，如果从同一条训练轨迹中用“RankIC 不可明显退化约束下的 Top precision/NDCG 平均分”选择 checkpoint，可能得到比纯 RankIC selection 更好的 top-tail 决策效果。

## 唯一允许变量

本轮 control 与 candidate 必须使用同一条、只训练一次的 TCN Top-tail epoch trajectory。两者共享每个 epoch 的完全相同 checkpoint hash；唯一变量是从 epoch `0..8` 中选择哪个 checkpoint。

- control selection：选择 mean daily RankIC 最大的 epoch；严格并列时选择更早 epoch。
- candidate selection：先计算该 seed/fold 轨迹的最大 RankIC；只允许 `RankIC >= max RankIC - 0.002` 的 epoch；在可行 epoch 中最大化 `tail_selection_score = 0.5 × mean_top_precision + 0.5 × mean_ndcg_at_top`；严格并列时依次选择更高 RankIC、更低 mean_top_turnover、更早 epoch。

不得同时调整 Top-tail weight、fraction、temperature、模型容量、学习率、数据顺序、early-stopping patience 或选择公式。不得根据运行结果改变 `0.002` 容忍度或 `0.5/0.5` 权重。

## 固定训练协议

- 数据及 SHA-256 与 v34 完全相同，只使用 train/validation/purged；任何 sealed 行立即 fail closed。
- seeds `7,17,27`；folds `0..4`。
- TCN：`dynamic_horizon_skip`、chomp causal path、channels `16`、kernel `3`、dilations `1,2,4,8,16,32,64,128`、dropout `0`。
- 加载与 v34 相同的 raw frozen parent checkpoints；shape residual scale `0.25`；仅 88 个 trainable 参数；父模型 state/prediction drift 必须为 `0`。
- objective 固定为 v34：`SmoothL1 + 0.05 × Top-tail pairwise logistic`，真实 top fraction `0.10`、temperature `0.10`、date order `fixed_once`。
- optimizer/lr/batch：Adam、`0.003`、`128`；float32；torch threads `8`；workers `0`。
- 固定训练 epoch `1..8`，另含 epoch 0 frozen parent；不得 early stop。每个 epoch checkpoint、验证分数与 hash 全量保存。
- 每个 batch 继续记录 Top-tail pair 数、分量梯度 cosine、总梯度和训练吞吐，保持与 v34 机制证据可比。

## 每 epoch 普通验证

每个 seed/fold/epoch 都必须输出与 v33/v34 完全相同的逐样本 score/target 契约，并计算：RankIC、Pearson IC、top10% raw/excess return、long-short spread、top precision、NDCG@top、quantile monotonicity、top turnover。

selection 只能读取当前 fold 的 ordinary validation 指标，不能读取 test/sealed、未来 fold 或 LSTM 结果。LSTM 只在 control/candidate checkpoint 选择完成后重放作上下文比较。

## 预注册门槛

完整性/机制门：

- 15 条 trajectory 均完整覆盖 epoch `0..8`，共 135 个唯一 epoch checkpoints。
- control/candidate 对每个 epoch 共享完全相同 checkpoint hash，证明没有重复训练或轨迹漂移。
- candidate 所选 epoch 全部满足 `RankIC >= unit max RankIC - 0.002`，选择结果可由公开公式精确重放。
- 至少 1/15 单元的 candidate epoch 与 control epoch 不同；否则判定为没有 selection opportunity。
- loss identity 与 `0.05/0.10/0.10` 参数正确；父模型 drift/prediction error `0`；梯度 cosine 有限；无 sealed 访问。

预测效果门（candidate minus control）：

- mean Top precision delta `> 0`。
- mean NDCG@top delta `> 0`。
- 两者至少一个 paired block-bootstrap 95% CI low `> 0`，另一个 CI low `>= -0.002`。
- mean RankIC delta `>= -0.002`。
- Top return CI low `>= -0.0005`。
- mean Top turnover delta `<= +0.02`。

速度门：

- 固定 8-epoch Top-tail TCN 相对历史 8-epoch LSTM 的 model-step 和端到端速度均 `>= 3.0x`。
- checkpoint evaluation/selection 的额外耗时必须单列，不得计入模型训练吞吐，也不得把删除研究诊断后的数字冒充模型优化。

只有完整性、机制、效果和速度全部通过，candidate 才能成为 ordinary-validation 候选；仍不得打开 sealed test。若失败，必须保留 TCN 路线并准确区分：没有 selection opportunity、Top-tail checkpoint 效果不稳、RankIC 约束失败或速度失败。

## 必需产物

- `trajectory-epoch-history.parquet`
- `trajectory-epoch-metrics.parquet`
- `trajectory-checkpoint-manifest.parquet`
- `checkpoint-selection.parquet`
- `selected-predictions.parquet`
- `selected-task-aligned-metrics.parquet`
- `selected-model-summary.parquet`
- `control-candidate-comparison.json`
- `candidate-lstm-context-comparison.json`
- `bootstrap-summary.parquet`
- `gradient-conflict-diagnostics.parquet`
- `parent-checkpoint-manifest.parquet`
- `selection.json`
- `config.resolved.json`
- `report.md`
- 135 个 epoch checkpoints、全部输出 SHA-256 及不可变 receipt；receipt 必须声明 `sealed_test_accessed: false`。

## 执行顺序

先为 selection 规则、epoch 完整性和固定全轨迹捕获编写失败测试；实现最小纯函数和训练轨迹接口；运行聚焦测试、Ruff、Mypy；再执行真实五年 3-seed × 5-fold × 9-epoch 普通验证。运行后逐项校验 artifact hash，更新研究结果与 work item，最后执行完整 pytest、Ruff、Mypy、preflight、统一测试入口和 production build。不得手写、推测或选择性报告实验数字。
