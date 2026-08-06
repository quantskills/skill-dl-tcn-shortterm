# TCN v43：梯度归一化的全横截面共识排序蒸馏提示词

你正在维护 `skill-dl-tcn-shortterm`。v42 已证明三种子 TCN 共识可以被单 student 大量吸收：相对原始 TCN，3 seeds × 5 folds 的 RankIC `+0.014508`、15/15 seed-fold 单元和 4/4 horizons 为正，六项指标改善 5 项，单模型速度约为 LSTM 的 `4.88×`。但点式目标混合使 Top precision 下降 `-0.003344`，低于预注册容忍线 `-0.002`，因此 v42 必须保持停止状态，不能事后改门槛或搜索 teacher weight。

v43 不追逐 Top precision、top-k、换手率或方向标签。它检验一个全局机制假设：`0.75 × true_rank + 0.25 × teacher_rank` 的点式回归只能学习每个样本的中心位置，不能直接保存 teacher 在完整 `(signal_date, horizon)` 横截面上的相对排序。v43 保持真实 rank target 不变，用全横截面 teacher 排序损失代替点式 teacher target 混合。

## 冻结项

- 数据、top50 PIT 股票池、relative10 特征、2021–2025 ordinary validation、fold、purge/embargo、1/2/3/5 日目标全部冻结。
- TCN 架构继续使用 16 channels、dilations `1..128`、causal chomp padding、WeightNorm、batch 128、float32、8 epochs、Adam `0.003`、无 EMA。
- teacher 继续使用 v42 已审计的 TCN seeds `{7,17,27}`、5 folds 及 train-only teacher consensus ranks；不得重新训练、替换或选择 teacher。
- 不读取 validation teacher scores、validation labels 作为训练输入，不读取 sealed/test。
- checkpoint 仍只按 ordinary-validation mean daily RankIC 选择；不使用 Top precision 选 epoch。

## 唯一变量：teacher ordering 的梯度归一化 listwise loss

control 是 v42 已冻结的点式 student。candidate 从头训练相同单 TCN，但不改写真实目标：

1. `L_true = SmoothL1(student_score, true_rank_target)`。
2. 对训练 batch 内每个完整 `(signal_date, horizon)`，用 temperature `0.1` 的 pairwise sigmoid soft-rank 得到 student differentiable rank。
3. `L_teacher = mean(1 - corr(student_soft_rank, teacher_consensus_rank))`，所有有效横截面等权；不截 top-k，不使用收益符号。
4. 分别计算 `g_true = dL_true/dscore`、`g_teacher = dL_teacher/dscore`。
5. 使用无梯度的尺度 `s = ||g_true|| / max(||g_teacher||, 1e-12)`，最终：

   `L = L_true + 0.25 * stop_gradient(s) * L_teacher`

这样 teacher 分量在 prediction-space 的梯度范数固定为 true 分量的 25%，避免先前 soft-RankIC 因损失量纲主导 SmoothL1。`0.25` 和 temperature `0.1` 均固定，不搜索。

listwise candidate 必须使用 date-grouped batch sampler，确保同一日期横截面不被拆散；这是该 objective 的组成部分，不得把 batch-order 收益单独宣称为 teacher loss 收益。保存每 batch 的 group count、有效标签数、真实/teacher 梯度范数比和梯度 cosine。

## 明确不做

- 不改 teacher weight、temperature、channels、learning rate、epoch、checkpoint selection 或 top fraction。
- 不加入 Top-tail loss、方向分类、收益符号标签、换手率 loss、组合规则或 LSTM teacher。
- 不把 v42 失败重新判为通过；v43 是新的前瞻性协议。
- 不使用三模型在线 ensemble；candidate 必须为一个标准 TCN state dict、一次前向。
- 不使用既有 sealed v36 结果选择任何参数，也不打开新的 sealed 数据。

## 完整性门

- teacher targets 的 SHA-256、表 fingerprint、teacher seeds/folds、train positions、日期、期限和 masks 必须与 v42 Phase A receipt 完全一致。
- teacher target 在 validation positions 必须为 NaN/不可访问；每 fold 训练覆盖必须严格等于 train positions。
- candidate 的真实训练 target 必须逐单元等于原始 rank target；不得再使用 v42 blended target。
- listwise group 不得跨日期或期限；每组至少两个有效标的。
- 每个 batch 的 teacher/true prediction-gradient ratio 应为 `0.25 ± 0.05`；非有限梯度、空分组或 validation teacher 访问立即失败。
- candidate/control 参数量、结构和推理次数一致；`sealed_test_accessed=false`。

## Phase A：seed 7 × 5 folds

候选首先相对原始 v42 control 通过非补偿全局门：

- mean RankIC delta `>=+0.002`；至少 3/5 folds 为正；RankIC bootstrap 95% CI low `>=-0.002`；
- RankIC、Pearson IC、Top return、Top precision、NDCG、quantile monotonicity 至少 4/6 改善；
- Top return `>=-0.0001`、Top precision `>=-0.002`、NDCG `>=-0.001`、quantile monotonicity `>=-0.002`；
- 至少 3/4 horizons RankIC 为正，最差 horizon `>=-0.003`。

同时相对 v42 点式 student 做 Pareto 保护：

- 六项至少 3 项严格改善；
- RankIC/Pearson IC 均 `>=-0.002`，Top return `>=-0.0001`、Top precision `>=-0.001`、NDCG `>=-0.001`、quantile monotonicity `>=-0.002`；
- candidate 对冻结 teacher validation ensemble 的平均 score RankIC 至少比点式 student 高 `+0.002`，只作为机制门，不覆盖任务指标门。

速度门：相对点式 student model-step 与 complete-cycle retention 均 `>=0.70`，折算 TCN/LSTM model-step ratio `>=3.0×`，推理前向次数为 1。

全部通过状态为 `listwise_consensus_seed7_holistic_admitted_v43`，才授权 Phase B；否则状态为 `stop_listwise_consensus_seed7_v43`，不得调 ratio 或 temperature。

## Phase B：seeds 17/27

仅 Phase A 通过才用完全相同协议训练 seeds 17/27。合并 3 seeds × 5 folds 后继续要求：

- 相对原始 control mean RankIC delta `>=+0.002`，15 单元至少 9 个为正，每 seed mean `>=-0.001`，CI low `>=-0.001`；
- 六项至少 4 项改善，四项下行容忍不变；3/4 horizons 为正，最差 `>=-0.002`；
- 相对 v42 点式 student 六项至少 3 项改善且继续满足 Pareto 下行容忍；
- teacher fidelity、梯度比例、速度和单模型推理门继续通过。

通过状态为 `listwise_consensus_multiseed_admitted_v43`；否则停止。

## 输出与验收

保存 frozen source manifest、teacher target 审计、loss/gradient 审计、训练历史、单模型 checkpoints、predictions、六项任务指标、相对 original control 与 pointwise student 的成对比较、fold/horizon/seed deltas、bootstrap、teacher fidelity、速度、model gate、receipt 和报告。runtime teacher data 与 checkpoints 不提交 Git。

实现 objective 数学、日期/期限隔离、teacher target 泄漏防护、梯度比例、配置解析和全局门测试。最后运行 Ruff、Mypy、完整 pytest、preflight 与 wheel/sdist build。

结论上限始终为 ordinary-validation single-TCN research evidence；不得宣称 Alpha-ready、sealed 通过、可以部署或 TCN 普遍优于 LSTM。
