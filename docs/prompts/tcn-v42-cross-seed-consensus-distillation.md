# TCN v42：跨种子共识蒸馏为单模型的全局优化提示词

你正在维护 `skill-dl-tcn-shortterm`。v41 已严格证伪“单轨迹参数抖动可由 EMA 修复”：EMA 机制与45个 raw epoch state 完全一致、速度保持98.33%，但六项预测指标只改善 Top return，RankIC/NDCG/单调性下降，未授权多种子。因此不得继续调 EMA decay。

v40 的三个不同初始化 TCN prediction ensemble 相对单模型同时改善 RankIC、Top return、Top precision、NDCG 和 quantile monotonicity，且几乎追平 LSTM。这说明有效信号来自 **不同参数盆地的横截面共识**，不是同一轨迹的时间平滑。v42 的唯一目标是把已冻结的三种子 TCN 教师共识蒸馏到一个推理时只需一次前向的 TCN student。

## 冻结项

- 继续使用 v40 top50 relative10、2021–2025 ordinary validation、1/2/3/5日横截面排序。
- student 架构、16 channels、dilations、因果 padding、WeightNorm、batch 128、float32、8 epochs、Adam 0.003、Smooth L1、fold 和日期顺序全部不变。
- teacher 只能使用 v40 已冻结的 relative10 TCN seeds 7/17/27 对应 fold checkpoints。
- 不使用 LSTM 作为教师；项目仍是 TCN-only 推理模型。
- 不读取 test/sealed，不允许 validation 标签或 validation teacher score 进入训练。

## 唯一变量：训练目标的固定共识混合

对每个 fold：

1. 只在该 fold 的 train positions 上加载 seeds 7/17/27 三个冻结 TCN checkpoint 做预测。
2. 对三模型 raw score 等权平均。
3. 在每个 `(signal_date, horizon)` 的训练横截面内，把 teacher ensemble score 转为 `[-1,1]` 百分位秩；不跨日期、不跨期限归一化。
4. student control 使用原始 rank target。
5. student candidate 使用固定目标：

   `distilled_target = 0.75 * true_rank_target + 0.25 * teacher_consensus_rank`

6. `teacher_weight=0.25` 本轮固定且不得搜索；候选继续使用相同 Smooth L1，不增加额外 forward、loss head 或可训练参数。

训练目标 override 必须按 fold 隔离。一个样本若在早期 fold 是 validation、在后期 fold 是 train，只能在后期 fold 作为训练 override；禁止用一张跨 fold 改写的 labels 表造成 validation 污染。

## 明确不做

- 不改架构、特征、损失类型、学习率、epoch、checkpoint selection、组合策略或换手率。
- 不继续 EMA；不同时加入 temporal pooling、PCGrad、top-tail loss 或 top100。
- 不用 validation ensemble score 训练 student。
- 不把三模型 ensemble 当最终产物；candidate 必须是一个标准 TCN state dict，一次前向。
- 不要求击败 LSTM 才晋级；LSTM只做冻结描述性基准，防止为单一排名无界调参。

## 完整性门

- teacher seeds 必须恰为 `{7,17,27}`，fold 必须和 student fold 相同。
- 每个 teacher checkpoint、数据、split、标签和父 receipt 保存 SHA-256。
- teacher 预测位置必须与该 fold train positions 严格相等，且与 validation positions 交集为0。
- teacher consensus target必须有限、位于`[-1,1]`，逐 `(date,horizon)` 计算。
- control/candidate 除目标 override 和 training contract ID 外完全一致。
- candidate 参数量、模型结构、推理前向次数与 control 相同。
- `sealed_test_accessed=false`。

## Phase A：seed 7 × 5 folds

### 非补偿式全局模型门

candidate 减 control：

- mean RankIC `>= +0.002`；5 folds至少3个为正；RankIC block-bootstrap 95% CI下界 `>= -0.002`；
- RankIC、Pearson IC、Top return、Top precision、NDCG、quantile monotonicity六项至少4项严格改善；
- Top return `>= -0.0001`、Top precision `>= -0.002`、NDCG `>= -0.001`、quantile monotonicity `>= -0.002`；
- 4 horizons至少3个RankIC为正，最差horizon `>= -0.003`；
- candidate/control model-step throughput retention `>=0.95`，complete-cycle retention `>=0.90`；
- 折算 v40 LSTM 后的 student/LSTM model-step ratio `>=3.0×`。

另报告 candidate 相对三种子 teacher validation ensemble 的差距回收比例，但它只是机制诊断，不允许覆盖上述非补偿门。

全部通过状态为 `consensus_student_seed7_holistic_admitted_v42`，才授权 Phase B。失败则停止共识蒸馏，不调 teacher weight。

## Phase B：student seeds 17/27

只在 Phase A 通过后，冻结相同 teacher checkpoints、`teacher_weight=0.25` 和全部协议。合并为3 seeds×5 folds：

- mean RankIC delta `>=+0.002`，15单元至少9个为正；每seed平均delta `>=-0.001`；
- 六项指标至少4项严格改善，继续满足四项下行容忍；
- 4 horizons至少3个为正，最差 `>=-0.002`；
- RankIC CI下界 `>=-0.001`；
- 速度保持和单模型推理门继续通过。

通过状态为 `consensus_student_multiseed_admitted_v42`；否则停止。

## 输出和验收

保存 teacher checkpoint manifest、train-only teacher predictions/targets审计、target fingerprint、control/candidate训练历史、单模型checkpoints、validation predictions、六项指标、fold/horizon deltas、bootstrap、速度、model gate、receipt和报告。真实 teacher train scores 属于runtime artifact，不提交Git。

实现 fold-scoped training target override、泄漏防护、目标混合数学、teacher覆盖和单模型checkpoint测试。最后运行 Ruff、Mypy、完整pytest、preflight和wheel/sdist build。

结论上限始终是 ordinary-validation single-TCN stability evidence；不得宣称 Alpha-ready、sealed 通过、TCN 普遍优于 LSTM 或可部署。
