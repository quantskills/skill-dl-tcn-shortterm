# TCN v41–v42 全局稳定性优化结果

## 目标与预注册提示词

本轮不再围绕换手率、单个 fold、单个 horizon 或单个指标做局部补丁，而是检验两个可跨全部预测任务生效的模型级机制：

1. [v41 单模型 EMA 稳定性](../prompts/tcn-v41-single-model-ema-stability.md)
2. [v42 跨种子共识蒸馏](../prompts/tcn-v42-cross-seed-consensus-distillation.md)

所有门槛在真实运行前冻结。ordinary validation 继续使用 2021–2025、top50 PIT 股票池、relative10 特征、1/2/3/5 日任务；未读取 sealed test。

## 永久红灯诊断

`tasks/diagnose_tcn_v41_variance_gap.py` 把 v40 的跨种子方差缺口变成可重复失败的诊断：三种子 TCN prediction ensemble 相对单个 relative TCN 同时改善：

- RankIC：`+0.010440`
- Top return：`+0.000670`
- Top precision：`+0.000365`
- NDCG：`+0.009944`
- quantile monotonicity：`+0.016131`

这证明可利用的增益来自不同初始化到达的参数盆地共识，而非某个组合换手规则。

## v41：EMA 假设被证伪

固定 `EMA decay=0.99`，不搜索 decay；控制组和候选组的 45 个 raw epoch state 最大误差为 `0.0`，因此机制实现没有改变原始训练轨迹。真实 seed 7 × 5 folds 结果：

- RankIC delta：`-0.000137`
- Top return delta：`+0.000139`
- Top precision delta：`-0.000375`
- NDCG delta：`-0.001584`
- quantile monotonicity delta：`-0.004030`
- 六项仅 `1/6` 改善，正向 folds `2/5`，正向 horizons `2/4`
- model-step retention：`0.9833`
- 折算 TCN/LSTM model-step ratio：`4.7547×`

状态为 `stop_ema_seed7_no_holistic_gain_v41`。没有修改 decay 或重复筛选。

## v42 Phase A：seed 7 通过

唯一干预是 fold-scoped 训练目标：

`distilled_target = 0.75 * true_rank_target + 0.25 * teacher_consensus_rank`

teacher 只由冻结的 TCN seeds 7/17/27 在各 fold 的 train positions 上产生；validation teacher score、validation label、LSTM teacher 与 sealed test 均未进入训练。candidate 保持单个 TCN、单次推理前向。

seed 7 × 5 folds：

- RankIC delta：`+0.014599`
- Pearson IC delta：`+0.010170`
- Top return delta：`+0.000475`
- Top precision delta：`-0.001625`
- NDCG delta：`+0.006136`
- quantile monotonicity delta：`+0.011970`
- 正向 folds：`5/5`
- 正向 horizons：`4/4`
- RankIC 95% CI low：`+0.008981`
- model-step retention：`0.9876`
- 折算 TCN/LSTM model-step ratio：`4.7757×`

状态为 `consensus_student_seed7_holistic_admitted_v42`，按预注册协议授权 Phase B。

## v42 Phase B：多种子严格停止

Phase B 冻结相同 teacher checkpoints、teacher targets、`teacher_weight=0.25`、架构、损失、优化器与训练预算，只新增 student seeds 17/27。合并 3 seeds × 5 folds 后：

- RankIC delta：`+0.014508`
- Pearson IC delta：`+0.011381`
- Top return delta：`+0.000336`
- Top precision delta：`-0.003344`
- NDCG delta：`+0.008561`
- quantile monotonicity delta：`+0.017985`
- 正向 seed/fold 单元：`15/15`
- 各 seed RankIC delta：seed 7 `+0.014599`、seed 17 `+0.012896`、seed 27 `+0.016027`
- 正向 horizons：`4/4`，最差 horizon `+0.011556`
- RankIC 95% CI low：`+0.011021`
- model-step retention：`1.0090`
- complete-cycle retention：`1.0080`
- 折算 TCN/LSTM model-step ratio：`4.8789×`
- 推理前向次数：`1`

六项有 `5/6` 改善，但 Top precision delta 低于预注册下限 `-0.002`。因此正式状态必须是 `stop_consensus_student_multiseed_v42`，不能事后放宽门槛，也不能把显著的 RankIC 或速度增益拿来补偿该失败。

## 结论

1. **速度已达到原目标区间。** 当前单 student TCN 的实测 model-step 速度约为同预算 LSTM 的 `4.88×`，不是依靠三模型在线 ensemble 获得。
2. **不同参数盆地的共识确实能改善排序表征。** RankIC、Pearson IC、Top return、NDCG、单调性以及所有 seed/fold/horizon 的一致性均显著改善。
3. **当前仍不能晋级候选模型。** Top precision 的预注册非补偿门失败，且没有 sealed test 证据。
4. **不继续局部追逐 Top precision。** 本轮不搜索 teacher weight、不改 top-k、不改换手率、不重跑种子。下一轮若继续，应先前瞻性地澄清“横截面排序/平均收益”和“正收益命中率”之间的任务契约，而不是在同一验证集上为一个阈值调参。
5. **证据上限不变。** 结果仅是 ordinary-validation single-TCN stability evidence；不是 Alpha-ready、不是 sealed-test 通过、不是部署授权，也不证明 TCN 普遍优于 LSTM。

## 权威运行证据

- v41 receipt：`artifacts/tcn-v41-ema-seed7/receipt.json`
- v42 Phase A receipt：`artifacts/tcn-v42-consensus-student-seed7/receipt.json`，receipt id `21c855063fd1a249b2e5f357dbfc6f374b0fdac868ecaf1a5822820e8157fa77`
- v42 Phase B receipt：`artifacts/tcn-v42-consensus-student-multiseed/receipt.json`，receipt id `23df0908321989f2c006ee71afeac445e5089e4cd898c424acae1d6555128d05`
- 全部运行均记录 `sealed_test_accessed=false`、`alpha_ready=false`。
