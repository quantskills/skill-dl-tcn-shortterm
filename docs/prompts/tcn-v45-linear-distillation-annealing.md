# TCN v45：从共识教师线性退火到真实标签的全局蒸馏提示词

你正在维护 `skill-dl-tcn-shortterm`。目标保持为单个 TCN 对真实股票分钟线做 1/2/3/5 日横截面收益排序，LSTM 只作为冻结 benchmark。不得围绕 Top precision、换手率、某个 top-k、单折或单期限增加局部 loss。

## 已知证据

- V42 固定 `25%` teacher 共识在 3 seeds × 5 folds 上广泛改善 5/6 指标、15/15 单元和 4/4 horizons，但 Top precision 触发停止门。
- V43 listwise teacher loss 相对 V42 六项全退化，禁止继续调 teacher gradient ratio/temperature。
- V44 epoch 2–8 参数平均把 validation 轨迹波动降到 raw 的 `29.7%`，机制、速度和 teacher fidelity 均通过，但相对 V42 六项 `0/6` 改善；它更像 teacher `+0.026889`，RankIC 却下降 `-0.006526`。
- 对 V44 保存的 V42 raw epoch checkpoints 做零重训诊断：epoch 与 teacher fidelity Spearman `+0.547619`，epoch 与真实 validation RankIC `-0.571429`，teacher fidelity 与真实 RankIC `-0.476190`。当前可证伪假设是固定 teacher 权重让后期优化继续追随平滑共识，覆盖了真实标签区分度。

## 唯一机制变量

保留 V42 的全部数据、TCN、优化器、batch order、预算和 checkpoint selection，只把固定 teacher blend 改成预注册的逐 epoch 线性退火：

`w_e = 0.25 * (8 - e) / 7, e in {1,...,8}`

`target_e = (1 - w_e) * true_rank_target + w_e * teacher_consensus_rank`

因此 epoch 1 的 teacher 权重严格为 `0.25`，epoch 8 严格为 `0`，中间依次为 `3/14, 5/28, 1/7, 3/28, 1/14, 1/28`。不搜索起点、终点、曲线形状或 epoch 数。

teacher target 只允许来自 V42 已审计的 fold-scoped train positions。blend 必须在 batch 内即时计算，不复制完整 memmap，不修改 validation target，不把 validation teacher score、validation label 或 sealed 数据写入 schedule。

## 冻结项

- 真实 PandaData 2021–2025 股票分钟线、top50 PIT、relative10、相同 folds/purge/embargo、1/2/3/5 日 next-open rank target。
- teacher seeds `{7,17,27}`、5 folds、train-only consensus rank、所有文件指纹。
- 16-channel dynamic horizon-skip TCN、kernel 3、dilations `1..128`、causal chomp、WeightNorm、无 dropout、单模型单前向。
- float32、batch 128、Adam `lr=0.003`、weight decay 0、8 epochs、固定 batch order、无 scheduler/EMA/参数平均/PCGrad/listwise/top-tail。
- checkpoint 仍只按 ordinary-validation mean daily RankIC 严格改善选择；不得按六项任务指标选 epoch。
- 不读 sealed/test，不使用既有 sealed 结果选择任何参数。

## 实现契约

- `teacher_blend_start_weight` 与 `teacher_blend_end_weight` 必须显式、有限且满足 `0 <= end < start < 1`；仅允许 SmoothL1 trial，禁止与静态 target override、teacher listwise、EMA 或 epoch parameter average 混用。
- 每个 epoch 记录实际 teacher weight；5 folds 的权重序列必须逐值等于预注册 8 项序列，epoch 8 target 在有效训练单元上逐值等于 true target。
- teacher 矩阵在 validation/non-train positions 必须保持 NaN；训练覆盖严格等于 fold train positions。
- 保存 target schedule audit、teacher source identity、epoch history、最终单 TCN checkpoints 和完整 receipt。

## Phase A：seed 7 × 5 folds

相对原始 true-target control：mean RankIC delta `>=+0.002`，至少 3/5 folds 和 3/4 horizons 为正，最差 horizon `>=-0.003`，RankIC CI low `>=-0.002`；六项至少 4 项改善；Top return `>=-0.0001`、Top precision `>=-0.002`、NDCG `>=-0.001`、monotonicity `>=-0.002`。

相对冻结 V42 pointwise student：六项至少 3 项严格改善；RankIC/Pearson 均 `>=-0.002`、Top return `>=-0.0001`、Top precision `>=-0.001`、NDCG `>=-0.001`、monotonicity `>=-0.002`。teacher fidelity 仅作机制诊断，不允许补偿真实任务门。

速度：相对 V42 model-step retention `>=0.95`、complete-cycle retention `>=0.90`、折算 TCN/LSTM model-step ratio `>=3.0x`、推理前向次数为 1。

全部通过状态为 `linear_distillation_annealing_seed7_admitted_v45`，才授权 Phase B；否则 `stop_linear_distillation_annealing_seed7_v45`，禁止再试指数/余弦/分段 schedule 或调整 teacher 权重。

## Phase B 与结论边界

仅 Phase A 通过才以完全相同 schedule 运行 seeds 17/27。合并 15 单元后要求：相对原始 control RankIC `>=+0.002`、至少 9/15 单元为正、每 seed mean `>=-0.001`、4/4 horizons 为正、六项至少 4 项改善；相对 V42 pointwise 至少 3/6 改善并保持 Phase A Pareto 下限；速度、schedule、泄漏和单模型门全部通过。

最终运行 Ruff、mypy、完整 pytest、preflight、wheel/sdist build。结论只能是 ordinary-validation single-TCN research evidence，不能宣称 Alpha-ready、sealed 通过、可部署或 TCN 普遍优于 LSTM。
