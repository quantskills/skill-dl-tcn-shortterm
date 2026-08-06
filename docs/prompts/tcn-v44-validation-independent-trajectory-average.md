# TCN v44：验证集无关的训练轨迹等权平均提示词

你正在维护 `skill-dl-tcn-shortterm`。当前目标仍是使用单个 TCN 对真实股票分钟线形成 1/2/3/5 日横截面收益排序；LSTM 只作为冻结 benchmark。不得把项目改成 LSTM 项目，不得把方向分类、Top precision、换手率或某个 top-k 组合规则写入训练目标。

## 已知事实与本轮根因

- V42 的点式跨种子共识 student 相对原始 true-target TCN，在 3 seeds × 5 folds 上取得 RankIC `+0.014508`、Pearson IC `+0.011381`、Top return `+0.000336`、NDCG `+0.008561`、quantile monotonicity `+0.017985`，15/15 seed-fold 与 4/4 horizons 的 RankIC 增量为正；但 Top precision `-0.003344` 越过预注册下限，所以 V42 的正式状态仍是停止。
- V43 已证伪“用梯度归一化 listwise teacher loss 替代 V42 点式目标即可修复任务契约”的假设；其相对 V42 六项指标全部下降，不得继续搜索 teacher gradient ratio 或 temperature。
- V44 前置训练集诊断还证伪了简单 teacher reliability weighting：教师 rank 分歧与共识绝对误差未经控制时在 20/20 fold-horizon 单元中为负相关，均值 `-0.059393`；但教师分歧与共识绝对幅度相关 `-0.388860`，控制共识幅度后相关反转且只有 `+0.021697`。教师局部 score margin 控制共识幅度后与误差相关约 `-0.001204`。这些量没有足够稳定、独立的可靠性信息，因此不得据此做样本权重或 teacher-weight 搜索。
- V42 student 的训练损失在 3 seeds × 5 folds 中持续下降，但平均 ordinary-validation RankIC 从 epoch 1 的 `0.089573` 下降到 epoch 8 的 `0.084088`；每折所选 raw checkpoint 分散在 epoch 1–7。当前最可证伪的全局假设是：固定 `Adam(lr=0.003)` 的后半程参数在同一低损失盆地内追逐训练噪声，而逐折挑选验证峰值又放大了 checkpoint 方差。

## 唯一机制变量

保留 V42 pointwise student 的数据、标签、teacher、模型、损失、优化器和完整 8 epoch 原始训练轨迹，只改变最终参数来源：

1. epoch 1 仅作为固定 burn-in，不进入平均。
2. 每个 epoch 完成全部 optimizer steps 后，保存该时刻的所有可训练浮点参数 `theta_e`。
3. 从 epoch 2 开始做无衰减、等权的在线算术平均：

   `theta_bar_e = mean(theta_2, ..., theta_e)`

4. ordinary validation 可以在每个 epoch 记录 `theta_bar_e` 的诊断指标，但不得用任何验证指标选择平均起点、终点或中间 checkpoint。
5. candidate 的最终状态必须固定为 `theta_bar_8`；`best_epoch=8`，`checkpoint_parameter_source=epoch_uniform_average_final`，平均更新数严格为 7。
6. 非可训练 buffer 使用 epoch 8 的当前值；架构不含 BatchNorm。WeightNorm 的可训练参数必须像普通参数一样进入算术平均。
7. 最终 candidate 仍是一份标准 TCN state dict、一次推理前向；不允许在线 checkpoint ensemble。

## 完全冻结项

- 数据：真实 PandaData 2021–2025 股票分钟线、top50 PIT 股票池、relative10 特征、相同 window index/labels/fold/purge/embargo。
- 任务：1/2/3/5 日 next-open return 的横截面 rank target；方向只作诊断，不能进入 loss。
- teacher：复用 V42 已审计的 TCN seeds `{7,17,27}`、5 folds、train-only consensus ranks 与指纹；不得重训、替换或筛选 teacher。
- student target：`0.75 * true_rank_target + 0.25 * teacher_consensus_rank`；不得改变 `0.25`，不得使用 teacher agreement、score margin 或 validation 信息做逐样本权重。
- TCN：16 channels、kernel 2、dilations `1,2,4,8,16,32,64,128`、causal chomp、WeightNorm、无 dropout、单模型。
- 训练：float32、batch 128、Adam、learning rate `0.003`、weight decay 0、8 epochs、相同 batch order 与随机种子；不加 scheduler、EMA、SWA 周期、PCGrad、listwise、top-tail 或新特征。
- checkpoint 诊断继续只看 ordinary-validation mean daily RankIC；六项任务指标不得用于选 epoch。
- 不读 sealed/test，不使用既有 sealed 结果选参数；`sealed_test_accessed=false`。

## 实现与可验证契约

- 新增通用、可单测的 epoch 参数等权平均器。第一次更新必须精确复制参数；之后用在线均值 `avg += (value - avg) / count`，拒绝参数身份、shape、dtype、device 漂移。
- `ema_decay` 与 `epoch_average_start` 必须互斥；平均起点必须是正整数且不超过 `max_epochs`。
- candidate 必须关闭 early stopping 并完成 8 epochs；raw replay 与 candidate 每个 epoch 的原始 state dict 最大绝对误差必须严格为 0，证明平均器没有改变优化轨迹。
- 保存 raw epoch state、averaged checkpoint、每 epoch averaged validation RankIC、update count、起止 epoch、最终平均参数与离线重算算术平均的最大误差、raw-final/average-final 参数距离。
- 最终平均参数与 7 个源 epoch 的离线 float64/兼容 dtype 重算最大误差必须 `<=1e-6`；参数距离必须 `>0`；参数量、架构、推理次数与 V42 pointwise student 相同。
- V44 raw replay 的最终预测和任务指标必须与冻结 V42 seed-7 pointwise artifact 在数值容忍度内复现；否则实验因基础设施漂移失败。

## Phase A：seed 7 × 5 folds

候选先相对原始 true-target TCN 通过非补偿全局门：

- mean RankIC delta `>=+0.002`；至少 3/5 folds 为正；RankIC bootstrap 95% CI low `>=-0.002`。
- RankIC、Pearson IC、Top return、Top precision、NDCG、quantile monotonicity 至少 4/6 严格改善。
- Top return `>=-0.0001`、Top precision `>=-0.002`、NDCG `>=-0.001`、quantile monotonicity `>=-0.002`。
- 至少 3/4 horizons RankIC 为正，最差 horizon `>=-0.003`。

同时相对冻结 V42 pointwise student 做 Pareto 与稳定性保护：

- 六项至少 3 项严格改善。
- RankIC/Pearson IC 均 `>=-0.002`，Top return `>=-0.0001`，Top precision `>=-0.001`，NDCG `>=-0.001`，quantile monotonicity `>=-0.002`。
- candidate 对冻结 teacher validation ensemble 的 mean score RankIC fidelity 不得比 V42 pointwise student 低超过 `0.002`。
- 5 折中，epoch 2–8 的 averaged-checkpoint validation RankIC 时间标准差中位数必须不高于相同 raw 轨迹的 `0.90` 倍；这是机制门，不能补偿任务门失败。

速度门：相对 V42 pointwise student 的 model-step retention `>=0.95`，complete-cycle retention `>=0.90`，折算 TCN/LSTM model-step ratio `>=3.0x`，推理前向次数为 1。

所有门通过时状态为 `trajectory_average_seed7_holistic_admitted_v44`，才授权 Phase B；否则状态为 `stop_trajectory_average_seed7_v44`。失败后不得调整 burn-in、平均终点、teacher weight 或指标阈值。

## Phase B：seeds 17/27

只有 Phase A 通过才用完全相同协议运行 seeds 17/27。合并 3 seeds × 5 folds 后要求：

- 相对原始 control mean RankIC delta `>=+0.002`；15 个单元至少 9 个为正；每 seed mean delta `>=-0.001`；CI low `>=-0.001`。
- 六项至少 4 项改善且沿用 Phase A 下行保护；4/4 horizons 为正，最差 `>=-0.002`。
- 相对 V42 pointwise student 至少 3/6 改善并保持所有 Pareto 下限。
- 每个 seed 的轨迹平均更新数、算术身份、raw trajectory identity、稳定性、teacher fidelity、速度和单模型门全部通过。

通过状态为 `trajectory_average_multiseed_admitted_v44`；否则停止。不得因 Phase A 或 Phase B 的实际结果事后放宽门槛。

## 输出与最终验收

输出 frozen source manifest、前置 reliability 诊断、raw trajectory audit、epoch-average audit、epoch history、单模型 checkpoints、predictions、六项任务指标、相对 original control 与 V42 pointwise student 的成对比较、fold/horizon/seed deltas、bootstrap、teacher fidelity、stability、timing、model gate、receipt 与研究报告。

实现后运行 Ruff、mypy、完整 pytest、项目 preflight 和 wheel/sdist build。结论上限始终是 ordinary-validation single-TCN research evidence；不得宣称 Alpha-ready、sealed 通过、可部署或 TCN 普遍优于 LSTM。
