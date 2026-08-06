# TCN 速度—效果 Pareto 收敛执行提示词 v6

你是 `skill-dl-tcn-shortterm` 的 TCN 性能与普通验证优化执行者。项目方向固定为因果 TCN；LSTM只允许作为速度参照，不得替代TCN。目标是把v5的“最快TCN-lite-4”和“效果最佳Bai-TCN-16”收敛为至少一个同时具备可接受RankIC与训练吞吐的TCN候选。

## 不可变输入与边界

- 只读复用五年真实运行 `artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/` 和expanding普通验证清单 `artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`。
- 只访问train/ordinary validation；禁止访问test、sealed holdout、`sealed=true`，禁止扩大股票池或看到结果后追加trial。
- 保持输入`[sample, 8, 480]`、float32、因果左填充、WeightNorm、四期限共享表征与`signal_date × horizon` Spearman RankIC。
- 所有正式运行不可覆盖，必须保存配置、输入/输出SHA-256、线程/loader环境和`sealed_test_accessed=false`收据。

## 已知红灯与探针结论

- v5真实seed-7容量表没有任何候选同时达到`mean RankIC >= 0.09`与训练吞吐`>=5000 samples/s`：Bai-TCN-16为`0.09419/997sps`，TCN-lite-16为`0.09197/1694sps`，TCN-lite-4为`0.07575/11242sps`。
- 因果深度可分离TCN-16虽然把参数从6228降到2988，但CPU模型步由约3997/s降到3073/s；该结构在本轮被否定，不得进入正式实现。
- TCN-lite-16的执行甜点不同于lite-4：微探针中`8 threads/batch128`约5709/s，`12 threads/batch256`约6404/s。
- 真正的大开销是每个残差块对完整`[batch, channel, 480]`激活执行element-wise dropout：TCN-lite-16在4线程/batch128下，block dropout 0.1约1416/s，dropout 0约4147/s；8线程下约1665/s与5916/s。

## Phase 1：低成本正则化实现

1. 为TCN-lite增加独立`head_dropout`，只作用于最后时刻的共享表示`[batch, channel]`，默认0以保持旧模型完全兼容。
2. 保留现有block dropout参数；正式候选使用`block_dropout=0`，并比较`head_dropout=0/0.1`。
3. 参数验证必须拒绝不在`[0,1)`的head dropout；训练配置、leaderboard和receipt必须分别记录block/head dropout。
4. 回归测试必须证明：
   - 默认参数与旧TCN-lite状态字典、输出形状和因果性兼容；
   - eval模式下head dropout不改变确定性；
   - train模式下head dropout位于最终表示而不是480步序列；
   - tuning task从JSON正确透传并记录两个dropout字段。

## Phase 2：固定结构执行筛选

使用同一TCN-lite-16、seed 7、一个普通验证fold做单变量探针：

- block/head dropout：`0.1/0`、`0/0`、`0/0.1`；
- threads/batch：`4/128`、`8/128`、`12/256`；
- 固定学习率0.003、相同shuffle、相同最大epoch与早停。

记录模型步、data wait、完整训练吞吐、验证推理、RankIC、参数量和time-to-best。只有速度与RankIC同时不劣的组合进入Phase 3。

## Phase 3：预登记五折筛选

在5个expanding ordinary-validation folds、seed 7上只比较以下四个trial，最多8 epochs、patience 2、min_delta 0.002：

1. `lite-c16-block01`：channels16、block dropout0.1、head dropout0；
2. `lite-c16-head01`：channels16、block dropout0、head dropout0.1；
3. `lite-c16-no-dropout`：channels16、block/head dropout均0；
4. `lite-c12-head01`：channels12、block dropout0、head dropout0.1。

全部使用kernel3、dilations`1..128`、learning rate0.003，并采用Phase 2中对应结构的最快threads/batch。排序先按五折mean RankIC降序，再按最差折、训练吞吐、参数量和trial ID确定性决胜。禁止追加学习率或结构。

## Phase 4：多seed确认与门槛

只让seed-7预登记排序第一名与`lite-c4-lr1e2`速度控制组进入seeds 17/27确认。合并15个fold-seed单元后要求：

- 候选RankIC中位数`>=0.09`，正值比例`>=80%`，按seed平均后的最差fold`>=-0.01`；
- 相对lite-4的配对RankIC中位提升`>=0.005`，三个seed平均改善均为正；
- 候选训练吞吐`>=5000 samples/s`或至少为旧block-dropout TCN-lite-16的`2.5×`；
- 另行对LSTM-8报告模型步与端到端速度比，不要求为追逐比值而降低TCN容量。

通过时状态写为`pareto_candidate_confirmed`；失败时只说明相应结构/正则化计划未收敛，不得改写为放弃TCN。

## 验收

- 先红后绿执行聚焦测试；保留深度可分离结构的负探针证据，但不提交throwaway实现。
- 运行Ruff、Mypy、完整pytest、`python tasks/preflight.py`、`python tasks/test.py`和production build。
- 更新README、WORK_ITEMS、requirements traceability和verification；清楚区分微探针、普通验证证据与尚未授权的sealed test。

严格按该提示词直接执行。目标始终是优化TCN，并以真实收据决定TCN内部的结构和infra选择。
