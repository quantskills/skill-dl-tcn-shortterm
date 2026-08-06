# TCN 短线模型 Infra 与容量优化执行提示词 v5

你是 `skill-dl-tcn-shortterm` 的 TCN 性能工程与普通验证优化执行者。项目目标固定为：在五年真实 A 股分钟线窗口上优化并验证因果 TCN；LSTM/GRU 只能作为测量对照，不能替代 TCN、改变项目方向或触发“放弃 TCN”的结论。

## 不可变边界

- 复用只读源运行 `artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`，先核对输入 SHA-256。
- 只使用普通 train/validation；禁止读取 test、sealed holdout、`sealed=true` 或扩大股票池寻找有利结果。
- 保持输入为 `[sample, feature=8, time=480]`、float32、因果左填充、WeightNorm、横截面 `signal_date × horizon` Spearman RankIC。
- 当前机器无 CUDA，本轮只陈述 CPU 证据；不得把单机结果外推为普遍结论。
- 所有正式输出不可覆盖，必须带配置、环境、输入/输出哈希和 `sealed_test_accessed=false` 收据。

## 已知红灯与基线

- 20 个 PyTorch intra-op threads 会导致过度并行；8 threads 是当前机器的已观察候选甜点。
- 5 folds × 3 seeds 旧基线：TCN-lite/LSTM 模型步约 `2.905×`、完整周期约 `1.883×`。
- 旧 TCN-lite 完整周期中，验证约占 `44.4%`；重复执行的标签联接、Pandas 分组和目标排名是固定开销。
- 真实 fold 0 上旧 RankIC 单次约 `0.625s`；缓存验证计划后的数值等价探针约 `0.014s`，约 `43.9×`。
- memmap 的批量 fancy-index 探针比现有逐样本顺序路径慢约 `12%`，因此不得把它作为默认优化。
- 参数匹配诊断把 Bai-TCN/TCN-lite 通道压到 3/4，而 8 个输入特征会立即被压缩；该诊断不能代表合理容量 TCN 的效果上限。

## Phase 1：Infra 数值等价优化

1. 为验证集按 fold 预构建不可变 `ValidationRankICPlan`：一次解析 sample position、标签、日期/期限分组和目标秩；每轮只对模型分数排名并计算相关。
2. 缓存计划必须校验 sample positions 完全一致；顺序变化、未知位置、标签重复或形状错误必须 fail closed。
3. 用包含 ties、无效标签、乱序 positions 的回归测试证明缓存路径与旧定义数值等价。
4. TCN tuning 和跨模型 benchmark 都必须复用每个 fold 的计划，禁止每个 epoch 重建。
5. tuning 增加作用域化 `torch_threads`，异常和正常退出后都恢复进程原线程数；receipt/leaderboard 记录实际线程数。
6. 不采用已被真实探针否定的批量 memmap fancy-index；保留只读 lazy memmap。

Phase 1 门槛：真实 fold 上 RankIC 核心计算至少快 `10×`，绝对差小于 `1e-12`；完整测试保持通过。

## Phase 2：TCN 执行路径优化

在固定 seed、fold、输入、损失和学习率下单变量测试：

- threads：`4/8/12`，禁止重新纳入已知劣化的 20 threads 作为候选；
- batch：`128/256/512`；
- 模型：Bai-TCN 与 TCN-lite；LSTM-8 只用于速度比值分母；
- 分别记录 data wait、模型前后向、验证推理、RankIC 计算、完整周期、峰值 RAM 和参数量。

先用一折一 seed 筛选，再用全部 5 folds × seeds `7/17/27` 复核。执行优化必须保持同 fold/seed 的 shuffle 次序和预测口径一致。

Phase 2 门槛：TCN-lite 相对 LSTM 的模型步几何平均目标 `>=3.0×`，完整周期目标 `>=2.0×`；未达到时报告观测值和剩余算子占比，不得伪造“3–5×”。

## Phase 3：合理容量的 TCN-only 优化

把“速度模型”和“效果模型”分开，参数匹配小模型只保留为诊断控制组：

- 控制：Bai-TCN channels `8`，kernel `3`，dilations `1..64`；
- 候选：Bai-TCN channels `16`；TCN-lite channels `8/16`，感受野至少覆盖 480；
- batch 采用 Phase 2 甜点；threads 采用作用域化甜点；
- 学习率只在预登记集合 `{0.001, 0.003, 0.01}` 中选择；最大 12 epochs、patience 3、min_delta 0.002；
- 先执行 5 folds × seed 7 的有界筛选，只让预登记排序第一的 TCN 候选进入 seeds `17/27` 复核。

效果比较以当前 TCN 控制组为主，LSTM 仅作为外部参照。必须报告：每折每 seed 最佳 epoch、日度 RankIC、四期限 RankIC、正值比例、最差折、参数量、模型步吞吐、完整周期吞吐和 time-to-best。不得把仅 fold 0 的改善写成稳定改善。

Phase 3 进展门槛：候选 TCN 的中位 RankIC 至少比 TCN 控制组提高 `0.005`，至少 `80%` fold-seed 单元为正，按 seed 平均后的最差折不低于 `-0.01`。未通过只表示当前候选/训练计划未通过，不能改写为“TCN 不适合本项目”。

## 验收与输出

1. 先红后绿：缓存等价、position safety、线程恢复、任务配置透传和收据字段均有回归测试。
2. 运行 Ruff、mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py`、production build。
3. 更新 README、WORK_ITEMS、requirements traceability、verification 和新的不可变性能/调优收据。
4. 最终分开回答：infra 消除了多少固定开销；TCN 模型步与完整周期分别相对 LSTM 多快；合理容量 TCN 的普通验证效果比旧 TCN 改善多少；哪些仍是未知量。

严格按以上提示词直接执行。项目方向始终是优化 TCN；任何失败门槛都触发下一轮有界诊断，而不是未经授权地更换模型目标。
