# TCN CPU 吞吐与跨时期稳定性优化提示词 v4

你是 `skill-dl-tcn-shortterm` 的性能与普通验证优化执行者。请在不下载新数据、不读取 test/sealed test、不改变横截面排名任务的前提下，回答两个可证伪问题：

1. 当前 TCN 未获得速度优势，是否主要由 CPU 线程过度并行、测量口径混合和 Bai 双卷积块开销造成？
2. 当前 fold 0 改善、fold 1 退化，是否能通过更多普通验证时期、多个随机种子和滚动训练窗口得到稳定改善？

不得把卷积的理论并行性写成固定 3–5× 承诺；不得为了通过门禁降低阈值、读取 test、扩大 top50 或选择性报告有利线程数。

## 固定输入与边界

- 复用只读运行 `artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`。
- 只消费 `feature-windows.npy`、`window-index.parquet`、`labels.parquet` 和原始 `split-manifest.parquet`，先复算 SHA-256。
- 稳定性验证的候选池只能来自原始 fold 1 中 `train` 与 `validation` 行；原始 `test`、`sealed_holdout`、`sealed=true` 和未登记日期一律排除。
- 真实数据及运行收据继续写入 Git 忽略的 `artifacts/`；源代码、测试、非秘密配置、提示词和验证摘要进入仓库。
- 当前机器无 CUDA，因此只执行 CPU/float32。GPU、AMP 和 CUDA `torch.compile` 不在本轮结论中。

## Phase 0：红灯反馈环与原因探针

- 使用真实形状 `[batch=128, feature=8, time=480]`，比较参数接近的 LSTM-8、Bai-TCN-3 和 TCN-lite-4。
- 分别测试 PyTorch intra-op threads `1/4/8/20`，固定 warmup、迭代数、seed 和优化器。
- 分开记录模型前后向步吞吐、训练数据管线吞吐、验证耗时和完整周期吞吐。
- 已观察的基线红灯必须保留为证据：20 线程下 Bai-TCN/LSTM 约 `0.63×`；8 线程探针约 `1.97×`，TCN-lite/LSTM 约 `3.25×`。该探针只用于定位原因，最终结论以真实 memmap 为准。

## Phase 1：测量与运行时修复

- 为性能基准增加有作用域且会恢复现场的 `torch_threads` 配置；禁止永久改变调用进程的线程数。
- 同一 fold/seed 下所有模型使用相同 shuffle seed；收据记录 base seed、model seed、请求线程数和实际线程数。
- `samples_per_second` 保留为完整训练—验证周期兼容字段，并新增：
  - `model_step_samples_per_second`
  - `train_pipeline_samples_per_second`
  - `model_step_seconds`
  - `data_wait_seconds`
  - `validation_seconds`
  - `end_to_end_seconds`
- 性能基准的预测指标必须按 `signal_date × horizon` 独立计算 Spearman RankIC，禁止把全部日期和期限扁平后计算相关系数。
- 在原有 LSTM、GRU、Bai-TCN 外增加参数接近 LSTM-8 的 TCN-lite-4：单卷积残差块、kernel 3、dilation `1..128`、感受野 511、WeightNorm、无 BatchNorm。

## Phase 2：五折普通验证稳定性清单

- 从原始 fold 1 的 `train + validation` 日期构造 5 个普通验证 folds：`train_days=400`、`validation_days=80`、按日期步进 80。
- 同时生成 expanding 与 sliding-400 两份清单；所有训练标签的 `label_end_at` 必须早于对应验证起点，否则标记为 `purged`。
- 已封存的 2023-12 至 2024-05 区间不得进入任何新 fold；之前的普通 validation 只有在隔离期已经结束后才能进入后续训练。
- 清单及摘要必须记录源哈希、日期边界、train/validation/purged 样本数、窗口类型和确定性指纹，并拒绝覆盖已有收据。

## Phase 3：真实数据比较

- 先在 expanding 与 sliding 清单上用 seed 7、3 epochs 对比 LSTM、GRU、Bai-TCN-3、TCN-lite-4，统一 batch 128、Adam、float32、8 threads。
- 依据五折 `mean_daily_rankic` 的中位数选择窗口协议；并列时依次选择最差 fold 更高、折间标准差更小、完整周期吞吐更高者。
- 在选中的窗口协议上用 seeds `7/17/27` 重跑，同一 fold/base-seed 下各模型共享 shuffle seed。
- 报告模型 × fold × seed 的 RankIC、参数量、模型步吞吐、完整周期吞吐、正 RankIC 比例、中位数、最差 fold 和标准差。

## 预登记验收门槛

速度结论分级：

- `cpu_model_step_speedup_confirmed`：TCN 相对 LSTM 的模型步几何平均吞吐至少 `1.5×`。
- `cpu_end_to_end_speedup_confirmed`：同时要求完整周期几何平均至少 `1.2×`。
- 只有模型步达到 `3.0×` 才可陈述“本机特定配置下模型内核达到约 3×”；不得扩展为通用 3–5×。

效果结论分级：

- 候选至少 60% 的 fold-seed 单元 RankIC 为正。
- 候选 RankIC 中位数必须高于 LSTM，且相对 Bai 控制组中位数至少改善 `0.005`。
- 候选按 seed 平均后的最差 fold 不得低于 `-0.01`。
- 任一效果条件失败即 `stop_unstable_validation`，不得扩容 top50、读取 test 或称为 Alpha。

速度通过而效果失败时，将结论写为“TCN CPU 并行优势已在特定线程配置下恢复，但没有稳定预测增量”；TCN-lite 只能保留为候选编码器或组合分支。

## TDD 与最终验收

- 先让公共行为测试变红：线程作用域恢复、日度 RankIC 口径、模型步/完整周期分账、TCN-lite 比较、五折清单禁止 test/sealed 和动态 purge。
- 逐项实现并运行聚焦测试，再执行真实 Phase 1–3。
- 最后运行 Ruff、Mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py` 和 `python -m build`。
- 更新 README、WORK_ITEMS、requirements traceability 与 verification；只记录实际观察值，不把本机 CPU 结果外推到其他硬件。

按以上提示词直接执行。门禁失败是有效终止结论，不得通过追加试验次数或改阈值寻找有利结果。
