# TCN 追加式相对时序特征消融 v38

你是 `skill-dl-tcn-shortterm` 的量化研究、时序建模和验证执行者。v37 已在 ordinary validation 上证明：用 13 通道相对表征替换原始 `log_volume/log_amount`，并把五个 date-level 静态秩复制到 480 个时间步，会同时降低 TCN 和 LSTM 的预测效果；但 TCN/LSTM 完整周期速度仍为 5.7286×。本轮继续保留 TCN 方向，只检验“保留原始量价信息、追加两个因果相对时序通道”能否恢复并改善 TCN。

## 一、不可变边界

1. TCN 是主模型，LSTM 只在 seed-7 门禁通过后作为确认 benchmark。
2. 旧 v36 sealed 已永久消费，禁止读取、重放、派生特征、选择模型或调整门槛。本轮只允许 ordinary `train/validation`。
3. 不修改标签、split、fold、seed、epoch、optimizer、loss、TCN 架构、checkpoint 选择或速度线程设置。
4. 固定 dynamic-horizon-skip TCN：channels 16、kernel 3、dilations 1/2/4/8/16/32/64/128、chomp causal padding、Weight Normalization、dropout 0、SmoothL1、lr 0.003、batch 128、8 epochs、CPU 8 threads、float32、只读 memmap。
5. 不搜索更多比例、窗口、技术指标、层数、学习率或静态 adapter。本轮只允许一个候选；失败即停止 v38。
6. 不部署、不交易、不连接券商、不写凭据。所有产物必须不可覆盖，带源 SHA-256、代码身份、配置、环境和 `sealed_test_accessed=false`。

## 二、唯一候选特征

构造 `causal-base-plus-relative-sequence-v38` 十通道：

1. 完整保留 v37 base 的八个通道和顺序：
   `close_return/open_close_return/intrabar_range/log_volume/log_amount/vwap_deviation/time_sin/time_cos`；
2. 追加 v37 已审计的两个因果时序通道：
   `log1p(5m amount / signal-date ADV20)`、
   `log1p(5m amount / signal-date market cap)`；
3. 不追加五个静态横截面秩，不把任何 date-level 数值重复为伪时序；
4. 新 tensor 必须逐位保持前八通道与 base tensor 完全相等，追加两通道必须与 v37 已物化相对通道逐位相等；
5. 样本 position、sample ID、instrument、signal date、480 步、标签和折覆盖全部不变；
6. 继续沿用 v37 的 423 个 causal ADV fallback 及其审计，不重新计算或接触未来数据。

## 三、Phase A：seed-7 TCN-only 五折门禁

复用 v37 中同协议 `base_tcn` seed 7 的不可变 predictions、leaderboard 和 receipt；只训练十通道候选 `relative_tcn` 的 seed 7、folds 0..4。不得因看到某折结果提前停止或修改候选。

输出相同的 RankIC、Top return、Top excess return、Top precision、NDCG@Top、turnover、逐期限指标、fold 配对差、日期 block-bootstrap、参数量、samples/s 和 checkpoint replay。

只有同时满足以下条件，状态才为 `append_relative_sequence_seed7_admitted_v38`：

1. 五折 mean RankIC delta >= 0.002；
2. 至少 3/5 folds 的 RankIC delta > 0；
3. mean Top precision delta >= 0；
4. mean NDCG delta >= 0；
5. mean Top return delta >= -0.0001；
6. mean turnover delta <= 0.02；
7. RankIC block-bootstrap 95% CI low >= -0.002；
8. 候选 TCN median samples/s 至少为 base TCN 的 90%；
9. 五折覆盖、四期限、预测样本键、标签、checkpoint replay 和所有数值完整有限。

任何条件失败，状态为 `stop_append_relative_sequence_seed7_v38`，不得执行 Phase B。

## 四、Phase B：仅在 Phase A 通过时

若 Phase A 通过，才允许：

1. 训练候选 TCN seeds 17/27，合并 seed 7 形成 15 个 seed×fold 单元；
2. 用同十通道训练 LSTM seeds 7/17/27，与候选 TCN 作同样本多指标 benchmark；
3. 使用 v37 的完整准入门：mean RankIC delta >=0.002、至少9/15正单元、Top precision/NDCG 不退化、Top return >=-0.0001、turnover <=0.02、RankIC CI low >=-0.002、TCN 吞吐保留 >=90%；
4. LSTM 结果只用于报告模型相对表现，不得反向调参。

Phase B 通过才标记 `append_relative_sequence_multiseed_admitted_v38`，授权下一轮把十通道接回 v35 Top-tail/checkpoint 合同；否则标记 `stop_append_relative_sequence_multiseed_v38`。

## 五、解释边界与下一步

- 若十通道通过，说明 v37 的主要错误是破坏性替换或静态秩复制，而不是相对量价信息本身无效。
- 若十通道失败，说明两个相对通道在当前 top20 横截面没有稳定增量；下一步不再改输入通道，而应补齐 top50 PIT 状态或实现独立 post-encoder stock-context/DeepSets adapter。
- top50 仍缺 36,375 个 PIT 状态键，本轮不得绕过门禁扩容。
- ordinary validation 改善不等于可交易 Alpha，不产生新的 sealed 授权。

## 六、工程验收

先写行为测试，再实现 chunked memmap 物化和不可变 Phase A runner；执行 focused tests、全量 Ruff、Mypy、Pytest、preflight、统一测试入口和 production build。构建包不得包含 artifacts、checkpoint、Parquet、NPY、PT、`.env` 或凭据。更新 work item、verification 和独立结果文档，记录无论成功或失败的完整门禁值。

按以上提示词直接执行。不得降低门槛、扩大搜索空间或读取旧 sealed 来挽救候选。
