# TCN top50 PIT 横截面扩容验证提示词 v39

你是 `skill-dl-tcn-shortterm` 的量化研究、时序建模与证据治理执行者。v38 已证明，在 top20 ordinary validation 中完整保留 base8、只追加 `amount/ADV20` 与 `amount/market-cap` 两条因果序列后，TCN mean RankIC 提高 `+0.002257`，且 4/5 folds 为正；但 Top precision、NDCG 和 Top return 分别下降 `-0.005000/-0.012250/-0.000772`。top20 的 Top10% 每日只有约 2 只股票，当前首要未知是尾部退化是否来自横截面过窄，而不是继续搜索 TCN 结构或输入组合。本轮必须先补齐 top50 的 PIT 状态，再以冻结协议直接复验。

## 一、目标与不变边界

1. TCN 始终是主模型；LSTM 只在 seed-7 top50 门禁通过后作为 benchmark，不得反向决定 TCN 参数。
2. 旧 v36 sealed identity 已永久消费并拒绝。禁止读取、重放、派生特征、选择模型或调整门槛；本轮只允许 ordinary train/validation，并始终记录 `sealed_test_accessed=false`。
3. 不修改标签定义、四个期限、480 根 5 分钟回看、walk-forward、purge/embargo、损失、优化器、epoch、batch、线程、checkpoint 选择或 TCN 架构。
4. 固定 dynamic-horizon-skip TCN：channels 16、kernel 3、dilations `1/2/4/8/16/32/64/128`、严格 causal chomp、Weight Normalization、dropout 0、SmoothL1、lr 0.003、batch 128、8 epochs、CPU 8 threads、float32、memmap。
5. 本轮只比较 top50 base8 与 top50 `base8 + relative2`；禁止新增静态横截面秩、重复 date-level 数值、adapter、attention、更多学习率或其他结构。
6. 不部署、不交易、不连接券商、不写发布凭据。数据和实验产物不可覆盖，必须绑定 SHA-256、代码身份、环境和上游收据。

## 二、Phase 0：top50 PIT 数据修复

1. 只读复用 `pandadata-csi300-top100-2021-2025-v1` 的 29,071,440 条 1 分钟分区与每日 PIT 权重，不重新下载分钟线。
2. 每日严格按 `weight desc, instrument_id asc` 选择前 50；预期 1,212 个交易日、60,600 个 membership keys、111 个唯一股票。
3. 只读复用 top20 enrichment 的 46 个唯一股票及其日线、公告日股本、复权事件和状态产物；只为 top50 新增的 65 个唯一股票拉取 2021–2025 的 `get_stock_daily/get_stock_status_change/get_adj_factor/get_share_float`。
4. 下载必须按月、每批最多 25 只、分块可恢复；凭据只来自进程环境，不写入配置、日志、收据、SDK `user.json` 或 Git。永久错误不重试，瞬时错误有界退避。
5. 合并前验证上游 source manifest SHA、年份、top_n 和所有 artifact SHA；合并键冲突必须 fail-closed，不能用 `keep=last` 静默覆盖。
6. ADV20 仅使用当前及此前 19 个已完成交易日；股本只能从供应商信息日期起向后 `asof`；禁止未来回填。市值为 signal-date close × 当时已知 total shares。
7. 公司行为继续使用 `unadjusted-minute-label-invalidation-v1`，跨不可靠未复权事件的标签必须失效；资格、停牌、ST 和完整交易时段继续按原契约处理。
8. top50 runtime 只接受每日 240 根完整 1 分钟记录的 stock-day。状态门禁必须报告 membership、可用 state、正有限 market cap、ADV20、公司行为覆盖、缺失键和日期；任何训练样本所需 PIT state/market cap 缺失即停止，不允许插值、当前状态回填或用 top20 状态代替。

## 三、Phase 1：top50 base8 与 relative10 物化

1. 用 top50 enriched runtime 重新生成 5 分钟标准条、480 步窗口、四期限原始收益与横截面 rank labels，以及五折 expanding ordinary-validation manifest。
2. base8 顺序固定为 `close_return/open_close_return/intrabar_range/log_volume/log_amount/vwap_deviation/time_sin/time_cos`。
3. relative10 必须逐位保留 base8，再追加：
   - `log1p(5m amount / signal-date ADV20)`；
   - `log1p(5m amount / signal-date market cap)`。
4. 不得添加五个静态横截面秩。relative10 的前八通道必须与 base tensor 逐位相等；样本位置、ID、instrument、signal date、480 步和标签必须完全一致。
5. 记录每天实际可评分股票数和 Top10% 持仓数；若 validation 日期的横截面宽度不能把主组合稳定扩展到至少 4 只，则标记 `blocked_insufficient_effective_breadth_v39`，不训练。

## 四、Phase A：seed-7 TCN-only 五折门禁

分别训练 top50 `base_tcn` 与 `relative_tcn`，seed 固定为 7、folds 为 0..4；两者除输入通道数外完全同协议。不得看到部分 fold 后提前修改协议。

输出 RankIC、Top return、Top excess return、Top precision、NDCG@Top、turnover、逐期限指标、fold 配对差、日期 block-bootstrap、参数量、samples/s、完整周期耗时、checkpoint replay 和样本覆盖。

只有同时满足以下条件，状态才为 `top50_relative_sequence_seed7_admitted_v39`：

1. 五折 mean RankIC delta `>= 0.002`；
2. 至少 3/5 folds 的 RankIC delta `> 0`；
3. mean Top precision delta `>= 0`；
4. mean NDCG delta `>= 0`；
5. mean Top return delta `>= -0.0001`；
6. mean turnover delta `<= 0.02`；
7. RankIC block-bootstrap 95% CI low `>= -0.002`；
8. relative TCN median samples/s 至少为 base TCN 的 90%；
9. 五折、四期限、预测键、标签、checkpoint replay 和所有数值完整有限；
10. validation 的每日有效横截面与 Top10% 持仓宽度满足 Phase 1 门禁。

任一条件失败，状态为 `stop_top50_relative_sequence_seed7_v39`，不得执行 Phase B。失败只否定当前 relative2 在 top50 的稳定增量，不否定 TCN 速度路径或项目本身。

## 五、Phase B：仅在 Phase A 通过时

1. 训练 relative TCN seeds 17/27，与 seed 7 合并为 15 个 seed×fold 单元；必要时也训练相同 seeds 的 base TCN，保证配对完整。
2. 用相同 top50 relative10 训练参数与预算冻结的 LSTM seeds 7/17/27，仅作为效果和速度 benchmark。
3. 多 seed 准入沿用 v38 门槛：mean RankIC delta `>=0.002`、至少 9/15 正单元、Top precision/NDCG 不退化、Top return `>=-0.0001`、turnover `<=0.02`、RankIC CI low `>=-0.002`、TCN 吞吐保留 `>=90%`。
4. 如通过，标记 `top50_relative_sequence_multiseed_admitted_v39`；否则标记 `stop_top50_relative_sequence_multiseed_v39`。LSTM 的胜负只如实报告，不得用单一 RankIC 宣称 TCN 必然优于 LSTM。

## 六、解释边界与后续决策

- 若 top50 同时保留 RankIC 改善并消除尾部退化，说明 v38 的主要限制是有效横截面宽度，可在新的 future holdout 治理前继续普通验证研究。
- 若 top50 仍是 RankIC 改善但尾部退化，下一轮才允许预注册独立的 post-encoder stock-context/DeepSets adapter；不得把静态状态重复到时间轴。
- 若 RankIC 与尾部都失败，停止这两条 relative sequence，不继续排列组合；转向标签噪声、交易可实现性与 date-level cross-sectional objective 的未知项。
- ordinary validation 改善不等于 Alpha、候选模型、部署或交易授权，也不创建新的 sealed 授权。

## 七、工程验收

先写行为测试，再实现增量 enrichment、冲突检测、PIT 泄漏审计、top50 runtime、base8/relative10 memmap 和分阶段 runner。执行 focused tests、全量 Ruff、Mypy、Pytest、`tasks/preflight.py`、`tasks/test.py` 和 production build。构建包不得包含 artifacts、checkpoint、Parquet、NPY、PT、`.env` 或凭据。更新 work item、verification 和独立结果文档，记录实际执行范围、所有门禁值、阻塞项、收据与 `sealed_test_accessed=false`。

按以上提示词直接执行。不得降低门槛、扩大搜索空间、用缺失状态强行训练或读取旧 sealed 挽救结果。
