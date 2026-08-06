# TCN top50 PIT 横截面扩容验证 v39 结果

## 决策

v39 Phase A 已完整执行，状态为 `stop_top50_relative_sequence_seed7_v39`。top50 扩容消除了 v38 的预测效果退化：relative10 相对 base8 的 RankIC、Top precision、NDCG 和 Top return 均为正，RankIC、Top precision 与 NDCG 的日期 block-bootstrap 95% 下界也均高于零；但平均 turnover delta 为 `+0.030759`，超过预登记上限 `+0.02`。因此唯一阻塞项是 `turnover_delta_above_gate`，Phase B 的 seeds 17/27 与 LSTM benchmark 未获授权。旧 sealed test 未访问。

完整执行提示词见 `docs/prompts/tcn-top50-pit-breadth-validation-v39.md`。top50 训练数据 receipt 为 `dc13c0cdb73c0162060519784b8b581a16ae99f03c6d60bb34f87e477c67288b`，relative10 特征 receipt 为 `e54a14779ca9ed693bf8ca14af6d9bc6e3298dcc53cc61e58f8d627bf9a17867`，Phase A receipt 为 `2d9f7adf4d003c6efb88417e80fb384a49cc7521c5a9c8603ee5dfd3dfc25483`。

## 数据与 PIT 修复

- 本地 top100 分钟分区继续只读复用；top50 runtime 包含 14,533,440 条 1 分钟记录和 60,556 个完整 240 分钟 stock-day。
- top50 每日权重集合为 1,212 天、60,600 个 membership keys、111 个唯一股票。下载器复用 top20 的 46 个股票，仅为新增 65 个股票执行 720 个按月可恢复请求。
- 合并后包含 127,478 条日线、126,668 条公告日股本、621 条公司行为和 60,600 条 membership；凭据只进入一次性进程，未写 SDK 状态、配置、收据或 Git。
- 全 membership 中 44 个键没有完整 240 分钟时段，严格排除后正好对应 60,556 个 runtime state；所有这些 state 的 market cap 均正且有限。
- 最终生成 59,539 个 `[8,480]` base 窗口、238,156 条四期限标签；1,203 个信号日的窗口宽度为 44–50。五折 validation 在有效标签后每个日期/期限为 40–50 只，Top10% 为 4–5 只。
- relative10 为 59,539 个 `[10,480]` float32 memmap 窗口；base8 逐位保留，只追加 amount/ADV20 与 amount/market-cap。1,040 个早期 ADV 缺口仅使用窗口内此前完整交易日 fallback；没有静态横截面秩。

## preprocessing infra 修复

原 `build_feature_windows_with_quality` 对每个样本重新扫描整张 5 分钟表，top50 通用入口与剥离基线后的专用入口分别超过 20 分钟和 30 分钟保护时限。v39 将 bars 按股票建立日期偏移索引，再直接切取 signal date 之前最后 10 个有效交易日；不改变窗口、特征或拒绝规则。

完整 top20 真实产物复验中，23,821 个 tensor、window index 和 rejection rows 均逐位/逐行完全相等，窗口构造耗时为 `1.80s`。top50 正式收据中：

| 阶段 | 秒 |
|---|---:|
| 加载 1m | 0.22 |
| 1m→5m | 122.48 |
| PIT universe | 5.96 |
| 480 步窗口 | 4.84 |
| 四期限标签 | 100.47 |
| source splits | 1.49 |
| 写产物 | 1.64 |
| 总计 | 237.11 |

因此本轮新增 infra 已把窗口构造从超过 30 分钟降到 4.84 秒；当前 preprocessing 的主要成本转为分钟聚合与标签，而不是窗口扫描。

## Phase A 效果与速度

| 指标 | Base TCN | Relative TCN | Delta |
|---|---:|---:|---:|
| mean RankIC | 0.072535 | 0.080128 | +0.007593 |
| Top return | 0.001857 | 0.002112 | +0.000256 |
| Top precision | 0.091031 | 0.099031 | +0.008000 |
| NDCG@Top | 0.564740 | 0.571631 | +0.006891 |
| Turnover | 0.531930 | 0.562690 | +0.030759 |

五折 RankIC delta 为 `+0.011966/-0.004075/-0.002127/+0.014803/+0.017397`，3/5 folds 为正。RankIC 95% CI 为 `[+0.000111,+0.015364]`，Top precision CI 为 `[+0.001125,+0.014753]`，NDCG CI 为 `[+0.000306,+0.013617]`；三者下界均高于零。Top return CI 为 `[-0.000386,+0.000864]`。

base/relative TCN median samples/s 为 `6113.06/5960.08`，吞吐保留 `0.97497`；参数量为 `6348/6476`。本轮没有运行 LSTM，因此不产生新的 TCN/LSTM 比值；此前冻结的 v35 ordinary-validation 速度证据仍为 model-step `6.1094x`、end-to-end `5.6770x`。

## turnover 定位

换手增量集中在较长期限：1d/2d 分别为 `-0.060759/-0.019241`，3d/5d 分别为 `+0.125063/+0.077975`。按 fold 看，fold 0/1/2 为 `+0.144937/+0.068354/+0.023576`，fold 3/4 为 `-0.025949/-0.057120`。这说明 top50 宽度已经修复预测尾部，不应回退特征或继续堆输入通道；下一轮应单独预注册因果的多期限换手控制，不得降低本轮 `+0.02` 门槛或把 Phase A 追认为通过。

优先候选是只作用于执行/排序稳定性的固定 horizon-aware incumbent buffer 或因果 score smoothing，并同时对 base/relative 应用；必须保留原始 RankIC 报告、显式计入成本，并验证 3d/5d 换手下降时 Top return/NDCG 不被破坏。只有新的 ordinary-validation 门禁通过，才允许多 seed 和 LSTM benchmark。

## 边界

- v39 证明的是 top50 ordinary validation 中 relative2 对 TCN 的预测增量，不是 sealed 候选、Alpha 或部署授权。
- Phase B 未运行，不能据此宣称 TCN 预测效果已经稳定优于 LSTM。
- 旧 v36 sealed identity 未读取、重放或用于选择；`sealed_test_accessed=false`。
