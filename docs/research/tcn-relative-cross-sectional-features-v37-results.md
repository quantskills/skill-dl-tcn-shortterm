# TCN 因果相对特征与横截面扩容门禁 v37 结果

## 结论

v37 工程与真实普通验证均已完成，但候选特征被拒绝：`stop_relative_features_no_stable_gain_v37`。新增表征没有破坏 TCN 的速度优势，却同时降低 TCN 和 LSTM 的预测效果，因此本轮证据指向“特征替换方式有问题”，而不是“TCN 卷积结构单独失效”。旧 v36 sealed test 未被读取、重放或用于本轮选择。

完整执行提示词见 `docs/prompts/tcn-relative-cross-sectional-features-v37.md`。特征物化收据为 `artifacts/tcn-relative-features-top20-v37/receipt.json`，普通验证收据为 `artifacts/tcn-relative-feature-validation-v37/receipt.json`。

## 数据与因果门禁

- 基础数据：23,821 个 `[8,480]` 只读 memmap 窗口、1,203 个 signal date、每日期 17–20 只股票。
- 候选数据：23,821 个 `[13,480]` 窗口；三个价格通道、两个 amount/ADV/market-cap 相对通道、VWAP/time 通道和五个 date-level 横截面秩。
- PIT 状态：market cap 100% 可用；423 个早期样本缺少供应商 ADV20，严格使用窗口内 signal date 之前的完整交易日 amount 均值作为 causal fallback。
- 因果验证：每个 window 必须结束于自身 signal date；状态按 `(signal_date,instrument_id)` 一对一 join；未来日期变更不影响过去样本；所有值有限。
- top50：本地 top100 分钟权重覆盖 1,212 天且每天恰好 100 只。top50 共需要 60,600 个 PIT 状态键，现有增强状态缺 36,375 个，因此门禁为 `blocked_missing_pit_state`，没有用不完整 top50 训练。

## 固定实验

ordinary validation 使用 folds 0..4、seeds 7/17/27、8 epochs、batch 128、8 CPU threads。四组合使用同一样本、标签、折和 checkpoint 选择口径：

| 模型 | 特征 | mean RankIC | Top return | Top precision | NDCG@Top | Turnover |
|---|---|---:|---:|---:|---:|---:|
| base TCN | 原始 8 通道 | 0.100947 | 0.003663 | 0.1155 | 0.5641 | 0.6133 |
| relative TCN | 替换并扩展的 13 通道 | 0.065184 | 0.002893 | 0.1201 | 0.5496 | 0.7519 |
| base LSTM | 原始 8 通道 | 0.115545 | 0.002952 | 0.1252 | 0.5719 | 0.5700 |
| relative LSTM | 替换并扩展的 13 通道 | 0.083053 | 0.003365 | 0.1196 | 0.5533 | 0.7192 |

relative TCN 相对 base TCN：

- RankIC `-0.035763`，仅 4/15 seed×fold 单元为正；block-bootstrap 95% CI low `-0.044593`；
- Top precision `+0.004583`，但 NDCG `-0.014499`；
- Top return `-0.000769`，turnover `+0.138608`；
- 失败门：平均 RankIC、正单元数、NDCG、Top return、turnover、RankIC CI；
- 速度门通过：median samples/s 从 4,988.12 上升到 5,899.61，保留率 1.1827。

relative TCN 相对 relative LSTM 的 RankIC、Top return 和 NDCG 分别低 `0.017868`、`0.000471` 和 `0.003761`；Top precision 高 `0.000521`，因此仍是 mixed，而不是 TCN 效果优越。

## 速度与梯度状态

- relative TCN/LSTM 的完整训练+验证周期几何平均速度比为 `5.7286x`。
- 新增通道没有推翻 v35/v36 的 3–5×工程速度结论；本轮速度也通过候选相对基础 TCN 的 90% 保留门。
- 本轮固定 SmoothL1，用于隔离输入表征，不引入辅助 Top-tail 梯度；因此没有新的梯度冲突问题。v35 的分量梯度 cosine `+0.5143` 仍是已验证证据。

## 原因归因

13 通道候选做了两个同时发生的动作：删除原始 `log_volume/log_amount`，并把五个静态横截面秩重复到 480 个时间步。结果在 TCN 和 LSTM 上都产生近似的 RankIC 下降，说明问题具有模型无关性。普通 validation 内的单特征日度 RankIC 也显示五个秩并非一致强信号：return 1d `-0.0157`、10d volatility `-0.0379`、amount/ADV `+0.0148`、amount/market-cap `-0.0416`、market-cap `+0.0155`。这些负号理论上可以由模型学习反转，但弱信号、强相关和重复静态通道会增加不稳定性。

最可能的两个机制是：

1. 原始绝对成交规模仍携带 top20 大盘股内的流动性/状态信息，直接替换造成信息损失；
2. 将 date-level 静态秩复制到每个时间步，使序列模型把常量上下文当作 480 次时序观测，导致排名抖动和换手显著上升。

## 下一步

v38 应保持 ordinary-only，并做最小消融而不是继续搜索 TCN 结构：

1. 保留原始 8 通道，只追加 amount/ADV 与 amount/market-cap 两个时序通道；
2. 先用 seed 7×5 folds 的 TCN-only 屏幕与 base TCN 比较；通过预登记门后才运行 seeds 17/27 与 LSTM；
3. 静态横截面状态不再沿 480 步复制；若追加相对时序通道仍失败，再实现独立的 post-encoder static-context/DeepSets adapter；
4. top50 分支需要先补齐 36,375 个 PIT 状态键和公司行为状态，再以 5 只 Top10% 组合复跑。不能为了扩容绕过状态门禁。

无论下一步结果如何，都不得复用已消费的 v36 sealed test，也不得把 ordinary validation 改善宣称为可部署 Alpha。
