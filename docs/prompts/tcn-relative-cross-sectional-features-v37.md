# TCN 因果相对特征与横截面扩容门禁 v37

你是 `skill-dl-tcn-shortterm` 的量化研究、时序建模与验证执行者。当前项目已经完成 TCN CPU infra、感受野、因果卷积、Weight Normalization、memmap、梯度冲突与多指标 checkpoint selection；v35 在普通验证通过，但一次性 sealed v36 相对 LSTM 被拒绝。现在不得再读取、重放、调参或解释旧 sealed 明细来选择候选。本轮目标不是更换 TCN，而是检验一个可证伪的数据表征假设：当前 8 通道绝对量价特征没有充分表达股票之间的相对状态，可能导致 TCN 学到个股尺度而不是横截面排序信号。

## 一、不可变目标与边界

1. TCN 仍是主模型；LSTM 仅作为同数据、同折、同 seed、同 epoch 的 benchmark。
2. 只使用普通 `train/validation`，拒绝任何 `test`、`sealed=true` 或旧 sealed 消费目录。
3. 保留因果卷积、Weight Normalization、感受野至少覆盖 480 个 5 分钟步、float32、只读 memmap、`num_workers=0` 与固定 CPU 线程。
4. 不再搜索通道数、层数、学习率、loss 权重或 checkpoint 规则。本轮只改变输入表征；模型与训练协议固定。
5. 不宣称 TCN 必然优于 LSTM，不把理论 3–5× 当承诺。速度结论沿用已验证的 v36 合同，并检查新增通道没有造成明显吞吐回退。
6. 不连接券商、不交易、不部署、不写入凭据。所有产物必须带 SHA-256、代码身份、配置、环境和 `sealed_test_accessed=false`。

## 二、事实基线

- 基础运行目录有 23,821 个 10 日窗口、1,203 个 signal date、43 个历史成分股；每个日期 17–20 只，绝大多数为 20 只。
- Top10% 在当前横截面中固定等于 2 只，尾部指标方差很大；原始分钟行数多不等于横截面有效样本多。
- 现有特征只有 `close_return/open_close_return/intrabar_range/log_volume/log_amount/vwap_deviation/time_sin/time_cos`。
- PIT universe 已含 signal-date 可用的 `market_cap` 和 `adv20`；`market_cap` 完整，早期 `adv20` 少量缺失。
- 本地 top100 分钟分区完整到 2025-12-31，但增强状态只覆盖 top20。top50 必须先通过状态覆盖门禁，不能用缺失状态强行训练。

## 三、唯一允许的候选表征

从基础 8 通道生成固定的 `causal-relative-cross-sectional-v37` 13 通道：

1. 保留三个无量纲价格通道：close return、open-close return、intrabar range。
2. 用 `log1p(5m amount / signal-date ADV20)` 替代绝对 log volume。
3. 用 `log1p(5m amount / signal-date market cap)` 替代绝对 log amount。
4. 保留 VWAP deviation、time sin、time cos。
5. 增加五个在同一 signal date、PIT eligible universe 内计算并缩放到 [-1, 1] 的静态横截面秩，沿时间轴重复：
   - 最近 1 日 close return；
   - 10 日 realized volatility；
   - 最近 1 日 amount / ADV20；
   - 最近 1 日 amount / market cap；
   - market cap。
6. `adv20` 缺失时，只允许使用窗口内、signal date 之前的完整交易日 amount 均值作为有记录的 causal fallback；不得用未来日期、全样本均值或标签填充。
7. 每个样本的 `window_end_at` 必须落在自身 `signal_date`；状态 join 必须是 `(signal_date, instrument_id)` 一对一且 eligible。所有候选值必须有限。

## 四、top50 fail-closed 门禁

从本地 top100 PIT weight 每日按 `weight desc, instrument_id asc` 固定选前 50。检查每个所需 `(trade_date, instrument_id)` 是否具有可用的 signal-date PIT market cap/state。输出所需键数、缺失状态键数、缺失市值键数、日期范围和状态。只在 1,212 天每天恰好 50 只、状态覆盖完整时才返回 `ready_top50`；否则返回 `blocked_missing_pit_state`，本轮继续 top20，不发起外部下载也不读取 sealed。

## 五、预登记实验

使用同一 ordinary split manifest、fold 0..4、seed 7/17/27、8 epochs、batch 128、CPU 8 threads。比较四个固定组合：

- `base_tcn`：基础 8 通道 + 固定 dynamic-horizon-skip TCN；
- `relative_tcn`：候选 13 通道 + 同一 TCN；
- `base_lstm`：基础 8 通道 + 参数规模固定的 LSTM；
- `relative_lstm`：候选 13 通道 + 同一 LSTM 协议。

TCN 使用 `channels=16, kernel=3, dilations=1..128, chomp causal padding, Weight Normalization, dropout=0, SmoothL1, lr=0.003`。本轮用无辅助梯度冲突的固定 SmoothL1 隔离表征效应；若表征通过，后续才允许把它接回 v35 的 Top-tail/checkpoint 合同。LSTM 使用 hidden size 34、SmoothL1、lr=0.003。每个模型只能基于本折 validation RankIC 选训练期 checkpoint，禁止跨折或跨 seed 选择。

必须同时输出：

- 按 seed/fold/horizon 的 RankIC；
- Top10% return、excess return、precision、NDCG、turnover；
- `relative_tcn-base_tcn`、`relative_lstm-base_lstm`、`relative_tcn-relative_lstm` 的配对差；
- 以 signal date 为 block 的 bootstrap 置信区间；
- 参数量、训练耗时、samples/s；
- 特征质量、fallback、横截面宽度与 top50 readiness 收据。

## 六、v37 决策门禁

只有 `relative_tcn` 相对 `base_tcn` 同时满足下列条件，才标记 `relative_features_admitted_v37`：

1. 15 个 seed×fold 单元平均 RankIC delta >= 0.002，至少 9/15 单元为正；
2. mean Top precision delta >= 0，mean NDCG delta >= 0；
3. mean Top return delta >= -0.0001，mean turnover delta <= 0.02；
4. RankIC block-bootstrap 95% CI low >= -0.002；
5. TCN median samples/s 不低于基础 TCN 的 90%；
6. 15/15 单元、4 个 horizon、普通 validation、非 sealed、有限指标和 checkpoint replay 全部完整。

若任一条件失败，状态为 `stop_relative_features_no_stable_gain_v37`。失败只否定这组相对特征，不否定 TCN 项目；下一步转向补齐 top50 PIT 状态和显式 date-level cross-sectional adapter，而不是继续从旧 sealed 调参。

`relative_tcn` 对 `relative_lstm` 只作 benchmark 报告，不作为表征准入硬门槛。TCN 可以在不同指标上与 LSTM 混合胜负；必须如实报告，不能用单一 RankIC 覆盖尾部和收益结果。

## 七、落地与验证

1. 先写单元测试：尺度不变性、未来日期变更不影响过去、ADV causal fallback、缺失 PIT state fail-closed、top50 状态覆盖门禁。
2. 实现 chunked memmap 物化，禁止一次性加载全量分钟数据；写 feature manifest、audit parquet、readiness JSON 和 receipt。
3. 执行 focused tests、Ruff、Mypy、完整 pytest、preflight、统一测试入口和 production build。
4. 再执行真实 ordinary validation。实验目录不可覆盖；中断恢复只能验证已有 checkpoint 哈希后继续。
5. 最终更新结果文档，明确：实际执行范围、所有门禁值、TCN/LSTM 多指标差、top50 阻塞项、sealed 未访问、下一步是否授权。

按以上提示词直接实施。任何数据、哈希、协议或覆盖不满足时必须 fail closed，不得降低门槛、改搜索空间或读取旧 sealed 寻找有利结果。
