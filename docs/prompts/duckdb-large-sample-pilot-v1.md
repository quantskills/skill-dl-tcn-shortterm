# TCN 大样本真实分钟数据试验提示词 v1

## 角色

你是 `skill-dl-tcn-shortterm` 的量化研究工程负责人。你必须把现有的
DuckDB 小样本 smoke test 升级为可复核的大样本真实数据可训练性审计，并且
只在数据、时间切分和治理条件全部满足时运行非封存 TCN pilot。

项目仅用于离线研究。不得连接券商、执行订单、写入外部数据库、打开封存
holdout、承诺收益或宣称 TCN 比 LSTM/GRU 快 3–5 倍。

## 已知事实

- 项目根目录：当前仓库根目录
- 外部数据库：`E:\hermes-data\a_stock.duckdb`
- 数据库必须使用 DuckDB `read_only=True` 打开。
- `stocks_1m` 约有 11.83 亿行、5,294 个沪深 A 股标的，覆盖
  2000-06-09 至 2025-12-31。
- 表内至少混有两类来源谱系：`_source_file IS NULL` 的全市场级数据，以及
  带 ZIP 文件名但只覆盖少量股票的补充数据。
- 2024–2025 年满足“每日不少于 4,000 只股票”的日期有 266 天，但被切成
  106 段，最长连续完整区间只有 16 个交易日。
- MVP 的标准输入是每个标的连续 10 个完整交易日、每天 48 根标准 5 分钟条；
  标签期限为未来 1、2、3、5 个交易日。
- 现有 smoke test 只证明 DuckDB → 规范 1m → 因果 5m 链路正确，不证明真实
  TCN 可训练、Alpha、容量或速度优势。

## 最终目标

交付一个只读、确定性、可重复运行的大样本数据审计流程：

1. 统计日期级行数、标的数、平均分钟数和来源谱系比例；
2. 根据预登记阈值判定日期是否具备横截面训练资格；
3. 将相邻合格交易日组成连续区间，任何不合格日期都必须切断区间；
4. 根据 `lookback_days` 与 `max_horizon_days` 计算真正可形成标签的候选信号日；
5. 输出不可覆盖的 CSV + JSON receipt，记录源库身份、参数、摘要、连续区间、
   候选信号日和阻断原因；
6. 只有候选信号日数量、训练/验证/普通测试预算及 PIT 辅助数据都满足时，才能
   继续生成训练 manifest；否则状态必须是 `blocked`，不得训练模型；
7. 对本地真实 DuckDB 运行一次全范围审计，并把观察结果写入验证文档。

## 公共接口与测试边界

测试只能通过下列公共接口观察行为，不测试私有函数：

1. `audit_duckdb_training_coverage(...) -> TrainingCoverageAudit`
   - 输入明确的数据库路径、日期范围和质量阈值；
   - 返回日期明细、连续区间、候选信号日和总体结论。
2. `write_training_coverage_receipt(...) -> Path`
   - 写出不可覆盖的审计目录；
   - 返回 JSON receipt 路径；重复目标必须 fail closed。
3. `tasks/audit_duckdb_training_coverage.py`
   - 成功完成审计时返回 0，无论结论是 ready 还是 blocked；
   - 参数、数据库或输出契约错误时返回 2；
   - stdout 只输出机器可读 JSON 摘要。

## 日期资格契约

每个日期必须同时满足：

- `instrument_count >= min_instruments`；
- `average_bars >= min_average_bars`；
- `primary_source_ratio >= min_primary_source_ratio`；
- 日期内不存在不被声明的来源混合。

`primary_source_ratio` 定义为 `_source_file IS NULL` 的行数占该日 A 股分钟行数的
比例。该名字只描述已观察到的数据谱系，不推断其供应商质量。

日期明细必须至少包含：

- `trade_date`
- `row_count`
- `instrument_count`
- `average_bars`
- `primary_source_rows`
- `supplemental_source_rows`
- `primary_source_ratio`
- `eligible`
- `rejection_reasons`

拒绝原因必须是稳定的机器值，例如：

- `too_few_instruments`
- `too_few_average_bars`
- `too_little_primary_source`

## 连续窗口与候选信号日契约

- 按数据库中有序交易日期判断相邻性；周末和法定休市不会主动打断序列；
- 一个不合格交易日必须打断连续区间；
- 对长度为 `L` 的连续区间，可用候选信号日数量为：
  `max(L - lookback_days - max_horizon_days + 1, 0)`；
- 信号日之前必须有完整 lookback，信号日之后必须存在最长期限标签；
- 不得让窗口跨越不合格日期；
- 不得随机拆分；训练、验证和普通测试必须保持时间顺序；
- `required_signal_days = train_signal_days + validation_signal_days + ordinary_test_signal_days`；
- 候选信号日不足时，总体状态必须为 `blocked`。

## Receipt 契约

输出目录必须包含：

- `daily-coverage.csv`
- `eligible-runs.csv`
- `candidate-signal-days.csv`
- `coverage-receipt.json`

JSON 必须至少记录：

- schema version 和算法版本；
- 源数据库文件名、大小、mtime，禁止复制数据库；
- 所有输入参数；
- 总行数、标的数、日期范围、合格日期数；
- 连续区间数、最长区间、候选信号日数；
- 所需训练/验证/普通测试信号日数；
- `status: ready|blocked`；
- 稳定的 `blockers`；
- 三个 CSV 的 SHA-256；
- 由规范化内容派生的确定性 `audit_id`。

Receipt 不得包含凭据、完整外部路径、真实行情内容或源数据库哈希。对 63GB
数据库做全文件 SHA-256 会制造不必要 I/O；使用文件名、大小和 mtime 作为本轮
只读审计身份，并明确这一限制。

## 默认真实审计参数

- 日期：2000-06-09 至 2025-12-31
- `min_instruments = 300`
- `min_average_bars = 230`
- `min_primary_source_ratio = 0.95`
- `lookback_days = 10`
- `max_horizon_days = 5`
- `train_signal_days = 120`
- `validation_signal_days = 40`
- `ordinary_test_signal_days = 40`

这些参数用于判断当前本地库能否支撑最低限度的非封存 pilot，不表示最终研究
配置。若结果 blocked，不得降低阈值直到“碰巧通过”；任何阈值修改必须作为新
审计运行并保留新 receipt。

## PIT 与模型运行硬门禁

覆盖审计 `ready` 仍不等于真实 pilot readiness。运行 TCN 前还必须由现有
`check_pilot_readiness` 验证：

- PIT 上市/停牌/ST/退市状态；
- PIT 公司行动和可靠复权；
- 下一开盘成交状态；
- 数据许可和无幸存者偏差声明；
- chronological split、实际 label end purge、至少 5 日 embargo；
- Ridge、LightGBM、LSTM、GRU 与 Bai TCN 的同数据同预算比较；
- 未访问 sealed holdout。

任一条件缺失时，正确交付物是阻断收据，不是模型指标。

## TDD 执行顺序

按纵向切片执行 red → green：

1. 先写来源比例与日期拒绝行为测试，观察失败，再实现最小查询；
2. 写不合格日期切断区间及候选信号日手算测试，再实现最小规划逻辑；
3. 写 receipt 不可覆盖、指纹和确定性 ID 测试，再实现写出；
4. 写 CLI ready/blocked 与契约错误返回码测试，再实现入口；
5. 运行聚焦测试；
6. 对真实数据库执行全范围只读审计；
7. 运行 MyPy、Ruff、完整 pytest、preflight、统一测试入口和 production build。

## 停止条件

出现以下任一情况立即停止训练，但继续保存审计证据：

- 数据库不能只读打开；
- 表结构或来源字段与契约不符；
- 候选信号日少于预登记预算；
- 连续窗口跨越不合格日期；
- PIT/复权/下一开盘状态缺失；
- 输出目标已存在；
- 任一测试、类型检查、lint、preflight 或 build 失败。

## 完成定义

只有以下内容全部存在，才可声称“大样本可训练性审计已落地”：

- 完整提示词；
- 公共 Python API 和 CLI；
- 保留的 red→green 行为测试；
- 真实数据库审计 receipt；
- 验证文档中的观察证据；
- 全套工程门禁通过。

除非覆盖审计和独立 PIT readiness 同时通过，不得声称“已完成真实TCN训练”。
