# DuckDB 分钟规范化与真实窗口前置烟测执行提示词

把下面整段提示词交给实现代理时必须完整执行。目标是修复外部 DuckDB 分钟时段语义，使真实行情能够稳定形成规范 5 分钟交易日；不得借机虚构 PIT 状态、复权、公司行为、真实 TCN 训练或 Alpha 证据。

## 角色

你是 `skill-dl-tcn-shortterm` 的量化数据工程负责人。你负责将所有者维护的外部 `a_stock.duckdb/stocks_1m` 以严格只读、有界、可追溯的方式转换为仓库既有 `raw_1m` 契约，并验证到因果 5 分钟聚合边界。

## 已知观测事实

- 外部数据库约 63 GB，必须使用 `duckdb.connect(..., read_only=True)`，不得复制、迁移、修复、`VACUUM` 或写入。
- `stocks_1m` 包含 `exchange/symbol/open/high/low/close/volume/amount/bob/eob/trade_date`，`eob` 是右端时间，时区为 `Asia/Shanghai`。
- 数据跨年份存在供应商语义漂移：部分日期含 `09:30`、`11:31` 等连续交易时段之外的记录。
- 2025-12-29 的深市样本通常缺 `14:59`，沪市样本通常缺 `14:58/14:59`；相邻年份可观察到这些分钟为零成交记录，而 `15:00` 保存收盘集合竞价成交。
- 对已核验样本，上述缺口属于闭市集合竞价期间的零成交分钟省略，不应与连续竞价随机数据缺口混为一谈。
- 当前聚合器要求每个 5 分钟桶恰好 5 条，并把任何非 `ok` 来源标记为失败，导致系统性集合竞价省略使整日被误判为不完整。
- 当前真实烟测为 954 条 1 分钟记录、192 条 5 分钟记录，其中 4 条收盘桶被误判不完整。

## 本次唯一交付目标

交付 `DuckDB → 规范 raw_1m → 因果 5m` 的真实数据工程闭环：

1. 丢弃并计数连续交易时段外的供应商记录；
2. 只对预登记的收盘集合竞价零成交分钟进行确定性补齐；
3. 不补齐任何上午、午后连续竞价或其他未知缺口；
4. 为补齐行保留可审计的 `auction_no_trade_fill` 来源质量标记；
5. 聚合器把该标记视为允许的确定性规范化，而不是数据质量失败；
6. manifest 记录原始行数、丢弃行数、集合竞价补齐行数和仍未解决的缺口数；
7. 使用真实数据库重跑有界烟测，证明已核验样本形成 4 个完整的 48-bar 交易日；
8. 更新契约、工作项、追踪和验证证据。

## 公共测试边界

只测试两个公共接口：

- `export_duckdb_minute_slice`：显式日期/标的选择如何转换为规范 `raw_1m` 和 manifest；
- `aggregate_five_minute_bars`：带允许规范化标记的 1 分钟数据如何形成标准 5 分钟条。

不得测试私有 SQL 拼装、内部 helper 调用次数或 DuckDB 实现细节。

## 强制安全和数据语义

- 数据库连接强制只读，单次仍限制最多 31 个日期、512 只证券和调用者声明的 `max_rows`。
- `max_rows` 约束原始查询返回量，不能通过补齐规避上限审计。
- 只保留右端时间位于 `09:31–11:30` 或 `13:01–15:00` 的记录。
- 规范交易日预期时间网格为上午 120 分钟、下午 120 分钟，共 240 条。
- 只允许补齐 `14:58` 和 `14:59`，且该分钟必须缺失、同一证券同一交易日必须已有更早观察和 `15:00` 收盘观察。
- 补齐行的 OHLC 使用截至该分钟前最后一个已观察收盘价，`volume=0`、`amount=0`，禁止使用未来 `15:00` 价格回填。
- 补齐行 `quality_flag=auction_no_trade_fill`；真实观察行仍为 `ok`。
- 其他缺失分钟保持缺失，由既有聚合器产生 `incomplete`，不得前向填充或插值。
- 重复主键、非正价格、负成交量/成交额仍必须失败。
- manifest 必须保存导出 Parquet SHA-256；不得对 63 GB 数据库做全文件哈希，可保存文件大小、mtime、来源表和显式选择作为来源收据。
- 不生成或猜测 `is_st`、退市状态、上市日期、公司行为、复权因子、涨跌停可交易性或 PIT 标的池。
- 不打开封存 holdout，不连接券商，不推送 GitHub，不部署。

## TDD 垂直切片

严格执行以下红→绿切片，每次只增加一个可观察行为：

1. 包含 `09:30/11:31` 和收盘竞价省略的临时 DuckDB 导出后，只有 240 个规范时间点；manifest 精确记录丢弃和补齐数量。
2. `auction_no_trade_fill` 进入聚合后形成 48 条质量为 `ok` 的 5 分钟条和 1 个完整 session。
3. 缺少普通连续竞价分钟时不得补齐，聚合结果仍明确为 `incomplete`。
4. 原有有界选择、B 股排除、最大行数、CLI 和旧聚合测试继续通过。

## 实现位置

- DuckDB 规范化：`src/skill_dl_tcn_shortterm/duckdb_source.py`
- 5 分钟允许标记策略：`src/skill_dl_tcn_shortterm/market_data.py`
- CLI：`tasks/export_duckdb_pilot.py`
- 测试：`tests/test_duckdb_source.py`、`tests/test_intraday_aggregation.py`
- 契约和使用说明：`docs/duckdb-source.md`、`docs/data-contracts.md`
- 证据：`docs/verification.md`、`docs/requirements-traceability.md`、`docs/WORK_ITEMS.md`

## 真实数据烟测

使用未跟踪的新目录，不能覆盖旧收据。至少选择 2025-12-29 的：

- `000001.XSHE`
- `600000.XSHG`
- `300001.XSHE`
- `688001.XSHG`

质量门禁仍要求当日 A 股证券数和平均分钟数达到预登记阈值。记录原始/导出行数、规范化统计、5 分钟总数、完整 session 数、质量标记分布和不可变 run ID。

如果时间允许，再选一个 2024 年含 `09:30/11:31` 额外记录的日期做第二个烟测，证明跨年份语义漂移已被规范化；仍不得宣称完成真实训练。

## 完整验证

依次执行并记录：

```text
python -m pytest -q tests/test_duckdb_source.py tests/test_intraday_aggregation.py
python -m mypy
python -m ruff check .
python tasks/preflight.py
python tasks/test.py
python -m build
```

## 完成定义

- 新测试经历可观察的 red→green；
- 真实数据库始终只读，未产生外部写入；
- 2025 样本的系统性收盘竞价省略不再导致错误的不完整 session；
- 未知连续竞价缺口继续 fail closed；
- 所有门禁和构建通过；
- 最终报告明确：这里只完成真实分钟规范化和 5 分钟聚合证据，PIT 状态、复权、公司行为、标签、真实 TCN 训练、封存评估和 Alpha 仍未完成。

## 停止条件

出现以下任一情况立即停止并报告：

- 实际缺口不能由上述预登记集合竞价规则解释；
- 需要使用未来价格补齐；
- 需要猜测 PIT、复权或公司行为字段；
- 真实库需要写入、修复或迁移；
- 用户已有修改与本任务发生不可绕开的冲突；
- 完整验证失败且无法在当前授权范围内修复。
