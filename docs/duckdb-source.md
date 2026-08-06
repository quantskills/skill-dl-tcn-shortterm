# Hermes A 股 DuckDB 接入

## 外部数据源

本项目可只读消费所有者维护的 `a_stock.duckdb`，但数据库始终是仓库外部运行时资产。不得复制进 Git、修改表结构、执行修复、`VACUUM` 或批量写入。

当前适配器固定读取 `stocks_1m`，并把供应商字段转换为项目 `raw_1m` 契约：

- `SHSE`/`SZSE` 转换为 `.XSHG`/`.XSHE` 规范证券标识；
- 使用右端时间 `eob` 作为 `bar_end_at`；
- 输出 `open/high/low/close/volume/amount` 和 `quality_flag`；
- 仅允许沪深 A 股代码前缀，排除 B 股、基金、债券和其他证券；
- 连接强制使用 DuckDB `read_only=True`。

## 质量门禁

真实库存在部分日期只导入少数证券的情况。导出前必须按日期审计证券数量和平均分钟条数量；只有同时达到调用者预登记阈值的日期才能进入导出。阈值属于 pilot 配置和证据，不是隐含常量。

导出必须显式声明日期、规范证券标识和 `max_rows`。单次最多 31 个日期、512 只证券，且拒绝覆盖已有产物。输出 Parquet 和 manifest 位于调用者指定的未跟踪目录，manifest 记录选择范围、数据指纹、数据库文件元数据和来源表。

## 分钟时段规范化

供应商跨年份可能保存 `09:30`、`11:31` 等连续交易时段外记录，并省略收盘集合竞价期间的零成交分钟。适配器执行以下固定策略：

- 只保留右端时间位于 `09:31–11:30` 或 `13:01–15:00` 的记录；
- 只在同一证券同一交易日已有更早观察和 `15:00` 观察时，允许补齐缺失的 `14:58/14:59`；
- 补齐价格只使用此前最后收盘价，成交量和成交额为零，标记为 `auction_no_trade_fill`；
- 上午或午后连续竞价的其他缺口不补齐，后续聚合继续标记 `incomplete`；
- manifest 的 `normalization` 保存查询行数、越界丢弃数、集合竞价补齐数、未解决缺口数和最终导出数。

`auction_no_trade_fill` 是允许的确定性来源规范化标记，5 分钟聚合不会因此拒绝收盘桶；任何其他非 `ok` 来源标记仍然 fail closed。

## 全库可训练性审计

小样本导出前的日期门禁不能回答 10 日回看与 5 日标签是否连续。启动真实训练前必须运行全库只读审计：

```text
python tasks/audit_duckdb_training_coverage.py \
  --database E:\hermes-data\a_stock.duckdb \
  --output-dir D:\research-runs\tcn\coverage-v1 \
  --start-date 2000-06-09 \
  --end-date 2025-12-31 \
  --min-instruments 300 \
  --min-average-bars 230 \
  --min-primary-source-ratio 0.95 \
  --lookback-days 10 \
  --max-horizon-days 5 \
  --train-signal-days 120 \
  --validation-signal-days 40 \
  --ordinary-test-signal-days 40
```

入口输出 `daily-coverage.csv`、`eligible-runs.csv`、`candidate-signal-days.csv` 和 `coverage-receipt.json`，且拒绝覆盖已有产物。`blocked` 表示审计成功但数据不满足预登记预算，CLI 返回 0；契约或执行错误返回 2。

`primary_source_ratio` 只是当前库中 `_source_file IS NULL` 行的观察比例，不代表对供应商质量的推断。带文件名的稀疏补充来源与无文件名的全市场级来源不得被总行数掩盖。覆盖审计通过后仍必须单独满足 PIT、复权、下一开盘状态和 pilot readiness 门禁。

## 有界导出

```text
python tasks/export_duckdb_pilot.py \
  --database E:\hermes-data\a_stock.duckdb \
  --output-dir D:\research-runs\tcn\duckdb-slice \
  --start-date 2025-12-29 \
  --end-date 2025-12-29 \
  --instrument 000001.XSHE \
  --instrument 600000.XSHG \
  --min-instruments 5000 \
  --min-average-bars 200 \
  --max-rows 1000
```

得到的 `manifest.json` 可直接传给 `tasks/run_experiment.py`。一次小样本烟测只证明接入和聚合链路可运行，不构成真实训练、Alpha、封存测试或生产就绪证据。
