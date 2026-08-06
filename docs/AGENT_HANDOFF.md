# Agent handoff

Follow `AGENTS.md`; this is not a competing read order.

## Problem solved

创建一个 Python 深度学习与量化研究实验项目，使用 Temporal Convolutional Network 对金融分钟线进行未来 1–5 个交易日的多期限收益、方向和波动率预测；包含 Bai 2018 因果扩张残差 TCN 基线、TCN-lite 与 ModernTCN 实验对照、严格 PIT 数据契约、chronological walk-forward、purge/embargo 验证、Parquet/Arrow/memmap 大数据加载、Ridge/LightGBM/LSTM/GRU 基准、RankIC/ICIR、含成本的离线回测以及训练吞吐性能验证。项目仅用于离线研究，不连接券商，不执行订单，不写入外部系统，不承诺收益。

## Current structure

`src/`, `scripts/`, `tasks/`, `tests/`, `config/`, and `docs/` have explicit ownership.

## Program authority

`PROGRAM.md` defines durable direction and boundaries; change it only for an explicit long-term program decision.

## Verification

Canonical entrypoints are `python tasks/preflight.py` and `python tasks/test.py`; they run natively on Windows and POSIX.

## Known unknowns

Product schemas, external wiring, deployment targets, and production readiness remain pending owner definition.
