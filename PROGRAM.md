# Project program

This document is the durable direction and boundary contract for the project.

## Mission

创建一个 Python 深度学习与量化研究实验项目，使用 Temporal Convolutional Network 对金融分钟线进行未来 1–5 个交易日的多期限收益、方向和波动率预测；包含 Bai 2018 因果扩张残差 TCN 基线、TCN-lite 与 ModernTCN 实验对照、严格 PIT 数据契约、chronological walk-forward、purge/embargo 验证、Parquet/Arrow/memmap 大数据加载、Ridge/LightGBM/LSTM/GRU 基准、RankIC/ICIR、含成本的离线回测以及训练吞吐性能验证。项目仅用于离线研究，不连接券商，不执行订单，不写入外部系统，不承诺收益。

## Intended outcomes

Deliver and maintain the stated mission as an owned project with explicit contracts, observable acceptance evidence, and a safe path from scaffold to verified behavior.

## MVP boundary

首版只解决沪深 A 股的多标的横截面收益排序：使用截至信号日收盘的 5 分钟数据，在下一交易日开盘执行，并同时输出未来 1、2、3、5 个交易日的排名分数。方向仅是收益符号的辅助诊断，波动率作为独立后续模型，不进入首版损失函数。

## Scope

Own versioned source, tests, non-secret configuration contracts, documentation, and verification for this `data-quant` project in the `finance` domain.

## Non-goals

The scaffold does not authorize deployment, production readiness, external-system writes, credential provisioning, or invented product schemas. It does not claim behavior that has not been implemented and verified.

## Operating principles

Follow `AGENTS.md` for safety and execution rules. Prefer plan/read-only behavior, preserve external ownership, keep secrets and runtime artifacts out of canonical source, and record observed evidence instead of inferred success.

## Delivery path

1. Preserve the mission and boundaries in this program.
2. Define the first measurable behavior in `docs/WORK_ITEMS.md`.
3. Implement source and tests within repository ownership.
4. Run `python tasks/preflight.py` and `python tasks/test.py`.
5. Record evidence before claiming readiness or expanding scope.

## Success gates

工程完成要求规范数据契约、PIT 防泄漏、walk-forward、模型基准、含成本回测、性能测量和可复现报告均有实现与测试证据；它不代表策略盈利或 TCN 已产生有效 Alpha。

TCN 只有在配置、阈值和比较协议预先登记后，于封存测试集上按 RankIC/ICIR、净多头组合和训练性能指标优于最强基准，才可标记为候选模型。任何“比 LSTM 快 3–5×”的描述在实测前都只能称为假设。

所有权威文档必须保持一致；preflight 与项目测试必须通过；需求必须映射到实现和证据；秘密、真实行情和未授权运行状态不得提交；部署、券商连接、订单执行和外部写入始终不在本项目授权范围内。

## Change control

Update `PROGRAM.md` only when the long-term mission, scope, non-goals, operating principles, delivery path, or success gates change. Track current execution in `docs/WORK_ITEMS.md` and current facts in `CONTEXT.md`.
