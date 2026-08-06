# TCN 短线横截面项目落地方案

## 1. Outcome

首版交付一个离线研究系统：从供应商无关的本地分钟行情构造严格 PIT 的沪深 A 股样本，训练统一契约下的传统基准与 TCN 模型，执行 chronological walk-forward、保守成交回测和训练性能测量，最后生成不夸大结论的可复现报告。

首版预测产品不是“预测股价”，而是在每个信号日收盘后，为 PIT 标的池中的股票输出未来 1、2、3、5 个交易日的四个横截面收益分数。下一交易日开盘是最早执行时点。

## 2. Accepted decisions

| Area | Accepted decision |
|---|---|
| Primary task | 多标的横截面收益排序 |
| Market | 沪深 A 股普通股，动态 PIT 标的池 |
| Signal and execution | 收盘后生成信号，下一交易日开盘执行 |
| Horizons | 1、2、3、5 个交易日 |
| Model input | 5 分钟 OHLCV，10 日约 480 步基准；5/20 日消融 |
| Target | 每日每期限横截面百分位映射到 `[-1, 1]` |
| Main model | Bai 双卷积因果扩张残差 TCN，共享主干与四个输出头 |
| Receptive field | `k=3`、7 块、扩张率 1–64，感受野 509 |
| Normalization | WeightNorm；禁用 BatchNorm |
| Validation | expanding walk-forward、动态 purge、5 日 embargo、封存测试 |
| Portfolio | 前 10% 等权多头；多空只作诊断 |
| Execution | 不可买入则现金，无法卖出则延迟退出，ADV20 的 5% 容量上限 |
| Claims | 工程完成与 Alpha 有效分开；3–5× 速度仅是待测假设 |

## 3. Runtime architecture

```text
provider files
    -> provider adapter
    -> canonical Parquet/Arrow + manifest
    -> PIT universe and 5-minute aggregation
    -> features + labels + split manifest
    -> memmap window index / streaming dataset
    -> baselines and TCN family
    -> statistical evaluation
    -> conservative execution backtest
    -> reproducible report bundle
```

建议代码边界：

```text
src/skill_dl_tcn_shortterm/
  contracts/      # schema、manifest、配置验证
  ingest/         # provider adapter、1m -> 5m
  universe/       # PIT 准入和交易状态
  features/       # 因果特征与训练段拟合转换
  sampling/       # 标签、split、window index、Dataset
  models/         # baselines、tcn_lite、bai_tcn、moderntcn
  evaluation/     # RankIC、ICIR、方向和性能指标
  backtest/       # 批次持仓、成交、成本、容量
  reporting/      # 运行清单、表格和图形
  cli/            # 离线命令入口
```

各目录通过显式数据类或协议交互，不允许模型层读取供应商字段、回测结果或未来交易状态。

## 4. Canonical datasets

第一阶段应定义并验证下列逻辑数据集。精确字段在接入真实数据前通过 `DATA-001` 固化：

- `trading_calendar`：交易日、交易时段、半日市和规则版本。
- `instrument_state`：证券标识、上市区间、板块、ST/退市整理和当时可知状态。
- `bars_1m`：供应商适配后的原始分钟 OHLCV/amount，保留来源时间戳和质量标志。
- `bars_5m`：只从已结束的 1 分钟条因果聚合，午间休市不得形成虚假连续窗口。
- `corporate_actions`：复权因子及其公告/可知时间，禁止用最终复权序列回填历史特征。
- `universe_snapshot`：每个信号日的准入结果和逐项排除原因。
- `features_5m`：带版本和 as-of 时间的模型特征。
- `labels`：原始持有期收益、横截面排名标签、有效性与缺失原因。
- `split_manifest`：fold 日期、purge/embargo、封存状态和数据指纹。
- `window_index`：样本到只读分区或 memmap 切片的映射，不复制完整张量。

所有时间戳必须显式使用 `Asia/Shanghai` 或 UTC，并在 schema 中固定 bar timestamp 表示开始还是结束。训练前必须拒绝重复主键、乱序时间、跨午休聚合、非正价格和无法解释的成交量异常。

## 5. Feature baseline

首版特征应保持小而可审计：

- 5 分钟对数或简单收益、开收收益、最高最低振幅和跳空。
- 成交量、成交额及其仅基于过去数据的滚动标准化值。
- VWAP 偏离、日内累计量占比、时间位置编码。
- 只在当时可得时加入市场宽度、指数收益和行业上下文。
- 股票静态或慢变字段通过独立通道输入，不重复铺满所有时间步。

缺失值、裁剪和标准化必须在每个 fold 的训练段拟合。禁止全样本 z-score、后验复权、当日完整成交量占比或使用未来行业成分。

## 6. Models and fair comparison

### Required baselines

- 常数/历史均值基线，用于识别指标实现错误。
- Ridge：使用相同可用信息的展平或汇总特征。
- LightGBM：使用同一特征版本和四期限标签。
- LSTM 与 GRU：输入窗口、输出头、参数量级、批大小搜索预算与 TCN 对齐。

### TCN ladder

- **TCN-lite**：轻量因果卷积对照；仍必须覆盖完整输入窗口，不能沿用感受野只有 15 的玩具结构。
- **Bai TCN**：首版主模型，默认感受野 509、宽度 64、WeightNorm、Dropout 0.1。
- **ModernTCN**：在数据、基准和 Bai TCN 全部稳定后才进入实验；不得阻塞 MVP。

所有模型输出固定顺序 `[1d, 2d, 3d, 5d]`。Smooth L1 对有效标签等权求和，缺失标签用 mask 排除。应增加单期限模型消融，检查共享主干是否出现负迁移。

## 7. Data loading strategy

- Parquet/Arrow 用于分区持久化与列裁剪；数据按日期和证券保持稳定排序。
- 预处理阶段生成紧凑 `window_index`，训练 worker 根据索引读取只读 memmap 或 Arrow 切片。
- 禁止把全量分钟样本预先转成一个 `TensorDataset` 常驻内存。
- worker 独立打开文件句柄，避免 fork/spawn 下共享不可序列化对象。
- 训练段内可以 shuffle 样本，但数据集切分本身绝不能随机。
- 吞吐测试必须覆盖单 worker、多 worker、pin memory、prefetch 和不同 batch size，并记录峰值 RAM/VRAM。

## 8. Leakage and correctness tests

最低测试集应包括：

1. 改写未来输入不会改变因果层对应历史位置的输出。
2. 每个 TCN 配置的计算感受野不小于输入窗口。
3. 5 分钟聚合不跨午休、不读取未闭合的 1 分钟条。
4. 信号日收盘后的样本不能按同日收盘成交。
5. 1/2/3/5 日标签开盘索引满足已确认时间轴。
6. 当前上市状态或当前成分股不能改变历史 PIT 标的池。
7. 标准化器只拟合训练段，验证与测试修改不会改变其参数。
8. purge 能移除标签区间越界样本，embargo 和最终封存状态不可绕过。
9. 横截面标签只在同一日期、同一期限的有效样本内排名。
10. 不可买入、延迟退出、T+1、成本、容量裁剪和持仓批次有确定性场景测试。
11. memmap/Arrow Dataset 在多 worker 下不会重复、遗漏或复制全量数据。
12. 同一代码、配置、数据指纹和种子可复现预测与报告摘要。

## 9. Evaluation protocol

统计报告按期限和 fold 展示：

- Spearman RankIC、RankIC 均值、标准差、ICIR 和衰减曲线。
- 方向准确率仅作辅助诊断，不进入首版主损失。
- 分位数组合收益、单调性、换手率和横截面覆盖率。
- 行业、规模、流动性和市场状态分层结果。
- 日期块 bootstrap 的置信区间以及相对最强基准的配对差值。

经济报告分别展示 1/2/3/5 日多头组合：

- 毛收益、净收益、比较基准超额、年化波动、信息比率、最大回撤。
- 佣金、税费、滑点、未成交、延迟退出和容量裁剪的贡献。
- 单边 5/10/20 bps 压力情景。
- 多空前后 10% 仅放在诊断区并标注不可直接执行。

## 10. Performance hypothesis

TCN 是否比 LSTM 快 3–5× 必须按同一硬件、训练数据、有效样本数、精度、搜索预算和可比参数规模测量。至少记录：

- 每秒训练样本数和每 epoch 时间。
- 达到最佳验证 RankIC 的墙钟时间，而非只比较单步速度。
- CPU/GPU 型号、线程数、batch size、AMP 设置和峰值 RAM/VRAM。
- 数据加载等待占比，区分模型速度与 I/O 瓶颈。

结果不足 3×、没有优势或因 I/O 受限都必须如实报告。

## 11. Delivery phases

### Phase 0 — Contracts and synthetic evidence

固化 schema、时间语义、配置模型和微型合成数据；首先让 PIT、标签和成交场景测试失败，再实现到通过。

### Phase 1 — Canonical data and windows

实现 adapter 接口、1m→5m 聚合、PIT 标的池、特征、标签、split manifest 和 memmap window index。

### Phase 2 — Baselines and evaluator

先跑常数、Ridge、LightGBM、LSTM、GRU，验证 RankIC、回测和报告没有明显错误。

### Phase 3 — Bai TCN MVP

实现纯左填充、WeightNorm 残差块、四期限头、感受野验证和共享/单期限消融。

### Phase 4 — Conservative backtest and performance

实现批次持仓、无法成交、延迟退出、成本、容量以及同条件吞吐基准。

### Phase 5 — TCN variants

在主流程稳定后加入 TCN-lite；ModernTCN 单独设实验预算和停止条件，避免范围膨胀。

### Phase 6 — Frozen evaluation

冻结数据、配置、代码和候选规则后，仅运行一次封存测试，生成工程结论、Alpha 证据和失败项。

## 12. Promotion and stopping rules

- 工程完成：全部强制测试、基准、walk-forward、回测、性能和复现证据齐全。
- 候选模型：封存测试前已有版本化 promotion 配置，并在 RankIC/ICIR、净多头组合及性能约束上满足其中标准。
- 若 TCN 未超过最强基准，项目仍可工程完成，但报告结论必须是“未发现 TCN 增量证据”。
- 若出现 PIT 泄漏、测试集调参、成本规则缺失或结果无法复现，任何 Alpha 结论自动失效。
- ModernTCN 不得用于挽救已被反复查看的封存测试；需要新假设时必须使用新验证周期或新封存区间。

## 13. Immediate next step

从 `DATA-001` 开始：用微型合成沪深 A 股交易日构造规范 schema、时间轴和故意包含停牌/涨跌停/缺失 bar/公司行为的测试夹具。该步骤完成前不应编写 TCN 模型，因为模型无法弥补错误标签和前视数据。
