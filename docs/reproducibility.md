# Reproducibility

每次实验必须产生一个只追加、不覆盖的运行目录，位于配置指定且不受 Git 跟踪的位置。运行标识由解析配置、输入清单、数据指纹和可执行源码指纹（含 Git 修订与 dirty 状态）的规范 JSON 摘要确定；相同身份可在另一个输出根重放，但不能覆盖已有目录。

## Required run manifest

- 代码修订、dirty 状态和入口命令。
- 完整配置及其摘要，不含密钥或机器私有路径。
- 数据清单、schema/feature/label/universe/split 版本和指纹。
- Python、PyTorch、CUDA、驱动、CPU/GPU、内存和线程环境。
- 所有随机种子、确定性设置和已知非确定性算子。
- fold 边界、purge/embargo 数量和封存状态。
- 模型参数量、感受野、batch size、精度和早停规则。
- 成本、税费、滑点、容量和成交规则的生效版本。

## Required artifacts

- `run.json`：机器可读清单。
- `config.resolved.json`：冻结后的无秘密配置。
- `metrics.parquet`：按 fold、日期、期限、模型和情景记录指标。
- `predictions.parquet`：证券、信号日、期限、分数、标签和数据版本。
- `orders.parquet`：计划/实际成交、未成交原因和成本分解。
- `portfolio-ledger.parquet`：按事件时点记录并发 vintage、现金、暴露、换手与已实现贡献。
- `tcn-comparison.parquet`：TCN-lite、共享多期限头与单期限模型的同日配对 RankIC 差值及置信区间。
- `environment.json`：依赖、代码状态与硬件摘要。
- `report.md`：事实、失败项、假设和结论，禁止把工程通过写成 Alpha 证据。

## Replay rule

重放必须从规范数据清单重新生成窗口、预测和摘要指标。缓存可加速但不能成为唯一真源；缓存指纹不匹配时必须失败，而不是静默复用。
