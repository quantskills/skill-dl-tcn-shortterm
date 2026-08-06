# Work items

状态说明：`ready` 可立即开始，`blocked` 等待依赖，`done` 已有合成或契约验收证据。全部工程实现项已完成；真实硬件与封存数据的实际运行不由合成测试替代。

| ID | Priority | Status | Blocked by | Scope | Acceptance evidence |
|---|---|---|---|---|---|
| DATA-001 | P0 | done | — | 定义规范 schema、时间戳/复权/交易状态语义和合成数据夹具 | schema 校验测试及因果时间轴测试 |
| DATA-002 | P0 | done | DATA-001 | 实现 provider adapter 协议和 1m→5m 因果聚合 | 午休、缺失条、未闭合条和指纹测试 |
| UNIVERSE-001 | P0 | done | DATA-001 | 实现沪深 A 股 PIT 标的池及排除原因 | 幸存者偏差与状态回填防护测试 |
| SAMPLE-001 | P0 | done | DATA-002, UNIVERSE-001 | 实现特征、四期限收益和横截面排名标签 | 标签索引、排名集合和 train-only 转换测试 |
| SPLIT-001 | P0 | done | SAMPLE-001 | 实现 expanding walk-forward、purge、embargo 和封存清单 | 边界重叠、封存访问和指纹测试 |
| LOADER-001 | P1 | done | SAMPLE-001 | 实现 Arrow/memmap window index 与流式 Dataset | 多 worker 正确性、RAM 与吞吐证据 |
| BASELINE-001 | P1 | done | SPLIT-001, LOADER-001 | 实现常数、Ridge 和 LightGBM 基准 | 固定 fixture 预测与指标测试 |
| BASELINE-002 | P1 | done | SPLIT-001, LOADER-001 | 实现可比 LSTM/GRU 基准 | 参数量、输入输出和训练预算对齐证据 |
| TCN-001 | P0 | done | SPLIT-001, LOADER-001 | 实现 Bai TCN、四期限头和缺失 mask | 因果性、感受野 509、残差与损失测试 |
| TCN-002 | P2 | done | TCN-001 | 实现 TCN-lite 和单期限消融 | 完整感受野与负迁移对照报告 |
| EVAL-001 | P0 | done | BASELINE-001, TCN-001 | 实现 RankIC/ICIR、分层和配对 bootstrap | 手算 fixture 与退化场景测试 |
| BACKTEST-001 | P0 | done | SAMPLE-001 | 实现多头批次组合、成交、T+1、成本和容量 | 确定性账本与压力测试 |
| PERF-001 | P1 | done | BASELINE-002, TCN-001 | 同条件比较 TCN、LSTM、GRU 性能 | 测量契约通过；真实硬件收据待运行 |
| MODERN-001 | P3 | done | TCN-002, EVAL-001 | 在独立预算内实现 ModernTCN 实验 | 同契约对照及停止条件证据 |
| REPORT-001 | P0 | done | EVAL-001, BACKTEST-001, PERF-001 | 生成可复现研究报告包 | run manifest、指标、预测、账本和摘要 |
| FREEZE-001 | P0 | done | REPORT-001 | 预登记 promotion 规则与一次性封存门禁 | 状态机通过；真实封存消费需单独授权 |
| PILOT-001 | P0 | done | REPORT-001, FREEZE-001 | 实现真实数据 pilot 的独立 readiness descriptor、fail-closed 校验器、CLI 与运行手册 | 11 个聚焦测试覆盖字段、秘密、全部声明文件指纹、切分、封存隔离、比较集合与 CLI 返回码 |
| DATA-003 | P0 | done | DATA-002, PILOT-001 | 以只读、有界方式接入外部 Hermes A 股 DuckDB，审计残缺日期并导出规范 raw_1m slice | 临时 DuckDB 公共接口测试、真实库有界烟测、完整项目测试 |
| DATA-004 | P0 | done | DATA-003 | 规范化 DuckDB 跨年份时段漂移，因果补齐已核验的收盘竞价零成交省略且未知缺口继续 fail closed | 3 个新增行为测试、2024/2025 双真实样本、完整项目与构建门禁 |
| DATA-005 | P0 | done | DATA-004 | 审计全库来源谱系、连续合格区间和可形成 10 日回看/5 日标签的候选信号日，数据不足时阻断真实训练 | 3 个公共接口测试、不可变全库 receipt `e10cf64245dbe7f5`、完整项目与构建门禁 |
| DATA-006 | P0 | done | DATA-005 | 通过 PandaData 以沪深300每日 PIT 权重前100、2021–2025、1分钟粒度建立可恢复真实数据试验并执行覆盖审计与 TCN 工程测试 | 适配器/续传/审计测试，真实覆盖收据 `5acdde65d2cb7a47`，预处理 run `20d9b8b6ea82ea72`，TCN run `0186f345a4bbe622` |
| DATA-007 | P0 | done | DATA-006, PERF-001 | 为每日 PIT top20 补齐历史日线、公告日股本、公司行为、因果 ADV20 与市值，并在相同数据/折叠/预算下执行 TCN、LSTM、GRU 真实 CPU 基准 | 480 个可恢复供应商分片、3 个新增数据行为测试、增强 run `90b9c2a4f248b208`、等价优化 run `52787a471076e1be` 与550/612参数匹配收据；三组证据均不支持 TCN 3–5× |
| TCN-003 | P0 | done | DATA-007, TCN-001 | 在不可变top20 memmap上按普通validation执行有界TCN超参数筛选、两折确认和batch速度复核，并以预登记门禁决定是否扩大top50 | 5个公共接口行为测试；权威Phase A/B fixed-seed收据；两折平均仅改善0.00148，`stop_no_validation_gain`，未扩大top50 |
| TCN-004 | P0 | done | TCN-003, PERF-001 | 修正性能基准的日度横截面RankIC口径，隔离CPU线程/模型步/完整周期，比较参数匹配TCN-lite，并用5折3seed ordinary validation检验稳定性 | 线程恢复、指标口径、派生清单和门禁测试；expanding/sliding不可变清单；TCN-lite/LSTM模型步2.905×、完整周期1.883×；效果门禁`stop_unstable_validation` |
| TCN-005 | P0 | done | TCN-004, LOADER-001 | 保持TCN方向，缓存fold级RankIC验证计划、作用域化调优线程、支持TCN-only模型子集与Bai/TCN-lite统一容量筛选，并诊断memmap worker生命周期 | position-safe数值等价测试；RankIC核心43.9×；5折3seed TCN-lite/LSTM模型步2.889×、完整周期2.520×；Bai-TCN-16相对TCN-lite-4配对中位RankIC提升0.01559 |
| TCN-006 | P0 | done | TCN-005 | 尝试以低成本正则化把TCN-lite-16的速度与效果收敛为单一Pareto候选；支持head/channel dropout、AdamW与完整配置收据 | dropout位置/掩码/JSON行为测试；v6–v8五折收据；最佳快速配置0.08892/5806sps，未同时通过0.09/5000门槛，`stop_no_pareto_gain` |
| EVAL-002 | P0 | done | TCN-006, BASELINE-002 | 显式统一 TCN/LSTM 逐样本输出契约，并同时评估全池 RankIC、Top-tail 命中、NDCG、原始收益和换手诊断 | v33 真实普通验证覆盖 186,264 条预测、3 seeds × 5 folds；历史重放误差为 0/2.78e-17；状态 `task_aligned_metrics_mixed_v33`，TCN 优越性统计门未通过；sealed test 未访问 |
| TCN-007 | P0 | done | EVAL-002, TCN-006 | 保持冻结父模型 TCN 与全部训练协议不变，只增加固定 `0.05` 权重、top10%、温度 `0.1` 的 Top-tail pairwise loss，并审计分量梯度、任务对齐效果和速度开销 | v34 真实普通验证 3 seeds × 5 folds；梯度 cosine `+0.5142`、父模型 drift `0`、TCN/LSTM `6.23x/5.76x`；candidate/control 吞吐 `0.7377x`，Top precision 微升但区间跨零、NDCG不升、RankIC与换手退化；`stop_top_tail_no_gain_v34`；sealed test 未访问 |
| TCN-008 | P0 | done | TCN-007, EVAL-002 | 对同一条固定8-epoch Top-tail TCN轨迹完整保存epoch 0..8，只比较最大RankIC selection与`best RankIC-0.002`可行域内等权Top precision/NDCG selection | v35真实普通验证135个唯一checkpoints、7/15单元改选；Top precision `+0.001458`且95% CI `[+0.000729,+0.002292]`，NDCG `+0.000566`，RankIC `-0.000269`，turnover不变；TCN/LSTM `6.11x/5.68x`；`constrained_tail_ordinary_validation_candidate_v35`；sealed test 未访问/未授权 |
| TCN-009 | P0 | done | TCN-008, FREEZE-001 | 冻结 v35 task-aligned 候选、v33 LSTM、sealed 数据身份与严格按日期可用的 checkpoint 计划，在逐字授权前只执行 metadata-only readiness | v36 复核全部父输出/源/checkpoint 哈希；canonical sealed test 3973 样本、2×100 交易日；time-safe checkpoint exposures 9+15=24、其中12个改选；状态 `awaiting_explicit_sealed_authorization_v36`，未加载 sealed 特征/标签、未创建消费标记 |
| TCN-010 | P0 | done | TCN-009 | 在用户逐字授权后按冻结协议一次性消费 sealed test，比较 candidate、RankIC control 与冻结 LSTM，按日期先聚合模型单元再 block bootstrap，并永久关闭该 sealed 数据的调参复用 | 562,113 条预测、28,800 个模型指标组、800 个市场日期/期限 paired groups；速度门与 RankIC/收益/换手容忍门通过，但 Top precision/NDCG 均值及 robust-tail 门失败；`sealed_rejected_tcn_candidate_v36`，receipt `68205700cf0a21b807c6ea8c6e5aea351a9732cda0662cf89d27a1fe2fe02db9` |
| TCN-011 | P0 | done | TCN-010, DATA-007 | 在不复用旧 sealed 的前提下，将 PIT ADV/市值与 date-level 横截面秩物化为因果相对特征，并用固定 TCN/LSTM 四组合 ordinary validation 检验表征假设与 top50 状态门禁 | 7 个新增行为测试；23,821 个 `[13,480]` memmap 窗口；top50 缺 36,375 个 PIT 状态键而 fail closed；3 seeds × 5 folds 中 relative-base TCN RankIC `-0.035763`、4/15 单元为正，速度 `5.7286x` TCN/LSTM；`stop_relative_features_no_stable_gain_v37`，receipt `091cb88ca4bcc8827ea301750c9d207b9904c95281fac66e931846eaf64cc16d` |
| TCN-012 | P0 | done | TCN-011 | 保留 base8 原始量价通道，仅追加两个因果相对成交时序通道；先以 seed 7×5 folds TCN-only 门禁决定是否授权多seed/LSTM确认 | 23,821 个 `[10,480]` memmap 窗口，base8逐位保留；候选-base RankIC `+0.002257`、4/5 folds 为正、吞吐保留 `1.0036`，但 Top precision `-0.005000`、NDCG `-0.012250`、Top return `-0.000772`，状态 `stop_append_relative_sequence_seed7_v38`；Phase B未授权，receipt `e804c897fbf26f0c0a262cca19c142b21ee87401fe92d84e73bb1e123939030f` |
| TCN-013 | P0 | done | TCN-012, DATA-007 | 增量补齐 top50 PIT 状态，等价优化窗口 preprocessing，并在更宽横截面上以 seed 7×5 folds 重训 base8/relative10 TCN | 复用46只并增量补65只股票；59,539个top50窗口，Top10%为4–5只；窗口构造由>30分钟降至4.84秒且top20逐位等价；relative-base RankIC/precision/NDCG/return均改善，但turnover `+0.030759` 超过`+0.02`门，`stop_top50_relative_sequence_seed7_v39`；Phase B未授权，receipt `2d9f7adf4d003c6efb88417e80fb384a49cc7521c5a9c8603ee5dfd3dfc25483` |
| TCN-014 | P0 | done | TCN-013, BACKTEST-001, BASELINE-002 | 冻结v39模型与数据协议，分离模型门和策略门；按证券目标权重净额化成交与双边名义额计费；执行3seed×5fold同预算LSTM确认 | 12个相关聚焦测试；relative10-base8 RankIC `+0.008374`、11/15单元为正；10bps净收益 `+0.000191`、one-way turnover `-0.008736`；TCN/LSTM模型步 `4.835×`、完整周期 `4.337×`；模型门与组合研究门通过；sealed未访问；receipt `e3613f1e30c2e6f2500af55659f553e68c8b269b452a53affd07872329a2a836` |
