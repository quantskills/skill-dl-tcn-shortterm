# Verification

## TCN V46 效用对齐与一次性独立外推（2026-08-06）

V46 在读取结果前冻结评测语义、2025-04-07 至 2025-12-23 独立窗口、9 个 fold-4 检查点以及 student-vs-control 外推门和 student-vs-LSTM 非劣—速度门。窗口晚于旧 sealed test 并经过 embargo，fold-4 训练统计最晚到 2024-06-11；177 个交易日、708 个日期—期限组和 33,674 个有效标签一次性消费，receipt id 为 `364eebcfad46b6935c1a63f49788598ba24625455ea09ea593cc78614e3402a8`。

语义测试证明 `top_membership_precision`、`top_positive_return_rate` 和 `top_above_cross_section_mean_rate` 互不等价；旧 `top_precision` 仅作为集合重合率兼容别名，且 membership 不进入 V46 blockers。正式结果为 `v46_student_not_generalized`：student 相对 control 的 RankIC delta 为 `+0.000839`，但 RankIC CI low `-0.006556`、Top excess-return delta `-0.000668`、NDCG delta `-0.001701`，且只有 2/4 期限 RankIC 为正。student 相对 LSTM 的三项非劣置信下界和 `4.8789x` 训练 model-step 速度门全部通过。因此 TCN 架构与速度路径保留，V42 共识蒸馏的跨时期增益被否定；不授权 Alpha、部署或交易。完整解释见 `docs/research/tcn-v46-utility-aligned-independent-test-results.md`。

最终工程验收通过：Mypy 检查 184 个源文件无错误，Ruff 全库无错误，preflight 与 `python tasks/test.py` 通过，完整测试集合为 288 项；`python -m build --no-isolation` 成功生成 sdist 与 wheel。wheel 共 63 个成员，包含 V46 模块且不含 artifacts、checkpoints、Parquet、NPY、PT、`.env` 或凭据。

## TCN v40 模型—策略责任边界与多种子确认（2026-08-05）

v40 冻结 v39 的网络、数据、标签、fold 和8-epoch训练预算，只修正评估责任边界与证券级成交核算。seed 7 Phase A 将 Top10% membership churn 降级为模型诊断，并以目标权重变化净额重放组合：模型门和组合研究门均通过，10bps `raw_topk` 的净收益均值差为 `+0.000171`，one-way turnover 差为 `-0.008088`。

Phase B 在普通 validation 上完成3 seeds×5 folds的 relative10/base8 TCN 与同预算 LSTM 确认。relative10 TCN相对base8 TCN的RankIC为 `+0.008374`，11/15单元为正，95% bootstrap CI下界 `+0.003407`；Top precision/NDCG/Top return均改善。相同CPU 8线程、batch 128、float32、8 epochs下，15个配对单元的TCN/LSTM模型步速度比几何均值为 `4.8354×`，完整训练—验证周期为 `4.3372×`。TCN相对LSTM的RankIC为 `-0.009409`、Top precision为 `+0.003792`，预测效果属于mixed而非TCN全面胜出。

组合门覆盖3 seeds×5 folds×4 horizons共60个单元；10bps `raw_topk` 中relative10相对base8的净收益差为 `+0.000191`，one-way turnover差为 `-0.008736`。状态分别为 `top50_relative_model_multiseed_admitted_v40` 与 `portfolio_research_admitted_v40`，receipt为 `e3613f1e30c2e6f2500af55659f553e68c8b269b452a53affd07872329a2a836`。`sealed_test_accessed=false`，证据上限仍是ordinary-validation research evidence；不等于Alpha-ready、真实容量或部署授权。完整报告见 `docs/research/tcn-v40-model-strategy-boundary-results.md`。

最终工程验收通过：Mypy检查159个源文件无错误，Ruff全库无错误，`python tasks/preflight.py`成功，`python tasks/test.py`成功，直接运行pytest为266 passed；`python -m build --no-isolation`成功生成可读的wheel和sdist。隔离构建只在临时环境下载`setuptools`时超时，未进入项目构建阶段。

## Current evidence

`python tasks/preflight.py` 与 `python tasks/test.py` 当前通过266个测试，覆盖 1m→5m、PIT、窗口/标签、封存不回流的 walk-forward、训练前只读 memmap 转换、多 worker 惰性加载与规模化 RSS 观测、统计与神经模型、Bai TCN/TCN-lite/ModernTCN、普通validation-only早停与扩容门禁、缓存RankIC计划与position safety、TCN head/channel dropout及AdamW配置、同日配对消融、并发 vintage 账本、证券级净额成交换手、PIT 等权可执行基准、跨生效期成本、A 股成交、性能测量、线程作用域、五折稳定性清单与速度/效果分离门禁、`engineering_complete=True` 主入口、证据摘要、Promotion 状态机、真实数据 pilot readiness、只读 DuckDB 数据适配、PandaData 分区接入、PIT 日线/股本/公司行为增强、分钟时段规范化及连续覆盖审计。这些证据不等于现实 Alpha、容量、生产就绪或通用 3–5× 速度结论。

## 真实 DuckDB 有界烟测

2026-08-01 使用外部 `a_stock.duckdb` 的 `stocks_1m` 表执行只读烟测。日期质量门禁接纳 2025-12-29；显式选择四只沪深 A 股后导出 954 条 1 分钟记录，并由现有实验入口生成 192 条 5 分钟记录，其中 188 条完整、4 条不完整。不可变运行 ID 为 `5b80debd2b3fd2f8`，产物保存在被 Git 忽略的 `artifacts/duckdb-smoke-20260801/`。

该证据确认 DuckDB→规范 Parquet→因果 5 分钟聚合链路可运行，也暴露出供应商分钟缺口；它没有提供 PIT 状态、复权、标签、真实 TCN 训练或 Alpha 证据。初次完整校验通过 Ruff、Mypy（47 个源文件）、57 个 pytest、项目 preflight、统一测试入口以及 wheel/sdist production build；分钟规范化完成后的最终测试总数见本页顶部。

随后按 `docs/prompts/duckdb-canonicalization-real-smoke-v1.md` 完成时段规范化。2025-12-29 新收据查询 954 行，因果补齐 6 个收盘竞价零成交分钟，导出 960 行；四只股票各 240 行，生成 192 条全部为 `ok` 的 5 分钟条和 4 个完整 session，不可变运行 ID 为 `a30bb244041740d4`。

2024-01-02 跨年份收据查询 949 行，过滤 6 条 `09:30/11:31` 越界记录并补齐 3 条集合竞价记录；三只股票形成完整 session。`688001.XSHG` 仍有 14 个非竞价缺口，因此保留为不完整，证明未知缺口没有被静默插值。运行 ID 为 `40e7d20fed4eea50`。

## 真实 DuckDB 全库可训练性审计

2026-08-02 按 `docs/prompts/duckdb-large-sample-pilot-v1.md` 对外部数据库执行 2000-06-09 至 2025-12-31 全范围只读审计。A 股过滤后扫描 1,182,657,251 行、6,194 个日期，峰值横截面为 5,168 只股票；数据库没有被复制、修改或计算全文件 SHA-256。

在每日不少于 300 只股票、平均不少于 230 根分钟条、`_source_file IS NULL` 行比例不少于 95% 的预登记门槛下，1,469 天合格，但被分为 718 个连续区间，最长只有 17 个交易日。结合 10 日回看和最长 5 日标签后，只得到 5 个候选信号日，低于训练 120 日、验证 40 日、普通测试 40 日合计 200 日预算。审计状态为 `blocked`，blocker 为 `insufficient_candidate_signal_days`，不可变 audit ID 为 `e10cf64245dbe7f5`。

只有 2004-11-03 至 2004-11-24 和 2005-01-07 至 2005-01-31 两段产生候选信号日。该收据证明本地库总量很大但不满足当前 MVP 的连续大样本训练条件，因此没有启动 TCN，也没有通过降低阈值制造通过结果。运行产物位于 Git 忽略的 `artifacts/duckdb-training-coverage-20260802-v1/`。

## PandaData 五年 PIT 数据与 TCN 工程烟测

- 原始分区：`2021-01-01..2025-12-31`，沪深300每日当日权重前100，300个哈希分片，29,071,440条1分钟记录。
- 覆盖收据：`5acdde65d2cb7a47`，1212个交易日、121,131个完整股票日、69个不完整股票日；每日完整证券最低90只；10日回看和5日标签得到1198个候选信号日，门禁为 ready。
- 五年预处理：run `20d9b8b6ea82ea72`，top20运行切片产生1,162,800条5分钟记录、24,225个完整会话、23,838个有效窗口，缺损5分钟条为0。
- Bai TCN烟测：run `0186f345a4bbe622`，单fold、单epoch、8通道、509步感受野；11,702个训练样本，训练11.0秒，约1064样本/秒。
- 验证RankIC：TCN在1/2/3/5日分别为0.0285/0.0576/-0.0109/-0.0120；LightGBM为0.0812/0.0849/0.0588/0.1480。TCN没有增量证据，尤其5日配对差值置信区间完全为负。
- 限制：运行状态由PIT指数成员和完整会话派生，未补齐历史ST、公司行动、行业、市值与ADV20；未运行成交成本、真实硬件三模型速度对比或封存测试。因此该证据只证明真实数据工程链路和训练链路可运行。

## PandaData PIT 增强与三模型真实 CPU 基准

- 增强范围：每日 PIT top20 五年并集46个供应商代码，按自然月、每批最多25只、4类接口形成480个原子分片；规范结果包含52,116条历史日线、51,041条公告日股本、251条除权事件、42个供应商停牌日和24,240条PIT成员关系。认证前后均未生成 `user.json`。
- 供应商异常：2025-08-14至2025-08-26的权重数据将 `000333/000651/000858` 错标为沪市后缀，共15个成员日；对应分钟与日线均不存在，既有完整会话门禁将其排除，未做猜测性代码纠正。
- PIT 状态：24,225个有完整分钟会话的证券日均有公告时点市值；23,415个证券日具备满20日因果ADV20；17个会话因停牌状态被排除。行业因无历史生效时间保持 `unavailable`，上市日明确只代表有界数据首次观察日。
- 公司行为：标签总计95,284条，其中92,559条有效；2,080条跨未复权公司行为按契约作废，645条因未来数据不足作废。没有把未验证的复权因子冒充已复权分钟价。
- 公平基准：run `90b9c2a4f248b208`，2折、3 epochs、batch 128、float32、同一CPU/loader。TCN相对LSTM与GRU的两折几何平均吞吐分别为0.879×与0.951×。该旧版性能收据中的预测相关字段把日期和期限扁平后计算Spearman，现已明确降级为legacy diagnostic，不再作为严格RankIC证据；严格效果结论以v3调优和v4五折收据为准。
- 等价复跑：将PIT状态查找改为按证券 `merge_asof` 后，run `52787a471076e1be` 的 universe、labels、split manifest、window index 与前一运行字节级一致，RankIC与协议完全一致。总耗时902.8秒，对比913.7秒仅改善约1.2%；本轮TCN/LSTM与TCN/GRU吞吐比为0.950×与1.095×，说明wall-clock存在系统负载波动，但两轮都明确否定3–5×。
- 参数匹配复核：直接复用优化运行的只读memmap、标签与split，将TCN从8通道/2,948参数降至3通道/550参数，接近LSTM的612参数和GRU的468参数。默认20线程下两折几何平均TCN/LSTM吞吐仍仅0.765×，TCN/GRU为0.978×；该结果后来被v4线程探针定位为小型卷积过度并行，并由严格日度RankIC、分账计时的新收据取代。旧收据保留在忽略目录 `artifacts/pandadata-five-year-sequence-benchmark-param-matched-v2/` 作为原因链证据。
- 结论：增强数据提高了PIT可信度并实测了性能假设，但TCN没有超过LSTM或既有LightGBM证据；未触碰封存测试，不升级为候选模型或Alpha结论。

## TCN 普通验证调优与样本扩容门禁

- 协议：`docs/prompts/tcn-validation-optimization-and-scale-gate-v3.md` 固定6个Phase A配置、相同fold内共享seed、严格按 `signal_date × horizon` 计算Spearman RankIC、validation-only早停、两折确认和0.01最小提升门槛。任务入口复算四个不可变预处理产物哈希，显式删除test行并拒绝覆盖。
- TDD证据：计划验证会拒绝不足感受野和重复ID；手算样例证明RankIC不跨日期/期限混合；相同trial配置共享同一fold seed；修改test特征不改变leaderboard；字面量收据验证确定性排序与top50 fail-closed门禁。
- Phase A权威收据：`artifacts/pandadata-tcn-tuning-phase-a-v3-fixed-seed/`。`k3-c16-lr3e3` 在fold 0达到0.157925，控制组为0.127013；单折改善0.030912。其参数量11,028、纯训练吞吐约801样本/秒；控制组2,948参数、约946样本/秒。
- Phase B权威收据：`artifacts/pandadata-tcn-tuning-phase-b-v3-fixed-seed/`。控制组fold 0/1为0.145058/0.084359；最优 `k3-c16-lr3e3` 为0.157925/0.074452。候选只在fold 0改善，fold 1退化；两折平均相对控制组仅提升0.001480，低于0.01门槛，尽管3个期限不退化，最终状态仍为 `stop_no_validation_gain`。
- 扩容结论：门禁失败后没有物化或下载top50增强数据。这说明当前调优能改善单一时期，但没有跨时期稳定性；在此证据下继续扩大横截面不符合预登记成本规则。
- 同seed batch速度收据：`artifacts/pandadata-tcn-speed-batch-sweep-v3-fixed-seed/`。batch 128/256/512的两折几何平均纯训练吞吐分别约755/796/894样本/秒，完整训练—验证周期约712/748/830样本/秒；相应两折平均最佳RankIC为0.1077/0.1019/0.0834。batch 512只比128提升约18%纯训练吞吐，同时显著降低验证效果。
- 结论：TCN确有未调优空间，但它不是当前主要缺口的完整解释。更大的TCN改善了一个折、未改善另一个折，并进一步降低速度；当前证据既不支持top50扩容，也不支持3–5×或候选Alpha结论。

## TCN CPU线程、TCN-lite与五折三seed稳定性

- 协议：`docs/prompts/tcn-cpu-throughput-and-stability-v4.md` 预登记CPU线程探针、模型步/训练管线/完整周期分账、严格日度横截面RankIC、参数匹配TCN-lite、5折ordinary validation、expanding/sliding筛选、三seed确认和速度/效果分离门禁。
- 原因探针：输入固定为`[128, 8, 480]`。20线程下LSTM/Bai-TCN/TCN-lite纯模型步约1664/1044/1450样本每秒，Bai-TCN/LSTM仅0.63×；8线程时约5706/11255/18525，Bai-TCN/LSTM约1.97×、TCN-lite/LSTM约3.25×。这证明旧速度失败的主要工程原因是本机20线程对小型Conv1d过度并行，但探针不替代真实memmap收据。
- 派生清单：只使用原始fold 1的ordinary `train + validation` 行，以400个初始训练日、80个验证日、5折构造。expanding指纹为`4917321f7398329eef967381b1454d343fd968bb3122bdeddc52a14521a9f800`，sliding指纹为`88d8b32d0f54a6cd5377028d6205883c0fd37b6e7bc7c3f3ea6ec27c728d7e3d`；每折动态purge 124–138个标签重叠样本，test/sealed样本为0。
- 窗口筛选：seed 7、3 epochs、8 threads下，expanding的TCN-lite RankIC中位数/最差折为0.03315/0.01562，优于sliding的0.02788/0.00280，因此按预登记顺序选择expanding。expanding单seed中TCN-lite/LSTM模型步与完整周期分别为2.318×和1.646×。
- 三seed权威收据：`artifacts/tcn-cpu-stability-expanding-three-seed-v4/` 完整覆盖4模型×5折×3seed共60个单元。TCN-lite/LSTM模型步几何平均为2.905×，完整周期为1.883×；Bai-TCN/LSTM分别为1.921×和1.518×。本机特定配置的CPU速度门禁为`cpu_end_to_end_speedup_confirmed`，但2.905×未达到预登记3.0×陈述阈值，更不能外推为通用3–5×。
- 效果收据：LSTM、GRU、Bai-TCN、TCN-lite严格RankIC中位数分别为0.09968、0.09041、0.04337、0.04325。TCN-lite 15个fold-seed单元全部为正，按seed平均的最差fold为0.02094；但它低于LSTM，且相对Bai控制组中位数为-0.00013，未达到+0.005门槛。
- 最终门禁：`artifacts/tcn-cpu-stability-gate-v4/decision.json` 为`speed_status=cpu_end_to_end_speedup_confirmed`、`effect_status=stop_unstable_validation`、`model_step_three_x=false`。结论是特定CPU配置下TCN并行优势已经恢复，但没有稳定预测增量；未读取test/sealed test，未扩容top50，未产生候选Alpha或部署授权。
- 工程门禁：Ruff通过；Mypy通过66个源文件；完整pytest为85 passed；preflight与`python tasks/test.py`通过。隔离build在安装`setuptools>=69`时300秒超时，随后使用已安装且满足声明的构建后端执行`python -m build --no-isolation`，成功生成wheel与sdist；该超时没有被记录为隔离构建成功。

## TCN v5 infra缓存与合理容量复核

- 协议：`docs/prompts/tcn-infra-and-capacity-v5.md` 固定项目方向为TCN，LSTM/GRU仅作测量参照；继续使用expanding普通验证五折和seeds `7/17/27`，不读取test/sealed test。
- 根因分账：旧v4收据中TCN-lite每个运行单元平均模型步2.051秒、验证1.983秒、完整周期4.464秒，验证固定成本占44.4%。真实fold 0上旧标签联接/Pandas分组RankIC约0.625秒/次。
- 缓存实现：`ValidationRankICPlan` 每fold一次解析sample position、标签、日期/期限组和目标秩；epoch只对变化的模型分数排名。包含ties、invalid label与乱序position的测试证明数值等价，并在position不一致时fail closed。真实探针为约0.014秒/次，核心计算约43.9×；批量memmap fancy-index探针反而慢约12%，因此没有进入默认路径。
- 正式速度收据：`artifacts/tcn-infra-cached-expanding-three-seed-v5/` 覆盖4模型×5折×3seed。4线程下TCN-lite/LSTM模型步几何平均为2.889×、完整周期为2.520×；Bai-TCN/LSTM分别为2.088×和1.919×。TCN-lite验证占比降至5.8%。单折探针曾达到3.349×，但跨折三seed没有达到3.0×模型步陈述阈值，因此仍不得宣称通用3–5×。
- worker诊断：双worker先把训练data-wait由约0.46秒降至0.17秒，但每epoch重建validation loader使验证膨胀到约15.2秒；复用fold级validation loader后，后续模型验证恢复约0.16秒。首个worker初始化仍约4.85秒，超过当前2.38万样本、3 epochs节省的数据等待，因此本规模默认`num_workers=0`，双worker仅保留为显式更大规模选项。
- 容量筛选：`artifacts/tcn-capacity-screen-expanding-seed7-v5/` 在同五折上比较TCN-lite channels 4/8/16和Bai-TCN channels 8/16，最多8 epochs、patience 2。seed 7中Bai-TCN-16平均RankIC 0.09419，TCN-lite-4为0.07575；5/5折均为正。
- 多seed确认：`artifacts/tcn-capacity-confirm-expanding-seed17-v5/` 与 `artifacts/tcn-capacity-confirm-expanding-seed27-v5/` 的候选相对控制平均改善为0.01112和0.02983，均为4/4期限不退化、`expand_top50`。合并seed 7后，Bai-TCN-16与TCN-lite-4的15单元中位RankIC为0.09409/0.06708，配对中位差0.01559、平均差0.01980、候选赢11/15、15/15为正；三个seed平均改善均为正。
- 解释：旧TCN效果弱不是“样本太少”或“TCN无效”的单一结论。已确认的工程因素包括过度线程、重复验证固定成本、3/4通道压缩和3 epochs训练截断。当前最快配置是TCN-lite-4，普通验证效果最强配置是Bai-TCN-16；尚未证明一个配置同时具备2.5×端到端速度与最佳RankIC，也没有封存测试、Alpha、top50物化或部署证据。

## TCN v6–v8 单模型Pareto收敛

- 红灯定义：从`artifacts/tcn-capacity-screen-expanding-seed7-v5/leaderboard.parquet`快速重放，要求同一TCN同时达到mean RankIC 0.09和训练吞吐5000样本/秒。既有Bai-TCN-16为0.09419/997，TCN-lite-16为0.09197/1694，TCN-lite-4为0.07575/11242，没有单一Pareto候选。
- 结构负探针：相同`[128,8,480]`、4线程下，因果深度可分离TCN-16将参数6228降至2988，但模型步从约3997/s降至3073/s；本机CPU depthwise Conv1d效率不足，因此throwaway结构没有进入产品代码。
- 执行与dropout根因：TCN-lite-16微探针的4线程/batch128约4147/s，8线程/batch128约5916/s，12线程/batch256约6159/s。block element dropout 0.1在8线程下把吞吐降至约1665/s，而dropout 0约5916/s；瓶颈来自对8个块的完整480步激活生成并应用随机掩码。
- v6收据：`artifacts/tcn-pareto-screen-expanding-seed7-v6/`。element block dropout、无dropout、head dropout与12通道head dropout的mean RankIC/吞吐分别为0.09180/1961、0.08859/5980、0.08598/5933、0.07657/7206。效果通过者速度失败，速度通过者效果失败。
- v7收据：`artifacts/tcn-channel-dropout-expanding-seed7-v7/`。时间共享channel dropout在微探针约6062/s，真实五折为0.08329/5477，虽是element dropout吞吐的2.79倍，但低于无dropout效果，因此未进入额外seed。
- v8收据：`artifacts/tcn-weight-decay-expanding-seed7-v8/`。AdamW weight decay `1e-5/1e-4/1e-3`分别得到mean RankIC 0.08892/0.08639/0.08875，吞吐5806/5940/6028；仍没有同时达到0.09/5000门槛。
- 最终状态：`stop_no_pareto_gain`。head dropout、channel dropout与非零weight decay已作为默认兼容的显式实验能力落地，但v6–v8没有候选获得seeds17/27或sealed test资格。下一阶段转向TCN内部多尺度skip聚合与多期限负迁移研究；不得把本轮失败解释为放弃TCN。

## TCN/LSTM 任务对齐多指标评测 v33

- 协议：`docs/prompts/tcn-task-aligned-multimetric-evaluation-v33.md`。本轮只新增统一预测/评测契约，没有同时修改模型、损失、优化器、数据、split 或训练预算。
- 真实证据：`artifacts/tcn-task-aligned-evaluation-v33/`，覆盖 2021–2025 数据派生的普通 validation、3 seeds × 5 folds、TCN/LSTM 各 93,132 条逐样本预测；两模型样本键、标签和评测契约完全一致。
- 复现：LSTM 对 v32 历史最佳 RankIC 的最大误差为 `0.0`；TCN checkpoint 重放最大误差为 `2.78e-17`。运行曾在 14/15 单元后触发命令时限，恢复路径只读验证已落盘 checkpoint 并训练缺失单元；receipt 明示 epoch history 不完整。
- 结果：TCN/LSTM mean RankIC 为 `0.099791/0.115545`；TCN-LSTM Top-return 差为 `+0.000028`，95% CI `[-0.000722,+0.000769]`；Top precision 差为 `-0.011042`，NDCG 差为 `-0.007082`，后二者置信区间均低于零。形式状态为 `task_aligned_metrics_mixed_v33`，但 `tcn_prediction_effect_passed=false`。
- 边界：RankIC 对共同的全池排序输出是合理指标，但不足以代表 Top10% 多头经济决策；当前证据不支持 TCN 预测效果必然或已经优于 LSTM。未访问 test/sealed test，未授权扩容、部署或交易。
- 工程验收：Ruff 全量通过；Mypy 133 个源文件通过；Pytest `225 passed`；preflight 与 `python tasks/test.py` 通过；`python -m build --no-isolation` 成功生成 sdist/wheel，wheel 未包含 runtime artifacts、checkpoint、Parquet 或凭据。

## TCN Top-tail 任务对齐优化 v34

- 协议：`docs/prompts/tcn-top-tail-alignment-v34.md`。control/candidate 只允许 objective 不同；数据、冻结父 checkpoints、TCN 架构、88 个可训练 shape-residual 参数、seeds/folds、日期顺序、优化器与预算完全相同。
- 实现：增加按 `(signal_date,horizon)` 等权的真实 top10% 对 non-top pairwise logistic loss，固定权重 `0.05`、温度 `0.1`；解析器强制显式公开参数和冻结父模型契约。每个训练 batch 记录 pair 数、SmoothL1/Top-tail prediction-space 梯度 cosine、总梯度与吞吐。
- 真实证据：`artifacts/tcn-top-tail-alignment-v34/`，receipt `de87e54264b942da3b67c0e9da3b9f2aa302e41a6b7ac99f8c0ad9ec9185cb2c`；46 个 outputs 的 SHA-256 已逐项复核。TCN 普通验证覆盖 2 objectives × 3 seeds × 5 folds，并与 v33 LSTM 在同一逐样本契约上对比；RankIC 重放误差 `2.78e-17`，父模型 drift `0`，sealed test 未访问。
- 机制与速度：Top-tail 每 batch 平均约 826 pairs，分量梯度 cosine 中位数 `+0.5142`，没有梯度冲突复发。candidate TCN/LSTM model-step `6.2309x`、端到端 `5.7578x`，继续达到 3–5×目标；但 candidate/control 训练吞吐仅 `0.7377x`，新增 loss/诊断开销门失败。
- 效果：candidate-control RankIC `-0.000610`，95% CI `[-0.001134,-0.000095]`；Top precision `+0.000208`、Top return `+0.000056` 的区间均跨零；NDCG `-0.000027`；turnover `+0.003059` 且区间高于零。状态 `stop_top_tail_no_gain_v34`，未晋级 sealed test。
- 工程验收：Top-tail 红测先失败后转绿；相关集成测试 47 passed；全量 Ruff 通过；Mypy 135 个源文件通过；Pytest `233 passed`；preflight 与 `python tasks/test.py` 通过；`python -m build --no-isolation` 成功生成 sdist/wheel，wheel 49 个成员中不含 artifacts、checkpoints、Parquet 或凭据。

## TCN 受约束 Top-tail checkpoint selection v35

- 协议：`docs/prompts/tcn-constrained-tail-checkpoint-selection-v35.md`。只训练一次固定8-epoch Top-tail TCN轨迹；control按最大RankIC选择，candidate只在`unit best RankIC-0.002`可行域内最大化等权Top precision/NDCG，二者共享全部epoch checkpoints。
- 实现：`TCNTuningResult`可选审计式捕获全部epoch states并关闭early stopping；纯`select_constrained_tail_checkpoints`模块验证epoch完整性、重复键、有限指标、RankIC可行域与确定性tie-break。真实运行器逐epoch重放统一任务契约、保存135个checkpoints并绑定SHA-256。
- 真实证据：`artifacts/tcn-constrained-tail-checkpoint-selection-v35/`，receipt `96f673676f73056efe28d67913e8e8ab1029b28d733af4cb241a8fd91eb85f73`；154个outputs及135个checkpoint哈希逐项复核通过。RankIC历史重放误差`5.55e-17`、selected metric重放误差`1.11e-16`、control best-epoch误差`0`；sealed test未访问。
- 机制：15/15轨迹均覆盖epoch 0..8，7/15单元改选；4个改选更早、3个更晚，candidate最晚只到epoch 2。梯度cosine中位数`+0.5143`、父模型drift `0`。证据支持任务对齐selection，而不支持简单延长训练。
- 效果：candidate-control Top precision `+0.001458`，95% CI `[+0.000729,+0.002292]`；NDCG `+0.000566`，CI `[-0.000053,+0.001192]`；RankIC `-0.000269`，CI跨零且在容忍域内；Top return `+0.000100`，CI跨零；turnover不变。四门通过，状态`constrained_tail_ordinary_validation_candidate_v35`。相对LSTM仍在RankIC、precision和turnover上落后，未宣称TCN架构优越。
- 速度：固定8-epoch TCN/LSTM model-step `6.1094x`、end-to-end `5.6770x`；逐epochselection评测`166.56s`独立分账。
- 工程验收：选择与轨迹红测先失败后转绿；全量Ruff通过；Mypy 138个源文件通过；Pytest `237 passed`；preflight与`python tasks/test.py`通过；`python -m build --no-isolation`成功生成sdist/wheel，wheel 50个成员中不含runtime artifacts、checkpoint文件、Parquet或凭据。

## Evidence levels

1. **Contract evidence**：schema、配置和清单可验证。
2. **Correctness evidence**：PIT、因果性、标签、切分、成交和成本场景测试通过。
3. **Reproduction evidence**：相同代码、数据指纹、配置和种子可重放摘要结果。
4. **Comparison evidence**：所有基准使用相同信息集、切分和计算预算。
5. **Frozen-test evidence**：候选规则预登记后只运行一次封存测试。

前四级的工程路径和合成证据已齐全；一次完整真实数据运行仍必须提供所有强制产物才能标记工程完成。第五级尚未在真实封存数据上执行，只有满足预登记门槛后才可称为候选模型。

## Required commands

所有工作项至少运行聚焦测试、`python tasks/preflight.py` 和 `python tasks/test.py`。离线运行使用：

```text
python tasks/run_experiment.py --config config/minimal.example.json --manifest <manifest.json> --output-root <untracked-run-root>
```

真实硬件基准与 Promotion 封存消费必须显式执行并保存收据，不能以合成测试代替。

真实数据 pilot 在运行前还必须执行：

```text
python tasks/check_pilot_readiness.py --descriptor <untracked-pilot-readiness.json>
```

该命令只验证输入身份、治理声明、切分、比较协议、预算和封存隔离，不产生真实效果证据或运行授权。

## TCN v35 候选冻结与 sealed readiness v36

- 协议：`docs/prompts/tcn-v35-once-only-sealed-readiness-v36.md` 固定 v35 candidate/control checkpoint、v33 LSTM、源数据身份、Top-tail/RankIC/成本/换手/速度门槛及逐字授权。readiness 只允许读取 split 元数据与流式文件哈希，不读取 sealed 特征值、标签值、预测或指标。
- 时间安全修正：原始 canonical test 有两个 100 交易日区间。第一段从 2023-12-13 开始，只允许 validation 于此前结束的 ordinary folds 0–2；folds 3/4 即使没读 sealed 标签，其选择日期也晚于测试起点，故被拒绝。第二段从 2024-10-30 开始，允许 folds 0–4。最终 24 个 time-safe seed/fold exposures，12 个使用 v35 改选 checkpoint；`sealed_holdout` 作为前段重复登记不计入 canonical test。
- 真实 readiness：`artifacts/tcn-v35-sealed-readiness-v36/`；freeze ID `642a485b09a2c381e5c1dac0fbd6d1938edf44e21c54d738cf3c4ff17ce486a6`，receipt `4f86b1bcbe1846c7ec8c0c5fdf311b58753bd7e4aba1a6fe3ee190964b2655ad`。sealed manifest SHA-256 `04cd32ec7f69fa3260ffa59f079dbdf03a8e26a204be16e96ffe551a578b7649`，canonical test 共 3973 样本；全部 readiness 输出哈希复核通过。
- 当前状态：`awaiting_explicit_sealed_authorization_v36`、`authorization_received=false`、`sealed_test_accessed=false`、`evaluation_executed=false`、`consumed_marker_created=false`。历史“同意”或近似表述不构成本协议授权；必须由用户新消息逐字给出协议规定的短语后，才允许一次性消费。

## TCN v35 一次性 sealed test v36

- 授权与消费：用户逐字授权后执行 `tasks/run_tcn_once_only_sealed_evaluation.py`。runner 在加载 sealed 特征/标签前复核 freeze、readiness、全部源数据与 checkpoint SHA-256，随后原子创建全局消费标记。标记状态已永久完成；同一 freeze/sealed 身份不得再次运行或调参复用。
- 真实产物：`artifacts/tcn-v35-once-only-sealed-evaluation-v36/`，receipt `68205700cf0a21b807c6ea8c6e5aea351a9732cda0662cf89d27a1fe2fe02db9`。共 562,113 条逐样本/期限预测、28,800 个模型-日期-期限指标组和 800 个先聚合模型单元的 sealed 日期/期限 paired groups；9 个输出哈希逐项复核通过。
- Candidate-control：RankIC `-0.000092`，95% CI `[-0.000336,+0.000164]`；Top precision `-0.000097`，CI `[-0.000500,+0.000333]`；NDCG `-0.000112`，CI `[-0.000598,+0.000363]`；Top return `-0.000011`，CI `[-0.000038,+0.000017]`；单边 10 bps 成本后收益 delta `-0.000010`，CI `[-0.000038,+0.000019]`；turnover `-0.000701`，CI `[-0.001894,+0.000449]`。
- Candidate-LSTM：RankIC `-0.005624`、Top precision `-0.006389`、NDCG `-0.003410`、Top return `-0.000162`；四项均为 candidate 落后。
- 决策：冻结速度门继续通过（model-step `6.1094x`、end-to-end `5.6770x`），RankIC、收益、成本后收益与换手容忍门通过；但 Top precision/NDCG 均值门和 robust-tail CI 门失败。因此最终状态为 `sealed_rejected_tcn_candidate_v36`、`candidate_model=false`，不授权部署、交易或在同一 sealed 数据上继续优化。

## TCN 因果相对特征普通验证 v37

- 协议：`docs/prompts/tcn-relative-cross-sectional-features-v37.md`。旧 v36 sealed 永久保持已消费状态，本轮只读取 ordinary `train/validation`；模型、折、seed、epoch、优化器和 checkpoint 规则固定，只检验输入表征。
- 特征工程：新增 chunked memmap 物化与 `(signal_date,instrument_id)` PIT 一对一门禁。23,821 个 `[8,480]` 基础窗口生成 `[13,480]` 候选；423 个早期 ADV 缺失只使用窗口内先前完整日 fallback。尺度不变、未来日期隔离、缺失状态 fail-closed 和 top50 覆盖测试通过。
- top50 门禁：本地 PIT top100 权重完整覆盖 1,212 天；top50 需要 60,600 个状态键，缺 36,375 个 market-cap/state 键，状态 `blocked_missing_pit_state`，未强行扩容。
- 真实证据：`artifacts/tcn-relative-feature-validation-v37/`，receipt `091cb88ca4bcc8827ea301750c9d207b9904c95281fac66e931846eaf64cc16d`；覆盖 base/relative TCN/LSTM 四组合、3 seeds × 5 folds、4,800 个模型-日期-期限指标组/模型，sealed test 未访问。
- 效果：relative-base TCN RankIC `-0.035763`，仅 4/15 单元为正，CI low `-0.044593`；Top precision `+0.004583`，但 NDCG `-0.014499`、Top return `-0.000769`、turnover `+0.138608`。LSTM 也下降 RankIC `-0.032493`，说明失败主要来自该替换式表征，而非 TCN 独有。
- 速度与梯度：relative TCN 吞吐保留 `1.1827`，相对 relative LSTM 完整周期 `5.7286x`，速度仍满足目标；本轮使用单一 SmoothL1 隔离表征，不产生辅助梯度冲突。最终状态 `stop_relative_features_no_stable_gain_v37`。
- 工程验收：Ruff 全量通过；Mypy 150 个源文件通过；Pytest `253 passed`；preflight 与 `python tasks/test.py` 通过；production build 成功生成 sdist/wheel。wheel 54 个成员、sdist 124 个成员，均不含 artifacts、checkpoints、Parquet、NPY、PT、`.env` 或凭据。

## TCN 追加式相对时序特征消融 v38

- 协议：`docs/prompts/tcn-append-relative-sequence-ablation-v38.md`。保留 base8 全部通道，只追加 amount/ADV20 与 amount/market-cap 两个已审计因果时序通道；不再复制静态横截面秩。Phase A 固定 seed 7×5 folds，只有全部效果与速度门通过才允许多seed/LSTM Phase B。
- 物化：`artifacts/tcn-appended-relative-sequence-top20-v38/`，23,821 个 `[10,480]` float32 memmap 窗口；前八通道与 base 逐位相同，后两通道与 v37 对应通道逐位相同；receipt `f41fb46d3e7db458754063a05492042d92e6886d2e0b3bc4b0304fda61991f74`。
- 真实 Phase A：`artifacts/tcn-appended-relative-seed7-screen-v38/`，receipt `e804c897fbf26f0c0a262cca19c142b21ee87401fe92d84e73bb1e123939030f`。候选-base RankIC `+0.002257`、4/5 folds 为正，说明保留原始成交尺度修复了 v37 的 RankIC 崩塌；TCN 吞吐保留 `1.0036`。
- 尾部结果：Top precision `-0.005000`、NDCG `-0.012250` 且 CI `[-0.023463,-0.000784]`、Top return `-0.000772`；RankIC CI low `-0.008089`。四门失败，状态 `stop_append_relative_sequence_seed7_v38`，未执行 seeds 17/27 或 LSTM Phase B，sealed test 未访问。
- 工程验收：Ruff 全量通过；Mypy 152 个源文件通过；Pytest `255 passed`；preflight 与 `python tasks/test.py` 通过；production build 成功。wheel 54 个成员、sdist 124 个成员，均不含 artifacts、checkpoints、Parquet、NPY、PT、`.env` 或凭据。

## TCN top50 PIT 横截面扩容 v39

- 协议：`docs/prompts/tcn-top50-pit-breadth-validation-v39.md`。只复验 top50 base8 与 `base8+relative2`；固定 dynamic-horizon-skip TCN、seed 7、五折、8 epochs、SmoothL1、lr 0.003、batch 128、8 CPU threads。旧 sealed test 禁止访问。
- PIT 修复：从 top100 权重得到 1,212 天、60,600 个 top50 membership keys、111 个唯一股票；复用 top20 的 46 只，仅增量拉取 65 只的 720 个分块请求。60,556 个完整 stock-day 均有正有限 market cap；44 个无完整分钟时段的 membership keys 被严格排除。
- infra 等价优化：窗口构造由每样本全表扫描改为按股票日期偏移切片。完整 top20 的 23,821 个 tensor/index/rejections 全部精确相等，耗时 1.80 秒；top50 59,539 个窗口只需 4.84 秒，完整数据物化总计 237.11 秒。训练数据 receipt `dc13c0cdb73c0162060519784b8b581a16ae99f03c6d60bb34f87e477c67288b`。
- 特征与宽度：relative10 物化 receipt `e54a14779ca9ed693bf8ca14af6d9bc6e3298dcc53cc61e58f8d627bf9a17867`；前8通道逐位保留、无静态秩，1,040 个 ADV 仅用因果 fallback。validation 横截面 40–50 只，Top10% 4–5 只。
- Phase A：artifact `artifacts/tcn-top50-relative-seed7-screen-v39/`，receipt `2d9f7adf4d003c6efb88417e80fb384a49cc7521c5a9c8603ee5dfd3dfc25483`。relative-base RankIC `+0.007593`、Top precision `+0.008000`、NDCG `+0.006891`、Top return `+0.000256`；RankIC/precision/NDCG bootstrap 下界均高于零；TCN 吞吐保留 `0.97497`。
- 决策：turnover delta `+0.030759` 超过预登记上限 `+0.02`，唯一 blocker 为 `turnover_delta_above_gate`，状态 `stop_top50_relative_sequence_seed7_v39`。Phase B seeds17/27 与 LSTM 未运行，sealed test 未访问；不构成 Alpha、候选模型或部署授权。
- 工程验收：Git diff check 与 Ruff 全量通过；Mypy 155 个源文件通过；Pytest `258 passed`；preflight 与 `python tasks/test.py` 通过；`python -m build --no-isolation` 成功。wheel 54 个成员、sdist 124 个成员，均不含 runtime artifacts、checkpoint 二进制、Parquet、NPY、PT、`.env` 或凭据。
