# TCN PIT 市场状态条件化提示词 v17

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化与真实实验执行者。主模型必须保持为 Temporal Convolutional Network；LSTM 只作为固定 benchmark，不得替换 TCN。v14–v16 已经证明 signed temporal readout 不是稳定增益来源；v16 的 TCN/LSTM model-step 与端到端速度分别为 `4.2381x` 和 `4.0663x`，因此本轮冻结已经达标的 causal TCN trunk、WeightNorm、chomp、感受野、训练基础设施与速度路径，只处理预测效果缺口。

## 1. 问题与父证据

- 父 artifact：`artifacts/tcn-decoupled-signed-residual-v16-seed7/`
- 父 receipt：`77731f043377cdab2aca535ff4185635a0896ec9660ad168b2255519ea1d6be6`
- 父状态：`stop_decoupled_residual_seed7_effect_v16`
- 稳定 TCN control：`context-c16-chomp-smooth`
- control mean RankIC：`0.0874870052`
- 固定 LSTM mean RankIC：`0.1115955`
- signed readout 路线在本轮明确停止，不得继续调整 residual LR、scale、seed 或门槛。

当前 TCN 为每只股票独立编码 480 根 5 分钟条，但主任务是在同一 `signal_date × horizon` 内做横截面排序。模型没有显式看到同一天的共同市场冲击或横截面离散度，存在任务结构与输入表征错位。

真实五折 control checkpoint 的只读诊断探针得到 1,600 个 `date × horizon × fold` RankIC 单元，总均值严格复现为 `0.0874870052`。按当日股票收益离散度中位数分组，高减低状态的 RankIC 差为：1 日 `-0.0142`、2 日 `-0.0551`、3 日 `-0.1042`、5 日 `-0.1236`。这只证明状态依赖值得测试，不证明条件化必然提高 RankIC。

## 2. 数据可用性结论

- 源运行的 `universe.parquet` 虽有 `industry` 字段，但真实值全部为 `unavailable`。
- `input-manifest.json` 明确记录 `industry_history=unavailable`。
- 因此本轮禁止伪造、静态回填或使用当前行业分类；`industry_context_status` 必须记录为 `blocked_historical_industry_unavailable`。
- 本轮只从冻结的逐股票因果特征窗口构造同一信号日可得的市场中心与横截面离散度，不调用 PandaData、不下载或补齐数据。

## 3. 可证伪假设

1. **H1：共同市场状态缺失是效果缺口的一部分。** 若成立，低秩、有界、零初始化的市场条件门控应在不改变 TCN trunk 的前提下把五折 mean RankIC 提高至少 `0.003`，并达到 `0.09`。
2. **H2：横截面离散度比单纯市场方向更重要。** 构造的上下文必须同时包含中位数中心和 MAD 离散度；receipt 必须分别审计两类字段，不能只传一个市场收益标量。
3. **H3：额外容量而非状态机制制造表面收益。** 候选只允许增加 260 个参数（`+3.985%`）；必须零初始化为与 control 完全等价，并报告 gate 参数范数与容量差异。
4. **H4：市场条件化没有稳定增益。** 若 seed-7 未通过全部效果门槛，立即停止，不运行 seeds 17/27、不访问 sealed test、不增加第二种 gate 或上下文口径。

## 4. 不可变数据与训练协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- features SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 只允许 seed 7、folds 0–4、`train/ordinary validation`；`test`、`sealed_holdout`、`sealed=true`、未知 stage 或哈希漂移全部 fail closed。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、max 8 epochs、patience 2、min_delta 0.002、Adam、batch 128。
- TCN：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、LR 0.003、strict causal chomp、WeightNorm、masked SmoothL1、480 bars、48 bars/day。
- control 与 candidate 必须使用同一 seed、fold、batch 顺序、训练预算、标签、损失和 early-stopping 规则。
- LSTM benchmark：hidden 34、LR 0.003、batch 128、8 epochs、相同 seed/folds/线程/精度；它只提供固定参考，不参与本轮 TCN candidate 选择。

## 5. PIT 市场上下文契约

只使用特征索引 0–5：`close_return`、`open_close_return`、`intrabar_range`、`log_volume`、`log_amount`、`vwap_deviation`。排除纯日内时钟 `time_sin/time_cos`。

对每个 `signal_date`：

1. 只收集该日 ordinary manifest 中可用的全部股票窗口；所有窗口必须严格结束于该信号日，不得读取未来日期或标签。
2. 在每个 `feature × time` 单元上，跨股票计算中位数 `center`。
3. 计算相对该中位数的跨股票 MAD `dispersion`。
4. 分别对 `center` 和 `dispersion` 计算最近 48 bars 均值与完整 480 bars 均值。
5. 按固定顺序拼成 24 维向量：`center_last_day[6]`、`center_full_window[6]`、`dispersion_last_day[6]`、`dispersion_full_window[6]`。

约束：

- 同一信号日所有股票必须得到逐位相同的上下文；不同日期允许不同。
- 某日期至少需要 2 只股票，否则 fail closed。
- 上下文构造不接受 labels 参数，也不得根据 fold、horizon、目标或模型预测改变。
- 修改未来日期的特征不得改变更早日期的上下文。
- 每 fold 的上下文均值与标准差只用该 fold 的唯一训练日期拟合；每个日期等权，验证段只读应用。
- 常数维标准差置 1；非有限值、重复 sample position、日期/窗口结束时间不一致、请求不可用 position 全部 fail closed。
- 上下文 tensor、字段顺序、允许 position、源哈希与 SHA-256 必须进入 artifact/receipt。

## 6. 唯一候选结构

保留 control 的完整 `TemporalContextTCN`：TCN trunk、day/intraday simplex readout 和四个 horizon heads 全部不变。新增 `MarketConditionedTemporalContextTCN`：

1. 输入仍是个股 `[batch, 8, 480]`，另接收标准化后的 `[batch, 24]` PIT 市场上下文。
2. 对市场上下文使用低秩 MLP：`Linear(24,4) -> tanh -> Linear(4,32)`。
3. 第二个 Linear 的 weight/bias 严格零初始化，因此相同 seed 下候选初始输出必须与 control 逐点一致。
4. gate 定义为 `1 + 0.25 * tanh(raw_gate)`，范围严格位于 `[0.75,1.25]`。
5. 同一个 32 维 gate 作用于每个 horizon 的 `cross_day[16] + intraday[16]` 表征；按元素乘法后再进入原 horizon head。共同 additive bias 会在日内横截面排序中抵消，因此禁止增加 shift 分支。
6. control 参数固定为 6,524；候选固定为 6,784，差值必须恰好为 260。候选仍是 TCN 主模型，市场 MLP 只是低秩条件门控。

冻结两臂：

1. `context-c16-chomp-smooth`：原 6,524 参数 control，不接收上下文。
2. `market-conditioned-c16-chomp-smooth-h4-g025`：24 维上下文、hidden 4、gate scale 0.25、6,784 参数。

不得增加第三臂，不得同时修改 readout、loss、trunk、dropout、weight decay、PCGrad、soft-RankIC、batch、epoch 或 LR。

## 7. seed-7 预注册门槛

candidate 必须同时满足：

1. 五折 mean RankIC `>=0.09`；
2. 相对 control 的 mean RankIC delta `>=0.003`；
3. 5/5 folds RankIC 为正；
4. 至少 3/5 folds 相对 control 不退化；
5. 1 日 delta `>=0`；
6. 2 日 delta `>=-0.003`；
7. 3 日 delta `>=-0.005`；
8. 5 日 delta `>=-0.005`；
9. median samples/s `>=5000`；
10. 参数数为 control 6,524 / candidate 6,784，差值 260；
11. gate 输出层 L2 范数必须大于 0，证明分支得到更新；
12. candidate 相对固定 LSTM 的 model-step 与端到端速度都 `>=3.0x`。

效果失败状态：`stop_pit_market_conditioning_seed7_effect_v17`。效果通过但速度失败：`stop_pit_market_conditioning_seed7_speed_v17`。全部通过：`pit_market_conditioning_seed7_admitted_v17`，只授权 seeds `[17,27]`，本轮仍不得自动运行。

## 8. 红灯反馈环

实现前建立并实际运行秒级测试，至少锁定：

- 同日一致、跨日变化、未来扰动不影响过去上下文；
- 构造器无 labels/horizon 输入，拒绝未来结束窗口、重复 position、单股票日期、不可用 position 和非有限值；
- fold 标准化只用唯一训练日期，验证极值不能回流；
- control/candidate 相同 seed 初始输出严格一致；
- gate 范围、维度、参数数和 260 参数容量差；
- candidate 缺上下文、错误上下文维度、control 意外接收上下文全部 fail closed；
- config、trial 身份、seed/fold、门槛、source/parent hash 和输出覆盖全部 fail closed；
- 任一 mean/fold/horizon/吞吐/速度/容量/gate-use 门槛失败时返回正确 blocker/status。

真实五折 RankIC 是最终反馈环；结构测试变绿不代表预测效果已经改善。

## 9. 输出与证据

- 配置：`config/pandadata-tcn-pit-market-conditioning-seed7-v17.example.json`
- runner：`tasks/run_tcn_pit_market_conditioning.py`
- 输出：`artifacts/tcn-pit-market-conditioning-v17-seed7/`，存在时拒绝覆盖。
- 保存 `market-context.npy`、`market-context-manifest.json`、epoch history、leaderboard、effect/horizon summary、checkpoints、LSTM measurements/environment、comparison、selection、resolved config 和 receipt。
- leaderboard/receipt 记录 context schema/identity、feature indices、fold train-only scaler identity、hidden size、gate scale、gate 参数范数、参数容量、吞吐、速度与 `industry_context_status`。
- receipt schema：`tcn-pit-market-conditioning-v17/v1`；记录 v16 parent、全部源哈希、代码身份、输出哈希与 `sealed_test_accessed=false`。
- 输出完成后复算全部文件 SHA-256；不得打印或写入凭据。

## 10. 工程与安全边界

- 真实训练前通过 Ruff、完整 mypy、聚焦 pytest、统一测试入口、preflight 和 production build。
- 不调用 PandaData、不补数据、不联网下载、不访问 test/sealed、不部署、不交易、不连接券商、不进行外部写入。
- 所有真实实验顺序执行，避免并发污染 CPU 吞吐。
- seed 7 失败后立即停止；不得通过改门槛、加 arm、挑 horizon 或追 seed 挽救结果。

## 11. 执行顺序

1. 复算 v16 父 receipt、source hashes 与 ordinary-validation 边界。
2. 运行真实 checkpoint 状态依赖诊断并记录结果。
3. 写红灯测试并确认失败。
4. 实现 PIT context builder、fold-train-only scaler、context dataset、条件 TCN、配置解析、决策和 runner。
5. 运行聚焦测试与完整工程门禁。
6. 顺序运行 seed-7 两臂 TCN 与固定 LSTM benchmark。
7. 复算输出哈希，审计 effect、horizon、fold、gate-use、容量、吞吐、速度和泄漏边界。
8. 写 `docs/research/tcn-pit-market-conditioning-v17-results.md`，明确假设判定与下一步后停止。
