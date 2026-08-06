# TCN 有符号时间适配器多种子确认提示词 v14

你是 `skill-dl-tcn-shortterm` 项目的 TCN 多种子确认执行者。v13 已在 seed 7 五折 ordinary-validation 上准入 `signed-context-c16-chomp-smooth`；本轮不得修改模型、损失、数据、fold、训练预算或门槛，只运行已授权 seeds 17/27，判断 seed-7 的结构增益是否稳定。TCN 继续是主模型，LSTM 只作相同协议 benchmark。

## 前置授权

- v13 artifact：`artifacts/tcn-signed-temporal-adapter-v13-seed7/`
- v13 receipt：`31748f1a344630cbb3eac7b47f85ce33c949c4bc5cd1c9bd7a96cb7bb09bcced`
- v13 状态：`seed7_winner_admitted_v11`
- v13 winner：`signed-context-c16-chomp-smooth`
- v13 明确授权 confirmation seeds：`[17,27]`
- 执行前必须复算 v13 receipt 输出哈希，并验证 `sealed_test_accessed=false`；任一不符则 fail closed。

## 不可变数据与训练协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- 特征 SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 只允许 `train` 和 ordinary `validation`；遇到 `test`、`sealed=true`、未知 stage、哈希漂移或输出已存在时 fail closed。
- seeds 只能是 `[17,27]`，每个 seed 五个 expanding folds。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、最大 8 epochs、patience 2、min_delta 0.002、Adam、lr 0.003、batch 128。
- TCN 公共参数：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、strict causal chomp、WeightNorm、masked SmoothL1、bars per day 48。
- LSTM：hidden size 34、lr 0.003、batch 128、8 epochs，使用完全相同的 fold/base seed/线程/精度。

## 冻结实验臂

只运行以下两条 TCN，不得增加第三条：

1. `context-c16-chomp-smooth`：v12 simplex 双尺度控制。
2. `signed-context-c16-chomp-smooth`：v13 有符号适配器候选。

两者必须保持 6524 参数。候选的初始函数、trunk、heads 和训练协议继续与控制相同，唯一差异仍是 day/intraday 时间权重是否受 simplex 约束。

## 预注册确认门槛

新 seeds 17/27 必须同时满足：

1. 候选跨 10 个 fold-seed 单元的 mean RankIC `>=0.09`。
2. 候选 10/10 单元 RankIC 为正。
3. 候选相对控制的跨 10 单元平均 RankIC 增益严格大于 0。
4. seed 17 和 seed 27 各自至少 3/5 folds 候选不退化。
5. 候选相对控制的聚合 3 日和 5 日 RankIC delta 均 `>=-0.005`，防止 seed-7 的轻微中周期退化扩大。
6. 候选 median samples/s `>=5000`。
7. 候选相对参数匹配 LSTM 的 model-step 与端到端速度比均 `>=3.0x`。

全部通过时状态为 `signed_candidate_multiseed_confirmed_v14`；效果门槛失败为 `stop_signed_candidate_unstable_v14`；只因速度失败为 `stop_signed_candidate_speed_v14`。无论结果如何都不得访问 sealed test。

## 证据与输出

- 输出目录：`artifacts/tcn-signed-temporal-adapter-multiseed-v14/`，存在时拒绝覆盖。
- 保存两个 seed 的 TCN epoch history、leaderboard、checkpoint、LSTM measurements、逐 seed/fold/horizon delta、速度比较、权重符号摘要、selection 和 receipt。
- receipt schema：`tcn-signed-multiseed-confirmation-v14/v1`，必须记录 v13 parent receipt、源哈希、代码身份、seeds、完整门槛结论、输出哈希和 `sealed_test_accessed=false`。
- 完成后可将 v13 seed 7 与 v14 seeds 17/27 合并形成三 seed 描述性汇总；确认决策本身只使用预注册的新 seeds 17/27，不能让旧 seed 重复加权掩盖失败。

## 测试与工程门禁

1. 先为确认决策写红灯测试：错误 seed、缺 fold、重复单元、参数不匹配、3/5 日退化扩大和速度失败均 fail closed 或产生正确状态。
2. 运行器必须验证 v13 授权、source SHA-256、stage/sealed、目标不可覆盖和无 secret-like config key。
3. 运行 Ruff、完整 mypy、聚焦 pytest、统一测试入口、preflight 和 production build。

## 安全与停止条件

- 不调用 PandaData、不下载数据、不写入或打印凭据。
- 不提交原始数据、checkpoint、runtime receipt；运行产物只写入 Git 忽略的 `artifacts/`。
- 不访问 test/sealed、不部署、不交易、不连接券商、不进行外部写入。
- 若确认失败，记录具体失败维度并停止；不追跑新 seed、不调参、不修改门槛。

## 执行顺序

1. 验证 v13 parent receipt 和 seeds 授权。
2. 写并运行确认门槛红灯测试。
3. 实现不可覆盖的 v14 runner、配置与 receipt。
4. 完成全部工程门禁。
5. 顺序运行 seeds 17/27 的两条 TCN 和固定 LSTM，避免并发污染 CPU 吞吐。
6. 复算输出哈希，形成新 seeds 确认和三 seed描述性汇总，停止于 ordinary validation。
