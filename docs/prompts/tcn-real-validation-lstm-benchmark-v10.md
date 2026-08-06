# TCN 真实五年 ordinary-validation 与 LSTM 公平基准提示词 v10

你是 `skill-dl-tcn-shortterm` 的 TCN 真实实验执行者。TCN 是唯一待优化的主模型；LSTM 只作为相同数据、折叠、batch、精度、线程和训练预算下的 benchmark，不得替换 TCN，也不得因为 LSTM 指标更好而改变项目方向。

## 目标

在 2021-01-01 至 2025-12-31 的真实 PandaData 沪深 300 PIT top20、5 分钟标准条、480 步回看窗口上，运行五个 expanding ordinary-validation folds。先用 seed 7 对三个预登记 TCN 路径做一次有界筛选，同时用参数量近似匹配的 LSTM 建立效果和速度参考；只有唯一 TCN 通过 seed-7 门禁时，才允许运行 seeds 17/27 确认。

## 不可变输入

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`。
- 特征 memmap SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`。
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`。
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`。
- expanding ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`。
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`。
- 只消费 `train` 与 `validation`；任何 `test`、`sealed=true` 或未知 stage 必须在训练前 fail closed。

## 预登记协议

- 设备：当前 CPU；精度：float32；PyTorch threads：8；DataLoader workers：0。
- folds：0、1、2、3、4；seed-7 screen；最多 8 epochs；patience 2；min_delta 0.002。
- 共同 TCN 参数：16 channels、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、Adam、lr 0.003、batch 128、causal padding+chomp、WeightNorm。
- TCN 控制：`lite-c16-chomp-smooth`，最后时点读出，masked SmoothL1。
- TCN 候选一：`skip-c16-chomp-smooth`，horizon-specific causal block simplex skip，masked SmoothL1。
- TCN 候选二：`skip-c16-chomp-rank01`，相同 skip 架构，`SmoothL1 + 0.1 × date/horizon grouped pairwise logistic rank loss`。
- 不增加第四个 trial；不在看到结果后改变 channels、kernel、dilation、dropout、weight decay、学习率、batch 或 loss 权重。
- LSTM benchmark：hidden size 34（约 6124 参数，对齐 TCN 的约 6228–6260 参数），lr 0.003、batch 128、8 epochs、相同 folds/seed/线程/精度。LSTM 不参加 TCN 候选选择。

## Seed-7 TCN 门禁

候选必须同时满足：五折 mean RankIC `>=0.09`、5/5 folds RankIC 为正、训练吞吐中位数 `>=5000 samples/s`、mean RankIC 不低于 TCN 控制。若多于一个候选通过，依次按 mean RankIC、worst-fold RankIC、吞吐、参数量、稳定 trial ID 决胜。控制组本身不能被称为“优化候选通过”。

无候选通过时输出 `stop_no_seed7_pareto_v10`，不得运行额外 seed。唯一候选通过时输出 `seed7_winner_admitted_v10`，此时才允许只对该候选、TCN 控制与 LSTM 运行 seeds 17/27。

## LSTM 比较口径

报告但不硬编码结论：TCN 与 LSTM 的参数量、逐折最佳 ordinary-validation RankIC、模型步 samples/s、端到端训练 samples/s、data wait、validation、complete cycle、time-to-best，以及按相同 fold/seed 配对的模型步与端到端速度比。不得用不同 batch、线程、精度、容量或 epoch 预算制造 3–5×。

## 证据与安全

- 运行产物只写入 Git 忽略的 `artifacts/`，目标存在时拒绝覆盖。
- receipt 必须包含配置、代码状态、输入与输出哈希、折叠、seed、参数量、RankIC、时序分账、选择/停止原因和 `sealed_test_accessed=false`。
- 不写入或打印凭据；不调用 PandaData；不重复下载；不提交原始数据、checkpoint、运行日志或 runtime receipt。
- 不访问 sealed holdout，不进行部署、交易、券商连接或外部写入。

## 执行顺序

1. 校验输入路径、SHA-256、shape、fold/stage/sealed 契约和感受野。
2. 运行聚焦测试、Ruff、Mypy 和 preflight。
3. 执行 seed-7 三路径 TCN 五折训练及参数匹配 LSTM benchmark。
4. 生成不可变 leaderboard、comparison 和 receipt，按门禁决定停止或确认。
5. 只有 seed-7 准入才运行 seeds 17/27；否则将负结果作为本轮完成状态。
6. 最后运行完整 pytest、统一测试入口和 production wheel/sdist build；权威文档只能引用真实 receipt 中已观察的数值。
