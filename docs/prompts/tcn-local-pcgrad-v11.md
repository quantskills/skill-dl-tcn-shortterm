# TCN 局部梯度冲突修复真实验证提示词 v11

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化与真实实验执行者。项目主模型必须保持为 Temporal Convolutional Network；LSTM 只作为相同数据与资源协议下的 benchmark，不得替换 TCN。本轮只验证一个事前登记的机制假设：1 日与 5 日任务在 TCN 共享干路 block 4/6 上的梯度方向冲突，是否是当前 TCN 预测效果落后于 LSTM 的主要可修复原因。

## 已知证据与待证伪假设

- v10 真实 seed-7 五折结果：`skip-c16-chomp-smooth` mean RankIC `0.087731`，LSTM `0.111595`；TCN/LSTM model-step `3.8099×`，端到端 `3.6554×`。
- 5/5 folds TCN 均落后于 LSTM；这不是单折偶然，也不是全局感受野不足。
- 训练集梯度探针显示 1d/5d global cosine 中位数 `-0.0161`、负值率 `59.4%`，4/5 folds 满足 PCGrad 触发条件；冲突最强的共享层为 block 4（中位数 `-0.1100`）和 block 6（中位数 `-0.1357`）。
- 假设：只在 `(1d, 5d) × (block 4, block 6)` 上投影负梯度，其余 horizon 与参数保留普通 mean SmoothL1 梯度，能提高 TCN RankIC，同时把额外反传开销控制在速度门槛内。

## 不可变数据与协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`。
- 特征 memmap SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`。
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`。
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`。
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`，SHA-256 `7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`。
- 2021–2025 沪深 300 PIT top20、5 分钟标准条、8 特征、480 步窗口；5 个 expanding folds，仅 seed 7。
- 设备 CPU、float32、PyTorch threads 8、DataLoader workers 0、batch 128、Adam、lr 0.003、最多 8 epochs、patience 2、min_delta 0.002。
- 只消费 `train` 和 ordinary `validation`。出现 `test`、`sealed=true`、未知 stage、输入哈希漂移或目标目录已存在时必须 fail closed。

## TCN 不可破坏约束

1. 模型保持 `HorizonSkipTCN`：16 channels、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0。
2. 本实现的精确感受野公式为 `1 + (kernel_size - 1) × sum(dilations) = 511`，必须覆盖 480 步输入。不得把“约等于 2^层数”的经验式当作代码验收公式。
3. 卷积必须严格因果：只允许左侧 padding 后 chomp；任何未来 padding、对称 padding 后泄漏未来值或中心卷积均禁止。
4. 残差卷积继续使用 WeightNorm，禁止换成 BatchNorm。
5. `feature-windows.npy` 必须以 `np.load(..., mmap_mode="r", allow_pickle=False)` 读取，禁止把完整分钟线窗口 eager materialize 到内存；索引与标签继续使用 Parquet/Arrow。
6. fold 内特征 mean/std 只能由 train positions 拟合并应用于 validation，禁止跨折或全量拟合。

## 单变量候选

- 控制：`skip-c16-chomp-smooth`，普通 masked SmoothL1。
- 候选：`skip-c16-chomp-localpcgrad-b46-h15`，架构、初始化 seed、batch 顺序、优化器和早停协议全部与控制一致；唯一差异是局部 PCGrad。
- 局部 PCGrad 只读取 horizon 1 与 5 对 block 4/6 的梯度。先计算全部四 horizon mean SmoothL1 的普通梯度，再以固定 seed 顺序投影 1d/5d 的负内积分量，并只替换这两个任务在 block 4/6 上的贡献。block 0–3、5、7、skip logits、四个 horizon heads 及所有未选参数必须保留普通 mean-loss 梯度。
- PCGrad 不需要 date-grouped batch；控制与候选都使用相同的 seeded-random batch sampler。不得把 batch 组成变化混进本轮干预。
- 模型步计时必须包含额外 horizon backward 与 projection；另行记录 `pcgrad_horizon_backward_seconds` 和 `pcgrad_projection_seconds`，但不得从总时间中扣除。
- 不扫描 dropout、weight decay、rank-loss 权重、channels、kernel、dilation、batch、学习率、PCGrad block/horizon 组合或额外 seed。

## LSTM benchmark 与门禁

- LSTM hidden size 34（约 6124 参数），lr 0.003、batch 128、8 epochs；fold、base seed、线程、精度和数据与 TCN 完全一致。
- LSTM 不参加 TCN 选择，只测量逐折 RankIC、model-step 和端到端吞吐。
- 候选准入需同时满足：mean RankIC `>= 0.09`、5/5 folds RankIC 为正、median samples/s `>= 5000`、mean RankIC 不低于 TCN 控制。
- 同时报告 TCN/LSTM model-step 与端到端速度比；`>=3×` 才能支持“3–5×速度优势”，低于 `3×` 必须明确判定速度目标未保持。
- seed 7 未同时通过效果与速度门禁时立即停止，不运行 seeds 17/27，不访问 sealed holdout；负结果也是本轮完成状态。

## 测试驱动执行

1. 先为局部 PCGrad 写红灯测试：未选参数保持普通总损失梯度；选中 block 只修正 1d/5d 冲突贡献；配置完整解析 block/horizon scope；非法 scope fail closed。
2. 最小实现直至聚焦测试转绿；receipt 暴露 scope、batching identity、额外反传/投影时间。
3. 运行 Ruff、Mypy、preflight 和完整测试；校验 RF、因果性、WeightNorm 与 memmap 契约。
4. 使用 `config/pandadata-tcn-local-pcgrad-seed7-v11.example.json` 运行真实 seed-7 五折控制/候选与 LSTM benchmark，输出到全新的 `artifacts/tcn-local-pcgrad-v11-seed7/`。
5. 复算 receipt 和全部输出哈希；生成逐折、逐 horizon、速度分账、候选相对控制和相对 LSTM 的结果文档。所有数值只能来自真实产物。

## 安全与停止条件

- 不调用 PandaData、不下载数据、不打印或写入凭据、不提交数据/checkpoint/runtime receipt。
- 不覆盖已有 artifact，不部署、不交易、不连接券商、不做任何外部写入。
- 若局部 PCGrad 未提升效果、速度跌破门槛或证据不足，记录被证伪的具体机制并停止；不得看到结果后追加 trial。
