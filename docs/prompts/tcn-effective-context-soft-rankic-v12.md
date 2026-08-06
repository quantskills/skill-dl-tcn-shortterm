# TCN 有效上下文与 soft-RankIC 真实验证提示词 v12

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化与真实实验执行者。项目主模型必须保持为 Temporal Convolutional Network；LSTM 只作为相同数据、切分、精度、线程、batch 和 epoch 预算下的 benchmark，不得替换 TCN。本轮只验证已由 checkpoint 归因探针支持的两个机制假设：当前 `HorizonSkipTCN` 的最后时点读出导致有效感受野塌缩；逐样本 SmoothL1 与日期横截面 RankIC 的评估目标不完全一致。

## 已知基线与问题

- v10 父模型 `skip-c16-chomp-smooth`：五折 mean RankIC `0.087731`，LSTM `0.111595`，TCN/LSTM model-step `3.8099x`、端到端 `3.6554x`。
- v11 局部 PCGrad 仅把 mean RankIC 提高到 `0.087948`，同时把相对速度降到约 `2.75x`，因此梯度冲突不是本轮首要优化对象。
- 五折 checkpoint 的只读归因显示，四个 horizon 有约 74%–79% 的输入归因集中在最后一个交易日；原有 horizon skip 对浅层 block 的权重没有形成有效 horizon 专门化。
- 输入固定为 10 个完整交易日、每日 48 根标准 5 分钟线，共 480 步；预测 horizon 固定为 1/2/3/5 日横截面排名。

## 本轮目标

在保留一卷积 residual TCN、严格因果卷积、WeightNorm、只读 memmap 和现有五折 ordinary-validation 协议的前提下：

1. 用完整隐藏序列构造显式的日内与跨日上下文，取消候选模型对“每层最后一个时间点”的依赖。
2. 为四个 horizon 分别学习日内 48 步和跨日 10 天的 simplex 权重，直接为历史日期建立梯度路径。
3. 增加日期分组的 differentiable soft-rank Spearman surrogate，并与 SmoothL1 混合；不得复用已失败的固定 pairwise logistic 作为本轮候选。
4. 通过单变量消融区分读出改进和目标函数改进，并与冻结的 LSTM benchmark 公平比较。

## 不可变数据与协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- 特征 SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 仅消费 `train` 与 ordinary `validation`；出现 `test`、`sealed=true`、未知 stage 或哈希漂移时训练前 fail closed。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、seed 7、五个 expanding folds、最大 8 epochs、patience 2、min_delta 0.002。
- TCN 公共参数：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、Adam、lr 0.003、batch 128、causal chomp、WeightNorm。
- LSTM：hidden size 34、lr 0.003、batch 128、8 epochs，与 TCN 使用相同 fold/base seed/线程/精度。

## 预注册实验臂

只允许以下三条 TCN trial，不得看到结果后追加 trial 或调参：

1. `skip-c16-chomp-smooth`：冻结的 v10 `HorizonSkipTCN` 父模型，masked SmoothL1。
2. `context-c16-chomp-smooth`：相同 TCN trunk，改为多尺度完整时序读出，masked SmoothL1。
3. `context-c16-chomp-softrank20-tau10`：与 trial 2 完全相同的模型，使用 `SmoothL1 + 0.2 * soft_rankic_loss`；soft-rank temperature 固定为 `0.1`。

不得扫描 channels、kernel、dilation、dropout、weight decay、学习率、batch、soft-rank 权重、temperature 或其他模型组合。本轮不启用 PCGrad。

## 多尺度 TCN 读出契约

- 新模型仍以现有严格因果 `CausalLiteBlockChomp` 组成 TCN trunk，名义感受野按 `1 + (kernel_size - 1) * sum(dilations) = 511` 验证，必须覆盖 480 步。
- `input_steps` 必须能被 `bars_per_day=48` 整除；否则 fail closed。
- 只消费最终 TCN block 的完整 `[batch, channel, time]` 隐藏序列，不得只抽取 `time=-1`。
- 跨日分支：把 480 步重排为 10 天 × 48 步，对每个交易日内部做均值池化，再为每个 horizon 学习 10 天 simplex 权重。
- 日内分支：从最后一个交易日的 48 个隐藏状态中，为每个 horizon 学习 48 步 simplex 权重。
- 每个 horizon 拼接其跨日与日内上下文后进入独立线性 head。权重零初始化，使初始 simplex 均匀，避免先验偏向最后一天。
- receipt/leaderboard 必须记录 readout 身份、bars per day、day weights 和 intraday weights。
- 不得引入 BatchNorm、双向卷积、对称未来 padding、未来时刻注意力或 eager materialization。

## soft-RankIC 目标契约

- 使用日期分组 batch；每个日期/horizon 至少两个有效股票时才参与。
- 对每个日期/horizon，通过两两 score 差的 sigmoid 构造可微 soft ranks；temperature 固定且必须为正。
- 以 soft ranks 与横截面 rank target 的 Pearson correlation 构造 `1 - correlation`，按有效日期/horizon 等权平均。
- 目标必须有限、mask 必须为布尔值；无有效组时退化为只优化 SmoothL1，不得产生 NaN。
- 训练元数据必须记录 soft-rank 权重、temperature、有效组数或明确的 loss identity。

## 测试驱动要求

1. 先写红灯测试，覆盖：输出 shape；day/intraday simplex；均匀初始化；完整历史对输出有直接影响；严格因果 trunk；480/48 形状约束；receipt 字段。
2. 为 soft-rank 写手算测试：完美正序损失低于逆序；梯度有限且非零；mask/日期分组正确；无有效组稳定退化；非法 temperature/weight fail closed。
3. 配置解析测试必须证明所有行为参数均来自公开配置，无隐藏默认；非法 model kind、strategy、bars per day 或 loss 参数 fail closed。
4. 小型端到端 sweep 必须能训练三类 trial，输出可恢复 checkpoint、loss/batching/readout 身份和有限 RankIC。
5. 运行 Ruff、mypy、preflight、完整 pytest、统一测试入口和 production build。

## 真实五折验收

- 输出写入全新的、Git 忽略的 `artifacts/tcn-effective-context-soft-rankic-v12-seed7/`，目标存在时拒绝覆盖。
- 分别报告三条 TCN trial 的逐折、逐 horizon RankIC、最差折、吞吐、模型步吞吐、参数量、time-to-best、day/intraday 权重。
- 使用 `context-c16-chomp-softrank20-tau10` 与固定 LSTM 做配对速度和效果比较；同时保留 trial 2 对 trial 1 的架构消融，以及 trial 3 对 trial 2 的目标消融。
- 候选最低门槛：mean RankIC `>=0.09`、5/5 folds 为正、median samples/s `>=5000`、mean RankIC 不低于父模型。
- “3–5x”速度描述只在 model-step 与端到端相对 LSTM 均 `>=3.0x` 时成立。
- 本轮的机制成功标准优先于直接超过 LSTM：多尺度读出相对父模型 mean RankIC 为正，且至少 3/5 folds 不退化；soft-RankIC 再相对多尺度 SmoothL1 为正。若任一阶段失败，记录被证伪机制并停止，不追加 trial。

## 安全与停止条件

- 不调用 PandaData、不下载新数据、不打印或写入凭据、不提交原始数据/checkpoint/runtime receipt。
- 不覆盖已有 artifact，不访问 sealed holdout，不部署、不交易、不连接券商、不做外部写入。
- 验证失败或效果为负也是完整结果；不得通过改阈值、换 fold、追跑 seed 或事后调参制造通过。

## 执行顺序

1. 完成仓库 onboarding 与 preflight，确认输入身份和现有脏工作树边界。
2. 写红灯测试并确认只因缺少 v12 行为而失败。
3. 最小实现多尺度读出、soft-RankIC 和显式配置解析，直到聚焦测试转绿。
4. 运行完整工程验证。
5. 执行真实 seed-7 五折三臂 TCN + 固定 LSTM benchmark。
6. 复算产物哈希，形成逐折、逐 horizon、速度和消融结论；不触碰 sealed test。
