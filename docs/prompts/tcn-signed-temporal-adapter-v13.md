# TCN 有符号时间适配器真实验证提示词 v13

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化与真实实验执行者。TCN 必须继续作为项目主模型；LSTM 只是在相同数据、fold、seed、精度、线程、batch 和 epoch 预算下的 benchmark，不得替换 TCN。本轮只验证一个由 v12 真实证据直接导出的结构假设：多尺度完整时序读出已经修复有效感受野塌缩，但 horizon-specific simplex 权重始终接近均匀且只能为正，导致有方向的金融时间模式被平均稀释；在参数量和初始函数一致时，允许时间权重变为负数是否能提高 RankIC。

## v12 已知证据

- 冻结父模型 `skip-c16-chomp-smooth` mean RankIC `0.087731`。
- v12 `context-c16-chomp-smooth` mean RankIC `0.087487`，只低 `0.000244`，相对 LSTM model-step `3.513x`、端到端 `3.390x`。
- v12 将最后一天输入归因从约 74%–79% 降到约 43%–50%，前五天合计提高到约 23%–27%，因此完整时序读出确实修复了有效感受野塌缩。
- 但 day simplex 只在约 `0.0962..0.1064`，intraday simplex 只在约 `0.0185..0.0232`，几乎停留在均匀初始化，没有形成 horizon 专门化。
- v12 soft-RankIC mean RankIC `0.084704` 且速度低于 3x，本轮不得继续使用、扫描或修改 soft-RankIC；PCGrad 同样不进入本轮。

## 本轮唯一假设

正 simplex 加权只能做凸组合，不能表达“近期正贡献、较早历史负贡献”或相反的时间滤波。把同形状的 day/intraday 权重替换成无 bias 的线性适配器，并以相同均匀值初始化，可以在不改变初始函数、不增加参数量、不修改 TCN trunk 的情况下允许有符号时间滤波。如果该约束是瓶颈，有符号候选应稳定提高 ordinary-validation RankIC。

## 不可变数据与训练协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- 特征 SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 仅允许 `train` 与 ordinary `validation`；遇到 `test`、`sealed=true`、未知 stage、源哈希漂移或目标目录已存在时训练前 fail closed。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、seed 7、五个 expanding folds。
- 最大 8 epochs、patience 2、min_delta 0.002、Adam、lr 0.003、batch 128。
- TCN：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、strict causal chomp、WeightNorm、masked SmoothL1。
- 输入：10 天 × 48 根标准 5 分钟线 = 480 步；感受野必须按精确公式验证为 511。
- LSTM benchmark：hidden size 34、lr 0.003、batch 128、8 epochs，其他资源协议与 TCN 一致。

## 预注册实验臂

只运行两条参数匹配 TCN，不得事后追加 trial：

1. `context-c16-chomp-smooth`：v12 正 simplex 双尺度完整时序读出，作为控制。
2. `signed-context-c16-chomp-smooth`：唯一候选；相同 trunk、相同双尺度聚合、相同四个 horizon head、相同参数量和 SmoothL1，只把 day/intraday simplex logits 替换为无 bias 的有符号线性权重。

不得扫描或修改 channels、kernel、dilations、dropout、weight decay、学习率、batch、初始化、损失或训练轮数。

## 有符号读出契约

- 继续消费最终 TCN block 的完整 `[batch, channel, 480]` 隐藏序列。
- 跨日分支仍先重排成 10 天 × 48 步，并对日内 48 步均值池化，得到 `[batch, channel, 10]`。
- 日内分支仍消费最后交易日 48 个隐藏状态。
- day adapter 为 `Linear(10, 4, bias=False)`，intraday adapter 为 `Linear(48, 4, bias=False)`；对 channel 维共享，输出四个 horizon 上下文。
- day adapter 的四行全部初始化为 `1/10`，intraday adapter 的四行全部初始化为 `1/48`，保证候选与 simplex 控制的初始读出数值一致。
- 训练后权重不做 softmax、归一化、截断或符号限制；必须允许正、负和任意行和。
- 每个 horizon 拼接 day 与 intraday context 后继续使用独立 `Linear(2*channels,1)` head。
- 候选参数量必须与 simplex 控制完全相同；不相同则 fail closed。
- receipt/leaderboard 必须记录 `horizon_dual_scale_signed_adapter`、原始 day/intraday 权重、每个 horizon 的负权重数量、权重行和及参数量。

## 测试驱动要求

1. 先写红灯测试：候选输出 shape、初始读出与 simplex 控制一致、参数量完全相同、初始权重分别为 `1/10` 和 `1/48`。
2. 手动设置负权重，证明权重不会被归一化或截断且输出随之变化。
3. 继续验证 strict causal trunk、480/48 整除约束、WeightNorm 和 receipt 字段。
4. 配置解析必须显式支持 `signed_temporal_context`，不得依赖隐藏行为默认。
5. 小型 sweep 必须生成有限 RankIC、可恢复 checkpoint、有符号权重元数据和 `seeded-random` batching identity。
6. 运行 Ruff、完整 mypy、聚焦测试、统一测试入口、preflight 和 production build。

## 真实五折验收

- 输出到新的 Git 忽略目录 `artifacts/tcn-signed-temporal-adapter-v13-seed7/`，拒绝覆盖。
- 报告两条 TCN 的逐折、逐 horizon RankIC、最差折、正折数、吞吐、模型步吞吐、参数量、time-to-best。
- 候选相对控制的机制成功门槛：mean RankIC 严格为正增益，至少 3/5 folds 不退化，参数量完全相同，median samples/s `>=5000`。
- 正式候选门槛：mean RankIC `>=0.09`、5/5 folds 为正、mean RankIC 不低于控制。
- 候选与 LSTM 的 model-step 和端到端速度比均 `>=3.0x` 时，才保留“3–5x”速度陈述。
- 训练后审计每个 horizon 的 day/intraday 负权重数、权重范围和行和；如果仍接近全正均匀，则判定优化器没有利用符号自由度。
- 对候选 fold 0 checkpoint 使用相同 64 个 ordinary-validation 样本复核逐日输入归因，不得用 test/sealed 样本。

## 安全与停止条件

- 不调用 PandaData、不下载新数据、不打印或提交凭据、原始数据、checkpoint 或 runtime receipt。
- 不访问 sealed holdout、不部署、不交易、不连接券商、不做外部写入。
- 负结果也是本轮完整完成状态。若有符号候选未通过，不追加 full-480 adapter、不改变初始化、不扫描正则或增加 trial；下一阶段转向市场/行业/横截面上下文。

## 执行顺序

1. 完成 preflight，确认输入身份、已有脏工作树边界和目标不存在。
2. 写红灯测试，确认失败仅来自缺少 signed adapter。
3. 最小实现模型、配置解析、训练元数据和 receipt schema v13，直至聚焦测试转绿。
4. 完成 Ruff、mypy、统一测试入口和 build。
5. 运行真实五折两臂 TCN 与固定 LSTM benchmark。
6. 复算 receipt 输出哈希，执行权重符号和有效感受野审计，形成不可变结果文档。
