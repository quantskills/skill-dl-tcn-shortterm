# TCN 有界有符号残差稳定化提示词 v15

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化执行者。v14 已证明当前无约束 signed temporal adapter 在 seeds 17/27 上不能复现 seed-7 增益，但 TCN 相对 LSTM 的 3–5x 速度目标已经完成。本轮继续优化 TCN，不更换主模型；目标是隔离并修复 signed temporal weights 的优化稳定性，而不是重新搜索 trunk、损失、数据或感受野。

## 父实验与问题陈述

- 父 artifact：`artifacts/tcn-signed-temporal-adapter-multiseed-v14/`
- 父 receipt：`c0d7a8d4e976f6aade52cd6ffd40d143a7d6bb54c8de6a48025709960ed6eada`
- 父状态：`stop_signed_candidate_unstable_v14`
- v14 确认集 signed - simplex mean RankIC：`-0.005947`
- v14 2 日/3 日 delta：`-0.011287/-0.013606`
- v14 model-step/端到端速度：`4.571x/4.335x`
- 执行前必须复算父 receipt 的全部输出哈希并确认 `sealed_test_accessed=false`；不符则 fail closed。

问题被限定为：无约束 day/intraday adapter 与 TCN trunk 使用同一 `0.003` 学习率，时间权重的行和、幅度和符号均无约束，可能造成 horizon-specific 滤波与 trunk 快速共适应、提前停止和 2/3 日负迁移。

## 可证伪假设

1. **H1：无界参数化是主因。** 若成立，同学习率的有界、行和固定 signed residual 应优于 v14 的无约束参数化，并且不再出现权重幅度漂移。
2. **H2：adapter 学习率过高是附加主因。** 若成立，在相同有界参数化下，adapter LR 从 `0.003` 降至 `0.0003` 应进一步改善 mean/worst-fold RankIC 或减少 2/3 日退化。
3. **H3：early stopping 是主因。** 只有前两条候选仍明显更早停止且后段验证指标存在恢复证据时，才允许在独立后续实验改变 patience/min epochs；v15 不修改训练预算。
4. **H4：共享 horizon 梯度冲突仍是主因。** 若有界 residual 仍主要损害 2/3 日，下一轮才测试 horizon 分组；v15 不加入 PCGrad、soft-RankIC 或独立 trunk。

## 不可变数据与训练协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- features SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 只允许 seed 7、folds 0–4、`train`/ordinary `validation`；遇到 test、sealed、未知 stage、哈希漂移或输出已存在时 fail closed。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、最大 8 epochs、patience 2、min_delta 0.002、Adam、trunk/head LR 0.003、batch 128。
- 公共 TCN：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、strict causal chomp、WeightNorm、masked SmoothL1、480 bars、bars per day 48。
- LSTM benchmark：hidden size 34、LR 0.003、batch 128、8 epochs，使用相同 seed/folds/线程/精度。

## 稳定化参数化

对每个 horizon 的 day/intraday raw logits `z`，定义：

1. `simplex(z) = softmax(z)`；
2. `centered(z) = tanh(z) - mean(tanh(z))`；
3. `weight(z) = simplex(z) + residual_scale * centered(z)`；
4. `residual_scale = 0.05`。

该参数化必须满足：

- raw logits 全零初始化，初始权重与 simplex control 完全相同；
- 每行权重和恒为 1；
- signed residual 每元素绝对值不超过 `2 * residual_scale`；
- 权重可为负，保留方向性时间滤波能力；
- 复用 control 的 raw logits 形状，不增加参数，三臂都必须是 6524 参数；
- causal trunk、readout 输入、四个 heads 及初始输出保持相同。

## 冻结实验臂

只运行三条 TCN：

1. `context-c16-chomp-smooth`：v12 simplex 控制，所有参数 LR 0.003。
2. `stable-residual-c16-chomp-smooth-lr100`：有界 signed residual，`residual_scale=0.05`，adapter LR 0.003；只检验参数化。
3. `stable-residual-c16-chomp-smooth-lr010`：相同参数化，adapter LR 0.0003；只在第 2 臂基础上检验较低 adapter LR。

不得增加第四条、改变 seed、扩大 epoch、修改门槛或使用 v14 结果回填参数。

## seed-7 预注册效果门槛

候选必须同时满足：

1. 五折 mean RankIC `>=0.09`；
2. 5/5 folds RankIC 为正；
3. 相对 simplex control 的平均 RankIC 增益严格大于 0；
4. 至少 3/5 folds 相对 control 不退化；
5. 1 日 delta `>=0`；
6. 2 日 delta `>=-0.003`；
7. 3 日 delta `>=-0.005`；
8. 5 日 delta `>=-0.005`；
9. median samples/s `>=5000`；
10. 参数量与 control 相等且为 6524。

若两条候选均通过，依次按 mean RankIC、worst-fold RankIC、3 日 delta、median samples/s 和 trial ID 确定唯一 winner。效果状态为 `stabilized_residual_seed7_effect_admitted_v15`；无候选通过为 `stop_stabilized_residual_seed7_effect_v15`。

效果 winner 还必须相对固定 LSTM 同时满足 model-step 与端到端速度 `>=3.0x`。全部通过时最终状态为 `stabilized_residual_seed7_admitted_v15` 并只授权 seeds `[17,27]`；速度失败为 `stop_stabilized_residual_seed7_speed_v15`。效果未通过时可对最高 mean RankIC 候选做描述性 LSTM 比较，但不得授权确认 seeds。

## 快速反馈环与测试

在实现前建立红灯测试，至少覆盖：

- stabilized 模型与 simplex control 参数量、初始权重和初始输出严格相同；
- 极端 raw logits 下权重行和仍为 1，signed residual 满足理论边界且能产生负权重；
- adapter 参数组与 trunk/head 无重复、无遗漏，候选分别使用 0.003/0.0003 adapter LR；
- 非 stabilized 模型拒绝 adapter LR，非法 residual scale fail closed；
- seed-7 决策拒绝错误 seed、缺 fold、重复单元、参数漂移；
- mean、逐折、1/2/3/5 日和吞吐任一门槛失败时 blocker 正确；
- 速度失败不能授权 seeds 17/27。

快速命令必须在秒级稳定复现上述结构问题；真实五折命令是最终 RankIC 反馈环。

## 证据与输出

- 配置：`config/pandadata-tcn-stabilized-signed-residual-seed7-v15.example.json`
- runner：`tasks/run_tcn_residual_stabilization.py`
- 输出：`artifacts/tcn-stabilized-signed-residual-v15-seed7/`，存在时拒绝覆盖。
- 保存带 `seed`/`model_seed` 的 epoch history、leaderboard、逐候选 summary、逐 horizon summary、checkpoint、LSTM measurements、comparison、selection 和 receipt。
- leaderboard/receipt 必须记录 `residual_scale`、`adapter_learning_rate`、optimizer 参数组身份、权重行和、负权重数和 residual 范数。
- receipt schema：`tcn-stabilized-signed-residual-v15/v1`，记录父 receipt、源哈希、代码身份、门槛、输出哈希和 `sealed_test_accessed=false`。

## 工程门禁与安全边界

- 真实训练前通过 Ruff、完整 mypy、聚焦 pytest、统一测试入口、preflight 和 production build。
- 不调用 PandaData、不补数据、不下载数据、不打印或写入凭据。
- 不访问 test/sealed、不部署、不交易、不连接券商、不进行外部写入。
- 若 seed 7 失败，停止并记录失败维度；不得追跑 seeds 17/27、不得改门槛、不得选择性汇报。
- 若 seed 7 通过，只写入授权，不在同一轮自动运行 seeds 17/27。

## 执行顺序

1. 验证 v14 parent receipt、source hashes 和 ordinary-validation 边界。
2. 运行红灯反馈环并确认它能抓住无界权重/错误 optimizer 分组。
3. 实现有界 residual 模型、显式 adapter 参数组、解析/校验、决策与 runner。
4. 通过全部工程门禁。
5. 顺序运行 seed 7 三条 TCN 与固定 LSTM，避免并发污染 CPU 吞吐。
6. 复算输出哈希，审计效果、horizon、早停、权重和速度，形成结果文档后停止。
