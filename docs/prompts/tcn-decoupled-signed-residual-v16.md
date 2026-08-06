# TCN 解耦有符号残差提示词 v16

你是 `skill-dl-tcn-shortterm` 项目的 TCN 优化执行者。v15 已证明：有界 residual 与较低 adapter LR 能消除 v14 的提前停止和 2/3 日大幅退化，但因为 simplex base 与 residual 共用同一组 logits，降低 LR 也冻结了正常的正时间权重学习，最终没有超过 simplex control。本轮继续以 TCN 为主模型，只测试“正常 LR simplex base + 独立低 LR signed residual”的真正解耦结构。

## 父实验与授权边界

- 父 artifact：`artifacts/tcn-stabilized-signed-residual-v15-seed7/`
- 父 receipt：`c3a630c8a9365a3e51ac49cd7d577c4a1f5fda540241bfa191d723370045f23e`
- 父状态：`stop_stabilized_residual_seed7_effect_v15`
- v15 control/lr100/lr010 mean RankIC：`0.087487/0.084230/0.086615`
- v15 lr010 2/3/5 日 delta：`-0.000106/-0.001164/-0.000167`
- v15 lr010 五折后负时间权重数：0
- 执行前必须复算父 receipt 全部输出哈希并确认 `sealed_test_accessed=false`；不符则 fail closed。

v15 失败后没有授权新 seed。本轮 seed 7 结构实验由用户明确授权，但不得自动访问 seeds 17/27 或 sealed test。

## 可证伪假设

1. **H1：base/residual 参数耦合是 v15 的主因。** 若成立，独立 residual LR 0.0003 时，正常 LR 0.003 的 simplex base 应保持 control 的时间加权能力，候选不再系统性低于 control。
2. **H2：0.0003 对独立 residual 仍过弱。** 若成立，只把 residual LR 提到 0.001 应增加 residual 范数或负权重使用并改善 1 日 RankIC，同时不复现 LR 0.003 的提前停止与 3 日大幅退化。
3. **H3：signed 时间滤波没有稳定增益。** 若两条解耦候选在保留 simplex base 后仍不通过，则停止继续追 signed adapter；不得通过增加 seed、改变门槛或选择性 horizon 汇报挽救。
4. **H4：新增容量而非 residual 机制产生表面收益。** 候选明确增加 232 个参数；所有结论必须标注 3.56% 容量增长，不能称为等参数改进。

## 不可变数据与训练协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`
- features SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- ordinary-validation manifest：`artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet`
- manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- 只允许 seed 7、folds 0–4、train/ordinary validation；遇到 test、sealed、未知 stage、哈希漂移或输出已存在时 fail closed。
- CPU、float32、PyTorch threads 8、DataLoader workers 0、最大 8 epochs、patience 2、min_delta 0.002、Adam、batch 128。
- 公共 TCN：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、strict causal chomp、WeightNorm、masked SmoothL1、480 bars、48 bars/day。
- trunk、heads 和 simplex base LR 均为 0.003。
- LSTM benchmark：hidden size 34、LR 0.003、batch 128、8 epochs，相同 seed/folds/线程/精度。

## 解耦参数化

每个 horizon 分别维护两套参数：

- `base_logits`：day `[4,10]`、intraday `[4,48]`，零初始化，按 LR 0.003 学习；
- `residual_logits`：相同形状，独立零初始化，按冻结的 residual LR 学习。

权重定义：

1. `simplex = softmax(base_logits)`；
2. `centered_residual = tanh(residual_logits) - mean(tanh(residual_logits))`；
3. `weight = simplex + 0.05 * centered_residual`。

必须满足：

- residual 初始严格为零，候选初始输出与 control 完全相同；
- base logits 与 residual logits 是不同 Parameter，不共享 storage；
- 每行权重和恒为 1；
- residual 每元素绝对值不超过 0.1；
- base logits 始终位于 LR 0.003 参数组，只有 residual logits 位于低 LR 参数组；
- optimizer 参数组无重复、无遗漏；
- 候选参数量固定为 6756，control 固定为 6524，差值只能是 232 个 residual 参数。

## 冻结实验臂

只运行三条 TCN：

1. `context-c16-chomp-smooth`：simplex control，6524 参数，所有参数 LR 0.003。
2. `decoupled-residual-c16-chomp-smooth-lr010`：独立 residual，scale 0.05，residual LR 0.0003，6756 参数。
3. `decoupled-residual-c16-chomp-smooth-lr033`：结构完全相同，只把 residual LR 改为 0.001，6756 参数。

不得增加第四条、改变 residual scale、修改 trunk/loss/budget 或引入 PCGrad、soft-RankIC、dropout、weight decay。

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
10. control 参数量为 6524、候选为 6756，且每折一致。

若两条候选都通过，依次按 mean RankIC、worst-fold RankIC、3 日 delta、median samples/s、trial ID 选唯一 winner。效果状态为 `decoupled_residual_seed7_effect_admitted_v16`；无候选通过为 `stop_decoupled_residual_seed7_effect_v16`。

效果 winner 相对固定 LSTM 的 model-step 与端到端速度还必须均 `>=3.0x`。全部通过时状态为 `decoupled_residual_seed7_admitted_v16`，仅授权 seeds `[17,27]`；速度失败为 `stop_decoupled_residual_seed7_speed_v16`。效果失败时只允许对最高 mean RankIC 候选做描述性 LSTM 比较。

## 红灯反馈环

实现前用秒级测试锁定：

- candidate/control 初始输出严格一致；
- candidate 比 control 恰好多 232 个参数；
- base/residual 参数对象及 storage 独立；
- 修改 residual 不改变 base simplex，修改 base 不改变 residual；
- 极端 residual logits 下行和仍为 1、residual 有界且权重可为负；
- optimizer 的 base 组包含 trunk、heads、base logits，residual 组只包含 residual logits；
- 两档 residual LR 正确，参数组无重复、无遗漏；
- 错误 model kind、缺 residual LR、非法 scale、错误参数量、错误 seed/fold、重复单元全部 fail closed；
- mean、fold、horizon、吞吐和速度任一门槛失败时产生正确 blocker/status。

真实五折是最终 RankIC 反馈环；结构单测转绿不代表效果已经修复。

## 证据与输出

- 配置：`config/pandadata-tcn-decoupled-signed-residual-seed7-v16.example.json`
- runner：`tasks/run_tcn_decoupled_residual.py`
- 输出：`artifacts/tcn-decoupled-signed-residual-v16-seed7/`，存在时拒绝覆盖。
- 保存带 seed/model_seed 的 epoch history、leaderboard、candidate summary、horizon summary、checkpoints、LSTM measurements、comparison、selection 和 receipt。
- leaderboard/receipt 必须记录 base LR、residual LR、residual scale、参数组身份、base/residual 参数量、simplex 权重、最终权重、负权重数、权重行和和 residual L2。
- receipt schema：`tcn-decoupled-signed-residual-v16/v1`，记录 v15 parent、源哈希、代码身份、容量差异、门槛、输出哈希和 `sealed_test_accessed=false`。

## 工程门禁与安全边界

- 真实训练前通过 Ruff、完整 mypy、聚焦 pytest、统一测试入口、preflight 和 production build。
- 不调用 PandaData、不补数据、不下载数据、不写入或打印凭据。
- 不访问 test/sealed、不部署、不交易、不连接券商、不进行外部写入。
- seed 7 失败时停止，不运行 seeds 17/27、不改门槛、不增加实验臂。
- seed 7 通过时只记录授权，不在同一轮自动运行确认 seeds。

## 执行顺序

1. 复核 v15 receipt、source hashes 与 ordinary-validation 边界。
2. 建立并运行红灯反馈环。
3. 实现真正解耦的模型、optimizer 分组、配置解析、决策和 runner。
4. 完成全部工程门禁。
5. 顺序运行 seed 7 三臂 TCN 与固定 LSTM，避免并发污染 CPU 吞吐。
6. 复算输出哈希，审计效果、horizon、早停、权重符号、容量和速度，形成结果文档后停止。
