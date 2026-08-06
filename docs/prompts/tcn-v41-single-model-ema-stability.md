# TCN v41：单模型 EMA 全局稳定性优化执行提示词

你正在维护 `skill-dl-tcn-shortterm`。本轮目标不是追逐换手率、某个期限或某个 TopK 指标，而是解决 v40 已被多种子上界实验证实的 **TCN 单模型训练方差**：三种子预测平均相对单个 relative10 TCN 同时改善 RankIC、Top return、Top precision、NDCG 和 quantile monotonicity，并显著缩小与 LSTM 的差距。需要把这种广谱稳定性尽可能压缩到一个推理时只需一次前向的 TCN 中。

## 1. 冻结事实

- 数据：v40 top50 relative10 五年普通 validation 数据。
- 主模型：`DynamicHorizonSkipTCN`，16 channels，kernel 3，dilations `[1,2,4,8,16,32,64,128]`，纯左侧因果 padding/chomp，感受野覆盖480步。
- 任务：1/2/3/5日横截面收益排序；同一输入、标签、fold 和预测输出契约。
- 训练：float32、CPU 8 threads、batch 128、Smooth L1、学习率0.003、8 epochs、固定日期顺序。
- v40 relative10 TCN：RankIC `0.079566`、Top return `0.002075`、Top precision `0.097792`、NDCG `0.568532`、quantile monotonicity `0.130717`。
- 三种子 TCN prediction ensemble：RankIC `0.090007`、Top return `0.002745`、Top precision `0.098156`、NDCG `0.578475`、quantile monotonicity `0.146848`。
- ensemble 相对单模型五项全部改善，说明方差不是局部指标问题。
- v40 单模型 TCN/LSTM 模型步和完整周期速度分别为 `4.835×`、`4.337×`。

## 2. 唯一干预变量

只新增训练期参数指数移动平均（EMA）：

- `ema_decay = 0.99`，本轮不得搜索其他 decay。
- 第一次 optimizer step 后把当前参数完整复制为 EMA shadow，避免初始化偏置。
- 后续每次 `optimizer.step()` 后仅对浮点可训练参数执行 `shadow = 0.99*shadow + 0.01*parameter`。
- 不修改 optimizer 参数、梯度、随机数消费、batch 顺序或原始模型参数；EMA 只旁路观察训练轨迹。
- control 使用原始参数做 validation/checkpoint；candidate 使用 EMA shadow 做 validation/checkpoint。
- candidate 最终只保存一份 EMA state，推理仍是单个 TCN、一次前向；不得用三模型 prediction ensemble 冒充单模型优化。
- 非浮点 buffer 原样复制；WeightNorm 参数必须按实际 state/parameter 名称处理。

control 和 EMA candidate 可重复训练，但二者的 **raw training trajectory 必须逐 tensor 完全相同**。若有任何 raw state drift，按实现错误停止，不解释为效果差异。

## 3. 明确不做

- 不改网络结构、channels、dilation、readout、特征、标签、损失、学习率、batch、epoch、fold 或 checkpoint 选择指标。
- 不新增 top-tail loss、PCGrad、temporal pooling、horizon adapter、score smoothing 或组合 buffer。
- 不处理 membership turnover，不把任何换手指标放入模型门。
- 不扩 top100；top100 需要额外补78只股票、960次PIT请求，应作为独立数据实验。
- 不读取 test、旧 sealed 或新 sealed；所有输入必须是 `stage=validation, sealed=false`。
- 不以“超过 LSTM”为强制晋级条件；LSTM 只作为描述性外部基准，避免为了击败单个数字无界调参。

## 4. 反馈环和先验假设

固定复现命令：

```powershell
python tasks/diagnose_tcn_v41_variance_gap.py --run-dir artifacts/tcn-v40-multiseed-lstm-confirmation
```

它必须返回 `red_seed_variance_gap_v41`，且 ensemble 相对单模型至少4/5项改善、RankIC改善不少于0.005。若复现不成立，停止本轮。

待检验假设：EMA 能沿同一训练轨迹降低参数抖动，使单个 TCN 在多个预测指标、多个期限和多个 seed/fold 上同步改善，同时基本保持当前速度。若只改善一个指标或只改善一个期限，假设不成立。

## 5. Phase A：seed 7 × 5 folds

先实现测试和通用 EMA 机制，再只运行 seed 7。输出 control 与 EMA 的逐 epoch 历史、raw state drift、预测、任务指标、配对 bootstrap、逐 fold/逐 horizon 差值和速度。

### 完整性门

- control/candidate raw epoch state 最大绝对误差必须为 `0`。
- 数据、split、模型结构、参数量、训练预算和随机种子身份完全一致。
- candidate checkpoint 必须来自 EMA shadow，最终 state 只包含一个正常 TCN state dict。
- sealed/test 访问必须为 false。

### 非补偿式全局效果门

以 EMA candidate 减 control 计算：

- mean RankIC delta `>= +0.002`；
- 5个fold至少3个 RankIC delta为正；
- RankIC日期block-bootstrap 95% CI下界 `>= -0.002`；
- 在 RankIC、Pearson IC、Top return、Top precision、NDCG、quantile monotonicity 六项中至少4项均值严格改善；
- Top return delta `>= -0.0001`；
- Top precision delta `>= -0.002`；
- NDCG delta `>= -0.001`；
- quantile monotonicity delta `>= -0.002`；
- 4个期限至少3个 RankIC delta为正，最差期限 delta `>= -0.003`。

这些门是非补偿式的：RankIC 大涨不能覆盖 NDCG/收益/期限崩溃；某个 Top 指标上涨也不能覆盖全池排序退化。

### 速度门

- candidate/control model-step throughput retention `>= 0.90`；
- candidate/control complete-cycle retention `>= 0.85`；
- 参数量、单次前向次数与 control 完全相同；
- 用 v40 同硬件 LSTM 收据折算后的 candidate/LSTM model-step ratio 仍需 `>= 3.0×`。

只有完整性、全局效果和速度三门同时通过，状态才是 `ema_seed7_holistic_admitted_v41` 并授权 Phase B；否则停止且不得调 decay 重跑。

## 6. Phase B：seeds 17/27 确认

只在 Phase A 通过后执行，复用完全冻结的 `ema_decay=0.99` 和全部协议。与 seed7 合并为3 seeds×5 folds。

多种子门：

- mean RankIC delta `>= +0.002`；
- 15个seed-fold单元至少9个 RankIC delta为正；
- 每个seed的平均 RankIC delta不得低于 `-0.001`；
- 6项预测指标至少4项严格改善，并继续满足 Phase A 的四项下行容忍；
- 4个期限至少3个 RankIC delta为正，最差期限 `>= -0.002`；
- RankIC block-bootstrap 95% CI下界 `>= -0.001`；
- model-step retention `>=0.90`、complete-cycle retention `>=0.85`、折算TCN/LSTM model-step ratio `>=3.0×`。

通过状态为 `ema_single_tcn_multiseed_admitted_v41`。LSTM比较必须报告，但无论胜负都不得改变预注册门槛。

## 7. 工程要求

- 先写 EMA 数学、首步复制、参数恢复、非浮点状态、raw trajectory不漂移和 fail-closed 合约测试。
- 配置、输出和receipt不得含凭据；写入使用临时目录和原子替换，拒绝覆盖已有权威产物。
- 保存 `predictions.parquet`、`task-aligned-metrics.parquet`、`comparison.json`、`bootstrap.parquet`、`horizon-summary.parquet`、`trajectory-audit.json`、`timing.json`、`model-gate.json`、`receipt.json` 和 `report.md`。
- 运行 Ruff、Mypy、完整pytest、preflight和production build。

## 8. 解释上限与停止条件

- 若 Phase A 失败：结论是固定EMA不能把种子集成上界压缩为单TCN，停止EMA方向。
- 若 Phase A通过而Phase B失败：结论是EMA只对seed7有效，停止，不调decay。
- 若Phase B通过：只能称为ordinary-validation单模型稳定性候选；不得宣称Alpha-ready、sealed通过、TCN普遍优于LSTM或可部署。
- 不允许因某一局部指标失败而修改门槛，也不允许因某一局部指标成功而忽略其他门。
