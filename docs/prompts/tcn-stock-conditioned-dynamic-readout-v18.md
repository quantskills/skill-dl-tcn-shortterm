# TCN v18：个股样本相关动态时序读出

## 任务目标

在不放弃 TCN、不更换真实数据、不改变标签、损失、训练预算、折叠或 LSTM 基准的前提下，验证当前 TCN 预测效果受限的主要原因，是否是 `TemporalContextTCN` 对所有个股共用一组静态日间/日内时间权重。实现一个轻量、因果、样本相关的动态时序读出，并在真实五年分钟线、五折 expanding validation、seed 7 上与现有 TCN 控制组和固定 LSTM benchmark 比较。

这不是新的模型家族搜索，也不是用其他模型替代 TCN。TCN 因果卷积主干保持不变；唯一实验变量是 TCN 隐状态的时间读出。

## 已知证据

- 控制组 `context-c16-chomp-smooth` 五折 mean RankIC 为 `0.08748700523632261`。
- 固定 LSTM benchmark 五折 mean RankIC 为 `0.11159549099300774`。
- v17 TCN 候选相对 LSTM 的模型步速度为 `3.71524478606347x`，端到端速度为 `3.525780015349283x`，速度目标已满足。
- v11 PCGrad 只带来约 `+0.000216`，不能解释主要效果缺口。
- v17 共享 PIT 市场门控 mean RankIC 为 `0.08336592016618974`，相对控制组下降 `-0.004121085070132868`。
- 当前 `TemporalContextTCN` 的 `day_logits[H,D]` 和 `intraday_logits[H,T]` 对所有股票样本相同；LSTM 的状态门控随样本变化。
- 输入窗口 480 根 5 分钟 bar，`bars_per_day=48`，TCN-lite 感受野覆盖完整输入；因果 chomp、weight normalization、memmap 数据路径均保持。

## 待验证假设

> TCN 主干已能编码有效分钟序列，但静态 simplex 时间读出把所有股票压到相同时间模板，限制了个股异质性表达。以零初始化、低秩、样本相关的时间 logits 残差替换静态读出，可在不破坏初始控制组函数和速度优势的情况下提高 RankIC。

## 不可变实验合同

- 数据：现有 PandaData 五年真实分钟线 immutable run。
- 样本、特征、标签、universe、split SHA-256 必须与配置完全一致。
- folds 必须恰为 `0..4`；只允许 train/validation，禁止 sealed test。
- seed `7`、float32、CPU、torch threads `8`。
- 训练：SmoothL1、Adam、lr `0.003`、batch `128`、max epochs `8`、patience `2`、min delta `0.002`。
- TCN：channels `16`、kernel `3`、dilations `[1,2,4,8,16,32,64,128]`、dropout `0`、chomp causal padding、weight normalization。
- LSTM：使用经父 receipt 校验的 v17 固定 benchmark；不得按本轮结果重新调参。
- 不新增横截面 rank 特征、不改 label、不使用未来 padding、不引入 BatchNorm、不访问 sealed test。

## 唯一候选机制

实现 `DynamicTemporalContextTCN`，继承现有 `TemporalContextTCN`。

1. 先由原 TCN 主干得到 `sequence[B,C,480]`。
2. 日间 token 为每个交易日内均值：`daily[B,D,C]`，其中 `D=10`。
3. 日内 token 为最后一个交易日：`intraday[B,T,C]`，其中 `T=48`。
4. 日间和日内各使用一个共享低秩 scorer：`Linear(C,4) -> tanh -> Linear(4,H)`，`H=4`。
5. scorer 的最后一层权重和 bias 必须零初始化。
6. 动态权重：

   `softmax(static_logits + 1.0 * tanh(dynamic_logits), temporal_axis)`

7. 以每个样本、每个 horizon 的动态权重加权相应 token，再复用原四个线性 heads。
8. 新增参数必须精确为 `176`，总参数 `6700`；控制组为 `6524`。
9. 公共诊断接口返回日间 `[B,H,D]` 和日内 `[B,H,T]` 权重。
10. receipt 必须记录 readout identity、hidden size、scale、新增参数数和两个输出层联合 L2。

## TDD 顺序

先写失败测试，确认失败原因是功能缺失，再实现最小代码：

1. 同一 manual seed 下，候选初始化输出与控制组逐元素一致。
2. 动态权重形状正确、有限、沿时间轴和为 1。
3. 在公开接口上设置一个确定的非零 scorer 后，不同样本产生不同时间权重。
4. 改变输入未来后，`encode_sequence` 的历史前缀不变，证明卷积路径仍严格因果。
5. 控制组/候选参数量严格为 `6524/6700`，差值 `176`。
6. config parser 要求候选显式给出 `dynamic_attention_hidden` 与 `dynamic_attention_scale`，非法值 fail closed。
7. tuning factory 构造正确模型；tiny sweep 后输出层 L2 大于零并写入 leaderboard。
8. v18 evaluator 对容量、动态机制使用、五折效果、四期限、吞吐和相对 LSTM 速度全部 fail closed。
9. runner 拒绝覆盖输出目录、拒绝父 receipt/hash 漂移、拒绝 sealed split，并生成不可变 receipt。

## 真实 seed-7 门禁

候选 `dynamic-context-c16-chomp-smooth-h4-s1` 必须同时满足：

- mean RankIC `>= 0.09`；
- 相对控制组 mean RankIC delta `>= 0.003`；
- 五折全部 RankIC 为正；
- 至少 3/5 折不低于控制组；
- horizon delta：1d `>=0`，2d `>=-0.003`，3d `>=-0.005`，5d `>=-0.005`；
- median samples/s `>=5000`；
- 参数量严格为 `6524/6700`；
- 五折 `dynamic_attention_output_l2` 最小值 `>1e-12`；
- 五折样本间动态权重变异证据最小值 `>1e-6`；
- 相对固定 LSTM：model-step speed `>=3x`，end-to-end speed `>=3x`。

任一效果或机制门禁失败：状态为 `stop_dynamic_readout_seed7_effect_v18`，不授权多 seed。效果通过但速度失败：`stop_dynamic_readout_seed7_speed_v18`。全部通过才允许 seeds 17/27 确认；本轮不得自动执行确认或 sealed test。

## 产物

- 源码、红绿测试、示例配置、独立 runner。
- `tcn-epoch-history.parquet`
- `tcn-leaderboard.parquet`
- `tcn-summary.parquet`
- `horizon-summary.parquet`
- `attention-diagnostics.parquet`
- 经父 receipt 校验后复制的 `lstm-measurements.parquet` / environment
- `comparison.json`、`selection.json`、`config.resolved.json`
- 五折 checkpoints 和 `receipt.json`
- 中文结果报告，明确区分事实、推断和下一步。

## 执行命令

```powershell
python -m pytest tests/test_tcn_dynamic_readout.py -q
python tasks/run_tcn_dynamic_readout.py `
  --run-dir artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be `
  --split-manifest artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet `
  --config config/pandadata-tcn-stock-conditioned-dynamic-readout-seed7-v18.example.json `
  --output-dir artifacts/tcn-stock-conditioned-dynamic-readout-v18-seed7
python tasks/preflight.py
python tasks/test.py
```

## 完成定义

只有在提示词、测试、实现、真实五折产物、receipt、结果报告、完整 preflight 和完整测试均完成后，本任务才完成。不得因候选失败而改门槛、替换 TCN、访问 sealed test 或隐去负结果。
