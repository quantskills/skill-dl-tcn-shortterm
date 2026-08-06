# TCN v20：样本条件化多尺度膨胀块选择

## 任务目标

继续优化 TCN，不替换模型方向。v18/v19 已经证明样本条件化读出能够产生动态权重，且 TCN 相对固定 LSTM 仍有约 3.7x model-step 速度，但把动态能力放在最终隐藏序列的日内/跨日读出上没有转化成稳定预测增益。v19 把动态参数学习率提高后，动态变异中位数扩大约 3.07x，mean RankIC 反而从静态 control 的 `0.0874870` 降至 `0.0841727`；因此“动态读出只是学习不足”已被真实数据否定。

本轮只检验一个新假设：有效信息可能分布在 TCN 的不同 dilation block，而最终 block 已经压缩或稀释了短线信号。保留同一个 8 层因果 TCN trunk，让候选对每个样本、每个预测期限动态选择 8 个 block 的最后有效状态；静态 `HorizonSkipTCN` 是唯一对照。

## 父证据

- v19 receipt：`ca865f2a5470d0f35b50159634c85f8a774588f459f2d4b41f8cc9b36e5c64f2`。
- v19 状态：`stop_dynamic_lr_seed7_effect_v19`。
- 静态 temporal-context control mean RankIC：`0.08748700523632261`。
- v19 候选 mean RankIC：`0.08417268671094484`，delta `-0.0033143185253777614`。
- v19 候选相对 LSTM：model-step `3.71732x`，端到端 `3.58412x`。
- 固定 LSTM mean RankIC：`0.11159549099300774`。

## 可证伪假设

> 1–5 日横截面信号依赖于不同的局部/中期感受野。最终隐藏层上的动态时间加权太晚；若对 8 个 dilation block 的表征进行样本和期限条件化选择，应在保持因果性、轻量容量和 TCN 吞吐优势的同时，提升五折 RankIC。

若 scorer 明确非零且 block 权重随样本变化，但效果门禁失败，则说明当前数据/标签下“动态选择 dilation 尺度”不能带来足够增益；不得在本轮结果出来后修改阈值或偷偷搜索 hidden、scale、loss、学习率。

## 不可变合同

- 真实 PandaData 五年分钟线、八特征、480 根 5 分钟 bar、PIT top20、原标签和原五折 SHA-256 完全冻结。
- folds 必须恰为 `0..4`；seed `7`；float32；CPU；torch threads `8`。
- train/validation only；禁止 test/sealed test。
- 两组共享同一 TCN trunk：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、causal chomp、weight normalization。
- 控制组：`HorizonSkipTCN`，每个 horizon 一个静态 8-block simplex，参数量严格 6260。
- 候选：`DynamicHorizonSkipTCN`。对每个 block 的最后有效 16 维 token 使用共享 `Linear(16,4)→tanh→Linear(4,4)` scorer；输出层零初始化；动态 logits 经 `tanh`、scale 1.0 后加到静态 logits，再对 block 维 softmax。
- 候选只增加 88 参数，总参数严格 6348；不得增加额外 trunk、全连接预测层或非因果路径。
- 同 seed 初始化时，候选输出必须与静态 control 逐元素完全相等。
- 训练：SmoothL1、Adam lr `0.003`、batch 128、max epochs 8、patience 2、min delta 0.002。
- LSTM 继续复用并校验父 receipt 的固定五折 benchmark。
- 不引入 BatchNorm、未来 padding、新特征、市场横截面条件、PCGrad、额外 seed 或 sealed test。

## 公共实现合同

1. `DynamicHorizonSkipTCN.dynamic_skip_weights(block_sequences)` 返回 `[batch, 4, 8]` 样本/期限/尺度 simplex。
2. `dynamic_skip_parameters()` 返回全部且仅返回共享 scorer 的 88 个非重复参数。
3. `receipt_metadata()` 记录 readout identity、hidden、scale、动态参数量、输出 weight/bias L2。
4. `TCNTuningTrial`、公开 JSON parser、验证器和 `build_tcn_trial_model` 支持 `dynamic_horizon_skip`、`dynamic_skip_hidden`、`dynamic_skip_scale`；非法值 fail closed。
5. leaderboard 持久化上述参数和动态输出 L2；控制组字段为空。
6. v20 runner 校验父 v19 receipt 及其全部输出 SHA-256，拒绝覆盖目标目录，原子写出新 receipt。

## TDD 纵向切片

1. 红灯：`DynamicHorizonSkipTCN` 尚不存在；绿灯：同 seed 初始输出严格等于静态 control，参数量为 6260/6348。
2. 行为测试：动态权重 shape 正确、沿 block 维和为 1；训练更新后输出层非零且不同样本权重产生可测差异。
3. 因果测试：改变输入后缀不得改变任一 block 的对应历史前缀表征。
4. 红灯：parser/factory 尚不认识新模型；绿灯：合法配置 round-trip，非法 hidden/scale 或非平滑损失 fail closed。
5. 决策测试：效果、容量、动态参数使用、样本变异、吞吐或 LSTM 相对速度任一失败都不得授权新 seeds。

## 真实 seed-7 门禁

候选 `dynamic-horizon-skip-c16-chomp-smooth-h4-s1` 必须同时满足：

- mean RankIC `>=0.09`；
- 相对静态 `horizon-skip-c16-chomp-smooth` mean RankIC delta `>=0.003`；
- 五折全部 RankIC 为正；
- 至少 3/5 折不低于静态 control；
- horizon delta：1d `>=0`，2d `>=-0.003`，3d `>=-0.005`，5d `>=-0.005`；
- median samples/s `>=5000`；
- 参数量严格为 `6260/6348`，动态参数量严格为 88；
- 五折动态输出 weight L2 最小值 `>1e-12`；
- 五折 block 权重样本变异最小值 `>=1e-6`；
- 五折 simplex error 最大值 `<=1e-6`；
- 相对固定 LSTM：model-step speed `>=3x`，end-to-end speed `>=3x`。

效果或机制门禁失败：`stop_dynamic_multiscale_seed7_effect_v20`。效果通过但速度失败：`stop_dynamic_multiscale_seed7_speed_v20`。全部通过才授权 seeds 17/27；本轮不自动执行多 seed 或 sealed test。

## 产物

- 本提示词、非秘密配置、源码和 TDD 测试。
- `tcn-epoch-history.parquet`、`tcn-leaderboard.parquet`。
- `tcn-summary.parquet`、`horizon-summary.parquet`。
- `attention-diagnostics.parquet`，记录逐折 block 权重变异与 simplex error。
- 固定 LSTM measurements/environment、comparison、selection、resolved config。
- 五折 checkpoints、不可变 receipt 和中文结果报告。

## 执行命令

```powershell
python -m pytest tests/test_tcn_dynamic_multiscale.py -q
python tasks/run_tcn_dynamic_readout.py `
  --run-dir artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be `
  --split-manifest artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet `
  --config config/pandadata-tcn-stock-conditioned-multiscale-seed7-v20.example.json `
  --output-dir artifacts/tcn-stock-conditioned-multiscale-v20-seed7
python -m ruff check .
python -m mypy
python tasks/preflight.py
python tasks/test.py
python -m build --no-isolation
```

## 完成定义

只有在红绿测试、真实五折、固定 LSTM 对比、收据复核、中文结果报告和全量工程验收全部完成后才算落地。负结果是有效完成；不得根据 v20 validation 结果调整模型或门禁。
