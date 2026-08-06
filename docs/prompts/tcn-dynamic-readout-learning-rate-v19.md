# TCN v19：动态读出独立学习率

## 任务目标

继续优化 TCN，不替换模型方向。v18 已证明个股样本相关动态时间读出能够工作并保持 3x 以上速度，但五折动态权重的样本间变异仅约 `0.0010–0.0032`，mean RankIC 只改善 `+0.0001767`。v19 只检验一个原因：动态读出的 176 个参数与 6524 个基础参数共用 `0.003` 学习率，是否使动态条件项在 8 epoch 预算内学习不足。

本轮将动态参数学习率解耦为 `0.01`，基础 TCN、静态 logits 和预测 heads 继续使用 `0.003`。不得同时修改 bias、模型宽度、动态 scale、损失、数据、标签、折叠、epoch、早停或 LSTM benchmark。

## 父证据

- v18 receipt：`dd837cd6dffd6081d7decfaa42ff5c2754531fcdf7fdf57e29e2fb455f00b673`。
- v18 状态：`stop_dynamic_readout_seed7_effect_v18`。
- 控制组 mean RankIC：`0.08748700523632261`。
- v18 动态候选 mean RankIC：`0.0876636583338836`，delta `+0.00017665309756100152`。
- v18 动态 TCN 相对 LSTM：model-step `3.717551x`，端到端 `3.581821x`。
- v18 每折综合动态权重变异：`0.0010390, 0.0010323, 0.0017099, 0.0014674, 0.0031623`。

## 可证伪假设

> v18 的动态 scorer 不是结构无效，而是在固定 8 epoch 内相对基础 TCN 学得过慢。保持候选函数、容量和初始化完全不变，只把 176 个动态参数的 Adam 学习率由 `0.003` 提到 `0.01`，应显著增加逐股票动态权重变异，并把该变异转化为稳定 RankIC 增益。

如果动态权重变异明显增加但 RankIC 仍不过门禁，则“动态项仅仅学习不足”被否定，后续不应继续搜索该学习率。

## 不可变合同

- 真实 PandaData 五年分钟线、八特征、480 根 5 分钟 bar、PIT top20、原标签和原五折 SHA-256 完全冻结。
- folds 必须恰为 `0..4`；seed `7`；float32；CPU；torch threads `8`。
- train/validation only；禁止 test/sealed test。
- TCN trunk：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、causal chomp、weight normalization。
- 动态读出：hidden 4、scale 1.0、两个 `16 -> 4 -> 4` scorer、输出层零初始化，总参数 6700，动态参数精确 176。
- 训练：SmoothL1、Adam、batch 128、max epochs 8、patience 2、min delta 0.002。
- 基础参数学习率 `0.003`；唯一变化是动态参数学习率 `0.01`。
- LSTM 继续复用并校验父 receipt 的固定五折 benchmark。
- 不引入 BatchNorm、未来 padding、新特征、横截面数据、PCGrad、额外 seed 或 sealed test。

## 公共实现合同

1. `DynamicTemporalContextTCN.dynamic_attention_parameters()` 返回全部且仅返回两个 scorer 的 176 个参数，非空、唯一。
2. `TCNTuningTrial` 新增可选 `dynamic_attention_learning_rate`；为 `None` 时完整复现 v18 的单组 optimizer。
3. 当其为 `0.01` 时，optimizer 必须建立两个完备且互斥的参数组：
   - base：6524 参数，lr `0.003`；
   - dynamic attention：176 参数，lr `0.01`。
4. 动态学习率只允许用于 `dynamic_temporal_context`，必须有限、为正且不超过基础学习率的 10 倍。
5. leaderboard/receipt 记录动态学习率、动态参数量、optimizer group identity、输出 weight L2 和 bias L2。
6. v19 runner 校验父 v18 receipt 及全部输出 SHA-256，拒绝覆盖目标目录，原子写出新 receipt。

## TDD 纵向切片

1. 红灯：公开动态参数接口与独立 optimizer group 尚不存在。
2. 绿灯：同 seed 下 v19 候选初始输出仍与 v18 候选、静态 control 完全一致；参数量不变。
3. 红灯：parser/plan 尚不能保存和约束动态学习率。
4. 绿灯：合法配置形成 `6524@0.003 + 176@0.01`，非法模型或数值 fail closed。
5. 行为测试：同一初始模型、同一 batch、同一 loss 的第一次 Adam 更新中，base 参数更新与 v18 相同，动态参数更新幅度按学习率比率放大；不测试私有 optimizer 实现。
6. 决策测试：效果、容量、动态参数使用、相对父变异和速度任一失败都不得授权新 seeds。

## 真实 seed-7 门禁

候选 `dynamic-context-c16-chomp-smooth-h4-s1-dlr1e2` 必须同时满足：

- mean RankIC `>=0.09`；
- 相对静态 control mean RankIC delta `>=0.003`；
- 五折全部 RankIC 为正；
- 至少 3/5 折不低于静态 control；
- horizon delta：1d `>=0`，2d `>=-0.003`，3d `>=-0.005`，5d `>=-0.005`；
- median samples/s `>=5000`；
- 参数量严格为 `6524/6700`，动态参数量严格为 176；
- 五折动态输出 weight L2 最小值 `>1e-12`；
- 五折综合动态权重变异最小值 `>=0.002`；
- 相对 v18 同折动态权重变异的配对中位比率 `>=2.0`；
- 相对固定 LSTM：model-step speed `>=3x`，end-to-end speed `>=3x`。

效果或机制门禁失败：`stop_dynamic_lr_seed7_effect_v19`。效果通过但速度失败：`stop_dynamic_lr_seed7_speed_v19`。全部通过才授权 seeds 17/27；本轮不自动执行多 seed 或 sealed test。

## 产物

- 提示词、spec、issues、非秘密配置、源码和 TDD 测试。
- `tcn-epoch-history.parquet`、`tcn-leaderboard.parquet`。
- `tcn-summary.parquet`、`horizon-summary.parquet`。
- `attention-diagnostics.parquet`，包含与 v18 的逐折变异比率。
- 固定 LSTM measurements/environment、comparison、selection、resolved config。
- 五折 checkpoints、不可变 receipt 和中文结果报告。

## 执行命令

```powershell
python -m pytest tests/test_tcn_dynamic_learning_rate.py -q
python tasks/run_tcn_dynamic_readout.py `
  --run-dir artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be `
  --split-manifest artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet `
  --config config/pandadata-tcn-dynamic-readout-learning-rate-seed7-v19.example.json `
  --output-dir artifacts/tcn-dynamic-readout-learning-rate-v19-seed7
python -m ruff check .
python -m mypy
python tasks/preflight.py
python tasks/test.py
python -m build --no-isolation
```

## 完成定义

只有在红绿测试、独立参数组、真实五折、父变异对照、收据复核、结果报告和全量工程验收全部完成后才算落地。负结果是有效完成；不得根据 v19 validation 结果修改学习率或门禁。
