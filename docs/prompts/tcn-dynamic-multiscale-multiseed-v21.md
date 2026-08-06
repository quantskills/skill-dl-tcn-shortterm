# TCN v21：动态多尺度 seeds 17/27 确认

## 任务目标

继续以 TCN 为主模型，不修改 v20 已通过 seed-7 门禁的 `DynamicHorizonSkipTCN`。本轮只回答一个问题：v20 相对静态 `HorizonSkipTCN` 的 mean RankIC 增量 `+0.008647`，是否能在预先授权的 seeds 17/27 上稳定复现。

本轮不是新的超参数搜索。模型结构、初始化规则、损失、数据、fold、训练预算、动态 scorer、容量和门槛全部冻结；禁止根据 seeds 17/27 的结果调整任何设置。

## 父证据与授权

- 父 artifact：`artifacts/tcn-stock-conditioned-multiscale-v20-seed7`
- 父 receipt：`9771fb41fa3dfd9a3a01ab6fe50ddafd0c69fd7d43b08a29bc05ba7d9710a76b`
- 父状态：`dynamic_multiscale_seed7_admitted_v20`
- 父 winner：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`
- 父授权 seeds：`[17,27]`
- 父 seed-7 候选/control mean RankIC：`0.0963784870 / 0.0877312790`
- 父 seed-7 配对增量：`+0.0086472080`
- 父 seed-7 相对 LSTM：model-step `3.535003x`，端到端 `3.427631x`

执行前必须验证父 receipt ID、selection、全部 20 个输出 SHA-256、`sealed_test_accessed=false` 和 confirmation seeds 授权；任一漂移均 fail closed。

## 不可变数据协议

- 源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be`
- features SHA-256：`6a494fe56cd4594bfd24f1c3a053d93bf704af5acfb97089a095d69c4bb2983e`
- window index SHA-256：`977033bd1115a2d649600905ee9f3c115ceed6fa0ceacff24587ee5da4966db1`
- labels SHA-256：`4a64e2d3e6ff9285df5e229661b50f66535c0dfeed22779def541a848f378594`
- split manifest SHA-256：`7de75cf192d807b03ef777f496caca5a1ee774f083bd23afc04239f006a9a672`
- universe SHA-256：`fcf2e52245f47e2c98b4f39a055cf5e4f02e80a3805538282877828a057bcd7c`
- input manifest SHA-256：`f7907b1972a73f34f31a0577da6f3f9ba0c49ae519c46beeed2f54d64878d3e5`
- 只允许 ordinary train/validation；禁止 test、sealed 或未知 stage。
- 真实 PandaData 五年分钟线、8 特征、480 根 5 分钟 bar、PIT top20、原标签、五个 expanding folds 完全冻结。

## 冻结训练协议

- 只运行 seeds `[17,27]`，每个 seed 五折；不得追跑第三个新 seed。
- CPU、float32、torch threads 8、DataLoader workers 0。
- SmoothL1、Adam、lr 0.003、batch 128、max epochs 8、patience 2、min delta 0.002。
- TCN trunk：channels 16、kernel 3、dilations `[1,2,4,8,16,32,64,128]`、dropout 0、causal chomp、WeightNorm。
- control：`horizon-skip-c16-chomp-smooth`，参数 6260。
- candidate：`dynamic-horizon-skip-c16-chomp-smooth-h4-s1`，共享 `16→4→4` scorer、scale 1.0、输出层零初始化、参数 6348、动态容量 88。
- LSTM：hidden 34、SmoothL1、Adam lr 0.003、batch 128、8 epochs；必须使用 seeds 17/27 相同数据与 folds 的真实 measurements，不得复用 seed 7 的速度数值。允许复用 v14 已完成的同协议 seeds 17/27 LSTM，但必须验证其 receipt、全部输出哈希、源 SHA、环境、超参数和 10 个 seed/fold 单元覆盖。

### 固定 LSTM 证据

- artifact：`artifacts/tcn-signed-temporal-adapter-multiseed-v14`
- receipt：`c0d7a8d4e976f6aade52cd6ffd40d143a7d6bb54c8de6a48025709960ed6eada`
- measurements SHA-256：`4dd4261348cf9ad38e21f5a49da6eacf3c14e14964389aa661a1d2a299ad9754`
- environment SHA-256：`10535efa6b034c45dcaf5ef443e3f8f6269ef925c902bf2ccea39b0f9e3009e8`
- 该证据必须包含且仅包含 base seeds 17/27 × folds 0..4 的 LSTM；参数量 6124、CPU、float32、8 threads、workers 0、hidden 34、lr 0.003、batch 128、8 epochs。
- 固定证据只替代重复 LSTM 训练，不替代 v21 TCN seeds 17/27 的重新训练，不改变 effect 或 speed 门禁。第一次全量 v21 尝试因 15 分钟基础设施超时终止、未产生 selection/receipt，不属于模型试验结果。

## v21 准入判定

正式判定只使用 seeds 17/27 的 10 个候选单元及其同 seed、同 fold control；seed 7 不得重复加权。

必须同时满足：

1. 候选跨 10 单元 mean RankIC `>=0.09`。
2. 候选 10/10 单元 RankIC 为正。
3. 候选相对 control 的配对 mean RankIC delta `>=0.003`。
4. seed 17 和 seed 27 各自 mean delta 都严格大于 0。
5. 每个 seed 至少 3/5 folds 不低于同 seed control。
6. 聚合 horizon delta：1d `>=0`，2d `>=-0.003`，3d `>=-0.005`，5d `>=-0.005`。
7. 候选 median samples/s `>=5000`。
8. 参数量严格为 control 6260、candidate 6348、delta 88。
9. 候选五折×两 seed 的动态输出 weight L2 最小值 `>1e-12`。
10. 动态 block 权重样本变异最小值 `>=1e-6`，simplex error 最大值 `<=1e-6`。
11. 相对同 seed LSTM：model-step speed `>=3x`、end-to-end speed `>=3x`。

全部通过：`dynamic_multiscale_multiseed_confirmed_v21`。效果或机制失败：`stop_dynamic_multiscale_unstable_v21`。仅速度失败：`stop_dynamic_multiscale_speed_v21`。

无论结果如何，`sealed_test_authorized=false`、`sealed_test_accessed=false`。

## 公共实现与 TDD 合同

1. `evaluate_dynamic_multiscale_multiseed()` 接受 leaderboard、动态尺度 diagnostics、LSTM comparison 和显式门槛，返回状态、blockers、seed summary、horizon summary 和聚合指标。
2. 缺 seed、缺 fold、重复单元、trial 漂移、非有限数、容量漂移或 diagnostics 覆盖不完整必须抛出 `ContractError`。
3. 先写红灯：稳定样例准入；单 seed 平均退化、配对增量不足、机制未启用和速度不足分别产生预期停止状态；缺 fold fail closed。
4. runner 必须拒绝覆盖、拒绝 secret-like 配置、验证父 receipt/授权/输出哈希、源 SHA 和 sealed/stage。
5. runner 顺序训练两个 seed，保存 20 个 TCN checkpoints、经 receipt 验证的真实 LSTM measurements、diagnostics、selection、resolved config 和 receipt。
6. 将父 seed 7 leaderboard 与新 seeds 17/27 合并为三 seed 描述性汇总；该汇总不得改变 v21 正式判定。

## 输出

- 输出目录：`artifacts/tcn-dynamic-multiscale-multiseed-v21`
- `tcn-epoch-history.parquet`
- `tcn-leaderboard.parquet`
- `attention-diagnostics.parquet`
- `seed-summary.parquet`
- `horizon-summary.parquet`
- `three-seed-summary.parquet`
- `three-seed-horizon-summary.parquet`
- `lstm-measurements.parquet`、`lstm-environment.json`、`comparison.json`
- 20 个 TCN checkpoints、`selection.json`、`config.resolved.json`、不可变 receipt 和中文结果报告。

## 执行命令

```powershell
python -m pytest tests/test_tcn_dynamic_multiscale_confirmation.py -q
python tasks/run_tcn_multiseed_confirmation.py `
  --run-dir artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be `
  --split-manifest artifacts/tcn-stability-expanding-v4/validation-stability-manifest.parquet `
  --config config/pandadata-tcn-dynamic-multiscale-multiseed-v21.example.json `
  --output-dir artifacts/tcn-dynamic-multiscale-multiseed-v21
python -m ruff check .
python -m mypy
python tasks/preflight.py
python tasks/test.py
python -m build --no-isolation
```

## 完成定义

只有红绿测试、父证据复核、seeds 17/27 真实 TCN 与 LSTM、正式新-seed 判定、三 seed 描述性汇总、输出哈希、receipt、结果报告和全量工程验收全部完成才算落地。负结果也是有效完成，不得事后调参。
