# TCN 解耦有符号残差 v16 真实 seed-7 结果

日期：2026-08-04

状态：`stop_decoupled_residual_seed7_effect_v16`

Receipt：`77731f043377cdab2aca535ff4185635a0896ec9660ad168b2255519ea1d6be6`

## 结论

v16 成功修复了 v15 的 base/residual 参数耦合：低 LR residual 候选保留了正常 LR simplex base 的学习、训练 epoch 与 control 完全一致，mean RankIC 从 v15 的 `0.086615` 恢复到 `0.087508`。但它相对 control `0.087487` 仅提高 `+0.000021`，没有达到 `0.09`，1 日 delta 为 `-0.000652`；因此不构成有效增益，不授权 seeds 17/27，也不访问 sealed test。

把独立 residual LR 从 `0.0003` 提到 `0.001` 后，residual 范数约放大 3.3 倍，但 mean RankIC 降到 `0.087165`，同样没有产生任何负时间权重。由此，v16 支持“解耦修复了训练稳定性”，但不支持“稳定的 signed temporal residual 可以提高预测效果”。signed temporal adapter 路线应在此停止，TCN 主模型继续保留。

描述性速度继续通过：最佳候选相对 LSTM 的 model-step 为 `4.238x`，端到端为 `4.066x`。失败仍然是效果问题，不是 TCN infra 或速度问题。

## 三臂聚合

| Trial | Mean RankIC | Delta vs control | Worst fold | 不退化 folds | Positive folds | Median samples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `context-c16-chomp-smooth` | 0.087487 | 0 | 0.014449 | 5/5 | 5/5 | 5974.77 | 6524 |
| `decoupled-residual-c16-chomp-smooth-lr010` | 0.087508 | +0.000021 | 0.014476 | 4/5 | 5/5 | 5866.03 | 6756 |
| `decoupled-residual-c16-chomp-smooth-lr033` | 0.087165 | -0.000322 | 0.013445 | 2/5 | 5/5 | 5956.29 | 6756 |

候选比 control 多 232 个参数，即 `+3.56%`。`lr010` 在增加容量后只得到 `+0.000021`，远小于跨折波动，不能解释为有效容量收益或机制收益。

逐折 delta：

- `lr010`：`+0.000027`、`+0.001263`、`+0.000602`、`+0.000107`、`-0.001895`。
- `lr033`：`-0.001004`、`-0.000330`、`+0.001050`、`+0.000741`、`-0.002068`。

两条候选的 best epoch 和 completed epochs 在所有 fold 上均与 control 一致；三臂平均 best epoch 为 4.0、平均 completed epochs 为 6.0。因此本轮不存在 v14/v15 高 LR 臂的提前停止差异。

## 逐 horizon 结果

| Arm | 1 日 delta | 2 日 delta | 3 日 delta | 5 日 delta |
|---|---:|---:|---:|---:|
| `lr010` | -0.000652 | +0.001417 | -0.000438 | -0.000244 |
| `lr033` | -0.000234 | +0.000477 | -0.000366 | -0.001165 |

`lr010` 通过 2/3/5 日和 4/5 folds 不退化门槛，但没有通过 mean RankIC `>=0.09` 与 1 日 delta `>=0`。提高 residual LR 略微减小 1 日负差，却牺牲 2/5 日和跨折稳定性，不能形成可接受的交换。

## 假设判定

1. **H1：base/residual 参数耦合是 v15 主因。得到支持。** 解耦后 `lr010` 从 v15 的 `0.086615` 恢复到 `0.087508`，base simplex 偏离均匀权重的幅度与 control 几乎相同，训练 epoch 也完全一致。
2. **H2：0.0003 对独立 residual 过弱，0.001 可恢复 signed 增益。未得到支持。** `lr033` residual L2 明显增大，但 mean RankIC 更低，仍没有负权重。
3. **H3：signed 时间滤波没有稳定增益。进一步得到支持。** v13 的 seed-7 无约束增益先在 v14 多 seed 失败；v15 有界共享参数失败；v16 真正解耦后只复制 control，没有产生足够大的独立效应。
4. **H4：新增容量可能制造表面收益。没有观察到。** 两条候选固定增加 232 参数，但最优增益仅 `+0.000021`，不具备实质意义。

## 权重与 optimizer 审计

候选参数组严格解耦：

- `lr010`：`base-lr-0.003+residual-lr-0.0003`；
- `lr033`：`base-lr-0.003+residual-lr-0.001`；
- base temporal 参数 232，独立 residual 参数 232；optimizer 参数无重复、无遗漏。

| Arm | Intraday weight range | Mean intraday residual L2 | Max intraday residual L2 | 负权重数 |
|---|---:|---:|---:|---:|
| `lr010` | 0.012182..0.031110 | 0.002334 | 0.004615 | 0 |
| `lr033` | 0.007562..0.034737 | 0.007737 | 0.015226 | 0 |

最终权重行和最大数值误差分别为 `2.38e-7` 和 `1.79e-7`，有界零和约束正常。`lr033` 的 residual 确实更强，但仍未跨越为负权重，也没有带来效果；这排除了“optimizer 没有更新 residual”的实现问题。

## LSTM 描述性比较

- `lr010` TCN mean RankIC：`0.0875078`。
- LSTM mean RankIC：`0.1115955`。
- 配对平均差：`-0.0240877`。
- TCN/LSTM model-step：`4.2381x`。
- TCN/LSTM 端到端：`4.0663x`。
- TCN 参数量：6756；LSTM 参数量：6124。

因为效果门槛先失败，selection 中 `relative_speed_gate_passed=false` 表示没有 effect winner 可晋级，不表示描述性速度低于 3x。

## 工程与证据

- 红灯反馈环先复现缺失的独立参数与 optimizer 分组，最小实现后聚焦测试 34 passed。
- 完整 Ruff：通过。
- 完整 mypy：59 个源码文件通过。
- 统一测试入口：通过。
- preflight：通过。
- production wheel/sdist：通过。
- v16 receipt 的 24/24 个输出 SHA-256 复算一致。
- Receipt schema：`tcn-decoupled-signed-residual-v16/v1`。
- `sealed_test_accessed=false`；未调用 PandaData、未下载数据、未访问 test/sealed、未部署或执行外部写入。

## 下一步边界

停止继续调整 signed residual 的 LR、scale、seed 或约束；不授权 seeds 17/27。保留 `context-c16-chomp-smooth` 作为当前稳定 TCN control，并继续保留已经验证的 3–5x 速度优势。

下一轮 TCN 优化应转向更上游的表征缺口：当前模型对每只股票独立编码，但主目标是同一信号日的横截面排序。优先测试“TCN 编码 + 仅使用当日可知信息的市场/行业状态条件化”，让 TCN trunk 能区分个股时序变化与共同市场冲击；LSTM 继续只作 benchmark。该方向必须另立预注册提示词与 PIT/泄漏测试，不能把横截面当天全样本统计未经约束地送入模型。
