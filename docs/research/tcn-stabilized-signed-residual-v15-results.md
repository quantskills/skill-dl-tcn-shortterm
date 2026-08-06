# TCN 有界有符号残差 v15 真实 seed-7 结果

日期：2026-08-04

状态：`stop_stabilized_residual_seed7_effect_v15`

Receipt：`c3a630c8a9365a3e51ac49cd7d577c4a1f5fda540241bfa191d723370045f23e`

## 结论

v15 成功修复了无约束 signed adapter 的幅度漂移和较高学习率导致的提前停止，但没有得到相对 simplex TCN 的正预测增益，因此停止在 seed-7 ordinary validation，不授权 seeds 17/27，也不访问 sealed test。

同学习率有界臂 `lr100` 的 mean RankIC 为 `0.084230`，比 control 低 `0.003257`；低 adapter LR 臂 `lr010` 恢复到 `0.086615`，比 `lr100` 高 `0.002385`，但仍比 control `0.087487` 低 `0.000872`，且只有 2/5 folds 不退化。低 LR 把 2/3/5 日退化分别压至 `-0.000106/-0.001164/-0.000167`，说明高 adapter LR 确实是 v14 中周期不稳定的原因之一；1 日仍下降 `-0.002050`。

描述性速度继续通过：`lr010` 相对 LSTM 的 model-step 为 `3.892x`，端到端为 `3.751x`。失败仍然只在效果，不在 TCN infra 或速度。

## 三臂聚合

| Trial | Mean RankIC | Delta vs control | Worst fold | 不退化 folds | Positive folds | Median samples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `context-c16-chomp-smooth` | 0.087487 | 0 | 0.014449 | 5/5 | 5/5 | 5866.32 | 6524 |
| `stable-residual-c16-chomp-smooth-lr100` | 0.084230 | -0.003257 | 0.013731 | 2/5 | 5/5 | 5821.23 | 6524 |
| `stable-residual-c16-chomp-smooth-lr010` | 0.086615 | -0.000872 | 0.014805 | 2/5 | 5/5 | 5857.37 | 6524 |

逐折 delta：

- `lr100`：`-0.000718`、`+0.000335`、`+0.003074`、`-0.000873`、`-0.018102`。
- `lr010`：`+0.000356`、`-0.002632`、`-0.001448`、`+0.000611`、`-0.001246`。

低 LR 消除了 `lr100` 在 fold 4 的大幅退化，但没有形成一致正增益。

## 逐 horizon 结果

| Arm | 1 日 delta | 2 日 delta | 3 日 delta | 5 日 delta |
|---|---:|---:|---:|---:|
| `lr100` | -0.002816 | -0.002852 | -0.008334 | +0.000975 |
| `lr010` | -0.002050 | -0.000106 | -0.001164 | -0.000167 |

`lr010` 通过预注册的 2/3/5 日退化门槛，但没有通过 1 日 `>=0`、mean `>=0.09`、相对 control 平均增益 `>0` 和至少 3/5 folds 不退化四项门槛。

## 假设判定

1. **H1：仅靠有界参数化即可修复。未得到支持。** `lr100` 比 control 退化，并在 3 日和 fold 4 上出现明显损失。
2. **H2：adapter LR 过高。部分得到支持。** 在完全相同参数化下，`lr010` 比 `lr100` 提高 `0.002385`，3 日 delta 从 `-0.008334` 恢复到 `-0.001164`，完成 epochs 从平均 5.0 恢复到 6.0。
3. **H3：early stopping 是唯一主因。不成立。** `lr010` 每个 fold 的 best/completed epoch 与 control 完全一致，但 RankIC 仍略低，因此恢复训练长度不足以产生增益。
4. **H4：2/3 日共享梯度冲突仍是本轮主因。证据减弱。** `lr010` 的 2/3 日退化已很小，主要剩余损失转到 1 日；本轮无需重新引入 PCGrad。

## 权重与参数化审计

三臂参数量均为 6524。候选 temporal adapter 参数为 232 个，optimizer 参数组完整、互斥且无遗漏：

- `lr100`：`base-lr-0.003+adapter-lr-0.003`。
- `lr010`：`base-lr-0.003+adapter-lr-0.0003`。

所有 stabilized 权重行和保持为 1，满足有界、零和 residual 结构约束。但两条候选在五折训练后都没有产生负权重：

| Arm | Intraday weight range | Mean intraday residual L2 | Max intraday residual L2 | 负权重数 |
|---|---:|---:|---:|---:|
| `lr100` | 0.008247..0.042051 | 0.017117 | 0.027074 | 0 |
| `lr010` | 0.017947..0.023216 | 0.002345 | 0.004667 | 0 |

`lr010` 的 day 权重范围仅 `0.098130..0.102064`，intraday 权重也接近均匀的 `1/48`。这暴露了 v15 参数化的关键限制：同一组 raw logits 同时产生 simplex base 与 signed residual；降低 adapter LR 会同时冻结正常的正 simplex 时间加权和 signed residual。它稳定了训练，却也抹掉了 v13 在 1 日 horizon 上的方向性时间滤波。

## LSTM 描述性比较

- `lr010` TCN mean RankIC：`0.0866153`。
- LSTM mean RankIC：`0.1115955`。
- 配对平均差：`-0.0249802`。
- TCN/LSTM model-step：`3.8915x`。
- TCN/LSTM 端到端：`3.7515x`。
- TCN 参数量：6524；LSTM 参数量：6124。

因为效果门槛先失败，selection 中 `relative_speed_gate_passed=false` 表示“没有 effect winner 可晋级”，不是描述性速度低于 3x。

## 工程与证据

- 红灯反馈环先稳定复现缺失接口，最小实现后同一命令 29 passed。
- 完整 Ruff：通过。
- 完整 mypy：58 个源码文件通过。
- 统一测试入口：通过。
- preflight：通过。
- production wheel/sdist：通过。
- v15 receipt 的 24/24 个输出 SHA-256 复算一致。
- `tcn-epoch-history.parquet` 已显式包含 `seed` 和 `model_seed`。
- Receipt schema：`tcn-stabilized-signed-residual-v15/v1`。
- `sealed_test_accessed=false`；未调用 PandaData、未下载数据、未访问 test/sealed、未部署或执行外部写入。

## 下一步边界

v15 候选不得进入 seeds 17/27 或 sealed test。下一次结构实验应真正解耦两条学习路径：保留按 LR 0.003 学习的 horizon-specific simplex base，并增加独立、零初始化、低 LR、零和有界的 signed residual。这样 control 的正权重学习不会被低 LR 冻结，residual 又不会复现 v14 的无界漂移。

该解耦会增加最多 232 个 adapter 参数（6524 到 6756，约 3.56%）；下一轮必须显式接受并审计这项容量变化，或另行设计低秩 residual 后再做参数匹配，不能把它伪装成等参数改动。
