# TCN 有符号时间适配器 v13 真实五折结果

日期：2026-08-03

状态：`seed7_winner_admitted_v11`

候选：`signed-context-c16-chomp-smooth`

Receipt：`31748f1a344630cbb3eac7b47f85ce33c949c4bc5cd1c9bd7a96cb7bb09bcced`

## 结论

有符号双尺度时间适配器通过 seed-7 五折效果与速度门槛。候选在与 simplex 控制完全相同的 6524 参数、相同初始输出、相同 TCN trunk、相同 SmoothL1 和训练预算下，把 mean RankIC 从 `0.087487` 提高到 `0.091777`，平均增益 `+0.004290`，3/5 folds 改善，5/5 folds 为正，最差折从 `0.014449` 提高到 `0.036821`。

相对固定 LSTM，候选 model-step 速度为 `3.470x`，端到端速度为 `3.351x`，因此本机同协议下的 3x 速度事实继续成立。候选 RankIC 仍比 LSTM `0.111595` 低 `0.019819`，所以它是当前 TCN 的 ordinary-validation Pareto 候选，不是已经超过 LSTM 的 Alpha 结论，也没有 sealed-test 资格结论。

训练后所有 fold 的 day adapter 仍保持正权重，但日内 adapter 在每个 horizon 和 fold 都学出了大量负权重，说明优化器确实使用了符号自由度。效果提升主要来自 1 日与 2 日 horizon；3 日和 5 日略有退化，后续多 seed 确认必须重点检查这一负迁移是否稳定。

## 五折聚合

| Trial | Mean RankIC | Worst fold | Positive folds | Median samples/s | Median model-step samples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| `context-c16-chomp-smooth` | 0.087487 | 0.014449 | 5/5 | 5356.69 | 5811.74 | 6524 |
| `signed-context-c16-chomp-smooth` | 0.091777 | 0.036821 | 5/5 | 5352.75 | 5812.12 | 6524 |

候选相对控制的逐折变化：

- fold 0：`+0.022372`
- fold 1：`-0.004000`
- fold 2：`+0.003314`
- fold 3：`+0.001588`
- fold 4：`-0.001826`

平均变化 `+0.004290`，3/5 folds 不退化，满足预注册机制门槛。

## 逐 horizon 聚合

| Horizon | Simplex control | Signed candidate | Delta |
|---|---:|---:|---:|
| 1 日 | 0.059417 | 0.076113 | +0.016696 |
| 2 日 | 0.072771 | 0.078692 | +0.005921 |
| 3 日 | 0.105677 | 0.103724 | -0.001953 |
| 5 日 | 0.112083 | 0.108578 | -0.003505 |

有符号适配器对短周期帮助明确，对 3/5 日没有改善。该结果与“不同 horizon 需要不同时间滤波”一致，但仍需 seeds 17/27 判断是否稳定。

## LSTM 公平比较

- TCN mean RankIC：`0.0917768`
- LSTM mean RankIC：`0.1115955`
- 配对平均差：`-0.0198187`
- TCN/LSTM model-step：`3.4695x`
- TCN/LSTM 端到端：`3.3514x`
- TCN 参数量：6524；LSTM 参数量：6124

候选同时通过 mean RankIC `>=0.09`、5/5 正折、median samples/s `>=5000`、不低于控制以及两项相对速度 `>=3x` 门槛，配置因此授权 seeds `17/27` 确认；本轮没有擅自执行额外 seeds。

## 权重符号审计

day adapter 在五个 fold 中未出现负权重，但不再受行和为 1 的约束：不同 horizon/fold 的行和约为 `0.309..0.785`。

intraday adapter 明确使用负权重：

- fold 0 四个 horizon 的负权重数：`48/29/22/33`
- fold 1：`21/35/29/21`
- fold 2：`35/10/22/47`
- fold 3：`22/27/11/19`
- fold 4：`6/35/23/27`
- 全部权重范围约 `-0.1457..0.1075`

这证明正 simplex 对日内 48 步的凸组合约束确实限制了有方向的时间滤波；解除该约束在不增加参数量的情况下形成了可测效果增益。

## 有效感受野复核

对 fold 0 相同 64 个 ordinary-validation 样本进行输入梯度归因：

| Horizon | Simplex 最后一天 | Signed 最后一天 | Signed 前五天合计 |
|---|---:|---:|---:|
| 1 日 | 0.4269 | 0.3346 | 0.3485 |
| 2 日 | 0.4408 | 0.1825 | 0.4272 |
| 3 日 | 0.4562 | 0.5541 | 0.1250 |
| 5 日 | 0.5025 | 0.3960 | 0.2646 |

候选没有简单地把所有 horizon 都推向更长历史，而是形成明显 horizon-specific 归因：1/2/5 日更加分散，3 日更集中于近期。这与有符号适配器的设计目标一致。

## 工程与证据

- Ruff：通过。
- mypy：93 个源码文件通过。
- 聚焦测试：21 passed。
- 统一测试入口：通过。
- production wheel/sdist：通过。
- Receipt 输出哈希：18/18 复算一致。
- Receipt schema：`tcn-real-validation-v13/v1`。
- `sealed_test_accessed=false`。
- 未调用 PandaData、未下载数据、未访问 test/sealed、未部署或执行外部写入。

## 下一步门槛

只对本轮已准入的 signed candidate、参数匹配 simplex control 和固定 LSTM 运行 seeds 17/27。若多 seed 下平均增益仍为正、至少两个 seed 的 3/5 folds 不退化、3/5 日退化未扩大且速度继续 `>=3x`，再设计市场/行业/横截面上下文；否则回退为“seed-7 结构线索”，不进入 sealed test。
