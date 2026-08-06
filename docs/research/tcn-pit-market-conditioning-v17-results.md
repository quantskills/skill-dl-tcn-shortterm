# TCN PIT 市场状态条件化 v17 真实 seed-7 结果

日期：2026-08-04

状态：`stop_pit_market_conditioning_seed7_effect_v17`

Receipt：`aa78dd0f6d76c67d2a7d88670d4fc6bcc900f4d32d0085510c3efa1c8dd9a7cc`

## 结论

v17 完整落地并验证了严格 PIT 的市场中心/横截面离散度上下文、每 fold 唯一训练日期标准化以及低秩有界 TCN 条件门控，但该机制没有提高真实普通验证 RankIC。候选 mean RankIC 为 `0.0833659`，低于冻结 control 的 `0.0874870`，delta 为 `-0.0041211`；它没有达到预注册的 `0.09` 与 `+0.003` 门槛，因此不授权 seeds 17/27，也不访问 sealed test。

TCN 速度路径仍然有效。候选相对固定 LSTM 的 model-step 为 `3.7152x`、端到端为 `3.5258x`，候选 median samples/s 为 `5271.36`，均通过描述性速度门槛。`selection.relative_speed_gate_passed=false` 是因为效果门槛先失败、最终决策没有晋级速度阶段，不表示观测速度低于 3x。

本轮排除的是“所有股票共享同一个信号日市场状态，通过小型 FiLM gate 调整 TCN 表征”这一机制，不是 TCN。TCN 继续作为主模型，LSTM 只作 benchmark。

## 冻结两臂结果

| Trial | Mean RankIC | Delta vs control | Worst fold | 不退化 folds | Median samples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| `context-c16-chomp-smooth` | 0.087487 | 0 | 0.014449 | 5/5 | 5629.75 | 6524 |
| `market-conditioned-c16-chomp-smooth-h4-g025` | 0.083366 | -0.004121 | 0.015736 | 2/5 | 5271.36 | 6784 |

候选只增加 260 个参数，即 `+3.985%`。五折 delta 依次为 `+0.001287`、`+0.000701`、`-0.000646`、`-0.001221`、`-0.020727`。fold 4 的候选在 epoch 1 达到最佳并于 epoch 3 停止，而 control 在 epoch 6 达到最佳并完成 8 epochs；该折是总体退化的主要来源。但退化并非只存在于一个折：候选只有 2/5 folds 不退化，而且 1/2/3 日聚合结果均下降。

## 逐 horizon 结果

| Horizon | Control | Candidate | Delta |
|---|---:|---:|---:|
| 1 日 | 0.059417 | 0.054336 | -0.005081 |
| 2 日 | 0.072771 | 0.067784 | -0.004987 |
| 3 日 | 0.105677 | 0.096197 | -0.009480 |
| 5 日 | 0.112083 | 0.115147 | +0.003064 |

候选只有 5 日提高；1/2/3 日分别触发预注册的退化 blocker。它同时触发 `mean_rankic_below_gate`、`control_mean_rankic_degradation`、`fold_stability_below_gate` 和 `mean_rankic_delta_below_gate`。

## Gate 与容量审计

候选不是因为零初始化分支没有更新而失败：五折最佳 checkpoint 的 gate 输出层 L2 分别为 `0.4073`、`0.4073`、`0.5074`、`0.5508`、`0.4294`，全部显著大于零。

验证日期上的实际 gate 范围和平均偏离为：

| Fold | Gate min | Gate max | Mean abs(gate-1) | 95% abs(gate-1) |
|---|---:|---:|---:|---:|
| 0 | 0.9584 | 1.0332 | 0.0081 | 0.0230 |
| 1 | 0.9632 | 1.0353 | 0.0091 | 0.0237 |
| 2 | 0.9390 | 1.0672 | 0.0146 | 0.0491 |
| 3 | 0.9167 | 1.0899 | 0.0269 | 0.0682 |
| 4 | 0.9393 | 1.0700 | 0.0136 | 0.0374 |

没有样本接近预注册的 `[0.75, 1.25]` 边界，说明失败不是 gate 饱和或 scale 过大。进一步调大 gate scale、hidden size 或学习率没有证据支持，也违反本轮单机制边界。

## PIT 与数据审计

- context identity：`09ee7a6fc57869d8f521e2e1c717927033531c282ba596040a112e95f9ec9380`
- 24 维字段：6 个因果特征的市场中位数中心与 MAD 离散度，各自包含最近 48 bars 和完整 480 bars 均值。
- ordinary-validation 可用 position：15,683。
- 信号日期：800；每日股票数 16–20，没有单股票日期。
- fold 0–4 的训练标准化拟合日期数：394、474、554、634、714；每个日期等权。
- 所有上下文有限；同日股票逐位共享；窗口结束日严格等于信号日；未来日期扰动测试不改变过去上下文。
- 真实行业历史不可用，`industry_context_status=blocked_historical_industry_unavailable`；没有静态回填或伪造行业字段。
- 未调用 PandaData、未补数据、未下载、未读取 test/sealed、未部署或外部写入。

## 机制判定

1. **H1：共同市场状态缺失是效果缺口的一部分。未得到支持。** 共享状态 gate 在 seed 7 上使 mean RankIC 下降 `0.004121`。
2. **H2：中心与离散度能通过共享 gate 改善状态适配。未得到支持。** 诊断中的状态依赖是真实存在的，但不等于共享上下文本身有可学习的横截面增量。
3. **H3：表面效果来自额外容量。没有观察到。** 增加 260 参数后结果下降。
4. **H4：共享市场条件化没有稳定增益。得到支持。** 分支确实更新、未饱和，但只有 2/5 folds 不退化。

核心原因是任务仍然是同日横截面排序，而一个对同日所有股票完全相同的状态向量不直接提供股票之间的相对位置。它只能通过通道缩放间接改变排序；在只有 800 个独立日期、masked SmoothL1 训练的条件下，这种二阶交互没有形成稳定增益。诊断发现“低离散度状态更容易预测”，只说明模型难度随状态变化，不能推出共享状态足以修复排序。

## 工程与证据

- 红灯反馈环先因缺少 context 模块失败，最小实现后 7 项 v17 测试通过，覆盖 PIT、fold-only scaler、初始等价、gate 边界、容量、决策和真实 tuning batch seam。
- 完整 Ruff：通过。
- 完整 mypy：99 个文件通过。
- 统一测试入口：通过；全量套件 164 项。
- preflight：通过。
- production wheel/sdist：`python -m build --no-isolation` 通过。隔离构建仅在安装 `setuptools>=69` 的环境引导阶段超时，未进入源码构建；这不是打包错误。
- receipt schema：`tcn-pit-market-conditioning-v17/v1`。
- receipt ID 重算一致；21/21 个登记输出 SHA-256 重算一致。
- `sealed_test_accessed=false`。

## 下一步边界

停止继续调整共享 market gate 的 scale、hidden size、LR 或随机 seed。下一轮若继续优化 TCN，应另立 v18 预注册实验，把 PIT 横截面信息变成**每只股票不同**的输入，而不是同日共享向量：优先构造 `stock_sequence - same-date market median sequence` 的个股相对市场残差，以及按同日 MAD 缩放的 robust relative-strength 序列，再输入冻结的 causal TCN trunk。

该方向直接对应横截面排序：共同市场冲击被显式扣除，每只股票获得不同的相对序列。应先做 deterministic feature ablation（原始特征 control、原始+中心残差、原始+robust 残差），保持 readout/loss/infra 不变；LSTM 继续只作 benchmark。仍需先用 seed 7 普通验证，失败即停，不访问 sealed test。
