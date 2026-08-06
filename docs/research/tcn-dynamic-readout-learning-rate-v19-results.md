# TCN 动态读出独立学习率 v19 真实 seed-7 结果

日期：2026-08-04
状态：`stop_dynamic_lr_seed7_effect_v19`
Receipt：`ca865f2a5470d0f35b50159634c85f8a774588f459f2d4b41f8cc9b36e5c64f2`

## 结论

v19 成功把动态读出的 176 个参数从基础 TCN 参数中解耦，并将其 Adam 学习率从 `0.003` 提高到 `0.01`。干预按预期生效：相对 v18，同折动态权重变异的配对中位比率达到 `3.0738x`，输出 weight L2 五折均明显非零；但 mean RankIC 从静态控制组 `0.0874870` 降到 `0.0841727`，delta 为 `-0.0033143`。

因此，本轮直接否定了“v18 只是动态读出学习不足，提高学习率就能形成稳定增益”这一假设。更强的样本相关时间重加权没有弥补预测缺口，反而主要伤害 1d 和 3d。不得继续围绕同一 validation 结果搜索中间学习率，也不授权 seeds 17/27 或 sealed test。

## 冻结实验合同

- 完整提示词：`docs/prompts/tcn-dynamic-readout-learning-rate-v19.md`。
- 父证据：v18 receipt `dd837cd6dffd6081d7decfaa42ff5c2754531fcdf7fdf57e29e2fb455f00b673`。
- 数据、标签、PIT universe、五折 expanding split、seed 7、TCN trunk、动态 scorer、参数量、损失、batch、epoch、早停和 LSTM benchmark 全部不变。
- 控制组：6524 参数，全部 lr `0.003`。
- 候选：6700 参数；6524 个 base 参数 lr `0.003`，176 个 dynamic-attention 参数 lr `0.01`。
- 只允许 ordinary train/validation；未读取 test/sealed test。

## 参数组证据

- optimizer identity：`base-lr-0.003+dynamic-attention-lr-0.01`。
- 每折 `optimizer_dynamic_attention_parameter_count=176`。
- 候选总参数量每折均为 6700；控制组每折均为 6524。
- 动态输出 weight L2 为 `0.7437–1.2367`；bias L2 为 `0.4707–1.0273`。
- 公共 TDD 测试证明两个参数组完备、互斥，v18 默认单组 optimizer 路径保持兼容。

## 效果结果

| fold | 静态控制组 | v19 动态 LR | delta |
|---:|---:|---:|---:|
| 0 | 0.0144490 | 0.0109532 | -0.0034958 |
| 1 | 0.0936261 | 0.0970918 | +0.0034657 |
| 2 | 0.0556713 | 0.0568539 | +0.0011826 |
| 3 | 0.2012150 | 0.2036201 | +0.0024051 |
| 4 | 0.0724736 | 0.0523444 | -0.0201292 |
| mean | 0.0874870 | 0.0841727 | -0.0033143 |

候选五折 RankIC 仍全部为正，并有 3/5 折高于控制组；但 fold 4 的明显退化覆盖了其他三个折的小幅改善，说明增强动态性没有形成跨时期稳定性。

### 分期限

| horizon | 控制组 | v19 候选 | delta | 门禁 |
|---:|---:|---:|---:|---:|
| 1d | 0.0594171 | 0.0536439 | -0.0057732 | `>=0`，失败 |
| 2d | 0.0727708 | 0.0729806 | +0.0002097 | 通过 |
| 3d | 0.1056771 | 0.0967500 | -0.0089271 | `>=-0.005`，失败 |
| 5d | 0.1120831 | 0.1133163 | +0.0012333 | 通过 |

与 v18 类似，动态时间重加权没有稳定帮助所有期限；提高学习率后 1d/3d 的负迁移反而显著扩大。

## 动态机制结果

| fold | v18 综合变异 | v19 综合变异 | 比率 |
|---:|---:|---:|---:|
| 0 | 0.001039 | 0.004132 | 3.9765x |
| 1 | 0.001032 | 0.003173 | 3.0738x |
| 2 | 0.001710 | 0.003400 | 1.9887x |
| 3 | 0.001467 | 0.004541 | 3.0950x |
| 4 | 0.003162 | 0.001753 | 0.5544x |

- 配对中位比率 `3.0738x`，通过 `>=2x` 门禁。
- v19 最小绝对变异 `0.0017533`，未达到预注册的 `>=0.002`；失败来自 fold 4。
- simplex 最大求和误差仍为 `2.384e-7`。

这说明优化器干预整体有效，但同一学习率在不同市场时期产生的动态强度不稳定；最关键的是，更强动态性与更高 RankIC 不存在一致对应关系。

## 速度

| 指标 | 结果 | 门禁 |
|---|---:|---:|
| 候选 median samples/s | 5333.20 | `>=5000`，通过 |
| 相对 LSTM model-step | 3.7173x | `>=3x`，通过 |
| 相对 LSTM end-to-end | 3.5841x | `>=3x`，通过 |
| 候选 mean RankIC | 0.0841727 | — |
| LSTM mean RankIC | 0.1115955 | — |

速度继续不是当前瓶颈。

## 门禁失败项

1. `mean_rankic_below_gate`：`0.084173 < 0.09`；
2. `control_mean_rankic_degradation`：相对控制组下降；
3. `horizon_1d_degradation_below_gate`；
4. `horizon_3d_degradation_below_gate`；
5. `mean_rankic_delta_below_gate`：`-0.003314 < +0.003`；
6. `dynamic_weights_not_sample_conditioned`：预注册命名沿用 v18，实际含义是最差折绝对变异 `0.001753 < 0.002`，不是完全静态。

效果门禁优先失败，因此 selection 中 `relative_speed_gate_passed=false` 表示未获得最终晋级，不代表实测速率不足；comparison 中两个速度比率均已通过 3x。

## 根因更新

### 已证实

- v18 的小增益不能主要归因为动态 scorer 学习率过低。
- 动态强度可以在不破坏 TCN 速度的情况下提高，但强度提高本身不等于预测效果提高。
- 当前“只对最终 TCN 隐状态做日间/日内动态加权”的表达形式存在明显期限和时期不稳定性。

### 被本轮否定

- “把动态参数 LR 从 0.003 提到 0.01 就能把 v18 的弱动态性转化为 >=0.003 RankIC 增益”。
- “动态权重变异越大，RankIC 必然越高”。fold 4 给出了直接反例。

### 下一轮建议

停止搜索当前 dynamic-readout 的学习率、hidden size 或 scale。若继续 TCN，应转向不同的、仍保持 TCN 主干的结构假设：对各扩张卷积 block 的不同感受野做样本相关的多尺度选择，而不是只在最终 block 内重加权时间位置。该 v20 必须保持零初始化控制等价、轻量容量和 3x 速度门禁，并预先限制为一个多尺度读出候选。

## 完整性

- Receipt schema：`tcn-dynamic-readout-learning-rate-v19/v1`。
- 20/20 输出 SHA-256 复算一致。
- `sealed_test_accessed=false`，`sealed_test_authorized=false`。
- Ruff 全仓、mypy 103 个源文件、preflight、完整 172 项 pytest 与 production wheel/sdist build 均通过。
