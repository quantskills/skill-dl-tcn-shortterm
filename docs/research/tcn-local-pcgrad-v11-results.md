# TCN 局部 PCGrad v11 真实五年验证结果

日期：2026-08-03

最终状态：`stop_no_seed7_effect_pareto_v11`

Receipt：`45d2c8b6aef32d7f57b4a110c28e6e914d86472eed4345f4b3b63e0b7f848365`

## 本轮验证的问题

在保持 TCN 架构、数据、batch 顺序、优化器和验证协议不变时，只修复已观测到的 `(1d, 5d) × (block 4, block 6)` 梯度冲突，能否让 TCN 的预测效果超过 v10 父模型并继续保持相对 LSTM 至少 3× 的速度。

控制为 `skip-c16-chomp-smooth`；候选为 `skip-c16-chomp-localpcgrad-b46-h15`。两者均为 HorizonSkipTCN、16 channels、kernel 3、dilations 1–128、感受野 511、strict causal chomp、WeightNorm、dropout 0、seeded-random batches。输入继续通过只读 memmap 消费，未访问 test 或 sealed holdout。

## 五折聚合结果

| Trial | Mean RankIC | Worst fold | Positive folds | Median samples/s | Median model-step samples/s | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `skip-c16-chomp-smooth` | 0.087731 | 0.034488 | 5/5 | 5544.86 | 6011.60 | 控制 |
| `skip-c16-chomp-localpcgrad-b46-h15` | 0.087948 | 0.030698 | 5/5 | 3804.46 | 4016.92 | 效果与吞吐均未过门槛 |

候选相对控制 mean RankIC 只提高 `+0.000216`，距离 `0.09` 效果门槛仍差 `0.002052`。候选只有 fold 1 和 fold 4 改善，fold 0、2、3 退化；worst fold 也从 `0.034488` 降至 `0.030698`。因此不能把该变化视为稳定效果改进。

## Horizon 变化

候选相对控制的五折平均 RankIC 变化：

| Horizon | RankIC delta |
|---|---:|
| 1d | +0.021188 |
| 2d | -0.004026 |
| 3d | -0.004395 |
| 5d | -0.011901 |

局部 PCGrad 明显提高了 1d，但牺牲了 2d、3d，尤其是被共同投影的 5d。这说明训练期的负梯度余弦是一个真实现象，却不足以证明对称梯度投影能改善多期限样本外泛化；本轮干预更像把共享容量重新分配给 1d，而不是同时修复 1d/5d。

## LSTM 公平基准与速度

- TCN 候选 mean RankIC：`0.087948`
- LSTM mean RankIC：`0.111595`
- 配对 mean RankIC 差：`-0.023648`
- TCN/LSTM model-step 速度比：`2.7457×`
- TCN/LSTM 端到端速度比：`2.7022×`

局部 PCGrad 后，原本在 v10 成立的 3–5× 速度优势不再成立。候选吞吐只相当于本轮 TCN 控制的 `68.61%`。

五折候选累计 model-step 时间为 `71.224s`，其中额外 horizon backward 为 `20.414s`，占 `28.66%`；梯度投影本身只有 `0.674s`，占 `0.95%`。主要瓶颈不是向量投影，而是为 1d/5d 计算额外 autograd 梯度。因此，单纯优化 projection 实现不会恢复 3× 速度。

## 门禁与结论

候选同时触发：

- `mean_rankic_below_gate`：`0.087948 < 0.09`
- `throughput_below_gate`：`3804.46 < 5000 samples/s`
- relative model-step speed：`2.7457× < 3×`
- relative end-to-end speed：`2.7022× < 3×`

按预登记协议，本轮停止；未运行 seeds 17/27，未访问 sealed holdout。Receipt ID、18 个输出哈希和 4 个源输入哈希已复算，零不匹配。

本轮否定的是“每个 batch 对 1d/5d、block 4/6 做对称 PCGrad 可以形成效果/速度 Pareto 改进”，不是否定 TCN。下一轮更值得预登记的方向是 TCN 内部的 horizon-specific 表征解耦或多尺度 decoder，让 1d 与 5d 少共享冲突层，同时只做一次普通 backward；不应继续扩大 PCGrad block/horizon 扫描，也不应把 LSTM 替换为主模型。
