# TCN 有效上下文与 soft-RankIC v12 真实五折结果

日期：2026-08-03

状态：`stop_no_seed7_effect_pareto_v11`

Receipt：`a9e51bb6bb252ae23ac05ef74c0efc3e2e35bbbe7cc4fa72a532594d1026c5d1`

## 结论

本轮成功把候选 TCN 的有效输入归因从“最后一天约 74%–79%”降到约 43%–50%，前五天合计归因提高到约 23%–27%，同时多尺度 SmoothL1 TCN 继续保持相对 LSTM 约 3.51x model-step 和 3.39x 端到端速度。因此，完整时序读出修复了有效感受野塌缩这一结构问题，而且没有破坏速度优势。

但是该修复没有带来稳定 RankIC 增益。多尺度 SmoothL1 的五折 mean RankIC 为 `0.087487`，比冻结父模型 `0.087731` 低 `0.000244`，只在 2/5 folds 改善。固定 `0.2` 权重、temperature `0.1` 的 soft-RankIC 进一步降到 `0.084704`，并把相对 LSTM 速度降到约 2.62x。因此“有效感受野塌缩是效果差距的充分解释”和“当前 soft-RankIC 配置可改善效果”均被本轮证伪。

本轮结果不支持追加参数扫描，也不支持访问 sealed test。下一轮如继续，应转向 TCN 输入与任务结构：用有符号完整序列适配器替代正 simplex 平均，或增加市场/行业/横截面上下文；不得继续优先优化 PCGrad 或重复扫描相同 soft-rank 权重。

## 五折聚合

| Trial | Mean RankIC | Worst fold | Positive folds | Median samples/s | Median model-step samples/s | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| `skip-c16-chomp-smooth` | 0.087731 | 0.034488 | 5/5 | 4853.15 | 5233.16 | 6260 |
| `context-c16-chomp-smooth` | 0.087487 | 0.014449 | 5/5 | 5425.73 | 5904.04 | 6524 |
| `context-c16-chomp-softrank20-tau10` | 0.084704 | 0.017282 | 5/5 | 4170.52 | 4439.84 | 6524 |

## 架构消融

`context-c16-chomp-smooth` 相对父模型的逐折变化：

- fold 0：`-0.020039`
- fold 1：`+0.019926`
- fold 2：`-0.045758`
- fold 3：`+0.045926`
- fold 4：`-0.001277`
- 五折平均：`-0.000244`

候选的 day simplex 保持在约 `0.0962..0.1064`，intraday simplex 保持在约 `0.0185..0.0232`，接近均匀初始化。它扩大了直接历史梯度路径，但没有学习出明显 horizon 专门化；正权重均值池化很可能同时稀释了有符号的短期模式。

## 目标函数消融

`context-c16-chomp-softrank20-tau10` 相对相同架构 SmoothL1 的 mean RankIC 变化为 `-0.002783`。它在 3/5 folds 有小幅提升，但 fold 1 明显下降，导致平均值恶化；median samples/s 从 `5425.73` 降至 `4170.52`。因此固定参数下的直接 soft-rank surrogate 没有形成效果—速度 Pareto。

## LSTM 公平比较

LSTM hidden-34 的五折 mean RankIC 为 `0.111595`，参数量 6124。

| TCN | RankIC 差值 | Model-step 比值 | 端到端比值 |
|---|---:|---:|---:|
| 父模型 | -0.023864 | 3.101x | 3.031x |
| 多尺度 SmoothL1 | -0.024108 | 3.513x | 3.390x |
| 多尺度 soft-RankIC | -0.026891 | 2.622x | 2.608x |

多尺度 SmoothL1 保留了 3x 速度事实，但效果仍低于 LSTM。soft-RankIC 同时未通过效果门槛和 3x 相对速度门槛。

## 有效感受野复核

对 fold 0 多尺度 SmoothL1 checkpoint 的 64 个 ordinary-validation 样本执行输入梯度归因，四个 horizon 的最后一天归因分别为：

- 1 日：`0.4269`
- 2 日：`0.4408`
- 3 日：`0.4562`
- 5 日：`0.5025`

对应前五天合计归因为 `0.2679/0.2592/0.2335/0.2260`。这证明新读出确实修复了旧模型的严重近期集中，但 RankIC 未同步改善，说明长期分钟历史在当前八特征、逐股票独立建模条件下并未自动形成额外横截面信号。

## 工程与证据

- Ruff：通过。
- mypy：93 个源码文件通过。
- 聚焦测试：24 passed。
- 统一测试入口：通过。
- production wheel/sdist：通过。
- Receipt 输出哈希：23/23 复算一致。
- `sealed_test_accessed=false`。
- 未调用 PandaData、未下载数据、未访问 test/sealed、未执行外部写入或部署。
