# TCN v10 真实五年验证与 LSTM 基准结果

日期：2026-08-03

状态：`stop_no_seed7_pareto_v10`

Receipt：`51256455db5215ae5d6e007787edd056e1427faa6e89b00194c2783e3617b5b2`

## 运行范围

- 数据：2021–2025 PandaData 沪深 300 PIT top20，5 分钟标准条，8 个特征，480 步窗口，共 23,821 个窗口样本。
- 验证：5 个 expanding ordinary-validation folds，仅 seed 7。
- TCN：16 channels、kernel 3、dilations 1–128、lr 0.003、batch 128、8 threads、最多 8 epochs、patience 2。
- LSTM：hidden size 34、6124 参数；其他训练资源与 TCN 对齐，仅作 benchmark。
- 未读取 test 或 sealed holdout；未下载新数据；未运行 seeds 17/27。

## TCN 筛选结果

| Trial | Mean RankIC | Worst fold | Positive folds | Median samples/s | Parameters | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `skip-c16-chomp-smooth` | 0.087731 | 0.034488 | 5/5 | 5332.41 | 6260 | RankIC 低于 0.09 |
| `skip-c16-chomp-rank01` | 0.087027 | 0.030200 | 5/5 | 4457.07 | 6260 | RankIC 与吞吐均未过门槛 |
| `lite-c16-chomp-smooth` | 0.083115 | 0.027897 | 5/5 | 6334.40 | 6228 | 控制组 |

Horizon skip 相对快速控制的 mean RankIC 改善为 `+0.004616`，同时保持吞吐高于 5000 samples/s；但它距离 0.09 门槛仍差 `0.002269`。固定权重 0.1 的日期分组排名目标没有进一步改善效果，并把吞吐降低至门槛以下，因此不保留为当前推荐路径。

## 参数匹配 LSTM 比较

比较对象为效果最好的 `skip-c16-chomp-smooth`（6260 参数）与 LSTM hidden-34（6124 参数），共 5 个相同 fold/seed 单元：

- TCN mean RankIC：`0.087731`
- LSTM mean RankIC：`0.111595`
- 配对 mean RankIC 差：`-0.023864`
- TCN/LSTM 模型步速度比：`3.8099×`
- TCN/LSTM 端到端训练速度比：`3.6554×`

因此，“TCN 在本机 CPU、参数量近似匹配、相同真实数据协议下比 LSTM 快 3–5×”已在本次 seed-7 五折测量中成立；但统一 TCN 的预测效果仍明显低于 LSTM，不能称为 Pareto 候选或有效 Alpha。

## 门禁结论

没有非控制 TCN 同时满足 mean RankIC 至少 0.09、5/5 folds 为正、吞吐至少 5000 samples/s、且不低于控制组的全部条件。按预登记协议，本轮正式停止，不运行 seeds 17/27，不追加 trial，不访问 sealed holdout。

下一轮若继续，应把重点放在 TCN 表征效果而非通用 infra：保留 `skip-c16-chomp-smooth` 作为新的速度合格父模型，先分析不同 horizon 的 skip 权重与 fold 0/1/4 信息损失，再预登记新的 TCN 内部表征实验。此次结果不支持继续扫描 dropout、weight decay 或相同的 rank-loss 权重。

## 工程验收

- Receipt ID 与 23 个输出哈希、4 个源输入哈希均已复算通过。
- Mypy：93 个源码文件通过。
- Ruff、preflight、统一测试入口和 production wheel/sdist build 通过。
- 完整 pytest：133 passed。
- `sealed_test_accessed=false`。
