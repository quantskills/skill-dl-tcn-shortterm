# V47 公开 CLI 优化 TCN 对齐结果

## 结论

公开 Agent-neutral CLI 已真正接入冻结的 `optimized-tcn-v40-portable`，不再用基础 Bai TCN 的速度代表优化研究路径。五年真实行情验收中，model-step 与 end-to-end 速度门均通过；预测 RankIC 为正，但相对同协议 LSTM 的非劣门未通过。

因此本次状态为：

- 工程主路径：`passed`；
- TCN 速度门：`passed`；
- 因果性与 sealed/test 隔离：`passed`；
- TCN/LSTM 预测非劣门：`failed`；
- Alpha-ready：`false`；
- deployment/trading authorization：`false`。

## 运行身份

- run id：`3925a53612056e21`
- run directory：本地封存证据目录（不随公开仓库分发）
- engine：`skill-dl-tcn-shortterm 0.1.0`，从 V47 wheel 安装运行
- data SHA-256：`829b8a1336ac80781930c24a1e936f38b05187baeb312223138ebef4ba11f22a`
- code source SHA-256：`64e39ce42a9e5d02745d19e5c7ddd18cb8c2a4efb4c77b622997530e6cddb7bc`
- request schema：`1`
- wall time：`1175.2s`，约 `19.6min`
- evidence bundle：哈希验证通过

## 数据与特征

- 原始 1 分钟行数：`14,533,440`
- 标准 5 分钟 bars：`2,906,688`
- 完整 stock-days：`60,556`
- 有效 10 日窗口：`59,539`
- 拒绝窗口：`938`
- 特征：base8 + audited relative2
- 特征张量：`[59,539, 10, 480]`
- 横截面宽度范围：`44–50`
- base8 保留：`true`
- static rank channels：未加入
- relative feature sealed access：`false`

固定 `600 train / 100 validation / 5 embargo / 100 test` 协议在该有限时间跨度内形成 2 个完整 folds；`max_folds=5` 是上限，不代表必须产生 5 folds。

## 同协议模型

共同条件：CPU、8 threads、seed 7、float32、read-only memmap、batch 128、Adam LR 0.003、8 epochs、同一 features、同一 folds、同一 validation 代码。

| 模型 | 参数量 | Fold 0 train/val | Fold 1 train/val |
|---|---:|---:|---:|
| LSTM hidden 34 | 6,396 | 29,041 / 4,970 | 34,502 / 4,973 |
| optimized TCN | 6,476 | 29,041 / 4,970 | 34,502 / 4,973 |

优化 TCN 结构为 dynamic horizon skip、channels 16、kernel 3、dilations 1–128、causal chomp、WeightNorm、Smooth L1、dynamic skip hidden 4。两个 folds 都完成 8 epochs，最佳 checkpoint 均为 epoch 6。

## 速度证据

| Fold | TCN model-step samples/s | LSTM model-step samples/s | model-step ratio | end-to-end ratio |
|---:|---:|---:|---:|---:|
| 0 | 5,298.94 | 1,349.63 | 3.9262x | 3.5269x |
| 1 | 5,341.42 | 1,279.94 | 4.1732x | 3.7720x |

- model-step 几何平均速度比：`4.0478x`，通过 `>=3x` 门；
- end-to-end 几何平均速度比：`3.6474x`，通过 `>=3x` 门。

该结果是指定硬件和协议下的实测，不是 TCN 的跨环境架构保证。

## 预测证据

| 指标 | optimized TCN | LSTM | TCN - LSTM |
|---|---:|---:|---:|
| mean best validation RankIC | 0.079133 | 0.094083 | -0.014950 |

优化 TCN 的公开预测 RankIC：

| Horizon | RankIC |
|---:|---:|
| 1d | 0.061236 |
| 2d | 0.080492 |
| 3d | 0.085786 |
| 5d | 0.089019 |

TCN 的四期限信号均为正，但预注册非劣门要求 TCN-LSTM 不低于 `-0.002`，实际为 `-0.014950`，因此失败。V40 多 seed 证据中 relative TCN 相对 relative LSTM 也曾为负，方向并非本次 CLI 偶发反转；当前不能宣称 TCN 预测效果优于或不劣于 LSTM。

## 隔离与证据完整性

- TCN validation prediction rows：`38,543`
- unique prediction samples：`9,779`
- 与 `test` / `sealed_holdout` sample overlap：`0`
- training metadata `sealed_test_accessed`：两个 folds 均为 `false`
- evidence index 校验：通过

## 解释与下一门

V47 已排除“公开 CLI 仍在调用旧 Bai TCN”这一工程混淆，也再次证明速度与基础设施不是当前瓶颈。剩余问题是模型归纳偏置/训练目标与当前横截面排序任务的匹配度；参数量已基本匹配，不能把差距简单归因于 LSTM 更大。

本结果可以作为研究层 TCN 预测 Skill 的真实可运行基线，但不能作为 TCN 效果晋级证据。下一阶段若继续优化，应建立独立的结构性预测效果规格，在相同公开 folds 上做预注册多 seed 比较；不得回到换手率、Top-K 缓冲或零散 channels/dropout 网格搜索，也不得读取 test/sealed 反向调参。
