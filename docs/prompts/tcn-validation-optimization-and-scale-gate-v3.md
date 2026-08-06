# TCN 普通验证调优与样本扩容门禁提示词 v3

你是 `skill-dl-tcn-shortterm` 的时序模型优化执行者。请复用已经生成且带哈希的五年 PandaData top20 预处理产物，先回答“TCN 是否只是没有调优好”，再决定是否值得扩大到 top50。不得重复下载已有分钟线，不得读取普通 test 指标或 sealed test，不得以一次较有利的 wall-clock 替代完整证据。

## 固定输入与安全边界

- 固定源运行：`artifacts/pandadata-five-year-sequence-benchmark-runs-v2-optimized/52787a471076e1be/`。
- 只消费其中的只读 `feature-windows.npy`、`window-index.parquet`、`labels.parquet`、`split-manifest.parquet`；每个输入先复算 SHA-256。
- 调优只允许读取 split manifest 中的 `train` 与 `validation` 行。`test`、sealed test、组合收益和未来标签不得进入配置选择、早停或排序。
- 真实数据和调优产物写入 Git 忽略的 `artifacts/`；源代码、测试、非秘密配置、提示词和证据文档进入仓库。
- 固定随机种子7、float32、CPU、相同归一化、相同标签 mask、相同数据加载协议。不得把跨日期/跨期限混合相关系数冒充横截面 RankIC。

## 正确优化指标

每轮验证分数必须按 `signal_date × horizon` 独立计算 Spearman RankIC，再对有效日期和四个期限等权求平均，称为 `mean_daily_rankic`。至少两个有效证券才形成一个日期/期限相关系数。早停、候选排序和扩容门禁只使用该指标。

## Phase A：单折有界筛选

仅使用 fold 0；每个候选最多8 epochs，patience=2，min_delta=0.002，batch=128。矩阵固定为：

| ID | channels | kernel | dilations | dropout | learning rate | receptive field |
|---|---:|---:|---|---:|---:|---:|
| control-k3-c8 | 8 | 3 | 1,2,4,8,16,32,64 | 0.10 | 0.01 | 509 |
| k3-c8-lr3e3 | 8 | 3 | 1,2,4,8,16,32,64 | 0.10 | 0.003 | 509 |
| k3-c16-lr3e3 | 16 | 3 | 1,2,4,8,16,32,64 | 0.10 | 0.003 | 509 |
| k5-c8-lr3e3 | 8 | 5 | 1,2,4,8,16,32 | 0.10 | 0.003 | 505 |
| k5-c16-lr1e3 | 16 | 5 | 1,2,4,8,16,32 | 0.10 | 0.001 | 505 |
| k5-c8-lr1e3-d0 | 8 | 5 | 1,2,4,8,16,32 | 0.00 | 0.001 | 505 |

每个候选逐轮保存训练损失、训练秒数、验证秒数、`mean_daily_rankic`、按期限 RankIC、最佳 epoch、停止原因、参数量和纯训练吞吐。按最佳 `mean_daily_rankic` 降序选择前2名；并列时依次选择参数更少、训练吞吐更高、ID字典序更小者。

## Phase B：两折确认

只重跑 Phase A 前2名与控制组，使用 fold 0/1、最多12 epochs、patience=3、min_delta=0.002、batch=128。候选最终分数是两折各自最佳 `mean_daily_rankic` 的算术平均；同时必须报告最差折、折间差异和相对控制组提升。

## 扩容门禁

只有同时满足以下条件，才允许进入 top50 数据物化与同协议复跑：

1. 最优候选两折 `mean_daily_rankic` 都大于0；
2. 最优候选两折平均分相对控制组至少提升0.01；
3. 没有感受野、PIT、标签、哈希、样本数或非有限损失异常；
4. 改善不是只来自一个期限，至少两个期限的平均 RankIC 不低于控制组。

任一条件失败即停止扩容，状态写为 `stop_no_validation_gain`。这不是工程失败，而是预登记的节省数据与计算成本结论。

## 速度复核

对 Phase B 最优候选直接复用同一 memmap，固定3 epochs，分别测 batch 128/256/512。报告纯训练吞吐和包含逐轮验证的完整周期吞吐；CPU下不得承诺3–5×。只有未来显式授权GPU后，才能增加 CUDA、AMP 和 `torch.compile` 矩阵。

## TDD 公共接口

- `TCNTuningTrial`/调优计划验证：拒绝感受野不足、重复ID、非法学习率、非法早停预算。
- 普通验证调优执行器：只消费 train/validation，逐日期/期限计算 RankIC，保存最佳权重并按预登记规则早停。
- 候选选择与扩容门禁：用字面量收据验证排序、平局规则和 fail-closed 条件。
- 不可变任务入口：复算源哈希、拒绝覆盖、写出 trial/epoch/selection/receipt 产物且不含秘密。

## 验收

- 先逐个红—绿实现上述公共缝；单元测试不联网，不模拟内部实现。
- 执行 Phase A、Phase B；仅在扩容门禁通过时执行 top50。
- 运行 Ruff、Mypy、完整 pytest、preflight、统一测试入口和 wheel/sdist build。
- 结论必须区分：调优改善、样本扩容、训练速度、预测效果；不得称为Alpha或生产就绪。

按以上提示词直接执行。若门禁失败，按预登记规则停止 top50，而不是降低阈值或读取 test 寻找更好结果。
