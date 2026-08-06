# TCN 模型与输出契约

## 模型职责

TCN 只负责把截至信号日收盘可见的分钟序列映射为四个连续横截面分数。组合策略消费这些分数后自行决定 TopK、权重、持仓批次、缓冲、换手、容量和成本；策略规则不得反向改变模型是否有效的定义。

## 输入与输出

- 神经网络标准输入：`[batch, feature, time]`。
- 基准序列：约 480 个 5 分钟时间步，即最近 10 个完整交易日。
- 标准输出：`[batch, 4]`，列顺序固定为 `[1d, 2d, 3d, 5d]`。
- 下游记录至少包含：`signal_date, instrument_id, horizon, score`。
- `score` 只在同一 `signal_date × horizon` 横截面内排序，不解释为绝对收益率、上涨概率或仓位。
- 缺失标签使用显式掩码排除，不得填零参与损失。

## 因果 TCN 不变量

1. 每层卷积只做纯左侧 padding；输出不得依赖右侧或未来时间步。
2. 默认使用 WeightNorm，不使用 BatchNorm。
3. Bai 双卷积残差块的感受野按下式计算：

   `R = 1 + 2 × (kernel_size - 1) × Σ(dilation)`

4. 默认 Bai 基线使用 kernel size 3、dilation `1,2,4,8,16,32,64`，感受野为 509，可覆盖 480 步输入。
5. 任何新配置都必须记录实际感受野并拒绝 `R < input_length`。
6. 四个期限共享因果时序主干，但使用独立输出头；单期限模型仅作负迁移消融。

不要使用“层数对应 `2^层数`”作为完整感受野公式。该说法只描述指数 dilation 的量级，忽略 kernel size、每块卷积数和初始时间点。

## 当前模型角色

| 模型 | 当前用途 | 解释边界 |
|---|---|---|
| true-target/control TCN | 稳定研究参考 | 当前优先用于新的研究重放与比较 |
| V42 consensus student | 冻结研究变体 | ordinary validation 有增益，但 V46 未证明相对 control 跨时期泛化 |
| LSTM | 公平 benchmark | 共享相同四期限连续分数契约，不是本项目替代主模型 |
| TCN-lite | 速度/消融配置 | 不自动代表预测效果最优配置 |
| ModernTCN | 隔离后续实验 | 不属于当前默认路径 |

只有新的预注册证据可以改变这些角色。不得利用已经看过的 V46 独立窗口重新选择 teacher weight、seed、horizon、损失或门槛。
