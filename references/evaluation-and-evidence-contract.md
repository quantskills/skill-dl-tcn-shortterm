# 评测与证据契约

## 公平 TCN/LSTM 比较

固定以下条件后才能比较：数据指纹、样本集合、walk-forward folds、seed、训练/验证用途、特征、标签、batch、精度、epoch 或停止规则、PyTorch 线程数和硬件。

速度至少分成：

1. 模型 forward/backward/optimizer step；
2. 数据加载与批处理；
3. 含逐轮验证和 checkpoint 选择的完整周期；
4. 单独标注的推理吞吐。

不得用短批量推理速度替代训练 model-step 速度，也不得把一次硬件收据写成普遍的 3–5 倍结论。

## 预测效果指标

- `RankIC`：逐 `signal_date × horizon` 计算 Spearman 相关，再按注册协议汇总。
- `Top excess return`：预测 Top 集合相对同期横截面均值的原始持有期收益差。
- `NDCG@Top`：Top 区域排序质量。
- `top_membership_precision`：预测 Top 与实际 Top 的集合重合率。
- `top_positive_return_rate`：预测 Top 中原始收益大于零的比例。
- `top_above_cross_section_mean_rate`：预测 Top 中收益高于同期横截面均值的比例。

后三个指标不可互换。旧字段 `top_precision` 只可作为 `top_membership_precision` 的兼容别名，不能解释为正收益命中率。

TCN 和 LSTM 都必须输出相同的四期限连续分数，所以 RankIC 是公平的共同预测契约。它不覆盖全部业务效用，因此必须与 Top excess return、NDCG 和稳定性联合解释。

## 当前可引用证据

- V46 独立窗口点估计：control TCN RankIC `0.021199`，V42 student `0.022038`，LSTM `0.018008`。
- V42 父证据：单 student TCN/LSTM 训练 model-step 速度比 `4.8789x`。
- V46 状态：`v46_student_not_generalized`。失败的是 student 相对 control TCN 的跨时期机制增益，不是 TCN 架构本身。
- 当前证据支持“本项目冻结协议下 TCN 具有预测—速度研究价值”，不支持“TCN 一定优于 LSTM”。
- `alpha_ready=false`、`deployment_authorized=false`、`trading_authorized=false`。

## 最小结果收据

每次正式研究结果至少记录：

```text
schema/version
run identity and timestamp
data/manifest/config/model/checkpoint fingerprints
signal-date coverage, securities, horizons and effective labels
fold boundaries, purge/embargo counts and seeds
model parameters, receptive field, threads, batch, precision and hardware
RankIC/Top excess return/NDCG and diagnostic hit metrics
model-step, full-cycle and inference throughput
comparison status, failed gates and limitations
sealed_test_accessed, alpha_ready, deployment_authorized, trading_authorized
```

如果没有完整指纹、公平对照或预注册门槛，结果只能标为 exploratory。ordinary validation 通过只能产生研究候选；一次性独立测试失败后不得复用同一窗口继续调参。
