# TCN v40 模型—策略责任边界与多种子 LSTM 对照结果

## 结论

v39 的停止原因不是 TCN 模型失效，而是把 Top10% 成员集合变化率误当成了模型 promotion 阻断项。v40 冻结 v39 的 TCN 训练协议和预测，重新分离模型门与策略门，并在普通 validation 上完成 3 seeds × 5 folds 的 TCN/LSTM 同预算确认。

- TCN 模型门：`top50_relative_model_multiseed_admitted_v40`
- 组合研究门：`portfolio_research_admitted_v40`
- 证据等级：ordinary-validation research evidence
- `sealed_test_accessed=false`；不构成新 sealed 证据、Alpha-ready 或部署授权

## 完整执行提示词

实现前冻结的任务、约束、验收口径和停止条件见：

- `docs/prompts/tcn-v40-model-strategy-boundary-validation.md`

该提示词要求保持 TCN 为主模型，不修改 v39 网络、特征、标签、fold、epoch 或 checkpoint；只校正责任边界、成交换手核算和同预算 LSTM 基准。

## 修正内容

1. 模型门只判断预测质量、稳定性和 TCN 自身速度保持率；membership turnover 保留为诊断，不再阻断模型。
2. 组合门按证券目标权重的前后变化净额计算：`buy=sum(max(w_t-w_{t-1},0))`，`sell=sum(max(w_{t-1}-w_t,0))`，one-way turnover 为 `max(buy,sell)`。
3. 成交成本按真实双边成交名义额 `buy+sell` 计提，避免把成员集合变化率代替交易成本。
4. 同一证券在并发 vintage 退出与进入时先净额化，避免把内部换仓重复计为市场成交。
5. 增加只使用当时可见持仓的 `incumbent_buffer_20pct` 因果缓冲策略，并与 `raw_topk` 分开报告。

## Phase A：seed 7 冻结预测重放

- 模型门通过：5/5 fold 中 3 个 RankIC 单元为正，RankIC 均值差 `+0.007593`，bootstrap CI 下界 `+0.000111`。
- 原 v39 membership turnover 诊断仍为 `+0.030759`，但它不是证券级成交换手。
- 10 bps 下，`raw_topk` 的证券级 one-way turnover 均值差为 `-0.008088`，净收益均值差为 `+0.000171`。
- Phase B 因此被授权；收据 SHA-256：`d61b57725bc19b139f611e020fdebadcb8ae633cb4eeb33da396333fa5911f07`。

## Phase B：3 seeds × 5 folds 同预算确认

### TCN 内部改善

relative10 TCN 相对 base8 TCN：

- RankIC：`+0.008374`，11/15 单元为正，bootstrap CI 下界 `+0.003407`
- Top precision：`+0.003938`
- NDCG：`+0.008614`
- Top return：`+0.000366`
- membership turnover 诊断：`-0.005601`
- TCN 吞吐保持率：`0.9474`

因此相对时序特征在更宽 top50 横截面上获得稳定的 TCN 内部增量，模型门通过。

### TCN 与 LSTM

在相同数据、seed、fold、batch 128、float32、8 epochs、CPU 8 threads 条件下，共 15 个配对单元：

- 模型步吞吐比 TCN/LSTM：几何均值 `4.8354×`，中位数 `4.6087×`
- 训练—验证完整周期速度比：几何均值 `4.3372×`，中位数 `4.1549×`
- TCN-LSTM RankIC：`-0.009409`
- TCN-LSTM Top return：`-0.000503`
- TCN-LSTM NDCG：`-0.007585`
- TCN-LSTM Top precision：`+0.003792`

这说明“TCN 比 LSTM 快 3–5×”在本项目这套 top50 CPU 协议下成立，但不是跨硬件、跨数据或跨实现的普遍承诺。预测效果结论是 mixed：LSTM 的 RankIC、Top return 和 NDCG 更高，TCN 的 Top precision 更高；不能宣称 TCN 一定优于 LSTM。

### 可执行组合

3 seeds × 5 folds × 4 horizons 共 60 个策略单元，10 bps `raw_topk`：

- relative10 相对 base8 的净收益均值差：`+0.000191`
- one-way turnover 均值差：`-0.008736`
- membership turnover 未用于策略门

`incumbent_buffer_20pct` 在普通 validation 中进一步降低两模型换手并提高成本后净收益，但它是策略层候选，不能反向证明模型更好，也未获得新的 sealed 授权。

## 当前边界与下一步

v40 已解决当前 optimization loop 的三项核心问题：TCN 速度、TCN 内部预测增量，以及模型指标与组合换手的责任边界。项目可以进入组合研究候选落地，不需要继续为满足某个 membership turnover 数字而修改 TCN。

仍未解决的是独立的新封存期泛化、真实冲击成本/容量、停牌与涨跌停执行细节、线上漂移和生产监控。旧 sealed test 已永久消费，后续若要晋级 Alpha-ready，必须先冻结新候选与新时期 sealed 数据，再获得单独授权；不得使用 v40 ordinary validation 继续无界调参。

## 可复现产物

- Phase A：`artifacts/tcn-v40-model-strategy-boundary-seed7/`
- Phase B：`artifacts/tcn-v40-multiseed-lstm-confirmation/`
- Phase B receipt：`e3613f1e30c2e6f2500af55659f553e68c8b269b452a53affd07872329a2a836`
