---
name: skill-dl-tcn-shortterm
description: 使用因果扩张 Temporal Convolutional Network 对沪深 A 股分钟线执行未来 1、2、3、5 个交易日的横截面收益排序研究，并以同数据、切分和预算的 LSTM 作为基准，生成可审计的数据、训练、速度和预测效果证据。用于运行或诊断 TCN 短线预测、核验感受野与因果卷积、PIT/walk-forward/purge/embargo、防止未来泄漏、比较 RankIC/Top 区域指标与训练吞吐，或判断研究模型是否达到冻结候选条件；不用于组合换手优化、券商连接、实盘交易或收益承诺。
---

# TCN 短线时序卷积研究

## 什么情况下应该使用这个 Skill

在用户需要用沪深 A 股分钟线或预计算的 `[样本, 特征, 时间]` tensor 研究未来 `1 / 2 / 3 / 5` 个交易日的横截面收益排序时使用本 Skill。它适合以下任务：

- 训练、重放或诊断严格因果的 TCN 短线预测模型；
- 在相同数据、切分、训练预算和硬件口径下，以 LSTM 做预测效果与速度基准；
- 审计 causal padding、扩张卷积感受野、WeightNorm、PIT 标的池、walk-forward、purge/embargo 和未来泄漏；
- 生成 RankIC、Top 区域指标、吞吐、稳定性及冻结候选所需的可审计证据。

不要在组合构建、换手率调优、仓位分配、交易执行、券商连接或实盘部署任务中使用本 Skill；这些属于下游策略层。若目标不是 TCN 分钟线横截面预测，或用户要求收益承诺，也不应触发本 Skill。

始终调用已安装的 `skill_dl_tcn_shortterm` 实现。不得在本 Skill、提示词或 Agent adapter 中复制 TCN、数据切分、评估或证据算法，也不得要求调用方理解仓库内部的 `tasks/`、版本实验脚本或本机目录结构。

## 快速确认

先执行 `tcn-shortterm-skill demo`。命令不可用时，要求安装当前仓库的 wheel 或 Git 版本；不要自行重写运行逻辑。成功回执必须包含 `status=success`、engine version、request digest 和 `authoritative_run_manifest`，但不能据此声称 TCN 已训练、有效或可部署。

使用统一入口：

```text
tcn-shortterm-skill run --request <request.json>
```

需要最小接口样例或机器协议时执行：

```text
tcn-shortterm-skill example --output-dir <empty-directory>
tcn-shortterm-skill schema --kind request
tcn-shortterm-skill schema --kind result
```

命令不可用但 Python 包可导入时，可以等价执行 `python -m skill_dl_tcn_shortterm ...`。相对路径必须以 request 文件所在目录解析；未知字段、缺失文件、指纹漂移和重复运行 ID 都应 fail closed。

## 执行顺序

1. 明确任务是数据审计、TCN 训练、冻结模型重放、TCN/LSTM 公平比较，还是结果解释。
2. 要求调用方提供本地数据、manifest、config 和扁平 request；不自动下载数据、读取凭据或扫描整台机器。
3. 校验 manifest、数据指纹、PIT 标的池、信号时点、标签结束时间和 fold 边界；任一关键证据缺失时停止。
4. 只消费因果生成的 5 分钟序列。裁剪、归一化、特征选择和其他可学习预处理均在每折训练段拟合。
5. 运行 TCN 前记录输入长度、卷积核、每层 dilation、每块卷积数和实际感受野；感受野不足时停止，不得靠未来 padding 补齐。
6. 保持纯左侧 causal padding、WeightNorm 和显式有效标签掩码；不得改用对称 padding 或 BatchNorm 获得表面指标改善。
7. 比较 LSTM 时固定数据、fold、seed、线程、batch、精度、epoch/停止规则和评估代码；同时报告 model-step 与完整训练—验证周期。
8. 输出每个证券在 1、2、3、5 日期限上的连续横截面分数。TCN 不决定持仓、TopK、换手缓冲或成交规则。
9. 联合报告 RankIC、Top excess return、NDCG@Top、稳定性和吞吐收据；把收益排序、正收益命中和高于横截面均值命中分开解释。
10. 保留权威运行清单、配置与输入指纹。研究通过不等于 Alpha-ready、部署授权或交易授权。

## 当前证据边界

- 没有其他预注册协议时，把使用真实 rank target 的 `control_tcn` 作为稳定研究参考。
- V42 consensus student 只作为已冻结机制的研究变体；V46 独立外推未证明它稳定优于 control TCN，不得自动晋级为默认模型。
- LSTM 只作为相同输出契约下的 benchmark，不替代 TCN 项目方向。
- 新研究运行优先使用 `optimized_tcn.profile=v40-portable`，并在 `performance.models` 中同时注册 `lstm` 与 `optimized-tcn-v40-portable`；旧 `tcn.enabled` 只是基础 Bai TCN 兼容路径，不能用它代表优化路径的速度。
- `v40-portable` 复用 V42 student 的可移植架构但不依赖教师 checkpoint；需要教师资产的 consensus distillation 只能作为显式研究变体。
- 现有证据支持本项目特定 CPU 训练协议下约 `4.8789x` 的 TCN/LSTM model-step 速度比；不得外推为跨硬件、跨数据的普遍结论。

## 模型与策略边界

模型层的权威输出是：

```text
signal_date, instrument_id, horizon, score
```

其中 `horizon` 仅允许 `1, 2, 3, 5`，`score` 是用于同日、同期限横截面排序的连续值。组合构建、目标权重、持仓批次、换手率、容量、成本和执行约束属于下游策略层；不得把这些局部策略参数反向写成 TCN 模型门。

## 按需读取详细资料

- 创建 request、处理路径或解释回执时读取 [references/agent-contract.md](references/agent-contract.md)。
- 设计、训练或解释网络时读取 [references/model-and-output-contract.md](references/model-and-output-contract.md)。
- 接入分钟线、构造样本或排查泄漏时读取 [references/data-and-causality-contract.md](references/data-and-causality-contract.md)。
- 比较 TCN/LSTM、解释指标或决定研究状态时读取 [references/evaluation-and-evidence-contract.md](references/evaluation-and-evidence-contract.md)。

完整人类使用说明读取 `README.md`。只有维护包本身时才读取开发文档；不要把内部版本历史或长文复制回本文件。

## 禁止事项

不得索取或提交数据供应商凭据，不得连接券商、发送订单、写外部数据库、触碰未授权 sealed 数据、重复消费已封存窗口、用 test 反向调参、把组合换手问题伪装成模型损失问题，或把单次 RankIC、速度比、回测收益描述成普遍优越性或实盘能力。
