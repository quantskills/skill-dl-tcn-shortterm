# V47：公开 CLI 与冻结优化 TCN 主路径对齐

## 角色与任务

你是本项目的研究工程代理。请把公开、Agent-neutral 的 `tcn-shortterm-skill` CLI 从基础 Bai TCN 路径升级为可显式选择的冻结优化 TCN 路径，并在同一份原始行情、同一 fold、同一 seed、同一 epoch、同一 batch 和同一 CPU 线程预算下与 LSTM 比较。实现后运行自动化验证和五年真实行情验收，输出可复核证据。

## 已冻结事实

1. 旧 CLI 的 `tcn.enabled` 路径是基础 Bai TCN；其真实五年运行速度不能代表已优化研究路径。
2. V40 在 Top50、相对特征、5 folds、seeds 7/17/27、8 epochs、CPU 8 threads 下观察到：
   - TCN/LSTM model-step 几何平均速度比 `4.8354x`；
   - TCN/LSTM end-to-end 几何平均速度比 `4.3372x`。
3. 可移植的冻结学生核心为：
   - `model_kind=dynamic_horizon_skip`；
   - channels `16`，kernel `3`，dilations `[1,2,4,8,16,32,64,128]`；
   - causal chomp padding，WeightNorm，dropout `0`；
   - Smooth L1，Adam `lr=0.003`，batch `128`，fixed `8 epochs`；
   - dynamic skip hidden `4`，scale `1.0`；CPU threads `8`。
4. 参数匹配基准为单层 LSTM hidden size `34`，同为 Adam `lr=0.003`、batch `128`、8 epochs、CPU threads `8`。
5. V42 的共识蒸馏需要教师预测或教师 checkpoint；它不是默认可移植依赖。本切片复用 V42 学生架构，但不把教师资产隐式打包进 CLI。
6. sealed/test 数据不得参与训练、选模、调参或本次 ordinary-validation 验收。

## 实施目标

1. 保留请求 schema v1 和旧配置语义；旧 `tcn.enabled`、`sequence_models.enabled` 不改变默认行为。
2. 新增显式 `optimized_tcn` 配置段。只有 `optimized_tcn.enabled=true` 时才启用冻结路径。
3. 优化路径复用正式源码里的 `TCNTuningTrial`、`DynamicHorizonSkipTCN`、walk-forward protocol 和 lazy/memmap loader，不复制任务脚本里的模型实现。
4. 原始 `raw_1m` 输入在配置启用时，使用 PIT universe 生成 base8 + relative2 的 Top50 相对序列特征；中间大张量必须落到临时 memmap，不能要求调用者传入裸 `[sample, feature, time]` tensor。
5. 优化 TCN 输出普通 validation 预测、RankIC 指标和逐 fold 训练元数据；模型名固定为 `optimized-tcn-v40-portable`。
6. 性能模块新增同模型工厂 `optimized-tcn-v40-portable`，与 LSTM 在同协议下计量 model-step、train-pipeline、validation 和 end-to-end 时间。
7. 运行清单和 evidence bundle 必须包含性能环境、线程数、模型参数、fold/seed 覆盖和 observed speed ratio；禁止把目标阈值写成观测结果。

## 配置契约

推荐配置：

```json
{
  "optimized_tcn": {
    "enabled": true,
    "profile": "v40-portable",
    "relative_features": true,
    "torch_threads": 8,
    "epochs": 8,
    "batch_size": 128,
    "learning_rate": 0.003
  },
  "performance": {
    "enabled": true,
    "models": ["lstm", "optimized-tcn-v40-portable"],
    "hidden_size": 34,
    "optimized_tcn_profile": "v40-portable",
    "torch_threads": 8,
    "epochs": 8,
    "batch_size": 128,
    "learning_rate": 0.003,
    "device": "cpu"
  }
}
```

`profile` 必须 fail closed：本切片只接受 `v40-portable`。冻结结构字段不得由普通 CLI 配置静默覆盖；允许覆盖的只有明确列出的执行预算字段，且 resolved config 必须保存最终值。

## 验收标准

### 合约和正确性

- 请求 schema 仍为 `1`，旧 demo/example/旧配置测试全部通过。
- causal padding、WeightNorm、感受野覆盖输入窗口的现有测试继续通过。
- split 中 `test`、`sealed_holdout`、`purged`、`embargo` 样本不会进入优化训练或 ordinary validation 预测。
- 优化 TCN 与性能 benchmark 使用同一个公开模型构造函数和同一个冻结 profile。
- 相对特征保留 base8，只追加 audited relative2；只使用 signal date 当时可见的 PIT 状态。
- 训练元数据记录 profile、architecture、optimizer、learning rate、batch、epochs、thread count、参数量、best epoch、训练/模型步时间。

### 自动化验证

- 为冻结 profile、模型工厂、CLI 路由、相对特征路由和速度比汇总补充单元/集成测试。
- 运行 `python tasks/preflight.py`。
- 运行完整 `python tasks/test.py`。
- 运行完整 mypy。
- 构建 wheel/sdist，并从构建产物安装后执行 CLI demo 和 schema canary。

### 五年真实行情验收

- 输入仍为已登记的五年股票分钟行情 manifest，不重新下载、不改源数据。
- TCN/LSTM 使用完全相同的 features、folds、seeds、epochs、batch、lr、threads 和 precision。
- 主要速度门：`optimized TCN / LSTM model-step geomean >= 3.0x`。
- 次要速度门：`optimized TCN / LSTM end-to-end geomean >= 3.0x`；若失败，必须按 data wait、validation、序列化等组成报告 blocker，不能回到局部模型调参。
- 效果不是“TCN 必须天然优于 LSTM”的先验；报告相同 validation 样本上的 mean RankIC、各 horizon RankIC 和配对差值。研究层最低守门为 TCN mean RankIC 不低于 LSTM 超过 `0.002`。
- 明确区分：速度合格、效果非劣、Alpha-ready、实盘授权是四个不同结论。本任务不申请实盘、不部署、不执行交易。

## 禁止事项

- 不做新的 channels/dropout/learning-rate 网格搜索。
- 不以换手率、Top-K 缓冲等组合层局部参数替代模型层验收。
- 不读取 sealed test 来决定代码或参数。
- 不让 V42 教师 checkpoint 成为默认 CLI 的隐藏依赖。
- 不覆盖既有 V40/V42/V46 artifacts，不篡改历史 receipt。
- 不声称 3–5x 是架构保证；它只能是指定硬件和协议下的观测证据。

## 最终交付

提交源码、测试、示例配置、文档、构建产物验证记录以及一份新的五年真实验收 run。最终结论必须回答：公开 CLI 是否已真正调用优化 TCN、速度门是否通过、效果是否非劣、还剩哪些研究层 blocker。
