# TCN v35 候选冻结与一次性 sealed 验收 v36 — 完整执行提示词

你正在维护 `skill-dl-tcn-shortterm`。v35 已在 2021–2025 真实股票分钟线 ordinary validation 上通过完整性、机制、任务对齐效果与速度四门，状态为 `constrained_tail_ordinary_validation_candidate_v35`。本轮不得继续查看 ordinary validation 后改模型、改 checkpoint 选择、改门槛或换模型；目标是把 v35 精确冻结为一次性 sealed test 的可审计候选，并在未获得用户逐字授权前停在 readiness。

## 一、不可变目标

1. 主模型仍是 Bai 风格因果扩张残差 TCN；LSTM 只作为冻结 benchmark，不得取代 TCN。
2. 冻结 v35 的 3 seeds × 5 folds 的 control/candidate checkpoint 选择、checkpoint SHA-256、源数据 SHA-256、模型/损失/选择协议与 v33 的 15 个 LSTM checkpoint。
3. 只读取 split manifest 的 stage、sealed、sample id 与日期等元数据来建立 readiness；不得读取 sealed 标签值、特征值、预测或已有测试指标。
4. 只有用户在新的明确消息中逐字给出 `授权执行 sealed test`，一次性 evaluator 才可打开 sealed 特征和标签。近似表达、前后空格、引用历史消息或配置中的哈希都不算授权。
5. sealed 一旦成功消费，无论结果好坏均不得重复评测、调参后重试或更换门槛；失败只允许在 sealed loader 尚未打开或 evaluator 未产生任何指标时，依据原子状态收据处理。

## 二、时间安全的模型资格

原始 sealed manifest 有两个 canonical `stage=test, sealed=true` 测试段；`sealed_holdout` 是前一 fold 测试段在后一 fold 的重复登记，必须排除，避免重复计数。

对每个 sealed 测试段，只允许 `ordinary validation_end_date < sealed_test_start_date` 的 seed/fold checkpoint：

- sealed fold 0（2023-12-13 至 2024-05-16）：仅 ordinary folds 0、1、2，3 seeds，共 9 个模型单元；
- sealed fold 1（2024-10-30 至 2025-03-27）：ordinary folds 0–4，3 seeds，共 15 个模型单元；
- 总计 24 个 time-safe unit exposures；其中 v35 candidate 相对 RankIC control 改选 12 个 exposures。

禁止把 fold 3/4 checkpoint 用于第一段 sealed，因为其 checkpoint 选择使用了第一段 sealed 日期之后的 ordinary validation；即使模型训练本身没读 sealed 标签，这仍属于时间穿越。

## 三、冻结与 readiness 产物

新增纯模块与 CLI，先写失败测试，再实现：

- 严格验证 v35 receipt ID、candidate status、四门、`sealed_test_accessed=false`、`sealed_test_authorized=false`，并复算 receipt 与全部 154 个输出 SHA-256；
- 严格验证 v33 LSTM receipt ID、`sealed_test_accessed=false` 及全部声明输出；
- 验证 ordinary manifest 只含 train/validation/purged 且无 sealed；canonical test 全部 sealed、fold 恰为 0/1、sample id 唯一，并与 ordinary sample id 零交集；
- 根据日期硬门生成 `eligible-checkpoint-plan.parquet`，验证 24 个 exposures、12 个 changed exposures、所有 TCN/LSTM checkpoint 文件 SHA；
- 生成不可覆盖的 `frozen-plan.json`、`sealed-data-descriptor.json`、`readiness.json`、`state.json`、`receipt.json`；receipt 绑定配置、父 receipt、源数据、checkpoint plan、代码与全部输出哈希；
- readiness 状态只能是 `awaiting_explicit_sealed_authorization_v36`，且必须声明 `authorization_received=false`、`sealed_test_accessed=false`、`evaluation_executed=false`。

任何身份、哈希、coverage、日期、stage、sealed 或样本隔离异常都必须在 sealed loader 之前 fail closed；不得覆盖目标目录。

## 四、一次性评测协议（本轮只冻结，不执行）

授权后每个 time-safe seed/fold checkpoint 在对应 sealed 段产生相同逐样本输出契约，比较 candidate、RankIC control 与冻结 LSTM。评测按 `(sealed segment, signal_date, horizon)` 建组；先在同一日期/期限内对可用模型单元求 paired mean，再按交易日 block bootstrap 5000 次，禁止把同一批标签被 24 个模型重复预测误当成 24 份独立市场样本。

同时报告：

- RankIC；
- Top 10% precision；
- NDCG@Top10%；
- Top return 与 excess return；
- long-short spread；
- top turnover；
- 以单边 10 bps 成本扣减后的净收益；
- TCN/LSTM model-step 与 end-to-end 吞吐比（冻结普通验证速度证据，仅作同环境工程门，不因 sealed 结果重跑挑硬件）。

candidate-control 门槛在打开 sealed 前固定为：Top precision 与 NDCG 均值不降；二者至少一个 95% CI low 大于等于 0，另一个不低于 -0.002；mean RankIC delta 不低于 -0.002；Top return 与成本后净收益 delta 的 CI low 均不低于 -0.0005；mean turnover delta 不高于 0.02；model-step 与 end-to-end 均至少 3×。LSTM 比较必须完整报告，但 TCN 不被预设为一定优于 LSTM。

所有门通过才可记为 `sealed_confirmed_tcn_candidate_v36`；任一门失败即 `sealed_rejected_tcn_candidate_v36`。两种状态都结束该 sealed 数据上的模型选择，不自动部署、不交易、不连接券商、不承诺收益。

## 五、测试与验收

必须覆盖：

1. 缺失、近似或带空格的授权文本在 loader 前被拒绝；
2. receipt、checkpoint、source SHA 漂移被拒绝；
3. ordinary 中出现 sealed/test、canonical test 不 sealed、重复 sample、ordinary/sealed 交集被拒绝；
4. 第一 sealed 段不得纳入 fold 3/4；第二段允许 folds 0–4；总数 24、改选 exposures 12；
5. `sealed_holdout` 不重复计入；
6. 目标已存在时拒绝覆盖；
7. readiness 真实运行不加载 features/labels，不创建 consumed marker；
8. 完整 pytest、Ruff、Mypy、preflight、tasks 测试与 production build 通过。

本轮完成条件是：提示词、配置、测试、实现、真实不消费 sealed 的冻结产物、哈希收据、文档与完整工程验收均通过；然后停住，向用户报告精确授权短语及一旦消费不可回退的后果。没有逐字授权时绝不执行 sealed test。
