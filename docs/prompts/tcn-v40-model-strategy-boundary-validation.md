# TCN v40 模型—组合责任边界校正与真实换手验证：完整实施提示词

你是 `skill-dl-tcn-shortterm` 项目的实现与验证代理。请在不访问任何 sealed/test 数据、不修改 v39 TCN 架构、不重新选择 v39 seed-7 checkpoint、不连接券商、不部署、不写外部系统的前提下，完成一次普通 validation 上的模型门与组合门分离，并使用冻结预测重放真实的 horizon-specific vintage 持仓换手和成本后收益。

## 一、问题与已知事实

1. v39 top50 `relative_tcn` 相对 `base_tcn` 的 RankIC、Top10% return、Top precision、NDCG 和 TCN 吞吐保持率均通过预登记门槛；唯一失败项是 `mean_top_turnover_delta=+0.030759` 高于 `+0.02`。
2. v39 的 `top_turnover` 是相邻信号日 Top10% 名单的成员变化率；v33 已将它定义为“未计成本诊断”。它不是资金权重换手，也没有表达 1/2/3/5 日重叠 vintage、同股票跨 vintage 净额抵消或实际交易成本。
3. 项目已有 `build_executable_long_only`，每个期限 `h` 每日建立一批 `1/h` 资金的独立持仓；但旧账本的 `one_way_turnover=max(entry_capital, exit_capital)` 只计算批次资金流，没有按证券目标权重变化进行净额抵消。
4. 旧 sealed window 已消费且被拒绝。本轮只能使用普通 validation，任何结果都不得升级为 Alpha、candidate-model sealed 结论或部署授权。
5. v39 Phase B 的 seeds 17/27 与同预算 LSTM benchmark 是被名单换手硬门截断的；本轮只有在校正后的模型门通过后才可授权后续多种子/基准确认。

## 二、目标

1. 保留 `top_turnover`，但明确重命名其语义为 `membership_turnover_diagnostic`，只作输出稳定性诊断，不进入 TCN 预测模型准入门。
2. 在 vintage 账本中按每个事件时点、证券和目标权重净额计算：
   - `buy_turnover = Σ max(w_t - w_{t-1}, 0)`；
   - `sell_turnover = Σ max(w_{t-1} - w_t, 0)`；
   - `one_way_turnover = max(buy_turnover, sell_turnover)`；
   - `traded_notional_turnover = buy_turnover + sell_turnover`。
3. 成本只能从实际计划交易名义金额计算：`transaction_cost = traded_notional_turnover × one_way_cost_bps / 10000`，禁止再用名单变化率直接估算成本。
4. 分离两个互不替代的门：
   - 模型门：RankIC、正 fold/seed 单元数、Top precision、NDCG、Top return、RankIC bootstrap 下界、速度保持率；
   - 组合门：真实资金换手、成本后净收益、成本敏感性、最大回撤和相对 PIT 等权基准表现。
5. 增加可选、严格因果且模型无关的组合层 `incumbent-buffer`：上一信号日已入选股票只要仍位于 `TopK + buffer` 内即可保留，再按当日分数补足 TopK。相同政策必须同时应用于 base 和 candidate，禁止针对某一模型单独调参。

## 三、可证伪假设

1. `H1`：v39 的唯一失败属于责任边界错误；移除名单换手硬门后，relative10 seed-7 会通过模型门。
2. `H2`：旧 vintage `one_way_turnover` 高估了同一证券在同一事件时点的新旧持仓交易；按证券净额后，中间换仓事件的换手会下降。
3. `H3`：用名单变化率乘成本与用实际交易名义金额乘成本会产生不同的成本后收益，因此旧成本口径不可作为可交易性证据。
4. `H4`：组合层缓冲能降低真实交易名义金额，但可能牺牲部分预测收益；必须同时报告收益—换手前沿，不能只报告最低换手配置。
5. `H5`：seed-7 模型门通过不等于多随机种子稳定，也不证明 TCN 相对 LSTM 的一般优势。

## 四、冻结输入

- v39 父目录：`artifacts/tcn-top50-relative-seed7-screen-v39`。
- 冻结预测：`predictions.parquet`，父 receipt 中 SHA-256 为 `38365159947754dd6a278ba89214b3c03c9927f02359f073df4994e9a2957f2a`。
- 标签：`artifacts/pandadata-top50-training-v39/labels.parquet`，SHA-256 为 `cd1c785268790a469b99796383b55b5b04bbfcfb63973ab34fd2bcbd09047200`。
- v39 leaderboard、bootstrap 和 selection 必须只读重放并核对父 receipt。
- 仅允许 `stage=validation`、`sealed=false`、seed `7`、fold `0..4`、horizon `1/2/3/5`。
- 输出必须写入新的不可覆盖目录；禁止修改父产物。

## 五、唯一代码干预

本轮只允许：

1. 修正 vintage 目标持仓的证券级净额换手算法；
2. 新增模型门/组合门分离的纯函数；
3. 新增冻结预测经济重放 runner；
4. 新增可选 `incumbent-buffer` 组合政策；
5. 增加测试、配置、文档和证据收据。

禁止修改 TCN/LSTM 网络、损失、优化器、学习率、epoch、batch size、特征、标签、split 或 v39 checkpoint。

## 六、Phase A：冻结 seed-7 重放

固定策略：

- `raw_topk`：`buffer_fraction=0.0`，主验收策略；
- `incumbent_buffer_20pct`：`buffer_fraction=0.2`，组合层因果消融；当 `ceil(K×0.2)` 小于 1 时使用 1 个缓冲名额。

固定成本压力：`0/5/10/20 bps` 单边成本。成本应用于 `buy_turnover + sell_turnover`，而不是 membership turnover。

模型门沿用 v39 除换手外的全部阈值：

- mean RankIC delta `>= 0.002`；
- 正 fold 单元 `>= 3/5`；
- mean Top precision delta `>= 0`；
- mean NDCG delta `>= 0`；
- mean Top return delta `>= -0.0001`；
- RankIC block-bootstrap 95% CI low `>= -0.002`；
- relative10/base8 TCN 吞吐保持率 `>= 0.90`。

组合门不得反向否决“预测模型门是否通过”，但决定“是否具备继续做可交易组合研究的资格”。主策略 `raw_topk` 至少报告：

- 每个 model/seed/fold/horizon 的 membership turnover diagnostic；
- buy/sell/one-way/traded-notional turnover；
- gross return、PIT 等权基准、gross excess；
- 0/5/10/20 bps 下的净收益和 candidate-base delta；
- 最大回撤、并发 vintage、现金比例；
- `incumbent_buffer_20pct` 相对 raw_topk 的收益—换手变化。

Phase A 状态：

- 模型门通过：`top50_relative_model_seed7_admitted_v40`；
- 模型门失败：`stop_top50_relative_model_seed7_v40`；
- 组合证据充足且 10bps candidate-base 净收益不退化：`portfolio_research_admitted_v40`；
- 否则：`portfolio_research_not_admitted_v40`，但不得把它改写成 TCN 模型失败。

## 七、Phase B：多种子与 LSTM

只有 Phase A 模型门通过才可执行：

1. 使用完全相同的 top50 base8/relative10、训练配置和预算训练 seeds `17/27`；与 seed 7 合并为 15 个 fold-seed 单元。
2. 使用 top50 relative10、相同 folds/seeds、8 CPU threads、float32、batch 128、8 epoch、hidden size 34、Adam `lr=0.003` 的冻结预算 LSTM。
3. TCN 与 LSTM 必须具有相同逐样本、标签、split、预测和评测契约；报告 model-step 与 end-to-end 速度比，禁止把 seed-7 的 relative/base 速度保持率冒充 TCN/LSTM 速度比。
4. 多种子模型门沿用 v39 除名单换手外的门槛：mean RankIC delta `>=0.002`、至少 `9/15` 正单元、Top precision/NDCG 不退化、Top return `>=-0.0001`、RankIC CI low `>=-0.002`、TCN 吞吐保持 `>=90%`。
5. LSTM 只作同预算 benchmark；如 LSTM 获胜，必须如实记录，不得更换主目标或事后修改门槛。

## 八、行为测试

先写并观察失败，再实现：

1. 名单换手很高但全部预测门通过时，模型门必须通过，并在证据中保留名单换手诊断。
2. 相邻 vintage 在同一事件时点继续持有同一证券时，中间事件的证券级净交易换手必须为零；旧批次资金流诊断可保留但不得称为真实换手。
3. 从 A 换到 B 时必须同时产生 sell 和 buy；成本按二者之和计算。
4. incumbent buffer 只能使用当日分数与上一信号日入选集合，不得查看未来收益；base/candidate 使用同一参数。
5. 任一 sealed/test 行、输入哈希漂移、样本契约漂移、输出目录已存在时 fail closed。

## 九、产物

新目录至少包含：

- `model-gate.json`
- `strategy-gate.json`
- `policy-summary.parquet`
- `portfolio-ledger.parquet`
- `portfolio-holdings.parquet`
- `cost-sensitivity.parquet`
- `membership-diagnostics.parquet`
- `config.resolved.json`
- `report.md`
- `receipt.json`

receipt 必须记录父 receipt、输入 SHA-256、代码身份、策略参数、成本参数、输出哈希、`sealed_test_accessed=false`。

## 十、验证与停止条件

运行 focused pytest、Ruff、Mypy、完整 pytest、`python tasks/preflight.py`、`python tasks/test.py` 和 `python -m build`。任何失败必须如实记录。

完成后只允许得出以下三类结论：

1. `research-ready model`：模型门通过，但尚未经过新 future sealed；
2. `portfolio-research-ready`：普通 validation 的真实持仓/成本门也通过；
3. `not alpha-ready`：在新的未来 sealed 测试通过前始终成立。

不得连接券商、部署、执行订单、读取旧 sealed 数据或宣称收益保证。
