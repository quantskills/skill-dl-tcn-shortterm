# TCN V46：效用对齐合同与一次性独立外推测试

你是 `skill-dl-tcn-shortterm` 项目的时序建模、量化验证和证据治理执行者。请在不修改 V42 模型参数、不搜索新超参数、不重写历史结论的前提下，完成一次“评测语义纠偏 + 冻结模型独立外推”的整体优化。

本轮不是继续追逐某个局部指标。它只回答以下三个系统级问题：

1. V42 共识蒸馏 TCN 在 ordinary validation 上获得的广泛排序改善，能否外推到一个从未用于训练、调参、选模或旧 sealed test 的后续市场窗口？
2. 该 TCN 相对冻结的 true-target TCN 控制组是否稳定改善，相对参数量近似匹配的 LSTM 是否形成可接受的“预测非劣—训练速度显著更快”折中？
3. 历史 `top_precision` 到底测量什么，它与“Top 组合正收益命中率”是否被错误混为同一指标？

## 一、不可改写的历史事实

- V42 Phase B 的正式历史状态保持 `stop_consensus_student_multiseed_v42`。
- V42 历史 `top_precision` 的定义是：预测 Top10% 股票集合与实际收益 Top10% 股票集合的重合率，不是正收益命中率。
- V42 的历史普通验证结果不得按 V46 新合同追溯性改判为通过。
- V36 一次性 sealed test 已经消费，状态保持 `sealed_rejected_tcn_candidate_v36`。
- V42 当前可复核速度证据为单 student TCN 相对同预算 LSTM 的 model-step 比率约 `4.8789x`；V46 不通过改变模型或缩小 LSTM 工作量刷新该数字。

## 二、冻结的任务输出合同

主任务继续是 1/2/3/5 个交易日的横截面收益排序。模型输出仍是每个期限的连续横截面 score，不新增方向分类头，不把波动率、换手率或某个 top-k 指标升级成新的训练主任务。

效用层按以下优先级评估：

1. 全横截面排序质量：RankIC；
2. 可交易 Top10% 长组合的原始收益与相对横截面均值的超额收益；
3. Top10% 区域排序质量：NDCG@Top10%；
4. 跨 seed、期限和日期的一致性；
5. 相对 LSTM 的训练速度优势与单模型单前向推理约束。

以下指标只作诊断，不能单独阻止或触发模型晋级：

- `top_membership_precision`：预测 Top10% 与实际收益 Top10% 的集合重合率；
- `top_positive_return_rate`：预测 Top10% 股票中原始持有期收益大于 0 的比例；
- `top_above_cross_section_mean_rate`：预测 Top10% 股票中收益高于当期横截面均值的比例；
- Top 集合换手率、Pearson IC、分位数组合单调性。

必须保留旧字段 `top_precision` 作为历史兼容别名，但新产物与文档不得再把它称作“正收益命中”。

## 三、冻结的数据与模型边界

### 独立窗口

- 已消费的最晚历史测试日：`2025-03-27`；
- 隔离结束日：`2025-04-03`；
- V46 一次性独立测试开始日：`2025-04-07`；
- 为保证 1/2/3/5 日四期限都具有后续标签，结束日：`2025-12-23`；
- 股票池：现有 PandaData 沪深300每日 PIT 权重前50；
- 输入：现有 relative10、10 日、480 根 5 分钟标准条；
- 只使用 `valid=true` 的 PIT 标签，禁止补未来收益、禁止用测试窗口重估归一化参数。

### 归一化与检查点

- 只使用普通验证 expanding fold 4 的训练样本统计量；
- V46 测试窗口不得进入均值/标准差拟合；
- seeds 固定为 `7, 17, 27`；
- 只读取每个 seed 的 fold 4 冻结检查点；
- 模型固定为：
  - `control_tcn`：V40/V39 的 true-target relative10 TCN；
  - `consensus_student_tcn`：V42 25% train-only teacher consensus student；
  - `relative_lstm`：V40 参数量近似匹配 LSTM；
- 不训练、不微调、不校准、不做 checkpoint ensemble、不按 V46 结果挑 seed 或期限；
- 每个模型保持单模型、单次前向。

### 一次性与防复用

- 运行前必须验证窗口晚于旧 sealed test，并至少经过既有 embargo；
- 输出目录存在时拒绝覆盖；
- 新窗口读取后必须记录 `sealed_test_accessed=true`、`sealed_consumed_exactly_once=true`；
- 任何标为 historical replay、ordinary validation 或与旧窗口重叠的输入，都只能得到 `research_reference_only`，不得晋级；
- 运行失败也不得删除或覆盖已经形成的消费凭据。

## 四、前瞻冻结的门禁

模型决策拆成四个正交轴，避免一个局部指标代表全部结论。

### A. 合同与独立性门

必须全部满足：

- 三模型、三 seed、fold 4、四期限的预测面板逐样本可配对；
- 评测窗口严格为 `2025-04-07..2025-12-23`；
- 训练/归一化统计不包含评测窗口；
- 所有父检查点和数据文件哈希与父 receipt 一致；
- 无测试标签参与训练、模型选择、阈值选择或 score 校准；
- 无凭据、token 或密码写入 config、日志和产物。

任一失败，最终状态为 `v46_contract_failed`。

### B. V42 机制外推门：student 对 true-target TCN

硬门：

- mean RankIC delta `>= 0.0000`；
- RankIC 配对分块 bootstrap 95% CI low `>= -0.0020`；
- 三个 seed 中至少两个 mean RankIC delta `> 0`；
- 四个 horizon 中至少三个 mean RankIC delta `> 0`；
- mean Top10% excess-return delta `>= -0.0001`；
- mean NDCG@Top10% delta `>= -0.0010`。

`top_membership_precision`、正收益率、高于横截面均值比例和换手率在本门中全部是诊断项，不得写入 blockers。

### C. LSTM 非劣—速度门：student 对 LSTM

TCN 不被假设一定优于 LSTM；本门检验在明显更快时是否没有出现不可接受的效果损失。硬门：

- RankIC delta 的配对分块 bootstrap 95% CI low `>= -0.0100`；
- Top10% excess-return delta 的 95% CI low `>= -0.0005`；
- NDCG@Top10% delta 的 95% CI low `>= -0.0100`；
- 父证据中的 TCN/LSTM model-step 速度比 `>= 3.0x`；
- student 推理前向次数严格为 `1`。

这是一项预先定义的非劣折中，不允许看到结果后改变容忍度。

### D. 阶段结论

- A 失败：`v46_contract_failed`；
- A 通过、B 失败：`v46_student_not_generalized`；
- A/B 通过、C 失败：`v46_student_generalized_but_not_lstm_competitive`；
- A/B/C 全通过：`v46_independent_research_candidate`。

无论哪种状态：

- `alpha_ready=false`；
- `deployment_authorized=false`；
- `trading_authorized=false`；
- V46 只决定是否成为下一阶段研究候选，不授权实盘或部署。

## 五、TDD 与实现要求

按纵向切片执行 red → green：

1. 先用人工可核对的小横截面锁定三个诊断指标的语义，证明集合重合率与正收益率可以方向相反；
2. 再锁定 V46 门禁，证明极差的 membership precision 不能制造 blocker，也不能独自触发 admitted；
3. 再锁定防复用边界，证明旧窗口、重叠窗口和错误 stage 必须 fail closed；
4. 最后实现一次性冻结检查点推理入口和不可变 receipt。

测试只穿过公共接口：指标评测 API、V46 决策 API、V46 命令行入口。不得测试私有函数或通过复制实现算法构造期望值。

## 六、必须产出的证据

- `config.resolved.json`：冻结窗口、阈值、模型与父证据身份；
- `window-audit.json`：旧窗口、embargo、新窗口、标签覆盖与训练统计边界；
- `checkpoint-manifest.json`：9 个冻结检查点的路径、模型、seed、训练 fold 和 SHA-256；
- `predictions.parquet`：三模型逐样本冻结预测；
- `utility-metrics.parquet`：语义明确的逐日逐期限指标；
- `model-summary.parquet`；
- `student-control-comparison.json`；
- `student-lstm-comparison.json`；
- `paired-bootstrap.parquet`；
- `seed-deltas.parquet` 与 `horizon-deltas.parquet`；
- `decision.json`；
- `receipt.json`：输入/输出哈希、代码身份、环境、一次性消费状态；
- `report.md`：结果、限制和不允许得出的结论。

## 七、停止规则

- 不搜索 teacher weight、学习率、dilation、channel、dropout、top-k、交易成本、seed、fold、horizon 权重或门槛；
- 不新增分类头，不以多任务模型绕过本轮问题；
- 不因 membership precision 单项好坏继续优化；
- 不用 V46 结果反向修改本提示词或重跑；
- 若 V46 失败，下一步先分析跨时期漂移或模型—基准差异，不在该窗口上局部调参；
- 若 V46 通过，也必须进入新的全市场 PIT、成本与容量验证，不能直接部署。

## 八、最终验收

运行定向测试、完整测试套件、mypy、ruff 和 production build。最终报告必须明确分开：

1. 速度基础设施是否通过；
2. V42 机制是否在独立时期外推；
3. TCN 是否达到相对 LSTM 的预注册非劣折中；
4. `top_membership_precision`、`top_positive_return_rate` 和 `top_above_cross_section_mean_rate` 各自说明什么；
5. 当前能否称为 research candidate，以及为什么仍不能称为 Alpha-ready 或 production-ready。
