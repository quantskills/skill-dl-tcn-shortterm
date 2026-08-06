# TCN 真实数据 Pilot 就绪包执行提示词

把下面整段提示词交给实现代理时，必须完整执行；不得把“代码已实现”“合成测试通过”替换成真实 Alpha、真实速度或候选模型证据。

## 角色

你是 `skill-dl-tcn-shortterm` 的量化研究工程负责人。目标是在不接触真实行情、不打开封存测试集、不连接券商、不写入外部系统的前提下，把仓库推进到“真实数据 pilot 可被确定性检查、缺失输入可被机器明确拒绝”的状态。

## 已知事实

- 项目是沪深 A 股、5 分钟输入、未来 1/2/3/5 个交易日横截面收益排序的离线研究流水线。
- 现有 runtime manifest schema v1 已被 `run_experiment` 和合成 fixture 使用，只负责运行所需的数据路径、数据类型和 SHA-256 身份。
- `docs/data-contracts.md` 还要求真实数据运行明确供应商、许可、可用时间、复权、PIT 状态、交易日历、模式和特征/标签版本。
- 工程路径已有合成验证；真实 RankIC/ICIR、净多头收益、容量和 TCN 相对 LSTM/GRU 的速度仍未知。
- 负面或无增量结果仍可构成合格的工程研究结论，但不得晋升为候选模型。

## 本次唯一交付目标

交付“离线工程研究库 + 真实数据 pilot 就绪包”，包括：

1. 独立、版本化的 pilot readiness descriptor；
2. fail-closed 的 Python 校验器；
3. 跨平台 CLI；
4. 非秘密示例配置；
5. 自动化测试；
6. 运行手册、工作项、验证状态和需求追踪更新。

不要修改现有 runtime manifest v1 的兼容语义。readiness descriptor 是运行前治理层，验证通过后才允许把其引用的 runtime manifest 交给 `run_experiment`。

## 强制安全边界

- 不下载、复制、生成或提交真实行情。
- 不读取或消费封存 holdout。
- 不创建券商、订单、实盘或外部写入集成。
- 不配置 Git remote，不推送，不创建 PR，不部署。
- 不提交 token、password、secret、API key、私有证书、`.env` 内容或供应商凭据。
- 示例只能包含明显占位值；真实 descriptor 必须保存在 Git 忽略的运行目录。
- 任一必要字段缺失、占位、顺序错误、类型错误、指纹不匹配或出现秘密形态字段时，结果必须是 `not-ready`。

## Readiness descriptor v1

顶层必须精确包含：

- `schema_version`：只能为 `1`；
- `deliverable`：只能为 `engineering-research-library`；
- `runtime_manifest_path`：指向既有 runtime manifest v1；
- `data_governance`；
- `evaluation_protocol`；
- `compute_protocol`；
- `research_budget`。

`data_governance` 至少要求非空的 provider、license、source/version、owner、timezone、calendar、bar timestamp、adjustment、availability、raw schema、canonical schema、universe、feature 和 label policy/version，并明确以下布尔门禁为真：

- `license_approved`
- `pit_instrument_state`
- `pit_corporate_actions`
- `survivorship_bias_controlled`

`evaluation_protocol` 必须：

- 固定 train、validation、ordinary test、sealed holdout 的起止日期；
- 满足严格时间顺序且区间不重叠；
- `embargo_days >= 5`；
- 明确 `purge_uses_label_end_at = true`；
- 明确 `sealed_holdout_accessed = false`；
- 指定非空的 holdout custodian，且不能与 model owner 相同；
- 固定模型集合，至少包含 Ridge、LightGBM、LSTM、GRU、Bai TCN；
- 固定指标集合，至少包含 RankIC、ICIR、net long-only return、throughput、time-to-best-validation、peak memory；
- 固定 promotion 配置引用，但本次不得执行 promotion。

`compute_protocol` 必须固定硬件标识、device、precision、batch size、epoch/early-stop 预算、seed 和确定性设置。`research_budget` 必须给出正数的最大迭代次数、最大 wall-clock 小时数和停止规则。

## 校验行为

- 返回结构化报告：`ready`、`checks`、`errors`、`warnings`、descriptor SHA-256、runtime manifest SHA-256。
- errors 非空时 `ready=false`；CLI 返回码为 `2`。
- 通过时 `ready=true`；CLI 返回码为 `0`。
- 解析/IO 错误也必须给出机器可读 JSON，并返回 `2`。
- 校验器必须递归拒绝 key 名包含 `password`、`secret`、`token`、`api_key`、`private_key`、`credential` 的字段。
- 必须拒绝常见占位值，如 `TODO`、`TBD`、`CHANGE_ME`、`<...>`。
- 必须验证 runtime manifest 文件存在、schema v1、dataset kind、所有已声明文件及其 SHA-256；不得创建实验输出。
- 不得因为 readiness 通过而声称真实数据正确、模型有效、速度达标或获准运行封存测试。

## 实现位置

- 核心模块：`src/skill_dl_tcn_shortterm/readiness.py`
- 公共导出：`src/skill_dl_tcn_shortterm/__init__.py`
- CLI：`tasks/check_pilot_readiness.py`
- 示例：`config/pilot-readiness.example.json`
- 测试：`tests/test_pilot_readiness.py`
- 使用说明：`docs/real-data-pilot.md`
- 状态更新：`README.md`、`docs/WORK_ITEMS.md`、`docs/verification.md`、`docs/requirements-traceability.md`

## 测试要求

至少覆盖：

1. 完整 descriptor 与微型 runtime manifest 通过；
2. 缺失字段和占位值失败；
3. 秘密形态 key 在任意嵌套层失败；
4. runtime data SHA-256 不匹配失败；
5. split 重叠、乱序、embargo 小于 5 失败；
6. sealed holdout 已访问失败；
7. custodian 与 model owner 相同失败；
8. 基准或必需指标缺失失败；
9. CLI 对 ready/not-ready 返回 0/2 且输出 JSON；
10. 校验过程不创建实验运行目录、不消费 holdout。

## 最终验证

依次执行并记录实际结果：

```text
python -m mypy
python -m ruff check .
python tasks/preflight.py
python tasks/test.py
python -m build
```

最终报告必须区分：

- 已实现并经合成验证的 readiness 能力；
- 仍需所有者提供的数据、许可、预算和 holdout 管理信息；
- 尚未执行的真实 benchmark、封存评估、GitHub 发布和部署。

## 停止条件

出现以下任一情况立即停止并报告，不得自行放宽：

- 需要真实数据、付费服务、凭据或外部写入；
- 需要打开封存 holdout；
- 现有用户修改与本任务冲突；
- 无法在不破坏 runtime manifest v1 的情况下实现；
- 完整验证失败且无法在当前授权范围内修复。
