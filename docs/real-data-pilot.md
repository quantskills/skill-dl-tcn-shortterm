# Real-data pilot readiness

本项目把“是否具备运行条件”和“实际运行实验”分成两个动作。readiness 检查只读取 descriptor、runtime manifest、预登记 promotion 配置及其声明的数据文件并核验身份；它不会创建实验目录、训练模型或打开封存测试集。

## 当前边界

readiness PASS 只表示首个非封存真实数据 pilot 的输入、治理和比较协议已经完整声明。它不表示：

- 数据内容已经被独立审计；
- TCN 存在真实 Alpha；
- TCN 比 LSTM/GRU 快 3–5 倍；
- 模型满足候选晋升条件；
- 获得 GitHub 发布、部署、券商连接或封存测试授权。

## 准备 descriptor

1. 把 `config/pilot-readiness.example.json` 复制到不受 Git 跟踪的研究运行目录。
2. 替换全部 `<...>` 占位值；不要在 descriptor 中放置凭据。
3. 创建 runtime manifest v1，并为下列本地文件记录 SHA-256：
   - 原始 1 分钟条；
   - PIT 标的状态；
   - PIT 公司行动；
   - 下一开盘成交状态。
4. 在打开任何封存数据前固定 promotion 配置文件并记录其 SHA-256。
5. 指定相互独立的 model owner 与 sealed-holdout custodian。

现有 runtime manifest v1 继续作为 `run_experiment` 的输入身份契约。pilot readiness descriptor 是额外的运行前治理契约，不改变 manifest v1 的兼容行为。

## 运行检查

```text
python tasks/check_pilot_readiness.py --descriptor D:\research-runs\tcn\pilot-readiness.json
```

返回码：

- `0`：`ready=true`，所有声明条件通过；
- `2`：`ready=false`，至少一个条件缺失或无效。

CLI 始终输出 JSON。主要字段为：

- `ready`：整体结论；
- `checks`：逐项 pass/fail；
- `errors`：阻塞原因；
- `warnings`：不能从 readiness 推导的结论；
- `descriptor_sha256` 与 `runtime_manifest_sha256`：本次检查身份。

## Fail-closed 条件

以下任一情况都会阻止 pilot：

- descriptor 字段缺失、额外顶层字段、空值或占位值；
- 任意嵌套 key 含 credential、password、secret、token、API key 或 private key 形态；
- 数据许可、PIT 标的状态、PIT 公司行动或幸存者偏差门禁未确认；
- runtime manifest 或任一声明文件不存在、schema 不匹配或 SHA-256 漂移；
- train、validation、ordinary test、sealed holdout 乱序或重叠；
- embargo 少于 5 日，或 purge 不基于实际 `label_end_at`；
- 封存数据已被访问，或 holdout custodian 与 model owner 相同；
- Ridge、LightGBM、LSTM、GRU、Bai TCN 或必需比较指标缺失；
- promotion 预登记文件不存在或指纹漂移；
- 计算预算、随机种子、确定性设置或研究停止规则未固定。

## Readiness 之后

只有所有者另外提供真实数据运行授权后，才可以把通过检查的 runtime manifest 交给：

```text
python tasks/run_experiment.py --config <resolved-config.json> --manifest <runtime-manifest.json> --output-root <untracked-run-root>
```

首个真实运行应先限定为非封存 validation pilot。保存全部强制产物并审计 PIT、成本、容量、性能与失败项之后，再决定是否冻结候选规则。封存测试仍需独立授权并且只能按预登记状态机消费一次。

完整代理执行契约见 `docs/prompts/real-data-pilot-readiness.md`。
