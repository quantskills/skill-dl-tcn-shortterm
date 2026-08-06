# Agent 接口契约

不同 Agent 只调用同一个版本化命令，不直接调用 `tasks/`、实验版本脚本或内部 Python 模块：

```text
tcn-shortterm-skill run --request <request.json>
```

## Request v1

Request 是一个 JSON 对象，且只允许以下字段：

```json
{
  "schema_version": "1",
  "action": "run",
  "config_path": "config.json",
  "manifest_path": "manifest.json",
  "output_root": "runs"
}
```

- 所有字段必填，未知字段应拒绝。
- 相对路径以 request 文件所在目录为准。
- `output_root` 可以尚不存在；已存在的相同 run ID 不得覆盖。
- Request 本身不包含密码、token、供应商账户或内嵌行情。
- 机器可读的准确约束以 `tcn-shortterm-skill schema --kind request` 为准。

## Result v1

命令只向标准输出写一个 JSON 对象。成功结果至少包含：

```text
schema_version, status, action, request_digest, engine,
run_id, authoritative_run_manifest, warnings, errors
```

`authoritative_run_manifest` 是执行引擎生成的权威收据。Agent 可以解释它，但不得修改、替换或把派生摘要描述成权威结果。

失败使用非零退出码并返回 `status=failed` 与非空 `errors`。机器可读的准确约束以 `tcn-shortterm-skill schema --kind result` 为准。

## 可移植性

- 首选 console script；不可用时使用 `python -m skill_dl_tcn_shortterm`。
- 接口不依赖 Codex、Hermes、MCP、WSL、特定 shell 或本机绝对路径。
- 最小样例由 `tcn-shortterm-skill example --output-dir <empty-directory>` 生成。
- `demo` 和 `example` 只验证接口链路，不训练 TCN，也不是效果证据。

## 安全边界

Skill 只读取 request 明确引用的本地文件，并只写入 request 指定的输出根。它不自动下载数据、不登录供应商、不访问券商、不提交订单，也不把研究结果升级为部署或交易授权。
