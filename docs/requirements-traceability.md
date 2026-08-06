# Requirements traceability

`verified-synthetic` 表示实现已由确定性合成 fixture 验证；`verified-real-smoke` 表示只读真实数据已完成有界接入烟测；`verified-real-validation` 表示已经在真实普通验证集上执行，但没有触碰封存测试，也不代表 Alpha；`implemented-unexecuted` 表示门禁或测量能力已实现，但尚未在真实硬件/封存数据上执行。

| Requirement | Contract/design | Planned implementation | Verification | Status |
|---|---|---|---|---|
| Durable direction and safety | `PROGRAM.md`, ADR-0001/0003 | governance files | `python tasks/preflight.py` | verified-synthetic |
| Domain language | `CONTEXT.md` | all modules | terminology and ticket review | verified-synthetic |
| Provider-independent local data | `docs/data-contracts.md` | DATA-001/002 | aggregation, schema and fingerprint tests | verified-synthetic |
| Read-only Hermes DuckDB source | `docs/duckdb-source.md`, `docs/data-contracts.md` | DATA-003 | temporary database tests and bounded real-data aggregation run | verified-real-smoke |
| DuckDB session normalization | `docs/prompts/duckdb-canonicalization-real-smoke-v1.md`, `docs/duckdb-source.md` | DATA-004 | auction-fill, unknown-gap and 2024/2025 bounded real-data receipts | verified-real-smoke |
| DuckDB large-sample trainability audit | `docs/prompts/duckdb-large-sample-pilot-v1.md`, `docs/duckdb-source.md` | DATA-005 | provenance, contiguous-window and immutable-receipt tests plus full-range real receipt | verified-real-smoke |
| PandaData five-year PIT ingestion and TCN smoke | `docs/prompts/pandadata-five-year-pit-pilot-v1.md`, `docs/verification.md` | DATA-006 | adapter/resume/coverage tests, real receipt `5acdde65d2cb7a47`, preprocess run `20d9b8b6ea82ea72`, TCN run `0186f345a4bbe622` | verified-real-smoke |
| PIT A-share universe | `CONTEXT.md`, `docs/data-contracts.md` | UNIVERSE-001 | as-of, future-state and liquidity tests | verified-synthetic |
| 5-minute causal input windows | `docs/data-contracts.md` | DATA-002, LOADER-001 | aggregation, rejection and memmap tests | verified-synthetic |
| Next-open multi-horizon labels | ADR-0002, `docs/data-contracts.md` | SAMPLE-001 | hand-calculated and corporate-action fixtures | verified-synthetic |
| Cross-sectional rank targets | `docs/data-contracts.md` | SAMPLE-001 | date/horizon isolation tests | verified-synthetic |
| Bai TCN receptive field and causality | `docs/architecture.md` | TCN-001 | causal invariance and RF=509 tests | verified-synthetic |
| Shared four-head prediction | `docs/architecture.md` | TCN-001 | shape, mask and loss tests | verified-synthetic |
| Walk-forward leakage control | `docs/architecture.md` | SPLIT-001 | purge, embargo and sealed-state tests | verified-synthetic |
| Comparable baselines | `docs/architecture.md` | BASELINE-001/002 | shared samples/budget and metric tests | verified-synthetic |
| Conservative A-share backtest | `docs/architecture.md` | BACKTEST-001 | vintage and deterministic execution-ledger tests | verified-synthetic |
| Performance hypothesis | `PROGRAM.md`, `docs/tcn-landing-plan.md`, `docs/prompts/tcn-cpu-throughput-and-stability-v4.md`, `docs/prompts/tcn-infra-and-capacity-v5.md` | PERF-001, TCN-004, TCN-005 | fold-cached RankIC and scoped 4-thread real receipts; TCN-lite/LSTM 2.889× model-step and 2.520× end-to-end after validation fixed-cost removal | verified-real-validation |
| Validation-only TCN tuning and scale gate | `docs/prompts/tcn-validation-optimization-and-scale-gate-v3.md`, `docs/verification.md` | TCN-003 | five public behavior tests plus fixed-seed Phase A/B and batch-sweep receipts; gate result `stop_no_validation_gain` | verified-real-validation |
| Five-fold multi-seed TCN stability | `docs/prompts/tcn-cpu-throughput-and-stability-v4.md`, `docs/verification.md` | TCN-004 | leakage-safe expanding/sliding manifests, 5 folds × 3 seeds × 4 models, machine-readable speed/effect gate; result `stop_unstable_validation` | verified-real-validation |
| TCN infra and reasonable-capacity recovery | `docs/prompts/tcn-infra-and-capacity-v5.md`, `docs/verification.md` | TCN-005 | cached RankIC equivalence/position safety, loader lifecycle and thread restoration tests; 5-fold screen plus seeds 17/27 confirmation; Bai-TCN-16 paired median gain 0.01559 over TCN-lite-4 | verified-real-validation |
| TCN speed-effect Pareto regularization | `docs/prompts/tcn-pareto-convergence-v6.md`, `docs/prompts/tcn-channel-dropout-pareto-v7.md`, `docs/prompts/tcn-weight-decay-pareto-v8.md` | TCN-006 | head/channel dropout and AdamW behavior tests plus three bounded five-fold receipts; no candidate simultaneously reached RankIC 0.09 and 5000 samples/s | verified-real-validation |
| Reproducible reporting | `docs/reproducibility.md` | REPORT-001 | structured recomputation and digest drift tests | verified-synthetic |
| Frozen candidate decision | ADR-0003, `docs/prompts/tcn-v35-once-only-sealed-readiness-v36.md` | FREEZE-001, TCN-009, TCN-010 | real once-only sealed receipt `68205700cf0a21b807c6ea8c6e5aea351a9732cda0662cf89d27a1fe2fe02db9`; permanent consumption marker completed; result `sealed_rejected_tcn_candidate_v36` | verified-real-sealed-rejected |
| Real-data pilot readiness | `docs/data-contracts.md`, `docs/reproducibility.md`, `docs/real-data-pilot.md` | PILOT-001 | descriptor, secret, fingerprint, split, holdout and CLI tests | verified-synthetic |
