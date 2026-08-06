# TCN 追加式相对时序特征消融 v38 结果

## 决策

v38 Phase A 已完整执行，状态为 `stop_append_relative_sequence_seed7_v38`。候选在全池 RankIC 上达到预登记均值和 fold 稳定性门，但 Top precision、NDCG、Top return 和 RankIC bootstrap 下界门失败，因此没有授权 seeds 17/27 或 LSTM Phase B。旧 sealed test 未访问。

完整提示词见 `docs/prompts/tcn-append-relative-sequence-ablation-v38.md`。十通道特征收据为 `artifacts/tcn-appended-relative-sequence-top20-v38/receipt.json`；Phase A 收据为 `artifacts/tcn-appended-relative-seed7-screen-v38/receipt.json`，receipt ID `e804c897fbf26f0c0a262cca19c142b21ee87401fe92d84e73bb1e123939030f`。

## 候选与协议

- 完整保留 base8：`close_return/open_close_return/intrabar_range/log_volume/log_amount/vwap_deviation/time_sin/time_cos`；
- 只追加 v37 已审计的 `log_amount_to_adv20` 与 `log_amount_to_market_cap`；
- 不包含五个静态横截面秩，不复制 date-level 数值到 480 步；
- tensor 为 23,821 个 `[10,480]` float32 memmap 窗口，前 8 通道与 base 逐位相同；
- 固定 dynamic-horizon-skip TCN、seed 7、folds 0..4、8 epochs、SmoothL1、batch 128、8 threads；
- base TCN 直接复用 v37 带哈希的同协议 seed-7 predictions 和 leaderboard，不重训、不重新选择。

## Phase A 结果

| 指标 | Base TCN | 十通道 TCN | Delta |
|---|---:|---:|---:|
| mean RankIC | 0.096989 | 0.099245 | +0.002257 |
| Top return | 0.003985 | 0.003212 | -0.000772 |
| Top precision | 0.119375 | 0.114375 | -0.005000 |
| NDCG@Top | 0.573912 | 0.561662 | -0.012250 |
| Turnover | 0.589241 | 0.588291 | -0.000949 |

逐折 RankIC delta 为 `+0.013989/+0.002520/-0.017509/+0.009447/+0.002836`，4/5 folds 为正。RankIC 的日期 block-bootstrap 95% CI 为 `[-0.008089,+0.012853]`。Top precision CI 为 `[-0.015938,+0.006875]`，NDCG CI 为 `[-0.023463,-0.000784]`，Top return CI 为 `[-0.002353,+0.001102]`。

候选 TCN 参数量从 6,348 增至 6,476；median samples/s 为 5,006.37，对 base 4,988.17 的保留率为 `1.0036`。速度没有回退。

失败门为：

- `top_precision_delta_below_gate`；
- `ndcg_delta_below_gate`；
- `top_return_delta_below_gate`；
- `rankic_ci_low_below_gate`。

## 解释

保留原始 `log_volume/log_amount` 后，v37 的 RankIC 大幅下降已经消失，且十通道候选在 4/5 folds 有正增量。这支持“v37 的破坏性替换是主要错误”这一判断，也说明两个相对时序通道可能携带全池排序信息。

但当前 top20 的 Top10% 每日期只有 2 只股票。候选改善了整体排序，却降低 top2 的 NDCG、precision 和 realized return；其中 NDCG 置信区间完整低于零。这不能解释为单纯随机噪声，也不能只凭 RankIC 晋级。Phase B 按预登记规则停止。

## 下一步

下一优先级应是修复 top50 PIT 状态，而不是继续在相同 top20 validation 上删减或组合通道：

1. 本地 top100 分钟权重已完整，top50 需要 60,600 个 PIT 状态键；当前缺 36,375 个 market-cap/state 键；
2. 补齐 top50 的因果 ADV、市值、公司行为和资格状态后，Top10% 将从 2 只扩展到约 5 只，可直接检验尾部退化是否来自横截面宽度；
3. top50 上先固定 base8 与 v38 十通道各一个 TCN，不搜索新结构；
4. 若 top50 仍表现为 RankIC 改善但尾部退化，再实现独立 post-encoder stock-context/DeepSets adapter，而不是把静态值复制到时间轴；
5. 继续禁止复用 v36 sealed；新的普通验证改善也不构成部署或交易授权。
