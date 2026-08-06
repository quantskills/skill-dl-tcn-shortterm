# TCN 无激活开销 Weight Decay Pareto 提示词 v8

继续TCN-only优化。v6无dropout TCN-lite-16达到约5980 samples/s和0.08859 mean RankIC；v7 channel dropout达到5477/s但RankIC降至0.08329。不得再增加激活dropout变体。本轮只测试无激活开销的AdamW权重衰减能否恢复缺失的约0.00141 RankIC。

## 实现

- `TCNTuningTrial`增加非负`weight_decay`，默认0保持既有Adam行为完全兼容。
- `weight_decay=0`继续使用Adam；正值使用AdamW，并在leaderboard/receipt记录optimizer与weight decay。
- 非法负值fail closed；JSON透传有回归测试。

## 预登记筛选

相同5个expanding ordinary-validation folds、seed7、TCN-lite-16、kernel3、dilations`1..128`、block/head dropout均0、lr0.003、batch128、8 threads、最多8 epochs、patience2、min_delta0.002。只新增：

- `lite-c16-wd1e5`
- `lite-c16-wd1e4`
- `lite-c16-wd1e3`

与v6不可变的`lite-c16-no-dropout`比较。只有mean RankIC`>=0.09`、5/5折为正、吞吐`>=5000/s`且相对无dropout RankIC不退化的第一名进入seeds17/27。合并15单元后要求中位RankIC`>=0.09`、正值比例`>=80%`、三个seed均值为正、吞吐中位数`>=5000/s`，通过状态为`pareto_candidate_confirmed`。

禁止新增weight decay值、学习率、通道数或其他结构；禁止test/sealed test、top50、部署和交易。如果全部失败，记录为`stop_no_pareto_gain`，保留v5的速度TCN与效果TCN双配置，后续转向数据/目标与TCN表征研究而不是继续infra微调。

最后运行Ruff、Mypy、完整pytest、preflight、统一测试入口和production build，并更新权威文档。
