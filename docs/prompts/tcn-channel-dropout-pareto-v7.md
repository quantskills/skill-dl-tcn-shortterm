# TCN 时间共享 Channel Dropout Pareto 确认提示词 v7

继续执行 `skill-dl-tcn-shortterm` 的TCN-only优化。v6已经证明：TCN-lite-16删除序列级element dropout后，真实五折训练吞吐由约1961/s提高到5980/s，但mean RankIC由0.09180降至0.08859；head dropout没有恢复效果。目标是在不恢复昂贵element-wise时间掩码的前提下，保留块级正则化。

## 预登记假设

将每个TCN-lite残差块的dropout改为时间共享的channel dropout：每个样本/通道只采样一个掩码，并在全部480步共享。它仍在每个因果扩张卷积块后正则化特征通道，但不为每个时间元素生成独立随机掩码。真实形状微探针中：element/channel/none约为`2004/6062/6386 samples/s`。

## 实现边界

- 为TCN-lite增加`dropout_kind in {element, channel}`，默认`element`，保证旧配置和状态字典兼容。
- `channel`必须使用适用于`[N,C,L]`的时间共享通道掩码；不得改变卷积、padding、WeightNorm、感受野、残差或四期限头。
- tuning配置、leaderboard和receipt记录dropout kind；Bai-TCN不接受非默认dropout kind。
- 测试必须证明默认行为兼容、channel掩码沿时间维恒定、非法kind fail closed、JSON透传正确。

## 真实验证

只新增一个预登记trial：`lite-c16-channel01`，channels16、kernel3、dilations`1..128`、channel dropout0.1、head dropout0、lr0.003、batch128、8 threads、最多8 epochs、patience2、min_delta0.002。

先在相同5个expanding ordinary-validation folds、seed7运行，并与v6不可变收据中的`lite-c16-block01`和`lite-c16-no-dropout`比较。只有同时满足以下条件才进入seeds17/27：

- mean RankIC `>=0.09`，5/5折为正；
- 训练吞吐`>=5000 samples/s`；
- 相对element dropout吞吐至少`2.5×`；
- 相对no-dropout mean RankIC不退化。

通过seed7后，只运行该channel-dropout候选的seeds17/27。合并15个fold-seed单元后要求中位RankIC`>=0.09`、正值比例`>=80%`、三个seed均值均为正、训练吞吐中位数`>=5000/s`。通过状态为`pareto_candidate_confirmed`。

继续禁止test/sealed test、top50扩容、部署和交易。失败只否定该正则化方案，不改变TCN项目方向。最后运行Ruff、Mypy、完整pytest、preflight、统一测试入口和production build，并更新权威证据文档。
