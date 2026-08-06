# TCN 金融短线同类项目调研

调研日期：2026-08-01
范围：Hermes 知识库、知乎、微信公众号、GitHub、原始论文
证据原则：论文正文和源代码优先；README 中的性能自述只作为项目声明，不当作已复现实验结果。

## 结论

存在多个“TCN + 金融时序”项目，但未发现一个公开项目同时覆盖本项目的完整问题定义：分钟线输入、未来 1/2/3/5 个交易日横截面收益排序、严格 PIT、chronological walk-forward、purge/embargo、Arrow/memmap 大数据加载、统一基准、RankIC/ICIR、含成本回测和冻结测试。

最接近的公开研究是 Dai、An、Long 的超高频股票价格变化 TCN：它使用深证 100 中 99 只股票的 990 万笔成交数据，比较 TCN、TCN-attention、LSTM 和 GARCH。但其任务是下一笔价格变化的离散分类，采用 7:3 切分，不是 1–5 日横截面预测，也未提供本文核验到的开源实现、purge/embargo 或含成本组合回测。

因此，外部项目适合拆分借鉴，不适合整体复刻；当前项目的差异化主要在金融数据契约、验证治理和多期限组合评估，而不只是 TCN 网络本身。

## 主要同类项目

| 来源 | 接近点 | 主要差异/风险 | 证据状态 |
|---|---|---|---|
| [Dai et al., UHF TCN](https://arxiv.org/abs/2107.00261) | 中国股票、990 万笔超高频成交、TCN/attention/LSTM/GARCH 对比 | 下一笔价格变化分类；7:3 切分；类别极不平衡；无已核验代码、成本回测和 purge/embargo | 论文全文深读 |
| [j00000ker/DTNN](https://github.com/j00000ker/DTNN) | LOB 高频数据、TCN 分支、三分类、LSTM/CNN 等基线 | TCN 只是混合模型分支；窗口 100；不是多日横截面排序；文本数据加载 | README 与 TCN 源码深读 |
| [ByronBroughten/Machine-Learning-Trading-Bot](https://github.com/ByronBroughten/Machine-Learning-Trading-Bot) | OHLC、可配置预测期限、时间靠后的测试集、训练集归一化、Zarr 分批加载 | 老项目且验证证据有限；无 purge/embargo；连接券商 API，不符合本项目离线边界 | README 深读，性能未复现 |
| [alessandrobaldo/PredictionModelsInForexMarkets](https://github.com/alessandrobaldo/PredictionModelsInForexMarkets) | 1 分钟至日频、多资产、TCN/CNN/LSTM/ML 对比、交易模拟 | Forex 单序列下一时点价格回归；Notebook 为主；未核验 PIT、purge/embargo 和真实成本处理 | README 深读，结果未复现 |
| [michaelmc9666/thesis](https://github.com/michaelmc9666/thesis) | TCN/LSTM 对照、时间注意力、动态特征门控、chronological 80/20 | 日线、8 只美股、单股票方向预测；无分钟线和 walk-forward/purge/embargo | README 深读，结果未复现 |
| [paul-krug/pytorch-tcn](https://github.com/paul-krug/pytorch-tcn) | 可复用 PyTorch TCN、严格 causal 开关、WeightNorm、dilation reset、流式 buffer、ONNX | 通用模型库，不含金融标签、PIT、验证和回测 | README/接口深读 |
| [locuslab/TCN](https://github.com/locuslab/TCN) | Bai 2018 官方基线实现和序列基准 | 非金融项目，代码年代较早 | 官方仓库深读 |
| [reedfenno/tcn-trading-system](https://github.com/reedfenno/tcn-trading-system) | 1m–1d 多周期、方向/收益/波动率多任务、回测、Optuna | C++ 示例把 BatchNorm 放入 TCN 块；代码集中在大型测试文件；因果性和验证需独立审计 | README 和局部源码深读；不建议直接复用 |
| [taleblou/TemporalConvolutionalNetworks-Price-Prediction](https://github.com/taleblou/TemporalConvolutionalNetworks-Price-Prediction) | 多金融资产的 TCN 价格预测声明 | README 同时称 TCN 与 gradient boosting，且高 R² 缺少严格时序验证回执 | README 深读；低置信度 |

## 中文内容检索

### Hermes 知识库

- 精确检索 `TCN 股票预测` 没有独立同类项目命中。
- `Temporal Convolutional Network` 只命中通用深度学习模型设计知识页，其中建议分钟级长序列使用 Temporal CNN/Transformer。
- 当前 TCN 项目的两份每日摘要是知识库中最具体的 TCN 工程记录，但属于本项目自身证据，不是外部复现。
- “量化全栈 6 层 Pipeline”包含数据契约、walk-forward、purge/embargo 和回测思路，但知识库自身明确标注其模型层存在占位实现、purge/embargo 较浅，不能作为等价项目。

### 知乎

- [DolphinDB：高频行情低频化因子库](https://zhuanlan.zhihu.com/p/2030576049969419663)不是 TCN 项目，但与数据侧高度相关：将分钟/Tick 数据聚合为日频特征，并强调分区、并行计算、统一因子模板和海量数据性能。
- 搜索还发现一篇使用沪深 300 成分股 5 分钟 K 线的 CNN-LSTM/VAE 付费内容。它不是 TCN，且正文受付费页面限制，本次只达到候选发现状态，不据此引用性能结论。

### 微信公众号

- [拓端研究室：Python、R 时间卷积神经网络 TCN 与 CNN、RNN](https://mp.weixin.qq.com/s/XUJkI8ZMvat5iCHtxBMtZw)可读取全文，介绍 causal/dilated convolution、滑动窗口、训练/验证/测试划分和 WeightNorm 选项。
- 文章是通用时序教程，示例并未建立金融 PIT、walk-forward、交易成本或 1–5 日横截面任务，因此只能用于入门说明，不能作为量化效果证据。

## 对关键技术表述的校正

### 感受野

“感受野 = `2^层数`”只是过度简化。对 stride=1 的卷积序列，通用形式是：

```text
R = 1 + Σ_l (k_l - 1) d_l
```

若 dilation 为 `1, 2, 4, ..., 2^(B-1)`，每个残差块有两层卷积且 kernel size 固定为 `k`，则：

```text
R = 1 + 2 (k - 1) (2^B - 1)
```

Bai 2018 的残差块正是每块两层 dilated causal convolution。因此配置必须基于真实 block/conv 数计算，不能只按块数取 `2^B`。

### 因果 padding

可行实现包括显式左侧 padding，或先进行 padding 再删除右端多余输出（chomp）。验证应使用“修改未来输入不得影响过去输出”的自动化测试，而不是只检查 `causal=True` 配置名。

### 归一化

Bai 2018 基线对卷积滤波器使用 Weight Normalization，并在每层后使用 dropout。BatchNorm 不是论文基线；金融非平稳序列和小 batch 下更应谨慎使用依赖 batch 统计量的归一化。

### 训练速度

Bai 2018 支持的结论是卷积可以沿时间维并行，而 RNN 需要顺序递推；论文和本次找到的金融项目都没有给出可迁移到当前分钟线流水线的统一“快 3–5 倍”证据。该倍率必须继续标记为待本项目同硬件、同数据、同 batch、同参数预算实测的假设。

## 推荐借鉴顺序

1. 以 Bai 2018 和 `paul-krug/pytorch-tcn` 校验 causal、WeightNorm、感受野和流式 buffer 语义。
2. 以 Dai et al. 的超高频研究设计类别不平衡与 TCN-attention 消融，但保留本项目更严格的 walk-forward/PIT 门禁。
3. 借鉴 Byron 项目的 out-of-core Zarr 思路，与当前 Arrow/memmap 路径做同预算对照，不接入券商。
4. 若未来引入 Level-2，参考 DTNN 的 LOB 特征和 TCN 分支，不把复杂 hybrid 直接作为第一基线。
5. 借鉴动态特征门控做解释性实验，但不能让解释模块改变冻结测试规则。
6. 借鉴 DolphinDB 的“分钟/Tick → 稳定日频特征”分区与批处理思想，保持数据处理与模型训练解耦。

## 不应复制的模式

- 直接预测绝对价格并以高 R² 作为策略有效性证明。
- 随机切分时间序列，或在全量数据上拟合 scaler/特征选择器。
- 只做单股票单期限 accuracy/MAE，不做横截面 RankIC、成本和换手。
- 以 README 自述、图表或单次回测替代可重放收据。
- 在没有真实硬件基准前宣传固定的 TCN/LSTM 加速倍数。
- 将研究系统直接连接券商或自动下单。

## 核心一手来源

- Bai, Kolter, Koltun, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- Dai, An, Long, [Price change prediction of ultra high frequency financial data based on temporal convolutional network](https://arxiv.org/abs/2107.00261)
- [locuslab/TCN](https://github.com/locuslab/TCN)
- [paul-krug/pytorch-tcn](https://github.com/paul-krug/pytorch-tcn)
- [j00000ker/DTNN](https://github.com/j00000ker/DTNN)

## 2026-08-02 工程延展

- Bai 2018全文只论证卷积的结构性并行能力，没有报告可迁移的3–5× wall-clock倍率；`training time`与`wall-clock`均无对应实验表。
- [pytorch-tcn](https://github.com/paul-krug/pytorch-tcn)的NCL输入、dilation reset和流式buffer支持“减少热路径搬运、限制无效padding”的方向，但不构成金融效果证据。
- [OneNet](https://github.com/yfzhang114/OneNet)用在线组合、延迟反馈和漂移适应处理时序分布变化，支持将fold失稳优先视为regime/更新协议问题，而不是只增加TCN宽度。
- [Ubiquant量化预测方案](https://www.zhihu.com/en/article/666354415)使用PurgedGroupTimeSeries/TimeSeriesSplit，并因CV不稳定排除部分神经网络，支持“稳定验证优先于偶然线上或单折改善”的门禁原则。
- 本项目v4实测进一步表明：CPU线程数是速度能否兑现的关键工作负载参数；恢复速度并不会自动恢复预测效果，因此速度与RankIC必须分开验收。

## 检索覆盖说明

四个用户指定来源均进行了真实检索：Hermes 知识库、知乎、微信公众号和 GitHub 均达到 `effective`；知识库两篇文档、知乎 DolphinDB 正文、微信公众号 TCN 正文以及多个 GitHub README/源码达到 `deep_read`。知乎的 5 分钟 CNN-LSTM 付费内容只达到 `effective`，没有绕过访问限制。
