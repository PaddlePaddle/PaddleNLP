# 情感分析应用

## **1. 情感分析简介**
情感分析（sentiment analysis）是近年来国内外研究的热点，旨在对带有情感色彩的主观性文本进行分析、处理、归纳和推理。情感分析具有广泛的应用场景，可以被应用于消费决策、舆情分析、个性化推荐等领域。

按照分析粒度可以大致分为三类：篇章级的情感分析（Document-Level Sentiment Classification）、语句级的情感分析（Sentence-Level Sentiment Classification）和属性级的情感分析（Aspect-Level Sentiment Classification）。其中属性级的情感分析又包含多项子任务，例如属性抽取（Aspect Term Extraction）、观点抽取（Opinion Term Extraction）、属性级情感分析（Aspect-Based Sentiment Classification）等。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/199965793-f0933baa-5b82-47da-9271-ba36642119f8.png" />
</div>



## **2. 情感分析项目介绍**

PaddleNLP情感分析应用立足真实企业用户对情感分析方面的需求，同时针对情感分析领域的痛点和难点，基于前沿模型开源了细粒度的情感分析解决方案，助力开发者快速分析业务相关产品或服务的用户感受。针对情感分析应用，本项目不仅提供了基于Taskflow开箱即用的情感分析能力，还提供了从输入数据到情感分析结果可视化的能力，另外考虑到一些企业用户需要针对业务场景进行适配，本项目同时提供了完整的情感分析定制方案：数据标注 - 模型训练 - 模型测试 - 模型部署 - 情感分析可视化。

当前PaddleNLP情感分析应用更多聚焦于属性级的情感分析，支持文本评论中关于属性、观点词和情感倾向方面的分析。当前提供了两种情感分析方案：基于通用信息抽取模型UIE的情感分析方案和基于情感知识增强模型SKEP的情感分析方案。

基于UIE的情感分析方案采用 Prompt Learning 的方式进行情感信息抽取，该分析方式需要预先定义情感信息抽取的schema，然后通过该schema逐步分析和抽取情感信息。 相比基于SKEP的情感分析方案，UIE方案在测试中表现出了更好的效果。在测试中，通过精确匹配的方式对比抽取的 属性、情感倾向和观点词 三者信息，即当三者全部匹配才算抽取正确，下表展示了此次测试的评测指标：

|  模型 | 权重 | Precision | Recall | F1 |
|  :---: | :--------: | :--------: | :--------: | :--------: |
| `SKEP` | `skep_ernie_1.0_large_ch` | 0.76368 | 0.74710 | 0.75530 |
| `uie` | `uie-senta-base` | 0.89593 | 0.86125 | 0.87825 |


基于SKEP的情感分析方案主要采用两阶段式的情感分析抽取，首先通过序列标注的方式定位属性词和观点词，然后通过结合属性词和观点词两者信息进行属性情感极性分类。相比基于UIE的情感分析方案，基于SKEP的情感分析方案具有更快的预测速度。下表展示了在测试集上平均每分钟预测的样本数，可以看到SKEP方案的预测速度显著快于UIE方案。

|  模型 | 权重 | 预测样本数/m |
|  :---: | :--------: | :--------: |
| `SKEP` | `skep_ernie_1.0_large_ch` | 3428 |
| `uie` | `uie-senta-base` | 1104 |

备注： 当前只有基于UIE的方案支持情感分析结果可视化能力，基于SKEP的方案暂不支持。

## **3. 快速开始**

- 👉 [基于UIE的情感分析方案](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/sentiment_analysis/unified_sentiment_extraction)

- 👉 [基于SKEP的情感分析方案](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/sentiment_analysis/ASO_analysis)
