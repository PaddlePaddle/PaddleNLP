# 文本摘要应用

## **1. 文本摘要简介**
文本摘要的目标是自动地将输入文本转换成简短摘要,为用户提供简明扼要的内容描述，是缓解文本信息过载的一个重要手段。
文本摘要也是自然语言生成领域中的一个重要任务，有很多应用场景，如新闻摘要、论文摘要、财报摘要、传记摘要、专利摘要、对话摘要、评论摘要、观点摘要、电影摘要、文章标题生成、商品名生成、自动报告生成、搜索结果预览等。


## **2. 文本摘要项目介绍**

PaddleNLP文本摘要应用主要针对中文文本数据上的摘要需求，基于最前沿的文本摘要预训练模型，开源了文本摘要解决方案。针对文本摘要应用，本项目提供了基于Taskflow开箱即用的产业级文本摘要预置任务能力，无需训练，一键完成文本摘要预测。除此之外，本项目提供给用户定制化训练策略，可以结合用户自身的不同数据需求完成模型的训练、预测和推理部署工作。对于需要特殊能力的文本摘要预训练模型，本项目开源了摘要模型的预训练代码，用户可以使用大规模无标注数据定制在特定领域有摘要能力的预训练模型。

本项目使用的基础模型为[PEGASUS（Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence models)](https://arxiv.org/pdf/1912.08777.pdf)， 是由谷歌公司提出的文本摘要预训练模型。其预训练目标：Gap Sentences Generation (GSG)，是根据文本摘要任务形式特殊设计的自监督上游任务。PEGASUS有两个不同的版本（base和large），其模型参数分别为：


|  参数 | base（238M） | large（523M） |
|  :---: | :--------: | :--------: |
| encoder layers    |    12 | 16|
| encoder_attention_heads | 12 | 16|
| encoder_ffn_dim | 3072 |4096 |
| decoder layers | 12 | 16|
| decoder_attention_heads | 12 | 16|
| decoder_ffn_dim | 3072 |4096 |
| max_encode_length | 512 | 1024|


## **3. 快速开始**

- [预训练PEGASUS模型](./pretrain/)

- [微调PEGASUS模型](./finetune/)
