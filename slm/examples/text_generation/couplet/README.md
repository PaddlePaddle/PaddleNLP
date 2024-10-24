# 使用 Seq2Seq 模型完成自动对联


## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成 vector，再用解码器将该 vector 解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含 Seq2Seq 的一个经典样例：自动对联生成，带 attention 机制的文本生成模型。

上联：未出南阳天下论        下联：先登北斗汉中书

上联：朱联妙语千秋颂        下联：赤胆忠心万代传

上联：月半举杯圆月下        下联：花间对酒醉花间

上联：挥笔如剑倚麓山豪气干云揽月去       下联：落笔似龙飞沧海龙吟破浪乘风来

## 参考的开源数据集

我们的数据集采用了开源对联数据集[couplet-clean-dataset](https://github.com/v-zich/couplet-clean-dataset)，地址：https://github.com/v-zich/couplet-clean-dataset ，该数据集过滤了[couplet-dataset](https://github.com/wb14123/couplet-dataset)（地址：https://github.com/wb14123/couplet-dataset ）中的低俗、敏感内容。

使用请[参考](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/text_generation/couplet)
