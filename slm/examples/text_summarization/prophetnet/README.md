# Prophetnet

## 模型简介

ProphetNet（先知网络）是一种新型的 seq2seq 预训练模型。在训练时，Prophetnet 每一时刻将会学习同时预测未来的 N 个字符，这种自监督学习目标可以使得模型考虑未来更远的字符，防止模型对强局部相关（strong
local correlation）过拟合。

本项目是 Prophetnet 在 PaddlePaddle 2.4 上开源实现的文本摘要的例子，包含了在 CNN/DailyMail 数据集，Gigaword 数据集上微调和生成的代码。

具体实现[参考](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/text_summarization/pointer_summarizer)
