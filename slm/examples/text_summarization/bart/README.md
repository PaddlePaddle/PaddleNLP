# BART

## 模型简介

BART 是一种 Seq2Seq 结构的降噪自编码器，通过增加噪声来破环文本然后重建原文本来训练模型。它使用一个标准的 Transformer 结构，可以被看作泛化的 BERT（由于是双向编码器），GPT（由于是从左到右解码器），和一些其他的预训练模型结构。

本项目是 BART 在 PaddlePaddle 2.2上开源实现的文本摘要的例子，包含了在[CNN/DailyMail](https://arxiv.org/pdf/1704.04368.pdf)数据集上微调和生成的代码。

使用请[参考](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/text_summarization/bart)
