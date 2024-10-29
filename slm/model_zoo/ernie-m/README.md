# ERNIE-M

## 模型介绍

[ERNIE-M](https://arxiv.org/abs/2012.15674) 是百度提出的一种多语言语言模型。原文提出了一种新的训练方法，让模型能够将多种语言的表示与单语语料库对齐，以克服平行语料库大小对模型性能的限制。原文的主要想法是将回译机制整合到预训练的流程中，在单语语料库上生成伪平行句对，以便学习不同语言之间的语义对齐，从而增强跨语言模型的语义建模。实验结果表明，ERNIE-M 优于现有的跨语言模型，并在各种跨语言下游任务中提供了最新的 SOTA 结果。
原文提出两种方法建模各种语言间的对齐关系:

- **Cross-Attention Masked Language Modeling(CAMLM)**: 该算法在少量双语语料上捕捉语言间的对齐信息。其需要在不利用源句子上下文的情况下，通过目标句子还原被掩盖的词语，使模型初步建模了语言间的对齐关系。
- **Back-Translation masked language modeling(BTMLM)**: 该方法基于回译机制从单语语料中学习语言间的对齐关系。通过 CAMLM 生成伪平行语料，然后让模型学习生成的伪平行句子，使模型可以利用单语语料更好地建模语义对齐关系。


![framework](https://user-images.githubusercontent.com/40912707/201308423-bf4f0100-3ada-4bae-89d5-b07ffec1e2c0.png)

详细请参考: https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/ernie-m
