# XLNet

## 模型简介

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 是一款无监督的自回归预训练语言模型。 有别于传统的单向自回归模型，XLNet 通过最大化输入序列所有排列的期望来进行语言建模，这使得它可以同时关注到上下文的信息。 另外，XLNet 在预训练阶段集成了 [Transformer-XL](https://arxiv.org/abs/1901.02860) 模型，Transformer-XL 中的片段循环机制(Segment Recurrent Mechanism)和 相对位置编码(Relative Positional Encoding)机制能够支持 XLNet 接受更长的输入序列，这使得 XLNet 在长文本序列的语言任务上有着优秀的表现。

详细请参考[这里](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/examples/language_model/xlnet).
