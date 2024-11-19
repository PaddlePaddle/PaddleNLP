# ELECTRA with PaddleNLP

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) 在[BERT](https://arxiv.org/abs/1810.04805)的基础上对其预训练过程进行了改进：预训练由两部分模型网络组成，称为 Generator 和 Discriminator，各自包含1个 BERT 模型。Generator 的预训练使用和 BERT 一样的 Masked Language Model(MLM)任务，但 Discriminator 的预训练使用 Replaced Token Detection(RTD)任务（主要改进点）。预训练完成后，使用 Discriminator 作为精调模型，后续的 Fine-tuning 不再使用 Generator。

图片来源：来自[electra 论文](https://openreview.net/pdf?id=r1xMH1BtvB)

详细请参考[这里](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/electra).
