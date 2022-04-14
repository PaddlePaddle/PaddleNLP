# ERNIE-Health with PaddleNLP

[ERNIE-Health](https://arxiv.org/pdf/2110.07244.pdf) 预训练模型与 [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) 的结构相似，包括 Generator 和 Discriminator 两部分模型网络，各自包含1个 [ERNIE](https://arxiv.org/pdf/1904.09223.pdf) 模型。Generator 的预训练使用Masked Language Model(MLM)任务，主要作用是给 Discriminator 提供训练语料。而 Discriminator 的预训练为多任务：

- token 级别，使用 Replaced Token Detection(RTD)、Multi-Token Selection (MTS) 任务（主要改进点）。
- sequence 级别，使用 Contrastive Sequence Prediction（CSP）任务（主要改进点）。

预训练结束后，使用Discriminator作为精调模型，后续的Fine-tuning不再使用Generator。

![Overview_of_EHealth]()

图片来源：来自[ERNIE-Health论文](https://arxiv.org/pdf/2110.07244.pdf)

ERNIE-Health 取得了中文医疗自然语言处理榜单 CBLUE 的冠军，平均得分达到 77.822。

本项目是 ERNIE-Health 在 Paddle 2.3上的开源实现。
