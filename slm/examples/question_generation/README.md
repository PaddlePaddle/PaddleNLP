# 问题生成

Question Generation（QG），即问题生成，指的是给定一段上下文和答案，自动生成一个流畅且符合上下文主题的问句。问题生成技术在教育、咨询、搜索、问答等多个领域均有着巨大的应用价值。

PaddleNLP 提供英文和中文问题生成任务示例，分别基于英文预训练语言模型[t5](./t5)和中文预训练语言模型[unimo-text](./unimo-text)。


## 英文

[t5](./t5) 展示了如何使用英文预训练模型 T5完成问题生成任务，支持模型微调预测评估，并提供相关预训练模型。

## 中文

[unimo-text](./unimo-text) 展示了如何使用中文预训练模型 UNIMO-Text 完成问题生成任务，提供数据准备、训练、预测、推理部署全流程定制化训练，并提供相关预训练模型。

# 参考文献

1. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P.J., 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140), pp.1-67.

2. Li, Wei, et al. "Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning." arXiv preprint arXiv:2012.15409 (2020).
