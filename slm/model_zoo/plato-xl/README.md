# PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

PLATO-XL 是业界首个开源的百亿超大规模开放域对话预训练模型，其使用了参数高效(encoder-decoder 共享参数)的 UnifiedTransformer（prefix LM）模型架构，将模型参数量提升到了11B 量级，经过了十亿级样本对话数据的预训练，并引入 role embedding 区分多方对话中的对话角色提升预训练效果，最终模型闲聊测试效果超过了众多代表性的对话模型。可以直接使用 PLATO-XL 构建高质量的开放领域对话机器人。

PaddleNLP 内置了 PLATO-XL 英文预训练模型以供使用。由于 PLATO-XL 模型规模较大，这使得其在预测时生成对话回复的时间较长，并且 11B 的参数量也可能超出部分型号 GPU 显存容量，这是大模型推理与落地存在的普遍和关键问题。PaddleNLP FastGeneration 为 PLATO-XL 提供了 GPU 上的高性能生成加速能力，并且支持模型并行（张量并行）推理允许通过多张小显存容量的 GPU 使用百亿大模型，相比单卡代码中也只增加了`enable_ft_para()`一行，此外模型并行能进一步提升预测速度。

本项目提供了 PLATO-XL 英文模型使用 PaddleNLP FastGeneration 进行高性能预测的使用示例。PLATO-XL 的训练及更多内容请参考 [PaddlePaddle/Knover](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-XL)。

详细请参考: https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/plato-xl
