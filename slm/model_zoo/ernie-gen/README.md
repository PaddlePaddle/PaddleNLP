# ERNIE-Gen: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation

## 1. 简介

ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入**span-by-span 生成任务**，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过**填充式生成机制**和**噪声感知机制**来缓解曝光偏差问题。此外, ERNIE-GEN 采样**多片段-多粒度目标文本采样策略**, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。

![multi-flow-attention](https://github.com/PaddlePaddle/ERNIE/raw/repro/ernie-gen/.meta/multi-flow-attention.png)

详细参考这里: https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.8/model_zoo/ernie-gen
