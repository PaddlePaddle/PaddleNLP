# RLHF PPO

提供了基于强化学习 PPO 算法对 LLM 进行人类偏好对齐的代码及完整使用示例。其中 PPO 代码实现细节参考了 [PKU-Alignment/safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf)（PKU Beaver） 中的 PPO 实现，支持reward normalization、pretraining loss等常用的 PPO 稳定训练策略；示例使用 PKU-Alignment/safe-rlhf 提供的部分数据集和模型。后续将持续完善扩展，支持更好效果、更低成本、更高性能、更大规模的 RLHF 能力。

## 快速开始

### 环境准备

- Python >= 3.10
- PaddlePaddle >= 2.6.0
- PaddleNLP >= 2.6.0

### 数据准备

PPO 训练包括Supervised Fine-Tuning、Reward Model Fine-Tuning、RLHF三个阶段（可见下文训练部分），会涉及到多个数据集。

示例使用 PKU-Alignment/safe-rlhf 提供的 [PKU-Alignment/PKU-SafeRLHF-30K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K) 数据集。

### 训练

PPO 完整的训练过程包括以下 3 个阶段：
1. Supervised Fine-Tuning (SFT)：同[LLM 精调](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#2-%E7%B2%BE%E8%B0%83)，可以直接参考对应内容进行训练并使用其产出模型。注意，当前 RLHF PPO 暂不支持 LoRA，若请将 LoRA 权重合并入。
2. Reward Model Fine-Tuning：训练奖励模型，

```

```

3. RLHF：

### 推理

PPO 训练得到的为标准的，可以直接使用，请参考部分相应内容

## 参考
