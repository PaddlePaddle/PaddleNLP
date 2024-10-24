# Falcon

## 介绍

Falcon 是由[TII](https://www.tii.ae/)构建的 Causal decoder-only 模型，基于含有 1,500B 个 tokens 的[RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)数据集训练得来。
Falcon 引入了[FlashAttention](https://github.com/HazyResearch/flash-attention)和[Multi-Query Attention]等新特性。更详细的模型介绍见[论文](https://arxiv.org/abs/2306.01116)

## 推理

```
python predict_generation.py \
    --model_name_or_path tiiuae/falcon-7b
```
