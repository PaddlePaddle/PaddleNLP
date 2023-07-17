# ChatGLM2-6B

## 介绍

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B 引入了[FlashAttention](https://github.com/HazyResearch/flash-attention)和[Multi-Query Attention]等新特性。更详细的模型介绍见[ChatGLM2-6B GitHub](https://github.com/THUDM/ChatGLM2-6B)

## 协议

ChatGLM2-6B 模型的权重的使用则需要遵循[License](../../../paddlenlp/transformers/chatglm_v2/LICENSE)。

## 训练

WIP

## 推理

```
# 单卡
python predict_generation.py --model_name_or_path THUDM/chatglm2-6b

# 多卡
python -m paddle.distributed.launch --gpus "0,1" predict_generation.py --model_name_or_path THUDM/chatglm2-6b
```
