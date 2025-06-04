# Apollo 优化器

[论文链接](https://arxiv.org/pdf/2412.05270)

Apollo 是一种为大型语言模型(LLM)预训练和全参数微调设计的内存高效优化器，提供类似SGD的内存占用但达到AdamW级别的性能表现。

## 主要特点

- **内存效率高**：与AdamW相比，Apollo可以降低高达50%的内存使用量，同时保持高性能。
- **两种变体**：
  - **Apollo**：使用基于rank-256辅助空间的近似通道级梯度缩放。
  - **Apollo-Mini**：使用基于rank-1辅助空间的张量级梯度缩放，以获得极致的内存效率。
- **与LoRA集成**：优化了使用LoRA进行LLM微调的场景，特别适合内存受限的环境。

## 实现原理

Apollo整合了两种主要思想来实现LLM训练的内存效率：

1. **低秩近似**：基于GaLore的概念，实现参数高效更新。
2. **优化器状态冗余减少**：扩展Adam-mini的概念，消除不必要的状态信息。

## 使用方法

```python
from paddlenlp.utils import ApolloAdamW, ApolloMiniAdamW

# Apollo优化器（标准版本）
optimizer = ApolloAdamW(
    learning_rate=5e-5,
    parameters=model.parameters(),
    weight_decay=0.01,
    rank=256,       # 更高的秩 = 更好的性能但更多的内存
    scale_type='channel',
    scale=1)

# Apollo-Mini优化器（极致内存效率）
optimizer = ApolloMiniAdamW(
    learning_rate=5e-5,
    parameters=model.parameters(),
    weight_decay=0.01,
    scale=128)     # 对更大的模型，更高的scale可能提高性能
```

## 主要参数

- **rank**：辅助子空间的秩（Apollo使用256，Apollo-Mini使用1）。
- **scale_type**：
  - 'channel'：通道级梯度缩放（Apollo）
  - 'tensor'：张量级梯度缩放（Apollo-Mini）
- **scale**：补偿近似误差的缩放因子（Apollo使用1，Apollo-Mini使用128）。
- **scale_front**：是否在缩放前应用Norm-Growth Limiter（默认为False）。

## 内存对比

当与Llama3-8B一起使用时，与AdamW相比，Apollo可以减少超过50%的优化器状态内存使用，便于在有限硬件上训练或微调更大的模型。
