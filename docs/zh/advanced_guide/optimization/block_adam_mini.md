# Block Adam-mini 优化器

[论文链接](https://arxiv.org/pdf/2406.16793)

Block Adam-mini 是一种参数高效微调优化器，通过对优化器状态进行结构化稀疏处理，显著降低了内存消耗。该实现将参数划分为固定大小的块，每个块仅维护一个优化器状态，与 AdamW 相比，可以减少约 50% 的内存使用量。

## 主要特点

- **内存效率高**：通过块级别的二阶矩聚合，与 AdamW 相比，可以减少约 50% 的优化器状态内存使用。
- **兼容性强**：与现有的 PaddleNLP 优化器框架完全兼容，易于集成到现有训练流程中。
- **性能保持**：虽然降低了内存使用，但仍能保持与 AdamW 相当的训练性能和收敛速度。
- **与 LoRA 等 PEFT 方法兼容**：特别适合在内存受限环境下与 LoRA 等参数高效微调方法结合使用。

## 实现原理

Block Adam-mini 的核心思想是利用优化器状态的冗余性，将参数划分为固定大小的块（默认 32 个元素一块），并对每个块内的梯度平方使用相同的二阶矩值。具体实现包括：

1. **一阶矩（moment1）**：保持与标准 Adam/AdamW 相同，为每个参数保存一个一阶矩。
2. **二阶矩（moment2）**：将参数划分为固定大小的块，每个块共享一个二阶矩值，显著减少内存使用。
3. **块内梯度统计**：对每个块内的梯度平方取平均值，作为该块所有参数的共同二阶矩。

## 使用方法

```python
from paddlenlp.utils import BlockAdamMini

# 创建 Block Adam-mini 优化器
optimizer = BlockAdamMini(
    learning_rate=5e-5,
    parameters=model.parameters(),
    weight_decay=0.01,
    block_size=32)  # 块大小，默认为 32

# 在训练循环中使用
for batch in data_loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
```

## 主要参数

- **block_size**：每个块包含的参数数量，默认为 32。较小的块大小可能提高性能但增加内存使用，较大的块大小可能降低性能但进一步减少内存使用。

## 内存对比

在 Llama3-8B 模型上，Block Adam-mini 相比 AdamW 可降低约 50% 的优化器状态内存使用，使得在有限内存的环境中微调大型模型变得更加可行。

## 与 LoRA 结合使用

Block Adam-mini 特别适合与 LoRA 等参数高效微调方法结合使用。在使用 LoRA 进行微调时，虽然已经大大减少了需要更新的参数数量，但优化器状态仍可能占用大量内存。结合 Block Adam-mini 可以进一步减少内存占用：

```python
from paddlenlp.utils import BlockAdamMini
from paddlenlp.peft import LoRAModel, LoRAConfig

# 配置 LoRA
lora_config = LoRAConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=8,
    lora_alpha=16,
    dropout=0.1
)

# 初始化 LoRA 模型
model = LoRAModel(model, lora_config)

# 创建 Block Adam-mini 优化器
optimizer = BlockAdamMini(
    learning_rate=5e-5,
    parameters=model.parameters(),
    weight_decay=0.01,
    block_size=32)

# 训练过程与常规相同
```
