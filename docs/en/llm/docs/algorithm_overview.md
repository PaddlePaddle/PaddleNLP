# Introduction to Common Algorithms in PaddlePaddle Large Models

## 1. Training Acceleration Strategies

### 1.1 Greedy Zero Padding
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/4b69f8ac-ba3f-4dd6-bdaf-3a7386cb09ad">
</div>

Padding input sequences to the same length is a common approach when handling mini-batch inputs during training. However, this method introduces irrelevant padding tokens, typically resulting in a padding token ratio of about 50%. The Zero Padding strategy proposes concatenating multiple texts into long sequences within single data instances, while using attention_mask to ensure precision alignment, thereby reducing the padding token ratio to 20%. Building upon Zero Padding, PaddlePaddle's self-developed Grouped Greedy Zero Padding strategy further reduces the padding token ratio to approximately 5% through data grouping and greedy combination search.

To enable Zero Padding during fine-tuning/DPO training, simply set `zero_padding` to `True` in the configuration. To activate Grouped Greedy Zero Padding, additionally set `greedy_zero_padding` to `True`.

### 1.2 FlashAttention2

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/8d87bf43-9c9c-4be5-9d1b-520fe735934b">
</div>

[FlashAttention2](https://tridao.me/publications/flash2/flash2.pdf) is a faster and more optimized self-attention computation method. The computational complexity and memory footprint of native self-attention mechanisms are both O(N²), which can lead to memory bottlenecks and low computational efficiency in long-context scenarios. FlashAttention2 introduces tiling to partition long sequences into smaller blocks, reducing memory consumption while leveraging parallel computing to improve computational efficiency.

The standard attention mechanism uses High Bandwidth Memory (HBM) to store, read, and write keys, queries, and values. While HBM offers larger capacity, its access speed is slower compared to the faster but smaller SRAM. In standard attention implementations, the cost of loading and writing keys/queries/values from HBM is significant. FlashAttention2 loads keys, queries and values from HBM to GPU's on-chip SRAM only once, fuses the attention computation steps, and writes back to HBM just once, achieving significant optimization.

```
# FlashAttention2 Usage Example
from paddlenlp.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    dtype=dtype,
    use_flash_attention=True
)
```

To enable FlashAttention2 during fine-tuning/DPO training, simply set `use_flash_attention` to `True`.
To enable FlashAttention2 acceleration during training, simply configure `flash_attention` to `True` in the model configuration.

### 1.3 FlashMask
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/2867b49b-297b-42ea-88d5-1bbfc29f61d6">
</div>

FlashMask is an attention mechanism optimization algorithm independently developed by PaddlePaddle. Based on FlashAttention2, it introduces column-wise sparse mask representation to support various complex masks while maintaining accuracy. This reduces the complexity of attention mask representation from O(N<sup>2</sup>) to O(N), lowering training memory consumption. During block computation, it dynamically skips computation, applies masks, or omits masks based on mask conditions, thereby improving computational efficiency and enhancing training throughput for long contexts.

To enable FlashMask acceleration during fine-tuning/DPO training, simply add `flashmask=True` to the configuration while setting `use_flash_attention=True`.

## 2. Improved Fine-Tuning Strategies

### 2.1 NEFT
[NEFT](https://arxiv.org/abs/2310.05914) proposes adding controlled noise to model embeddings during training to improve fine-tuning performance. To enable NEFT training, simply configure `neftune=True` and set the noise intensity parameter `neftune_noise_alpha` in the configuration.

### 2.2 LoRA+

[LoRA+](https://github.com/user-attachments/assets/f367bf06-ad2f-4d41-8b5a-6508f90b46fa) proposes using different learning rates for matrices A and B during LoRA training to accelerate convergence. Configure the learning rate ratio between B and A using the `lora_plus_scale` parameter.

### 2.3 rsLoRA

[rsLoRA](https://arxiv.org/pdf/2312.03732) suggests that for improved LoRA performance, the LoRA rank should increase while maintaining the scaling-ratio relationship according to:
$$
\gamma_r \in \Theta_r\left(\frac{1}{\sqrt{r}}\right)
$$
Enable this by setting `rslora=True` during LoRA training.
### 2.4 PiSSA
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/d75ec708-2c40-4a12-8a01-17b9f276ba20">
</div>

[PiSSA](https://arxiv.org/abs/2404.02948) is an optimization work based on LoRA, sharing the same architecture. The A and B matrices are initialized through singular value decomposition (SVD) of the pre-trained model weights, with the residual matrix initializing the backbone weights. During training, the backbone weights remain frozen while updating matrices A and B (i.e., the principal components), achieving faster convergence and higher performance. Utilizing fast SVD technology, PiSSA's initialization requires only a few seconds, adding almost no cost when switching from LoRA to PiSSA.

To enable PiSSA during fine-tuning, simply set `pissa=True` in the LoRA training configuration.

### 2.5 VeRA
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/343acc9e-46a0-4a39-91cd-71e433ee59b2">
</div>

[VeRA](https://arxiv.org/abs/2310.11454) is a parameter-efficient fine-tuning technique similar to LoRA, but requiring fewer additional parameters while promising comparable or better performance. This makes VeRA particularly useful when parameter budgets are extremely limited, such as when scaling to very large models. The reduction in trainable parameters is achieved by sharing the same low-rank matrix across all layers, with only two additional vectors trained per layer.

To enable VeRA during fine-tuning, simply set `vera=True` and configure the corresponding `vera_rank` parameter.
