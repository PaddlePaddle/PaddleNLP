# 飞桨大模型常见算法介绍

## 1.训练加速策略

### 1.1 Greedy Zero Padding
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/4b69f8ac-ba3f-4dd6-bdaf-3a7386cb09ad">
</div>
在训练过程中，将输入序列填充到相同长度是一种常见的处理小批量输入的方法。然而，这种方式由于填充了无关的填充标记，空洞率（填充标记比例）通常达到50%左右。Zero Padding 策略提出在单条数据中拼接多个文本为长文本，使用 attention_mask 保证精度对齐，将空洞率降低为20%。在 Zero Padding 策略基础上中，飞桨自研分组贪心 Zero Padding 策略，通过将输入数据进行分组保证数据分布，组内贪心策略搜索组合可能，实现了高效的 Padding 策略，将空洞率降低至5%左右。


精调/DPO 训练只需要添加一个`zero_padding`为`True`的配置，即可开启 Zero Padding 训练。在开启 Zero Padding 的基础上，添加一个`greedy_zero_padding`为`True`即可开启分组贪心 Zero Padding 训练。

### 1.2 FlashAttention2

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/8d87bf43-9c9c-4be5-9d1b-520fe735934b">
</div>

[FlashAttention2](https://tridao.me/publications/flash2/flash2.pdf) 是一种更快、更优化的自注意力机制计算方法。原生的自注意力极致计算复杂度和内存都是 O(N<sup>2</sup>)，其中 N 是序列的长度，在长上下文场景容易导致内存不足，计算效率低。FlashAttention2提出分块处理（Tiling）将长序列划分为较小块，降低显存占用，利用并行计算提高计算效率。

标准的注意力机制使用高带宽内存（HBM）来存储、读取和写入键、查询和值。虽然 HBM 具有较大的内存，但处理速度较慢，而 SRAM 具有较小的内存，但操作速度更快。在标准的注意力实现中，从 HBM 加载和写入键、查询和值的成本较高。它将键、查询和值从 HBM 加载到 GPU 的片上 SRAM 中，执行一次注意力机制步骤，然后写回 HBM，并为每个注意力步骤重复这一过程。相反，Flash Attention 仅加载一次键、查询和值，将注意力机制的操作融合在一起，然后写回。


```
# FlashAttention2 使用样例
from paddlenlp.transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    dtype=dtype,
    use_flash_attention=True
)
```

精调/DPO 训练只需要添加一个`use_flash_attention`为`True`的配置，即可开启 FlashAttention2加速训练。

### 1.3 FlashMask
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/2867b49b-297b-42ea-88d5-1bbfc29f61d6">
</div>
FlashMask 是飞桨自研的注意力机制优化算法，在 FlashAttention2的基础上，在不损失精度的前提下，提出按列稀疏掩码表示支持多种复杂掩码表示，将注意力掩码表示复杂度由 O(N<sup>2</sup>) 降到 O(N)，降低训练显存。同时在进行块计算的过程中根据掩码情况分为跳过计算、应用掩码、不应用掩码三种情况，提高计算效率进而提高模型训练长上下文的吞吐能力。


精调/DPO 训练只需要添加一个`flashmask`为`True`的配置，同时配置`use_flash_attention`为`True`，即可开启 FlashMask 加速训练。

## 2.改进精调策略

### 2.1 NEFT
[NEFT](https://arxiv.org/abs/2310.05914)提出在模型训练过程中，在模型的 embedding 之后加入一定噪声，有利于提高模型精调性能。精调训练只需要添加一个`neftune`为`True`的配置，同时配置相应的 NEFT alpha 参数`neftune_noise_alpha`，即可使用 NEFT 训练。

### 2.2 LoRA+

[LoRA+](https://github.com/user-attachments/assets/f367bf06-ad2f-4d41-8b5a-6508f90b46fa)提出 LoRA 训练过程中低秩矩阵 A 和 B 使用不同的学习率，有利于提高模型的 LoRA 训练收敛速度。精调训练只需要设置`lora_plus_scale`调节 B 与 A 的学习率比例。

### 2.3 rsLoRA

[rsLoRA](https://arxiv.org/pdf/2312.03732)提出 LoRA 训练提升模型效果应该让 LoRA rank 增大的同时，使 LoRA 中 scaling 与 LoRA rank 关系应该满足下面的公式：
$$
\gamma_r \in \Theta_r\left(\frac{1}{\sqrt{r}}\right)
$$
调训练只需要在 LoRA 训练过程中设置`rslora`为 True，即可开启。

### 2.4 PiSSA
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/d75ec708-2c40-4a12-8a01-17b9f276ba20">
</div>

[PiSSA](https://arxiv.org/abs/2404.02948) 是 LoRA 基础上的优化工作，具有与 LoRA 相同的架构，A 和 B 矩阵通过预训练模型权重奇异值分解（SVD）进行初始化，将残差矩阵对主干权重进行初始化。训练过程中对主干权重进行冻结，更新 A 和 B 矩阵，也即主成分部分从而实现了更快的收敛速度和更高的性能。利用快速 SVD 技术，PiSSA 的初始化仅需几秒钟，几乎不会增加从 LoRA 转向 PiSSA 的成本。

精调训练只需要在 LoRA 训练过程中设置`pissa`为 True，即可开启。

### 2.5 VeRA
<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/343acc9e-46a0-4a39-91cd-71e433ee59b2">
</div>

[VeRA](https://arxiv.org/abs/2310.11454)是一种类似于 LoRA 的参数高效微调技术，但需要更少的额外参数，同时承诺提供相似甚至更好的性能。因此，当参数预算非常有限时，例如在扩展到非常大的模型时，VeRA 尤为有用。可训练参数数量的减少是通过在所有层之间共享相同的低秩矩阵，并且每层仅训练两个额外的向量来实现的。
精调训练只需设置`vera`为 True，设置相应的`vera_rank`即可开启。
