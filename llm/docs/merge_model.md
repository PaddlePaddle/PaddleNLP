# 飞桨大模型套件  模型融合文档
## 1.大模型融合介绍
模型融合，也称为模型合并，是一种有效的技术，通过融合多个具有不同能力的独立模型的参数，构建一个通用模型，而无需访问原始训练数据或进行昂贵的计算。与模型融合最相关的概念是集成学习，因为它们都促进了知识的融合与传递。它们之间的主要区别在于，集成学习必须保存所有的单个模型，并在推理阶段融合多个模型的预测（或输出），而模型融合则直接在参数层面进行合并，并且在推理时只有一个最终模型。

模型融合应用的场景：

- **提高模型能力**：模型融合能使模型拥有处理多领域任务的能力，并且可能提升在交叉领域的能力。

- **缓解对齐代价**：在对预训练模型进行微调 sft 后得到微调模型，之后我们通常进行 RLHF。RLHF 被证明有效提升了 LLMs 的用户友好性，但它会引入一个对齐税（alignment tax），即在对齐人类偏好后模型的性能可能有所下降。模型融合可以缓解对齐税。

## 2.大模型融合算法介绍
### 2.1 Linear
[Linear](https://arxiv.org/abs/2203.05482):融合方法通过对多个微调模型的权重进行线性加权平均实现模型的合并。这种方法基于以下假设：微调后的模型通常位于低误差盆地，线性组合权重能够保持其性能和特性。Linear 方法简单高效。
### 2.2 Slerp
SLERP（Spherical Linear Interpolation）是一种用于两个模型权重之间平滑插值的方法。不同于 Linear 直接在欧几里得空间内插值，SLERP 在球面空间进行操作。
### 2.3 TIES
[TIES](https://arxiv.org/abs/2306.01708)（Trimming, Elect Sign & Merge）是一种高效的多任务模型融合方法，专注于解决以下挑战：

冗余参数干扰：剔除对模型性能无影响的冗余参数。

符号冲突问题：不同模型中同一参数的符号可能相反，直接平均可能导致性能下降。

TIES 方法的三步流程：
修整（Trimming）：根据参数重要性，移除冗余的低权重参数。

选取（Elect Sign）：对符号冲突参数，选取幅度更大的方向。
合并（Merge）：仅融合符号一致的参数，避免相互抵消。 TIES 方法提高了融合模型在多任务场景中的性能稳定性和鲁棒性。
### 2.4 DARE
[DARE](https://arxiv.org/abs/2311.03099)（Drop and Rescale）通过识别和删除冗余的 delta 参数（微调参数与预训练参数的差异）实现高效融合。
主要步骤包括：

剪枝：删除比例为 p 的冗余参数，保留对性能影响大的部分。
重新缩放：对剩余参数进行 1/(1−p) 的幅度调整，确保模型的总体特性保持稳定。 DARE 方法尤其适用于参数冗余较高的大规模模型。在处理大型微调模型时，能够显著提高融合效率并减少资源消耗。
### 2.5 DELLA

[DELLA](https://arxiv.org/abs/2406.11617)（Drop and Rescale via Sampling with Magnitude）进一步优化了 DARE 的剪枝策略，采用 MAGPRUNE 算法对参数重要性进行排序，并基于以下步骤完成融合：
随机剪枝：按照参数幅度设置丢弃概率，幅度越小的参数丢弃概率越高。

重新缩放：对剩余参数根据丢弃比例调整幅度，保持嵌入表示的准确性。

融合：结合 TIES 的符号选择机制，最终合并关键任务参数。 DELLA 方法兼顾剪枝的随机性与重要性排序，适合需要在高维任务向量中保留方向性的复杂模型融合场景。

## 3.快速开始
接下来我们将介绍如何使用统一脚本进行模型融合。
### 3.1 环境准备

- PaddlePaddle 3.0-beta
- PaddleNLP   3.0.0b2
- PaddleSlim develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm/tools
    # 到达运行目录
```
### 3.2 模型融合

```
python merge_weight.py \
    --device cpu \
    --tensor_type np \
    --n_process 2 \
    --merge_method linear \
    --model_path_list ../checkpoints/model1 ../checkpoints/model \
    --output_path ../checkpoints/model_merge

```
<summary>&emsp; 脚本参数介绍</summary><div>

- `device`: 运行环境，默认为 `"cpu"`。
- `tensor_type`: 模型融合过程使用的张量类型，支持 `"np"`（仅支持 CPU）或 `"pd"`（支持 CPU/GPU），默认为 `"np"`。
- `n_process`: 模型融合的并行进程数，默认为 `1`。
- `merge_method`: 模型融合策略，支持`"linear"`, `"ties"`, `"slerp"`, `"della_linear"`, `"della"`, `"dare_linear"`, `"dare_ties"`，默认为 `"linear"`。
- `model_path_list`: 融合模型的路径或名称列表，需至少包含两个路径。
- `output_path`: 融合模型保存的目录路径。
</div>

## 4.Mergekit 参数介绍
<summary>&emsp; 通用参数（Common Parameters）</summary><div>

- `device`: 用于模型融合的设备类型，支持 `"cpu"`、`"gpu"` 或 `"low_gpu_mem"`，默认为 `"cpu"`。
- `tensor_type`: 模型融合过程使用的张量类型，支持 `"np"`（仅支持 CPU）或 `"pd"`（支持 CPU/GPU），默认为 `"np"`。
- `n_process`: 模型融合的并行进程数，默认为 `1`。
- `merge_preifx`: 模型文件的前缀名称，例如 `"model"` 或 `"master_weights"`，默认为 `"model"`。
- `merge_method`: 模型融合策略，支持`"linear"`, `"ties"`, `"slerp"`, `"della_linear"`, `"della"`, `"dare_linear"`, `"dare_ties"`，默认为 `"linear"`。
- `merge_type`: 模型融合过程的类型，支持`"linear"`, `"ties"`, `"slerp"`,默认为 `"linear"`。
- `sparsify_type`: 稀疏化处理的类型，支持`"trim"`, `"magprune"`, `"dare"`,默认为 `None`。

</div>

<summary>&emsp; 模型参数（Model Parameters）</summary><div>

- `model_path_list`: 融合模型的路径或名称列表，需至少包含两个路径。默认为 `None`。
- `model_path_str`: 融合模型名称或路径字符串,以逗号分隔，默认为 `None`。
- `base_model_path`: 基础模型的路径或名称，默认为 `None`。
- `output_path`: 融合模型保存的目录路径，默认为 `None`。

</div>

<summary>&emsp; 融合参数（Merge Parameters）</summary><div>

- `weight_list`: 融合过程中每个模型的相对（或绝对，如果 normalize=False）权重列表，默认为 `None`。如果未设置，将自动均匀分配权重。
- `normalize`: 是否对权重进行归一化处理，默认为 `False`。
- `slerp_alpha`: Slerp 插值参数，用于 Slerp 方法，默认为 `0.5`。
- `slerp_normalize_eps`: Slerp 归一化过程中的 epsilon 值，默认为 `1e-8`。
- `slerp_dot_threshold`: Slerp 点积阈值。如果点积值超过该阈值，则使用线性插值，默认为 `0.9995`。
- `ties_elect_type`: ties mask 的处理类型，支持 `"sum"` 或 `"count"`，默认为 `"sum"`。



</div>

<summary>&emsp; 稀疏化参数（Sparsify Parameters）</summary><div>

- `rescale`: 稀疏化后是否重新缩放权重，默认为 `True`。
- `reserve_p`: 稀疏化模型时保留的随机概率，默认为 `0.7`。
- `epsilon`: 稀疏化方法`”magprune“`中使用的 epsilon 值，默认为 `0.14`。

</div>
