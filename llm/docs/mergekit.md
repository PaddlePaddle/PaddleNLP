# 飞桨大模型套件  模型融合文档
## 1.大模型融合介绍
模型融合，也称为模型合并，是一种有效的技术，通过融合多个具有不同能力的独立模型的参数，构建一个通用模型，而无需访问原始训练数据或进行昂贵的计算。与模型融合最相关的概念是集成学习，因为它们都促进了知识的融合与传递。它们之间的主要区别在于，集成学习必须保存所有的单个模型，并在推理阶段融合多个模型的预测（或输出），而模型融合则直接在参数层面进行合并，并且在推理时只有一个最终模型。

模型融合应用的场景：

- **提高模型能力**：模型融合能使模型拥有处理多领域任务的能力，并且可能提升在交叉领域的能力。

- **缓解对齐代价**：在对预训练模型进行微调 sft 后得到微调模型，之后我们通常进行 RLHF。RLHF 被证明有效提升了 LLMs 的用户友好性，但它会引入一个对齐税（alignment tax），即在对齐人类偏好后模型的性能可能有所下降。模型融合可以缓解对齐税。

## 2.大模型融合算法介绍
### 2.1 快速开始
接下来我们将介绍如何使用统一脚本进行模型融合。
#### 2.1.1 环境准备

- PaddlePaddle 3.0-beta
- PaddleNLP   develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm/tools
    # 到达运行目录
```
#### 2.1.2 模型融合

```
python mergekit.py \
    --device cpu \
    --tensor_type np \
    --n_process 2 \
    --merge_method linear \
    --model_path_list ../checkpoints/model1 ../checkpoints/model \
    --output_path ../checkpoints/model_merge

```

### 2.2 权重融合方法
| merge_method   | 权重稀疏      | 权重融合   | 支持融合模型数 |
|----------------|---------------|------------|----------------|
| linear         | /             | linear     | >=2            |
| slerp          | /             | slerp      | =2             |
| ties           | trim          | ties       | >=2            |
| della          | magprune      | ties       | >=2            |
| della_linear   | magprune      | linear     | >=2            |
| dare_linear    | dare          | linear     | >=2            |
| dare_ties      | dare          | ties       | >=2            |

#### 权重融合方法介绍：
- **linear**: 权重线性融合。可选超参 `weight_list`、`normalize`。
- **slerp**: 球面线性插值融合，仅支持两个模型融合。可选超参 `slerp_alpha`。
- **ties**: [TIES](https://arxiv.org/abs/2306.01708)论文中提出权重融合方式，应用符号一致性算法来减少模型融合干扰。可选超参 `weight_list`、`normalize`。

#### 权重稀疏方法介绍：
- **trim**: [TIES](https://arxiv.org/abs/2306.01708)论文中提出稀疏方式，根据绝对值由大到小顺序，保留设定比例权重数值，将其余小数值权重设为0。可选超参 `reserve_p`、`rescale`。
- **dare**: [DARE](https://arxiv.org/abs/2311.03099)论文中提出稀疏方式，根据设定概率，随机选择保留原始权重或设为0。可选超参 `reserve_p`、`rescale`。
- **magprune**:[DELLA](https://arxiv.org/abs/2406.11617)论文中提出稀疏方式，根据权重绝对值大小给定不同保留概率，随机选择保留原始权重或设为0。可选超参 `reserve_p`、`rescale`、`epsilon`。
## 3.Mergekit 参数介绍
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
