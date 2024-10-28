<!-- vscode-markdown-toc -->
* [1. 背景](#)
* [2. FlashMask: 列式稀疏掩码表示](#FlashMask:)
* [3. FlashMask: 扩展 FlashAttention 支持复杂掩码](#FlashMask:FlashAttention)
    * [3.1 预处理阶段](#-1)
    * [3.2 实时块跳过计算阶段](#-1)
* [4. FlashMask: 速度与存储的双重提升](#FlashMask:-1)
    * [4.1 端到端训练吞吐量提升](#-1)
    * [4.2 端到端训练收敛验证](#-1)
* [4.3 稀疏度与 Kernel 计算时延的线性关系](#Kernel)
* [4.4 Kernel 性能对比](#Kernel-1)
* [5. Quick Start](#QuickStart)
* [5.1 Requirements](#Requirements)
* [5.2 SFT & LoRA](#SFTLoRA)
    * [5.2.1 Data Preparation](#DataPreparation)
    * [5.2.2 SFT](#SFT)
    * [5.2.3 LoRA](#LoRA)
* [5.3 DPO & RM](#DPORM)
    * [5.3.1 Data Preparation](#DataPreparation-1)
    * [5.3.2 DPO](#DPO)
    * [5.3.3 RM](#RM)

<!-- vscode-markdown-toc-config
    numbering=false
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# FlashMask

FlashMask 是 FlashAttention 的扩展，它利用了一种新颖的按列注意力掩码表示法。这种方法允许在不牺牲计算精度的情况下，更有效地处理更广泛类型的掩码。FlashMask 实现了线性内存复杂度，并且支持内核优化，减少不必要的计算，从而实现显著的计算加速和增强的训练效率。

## <a name=''></a>1. 背景

在 Transformer 类大模型训练任务中，注意力掩码（Attention Mask）一方面带来了大量的冗余计算，另一方面因其 $O(N^2)$ 巨大的存储占用导致难以实现长序列场景的高效训练（其中$N$为序列长度）。虽然业界已有 FlashAttention 等针对特定注意力掩码的计算加速方法，但其支持的注意力掩码模式有限，难以满足大模型训练任务对灵活注意力掩码的需求。为了解决上述问题，飞桨独创 FlashMask 技术，提出了列式稀疏的注意力掩码表示方法，支持灵活多样的注意力掩码模式，使得存储复杂度从 $O(N^2)$ 降低至 $O(N)$，并在此基础上实现了高效的算子 Kernel，极致加速大模型训练效率，尤其是长序列场景下的训练效率。

* arXiv 论文地址 https://arxiv.org/pdf/2410.01359
* PaddlePaddle 官方文档地址 https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/functional/flashmask_attention_en.html
* PaddleNLP 开源地址 https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/docs/flashmask.md


## <a name='FlashMask:'></a>2. FlashMask: 列式稀疏掩码表示

FlashMask 的核心发现是，在大模型常见的注意力掩码模式中，Query-Key token 的掩码模式具有一定的连续性。具体地，对于每一个 Key token 而言，不进行有效 Attention 计算的 Query token 是相邻的，即在图1中二维掩码矩阵中，Query token 作用在每一列的 Key token 的灰色部分在列方向上是连续分布的。基于这一洞察，FlashMask 巧妙地将二维的稠密掩码矩阵转换为一维的行索引区间这一更为紧凑的表示形式，显著降低存储需求。我们可以公式化表示为：

$M_{j} = [start_j, end_j), \quad \forall j \in \{1, \ldots, N\}$

其中 $N$ 为 Key 的序列长度，$M_j$ 为二维的稠密掩码矩阵的第 $j$ 列，$[start_j, end_j)$ 为连续的行索引区间，表示 $start_j$ 到 $end_{j} - 1$ 的连续 Query token 是被 mask 掉，置为无效 Attention 计算。


为了高效处理因果和双向注意力场景中的复杂掩码模式，FlashMask 提出了一种新颖的列式稀疏表示方法。以对角线为区分，它使用四个一维向量来表示掩码：
* 下三角起始行索引（Lower Triangular Start，简称 LTS）
* 下三角结束行索引（Lower Triangular End，简称 LTE）
* 上三角起始行索引（Upper Triangular Start，简称 UTS）
* 上三角结束行索引（Upper Triangular End，简称 UTE）

其中下三角被 mask 掉的行索引区间使用 $[𝐿𝑇𝑆, 𝐿𝑇𝐸)$ 表示，上三角被 mask 掉的行索引区间使用 $[𝑈𝑇𝑆, 𝑈𝑇𝐸)$ 表示。

<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/989cc61e-174b-489d-ba7a-d1e6d172ff91">
    <div align="center">
        <font size ="2">
        图1：较为复杂的二维稠密因果注意力的掩码矩阵示意图
        </font>
    </div>
</div>

如图1所示，我们展示了16个 Query token 和16个 Key token 做 Attention 计算时较为复杂的二维稠密因果注意力的掩码矩阵，灰色单元格是 mask 区域。

可以通过 $[LTS,LTE)$ 两个向量进行表达，如下所示：
| col_idx | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
|---------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| $LTS$   | 13 | 5  | 5  | 5  | 6  | 6  | 9  | 9  | 9  | 12 | 12 | 12 | 16 | 16 | 16 | 16 |
| $LTE$   | 15 | 14 | 14 | 15 | 12 | 12 | 11 | 11 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

以第1列为例，开始 mask 的行为13，结束 mask 的行为15（开区间），表示位置为13和14的 Query token 不与位置为0的 Key token 做有效 Attention 计算。

<div align="center">
    <div align="center">
        <img width="400" alt="llm" src="https://github.com/user-attachments/assets/55a023ec-f1d9-46c9-aa3a-9fed14ed89f0">
        <img width="300" alt="llm" src="https://github.com/user-attachments/assets/67c8076a-da8e-415b-988a-6b5f65023464">
    </div>
    <div align="center">
        <font size ="2">
        图2: 常见的注意力掩码类型及使用 FlashMask 的列式稀疏掩码表示方法表示的注意力掩码模式
        </font>
    </div>
</div>

更多的例子参考图2，FlashMask 使用列式稀疏掩码表示方法，表达了图1中所有的注意力掩码模式。其中 $-$ 的空缺表示在不同的场景下有不同的默认值，$LTS$ 和 $UTS$ 中的默认值是 0，表示 mask 区域默认从第0行开始，$LTE$和$UTE$中的默认值是 Query 的序列长度，表示 mask 区域默认结束于最后一行。


## <a name='FlashMask:FlashAttention'></a>3. FlashMask: 扩展 FlashAttention 支持复杂掩码

FlashMask 将列式掩码表示方法集成到 FlashAttention-2 算法中，扩展了其对注意力掩码的支持能力。FlashMask 的高性能 Kernel 实现包括两个关键步骤：预处理和实时块跳过计算。

在 FlashAttention 的 Kernel 实现中，得分矩阵（score matrix）的计算是分块（Tile Block）实现的。如图4的简化表示所示，整个得分矩阵计算被分为了 4 x 4 的块，每个块包含 4 个 Query token 和 4 个 Key token 交互的 4 x 4  Attention 计算。FlashMask 的原始输入是 token 级别的逐列表示，通过预处理阶段转化成块级别的表示，用于在实时跳过计算阶段快速实时计算出每个块的类型。

<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/1a244bb1-1b3c-4bc4-8839-5d3e77f02bed">
    <div align="center">
        <font size ="2">
        图3：FlashMask 计算过程示意图
        </font>
    </div>
</div>

### <a name='-1'></a>3.1 预处理阶段
在 FlashMask 的预处理阶段，列式稀疏掩码向量 $LTS$、 $LTE$、 $UTS$、 $UTE$ 首先被加载到高带宽存储（HBM）中，然后根据 FlashAttention 的分块列大小，将列式稀疏掩码向量分块，计算出每个分块中所有列的向量最大值和最小值，生成8个中间向量：

* $LTStart^{min}$, $LTStart^{max}$
* $LTEnd^{min}$, $LTEnd^{max}$
* $UTStart^{min}$, $UTStart^{max}$
* $UTEnd^{min}$, $UTEnd^{max}$

以图4最左边的4个分块为例，分块包含4个列，这4列的 $LTS=[13,5,5,5]$和 $LTE=[15,14,14,15]$，因此 $LTStart^{min}=min(LTS)=5$, $LTStart^{max}=max(LTS)=13$, $LTEnd^{min}=min(LTE)=14$, $LTEnd^{max}=max(LTE)=15$。剩余的计算结果如图5所示：

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/76a5cca9-c268-4bd8-b0f6-d84ba3948b68">
    <div align="center">
        <font size ="2">
        图4：预处理计算的分块最大值/最小值计算
        </font>
    </div>
</div>

### <a name='-1'></a>3.2 实时块跳过计算阶段
在实时计算阶段，FlashMask 利用预处理生成的最小值和最大值向量，对注意力得分矩阵的每个分块进行分类，以提升计算效率。分类依据为以下三种类型：

1. 完全掩码块：若 $BlockRow_{min} \geq Start^{max} \text{ and } BlockRow_{max} \leq End^{min}$ ，则此块的所有元素均被掩码，计算可直接跳过。
2. 部分掩码块：若 $BlockRow_{min} < End^{max} \text{ and } BlockRow_{max} > Start^{min}$ ，则此块的部分元素被掩码，因此需要对该块进行逐元素的掩码计算。
3. 未掩码块：其他情况则归为未掩码块，此类块中的所有元素均未被掩码，可以简化计算过程，不进行额外的掩码操作。
通过这种分类处理，FlashMask 可以显著提升计算效率：完全掩码块的计算被跳过，未掩码块的计算得以简化，仅对部分掩码块执行必要的掩码操作。



图4展示了在因果掩码场景下，使用 $LTS$ 和 $LTE$ 进行 Kernel 计算的完整过程。图中每种分块类型的实时计算公式都已标注，以下是具体例子说明：

* 完全掩码块，例如，图4中 [3, 2] 位置的块，其最小行号为12，大于等于 $LTStart^{max}=12$ ，最大行号为15，小于等于 $LTEnd^{max}=16$ ，因此块中所有元素被掩码，计算可以直接跳过。
* 部分掩码块，例如，图4中 [1, 1] 位置的块，其最小行号为4，小于 $LTEnd^{max}=12$ ，最大行号为7，大于 $LTStart^{min}=6$ ，因此块中部分元素被掩码，需要对该块逐元素进行掩码计算。
* 未掩码块，例如，图4中 [3, 1] 位置的块，其最小行号为12，大于等于 $LTEnd^{max}=12$ ，表明此块中所有元素未被掩码，计算时无需额外的掩码操作，从而减少计算开销。


## <a name='FlashMask:-1'></a>4. FlashMask: 速度与存储的双重提升
FlashMask 充分利用了注意力掩码中的稀疏性，通过跳过完全掩码块的计算，减少了计算开销，同时不改变算法的精度。与使用稠密掩码矩阵的注意力计算保持比特级别的数值等效性，确保了精度无损。更多分析详见 [FlashMask 论文](https://arxiv.org/pdf/2410.01359) 的第4.3节 [3]。

### <a name='-1'></a>4.1 端到端训练吞吐量提升
在 Llama-2 7B、13B、70B 等模型规模下，针对 SFT、LoRA、DPO、RM 四种下游训练场景和不同序列长度的实验表明，FlashMask 在各个模型规模和序列长度下均实现了端到端的加速和存储效率的提升。相比现有的基于稠密掩码矩阵的计算方法，FlashMask 实现了1.65倍至3.22倍的吞吐量提升，并支持更长的序列长度。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/49208427-49b8-4a74-aca4-e7782294071d">
    <div align="center">
        <font size ="2">
        图5：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，3 个 Llama2 模型规模，在不同序列长度下的端到端训练吞吐量
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/9bbe637b-9a04-4df4-a227-36f6eab38bbc">
    <div align="center">
        <font size ="2">
        图5：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，3 个 Llama2 模型规模，不同序列长度下的端到端训练峰值显存消耗
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f0f7880a-c439-4a9f-9232-6f4171090c90">
    <div align="center">
        <font size ="2">
        图5：在 Llama2 7B 模型上 FlashMask 对比 FlexAttention (Causal=True) 的显存消耗，单位(GB)
        </font>
    </div>
</div>

### <a name='-1'></a>4.2 端到端训练收敛验证
在 Llama 3.1 模型上的实验验证了 FlashMask 对收敛精度没有影响。作为一种精确的算法，通过控制计算过程的随机性（如去除 FlashAttention 反向 Query 梯度计算的 atomicAdd 操作），FlashMask 可以与使用稠密掩码的 FlashAttention 在比特级别精确对齐。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ad68e6f1-e100-42fe-a4dd-59f150487588">
    <div align="center">
        <font size ="2">
        图5：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，Llama3.1 8B 模型端到端训练 Loss 对比
        </font>
    </div>
</div>

## <a name='Kernel'></a>4.3 稀疏度与 Kernel 计算时延的线性关系

FlashMask 利用注意力掩码的块稀疏性，跳过完全掩码块的计算，将计算复杂度降低到 $O((1 - ρ)T_rT_c)$ ，其中 $ρ$ 表示块稀疏性。为了验证这一关系，FlashMask 进行了多组实验，测试了三种不同的掩码类型（因果文档掩码、共享问题掩码和文档掩码），并使用不同稀疏度的数据。实验结果（如图5所示）表明，Kernel 执行延迟与稀疏性之间呈线性关系，意味着随着稀疏性的增加，FlashMask 的计算速度进一步提升。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ff1f05b4-c469-4b55-82be-f1445dbafcc6">
    <div align="center">
        <font size ="2">
        图5：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，Llama3.1 8B 模型端到端训练 Loss 对比
        </font>
    </div>
</div>

## <a name='Kernel-1'></a>4.4 Kernel 性能对比
关注到近期 PyTorch 推出了 FlexAttention[4]（使用编译器技术支持 Attention Mask），FlashMask 与之在 Kernel 级别进行了对比。在各种常见的注意力掩码模式下，FlashMask 展现了更高的计算效率。在 TFLOPs/s 指标上，FlashMask 比 FlexAttention 高出12.1%至60.7%，在 A100 GPU 上实现了37.8%至62.3%的理论峰值计算性能。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f4ea0875-adf2-471c-bb55-fe254e062c0a">
    <div align="center">
        <font size ="2">
        图5：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，Llama3.1 8B 模型端到端训练 Loss 对比
        </font>
    </div>
</div>


## <a name='QuickStart'></a>5. 快速开始

## <a name='Requirements'></a>5.1 Requirements

- PaddlePaddle develop
- PaddleNLP  develop

## <a name='SFTLoRA'></a>5.2 SFT & LoRA

### <a name='DataPreparation'></a>5.2.1 数据准备


The data format we support is as follows:

- `src` : `str, List(str)`, the input instruction or prompt for the model.
- `tgt` : `str, List(str)`, the output generated by the model.

Sample data:
```
{"src": ["Show me the most compelling argument for the existence of God from a theists perspective and then contrast that with the most compelling argument from an atheists perspective. 1 / 1", "The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>"], "tgt": ["The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>", "Why are these arguments considered the most compelling?1 / 1"]}
...
```

We offer [allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) dataset for immediate use：

```bash
mkdir data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/tulu.jsonl
mv tulu.jsonl data/train.json
```

### <a name='SFT'></a>5.2.2 SFT
```
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py./config/llama/flashmask/sft.json
```

### <a name='LoRA'></a>5.2.3 LoRA
```
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/flashmask/lora.json
```

## <a name='DPORM'></a>5.3 DPO & RM

### <a name='DataPreparation-1'></a>5.3.1 数据准备

The data format we support is a JSON file where each line contains a dictionary, and each dictionary includes the following fields:

- `src` : `str, List(str)`,the user's dialogue content.
- `tgt` : `str, List(str)`, the system's response content.
- `response` : `str, List(str)`,contains both chosen and rejected responses.
- `sort` : `List(int)`, the sort value is used to distinguish between chosen and rejected responses (the smaller sort value corresponds to rejected, and the larger sort value corresponds to chosen).

Sample data:

```text
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
...
```

We offer [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset for immediate use：

```bash
mkdir dpo_data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/ultrafeedback.jsonl
mv ultrafeedback.jsonl dpo_data/
```
### <a name='DPO'></a>5.3.2 DPO

```bash
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/flashmask/dpo.json
```

### <a name='RM'></a>5.3.3 RM

```bash
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/rm/flashmask/run_reward.py ./config/llama/flashmask/rm.json
```
