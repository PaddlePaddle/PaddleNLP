# FlashMask

在 Transformer 类大模型训练任务中，注意力掩码（Attention Mask）一方面带来了大量的冗余计算，另一方面因其 $O(N^2)$ 巨大的存储占用导致难以实现长序列场景的高效训练（其中 $N$ 为序列长度）。虽然业界已有 FlashAttention 等针对特定注意力掩码的计算加速方法，但其支持的注意力掩码模式有限，难以满足大模型训练任务对灵活注意力掩码的需求。为了解决上述问题，飞桨独创 FlashMask 技术，提出了列式稀疏的注意力掩码表示方法，支持灵活多样的注意力掩码模式，使得存储复杂度从 $O(N^2)$ 降低至 $O(N)$，并在此基础上实现了高效的算子 Kernel，极致加速大模型训练效率，尤其是长序列场景下的训练效率。

我们在 NVIDIA A100 (80G) GPU 上对 FlashMask 在大语言模型微调和对齐训练中的表现进行了评估，包括 SFT、LoRA、DPO 和 RM。与现有的 FlashAttention 密集掩码方法相比，FlashMask 在端到端训练速度上实现了显著提升，速度提高幅度在1.65倍到3.22倍之间。此外，我们还评估了其内核层次上的性能。FlashMask 在理论最大浮点运算次数上达到了37.8%到62.3%，在内核每秒浮点运算次数（TFLOPs/s）方面，其性能超过 FlexAttention，提升幅度为12.1%到60.7%。

* arXiv 论文地址 https://arxiv.org/pdf/2410.01359
* PaddlePaddle 官方文档地址 https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/functional/flashmask_attention_en.html
* PaddleNLP 开源集成 https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/docs/flashmask.md
* 星河社区快速体验 [【PaddleNLP 3.0】FlashMask 灵活注意力掩码，长序列训练利器 - 飞桨 AI Studio 星河社区](https://aistudio.baidu.com/projectdetail/8459413)

**目录**
<!-- vscode-markdown-toc -->
* [1. 大语言模型的挑战](#1.)
* [2. FlashMask 的创新：列式稀疏掩码表示方法与高效计算](#2.)
    * [2.1 关键洞察](#2.1)
    * [2.2 注意力掩码的列式稀疏掩码表示方法](#2.2)
    * [2.3 扩展 FlashAttention 支持复杂掩码](#2.3)
        * [2.3.1 预处理阶段](#2.3.1)
        * [2.3.2 实时块跳过计算阶段](#2.3.2)
    * [2.4 效率提升与精度保证](#2.4)
* [3. FlashMask 的优势：速度与存储的双重提升](#3.)
    * [3.1 端到端训练吞吐量提升](#3.1)
    * [3.2 端到端训练收敛验证](#3.2)
    * [3.3 稀疏度与 Kernel 计算时延的线性关系](#3.3)
    * [3.4 Kernel 性能对比](#3.4)
* [4. FlashMask 的应用：赋能大语言模型](#4.)
    * [4.1 可广泛应用于大语言模型的下游训练加速](#4.1)
    * [4.2 支持单向/双向混合注意力掩码模式训练](#4.2)
    * [4.3 支持多模态图文数据的混合多分辨率训练](#4.3)
* [5. 快速开始](#5.)
    * [5.1 环境依赖](#5.1)
    * [5.2 SFT & LoRA](#5.2)
        * [5.2.1 数据准备](#5.2.1)
        * [5.2.2 SFT](#5.2.2)
        * [5.2.3 LoRA](#5.2.3)
    * [5.3 DPO & RM](#5.3)
        * [5.3.1 数据准备](#5.3.1)
        * [5.3.2 DPO](#5.3.2)
        * [5.3.3 RM](#5.3.3)
* [6. 参考文献](#6.)

<!-- vscode-markdown-toc-config
    numbering=false
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='1.'></a>1. 大语言模型的挑战

随着人工智能技术的迅猛发展，以 Transformer 为代表的大模型在自然语言处理、计算机视觉和多模态应用中展现出了非凡的能力。在这些大模型中，注意力（Attention）机制是一个关键环节。为了在大模型训练任务中确定哪些 Query-Key token 之间需要进行有效的 Attention 计算，业界通常使用注意力掩码（Attention Mask）。然而，目前的注意力掩码通常采用二维稠密矩阵表示，这导致了一些问题。一方面，这种表示方法引入了大量冗余计算，因为许多无效的 token 间 Attention 仍需计算；另一方面，这种掩码的空间复杂度为 $O(N^2)$（其中 $N$ 为序列长度），在长序列的训练场景中可能会造成巨大的存储压力，因此难以进行高效训练。为了解决这些问题，业界已经提出了一些方案，如 Memory Efficient Attention (MEA) [1] 和 FlashAttention [2]。然而，这些方案支持的注意力掩码类型较为有限。正如图1所示，FlashAttention 只能支持如纯因果掩码（Causal）、滑动窗口掩码（Sliding Window）、因果文档掩码（Causal Document Mask）和文档掩码（Document Mask）等几种固定形式的掩码。然而，实际训练任务中使用的注意力掩码形式往往丰富多变，当前技术难以满足大模型在不同训练任务中对注意力掩码灵活性的要求。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/7b1013c6-de4b-4965-bbe3-857511c2dc5f">
    <div align="center">
        <font size ="2">
        图1: 常见的注意力掩码类型
        </font>
    </div>
</div>

## <a name='2.'></a>2. FlashMask 的创新：列式稀疏掩码表示方法与高效计算

### <a name='2.1'></a>2.1 关键洞察
FlashMask 的核心发现是，在大模型常见的注意力掩码模式中，Query-Key token 的掩码模式具有一定的连续性。具体而言，对于每一个 Key token，无效注意力计算的 Query token 是相邻排列的。也就是说，在图1中二维掩码矩阵中，Query token 作用在每一列的 Key token 的灰色部分沿列方向连续分布。基于这一洞察，FlashMask 巧妙地将二维稠密掩码矩阵转换为一维的行索引区间，从而实现更为紧凑的表示形式，并显著降低了存储需求。我们可以公式化表示为：

$$M_{j} = [start_j, end_j), \quad \forall j \in \{1, \ldots, N\}$$

其中 $N$ 为 Key 的序列长度, $M_j$ 为二维的稠密掩码矩阵的第 $j$ 列, $[start_j, end_j)$ 为连续的行索引区间，表示 $start_j$ 到 $end_{j} - 1$ 的连续 Query token 是被 mask 掉，置为无效 Attention 计算。

### <a name='2.2'></a>2.2 注意力掩码的列式稀疏掩码表示方法
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
        图2：较为复杂的二维稠密因果注意力的掩码矩阵示意图
        </font>
    </div>
</div>

如图2所示，我们展示了16个 Query token 和16个 Key token 做 Attention 计算时较为复杂的二维稠密因果注意力的掩码矩阵，灰色单元格是 mask 区域。

可以通过 $[LTS,LTE)$ 两个向量进行表达，如下所示：
| col_idx | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
|---------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| $LTS$   | 13 | 5  | 5  | 5  | 6  | 6  | 9  | 9  | 9  | 12 | 12 | 12 | 16 | 16 | 16 | 16 |
| $LTE$   | 15 | 14 | 14 | 15 | 12 | 12 | 11 | 11 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

以第1列为例，开始 mask 的行为13，结束 mask 的行为15（开区间），表示位置为13和14的 Query token 不与位置为0的 Key token 做有效 Attention 计算。

<div align="center">
    <div align="center">
        <img width="300" alt="llm" src="https://github.com/user-attachments/assets/67c8076a-da8e-415b-988a-6b5f65023464">
    </div>
    <div align="center">
        <font size ="2">
        图3: 使用 FlashMask 的列式稀疏掩码表示方法表示图1的注意力掩码模式
        </font>
    </div>
</div>

更多的例子参考图3，FlashMask 使用列式稀疏掩码表示方法，表达了图1中所有的注意力掩码模式。其中 $-$ 的空缺表示在不同的场景下有不同的默认值, $LTS$ 和 $UTS$ 中的默认值是 0，表示 mask 区域默认从第0行开始, $LTE$和 $UTE$中的默认值是 Query 的序列长度，表示 mask 区域默认结束于最后一行。


### <a name='2.3'></a>2.3 扩展 FlashAttention 支持复杂掩码

FlashMask 将列式掩码表示方法集成到 FlashAttention-2 算法中，扩展了其对注意力掩码的支持能力。FlashMask 的高性能 Kernel 实现包括两个关键步骤：预处理和实时块跳过计算。

在 FlashAttention 的 Kernel 实现中，得分矩阵（score matrix）的计算是分块（Tile Block）实现的。如图4的简化表示所示，整个得分矩阵计算被分为了 4 x 4 的块，每个块包含 4 个 Query token 和 4 个 Key token 交互的 4 x 4  Attention 计算。FlashMask 的原始输入是 token 级别的逐列表示，通过预处理阶段转化成块级别的表示，用于在实时跳过计算阶段快速实时计算出每个块的类型。

<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/1a244bb1-1b3c-4bc4-8839-5d3e77f02bed">
    <div align="center">
        <font size ="2">
        图4：FlashMask 计算过程示意图
        </font>
    </div>
</div>

#### <a name='2.3.1'></a>2.3.1 预处理阶段
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
        图5：预处理计算的分块最大值/最小值计算
        </font>
    </div>
</div>

#### <a name='2.3.2'></a>2.3.2 实时块跳过计算阶段
在实时计算阶段，FlashMask 利用预处理生成的最小值和最大值向量，对注意力得分矩阵的每个分块进行分类，以提升计算效率。分类依据为以下三种类型：

* 完全掩码块：若 $BlockRow_{min} \geq Start^{max} \text{ and } BlockRow_{max} \leq End^{min}$ ，则此块的所有元素均被掩码，计算可直接跳过。
* 部分掩码块：若 $BlockRow_{min} < End^{max} \text{ and } BlockRow_{max} > Start^{min}$ ，则此块的部分元素被掩码，因此需要对该块进行逐元素的掩码计算。
* 未掩码块：其他情况则归为未掩码块，此类块中的所有元素均未被掩码，可以简化计算过程，不进行额外的掩码操作。

通过这种分类处理，FlashMask 可以显著提升计算效率：完全掩码块的计算被跳过，未掩码块的计算得以简化，仅对部分掩码块执行必要的掩码操作。

图4展示了在因果掩码场景下，使用 $LTS$ 和 $LTE$ 进行 Kernel 计算的完整过程。图中每种分块类型的实时计算公式都已标注，以下是具体例子说明：

* 完全掩码块，例如，图4中 [3, 2] 位置的块，其最小行号为12，大于等于 $LTStart^{max}=12$ ，最大行号为15，小于等于 $LTEnd^{max}=16$ ，因此块中所有元素被掩码，计算可以直接跳过。
* 部分掩码块，例如，图4中 [1, 1] 位置的块，其最小行号为4，小于 $LTEnd^{max}=12$ ，最大行号为7，大于 $LTStart^{min}=6$ ，因此块中部分元素被掩码，需要对该块逐元素进行掩码计算。
* 未掩码块，例如，图4中 [3, 1] 位置的块，其最小行号为12，大于等于 $LTEnd^{max}=12$ ，表明此块中所有元素未被掩码，计算时无需额外的掩码操作，从而减少计算开销。

算法1详细描述了 FlashMask 扩展 FlashAttention-2 的前向计算过程，其中浅蓝色阴影部分表示 FlashMask 新增的计算步骤 [3]。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/91153ab6-240c-4787-9469-ef29cdc8eb12">
    <div align="center">
        <font size ="2">
        算法1: FlashMask 的前向计算伪代码
        </font>
    </div>
</div>

### <a name='2.4'></a>2.4 效率提升与精度保证
FlashMask 充分利用了注意力掩码中的稀疏性，通过跳过完全掩码块的计算，减少了计算开销，同时不改变算法的精度。与使用稠密掩码矩阵的注意力计算保持比特级别的数值等效性，确保了精度无损。

## <a name='3.'></a>3. FlashMask 的优势：速度与存储的双重提升

### <a name='3.1'></a>3.1 端到端训练吞吐量提升
在 Llama-2 7B、13B、70B 等模型规模下，针对 SFT、LoRA、DPO、RM 四种下游训练场景和不同序列长度的实验表明，FlashMask 在各个模型规模和序列长度下均实现了端到端的加速和存储效率的提升。相比现有的基于稠密掩码矩阵的计算方法，FlashMask 实现了1.65倍至3.22倍的吞吐量提升，并支持更长的序列长度。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/49208427-49b8-4a74-aca4-e7782294071d">
    <div align="center">
        <font size ="2">
        图6：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，3 个 Llama2 模型规模，在不同序列长度下的端到端训练吞吐量
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/9bbe637b-9a04-4df4-a227-36f6eab38bbc">
    <div align="center">
        <font size ="2">
        图7：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，3 个 Llama2 模型规模，不同序列长度下的端到端训练峰值显存消耗
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f0f7880a-c439-4a9f-9232-6f4171090c90">
    <div align="center">
        <font size ="2">
        表2：在 Llama2 7B 模型上 FlashMask 对比 FlashAttention (Causal=True) 的显存消耗，单位(GB)
        </font>
    </div>
</div>

### <a name='3.2'></a>3.2 端到端训练收敛验证
在 Llama 3.1 模型上的实验验证了 FlashMask 对收敛精度没有影响。作为一种精确的算法，通过控制计算过程的随机性（如 FlashAttention 反向 Query 梯度计算使用 atomicAdd 操作），FlashMask 可以与使用稠密掩码的 FlashAttention 在比特级别精确对齐。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ad68e6f1-e100-42fe-a4dd-59f150487588">
    <div align="center">
        <font size ="2">
        图8：在四个下游训练任务（SFT、LoRA、DPO 和 RM）中，Llama3.1 8B 模型端到端训练 Loss 对比
        </font>
    </div>
</div>

### <a name='3.3'></a>3.3 稀疏度与 Kernel 计算时延的线性关系

FlashMask 利用注意力掩码的块稀疏性，跳过完全掩码块的计算，将计算复杂度降低到 $O((1 - ρ)T_rT_c)$ ，其中 $ρ$ 表示块稀疏性。为了验证这一关系，FlashMask 进行了多组实验，测试了三种不同的掩码类型（因果文档掩码、共享问题掩码和文档掩码），并使用不同稀疏度的数据。实验结果（如图5所示）表明，Kernel 执行延迟与稀疏性之间呈线性关系，意味着随着稀疏性的增加，FlashMask 的计算速度进一步提升。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ff1f05b4-c469-4b55-82be-f1445dbafcc6">
    <div align="center">
        <font size ="2">
        图9：不同块稀疏度下的 Kernel 计算时延
        </font>
    </div>
</div>

### <a name='3.4'></a>3.4 Kernel 性能对比
关注到近期 PyTorch 推出了 FlexAttention[4]（使用编译器技术支持 Attention Mask），FlashMask 与之在 Kernel 级别进行了对比。在各种常见的注意力掩码模式下，FlashMask 展现了更高的计算效率。在 TFLOPs/s 指标上，FlashMask 比 FlexAttention 高出12.1%至60.7%，在 A100 GPU 上实现了37.8%至62.3%的理论峰值计算性能。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f4ea0875-adf2-471c-bb55-fe254e062c0a">
    <div align="center">
        <font size ="2">
        图10：在 A100-SXM 80G GPU 上的 Kernel 前向和反向速度对比。FlexAttention 使用 PyTorch 2.6.0.dev20240920+cu124
        </font>
    </div>
</div>

## <a name='4.'></a>4. FlashMask 的应用：赋能大语言模型
FlashMask 的创新和优势为 Transformer 类大模型的注意力机制训练加速开辟了新的可能，可广泛应用于各种任务，并支持超长序列高效训练。

### <a name='4.1'></a>4.1 可广泛应用于大语言模型的下游训练加速
FlashMask 可以应用于大语言模型的下游任务训练，例如 SFT、LoRA、DPO、RM 等。特别是在 DPO 和 RM 的训练中，其数据由问题和回答对组成，训练时多个答案可以共享一个问题，从而大幅减少对问题 token 的冗余计算。

### <a name='4.2'></a>4.2 支持单向/双向混合注意力掩码模式训练
FlashMask 支持多种注意力模式，包括因果掩码（单向注意力）和文档掩码（双向注意力），因此能够灵活地应用于需要混合注意力的场景。例如：

* 全局 + 滑动窗口掩码：这种掩码结合了全局注意力和滑动窗口注意力，既能捕捉全局上下文信息，又能关注局部细节。FlashMask 能高效处理这种混合掩码，提升模型性能。
* 前缀语言模型：在生成文本时，前缀部分需要关注所有的 token，而其他部分使用因果掩码（如 T5 模型的预训练）。FlashMask 可以同时支持这两种注意力模式，提高前缀语言模型的训练和推理效率。

### <a name='4.3'></a>4.3 支持多模态图文数据的混合多分辨率训练
在多模态数据处理中，不同模态的数据可能具有不同的分辨率。虽然文中未明确提及 FlashMask 在多模态和多分辨率训练中的应用，但 FlashMask 可以通过不同的注意力模式和掩码策略，有效处理这些具有不同分辨率的数据。针对长序列处理能力的优化，使得 FlashMask 能够帮助模型更好地学习不同模态数据之间的关联。例如，在图文匹配任务中，FlashMask 可以帮助模型更有效地对齐图像和文本中的关键信息。

FlashMask 的开源代码已在 PaddlePaddle 和 PaddleNLP 平台发布，支持超过千亿参数的模型以及超过 128K tokens 的上下文长度。我们相信，FlashMask 将成为推动大语言模型发展的重要力量，为算法研究人员提供更广阔的注意力掩码创新与研究空间。

## <a name='5.'></a>5. 快速开始

### <a name='5.1'></a>5.1 环境依赖

* python >= 3.8
* paddlepaddle >= 3.0.0b0
如果您尚未安装 PaddlePaddle，请参考 [飞桨官网](https://www.paddlepaddle.org.cn/) 进行安装。

通过以下命令安装最新 develop 分支代码：

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### <a name='5.2'></a>5.2 SFT & LoRA

#### <a name='5.2.1'></a>5.2.1 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```json
{
   "src": ["Show me the most compelling argument for the existence of God from a theists perspective and then contrast that with the most compelling argument from an atheists perspective. 1 / 1", "The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>"],
   "tgt": ["The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>", "Why are these arguments considered the most compelling?1 / 1"]
}
```

为了方便测试，我们也提供了 [allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) 数据集可以直接使用：

```bash
mkdir data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/tulu.jsonl
mv tulu.jsonl data/train.json
```

#### <a name='5.2.2'></a>5.2.2 SFT
```shell
# SFT 启动命令参考
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/flashmask/sft.json
```

#### <a name='5.2.3'></a>5.2.3 LoRA
```shell
# LoRA 启动命令参考
python  -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7"  run_finetune.py ./config/llama/flashmask/lora.json
```

### <a name='5.3'></a>5.3 DPO & RM

#### <a name='5.3.1'></a>5.3.1 数据准备

我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：

- `src` : `str, List(str)`, 用户对话内容。
- `tgt` : `str, List(str)`, 系统回复内容。
- `response` : `str, List(str)`, 包含 chosen 和 rejected 回复。
- `sort` : `List(int)`, sort 值用于区分 response 中 chosen 和 rejected（sort 值小的是 rejected，sort 值大的是 chosen）。

样例数据：

```json
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
```

为了方便测试，可以下载 [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 数据集直接使用：

```bash
mkdir dpo_data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/ultrafeedback.jsonl
mv ultrafeedback.jsonl dpo_data/
```
#### <a name='5.3.2'></a>5.3.2 DPO

```bash
# DPO 启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/flashmask/dpo.json
```

#### <a name='5.3.3'></a>5.3.3 RM

```bash
# RM 启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/rm/flashmask/run_reward.py ./config/llama/flashmask/rm.json
```


## <a name='6.'></a>6. 参考文献

[1] Self-attention Does Not Need O(n^2) Memory. https://arxiv.org/pdf/2112.05682

[2] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. https://arxiv.org/pdf/2307.08691

[3] FlashMask: Efficient and Rich Mask Extension of FlashAttention. https://arxiv.org/pdf/2410.01359

[4] FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention. https://pytorch.org/blog/flexattention/
