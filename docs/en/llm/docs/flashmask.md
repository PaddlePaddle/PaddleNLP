# FlashMask Flexible Attention Mask

In large-scale model training tasks such as Transformers, attention masks (Attention Mask) introduce significant redundant computations on one hand, and due to their $O(N^2)$ memory footprint, make it challenging to achieve efficient training in long-sequence scenarios (where $N$ is the sequence length). While the industry has developed computation acceleration methods like FlashAttention for specific attention masks, their supported attention mask patterns are limited, failing to meet the diverse requirements of large model training. To address these issues, PaddlePaddle's innovative FlashMask technology proposes a column-wise sparse attention mask representation method, supporting flexible and diverse attention mask patterns, thereby reducing the storage complexity from $O(N^2)$ to $O(N)$.
# FlashMask: Flexible Attention Mask for Accelerating Large Language Model Training

We implemented highly efficient operator kernels to maximize the training efficiency of large models, particularly in long sequence scenarios.

We evaluated FlashMask's performance on NVIDIA A100 (80G) GPUs for large language model fine-tuning and alignment training, including SFT, LoRA, DPO, and RM. Compared to existing dense masking methods like FlashAttention, FlashMask achieves significant end-to-end training speed improvements ranging from 1.65x to 3.22x. Additionally, we evaluated its kernel-level performance. FlashMask reaches 37.8% to 62.3% of theoretical maximum FLOPS utilization, and outperforms FlexAttention by 12.1% to 60.7% in kernel-level TFLOPS/s.

* arXiv paper: https://arxiv.org/pdf/2410.01359
* PaddlePaddle documentation: https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/functional/flashmask_attention_en.html
* PaddleNLP implementation: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/docs/flashmask.md
* Quick experience on Paddle AI Studio: [【PaddleNLP 3.0】FlashMask Flexible Attention Mask - PaddlePaddle AI Studio](https://aistudio.baidu.com/projectdetail/8459413)

**Table of Contents**
<!-- vscode-markdown-toc -->
* [1. Challenges of Large Language Models](#1)
* [2. FlashMask Innovation: Column-Sparse Mask Representation and Efficient Computation](#2)
    * [2.1 Key Insights](#21)
    * [2.2 Column-Sparse Representation for Attention Masks](#22)
    * [2.3 Extending FlashAttention for Complex Masks](#23)
        * [2.3.1 Preprocessing Phase](#231)
        * [2.3.2 Real-time Block Skipping](#232)
    * [2.4 Efficiency Improvements with Accuracy Guarantees](#24)
* [3. FlashMask Advantages: Dual Gains in Speed and Memory](#3)
    * [3.1 End-to-End Training Throughput Improvement](#31)
    * [3.2 End-to-End Training Convergence Validation](#32)
    * [3.3 Linear Relationship Between Sparsity and Kernel Latency](#33)
    * [3.4 Kernel Performance Comparison](#34)
* [4. FlashMask Applications: Empowering Large Language Models](#4)
    * [4.1 Broad Applicability for Downstream Training Acceleration](#41)
    * [4.2 Support for Unidirectional/Bidirectional Hybrid Attention Modes](#42)
    * [4.3 Multi-modal Training with Mixed Resolutions](#43)
* [5. Quick Start](#5)
    * [5.1 Environment Requirements](#51)
    * [5.2 SFT & LoRA](#52)
        * [5.2.1 Data Preparation](#521)
        * [5.2.2 SFT Training](#522)
        * [5.2.3 LoRA Training](#523)
    * [5.3 DPO & RM](#53)
        * [5.3.1 Data Preparation](#531)
        * [5.3.2 DPO Training](#532)
        * [5.3.3 RM Training](#533)
* [6. References](#6)

<!-- vscode-markdown-toc-config
    numbering=false
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='1'></a>1. Challenges of Large Language Models

With the rapid development of AI technology, Transformer-based large models have demonstrated remarkable capabilities in NLP, computer vision, and multi-modal applications. The attention mechanism plays a crucial role in these models. To determine which Query-Key token pairs require valid attention computation during training, the industry typically uses attention masks. However, current 2D dense matrix representations of attention masks introduce two major issues: (1) redundant computations from unnecessary token-pair attention calculations, and (2) O(N²) space complexity that becomes...
$O(N^2)$ (where $N$ is the sequence length) may cause significant memory pressure in long sequence training scenarios, making efficient training challenging. To address these issues, industry has proposed solutions such as Memory Efficient Attention (MEA) [1] and FlashAttention [2]. However, these solutions support limited types of attention masks. As shown in Figure 1, FlashAttention can only support fixed mask patterns like pure causal mask (Causal), sliding window mask (Sliding Window), causal document mask (Causal Document Mask), and document mask (Document Mask). Yet, actual training tasks often require diverse attention mask patterns, and current technologies struggle to meet the flexibility requirements of attention masks for large models in different training scenarios.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/7b1013c6-de4b-4965-bbe3-857511c2dc5f">
    <div align="center">
        <font size ="2">
        Figure 1: Common attention mask types
        </font>
    </div>
</div>

## <a name='2.'></a>2. FlashMask's Innovation: Column-wise Sparse Mask Representation and Efficient Computation

### <a name='2.1'></a>2.1 Key Insight
The core discovery of FlashMask is that in common attention mask patterns of large models, the mask patterns between Query-Key tokens exhibit certain continuity. Specifically, for each Key token, the invalid Query tokens for attention computation are arranged consecutively. That is, in Figure 1, the grayed-out regions along the column direction in the 2D mask matrix correspond to consecutive Query tokens. Based on this insight, FlashMask cleverly converts the 2D dense mask matrix into 1D row index intervals, achieving a more compact representation and significantly reducing memory requirements. We can formalize this as:

$$M_{j} = [start_j, end_j), \quad \forall j \in \{1, \ldots, N\}$$

where $N$ is the sequence length of Keys, $M_j$ represents the $j$-th column of the 2D dense mask matrix, and $[start_j, end_j)$ denotes the consecutive row index interval, indicating that Query tokens from $start_j$ to $end_j - 1$ are masked as invalid for attention computation.

### <a name='2.2'></a>2.2 Column-wise Sparse Mask Representation for Attention
To efficiently handle complex mask patterns in causal and bidirectional attention scenarios, FlashMask proposes a novel column-wise sparse representation. Using the diagonal as a boundary, it employs four 1D vectors to represent masks:
* Lower Triangular Start (LTS)
* Lower Triangular End (LTE)
* Upper Triangular Start (UTS)
* Upper Triangular End (UTE)

Here, the masked row index interval in the lower triangular region is represented by $[LTS, LTE)$, while the upper triangular region uses $[UTS, UTE)$.
<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/989cc61e-174b-489d-ba7a-d1e6d172ff91">
    <div align="center">
        <font size ="2">
        Figure 2: Schematic diagram of a more complex 2D dense causal attention mask matrix
        </font>
    </div>
</div>

As shown in Figure 2, we present a more complex 2D dense causal attention mask matrix when performing attention calculations between 16 Query tokens and 16 Key tokens, where the gray cells represent the masked regions.

This can be expressed using two vectors $[LTS, LTE)$, as shown below:
| col_idx | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
|---------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| $LTS$   | 13 | 5  | 5  | 5  | 6  | 6  | 9  | 9  | 9  | 12 | 12 | 12 | 16 | 16 | 16 | 16 |
| $LTE$   | 15 | 14 | 14 | 15 | 12 | 12 | 11 | 11 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

Taking the first column as an example, the starting row for masking is 13, and the ending row is 15 (open interval), indicating that Query tokens at positions 13 and 14 do not perform valid attention calculations with the Key token at position 0.

<div align="center">
    <div align="center">
        <img width="300" alt="llm" src="https://github.com/user-attachments/assets/67c8076a-da8e-415b-988a-6b5f65023464">
    </div>
    <div align="center">
        <font size ="2">
        Figure 3: Columnar sparse mask representation using FlashMask to represent the attention mask pattern from Figure 1
        </font>
    </div>
</div>

As shown in Figure 3, FlashMask uses a columnar sparse mask representation to express all the attention mask patterns from Figure 1. The '-' entries indicate different default values in various scenarios. The default values in $LTS$ and $UTS$ are 0, meaning the masked region starts from row 0 by default, while $LTE$ and $UTE$...
### <a name='2.3'></a>2.3 Extending FlashAttention to Support Complex Masks

FlashMask integrates column-wise mask representation into the FlashAttention-2 algorithm, extending its support for attention masks. The high-performance kernel implementation of FlashMask consists of two key steps: preprocessing and real-time block skipping computation.

In FlashAttention's kernel implementation, the score matrix computation is implemented using tiling. As shown in the simplified representation in Figure 4, the entire score matrix computation is divided into 4x4 tiles, where each tile contains 4 query tokens and 4 key tokens interacting through 4x4 attention computations. FlashMask's original input is token-level column-wise representation, which is converted into tile-level representation during the preprocessing phase. This is used to quickly determine the type of each tile in real-time computation skipping phase.

<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/1a244bb1-1b3c-4bc4-8839-5d3e77f02bed">
    <div align="center">
        <font size ="2">
        Figure 4: Schematic diagram of FlashMask computation process
        </font>
    </div>
</div>

#### <a name='2.3.1'></a>2.3.1 Preprocessing Phase
In FlashMask's preprocessing phase, the column-wise sparse mask vectors $LTS$, $LTE$, $UTS$, $UTE$ are first loaded into high-bandwidth memory (HBM). Then, based on FlashAttention's tile column size, the column-wise sparse mask vectors are tiled. The maximum and minimum values of all columns within each tile are computed, generating 8 intermediate vectors:

* $LTStart^{min}$, $LTStart^{max}$
* $LTEnd^{min}$, $LTEnd^{max}$
* $UTStart^{min}$, $UTStart^{max}$
* $UTEnd^{min}$, $UTEnd^{max}$

Taking the leftmost 4 tiles in Figure 4 as example, a tile contains 4 columns with $LTS=[13,5,5,5]$ and $LTE=[15,14,14,15]$. Therefore, $LTStart^{min}=min(LTS)=5$, $LTStart^{max}=max(LTS)=13$, $LTEnd^{min}=min(LTE)=14$, $LTEnd^{max}=max(LTE)=15$
The remaining computation results are shown in Figure 5:

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/76a5cca9-c268-4bd8-b0f6-d84ba3948b68">
    <div align="center">
        <font size ="2">
        Figure 5: Chunked Max/Min Value Computations in Preprocessing
        </font>
    </div>
</div>

#### <a name='2.3.2'></a>2.3.2 Real-time Block Skipping Computation Stage
In the real-time computation phase, FlashMask leverages the min/max vectors generated during preprocessing to classify each chunk of the attention score matrix into three types for computational efficiency:

* **Fully masked block**: If $BlockRow_{min} \geq Start^{max} \text{ and } BlockRow_{max} \leq End^{min}$, all elements in this block are masked and the computation can be directly skipped.
* **Partially masked block**: If $BlockRow_{min} < End^{max} \text{ and } BlockRow_{max} > Start^{min}$, partial elements in this block are masked, thus requiring element-wise masking computation.
* **Unmasked block**: All other cases are classified as unmasked blocks, where no elements are masked, allowing simplified computation without additional masking operations.

This classification enables FlashMask to significantly improve computational efficiency: computations for fully masked blocks are skipped, computations for unmasked blocks are simplified, and only partially masked blocks require element-wise masking operations.

Figure 4 demonstrates the complete process of Kernel computations using $LTS$ and $LTE$ under causal masking scenarios. The real-time computation formulas for each block type are labeled in the figure. Specific examples include:

* **Fully masked block**: For example, the block at position [3, 2] in Figure 4, with min row number 12 (≥ $LTStart^{max}=12$) and max row number 15 (≤ $LTEnd^{max}=16$). All elements in this block are masked, allowing direct computation skipping.
* **Partially masked block**: For example, the block at position [1, 1] in Figure 4, with min row number 4 (< $LTEnd^{max}=12$) and max row number 7 (> $LTStart^{min}=6$). This block requires element-wise masking computation.
* **Unmasked block**: For example, the block at position [3, 1] in Figure 4, with min row number 12 (≥ $LTEnd^{max}=12$)
, indicating that all elements in this block are unmasked, eliminating the need for additional masking during computation to reduce computational overhead.

Algorithm 1 details the forward computation process of FlashMask extended from FlashAttention-2, where the light blue shaded sections represent new computational steps added by FlashMask [3].

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/91153ab6-240c-4787-9469-ef29cdc8eb12">
    <div align="center">
        <font size ="2">
        Algorithm 1: Pseudo-code for FlashMask Forward Computation
        </font>
    </div>
</div>

### <a name='2.4'></a>2.4 Efficiency Improvement and Precision Guarantee
FlashMask fully exploits the sparsity in attention masks by skipping computations for fully masked blocks, reducing computational overhead while maintaining algorithm precision. It preserves bit-level numerical equivalence with dense mask matrix-based attention computations, ensuring no loss of precision.

## <a name='3.'></a>3. FlashMask Advantages: Dual Improvement in Speed and Memory

### <a name='3.1'></a>3.1 End-to-End Training Throughput Improvement
Experiments on model scales including Llama-2 7B, 13B, 70B, and across four downstream training scenarios (SFT, LoRA, DPO, RM) with varying sequence lengths demonstrate that FlashMask achieves consistent acceleration and memory efficiency improvements. Compared to existing dense mask matrix-based methods, FlashMask delivers 1.65x to 3.22x throughput improvement while supporting longer sequence lengths.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/49208427-49b8-4a74-aca4-e7782294071d">
    <div align="center">
        <font size ="2">
        Figure 6: End-to-End Training Throughput Across Four Downstream Tasks (SFT, LoRA, DPO, RM) for Three Llama2 Model Scales at Different Sequence Lengths
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/9bbe637b-9a04-4df4-a227-36f6eab38bbc">
    <div align="center">
        <font size ="2">
        Figure 7: Peak GPU Memory Consumption During End-to-End Training for Four Downstream Tasks (SFT, LoRA, DPO, RM) Across Three Llama2 Model Scales at Different Sequence Lengths
        </font>
    </div>
</div>

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f0f7880a-c439-4a9f-9232-6f4171090c90">
    <div align="center">
        <font size ="2">
        Table 2: GPU Memory Consumption Comparison Between FlashMask and FlashAttention (Causal=True) on Llama2 7B Model, Unit(GB)
        </font>
    </div>
</div>

### <a name='3.2'></a>3.2 End-to-End Training Convergence Validation
Experiments on Llama 3.1 models confirm that FlashMask does not affect convergence accuracy. As a precise algorithm, FlashMask achieves bit-level alignment with dense mask-based FlashAttention by controlling computational randomness (e.g., using atomicAdd operations for backward query gradient computation in FlashAttention).

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ad68e6f1-e100-42fe-a4dd-59f150487588">
    <div align="center">
        <font size ="2">
        Figure 8: Training Loss Comparison for Llama3.1 8B Model Across Four Downstream Tasks (SFT, LoRA, DPO, RM)
        </font>
    </div>
</div>

### <a name='3.3'></a>3.3 Linear Relationship Between Sparsity and Kernel Computation Latency

FlashMask leverages block-wise sparsity in attention masks to skip fully masked blocks, reducing computational complexity to... [Remaining text truncated]
$O((1 - \rho)T_rT_c)$, where $\rho$
The term indicates block sparsity. To validate this relationship, FlashMask conducted multiple experiments with three different mask types (causal document mask, shared question mask, and document mask) using varying sparsity levels. The results (shown in Figure 5) demonstrate a linear relationship between kernel execution latency and sparsity, indicating that FlashMask's computational speed improves as sparsity increases.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/ff1f05b4-c469-4b55-82be-f1445dbafcc6">
    <div align="center">
        <font size ="2">
        Figure 9: Kernel computation latency under different block sparsity levels
        </font>
    </div>
</div>

### <a name='3.4'></a>3.4 Kernel Performance Comparison
Noting that PyTorch recently introduced FlexAttention[4] (using compiler technology to support Attention Mask), FlashMask was benchmarked against it at the kernel level. FlashMask demonstrated higher computational efficiency across common attention mask patterns. In TFLOPs/s metrics, FlashMask outperformed FlexAttention by 12.1% to 60.7%, achieving 37.8% to 62.3% of the theoretical peak performance on A100 GPUs.

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/f4ea0875-adf2-471c-bb55-fe254e062c0a">
    <div align="center">
        <font size ="2">
        Figure 10: Kernel forward/backward speed comparison on A100-SXM 80G GPU. FlexAttention uses PyTorch 2.6.0.dev20240920+cu124
        </font>
    </div>
</div>

## <a name='4.'></a>4. Applications of FlashMask: Empowering Large Language Models
FlashMask's innovations create new possibilities for accelerating attention mechanism training in Transformer-based models, enabling efficient training on ultra-long sequences across various applications.

### <a name='4.1'></a>4.1 Broad Applicability for Downstream Training Acceleration
FlashMask can accelerate downstream training tasks for large language models, including SFT, LoRA, DPO, RM, etc. Particularly in DPO and RM training where question-answer pairs allow multiple answers to share a question, FlashMask significantly reduces redundant computation on question tokens.

### <a name='4.2'></a>4.2 Supporting Uni-/Bi-directional Hybrid Attention Modes
FlashMask supports multiple attention modes (causal mask for unidirectional attention, document mask for bidirectional attention), enabling flexible hybrid scenarios:

* **Global + Sliding Window Masks**: Combining global and local attention, FlashMask efficiently processes such hybrid masks to enhance model performance while capturing both global context and local details.
* **Prefix Language Models**: During text generation, prefixes require full token attention while other parts use causal masks (e.g., T5 pretraining). FlashMask supports both modes simultaneously, improving training/inference efficiency.

### <a name='4.3'></a>4.3 Supporting Multimodal Training with Mixed Resolutions
For multimodal data with varying resolutions, FlashMask can employ different attention patterns and mask strategies to effectively process such data. Its optimized long-sequence handling helps models better learn cross-modal relationships. For example, in image-text matching tasks, FlashMask enables more effective alignment between visual and textual key information.

The open-source implementation of FlashMask is available on PaddlePaddle and PaddleNLP, supporting models with over 100B parameters and context lengths exceeding 128K tokens. We believe FlashMask will become a key enabler for large language models, providing researchers with expanded possibilities for attention mask innovation.

## <a name='5.'></a>5. Quick Start

### <a name='5.1'></a>5.1 Environment Requirements

* Python >= 3.8
* PaddlePaddle >= 3.0.0b0

If you haven't installed PaddlePaddle, please refer to the [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/) for installation.

Install the latest develop branch code via:

```bash
pip install -U https://www.paddlepaddle.org.cn/whl/paddle_nightly_cuda118.html
```
```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

### <a name='5.2'></a>5.2 SFT & LoRA

#### <a name='5.2.1'></a>5.2.1 Data Preparation

The supported fine-tuning data format is a json file where each line contains a dictionary with the following fields:

- `src`: `str, List(str)`, the model's input instruction (instruction), prompt, and the task the model should perform.
- `tgt`: `str, List(str)`, the model's output.

Sample data:
```json
{
   "src": ["Show me the most compelling argument for the existence of God from a theists perspective and then contrast that with the most compelling argument from an atheists perspective. 1 / 1", "The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>"],
   "tgt": ["The most compelling argument for the existence of God from a theist's perspective is the cosmological argument, which posits that the universe had a cause or a beginning and that cause must be God. The argument goes like this: everything that begins to exist has a cause, the universe began to exist, therefore the universe has a cause. The cause of the universe must be uncaused, eternal, and necessary, and that cause is God.\n\nOn the other hand, from an atheist's perspective, the most compelling argument against the existence of God is the problem of evil. The problem of evil argues that the existence of evil in the world, such as suffering and injustice, is incompatible with the idea of an all-powerful, all-knowing, and all-good God. The argument goes like this: if God is all-powerful, he could prevent evil; if God is all-knowing, he would know about evil; if God is all-good, he would want to prevent evil. Therefore, the existence of evil is evidence against the existence of God.\n\nTo summarize, the cosmological argument is the most compelling argument for the existence of God from a theist's perspective, while the problem of evil is the most compelling argument against the existence of God from an atheist's perspective.", "Please cite your sources for these.1 / 1", "Sure! Here are the sources for the arguments I presented:\n\nThe cosmological argument:\n\n* Rowe, William L. \"Cosmological Argument.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/cosmological-argument/>\n* Craig, William Lane. The Kalam Cosmological Argument. Eugene, OR: Wipf and Stock Publishers, 1979.\n\nThe problem of evil:\n\n* Mackie, J.L. \"Evil and Omnipotence.\" Mind 64, no. 254 (1955): 200-212.\n* Rowe, William L. \"The Problem of Evil.\" Stanford Encyclopedia of Philosophy. <https://plato.stanford.edu/entries/evil/>", "Why are these arguments considered the most compelling?1 / 1"]
}
```
To facilitate testing, we also provide the [allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) dataset which can be directly used:

```bash
mkdir data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/tulu.jsonl
mv tulu.jsonl data/train.json
```

#### <a name='5.2.2'></a>5.2.2 SFT
```shell
# SFT startup command reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/flashmask/sft.json
```

#### <a name='5.2.3'></a>5.2.3 LoRA
```shell
# LoRA startup command reference
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/flashmask/lora.json
```

### <a name='5.3'></a>5.3 DPO & RM

#### <a name='5.3.1'></a>5.3.1 Data Preparation

The supported fine-tuning data format is a json file where each line contains a dictionary with the following fields:

- `src`: `str, List(str)`, user dialogue content.
- `tgt`: `str, List(str)`, system response content.
- `response`: `str, List(str)`, contains chosen and rejected responses.
- `sort`: `List(int)`, sort values are used to distinguish between chosen and rejected responses in `response` (responses with smaller sort values are rejected, those with larger values are chosen).

Example data:
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
To facilitate testing, you can download the [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset directly:

```bash
mkdir dpo_data
wget https://paddlenlp.bj.bcebos.com/datasets/examples/ultrafeedback.jsonl
mv ultrafeedback.jsonl dpo_data/
```

#### <a name='5.3.2'></a>5.3.2 DPO

```bash
# DPO startup command reference
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/flashmask/dpo.json
```

#### <a name='5.3.3'></a>5.3.3 RM

```bash
# RM startup command reference
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" ./alignment/rm/flashmask/run_reward.py ./config/llama/flashmask/rm.json
```

## <a name='6.'></a>6. References

[1] Self-attention Does Not Need O(n^2) Memory. https://arxiv.org/pdf/2112.05682

[2] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. https://arxiv.org/pdf/2307.08691

[3] FlashMask: Efficient and Rich Mask Extension of FlashAttention. https://arxiv.org/pdf/2410.01359

[4] FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention. https://pytorch.org/blog/flexattention/
