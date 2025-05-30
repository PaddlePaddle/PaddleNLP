# PaddleNLP 大模型新手指南-精调
从零开始了解并实践如何基于 PaddleNLP 对大语言模型（如 LLaMA、Baichuan、Qwen 等）进行精调（Fine-tuning）。

我们在 Ai Studio 上同步公开了项目，也可以点击[链接](https://aistudio.baidu.com/projectdetail/9169303)在线体验大模型精调。


## 1. 什么是精调 （Fine-tuning）
经过预训练，我们拥有了经过预训练之后的基础模型，但是这个模型如果直接应用在某一个特定领域，效果可能并不太好。这是由于我们的预训练语料没有针对特定的任务场景进行特化。但是大模型经过预训练已经拥有了很强的通用能力，只需要少量特定数据，就可以大幅提升大模型在特定领域的能力。

我们简单对比下预训练与精调：
* 数据量:（预训练）海量通用文本/(精调)少量特定任务数据
* 训练目标：（预训练）学习通用语言知识/(精调)优化具体任务表现
* 训练时间：（预训练）很长/(精调)较短

## 2. 精调数据
与预训练的无监督数据不同，精调使用的数据是有监督数据，一般是一个指令（向大模型说的话）和一个输出（大模型应该的回应）。常用的数据格式是[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)，我们用一个简单的例子来展示一下其基本组成：
```
{
  "instruction": "给出首都：",
  "input": "法国",
  "output": "巴黎"
}
```
* instruction：用户向大模型输入的指令
* input：任务相关的输入，通常为空
* output：大模型的预期输出

我们本次使用[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)的 demo 数据集。


```python
# clone PaddleNLP仓库，如果之前已经操作过，可以跳过这一步
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```


```python
cd PaddleNLP/llm
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```


## 3. 精调（Fine-tuning）
对大模型的精调本质上是对其参数进行微调，在这里我们就有两种选择：
* 精调大模型的全部参数->SFT（Supervised Fine-Tuning，监督式微调）
* 精调大模型的部分参数->PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）
如果我们有足够的资源对所有的参数进行微调，那我们可以选择 SFT；如果资源不足或受限，我们可以选择 PEFT 对模型的小部分关键参数进行微调。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/user-attachments/assets/4556e9f0-d855-418f-914f-bcecccce6dba">
</div>
<div align="center">
    <font size ="1">
    大模型精调原理介绍
     </font>
</div>


与预训练一样，PaddleNLP 封装好了精调脚本，可以根据需要切换不同模型的配置文件，直接执行命令即可进行精调。

### 3.1 SFT
我们首先尝试对 Qwen2.5-0.5B 模型进行 SFT，SFT 的训练过程就是让模型根据输入尽可能产生与给定输出一样的输出。需要执行的指令如下：


```python
# 需要12G显存左右
python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json
```


```python
# 多卡命令如下
# SFT 启动命令参考，需要45G显存左右
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument.json
```

### 3.2 PEFT
在实际业务环境中，可能需要对参数量较大的模型进行微调，此时如果使用 SFT，带来的显存和时间开销可能过大。此时我们可以选择 PEFT 方法，只更新小部分参数。常用的 PEFT 方法有参数附加方法、低秩分解方法等，我们分别选择其中的一种方法进行展示，代码中的 API 可以参考[PaddleNLP PEFT API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/peft.md)。

#### 3.2.1 Prefix Tuning
[Prefix Tuning](https://arxiv.org/abs/2101.00190)（前缀调优）是一种参数高效微调（PEFT）技术，它通过在预训练大语言模型的每一层前面插入一段可学习的“前缀向量”，而不是更新模型的全部参数，从而实现任务适配。

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/8baf6943-4540-4c02-8540-35f977acc077 width=40% height=40% />
</div>

我们举个简单的例子，在 SFT 中，模型的输入数据为：“输入句”

在 Prefix Tuning 中，这个输入会变成：
```
prefix + 输入句
```

也就是模型会新增一部分专门处理 prefix，这部分多出来的参数就是我们要进行微调的部分。


```python
# 需要10G左右显存
python run_finetune.py ./config/qwen/pt_argument_0p5b.json
```

#### 3.2.2 LoRA

<div align="center">
<img src=https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/63d56558-247a-4a8d-a6ca-121c820f7534 width=30% height=30% />
</div>

LoRA(Low-rank Adaptation)方法向模型插入少量可训练的低秩矩阵，在不改变原模型参数的前提下，完成微调任务。参考图里面的示例，一个很大的灰色参数矩阵可以近似分解为两个较小的矩阵 A 与 B 相乘。这样一个大矩阵就被分解为了两个小矩阵，需要微调的参数量大大减少。在训练完成后，在推理的过程中将原本的参数矩阵与微调的参数矩阵相加，既不影响原本的预训练参数，也可以实现微调效果，这样的优点也让 LoRA 成为了大模型微调的热门方法。


```python
# 需要9G左右显存
python run_finetune.py ./config/qwen/lora_argument_0p5b.json
```

除了 LoRA、Prefix Tuning 外，PaddleNLP 还支持 LoKr、VeRA、MoRA、ReFT、rsLoRA、LoRA+、PiSSA、MoSLoRA 等多种精调算法，更多大模型精调使用文档、训练细节和效果请参见[大模型精调教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/finetune.md)。
