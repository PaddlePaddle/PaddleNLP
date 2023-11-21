# LLM全流程导览图
我们提供了模型预训练、精调（SFT、LoRA、Prefix Tuning）、量化、推理、部署全流程脚本，开发者可以根据自己的需求定制化自己的大语言模型。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a12b8c20-02f3-4a01-8c4b-7020f808b655">
</div>

# LLM模块特性(Features)
## 1. 预训练
### 1.1. 统一全场景分布式 Trainer
Trainer是PaddleNLP中的一个重要模块，用于实现自然语言处理任务的训练过程。Trainer 对通用训练配置做了封装支持，比如：
* 开箱即用4D并行配置，涵盖数据并行，张量并行，流水线并行及 Sharding 并行
* 屏蔽多硬件编程复杂性
* 预训练、精调、对齐复用
* 统一日志、打点、监控
用户输入模型，数据集，就可以使用Trainer API高效快速的实现预训练、微调等任务。为了满足不同用户的需求，Trainer支持通用分布式能力和混合并行分布式能力，以提供更高效、更稳定的训练体验。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a2f0261d-7f76-4faf-ae01-cc9d37d5fcc0">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Megatron 预训练性能比对
     </font>
</div>

## 2. 精调
### 2.1. 全参数微调（SFT）
对模型的所有参数进行微调的方法。它涉及调整模型中的所有参数，而不仅仅是部分参数。这意味着在微调过程中，每个参数都会根据训练数据的反馈进行更新和调整。它能够充分利用预训练模型的知识，并且通过全面调整参数，有可能实现更好的任务性能。然而，这种方法也可能带来过拟合的风险，特别是当训练数据有限时。
### 2.2. 分布式低比特PEFT (参数高效微调)
参数高效微调技术（PEFT）是大模型微调中一个重要组成，可支持 LoRA，Prefix 两种主流精调方法。在此基础上 PEFT 还支持低比特和分布式并行策略。

#### 2.2.1. PEFT低比特策略
LoRA和PrefixTuning相比于全量参数大大降低了所需的显存资源，但对于百亿级别的模型对训练资源仍然要求很高。在PEFT训练中尤其是百亿参数界别模型，占比最大的是加载模型本身权重所占的资源，为了减少显存资源占用，PEFT中提供将16位浮点数的主干模型转化为4比特或8比特的量化模型，只有当权重参与计算时才将低比特的主干模型反量化为浮点数模型。这在保存原有性能和精度的前提下，有效减少了训练所需资源。对于主干模型位16位浮点数，量化为8比特显存资源占用减少一半，量化为4比特显存资源占用减少3/4。PaddleNLP中提供量化为INT4、INT8、NF4、FP4等多种低比特数据类型。

#### 2.2.2. PEFT中的分布式并行策略
对于千亿参数级别的模型，PEFT配合低比特策略并不能在单卡训练。PaddleNLP中支持上述所有PEFT策略包含低比特策略使用数据并行（data parallel）、张量并行（tensor parallel）和流水线并行（pipeline parallel）策略。PEFT、低比特策略、分布式能力三者组合，PaddleNLP将模型微调拓展到单机千亿参数级别。

支持的微调算法：
* 低秩适配矩阵微调（LoRA）：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
* Prefix Tuning：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
* Q-LoRA(低比特): [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)
* WINT-LoRA(低比特): Weight Only INT8量化策略和 LoRA 的结合

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/f3a12366-3b7d-428c-9466-90e59ef3a3ed">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Huggingface Transformers 微调性能比对
     </font>
</div>

## 3. 量化
### 3.1. 模型参数量化 (Quantization)
飞桨PaddleNLP框架内的量化模块专门为主流大模型提供量化功能。量化是一种将浮点数转换为低精度的整数表示的技术，可以显著减少模型的存储空间和计算资源需求，同时加速模型的推理速度。飞桨内置了两种业界主流的量化算法：GPTQ和SmoothQuant。为了满足更多量化的需求，飞桨还开源了PaddleSlim团队自研的自适应Shift-SmoothQuant算法。该算法解决了SmoothQuant无法处理的某些量化场景。
* A8W8: SmoothQuant, Shift-SmoothQuant
* W4: GPTQ

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/969b62db-9692-4d50-b91a-85cff305d153">
</div>
<div align="center">
    <font size ="1">
    飞桨量化性能比对
     </font>
</div>

这些方法根据所需的样本数据和计算资源有所不同，用户可以根据自己的需求选择适合的方法。

## 4. 推理
### 4.1. Predictor统一推理模块
飞桨PaddleNLP框架内的Predictor统一推理模块是用于在训练完成后对模型进行推理的模块。Predictor推理模块提供了一套高效、便捷的推理工具，旨在加速模型的部署和推理过程。它隐藏了底层实现的细节，使用户能够轻松地将训练好的模型应用于实际场景中。
该推理模块支持以下几种推理方式：
* 动态图推理（Dynamic Graph)
* 静态图推理（Static Graph）
* 高性能推理（Inference Model)

# LLM模块性能数据


# LLM模型支持工具链

| Model | Pretrain | SFT | LoRA | Prefix Tuning | Generation | Quantization | weight convert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [LLaMA v1/v2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  | ✅  |
| [ChatGLM-6B](./chatglm) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ❌  |
| [ChatGLM v2/v3](./chatglm2) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ✅  |
| [Bloom](./bloom) | ❌  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |  ✅  |  🚧  | ✅   | 🚧 | ✅  |
| [OPT](./opt) | 🚧 | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |
| [GLM](./glm) | ❌  | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |
| [Qwen](./qwen) | ✅ | ✅ | ✅ | ✅ |  ✅ | 🚧 | ✅  |j


* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported


# 快速开始
## 1. 预训练
[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3)、[Qwen](./qwen) 目录中提供了模型预训练的数据准备和训练细节，整个预训练过程搭载了飞桨统一全场景分布式 Trainer，可实现预训练 4D 并行加速。具体预训练配置细节[参考](./docs/pretrain.md)
```
# 千问模型预训练启动脚本
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json

```

## 2. SFT精调
目前精调统一脚本只已支持大部分主流模型，详见对应模型目录。更多LoRA、Prefix Tuning请参见[精调文档](./docs/finetune.md)。

```bash
# 张量并行分布式训练（常用）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_argument.json

# ChatGLM2、OPT不支持张量并行，默认使用Sharding策略（Paddle 2.5.1支持Sharding Stage2，Sharding Stage3需要使用Paddle develop版本）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm2/sft_argument.json

# 张量并行&流水线并行分布式训练
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_pp_argument.json
```

## 3. LoRA精调
```bash
# 单卡LoRA训练
python  finetune_generation.py ./llama/lora_argument.json

# 张量并行分布式训练
# 只需将lora_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/lora_argument.json
```

## 4. Prefix精调
```bash
# 单卡训练
python  finetune_generation.py ./llama/pt_argument.json

# 张量并行分布式训练
# 只需将pt_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/pt_argument.json
```
## 5. 张量并行参数合并
我们使用张量并行（TP，Tensor Parallelism）训练过程中，为了节省TP参数合并时间通常在中间checkpoint将参数存储为多个TP参数分片，可以使用提供的分片合并参数脚本进行参数合并。

```bash
# model_name_or_path为本地 TP 分片参数路径
python merge_tp_params.py \
    --model_name_or_path ./checkpoints/llama_sft_ckpts/checkpoint-100
```

## 6. LoRA 参数合并

为了后续的**压缩**和**静态图推理**方便，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。
```
python merge_lora_params.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --lora_path ./checkpoints/llama_lora_ckpts
```
<details><summary>&emsp; 脚本参数介绍</summary><div>

- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `merge_model_path`: 必须，合并参数后保存路径，默认为None。
- `device`: 运行环境，默认为gpu。
</div></details>


## 7. 多轮对话精调
PaddleNLP LLM 模块现已支持多轮对话模式的精调节，用户只需提前准备好 chat_template.json 文件便可一键启动多轮对话模式的 SFT，LoRA，Prefix 精调。具体chat_template.json的配置可看[多轮对话文档](./docs/chat_template.md)

* 使用模型权重自带 chat_template
```shell
python finetune_generation.py ... --model_name_or_path qwen/qwen-7b-chat --chat_template qwen/qwen-7b-chat
```

* 使用自定义 chat-template

```shell
python finetune_generation.py ... --chat_template ./qwen_14b_chat_template.json
```

## 8. 模型推理

此外 PaddleNLP 还提供了高性能推理模型，从而加速 LLM 模型的部署落地，详细文档请看：[Inference Model](./docs/inference.md)

### 8.1 动态图推理

```shell
# 预训练&SFT动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --data_file ./data/dev.json \
    --dtype float16

# LoRA动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --lora_path ./checkpoints/llama_lora_ckpts

# Prefix Tuning动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --data_file ./data/dev.json \
    --prefix_path ./checkpoints/llama_pt_ckpts
```

### 8.2 静态图推理

```shell
# 首先需要运行一下命令将动态图导出为静态图
# LoRA需要先合并参数，详见3.7LoRA参数合并
# Prefix Tuning暂不支持
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16


# 静态图模型推理
python predictor.py \
    --model_name_or_path inference \
    --data_file ./data/dev.json \
    --dtype float16 \
    --mode static
```

### 8.3 Inference Model 高性能推理

此外 PaddleNLP 还提供了高性能推理模型，从而加速 LLM 模型的部署落地

支持的模型列表如下所示：

| Model                       | Inference Model | PTuning | Wint8 | PTQ |
|-----------------------------|-----------------|---------|-------|-----|
| [LLaMA1/2](./llama)         | ✅               | ✅       | ✅     | ✅   |
| [ChatGLM](./chatglm)        | ✅               | ✅       | ✅     | ❌   |
| [ChatGLM2](./chatglm2)      | ✅               | ❌       | ❌     | ❌   |
| [BaiChuan1](./baichuan)     | ✅               | ✅       | ✅     | ✅   |
| [BaiChuan2-7B](./baichuan)  | ❌               | ❌       | ❌     | ❌   |
| [BaiChuan2-13B](./baichuan) | ✅               | ✅       | ✅     | ✅   |
| [Bloom](./bloom)            | ✅               | ✅       | ✅     | ❌   |
| [GPT-3](./gpt-3)            | ✅               | ❌       | ❌     | ❌   |
| [Qwen](./qwen)              | ❌               | ❌       | ❌     | ❌   |


## 9. 量化

量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。下面我们提供GPTQ和PaddleSlim自研的PTQ策略，分别实现WINT4和W8A8量化。更多技术细节详见[量化策略详细教程](./docs/quantization.md)

### 9.1 环境安装
- PaddleSlim develop版本
- PaddlePaddle develop版本

### 9.2 数据准备

量化中默认使用训练集作为校正（Calibartion）数据集，开发集作为评估数据集。如果希望使用其他数据作为校正数据集，则在数据目录下新增`quant.json`文件，文件格式请参照精调训练数据格式。

### 9.3 PTQ 量化

```
python  finetune_generation.py ./llama/ptq_argument.json
```

### 9.4 GPTQ 量化

```
python  finetune_generation.py ./llama/gptq_argument.json
```

## 10. 服务部署

### 10.1 环境准备

- python >= 3.8
- gradio
- flask

### 10.2 Flask & Gradio UI服务化部署

我们提供了一套简单易用的UI服务化部署脚本:


```
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```

<details><summary>&emsp; 脚本参数介绍</summary><div>

- `port`: Gradio UI 服务端口号，默认8011。
- `flask_port`: Flask服务端口号，默认8010。
- 其他参数请参见动态图推理中参数。

</div></details>

## 11. 转化 Pytorch 权重
PaddleNLP 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，代码如下：

```python
from paddlenlp.transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

> dtype 为转化权重的真实 dtype 数据类型，通常为：float16, bloat16 和 float32。

以上代码可自动加载 pytorch 权重并转化为对应 paddle 权重保存在 `/path/to/pytorch/model` 目录下。
转换 torch 分片权重等方法具体参考[文档](./docs/torch2paddle.md)

