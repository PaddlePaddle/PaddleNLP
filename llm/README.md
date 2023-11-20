# 飞桨大语言模型工具链

飞桨大语言模型工具链基于飞桨4D分布式并行技术开发，旨在提供高性能、灵活易用大语言模型全流程开发能力，覆盖开发、预训练、精调、压缩、推理、部署的全流程。

| Model | Pretrain | SFT | LoRA | Prefix Tuning | Generation | Quantization | weight convert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [LLaMA v1/v2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  | ✅  |
| [ChatGLM-6B](./chatglm) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ❌  |
| [ChatGLM v2/v3](./chatglm2) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ✅  |
| [Bloom](./bloom) | ❌  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |  ✅  |  🚧  | ✅   | 🚧 | ✅  |
| [OPT](./opt) | 🚧 | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |
| [GLM](./glm) | ❌  | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |
| [Qwen](./qwen) | ✅ | ✅ | ✅ | ✅ |  ✅ | 🚧 | ✅  |


* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

# LLM全流程导览图
我们提供了模型预训练、精调（SFT、LoRA、Prefix Tuning）、量化、推理、部署全流程脚本，开发者可以根据自己的需求定制化自己的大语言模型。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/009bbb4e-baee-4c4a-a52e-94ac44c73c90">
</div>

<div align="center">
    <font size ="1">
    LLM全流程工具流程图（上图：PaddleNLP 2.6进展 下图：最终目标）
     </font>
</div>

# LLM模块特性(Features)
## 1.统一全场景分布式 Trainer
Trainer是PaddleNLP中的一个重要模块，用于实现自然语言处理任务的训练过程。Trainer 对通用训练配置做了封装支持，比如：
* 开箱即用4D并行配置，涵盖数据并行，张量并行，流水线并行及 Sharding 并行
* 屏蔽多硬件编程复杂性
* 预训练、精调、对齐复用
* 统一日志、打点、监控
用户输入模型，数据集，就可以使用Trainer API高效快速的实现预训练、微调等任务。为了满足不同用户的需求，Trainer支持通用分布式能力和混合并行分布式能力，以提供更高效、更稳定的训练体验。

## 2.PEFT (参数高效微调)
参数高效微调技术（PEFT）是大模型微调中一个重要组成，在使用较少的训练步骤和计算资源的情况下实现更好的性能。PaddleNLP将PEFT工作与现有的低比特和飞桨并行策略，对多种主流预训练模型进行适配，满足用户在不同硬件资源对多种尺寸的预训练模型微调、量化需求，有效降低模型训练成本。
支持的微调算法：
* 全参数微调（SFT）
* 低秩适配矩阵微调（LoRA）：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
* Prefix Tuning：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)

## 3.模型参数量化 (Quantization)
飞桨PaddleNLP框架内的量化模块专门为主流大模型提供量化功能。量化是一种将浮点数转换为低精度的整数表示的技术，可以显著减少模型的存储空间和计算资源需求，同时加速模型的推理速度。PaddleNLP的量化模块提供了一系列的工具和功能，使用户能够轻松地进行模型量化。它支持不同的量化方法:
* PTQ
* GPTQ

这些方法根据所需的样本数据和计算资源有所不同，用户可以根据自己的需求选择适合的方法。

## 4.Predictor推理模块
飞桨PaddleNLP框架内的Predictor推理模块是用于在训练完成后对模型进行推理的模块。Predictor推理模块提供了一套高效、便捷的推理工具，旨在加速模型的部署和推理过程。它隐藏了底层实现的细节，使用户能够轻松地将训练好的模型应用于实际场景中。
该推理模块支持以下几种推理方式：
* 动态图推理（Dynamic Graph)
* 静态图推理（Static Graph）
* 融合组网推理（Inference Model)


# 开用!
## 1. 预训练
[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3)、[Qwen](./qwen) 目录中提供了模型预训练的数据准备和训练细节，整个预训练过程搭载了飞桨统一全场景分布式 Trainer，可实现预训练 4D 并行加速。
```
# 千问模型预训练启动脚本
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json

```

## 2. SFT精调
目前精调统一脚本只已支持大部分主流模型，详见对应模型目录。更多LoRA、Prefix Tuning请参见[PEFT文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/peft.md)。

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

当前开源Chat 类型模型越来越多，PaddleNLP 已经集成了 [Llama](./llama/README.md)、[Qwen](./qwen/README.md)、[ChatGLM](./chatglm/README.md) 等系列模型，也支持[多轮对话 Prompt Template 推理](https://paddlenlp.readthedocs.io/zh/latest/get_started/chat_template.html)，只需要调用`apply_chat_template` 函数即可构造将对话历史和用户最新 query 按照模型指定规则拼接到一起，实现不同模型的定制化 Prompt 规则推理。

此外多轮对话训练精调的应用场景也是越来越多，不同模型的多轮对话模板构造规则都不一致，为了在训练侧标准化前处理上的区别，设计了`chat_template`来解决此问题。

### 7.1 如何构造 `chat_template`

只需要添加一个 chat_template 的配置即可为该模型添加相应的多轮对话精调训练支持，以`qwen-14b-chat`配置文件

> 以下配置参考：https://huggingface.co/Qwen/Qwen-14B-Chat/blob/main/qwen_generation_utils.py#L119

```json
{
    "system": "You are a helpful assistant.",
    "conversation": ["\n<|im_start|>user\n{{user}}<|im_end|>\n<|im_start|>assistant\n", "{{bot}}<|im_end|>"],
    "query": "\n<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n",
}
```

注意点：

1. 配置文件名默认为：`chat_template.json`。
1. 对于 `chat_template.json`配置文件 `query`和`conversation`字段为必选项，且内容非常类似，主要是为应对推理和训练两种场景设计使用：query 只用于推理，query 和 conversation 用于训练。
1. 由于训练和推理过程中会在文本中添加 独特token 标记，其中包括 bos_token, eos_token 以及像上述的 <|im_start|> 自定义标记等，故基于 chat_template 的分词是不会添加 special_token，也就是说 tokenizer 中的 `add_special_tokens` 参数始终要设置为 `False`。
1. `conversation`字段为数组，且必须为两个元素，分别对应着 User 和 Bot 的对话内容，前者在训练过程中不参与 loss 的计算，后者的参与 Loss 的计算。
1. 在训练过程中，system 文本的长度不可大于 `max_length`，当对话轮次只有一轮时，基于 token 长度来截断，伪代码为：`(system_tokens + conversation_tokens)[:max_length]`；否则将基于对话轮次来截断，详细来说就是在计算训练 token 总长度时，会从后往前计算每一轮的对话长度，如果截止当前的对话（包含 User 和 Bot 的总 tokens 长度）token 长度大于 `max_length`，此时将当前对话轮次给截断，也不计算后续历史对话数据，直接构造训练数据。
1. 在训练过程中，system 必须存在，不能被截断。

#### 7.2 如何使用 `chat_template` 进行训练

以`qwen-14b-chat`基座模型为例，首先需要调整的是训练数据部分，需要保证如下格式：

```json
{"src": ["user-1", "user-2", ..., "user-n"], "tgt": ["bot-1", "bot-2", ..., "bot-n"]}
...
```

其次就是将构造好的`chat_template.json`文件传入到 `llm/finetune_generation.py` 模块当中：

* 使用模型自带chat-template

> 并不是所有的模型支持chat-template，PaddleNLP 正在全力支持，可根据是否有下载 `chat_template.json` 文件来判断该模型是否支持 chat-template。

```shell
python finetune_generation.py ... --model_name_or_path qwen/qwen-7b-chat --chat_template qwen/qwen-7b-chat
```

此时当 `chat_template` 参数和 `model_name_or_path` 参数一致时，此时将默认使用模型自带的chat_template.json` 文件。

* 使用自定义 chat-template

```shell
python finetune_generation.py ... --chat_template ./qwen_14b_chat_template.json
```

1. 当 `chat_template` 参数和 `model_name_or_path` 参数一致时，此时将默认使用模型自带的 `chat_template.json` 文件。
1. 当 `chat_template` 参数为文件路径时，此时将使用该文件中的 `chat_template` 配置。
1. 当 `chat_template` 参数为空时，此时不使用 `chat_template` 配置进行训练。

## 8. 模型推理

此外 PaddleNLP 还提供了高性能推理模型，从而加速 LLM 模型的部署落地，详细文档请看：[Inference Model](./inference.md)

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

### 8.3 Inference Model 推理

此外 PaddleNLP 还提供了高性能推理模型，从而加速 LLM 模型的部署落地，详细文档请看：[Inference Model](./inference.md)

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

量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。下面我们提供GPTQ和PaddleSlim自研的PTQ策略，分别实现WINT4和W8A8量化。更多技术细节详见[量化策略详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md)

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

### 11.1 合并 Pytorch 分片权重

当前 PaddleNLP 仅支持转化单个 Pytorch 权重：`pytorch_model.bin`文件。所以当Pytorch 权重为分片权重时，需要将其合并，合并脚本如下所示：

```python
import torch, os
state_dict = {}

files = [file for file in os.list("./path/to/pytorch/weight") if file.startswith("pytorch_model-")]

for file in files:
    state_dict.update(torch.load(file))

torch.save(state_dict, "pytorch_model.bin")
```
