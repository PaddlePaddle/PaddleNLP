# LLM全流程工具链
我们提供了模型预训练、精调（SFT、LoRA、Prefix Tuning）、量化、推理、部署全流程通用脚本，开发者可以根据自己的需求定制化自己的大语言模型。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/72c5b613-dca2-4f99-9a35-4dee089ee71c">
</div>

## 工具链特性
### 1. 飞桨 4D 并行
飞桨 4D 并行是PaddleNLP中搭载的一个重要模块，用于实现多卡多机训练大语言模型。PaddleNLP 中的全场景 Trainer 对分布式训练配置做了封装支持，比如：
* 开箱即用4D并行配置，涵盖数据并行，张量并行，流水线并行及 Sharding 并行
* 屏蔽多硬件编程复杂性
* 预训练、精调、对齐复用
* 统一日志、打点、监控
用户输入模型，数据集，就可以使用Trainer API高效快速的实现多卡预训练、微调等任务,以提供更高效、更稳定的训练体验。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a2f0261d-7f76-4faf-ae01-cc9d37d5fcc0">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Megatron 预训练性能比对
     </font>
</div>

### 2. 高效精调
* 高效精调（PEFT）是大模型微调中一个重要组成，可支持 LoRA，Prefix 两种主流精调方法。在此基础上 PEFT 还支持低比特和分布式并行策略。PEFT中提供将16位浮点数的主干模型转化为4比特或8比特的量化模型。
* Intokens 策略：高效精调模块还搭载了 Intokens 策略，对变长数据流做了比较极致的优化，大大减少了pad token的占比，从而提升了有效token率和训练的效率。

支持的微调算法：
* [低秩适配矩阵微调（LoRA）](https://arxiv.org/abs/2106.09685)
* [Prefix Tuning](https://aclanthology.org/2021.acl-long.353/)
* [Q-LoRA(低比特)](https://arxiv.org/pdf/2305.14314.pdf)
* WINT-LoRA(低比特): Weight Only INT8量化策略和 LoRA 的结合

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/f3a12366-3b7d-428c-9466-90e59ef3a3ed">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Huggingface Transformers 微调性能比对
     </font>
</div>

### 3. 模型参数量化 (Quantization)
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

### 4. 高性能推理 
飞桨PaddleNLP框架高性能推理模块，相比与普通的静态图导出推理方式，有更高的吞吐。同时它隐藏了底层实现的细节，使用户能够开箱即用。高性能推理模块主要包含：
* 算子融合策略将大模型的transformer结构上做了全环节的算子融合， 比如每个大模型都不可或缺的Self-Attetnion层，FeedForward层，甚至最后的Optimizer都做了融合。
* 动态插入策略在推理时能及时换出推理结束的样本并插入还未推理的样本，极大加快并行推理的速度。

## 快速开始

| Model | Pretrain | SFT | LoRA | Prefix Tuning | Generation | Quantization | weight convert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [LLaMA/LLaMA2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  | ✅  |
| [Baichuan/Baichuan2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅ | ✅  | ✅  |
| [ChatGLM-6B](./chatglm) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ❌  |
| [ChatGLM2/ChatGLM3](./chatglm2) |  ❌  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  | ✅  |
| [Qwen](./qwen) | ✅ | ✅ | ✅ | ✅ |  ✅ | 🚧 | ✅  |j
| [Bloom](./bloom) | ❌  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |  ✅  |  🚧  | ✅   | 🚧 | ✅  |
| [OPT](./opt) | 🚧 | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |
| [GLM](./glm) | ❌  | ✅ | ✅ | 🚧 |  ✅ | 🚧 | ✅  |


* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

### 1. 预训练
[LLaMA v1/v2](./llama)、[GPT-3](./gpt-3)、[Qwen](./qwen) 目录中提供了模型预训练的数据准备和训练细节，整个预训练过程搭载了飞桨统一全场景分布式 Trainer，可实现预训练 4D 并行加速。具体预训练配置细节[参考](./docs/pretrain.md)
```
# 千问模型预训练启动脚本
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json

```

### 2. 精调
目前精调统一脚本只已支持大部分主流模型，详见对应模型目录。更多LoRA、Prefix Tuning请参见[精调文档](./docs/finetune.md)。除此以外还支持了高效多轮对话模式精调，具体的配置可看[多轮对话文档](./docs/chat_template.md)

#### 2.1. 精调训练数据格式

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```


#### 2.2. SFT
```bash
# 张量并行分布式训练（常用）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_argument.json

# ChatGLM2、OPT不支持张量并行，默认使用Sharding策略（Paddle 2.5.1支持Sharding Stage2，Sharding Stage3需要使用Paddle develop版本）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm2/sft_argument.json

# 张量并行&流水线并行分布式训练
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_pp_argument.json
```

#### 2.3. LoRA
```bash
# 单卡LoRA训练
python  finetune_generation.py ./llama/lora_argument.json

# 张量并行分布式训练
# 只需将lora_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/lora_argument.json
```

#### 2.4. Prefix Tuning
```bash
# 单卡训练
python  finetune_generation.py ./llama/pt_argument.json

# 张量并行分布式训练
# 只需将pt_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/pt_argument.json
```

### 3. 量化
量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。下面我们提供GPTQ和PaddleSlim自研的PTQ策略，分别实现WINT4和W8A8量化。更多技术细节详见[量化文档](./docs/quantization.md)
#### 3.1 PTQ 量化

```
python  finetune_generation.py ./llama/ptq_argument.json
```

#### 3.2 GPTQ 量化

```
python  finetune_generation.py ./llama/gptq_argument.json
```

### 4. 模型推理

此外 PaddleNLP 还提供了高性能推理模型，从而加速 LLM 模型的部署落地，详细文档请看：[Inference Model](./docs/inference.md)

#### 4.1 动态图推理

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

#### 4.2 静态图推理

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

#### 4.3 Inference Model 高性能推理

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



### 5. 服务部署

#### 5.1 环境准备

- python >= 3.8
- gradio
- flask

#### 5.2 Flask & Gradio UI服务化部署

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

### 6. 权重自动转换
PaddleNLP 提供了可自动将 PyTorch 相关的权重转化为 Paddle 权重的接口，代码如下：

```python
from paddlenlp.transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("/path/to/pytorch/model", convert_from_torch=True, dtype="float16")
```

> dtype 为转化权重的真实 dtype 数据类型，通常为：float16, bloat16 和 float32。

以上代码可自动加载 pytorch 权重并转化为对应 paddle 权重保存在 `/path/to/pytorch/model` 目录下。
转换 torch 分片权重等方法具体参考[文档](./docs/torch2paddle.md)

