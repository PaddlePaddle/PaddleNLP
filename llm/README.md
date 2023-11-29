#  飞桨大模型全流程工具链

飞桨大模型套件秉承了一站式体验、性能极致、生态兼容的设计理念，旨在提供业界主流大模型预训练、精调（含SFT、PEFT）、量化、推理等全流程统一工具链， 帮助开发者低成本、低门槛、快速实现大语言模型定制化。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/17710a9a-d972-4772-9bf4-19ff938b5fe9">
</div>


##  🚣‍♂️ 飞桨大模型工具链特性 🚣‍♂️

-  **飞桨4D并行分布式策略**。 PaddleNLP Trainer 封装支持了飞桨4D并行配置（数据并行、张量并行、流水线并行、Sharding 并行），屏蔽多硬件编程复杂性，用户可以修改Trainer配置组合多种预训练或精调过程的分布式策略，获得更高效、更稳定的训练体验。

-  **高效精调策略**。飞桨大模型套件提供SFT、PEFT等多种精调策略，搭载自研Intokens策略有效减少了pad token的占比，提高模型训练效率。独创PEFT结合低比特和分布式并行策略，大幅降低大模型精调硬件门槛。


- **大模型无损量化**。工具链内置了PaddleSlim 团队自研的自适应Shift-SmoothQuant的A8W8量化算法和业界主流GPTQ的W4量化算法，实现了主流大模型的无损量化，有效加速模型推理。


- **高性能推理**。工具链高性能推理模块内置动态插入和全环节算子融合策略，极大加快并行推理的速度。同时隐藏了底层实现的细节，实现高性能推理开箱即用。


##  🧨 支持模型列表 🧨

| Model | Pretrain | SFT | LoRA | Prefix Tuning |  Quantization | weight convert |
| --- | --- | --- | --- | --- | --- |  --- |
| [LLaMA/LLaMA2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅  | ✅  |
| [Baichuan/Baichuan2](./llama) | ✅  | ✅ | ✅ | ✅ | ✅  | ✅  |
| [ChatGLM-6B](./chatglm) |  ❌  |  ✅  |    ✅  |  ✅  |  ✅  | ❌  |
| [ChatGLM2/ChatGLM3](./chatglm2) |  ❌  |    ✅  |  ✅  |  ✅  |  ✅  | ✅  |
| [Qwen](./qwen) | ✅ | ✅ | ✅ | ✅ |  🚧 | ✅  |j
| [Bloom](./bloom) | ❌  | ✅ | ✅ |  ✅ | ✅ | ✅  |
| [GPT-3](./gpt-3) |   ✅  |  ✅  |    🚧  | 🚧  | 🚧 | ✅  |
| [OPT](./opt) | 🚧 | ✅ | ✅ | 🚧 |  🚧 | ✅  |
| [GLM](./glm) | ❌  | ✅ | ✅ | 🚧 |   🚧 | ✅  |

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported


##  🚀 快速开始 🚀



### 1. 预训练
PaddleNLP将飞桨4D并行策略加入到Trainer API中， 用户只需修改Trainer配置即可使用不同的分布式策略。目前工具链提供[LLaMA/LLaMA2](./llama)、[GPT-3](./gpt-3)、[Qwen](./qwen) 等模型预训练功能，更多模型支持持续更新中。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/a2f0261d-7f76-4faf-ae01-cc9d37d5fcc0">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Megatron 预训练性能比对
     </font>
</div>

单机8卡qwen预训练启动命令参考：
```
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./qwen/pretrain_argument_stage2.json
```
更多模型预训练数据准备、环境准备、参数配置细节请参考[预训练文档](./docs/pretrain.md)。


### 2. 精调
目前精调统一脚本只已支持大部分主流模型，详见对应模型目录。更多LoRA、Prefix Tuning请参见[精调文档](./docs/finetune.md)。除此以外还支持了高效多轮对话模式精调，具体的配置可看[多轮对话文档](./docs/chat_template.md)

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/b2b4db4f-0cf3-4d28-989c-e3c00d24f397">
</div>
<div align="center">
    <font size ="1">
    飞桨与 Huggingface Transformers 微调性能比对
     </font>
</div>

#### 2.1. 环境准备
- paddlepaddle-gpu >= 2.5.1
- paddlenlp >= 2.6.1
- tiktoken (仅 Qwen 需要)

#### 2.2. 精调训练数据格式

为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```



#### 2.3. SFT
```bash
# 张量并行分布式训练（常用）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_argument.json

# ChatGLM2、OPT不支持张量并行，默认使用Sharding策略（Paddle 2.5.1支持Sharding Stage2，Sharding Stage3需要使用Paddle develop版本）
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./chatglm2/sft_argument.json

# 张量并行&流水线并行分布式训练
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py ./llama/sft_pp_argument.json
```

#### 2.4. LoRA
```bash
# 单卡LoRA训练
python  finetune_generation.py ./llama/lora_argument.json

# 张量并行分布式训练
# 只需将lora_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/lora_argument.json
```

#### 2.5. Prefix Tuning
```bash
# 单卡训练
python  finetune_generation.py ./llama/pt_argument.json

# 张量并行分布式训练
# 只需将pt_argument.json中tensor_parallel_degree修改为2
# 并用 -m paddle.distributed.launch --gpus "0,1"指定一下卡数
python  -u  -m paddle.distributed.launch --gpus "0,1"  finetune_generation.py ./llama/pt_argument.json
```

### 3. 量化
大模型量化将16位、32位浮点数的模型参数或激活量化为4位或8位整数能够有效降低模型存储空间和计算资源需求，同时加速推理速度。工具链量化算法包含：
- **PTQ**。PaddleSlim 团队自研的自适应Shift-SmoothQuant量化算法，在[SmoothQuant](https://arxiv.org/abs/2211.10438)和[Outlier Suppression+](https://arxiv.org/abs/2304.09145)基础上
新增PieceWiseSearch参数搜索算法，对模型权重和激活分布进行调整，减少后续A8W8 PTQ量化损失。


- **GPTQ**。[GPTQ](https://arxiv.org/abs/2210.17323)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/969b62db-9692-4d50-b91a-85cff305d153">
</div>
<div align="center">
    <font size ="1">
    飞桨量化算法效果展示
     </font>
</div>


```
# PTQ 量化启动命令参考
python  finetune_generation.py ./llama/ptq_argument.json

# GPTQ 量化启动命令参考
python  finetune_generation.py ./llama/ptq_argument.json
```

更多技术细节和模型量化使用详见[量化文档](./docs/quantization.md)。


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

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fb248224-0ad1-4d6a-a1ca-3a8dd765c41d">
</div>
<div align="center">
    <font size ="1">
    推理部署性能业界领先
     </font>
</div>


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
