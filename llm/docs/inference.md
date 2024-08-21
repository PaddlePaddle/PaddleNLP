# 大模型推理教程

PaddleNLP以一站式体验、极致性能为设计理念，实现大模型的快速推理：

* 提供常用模型推理，方便用户快速验证模型推理效果。
* 提供高性能推理，内置动态插入和全环节算子融合策略，极大加快并行推理的速度。
* 实现BlockAttention，在保持高性能推理和动态插入的基础上可以动态地为CacheKV分配存储空间，极大地节省显存。


## 1. 常用推理

PaddleNLP 为常用模型提供了动态图推理和静态图推理两种方式（包含LoRA、PrefixTuning），用户能够根据自己的需求灵活的选择最适合的推理方式，从而快速验证模型的推理效果。命令参数详情请参考模型页面介绍。

### 1.1 动态图推理 

动态图推理是一种灵活的推理方式：

- 即时执行：每个操作都会立即执行，便于调试和可视化。
- 灵活性高：支持动态变化的网络结构。

### **1.2 静态图推理**

静态图推理是一种高效的推理方式（在运行静态图之前需将动态图转为静态图）：

- 预先编译：整个计算图在执行前被完整编译，有利于全局优化，性能通常优于动态图。
- 部署便利：更适合产品化部署，特别是在对性能要求较高的场景。

## 2. 高性能推理

PaddleNLP提供了常用模型的高性能推理，在推理过程中动态地插入或调整计算图中的节点或操作，同时在推理过程的各个阶段实现了算子融合技术，减少内存访问和计算开销，从而全面提升推理性能。高性能推理的内置动态插入和全环节算子融合策略，隐藏了底层实现的细节，实现了开箱即用高性能并行推理能力。同时为了进一步提升推理的吞吐，我们基于PageAttention的思想设计并实现了BlockAttention，将 KV 缓存划分为固定大小的块（blocks），从而可以更灵活的分配cachekv。在保持高性能推理和动态插入的基础上可以动态地为cachekv分配存储空间，极大地节省显存，从而在同一时刻处理更多的query以获得吞吐的提升。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/42174b47-f765-48d6-9fef-907b69bf6706">
</div>
<div align="center">
    <font size ="1">
    飞桨高性能推理算子融合示意图
     </font>
</div>

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/616b3fc5-b9b2-4b10-a5c8-2f892a65ae6b">
</div>
<div align="center">
    <font size ="1">
    动态插入图解 & 飞桨高性能模型推理性能图
     </font>
</div>

### 2.1 模型支持

PaddleNLP 中已经添加高性能推理模型相关实现，支持：

| Model                            | FP16/BF16 | WINT8 | INT8-A8W8C16 | FP8-A8W8C16 | INT8-A8W8C8 | FP8-A8W8C8 |
|----------------------------------|-----------|-------|--------------|-------------|-------------|------------|
| [LLaMA1/2/3/3.1](../config/llama)| ✅        | ✅    | ✅        | ✅       | ✅    | ❌        |
| [Qwen1.5/2](../config/qwen)      | ✅        | ✅    | ✅        | ✅       | ✅    | ❌        |
| [Qwen-Moe]()       | ✅        | 🚧    | ❌        | ❌       | ❌    | ❌        |
| [Mixtral]()        | ✅        | 🚧    | ❌        | ❌       | ❌    | ❌        |
| [ChatGLM](../config/chatglm)     | ✅        | ✅    | ❌        | ❌       | ❌    | ❌        |
| [ChatGLM2](../config/chatglm2)   | ✅        | ❌    | ❌        | ❌       | ❌    | ❌        |
| [Bloom](../config/bloom)         | ✅        | ✅    | ❌        | ❌       | ❌    | ❌        |
| [GPT-3](../config/gpt-3)         | ✅        | ❌    | ❌        | ❌       | ❌    | ❌        |
| [BaiChuan-7B](../config/baichuan)   | ✅     | ✅     | 🚧       | ❌       | ❌    | ❌        |
| [BaiChuan2-7B](../config/baichuan)  | ✅     | ✅     | 🚧       | ❌       | ❌    | ❌        |
| [BaiChuan2-13B](../config/baichuan) | 🚧              | 🚧    | 🚧     | ❌       | ❌    | ❌        |

* ✅: Supported

* 🚧: In Progress

* ❌: Not Supported

* WINT8:指Weight-Only Quantization INT8，即对权重进行INT8量化的模型。

* INT8-A8W8C16:指使用PTQ对线性层的激活和权重都量化为INT8的模型。
* FP8-A8W8C16:指使用PTQ对线性层的激活和权重都量化为FP8的模型。
* INT8-A8W8C8:指使用PTQ对Cache KV、线性层的激活和权重都量化为INT8的模型。
* FP8-A8W8C8:指使用PTQ对Cache KV、线性层的激活和权重都量化为FP8的模型。

### 2.2 硬件&精度支持

PaddleNLP 提供了多种硬件平台和精度支持，包括：

| Precision      | Ada | Ampere | Turing | Volta | x86 CPU | XPU |
|----------------|-----|--------|--------|-------|---------|-----|
| FP32      | ✅ | ✅ | ✅ | ✅  | ✅  | ✅  |
| FP16      | ✅ | ✅ | ✅ |  ✅ | ✅  |  ✅ |
| BF16      | ✅ | ✅ | ✅ |  ❌ |  ❌ |  ❌ |
| INT8      | ✅ | ✅ | ✅ | ✅  | ✅ |   ✅|
| INT4      | ❌ | ❌ | ❌ | ❌  | ❌  | ❌  |
| FP8       | ✅ | ❌ | ❌ | ❌  | ❌  |  ❌ |


### 2.4 性能优化选项

#### Inference Model
`--inference_model` : 开启高性能推理模式

#### Block Attention
`--block_attn` : 为了进一步提升推理的吞吐，我们基于PageAttention的思想设计并实现了BlockAttention，在保持高性能推理和动态插入的基础上可以动态地为cachekv分配存储空间，极大地节省显存，从而在同一时刻处理更多的query以获得吞吐的提升。

#### Weight Only
`--quant_type weight_only_int8` : 即对权重进行INT8量化的模型。


#### PTQ
`--quant_type a8w8` :


#### Cache KV Quantization
`--quant_type a8w8c8` : 

`--cachekv_int8_type` : cachekv量化类型，支持dynamic和static两种模式。


### 2.5 性能分析选项

#### benchmark

`--benchmark` : 开启性能分析模式

#### src_length & max_length

`--src_length`: 模型输入上下文最大token长度，默认为1024。

`--max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。

#### batch_size

`--batch_size` : 批处理大小，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。





## 3. 环境准备
- [PaddlePaddle develop](https://github.com/PaddlePaddle/Paddle)
- PaddleNLP develop

git clone 代码到本地，即可开始。

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
# pip install ./PaddleNLP 使用develop版本
cd PaddleNLP/llm
# 到达运行目录
```

PaddleNLP 针对于Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能，如需使用高性能推理模式需要预先安装自定义算子库：

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP
#GPU设备安装自定义算子
cd ./paddlenlp/csrc && python setup_cuda.py install
#XPU设备安装自定义算子
cd ./paddlenlp/csrc/xpu/src && sh cmake_build.sh
```

## 4. 快速开始
安装PaddleNLP

```bash
cd PaddleNLP
python setup.py install
```

到达运行目录
```bash
cd PaddleNLP/llm
```


### 4.1. 常用推理
#### 4.1.1 动态图推理
```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --data_file ./data/dev.json --dtype float16
```
对于LoRA、PrefixTuning 模型只需额外传入相应的lora_path或prefix_path即可，如：--lora_path ./checkpoints/llama_lora_ckpts或--prefix_path ./checkpoints/llama_prefix_ckpts，详见推理参数介绍。


#### 4.1.2 静态图推理
```shell
# 静态图模型推理命令参考， LoRA需要先合并参数，Prefix Tuning暂不支持
# step1 : 静态图导出
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --output_path ./inference --dtype float16
# step2: 静态图推理
python ./predict/predictor.py --model_name_or_path ./inference --data_file ./data/dev.json --dtype float16 --mode static
```

### 4.2 高性能推理
#### 4.2.1 动态图推理

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn

# XPU设备动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --device xpu

# Weight Only Int8 动态图推理参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --quant_type weight_only_int8 --block_attn

# PTQ-A8W8推理命令参考
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn

# CacheKV 动态量化推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --cachekv_int8_type dynamic
```

#### 4.2.2 静态图推理
**step1：动转静**
```shell
# 动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --block_attn

# XPU设备动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --block_attn --device xpu

# Weight Only Int8 动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --quant_type weight_only_int8 --block_attn

# PTQ-A8W8动转静命令参考
python ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16 --block_attn

# CacheKV 动态量化动转静命令参考
python ./predict/export_model.py  --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --block_attn --cachekv_int8_type dynamic
```

**step2：静态图推理**
```shell
# 静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --block_attn

# XPU设备静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --block_attn --device xpu

# Weight Only Int8 静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --quant_type weight_only_int8 --block_attn

# PTQ-A8W8静态图推理命令参考
# 以下环境变量用于开启int8矩阵乘的算法选择以获得更快的推理速度，打开之后第一次执行会执行算法选择从而导致速度较慢。
export FLAGS_use_autotune=1
export FLAGS_cublaslt_exhaustive_search_times=10
export FLAGS_cache_inference_while_scope=1

python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --block_attn

# CacheKV 动态量化int8静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --cachekv_int8_type dynamic --block_attn
```
**Note**：
1. `quant_type`可选的数值有`weight_only_int8`，`weight_only_int4`，`a8w8`, `a8w8c8`。
2. `a8w8`推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型，需要额外的act和weight的scale校准表。
3. `cachekv_int8_type`可选`dynamic`和`static`两种，`static`需要额外的cache kv的scale校准表。



更多模型推理教程，参考[examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)

## 5. 推理参数介绍

- `model_name_or_path`: 必须，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。
- `batch_size`: 批处理大小，默认为8。该参数越大，占用显存越高；该参数越小，占用显存越低。
- `src_length`: 模型输入上下文最大token长度，默认为1024。
- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。
- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。
- `prefix_path`: Prefix Tuning参数和配置路径，对Prefix Tuning参数进行初始化，默认为None。
- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认为1，等价于贪心策略。
- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认为1.0，表示不起作用。
- `temperature`:“采样”策略中会对输出logit除以temperature。默认为1.0，表示不起作用。
- `data_file`:必须，待推理json文件，默认为None。
- `output_file`:保存推理结果文件名，默认为output.json。
- `device`: 运行环境，默认为gpu。
- `dtype`: 模型参数dtype，默认为None。如果没有传入`lora_path`、`prefix_path`则必须传入
- `model_type`: 初始化不同类型模型，gpt-3: GPTForCausalLM; ernie-3.5-se: Ernie35ForCausalLM; 默认为 None。
- `mode`: 使用动态图或者静态图推理，值为：[dynamic, static]，默认为 dynamic。
- `inference_model`: 是否使用Inference Model 推理，默认值为 False。
- `block_attn`: 是否使用Block Attention 推理， 默认值为False。
- `block_size`: 如果使用Block Attention 推理，指定一个Block可以存储的token数量，默认值为64。
- `cachekv_int8_type`: 是否使用cachekv int8量化用于节省显存，可以是动态或者静态，默认值为None。
