# 大模型推理教程

PaddleNLP以一站式体验、极致性能为设计理念，实现大模型的快速推理。

PaddleNLP大模型推理构建了高性能推理方案：

- 内置动态插入和全环节算子融合策略

- 支持PageAttention、FlashDecoding优化

- 支持Weight Only INT8及INT4推理，支持权重、激活、Cache KV进行INT8、FP8量化的推理

- 提供动态图推理和静态图推理两种方式


PaddleNLP大模型推理提供压缩、推理、服务全流程体验 ：

- 提供多种PTQ技术，提供WAC（权重/激活/缓存）灵活可配的量化能力，支持INT8、FP8、4Bit量化能力

- 支持多硬件大模型推理，包括[昆仑XPU](../../xpu/llama/README.md)、[昇腾NPU](../../npu/llama/README.md)、[海光K100](../dcu_install.md)、[燧原GCU](../../gcu/llama/README.md)、[X86 CPU](../../../csrc/cpu/README.md)等

- 提供面向服务器场景的部署服务，支持连续批处理(continuous batching)、流式输出等功能，支持HTTP、RPC、RESTful多种Clent端形式


PaddleNLP大模型推理支持如下主流开源大模型：

- 基于Transformer的大模型（如Llama、Qwen）

- 混合专家大模型（如Mixtral）


## 1. 模型支持

PaddleNLP 中已经添加高性能推理模型相关实现，支持：
| Models | Example Models |
|--------|----------------|
|Llama 3.1, Llama 3, Llama 2|`meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Meta-Llama-3.1-405B`,  etc.|
|Qwen 2| `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-7B`, `Qwen/Qwen2-72B`, etc|
|Qwen-Moe| `Qwen/Qwen1.5-MoE-A2.7B`, etc|
|Mixtral| `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc|
|ChatGLM 3, ChatGLM 2| `THUDM/chatglm3-6b`, `THUDM/chatglm2-6b`, etc|
|Baichuan 3, Baichuan 2, Baichuan|`baichuan-inc/Baichuan2-7B-Base`, `baichuan-inc/Baichuan2-7B-Chat` , etc|
|Bloom|`bigscience/bloom-560m`, `bigscience/bloom-1b1`, `bigscience/bloom-3b`, etc|


## 2. 硬件&精度支持

PaddleNLP 提供了多种硬件平台和精度支持，包括：

| Precision      | Ada | Ampere | Turing | Volta | 昆仑XPU | 昇腾NPU | 海光K100 | 燧原GCU | x86 CPU |
|----------------|-----|--------|--------|-------|--------|---------|---------|--------|---------|
| FP32           |  ✅ | ✅     | ✅      | ✅    | ✅      |  ✅     | ✅      | ✅      |   ✅    |
| FP16           |  ✅ | ✅     | ✅      | ✅    | ✅      |  ✅     | ✅      | ✅      |   ✅    |
| BF16           |  ✅ | ✅     | ❌      | ❌    | ❌      |  ❌     | ❌      | ❌      |   ✅    |
| INT8           |  ✅ | ✅     | ✅      | ✅    | ❌      |  ❌     | ✅      | ❌      |   ✅    |
| FP8            |  ✅ | ❌     | ❌      | ❌    | ❌      |  ❌     | ❌      | ❌      |   ❌    |


## 3. 推理参数

PaddleNLP 提供了多种参数，用于配置推理模型和优化推理性能。

### 3.1 常规参数

- `model_name_or_path`: 必需，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为None。

- `dtype`: 必需，模型参数dtype，默认为None。如果没有传入`lora_path`、`prefix_path`则必须传入。

- `lora_path`: LoRA参数和配置路径，对LoRA参数进行初始化，默认为None。

- `prefix_path`: Prefix Tuning参数和配置路径，对Prefix Tuning参数进行初始化，默认为None。

- `batch_size`: 批处理大小，默认为1。该参数越大，占用显存越高；该参数越小，占用显存越低。

- `data_file`: 待推理json文件，默认为None。

- `output_file`: 保存推理结果文件，默认为output.json。

- `device`: 运行环境，默认为gpu。可选的数值有`cpu`, `gpu`, `xpu`。

- `model_type`: 初始化不同类型模型，gpt-3: GPTForCausalLM; ernie-3.5-se: Ernie35ForCausalLM; 默认为 None。

- `mode`: 使用动态图或者静态图推理，可选值有`dynamic`、 `static`，默认为`dynamic`。

- `avx_model`: 当使用CPU推理时，是否使用AvxModel，默认为False。参考[CPU推理教程]()。

- `avx_type`: avx计算类型，默认为None。可选的数值有`fp16`、 `bf16`。

- `src_length`: 模型输入上下文最大token长度，默认为1024。

- `max_length`:模型输入（上下文+生成内容）的最大token长度, 默认为2048。


### 3.2 性能优化参数

- `inference_model`: 是否使用 Inference Model 推理，默认值为 False。Inference Model 内置动态插入和全环节算子融合策略，开启后性能更优。

- `block_attn`: 是否使用 Block Attention 推理， 默认值为False。Block Attention 是基于 PageAttention 的思想设计并实现的，在保持高性能推理和动态插入的基础上可以动态地为 cachekv 分配存储空间，极大地节省显存并提升推理的吞吐。

- `block_size`: 如果使用 Block Attention 推理，指定一个 Block 可以存储的 token 数量，默认值为64。


### 3.3 量化参数

PaddleNLP 提供了多种量化策略，支持Weight Only INT8及INT4推理，支持WAC（权重、激活、Cache KV）进行INT8、FP8量化的推理

- `quant_type`: 是否使用量化推理，默认值为None。可选的数值有`weight_only_int8`、`weight_only_int4`、`a8w8`和`a8w8_fp8`。`a8w8`与`a8w8_fp8`需要额外的act和weight的scale校准表，推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。

- `cachekv_int8_type`: 是否使用cachekv int8量化，默认值为None。可选`dynamic`和`static`两种，`static`需要额外的cache kv的scale校准表，传入的 `model_name_or_path` 为PTQ校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。


### 3.4 解码策略参数

- `decode_strategy`: 推理解码策略，默认值为`sampling`，可选的数值有`greedy_search`、`beam_search`和`sampling`。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认值为1，等价于贪心策略。

- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认值为1.0，表示不起作用。

- `temperature`:“采样”策略中会对输出logit除以temperature。默认值为1.0，表示不起作用。

### 3.4 性能分析参数

- `benchmark`: 是否开启性能分析，默认值为False。如果设为true，会将模型输入填充为src_length并强制解码到max_length，并计算模型推理吞吐量、记录推理时间。


## 4. 快速开始

### 4.1 环境准备

git clone 代码到本地：

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
export PYTHONPATH=/path/to/PaddleNLP:$PYTHONPATH
```

PaddleNLP 针对于Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能，使用之前需要预先安装自定义算子库：


```shell
git clone https://github.com/PaddlePaddle/PaddleNLP
#GPU设备安装自定义算子
cd ./paddlenlp/csrc && python setup_cuda.py install
#XPU设备安装自定义算子
cd ./paddlenlp/csrc/xpu/src && sh cmake_build.sh
#DCU设备安装自定义算子
cd ./paddlenlp/csrc && python setup_hip.py install
```

到达运行目录，即可开始：

```shell
cd PaddleNLP/llm
```

### 4.2 推理示例

下面给出Llama2-7B的动态图推理示例：

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn

# XPU设备动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --device xpu

# Weight Only Int8 动态图推理参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --quant_type weight_only_int8 --block_attn

# PTQ-A8W8推理命令参考
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn --quant_type a8w8

# PTQ-A8W8C8推理命令参考
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn --quant_type a8w8  --cachekv_int8_type static

# CacheKV 动态量化推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --cachekv_int8_type dynamic
```

**Note:**

1. `quant_type`可选的数值有`weight_only_int8`、`weight_only_int4`、`a8w8`和`a8w8_fp8`。
2. `a8w8`与`a8w8_fp8`需要额外的act和weight的scale校准表，推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。
3. `cachekv_int8_type`可选`dynamic`和`static`两种，`static`需要额外的cache kv的scale校准表，传入的 `model_name_or_path` 为PTQ校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。

更多大模型推理教程，参考：

-  [llama](./llama.md)
-  [qwen](./qwen.md)
-  [mixtral](./mixtral.md)

环境准备，参考：

- [安装教程](./installation.md)

获取最佳推理性能：

- [最佳实践](./best_practices.md)

更多压缩、服务化推理体验：

- [大模型量化教程](../quantization.md)
- [服务化部署教程](https://github.com/PaddlePaddle/FastDeploy/blob/develop/README_CN.md)

更多硬件大模型推理教程：

- [X86 CPU](../../../csrc/cpu/README.md)
- [昆仑XPU](../../../csrc/xpu/README.md)
- [昇腾NPU](../../npu/llama/README.md)
- [海光DCU](../dcu_install.md)
- [海光K100]()
- [燧原GCU]()



