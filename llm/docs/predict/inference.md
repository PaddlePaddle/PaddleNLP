# 大模型推理教程

PaddleNLP 以一站式体验、极致性能为设计理念，实现大模型的快速推理。

PaddleNLP 大模型推理构建了高性能推理方案：

- 内置动态插入和全环节算子融合策略

- 支持 PageAttention、FlashDecoding 优化

- 支持 Weight Only INT8及 INT4推理，支持权重、激活、Cache KV 进行 INT8、FP8量化的推理

- 提供动态图推理和静态图推理两种方式


PaddleNLP 大模型推理提供压缩、推理、服务全流程体验 ：

- 提供多种 PTQ 技术，提供 WAC（权重/激活/缓存）灵活可配的量化能力，支持 INT8、FP8、4Bit 量化能力

- 支持多硬件大模型推理，包括[昆仑 XPU](../../devices/xpu/llama/README.md)、[昇腾 NPU](../../devices/npu/llama/README.md)、[海光 K100](../dcu_install.md)、[燧原 GCU](../../devices/gcu/llama/README.md)、[X86 CPU](../cpu_install.md)等

- 提供面向服务器场景的部署服务，支持连续批处理(continuous batching)、流式输出等功能，HTTP 协议的服务接口


## 1. 模型支持

PaddleNLP 中已经添加高性能推理模型相关实现，已验证过的模型如下：

| Models | Example Models |
|--------|----------------|
|Llama 3.x, Llama 2|`meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3.1-405B`, `meta-llama/Meta-Llama-3.1-405B-Instruct`,`meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-Guard-3-8B`, `Llama-2-7b, meta-llama/Llama-2-7b-chat`, `meta-llama/Llama-2-13b`, `meta-llama/Llama-2-13b-chat`, `meta-llama/Llama-2-70b`, `meta-llama/Llama-2-70b-chat`|
|Qwen 2.x|`Qwen/Qwen2-1.5B`, `Qwen/Qwen2-1.5B-Instruct`, `Qwen/Qwen2-7B`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-72B`, `Qwen/Qwen2-72B-Instruct`, `Qwen/Qwen2-57B-A14B`, `Qwen/Qwen2-57B-A14B-Instruct`, `Qwen/Qwen2-Math-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`|
|Qwen-MoE| `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, `Qwen/Qwen2-57B-A14B`, `Qwen/Qwen2-57B-A14B-Instruct`|
|Mixtral| `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-Instruct-v0.1`|
|ChatGLM 3, ChatGLM 2| `THUDM/chatglm3-6b`, `THUDM/chatglm2-6b`|
|Baichuan 2, Baichuan|`baichuan-inc/Baichuan2-7B-Base`, `baichuan-inc/Baichuan2-7B-Chat`, `baichuan-inc/Baichuan2-13B-Base`, `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, `baichuan-inc/Baichuan-13B-Base`, `baichuan-inc/Baichuan-13B-Chat`|


## 2. 硬件&精度支持

PaddleNLP 提供了多种硬件平台和精度支持，包括：

| Precision      | Hopper| Ada | Ampere | Turing | Volta | 昆仑 XPU | 昇腾 NPU | 海光 K100 | 燧原 GCU  | 太初 SDAA| x86 CPU |
|:--------------:|:-----:|:---:|:------:|:------:|:-----:|:------:|:-------:|:-------:|:------:|:------:|:-------:|
| FP32           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ✅   |  ✅    |   ✅    |
| FP16           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ✅   |  ✅    |   ✅    |
| BF16           |  ✅   |  ✅ | ✅     | ❌      | ❌    | ❌    |  ❌    | ❌    | ❌   |  ❌    |   ✅    |
| INT8           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ❌   |  ✅    |   ✅    |
| FP8            |  🚧   |  ✅ | ❌     | ❌      | ❌    | ❌    |  ❌    | ❌    | ❌   |  ❌    |   ❌    |


## 3. 推理参数

PaddleNLP 提供了多种参数，用于配置推理模型和优化推理性能。

### 3.1 常规参数

- `model_name_or_path`: 必需，预训练模型名称或者本地的模型路径，用于热启模型和分词器，默认为 None。

- `dtype`: 必需，模型参数 dtype，默认为 None。如果没有传入`lora_path`或`prefix_path`则必须传入`dtype`参数。

- `lora_path`: LoRA 参数和配置路径，对 LoRA 参数进行初始化，默认为 None。

- `prefix_path`: Prefix Tuning 参数和配置路径，对 Prefix Tuning 参数进行初始化，默认为 None。

- `batch_size`: 批处理大小，默认为1。该参数越大，占用显存越高；该参数越小，占用显存越低。

- `data_file`: 待推理 json 文件，默认为 None。样例数据：

    ```json
    {"tgt":"", "src": "写一个300字的小说大纲，内容是李白穿越到现代，最后成为公司文职人员的故事"}
    {"tgt":"", "src": "我要采访一位科幻作家，创建一个包含5个问题的列表"}
    ```

- `output_file`: 保存推理结果文件，默认为 output.json。

- `device`: 运行环境，默认为 gpu，可选的数值有 gpu、[cpu](../cpu_install.md)、[xpu](../../devices/xpu/llama/README.md)、[npu](../../devices/npu/llama/README.md)、[gcu](../../devices/gcu/llama/README.md)等（[dcu](../dcu_install.md)与 gpu 推理命令一致）。

- `model_type`: 初始化不同类型模型，gpt-3: GPTForCausalLM; ernie-3.5-se: Ernie35ForCausalLM; 默认为 None。

- `mode`: 使用动态图或者静态图推理，可选值有`dynamic`、 `static`，默认为`dynamic`。

- `avx_model`: 当使用 CPU 推理时，是否使用 AvxModel，默认为 False。参考[CPU 推理教程](../cpu_install.md)。

- `avx_type`: avx 计算类型，默认为 None。可选的数值有`fp16`、 `bf16`。

- `src_length`: 模型输入（仅 prompt）最大 token 长度，默认为 1024。

- `max_length`: 模型输出（仅生成内容）的最大 token 长度,默认为 1024。

- `total_max_length`: 模型输入+输出（prompt+生成内容）的最大 token 长度,默认为 4096。

- `mla_use_matrix_absorption`: 跑 DeepSeek-V3/R1 模型时，是否使用 MLA 模块性能更优的矩阵吸收实现，默认为 True。


### 3.2 性能优化参数

- `inference_model`: 是否使用 Inference Model 推理，默认值为 False。Inference Model 内置动态插入和全环节算子融合策略，开启后性能更优。

- `block_attn`: 是否使用 Block Attention 推理， 默认值为 False。Block Attention 是基于 PageAttention 的思想设计并实现的，在保持高性能推理和动态插入的基础上可以动态地为 cachekv 分配存储空间，极大地节省显存并提升推理的吞吐。

- `append_attn`: Append Attention 在 Block Attention 实现的基础上，进一步借鉴 FlashInfer 的实现对 Attention 模块进行了优化，并增加了 C4的高性能支持，极大地提升了推理性能。属于是 Block Attention 实现的升级版，此选项可替代`block_attn`单独开启。

- `block_size`: 如果使用 Block Attention 或者 Append Attention 推理，指定一个 Block 可以存储的 token 数量，默认值为64。


### 3.3 量化参数

PaddleNLP 提供了多种量化策略，支持 Weight Only INT8及 INT4推理，支持 WAC（权重、激活、Cache KV）进行 INT8、FP8量化的推理

- `quant_type`: 是否使用量化推理，默认值为 None。可选的数值有`weight_only_int8`、`weight_only_int4`、`a8w8`和`a8w8_fp8`。`a8w8`与`a8w8_fp8`需要额外的 act 和 weight 的 scale 校准表，推理传入的 `model_name_or_path` 为 PTQ 校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。

- `cachekv_int8_type`: 是否使用 cachekv int8量化，默认值为 None。可选`dynamic`（已不再维护，不建议使用）和`static`两种，`static`需要额外的 cache kv 的 scale 校准表，传入的 `model_name_or_path` 为 PTQ 校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。

- `weightonly_group_size`: `weight_only`模式下，使用`group wise`量化方式，`group size`目前支持 为 `64` 和 `128`，默认值为`-1`表示`channel wise`模式。

- `weight_block_size`: FP8 权重量化粒度， 当前支持 DeepSeek-V3/R1 模型， 默认为[128 128]。

- `moe_quant_type`: MoE 量化类型， 支持 DeepSeek-V3/R1-FP8 模型的 MoE 量化推理， 默认为空， 可选值`weight_only_int4`、`weight_only_int8`。

### 3.4 投机解码参数

PaddleNLP 提供了多种投机解码方法，具体细节请查阅[投机解码教程](./speculative_decoding.md).

- `speculate_method`: 推理解码算法，默认值为`None`，可选的数值有`None`、`inference_with_reference`、 `mtp`、 `eagle`。为`None`时为正常自回归解码，为`inference_with_reference`时为基于上下文的投机解码[论文地址](https://arxiv.org/pdf/2304.04487)。

- `speculate_max_draft_token_num`: 投机解码算法中每轮产生的最大 draft tokens 数目，默认值为 1，最大支持 5。

- `speculate_max_ngram_size`: n-gram 匹配 draft tokens 时的最大窗口大小，默认值为`1`。inference_with_reference 算法中会先从 prompt 中使用 ngram 窗口滑动匹配 draft tokens，窗口大小和输入输出重叠程度共同决定了产生 draft tokens 的开销从而影响 inference_with_reference 算法的加速效果。

- `speculate_verify_window`(暂时废弃): 投机解码 verify 策略默认采用 TopP + window verify 中的 window 大小，默认值为`2`。更多有关 TopP + window verify 的详细介绍参考[投机解码教程](./speculative_decoding.md)。

- `speculate_max_candidate_len`(暂时废弃): 产生的最大候选 tokens 数目，根据候选 tokens 与 draft tokens 比较来进行 verify(仅在 TopP + window verify 时生效)，默认值为`5`。

- `draft_model_name_or_path`: 在`MTP`或者`EAGLE`模式下，`Draft Model`的路径。

- `draft_model_quant_type`: 在`MTP`或者`EAGLE`模式下，`Draft Model`的推理量化精度，参考`--quant_type`。

- `return_full_hidden_states`: 在`MTP`或者`EAGLE`模式下，是否返回全部的隐藏层状态，默认为`False`。

### 3.5 解码策略参数

- `decode_strategy`: 推理解码策略，默认值为`sampling`，可选的数值有`greedy_search`、`beam_search`和`sampling`。

- `top_k`: “采样”策略中为 top-k 过滤保留的最高概率标记的数量。默认值为1，等价于贪心策略。

- `top_p`:“采样”策略中 top-p 过滤的累积概率。默认值为1.0，表示不起作用。

- `temperature`:“采样”策略中会对输出 logit 除以 temperature。默认值为1.0，表示不起作用。

### 3.6 性能分析参数

- `benchmark`: 是否开启性能分析，默认值为 False。如果设为 true，会将模型输入填充为 src_length 并强制解码到 max_length，并计算模型推理吞吐量、记录推理时间。


## 4. 快速开始

### 4.1 环境准备

参考[安装教程](./installation.md)。

### 4.2 推理示例

下面给出 Llama2-7B 的动态图推理示例：

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
2. `a8w8`与`a8w8_fp8`需要额外的 act 和 weight 的 scale 校准表，推理传入的 `model_name_or_path` 为 PTQ 校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。
3. `cachekv_int8_type`可选`dynamic`（已不再维护，不建议使用）和`static`两种，`static`需要额外的 cache kv 的 scale 校准表，传入的 `model_name_or_path` 为 PTQ 校准产出的量化模型。量化模型导出参考[大模型量化教程](../quantization.md)。


## 5. 服务化部署

**高性能服务化部署请参考**：[静态图服务化部署教程](../../server/docs/deploy_usage_tutorial.md)。

如果您想简单体验模型，我们提供了**简易的 Flash Server 动态图部署**方式，我们提供了一套基于动态图推理的简单易用 UI 服务化部署方法，用户可以快速部署服务化推理。

环境准备

- python >= 3.9
- gradio
- flask

服务化部署脚本

```shell
# 单卡，可以使用 paddle.distributed.launch 启动多卡推理
python  ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```

- `port`: Gradio UI 服务端口号，默认8010。
- `flask_port`: Flask 服务端口号，默认8011。

图形化界面: 打开 `http://127.0.0.1:8010` 即可使用 gradio 图形化界面，即可开启对话。
API 访问: 您也可用通过 flask 服务化 API 的形式.

1. 可参考：`./predict/request_flask_server.py` 文件。
```shell
python predict/request_flask_server.py
```

2. 或者直接使用 curl,调用开始对话
```shell
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "你好"}]}'
```
3.使用 OpenAI 客户端调用：
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8011/v1/",
)

# Completion API
stream = True
completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "PaddleNLP好厉害！这句话的感情色彩是？"}
    ],
    max_tokens=1024,
    stream=stream,
)

if stream:
    for c in completion:
        print(c.choices[0].delta.content, end="")
else:
    print(completion.choices[0].message.content)
```
该方式部署，性能一般，高性能服务化部署请参考：[静态图服务化部署教程](../../server/docs/deploy_usage_tutorial.md)。



更多大模型推理教程：

-  [llama](./llama.md)
-  [qwen](./qwen.md)
-  [deepseek](./deepseek.md)
-  [mixtral](./mixtral.md)
-  [投机解码](./speculative_decoding.md)

环境准备，参考：

- [安装教程](./installation.md)

获取最佳推理性能：

- [最佳实践](./best_practices.md)

更多压缩、服务化推理体验：

- [大模型量化教程](../quantization.md)
- [静态图服务化部署教程](../../server/docs/deploy_usage_tutorial.md)

更多硬件大模型推理教程：

- [昆仑 XPU](../../devices/xpu/llama/README.md)
- [昇腾 NPU](../../devices/npu/llama/README.md)
- [海光 K100](../dcu_install.md)
- [燧原 GCU](../../devices/gcu/llama/README.md)
- [太初 SDAA](../../devices/sdaa/llama/README.md)
- [X86 CPU](../cpu_install.md)

## 致谢

我们参考[FlashInfer 框架](https://github.com/flashinfer-ai/flashinfer)，在 FlashInfer 的基础上，实现了 append attention。参考[PageAttention](https://github.com/vllm-project/vllm)的 page 分块的思想实现了 generation 阶段的 block attention。基于[Flash Decoding](https://github.com/Dao-AILab/flash-attention)的 KV 分块思想实现了长 sequence 场景下的推理加速。基于[Flash Attention2](https://github.com/Dao-AILab/flash-attention)实现了 prefill 阶段的 attention 加速。FP8 GEMM 基于[CUTLASS](https://github.com/NVIDIA/cutlass)的高性能模板库实现。有部分算子如 gemm_dequant 参考了[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)和[FasterTransformer](https://github.com/NVIDIA/FasterTransformer.git)的实现和优化思路。
