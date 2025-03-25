# Large Model Inference Tutorial

PaddleNLP is designed with a one-stop experience and extreme performance in mind, enabling fast inference of large models.

PaddleNLP's large model inference implements high-performance solutions:

- Built-in dynamic insertion and full-loop operator fusion strategy
- Supports PageAttention and FlashDecoding optimizations
- Supports Weight Only INT8 and INT4 inference, enabling INT8 and FP8 quantization for weights, activations, and Cache KV
- Provides both dynamic graph and static graph inference modes

PaddleNLP's large model inference offers an end-to-end experience from compression to inference to serving:

- Provides various PTQ techniques and flexible WAC (Weight/Activation/Cache) quantization capabilities, supporting INT8, FP8, and 4Bit quantization
- Supports multi-hardware large model inference, including [Kunlun XPU](../../devices/xpu/llama/README.md), [Ascend NPU](../../devices/npu/llama/README.md), [Hygon DCU K100](../dcu_install.md), [Enflame GCU](../../devices/gcu/llama/README.md), [X86 CPU](../cpu_install.md), etc.
- Provides deployment services for server scenarios, supporting continuous batching and streaming output, with HTTP protocol service interfaces

## 1. Model Support

PaddleNLP has implemented high-performance inference models. Verified models include:

| Models | Example Models |
|--------|----------------|
| Llama 3.x, Llama 2 | `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3.1-405B`, `meta-llama/Meta-Llama-3.1-405B-Instruct`,`meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-Guard-3-8B`, `Llama-2-7b, meta-llama/Llama-2-7b-chat`, `meta-llama/Llama-2-13b`, `meta-llama/Llama-2-13b-chat`, `meta-llama/Llama-2-70b`, `meta-llama/Llama-2-70b-chat` |
| Qwen 2.x | `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-1.5B-Instruct`, `Qwen/Qwen2-7B`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-72B`, `Qwen/Qwen2-72B-Instruct`, `Qwen/Qwen2-57B-A14B`, `Qwen/Qwen2-57B-A14B-Instruct`, `Qwen/Qwen2-Math-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, |
|Model Name|Model ID|
|----------|--------|
|Qwen 2.5| `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`|
|Qwen-MoE| `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, `Qwen/Qwen2-57B-A14B`, `Qwen/Qwen2-57B-A14B-Instruct`|
|Mixtral| `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-Instruct-v0.1`|
|ChatGLM 3, ChatGLM 2| `THUDM/chatglm3-6b`, `THUDM/chatglm2-6b`|
|Baichuan 2, Baichuan|`baichuan-inc/Baichuan2-7B-Base`, `baichuan-inc/Baichuan2-7B-Chat`, `baichuan-inc/Baichuan2-13B-Base`, `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, `baichuan-inc/Baichuan-13B-Base`, `baichuan-inc/Baichuan-13B-Chat`|
## 2. Hardware & Precision Support

PaddleNLP provides support for multiple hardware platforms and precisions, including:

| Precision      | Hopper| Ada | Ampere | Turing | Volta | Kunlun XPU | Ascend NPU | Hygon K100 | Tianshu GCU  | SDAA | x86 CPU |
|:--------------:|:-----:|:---:|:------:|:------:|:-----:|:------:|:-------:|:-------:|:------:|:------:|:-------:|
| FP32           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ✅   |  ✅    |   ✅    |
| FP16           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ✅   |  ✅    |   ✅    |
| BF16           |  ✅   |  ✅ | ✅     | ❌      | ❌    | ❌    |  ❌    | ❌    | ❌   |  ❌    |   ✅    |
| INT8           |  ✅   |  ✅ | ✅     | ✅      | ✅    | ✅    |  ✅    | ✅    | ❌   |  ✅    |   ✅    |
| FP8            |  🚧   |  ✅ | ❌     | ❌      | ❌    | ❌    |  ❌    | ❌    | ❌   |  ❌    |   ❌    |


## 3. Inference Parameters

PaddleNLP provides various parameters for configuring inference models and optimizing inference performance.

### 3.1 General Parameters

- `model_name_or_path`: Required. The name of the pre-trained model or the path to the local model, used to initialize the model and tokenizer. Default is None.

- `dtype`: Required. Data type of model parameters. Default is None. If `lora_path` or `prefix_path` is not provided, `dtype` must be specified.

- `lora_path`: Path to LoRA parameters and configuration, used to initialize LoRA parameters. Default is None.

- `prefix_path`: Path to Prefix Tuning parameters and configuration, used to initialize Prefix Tuning parameters. Default is None.

- `batch_size`: Batch size. Default is 1. Larger batch sizes consume more memory, while smaller values reduce memory usage.

- `data_file`: Input JSON file for inference. Default is None. Example data:

    ```json
    {"tgt":"", "src": "Write a 300-word novel outline about Li Bai traveling to modern times and eventually becoming a corporate office worker"}
    {"tgt":"", "src": "I need to interview a science fiction author, create a list of 5 interview questions"}
    ```

- `output_file`: File to save inference results. Default is output.json.

- `device`
### 3.1 Basic Parameters

- **Running environment**: Default is GPU. Optional values include GPU, [CPU](../cpu_install.md), [XPU](../../devices/xpu/llama/README.md), [NPU](../../devices/npu/llama/README.md), [GCU](../../devices/gcu/llama/README.md), etc. ([DCU](../dcu_install.md) uses the same inference commands as GPU).

- **`model_type`**: Initialize different model types. gpt-3: GPTForCausalLM; ernie-3.5-se: Ernie35ForCausalLM; default is None.

- **`mode`**: Use dynamic graph or static graph for inference. Optional values: `dynamic`, `static`. Default is `dynamic`.

- **`avx_model`**: When using CPU inference, whether to use AvxModel. Default is False. Refer to [CPU Inference Tutorial](../cpu_install.md).

- **`avx_type`**: AVX computation type. Default is None. Optional values: `fp16`, `bf16`.

- **`src_length`**: Maximum token length for model input (prompt only). Default is 1024.

- **`max_length`**: Maximum token length for model output (generated content only). Default is 1024.

- **`total_max_length`**: Maximum token length for model input + output (prompt + generated content). Default is 4096.

- **`mla_use_matrix_absorption`**: Whether to use the matrix absorption implementation with better performance for MLA module when running DeepSeek-V3/R1 models. Default is True.

### 3.2 Performance Optimization Parameters

- **`inference_model`**: Whether to use Inference Model for inference. Default is False. The Inference Model incorporates dynamic insertion and full-cycle operator fusion strategies, offering better performance when enabled.

- **`block_attn`**: Whether to use Block Attention for inference. Default is False. Block Attention is designed based on PageAttention, enabling dynamic cachekv memory allocation while maintaining high-performance inference and dynamic insertion, significantly saving memory and improving inference throughput.

- **`append_attn`**: Append Attention further optimizes the Attention module based on Block Attention implementation, incorporating C4 high-performance support to significantly improve inference performance. This is an enhanced version of Block Attention and can be enabled separately instead of `block_attn`.

- **`block_size`**: If using Block Attention or Append Attention, specifies the number of tokens stored per block. Default is 64.

### 3.3 Quantization Parameters

PaddleNLP provides multiple quantization strategies, supporting Weight Only INT8 and INT4 inference, and supporting WAC (Weight, Activation, Cache KV) with INT8/FP8 quantization.

- **`quant_type`**: Whether to use quantized inference. Default is None. Optional values: `weight_only_int8`, `weight_only_int4`, `a8w8`, etc.
`a8w8_fp8`. Both `a8w8` and `a8w8_fp8` require additional scale calibration tables for activation and weight. During inference, the `model_name_or_path` should be the quantized model produced by PTQ calibration. For quantization model export, refer to the [Large Model Quantization Tutorial](../quantization.md).

- `cachekv_int8_type`: Whether to use cachekv int8 quantization. Default is None. Options: `dynamic` (no longer maintained, not recommended) and `static`. `static` requires additional scale calibration tables for cache kv. The input `model_name_or_path` should be the quantized model produced by PTQ calibration. For quantization model export, refer to the [Large Model Quantization Tutorial](../quantization.md).

- `weightonly_group_size`: In `weight_only` mode, use `group wise` quantization. The `group size` currently supports `64` and `128`. Default is `-1`, which indicates `channel wise` mode.

- `weight_block_size`: FP8 weight quantization granularity. Currently supports DeepSeek-V3/R1 models. Default is [128 128].

- `moe_quant_type`: MoE quantization type. Supports FP8 MoE quantization inference for DeepSeek-V3/R1 models. Default is empty. Optional values: `weight_only_int4`, `weight_only_int8`.

### 3.4 Speculative Decoding Parameters

PaddleNLP provides multiple speculative decoding methods. For details, refer to the [Speculative Decoding Tutorial](./speculative_decoding.md).

- `speculate_method`: Speculative decoding algorithm. Default is `None`. Valid options: `None`, `inference_with_reference`, `mtp`, `eagle`. When set to `None`, normal autoregressive decoding is used. When set to `inference_with_reference`, context-based speculative decoding is used [Paper Link](https://arxiv.org/pdf/2304.04487).

- `speculate_max_draft_token_num`: Maximum number of draft tokens generated per round in the speculative decoding algorithm. Default is 1, maximum supported value is 5.

- `speculate_max_ngram_size`: Maximum window size for n-gram matching of draft tokens. Default is `1`. In the `inference_with_reference` algorithm, draft tokens are first matched from the prompt using n-gram window sliding. The window size and input-output overlap jointly determine the overhead of generating draft tokens, thus affecting the acceleration performance of the `inference_with_reference` algorithm.

- `speculate_verify_window` (temporarily deprecated): The speculative decoding verify strategy defaults to TopP + window verify with window size. Default is `2`. For more details on TopP + window verify, refer to the [Speculative Decoding Tutorial](./speculative_decoding.md).

- `speculate_max_candidate_len`
## 4. Fast Start

### 4.1 Environment Preparation

Refer to the [Installation Guide](./installation.md).

### 4.2 Inference Example

Below is a dynamic graph inference example for Llama2-7B:

```python
from transformers import AutoTokenizer
from flashllm import AutoModel

model_path = "meta-llama/Llama-2-7b-chat-hf"
prompt = "Hello, my name is"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModel.from_pretrained(model_path, dtype="float16")

model.set_chat_template("llama-2")
input_ids = tokenizer([prompt]).input_ids
output = model.generate(
    inputs=input_ids,
    max_new_tokens=100,
    do_sample=True
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 4.3 Performance Analysis

Use the following command to test inference speed:
```bash
python -m flashllm.llm \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dtype "float16" \
    --benchmark True \
    --src_length 1024 \
    --max_length 2048
```
```shell
# Dynamic Graph Model Inference Command Reference
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn

# XPU Device Dynamic Graph Model Inference Command Reference
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --device xpu

# Weight Only Int8 Dynamic Graph Inference Reference
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --quant_type weight_only_int8 --block_attn

# PTQ-A8W8 Inference Command Reference
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn --quant_type a8w8

# PTQ-A8W8C8 Inference Command Reference
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn --quant_type a8w8 --cachekv_int8_type static

# CacheKV Dynamic Quantization Inference Command Reference
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --cachekv_int8_type dynamic

```

**Note:**

1. The available options for `quant_type` include `weight_only_int8`, `weight_only_int4`, `a8w8`, and `a8w8_fp8`.
2. `a8w8` and `a8w8_fp8` require additional act and weight scale calibration tables. The `model_name_or_path` parameter for inference should point to the PTQ-calibrated quantized model. For quantized model export, refer to the [Large Model Quantization Tutorial](../quantization.md).
3. The `cachekv_int8_type` supports two options: `dynamic` (no longer maintained, not recommended) and `static`. `static` requires additional cache kv scale calibration tables, and the `model_name_or_path` parameter should point to the corresponding PTQ-calibrated model.
for PTQ calibrated quantized models. For exporting quantized models, refer to [Large Model Quantization Tutorial](../quantization.md).

## 5. Service Deployment

**For high-performance serving deployment, refer to**: [Static Graph Serving Deployment Tutorial](../../server/docs/deploy_usage_tutorial.md).

For quick model experience, we provide **simplified Flash Server dynamic graph deployment** with an easy-to-use UI serving method based on dynamic graph inference.

Environment Preparation

- python >= 3.9
- gradio
- flask

Serving Deployment Script

```shell
# Single GPU, use paddle.distributed.launch for multi-GPU inference
python ./predict/flask_server.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 \
    --flask_port 8011 \
    --dtype "float16"
```

- `port`: Gradio UI service port, default 8010.
- `flask_port`: Flask service port, default 8011.

UI Interface: Access `http://127.0.0.1:8010` to use the gradio interface for conversations.
API Access: You can also access via flask service API.

1. Reference: `./predict/request_flask_server.py` file.
```shell
python predict/request_flask_server.py
```

2. Or directly use curl to start conversation:
```shell
curl 127.0.0.1:8011/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"message": [{"role": "user", "content": "Hello"}]}'
```

3. Using OpenAI client:
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
        {"role": "user", "content": "PaddleNLP is awesome! What's the sentiment of this sentence?"}
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
This deployment method offers average performance. For high-performance serving deployment, refer to: [Static Graph Serving Deployment Tutorial](../../server/docs/deploy_usage_tutorial.md).

More large model inference tutorials:

- [llama](./llama.md)
- [qwen](./qwen.md)
- [deepseek](./deepseek.md)
- [mixtral](./mixtral.md)
- [Speculative Decoding](./speculative_decoding.md)

Environment setup, refer to:

- [Installation Tutorial](./installation.md)

For optimal inference performance:

- [Best Practices](./best_practices.md)

More compression and serving inference experiences:

- [Large Model Quantization Tutorial](../quantization.md)
- [Static Graph Serving Deployment Tutorial](../../server/docs/deploy_usage_tutorial.md)

More hardware-specific large model inference tutorials:

- [Kunlun XPU](../../devices/xpu/llama/README.md)
- [Ascend NPU](../../devices/npu/llama/README.md)
- [Hygon K100](../dcu_install.md)
- [Suiyuan GCU](../../devices/gcu/llama/README.md)
- [Taichu SDAA](../../devices/sdaa/llama/README.md)
- [X86 CPU](../cpu_install.md)

## Acknowledgements

We reference the [FlashInfer framework](https://github.com/flashinfer-ai/flashinfer) and implement append attention based on FlashInfer. Inspired by [PageAttention](https://github.com/vllm-project/vllm)'s paging concept, we implement block attention during generation phase. Leveraging [Flash Decoding](https://github.com/Dao-AILab/flash-attention)'s KV chunking technique, we accelerate long-sequence inference. Utilizing [Flash Attention2](https://github.com/Dao-AILab/flash-attention), we optimize attention during prefill phase. FP8 GEMM operations are implemented using high-performance templates from [CUTLASS](https://github.com/NVIDIA/cutlass). Some operators like gemm_dequant draw inspiration from implementations and optimizations in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer.git).
