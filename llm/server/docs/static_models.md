## 静态图模型下载支持

* 静态图模型已支持 Deepseek 系列、Qwen 系列、llama 系列等模型 详细支持列表如下：

### DeepSeekV2

|模型名称|静态图下载 model_name|
|:------|:-:|
| deepseek-ai/DeepSeek-V2-Chat              | 🚧 |
| deepseek-ai/DeepSeek-V2-Lite-Chat         | 🚧 |

### DeepSeekV3

|模型名称|静态图下载 model_name|
|:------|:-:|
| deepseek-ai/DeepSeek-V3                   | 🚧 |

### DeepSeekR1
部署硬件要求：
- 除MTP模型，Fp8模型之外支持的最低版本是SM80 (机器：A100 / A800) 要求CUDA 11.8 以上
- DeepSeek-R1-MTP 与 Fp8 模型 支持的最低版本是SM90 (机器：H800) 要求CUDA 12.4 以上

|模型名称|精度|MTP|节点数|静态图下载 model_name|
|:------|:-:|:-:|:-:|:-:|
| deepseek-ai/DeepSeek-R1  |weight_only_int4|否|1| deepseek-ai/DeepSeek-R1/weight_only_int4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int4|是|1| deepseek-ai/DeepSeek-R1-MTP/weight_only_int4 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|否|2| deepseek-ai/DeepSeek-R1-2nodes/weight_only_int8 |
| deepseek-ai/DeepSeek-R1  |weight_only_int8|是|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/weight_only_int8 |
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|否|2| deepseek-ai/DeepSeek-R1-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1  |a8w8_fp8|是|2| deepseek-ai/DeepSeek-R1-MTP-2nodes/a8w8_fp8|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |weight_only_int8|-|-| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B  |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-8B/weight_only_int8 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B |weight_only_int8|-|-|deepseek-ai/DeepSeek-R1-Distill-Llama-70B/weight_only_int8 |


### LLaMA

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| facebook/llama-7b | 🚧 |
| facebook/llama-13b | 🚧 |
| facebook/llama-30b | 🚧 |
| facebook/llama-65b | 🚧 |

### Llama2

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| meta-llama/Llama-2-7b | 🚧 |
| meta-llama/Llama-2-7b-chat | 🚧 |
| meta-llama/Llama-2-13b | 🚧 |
| meta-llama/Llama-2-13b-chat | 🚧 |
| meta-llama/Llama-2-70b | 🚧 |
| meta-llama/Llama-2-70b-chat | 🚧 |

### Llama3

部署硬件要求：
- Append-Attn：
  - 支持的最低版本是SM80 (机器：A100 / A800)
  - 要求CUDA 11.8 以上
- Block-Attn:
  - 支持的最低版本是SM70 (机器：V100)
  - 要求CUDA 11.8 以上


| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| meta-llama/Meta-Llama-3-8B | 🚧 |
| meta-llama/Meta-Llama-3-8B-Instruct |meta-llama/Meta-Llama-3-8B-Instruct-Append-Attn/bfloat16,meta-llama/Meta-Llama-3-8B-Instruct-Block-Attn/float16|
| meta-llama/Meta-Llama-3-70B | 🚧 |
| meta-llama/Meta-Llama-3-70B-Instruct | 🚧 |

### Llama3.1

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| meta-llama/Meta-Llama-3.1-8B | 🚧 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 🚧 |
| meta-llama/Meta-Llama-3.1-70B | 🚧 |
| meta-llama/Meta-Llama-3.1-70B-Instruct | 🚧 |
| meta-llama/Meta-Llama-3.1-405B | 🚧 |
| meta-llama/Meta-Llama-3.1-405B-Instruct | 🚧 |
| meta-llama/Llama-Guard-3-8B | 🚧 |

### Llama3.2

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| meta-llama/Llama-3.2-1B | 🚧 |
| meta-llama/Llama-3.2-1B-Instruct | 🚧 |
| meta-llama/Llama-3.2-3B | 🚧 |
| meta-llama/Llama-3.2-3B-Instruct | 🚧 |
| meta-llama/Llama-Guard-3-1B | 🚧 |

### Llama3.3

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| meta-llama/Llama-3.3-70B-Instruct | 🚧 |


### Mixtral

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 🚧 |

### Qwen

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| qwen/qwen-7b | 🚧 |
| qwen/qwen-7b-chat | 🚧 |
| qwen/qwen-14b | 🚧 |
| qwen/qwen-14b-chat | 🚧 |
| qwen/qwen-72b | 🚧 |
| qwen/qwen-72b-chat | 🚧 |

### Qwen1.5
部署硬件要求：
- Block-Attn:
  - 支持的最低版本是SM70 (机器：V100)
  - 要求CUDA 11.8 以上

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen1.5-0.5B | Qwen/Qwen1.5-0.5B-Block-Attn/bfloat16,Qwen/Qwen1.5-0.5B-Block-Attn/float16 |
| Qwen/Qwen1.5-0.5B-Chat | 🚧 |
| Qwen/Qwen1.5-1.8B | 🚧 |
| Qwen/Qwen1.5-1.8B-Chat | 🚧 |
| Qwen/Qwen1.5-4B | 🚧 |
| Qwen/Qwen1.5-4B-Chat | 🚧 |
| Qwen/Qwen1.5-7B | 🚧 |
| Qwen/Qwen1.5-7B-Chat | 🚧 |
| Qwen/Qwen1.5-14B | 🚧 |
| Qwen/Qwen1.5-14B-Chat | 🚧 |
| Qwen/Qwen1.5-32B | 🚧 |
| Qwen/Qwen1.5-32B-Chat | 🚧 |
| Qwen/Qwen1.5-72B | 🚧 |
| Qwen/Qwen1.5-72B-Chat | 🚧 |
| Qwen/Qwen1.5-110B | 🚧 |
| Qwen/Qwen1.5-110B-Chat | 🚧 |
| Qwen/Qwen1.5-MoE-A2.7B | 🚧 |
| Qwen/Qwen1.5-MoE-A2.7B-Chat | 🚧 |

### Qwen2
部署硬件要求：
- Append-Attn：
  - 支持的最低版本是SM80 (机器：A100 / A800)
  - 要求CUDA 11.8 以上
- Block-Attn:
  - 支持的最低版本是SM70 (机器：V100)
  - 要求CUDA 11.8 以上

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen2-0.5B | 🚧 |
| Qwen/Qwen2-0.5B-Instruct | 🚧 |
| Qwen/Qwen2-1.5B | 🚧 |
| Qwen/Qwen2-1.5B-Instruct | Qwen/Qwen2-1.5B-Instruct-Append-Attn/bfloat16, Qwen/Qwen2-1.5B-Instruct-Block-Attn/float16|
| Qwen/Qwen2-7B | 🚧 |
| Qwen/Qwen2-7B-Instruct | 🚧 |
| Qwen/Qwen2-72B | 🚧 |
| Qwen/Qwen2-72B-Instruct | 🚧 |
| Qwen/Qwen2-57B-A14B | 🚧 |
| Qwen/Qwen2-57B-A14B-Instruct | 🚧 |

### Qwen2-Math

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen2-Math-1.5B | 🚧 |
| Qwen/Qwen2-Math-1.5B-Instruct | 🚧 |
| Qwen/Qwen2-Math-7B | 🚧 |
| Qwen/Qwen2-Math-7B-Instruct | 🚧 |
| Qwen/Qwen2-Math-72B | 🚧 |
| Qwen/Qwen2-Math-72B-Instruct | 🚧 |
| Qwen/Qwen2-Math-RM-72B | 🚧 |

### Qwen2.5

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen2.5-0.5B | 🚧 |
| Qwen/Qwen2.5-0.5B-Instruct | 🚧 |
| Qwen/Qwen2.5-1.5B | 🚧 |
| Qwen/Qwen2.5-1.5B-Instruct | 🚧 |
| Qwen/Qwen2.5-3B | 🚧 |
| Qwen/Qwen2.5-3B-Instruct | 🚧 |
| Qwen/Qwen2.5-7B | 🚧 |
| Qwen/Qwen2.5-7B-Instruct | 🚧 |
| Qwen/Qwen2.5-14B | 🚧 |
| Qwen/Qwen2.5-14B-Instruct | 🚧 |
| Qwen/Qwen2.5-32B | 🚧 |
| Qwen/Qwen2.5-32B-Instruct | 🚧 |
| Qwen/Qwen2.5-72B | 🚧 |
| Qwen/Qwen2.5-72B-Instruct | 🚧 |

### Qwen2.5-Math

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen2.5-Math-1.5B | 🚧 |
| Qwen/Qwen2.5-Math-1.5B-Instruct | 🚧 |
| Qwen/Qwen2.5-Math-7B | 🚧 |
| Qwen/Qwen2.5-Math-7B-Instruct | 🚧 |
| Qwen/Qwen2.5-Math-72B | 🚧 |
| Qwen/Qwen2.5-Math-72B-Instruct | 🚧 |
| Qwen/Qwen2.5-Math-RM-72B | 🚧 |

### Qwen2.5-Coder

| 模型名称 | 静态图下载 model_name |
|:------|:-:|
| Qwen/Qwen2.5-Coder-1.5B | 🚧 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 🚧 |
| Qwen/Qwen2.5-Coder-7B | 🚧 |
| Qwen/Qwen2.5-Coder-7B-Instruct | 🚧 |
