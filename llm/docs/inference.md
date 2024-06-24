# 大模型推理教程

PaddleNLP除了提供常用模型推理外，还提供了高性能推理，内置动态插入和全环节算子融合策略，极大加快并行推理的速度。

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```

## 1. 常用模型推理
PaddleNLP 提供了动态图推理和静态图推理两种方式，方便用户快速验证模型推理效果（包含LoRA、PrefixTuning）

### 1.1 动态图推理
```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --data_file ./data/dev.json --dtype float16
```
对于LoRA、PrefixTuning 模型只需额外传入相应的lora_path或prefix_path即可，如：`--lora_path ./checkpoints/llama_lora_ckpts`或`--prefix_path ./checkpoints/llama_prefix_ckpts`，详见推理参数减少。

### 1.2 静态图推理

```shell
# 静态图模型推理命令参考， LoRA需要先合并参数，Prefix Tuning暂不支持
# step1 : 静态图导出
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --output_path ./inference --dtype float16
# step2: 静态图推理
python ./predict/predictor.py --model_name_or_path ./inference --data_file ./data/dev.json --dtype float16 --mode static
```

## 2. 高性能模型推理

高性能推理内置动态插入和全环节算子融合策略，隐藏了底层实现的细节，实现了开箱即用高性能并行推理能力。
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

PaddleNLP 中已经添加高性能推理模型相关实现，支持：

| Model                       | Inference Model | PTuning | WINT8 | PTQ-A8W8 |
|-----------------------------|-----------------|---------|-------|-----|
| [LLaMA1/2](../llama)         | ✅               | ✅       | ✅     | ✅   |
| [ChatGLM](../chatglm)        | ✅               | ✅       | ✅     | ❌   |
| [ChatGLM2](../chatglm2)      | ✅               | ❌       | ❌     | ❌   |
| [Bloom](../bloom)            | ✅               | ✅       | ✅     | ❌   |
| [GPT-3](../gpt-3)            | ✅               | ❌       | ❌     | ❌   |
| [Qwen](../qwen)              | ✅               | ❌       | ❌     | ❌   |
| [BaiChuan-7B](../llama)     | ✅               | ✅       | ✅     | 🚧   |
| [BaiChuan2-7B](../llama)     | ✅               | ✅       | ✅     | 🚧   |
| [BaiChuan2-13B](../llama) | 🚧               | 🚧       | 🚧     | 🚧   |

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported
* WINT8:指Weight-Only Quantization INT8，即对权重进行INT8量化的模型。
* PTQ-A8W8:指使用PTQ对线性层的激活和权重都量化为INT8的模型。

为了进一步提升推理的吞吐，我们基于PageAttention的思想设计并实现了BlockAttention，在保持高性能推理和动态插入的基础上可以动态地为cachekv分配存储空间，极大地节省显存，从而在同一时刻处理更多的query以获得吞吐的提升。下面分别给出关闭BlockAttention和打开BlockAttention进行高性能推理的命令参考。

### 2.2 环境准备

- PaddleNLP develop
- PaddlePaddle develop

PaddleNLP 针对于Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能，使用之前需要预先安装自定义算子库：

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP
#GPU设备安装自定义算子
cd ./paddlenlp/csrc && python setup_cuda.py install
#XPU设备安装自定义算子
cd ./paddlenlp/csrc/xpu/src && sh cmake_build.sh
```

### 2.3 关闭BlockAttention的高性能推理

#### 2.3.1 动态图推理

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16

# PrefixTuning动态图推理参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --export_precache true --prefix_path ./checkpoints/llama_prefix_ckpts

# Weight Only Int8 动态图推理参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --quant_type weight_only_int8

# PTQ-A8W8推理命令参考
python ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16
```
**Note**：
1. LoRA 模型在推理之前是需要合并参数，详细可见：[合并 LoRA 参数](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/merge_lora_params.py)。
2. PrefixTuning推理需要传入相应的pre_cache，需要额外设置`export_precache`为`true`，并且传入对应的PrefixTuning参数保存路径`prefix_path`。
3. 使用Weight Only Int8 推理需要额外传入 `quant_type`。

#### 2.3.2 静态图推理
**step1：动转静**
```shell
# 动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16

# PrefixTuning动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --export_precache true

# Weight Only Int8 动转静命令参考
python ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --quant_type weight_only_int8

# PTQ-A8W8动转静命令参考
python ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16
```
**Note**：
1. LoRA 模型在推理之前是需要合并参数，详细可见：[合并 LoRA 参数](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/merge_lora_params.py)。
2. PrefixTuning推理需要传入相应的pre_cache，需要额外设置`export_precache`为`true`。
3. 使用Weight Only Int8 推理需要额外传入 `quant_type`。
4. A8W8推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型。

**step2：静态图推理**
```shell
# 静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --quant_type weight_only_int8 --dtype "float16" --mode "static"

# PrefixTuning静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --quant_type weight_only_int8 --dtype "float16" --mode "static" --export_precache true --prefix_path ./checkpoints/llama_prefix_ckpts

# Weight Only Int8 静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --quant_type weight_only_int8 --dtype "float16" --mode "static" --quant_type weight_only_int8

# PTQ-A8W8静态图推理命令参考
# 以下环境变量用于开启int8矩阵乘的算法选择以获得更快的推理速度，打开之后第一次执行会执行算法选择从而导致速度较慢。
export FLAGS_use_autotune=1
export FLAGS_cublaslt_exhaustive_search_times=10
export FLAGS_cache_inference_while_scope=1

python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --quant_type weight_only_int8 --dtype "float16" --mode "static"
```
**Note**：
1. LoRA 模型在推理之前是需要合并参数，详细可见：[合并 LoRA 参数](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/merge_lora_params.py)。
2. PrefixTuning推理需要传入相应的pre_cache，需要额外设置`export_precache`为`true`，并且传入对应的PrefixTuning参数保存路径`prefix_path`。
3. 使用Weight Only Int8 推理需要额外传入 `quant_type`。
4. A8W8推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型。


### 2.4 打开BlockAttention的高性能推理

#### 2.4.1 动态图推理

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
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --block_attn --cachekv_int8
```

#### 2.4.2 静态图推理
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
python ./predict/export_model.py  --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --output_path ./inference --dtype float16 --block_attn --cachekv_int8
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

# CacheKV 动态量化8静态图推理命令参考
python ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype "float16" --mode "static" --cachekv_int8 --block_attn
```
**Note**：
1. 使用Weight Only Int8 推理需要额外传入 `quant_type`。
2. A8W8推理传入的 `model_name_or_path` 为PTQ校准产出的量化模型。


## 3. 推理参数介绍

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
- `cachekv_int8`: 是否使用cachekv int8量化用于节省显存，默认值为False。
