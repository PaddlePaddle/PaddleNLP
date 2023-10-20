# 高性能推理

PaddleNLP 中已经添加高性能推理模型相关实现，支持：

|          Model          | Inference Model | PTuning | Wint8/4 | PTQ |
|:-----------------------:|:---------------:|:-------:|:-----:|:---:|
|   [LLaMA1/2](./llama)   |        ✅        |    ✅    |   ✅   |  ✅  |
|   [ChatGLM](./chatglm)  |        ✅        |    ✅    |   ✅   |  ❌  |
|  [ChatGLM2](./chatglm2) |        ✅        |    🚧    |   🚧   |  ❌  |
|     [Bloom](./bloom)    |        ✅        |    ✅    |   ✅   |  ❌  |
|     [GPT-3](./gpt-3)    |        ✅        |    ❌    |   ❌   |  ❌  |
|      [Qwen](./qwen)     |        ❌        |    ❌    |   ❌   |  ❌  |
| [BaiChuan2](./baichuan) |        ❌        |    ❌    |   ❌   |  ❌  |

[TOC]

## 预训练 & SFT 模型 & Lora 推理

> Lora 模型在推理之前是需要合并参数，详细可见：[合并 Lora 参数](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#37-lora-%E5%8F%82%E6%95%B0%E5%90%88%E5%B9%B6)。

预训练模型和 SFT 模型在结构上一样，推理功能包含：

* 动态图推理
* 静态图推理

### 动态图推理

```python
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --dtype float16
```

### 静态图推理

在静态图推理之前需要执行动转静，将模型转化为静态图，命令如下：

* 动转静

```python
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --output_path ./inference \
    --dtype float16
```

* 静态图推理

```python
python predictor.py \
    --model_name_or_path ./inference \
    --inference_model \
    --dtype "float16" \
    --mode "static"
```

## PTuning 模型推理

PTuning 模型和非 PTuning 模型推理非常类似，区别在于前者会添加 pre_caches.npy 的输入，动静推理命令可见：

### 动态图推理

```python
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --export_precache true \
    --prefix_path /path/to/pre_caches \
    --dtype float16
```

### 静态图推理

在静态图推理之前需要执行动转静，将模型转化为静态图，命令如下：

* 动转静

```python
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --export_precache true \
    --output_path ./inference_ptuning \
    --dtype float16
```

* 静态图推理

```python
python predictor.py \
    --model_name_or_path ./inference_ptuning \
    --inference_model \
    --dtype "float16" \
    --export_precache true \
    --prefix_path /path/to/pre_caches \
    --mode "static"
```

## Weight Only Int8/4 推理

Weight Only Int8/4 的推理脚本相比SFT 模型推理仅增加了：`quant_type`参数，值为：`weight_only_int8`和 `weight_only_int4`。

> 当前 weight_only_int8/4 仅支持A100，V100 上的 weight only int8/4 存在精度问题。

### 动态图推理

```python
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --quant_type weight_only_int8 \
    --dtype float16
```

### 静态图推理

在静态图推理之前需要执行动转静，将模型转化为静态图，命令如下：

* 动转静

```python
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --inference_model \
    --quant_type weight_only_int8 \
    --output_path ./inference \
    --dtype float16
```

* 静态图推理

```python
python predictor.py \
    --model_name_or_path ./inference \
    --inference_model \
    --quant_type weight_only_int8 \
    --dtype "float16" \
    --mode "static"
```

## 多卡推理

TODO: 未来将支持更多多卡推理文档说明

## FastLLMDeploy 部署

TODO: 未来将联合 [FastLLMDeploy](https://github.com/PaddlePaddle/FastDeploy) 给出更多生产环境下的高性能推理模型部署解决方案。

## 参数介绍

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
