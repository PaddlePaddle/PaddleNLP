# DeepSeek

本文档展示了如何在 PaddleNLP 中构建和运行[DeepSeek](https://www.deepseek.com/) 系列大模型。

## 模型介绍

* DeepSeek 系列大模型是由深度求索（DeepSeek Inc.）研发的高效开源语言模型，专注提升模型推理效率与多场景应用能力。

* [DeepSeek V3](https://www.deepseek.com/): 2024年12月，DeepSeek-V3首个版本上线并同步开源，DeepSeek-V3为 MoE 模型，671B 参数，激活37B。
* [DeepSeek R1](https://www.deepseek.com/): 2025年1月，深度求索发布 DeepSeek-R1，并同步开源模型权重。
* [DeepSeek R1 Distill Model](https://www.deepseek.com/): 2025年1月，深度求索在开源 R1模型的同时，通过 DeepSeek-R1的输出，蒸馏了6个小模型并开源，分别是 Qwen1.5B、7B、14B、32B 以及 Llama8B、70B。

## 已验证的模型

|Model|
|:-|
|deepseek-ai/DeepSeek-V3-Base|
|deepseek-ai/DeepSeek-V3|
|deepseek-ai/DeepSeek-R1|
|deepseek-ai/DeepSeek-R1-Zero|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|
|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|
|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|

## 已验证的预量化模型
|Model|
|:-|
|deepseek-ai/DeepSeek-V3-A8W8-FP8|

## 模型推理

deepseek-ai/DeepSeek-V3以单机 WINT8/FP8+WINT4、双机 FP8以及双机 WINT8为例；

deepseek-ai/DeepSeek-R1以单机 WINT8/FP8+WINT4为例；

DeepSeek R1系列蒸馏模型以 deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 的单机单卡 Wint8推理为例。

### deepseek-ai/DeepSeek-V3
单机 WINT8推理

```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```

单机 FP8+WINT4
```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```

双机 FP8
```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```

双机 WINT8
```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```

### deepseek-ai/DeepSeek-R1
单机 WINT8推理

```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```

单机 FP8+WINT4
```shell
# 动态图推理

# 动转静导出模型

# 静态图推理

```
### deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
单机单卡 WINT8推理

```shell
# 动态图推理
python ./predict/predictor.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 动转静导出模型
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type weight_only_int8
```
