# Qwen

本文档展示了如何在 PaddleNLP 中构建和运行[Qwen](https://huggingface.co/Qwen) 系列大模型。

## 模型介绍

* [通义千问（Qwen）](https://arxiv.org/abs/2205.01068) 是阿里云研发的通义千问大模型系列的模型, 包括 Qwen-1.8B、Qwen-7B、Qwen-14B 和 Qwen-72B 等4个规模。Qwen 是基于 Transformer 的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

* [通义千问（Qwen1.5）](https://qwenlm.github.io/blog/qwen1.5/) 是阿里云研发的通义千问系列模型升级版。Qwen1.5包括0.5B、1.8B、4B、7B、14B、32B、72B 和110B 共计8个不同规模的 Base 和 Chat 模型。

* [通义千问（Qwen2）](https://qwenlm.github.io/blog/qwen2/) 是阿里云研发的通义千问系列模型升级版。Qwen2包括 Qwen2-0.5B、Qwen2-1.5B、Qwen2-7B、Qwen2-57B-A14B 以及 Qwen2-72B 共计5个不同规模的 Base 和 Instruct 模型。

* [通义千问（Qwen-MoE）](https://qwenlm.github.io/blog/qwen2/) 是阿里云研发的通义千问系列模型升级版。Qwen-MoE 包括 Qwen1.5-MoE-A2.7B 以及 Qwen2-57B-A14B 共计2个不同规模的 Base、Chat 和 Instruct 模型。

## 已验证的模型

|Model|
|:-|
|Qwen/Qwen2-0.5B-Instruct|
|Qwen/Qwen2-1.5B-Instruct|
|Qwen/Qwen2-7B-Instruct|
|Qwen/Qwen1.5-MoE-A2.7B-Chat|
|Qwen/Qwen2-57B-A14B-Instruct|
|Qwen/Qwen2.5-1.5B-Instruct|
|Qwen/Qwen2.5-7B-Instruct|
|Qwen/Qwen2.5-14B-Instruct|
|Qwen/Qwen2.5-32B-Instruct|
|Qwen/Qwen2.5-72B-Instruct|

## 已验证的预量化模型

|Model|
|:-|
|Qwen/Qwen2-1.5B-Instruct-A8W8C8|
|Qwen/Qwen2-1.5B-Instruct-A8W8-FP8|
|Qwen/Qwen2-7B-Instruct-A8W8C8|
|Qwen/Qwen2-7B-Instruct-A8W8-FP8|

## 模型推理

以 Qwen/Qwen2-1.5B-Instruct 为例。

BF16推理

```shell
# 动态图推理
python ./predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1

# 动转静导出模型
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1

```

WINT8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 动转静导出模型
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type weight_only_int8
```

下面量化推理所需要的模型需要根据[大模型量化教程](../quantization.md)产出，如 checkpoints/qwen_ptq_ckpts，或者使用所提供的预先量化好的模型，如 Qwen/Qwen2-1.5B-Instruct-A8W8C8。

INT8-A8W8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path checkpoints/qwen_ptq_ckpts --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8

# 动转静导出模型
python predict/export_model.py --model_name_or_path checkpoints/qwen_ptq_ckpts --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8
```

INT8-A8W8C8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct-A8W8C8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# 动转静导出模型
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct-A8W8C8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static
```

FP8-A8W8推理
```shell
# 动态图推理
python predict/predictor.py --model_name_or_path Qwen/Qwen2-7B-Instruct-A8W8-FP8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# 动转静导出模型
python predict/export_model.py --model_name_or_path Qwen/Qwen2-7B-Instruct-A8W8-FP8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8_fp8
```
