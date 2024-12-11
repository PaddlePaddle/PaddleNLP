# LLaMA

本文档展示了如何在 PaddleNLP 中构建和运行[LLaMA](https://llama.meta.com/) 系列大模型。

## 模型介绍

* LLaMA 系列大模型是由 Meta AI 发布的一个开放且高效的大型基础语言模型。

* [Llama 2](https://llama.meta.com/llama2/)：2023年7月，Meta 发布了 Llama 2系列，有7B、13B、34B 和70B 四个版本。该版本实现了开源商用，降低了初创公司创建类似 ChatGPT 聊天机器人的成本。

* [Llama 3](https://llama.meta.com/)：2024年4月19日，Meta 推出了 Llama 3系列，包括8B 和70B 两个版本，400B 的 Llama-3还在训练中。该版本在多个基准测试中取得了全面进步，性能优异。

* [Llama 3.1](https://llama.meta.com/)：2024年7月23日，Meta 发布了 Llama 3.1 8B、70B、405B 模型，进一步提升了模型的性能和效率。

## 已验证的模型

|Model|
|:-|
|meta-llama/Llama-2-7b-chat|
|meta-llama/Llama-2-13b-chat|
|meta-llama/Llama-2-70b-chat|
|meta-llama/Meta-Llama-3-8B-Instruct|
|meta-llama/Meta-Llama-3-70B-Instruct|
|meta-llama/Meta-Llama-3.1-8B-Instruct|
|meta-llama/Meta-Llama-3.1-70B-Instruct|
|meta-llama/Meta-Llama-3.1-405B-Instruct|
|meta-llama/Llama-3.2-3B-Instruct|

## 已验证的预量化模型

|Model|
|:-|
|meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8|
|meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8|
|meta-llama/Meta-Llama-3.1-8B-Instruct-A8W8C8|
|meta-llama/Meta-Llama-3.1-8B-Instruct-A8W8-FP8|


## 模型推理

以 meta-llama/Meta-Llama-3-8B-Instruct 单卡和 meta-llama/Meta-Llama-3.1-405B-Instruct 多卡为例。

BF16推理

```shell
# 动态图推理
python ./predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1

# 动转静导出模型
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1

```

WINT8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 动转静导出模型
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type weight_only_int8
```

下面量化推理所需要的模型需要根据[大模型量化教程](../quantization.md)产出，如 checkpoints/llama_ptq_ckpts，或者使用所提供的预先量化好的模型，如 meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8。

INT8-A8W8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8

# 动转静导出模型
python predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8
```

INT8-A8W8C8推理

```shell
# 动态图推理
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# 动转静导出模型
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static
```

FP8-A8W8推理
```shell
# 动态图推理
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# 动转静导出模型
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# 静态图推理
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8_fp8
```

405B INT8-A8W8C8 TP8推理

```shell
# 由于模型较大，可执行如下脚本预先下载模型
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.generation import GenerationConfig
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
generation_config = GenerationConfig.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
```

这里通过--use_fake_parameter 使用 fake parameters，如需要推理正确的量化模型，请自行参考[大模型量化教程](../quantization.md)进行量化。

```shell
# 导出模型 (可在predict/export_model.py中设置paddle.set_device("cpu")，通过内存导出模型)
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3.1-405B-Instruct --output_path /path/to/a8w8c8_tp8 --inference_model 1 --append_attn 1 --dtype bfloat16 --quant_type a8w8 --cachekv_int8_type static --use_fake_parameter 1

# 推理
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" predict/predictor.py --model_name_or_path /path/to/a8w8c8_tp8 --mode static --inference_model 1 --append_attn 1 --dtype bfloat16 --quant_type a8w8 --cachekv_int8_type static
```
