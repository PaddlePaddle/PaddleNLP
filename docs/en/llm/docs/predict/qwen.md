# Qwen

This document demonstrates how to build and run [Qwen](https://huggingface.co/Qwen) series of large models in PaddleNLP.

## Model Introduction

* [Qwen](https://arxiv.org/abs/2205.01068) is the model series developed by Alibaba Cloud, including Qwen-1.8B, Qwen-7B, Qwen-14B and Qwen-72B. Qwen is a Transformer-based large language model trained on massive pretraining data. The pretraining data contains various types of text including web texts, professional books, and code.

* [Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/) is an upgraded version of Qwen series developed by Alibaba Cloud. Qwen1.5 includes 8 models at different scales: 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B and 110B, with both Base and Chat versions.

* [Qwen2](https://qwenlm.github.io/blog/qwen2/) is the next generation of Qwen series developed by Alibaba Cloud. Qwen2 includes 5 models at different scales: Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B and Qwen2-72B, with both Base and Instruct versions.

* [Qwen-MoE](https://qwenlm.github.io/blog/qwen2/) is the MoE version of Qwen series developed by Alibaba Cloud. Qwen-MoE includes 2 models at different scales: Qwen1.5-MoE-A2.7B and Qwen2-57B-A14B, with Base, Chat and Instruct versions.

## Verified Models

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

## Verified Pre-quantized Models

|Model|
|:-|
|Qwen/Qwen2-1.5B-Instruct-A8W8C8|
|Qwen/Qwen2-1.5B-Instruct-A8W8-FP8|
|Qwen/Qwen2-7B-Instruct-A8W8C8|
|Qwen/Qwen2-7B-Instruct-A8W8-FP8|

## Model Inference

Take Qwen/Qwen2-1.5B-Instruct as example.

BF16 Inference:

```python
from paddlenlp import Qwen2ForCausalLM, Qwen2Tokenizer

model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", dtype="bfloat16")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

inputs = tokenizer("Human: Hello\nAssistant:", return_tensors="pd")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.batch_decode(outputs[0]))
```
```shell
# Dynamic graph inference
python ./predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1

# Dynamic to static model export
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1

```

WINT8 Inference

```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# Dynamic to static model export
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type weight_only_int8
```

The following quantization inference requires models to be produced according to the [Large Model Quantization Tutorial](../quantization.md), such as checkpoints/qwen_ptq_ckpts, or using provided pre-quantized models like Qwen/Qwen2-1.5B-Instruct-A8W8C8.

INT8-A8W8 Inference
```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path checkpoints/qwen_ptq_ckpts --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8

# Dynamic to static model export
python predict/export_model.py --model_name_or_path checkpoints/qwen_ptq_ckpts --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8
```

INT8-A8W8C8 Inference

```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct-A8W8C8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# Dynamic to static model export
python predict/export_model.py --model_name_or_path Qwen/Qwen2-1.5B-Instruct-A8W8C8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static
```

FP8-A8W8 Inference
```shell
# Dynamic Graph Inference
python predict/predictor.py --model_name_or_path Qwen/Qwen2-7B-Instruct-A8W8-FP8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# Export model via dynamic to static
python predict/export_model.py --model_name_or_path Qwen/Qwen2-7B-Instruct-A8W8-FP8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# Static Graph Inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8_fp8
```
