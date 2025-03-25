# LLaMA

This document demonstrates how to build and run the [LLaMA](https://llama.meta.com/) series of large models in PaddleNLP.

## Model Introduction

* The LLaMA series of large models is an open and efficient collection of foundational language models released by Meta AI.

* [Llama 2](https://llama.meta.com/llama2/): In July 2023, Meta released the Llama 2 series with four versions: 7B, 13B, 34B, and 70B. This version enables commercial use under open-source license, reducing the cost for startups to create ChatGPT-like chatbots.

* [Llama 3](https://llama.meta.com/): On April 19, 2024, Meta launched the Llama 3 series, including 8B and 70B versions, with a 400B Llama-3 still in training. This version demonstrates comprehensive improvements across multiple benchmarks.

* [Llama 3.1](https://llama.meta.com/): On July 23, 2024, Meta released Llama 3.1 8B, 70B, and 405B models, further enhancing model performance and efficiency.

## Verified Models

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

## Verified Pre-quantized Models

|Model|
|:-|
|meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8|
|meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8|
|meta-llama/Meta-Llama-3.1-8B-Instruct-A8W8C8|
|meta-llama/Meta-Llama-3.1-8B-Instruct-A8W8-FP8|

## Model Inference

Taking meta-llama/Meta-Llama-3-8B-Instruct (single GPU) and meta-llama/Meta-Llama-3.1-405B-Instruct (multi-GPU) as examples.

BF16 Inference
```shell
# Dynamic Graph Inference
python ./predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1

# Dynamic to Static Model Exportation
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1

# Static Graph Inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1

```

WINT8 Inference

```shell
# Dynamic Graph Inference
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# Dynamic to Static Model Exportation
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type weight_only_int8

# Static Graph Inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type weight_only_int8
```

The following quantization inference requires models generated according to the [Large Model Quantization Tutorial](../quantization.md), such as checkpoints/llama_ptq_ckpts, or using pre-quantized models provided, e.g., meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8.

INT8-A8W8 Inference
```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8

# Dynamic to static model export
python predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8
```

INT8-A8W8C8 Inference

```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# Dynamic to static model export
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8C8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8 --cachekv_int8_type static
```

FP8-A8W8 Inference
```shell
# Dynamic graph inference
python predict/predictor.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8 --dtype bfloat16 --mode dynamic --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# Dynamic to static model export
python predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct-A8W8-FP8 --output_path /path/to/exported_model --dtype bfloat16 --inference_model 1 --append_attn 1 --quant_type a8w8_fp8

# Static graph inference
python predict/predictor.py --model_name_or_path /path/to/exported_model --dtype bfloat16 --mode static --inference_model 1 --append_attn 1 --quant_type a8w8_fp8
```

405B INT8-A8W8C8 TP8 Inference

```shell
# Due to model size, execute the following script to pre-download the model
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.generation import GenerationConfig
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
generation_config = GenerationConfig.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
```

Here we use fake parameters via --use_fake_parameter. For correctly inferring quantized models, please refer to the [Large Model Quantization Tutorial](../quantization.md) for quantization.
```shell
# Export model (Set paddle.set_device("cpu") in predict/export_model.py to export model via memory)
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" predict/export_model.py --model_name_or_path meta-llama/Meta-Llama-3.1-405B-Instruct --output_path /path/to/a8w8c8_tp8 --inference_model 1 --append_attn 1 --dtype bfloat16 --quant_type a8w8 --cachekv_int8_type static --use_fake_parameter 1

# Inference
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" predict/predictor.py --model_name_or_path /path/to/a8w8c8_tp8 --mode static --inference_model 1 --append_attn 1 --dtype bfloat16 --quant_type a8w8 --cachekv_int8_type static
```
