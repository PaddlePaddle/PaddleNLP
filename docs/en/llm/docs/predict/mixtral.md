# Mixtral

This document demonstrates how to build and run the [Mxtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model in PaddleNLP.

## Model Introduction

* [Mistral Series](https://arxiv.org/abs/2310.06825) are foundation models developed by Mistral AI, using Grouped Query Attention and Sliding Window Attention mechanisms to improve performance and inference speed, including 7B-scale Base and Instruct models.
* [Mixtral Series](https://arxiv.org/abs/2401.04088) are foundation models designed by Mistral AI using MoE (Mixture of Experts) architecture, outperforming comparable llama models in most benchmarks. MoE combines the advantages of multiple expert models to solve problems, requiring only activation of a small number of experts during inference to achieve excellent results, significantly reducing computational requirements compared to traditional large models. Current open-source models include 8x7B and 8x22B-scale Base and Instruct models.

## Verified Models

|Model|
|:-|
|mistralai/Mixtral-8x7B-Instruct-v0.1|

## Model Inference

The following example demonstrates the complete inference workflow for Mixtral-8x7B-Instruct-v0.1 on 2 GPUs.

BF16 Inference

```shell
# Dynamic graph inference
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/predictor.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype bfloat16 \
    --mode "dynamic" \
    --inference_model \
    --append_attn

# Convert dynamic graph to static graph
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output_path ./inference \
    --dtype bfloat16 \
    --inference_model \
    --append_attn

# Static graph inference
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    predict/predictor.py \
    --model_name_or_path ./inference \
    --dtype bfloat16 \
    --mode "static" \
    --inference_model \
    --append_attn
```

WINT8 Inference
```shell
# Dynamic Graph Inference
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/predictor.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype bfloat16 \
    --quant_type "weight_only_int8" \
    --mode "dynamic" \
    --inference_model \
    --append_attn

# Export Model with Dynamic to Static
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output_path ./inference \
    --dtype bfloat16 \
    --quant_type weight_only_int8 \
    --inference_model \
    --append_attn

# Static Graph Inference
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    predict/predictor.py \
    --model_name_or_path ./inference \
    --dtype bfloat16 \
    --quant_type weight_only_int8 \
    --mode "static" \
    --inference_model \
    --append_attn
```
