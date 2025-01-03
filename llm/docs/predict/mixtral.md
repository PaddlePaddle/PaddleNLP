# Mixtral

本文档展示了如何在 PaddleNLP 中构建和运行 [Mxtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 模型。

## 模型介绍


* [Mistral 系列](https://arxiv.org/abs/2310.06825) 是 Mistral AI 研发的基座大模型，使用了分组查询注意力和滑动窗口注意力机制来提高模型性能表现和推理速度，包括7B 不同规模的 Base 和 Instruct 模型。
* [Mixtral 系列](https://arxiv.org/abs/2401.04088) 是 Mistral AI 采用 MoE(Mixture of Experts)架构设计的基座大模型，在大多数基准测试中优于同级别的 llama 模型，MoE 结合了多个专家模型的优势来解决问题，在推理中仅需激活少量专家就可以达到非常好的效果，相比于传统大模型减少了较多的计算量；目前开源模型包括8x7B 和8x22B 两种不同规模的 Base 和 Instruct 模型。

## 已验证的模型

|Model|
|:-|
|mistralai/Mixtral-8x7B-v0.1-Instruct|


## 模型推理

下面以 Mixtral-8x7B-Instruct-v0.1两卡为例介绍整体推理流程。

BF16推理

```shell
# 动态图推理
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/predictor.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype bfloat16 \
    --mode "dynamic" \
    --inference_model \
    --append_attn

# 动转静导出模型
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output_path ./inference \
    --dtype bfloat16 \
    --inference_model \
    --append_attn

# 静态图推理
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

WINT8推理
```shell
# 动态图推理
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

# 动转静导出模型
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

# 静态图推理
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
