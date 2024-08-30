# Mixtral

本文档展示了如何在 PaddleNLP中构建和运行 [Mxtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 模型。

## 模型介绍


* [Mistral系列](https://arxiv.org/abs/2310.06825) 是Mistral AI研发的基座大模型，使用了分组查询注意力和滑动窗口注意力机制来提高模型性能表现和推理速度，包括7B不同规模的Base和Instruct模型。
* [Mixtral系列](https://arxiv.org/abs/2401.04088) 是Mistral AI采用MoE(Mixture of Experts)架构设计的基座大模型，在大多数基准测试中优于同级别的llama模型，MoE结合了多个专家模型的优势来解决问题，在推理中仅需激活少量专家就可以达到非常好的效果，相比于传统大模型减少了较多的计算量；目前开源模型包括8x7B和8x22B两种不同规模的Base和Instruct模型。

## 模型支持

|              Model              |
| :-----------------------------: |
| mistralai/Mixtral-8x7B-v0.1(-Instruct) |


## 模型推理

下面以Mixtral-8x7B-Instruct-v0.1两卡为例介绍整体推理流程。

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
    --block_attn

# 动转静导出模型
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    ./predict/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --output_path ./inference \
    --dtype bfloat16 \
    --inference_model \
    --block_attn

# 静态图推理
# 需要设置下面的环境变量，否则会导致多卡推理阻塞
export FLAGS_dynamic_static_unified_comm=false
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    predict/predictor.py \
    --model_name_or_path ./inference \
    --dtype bfloat16 \
    --mode "static" \
    --inference_model \
    --block_attn

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
    --block_attn

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
    --block_attn

# 静态图推理
export FLAGS_dynamic_static_unified_comm=false
export DEVICES=0,1
python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    predict/predictor.py \
    --model_name_or_path ./inference \
    --dtype bfloat16 \
    --quant_type weight_only_int8 \
    --mode "static" \
    --inference_model \
    --block_attn
```