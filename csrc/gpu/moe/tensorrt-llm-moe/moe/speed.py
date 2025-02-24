

import paddle

import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from paddle.incubate.nn.functional import (
    fused_bias_act,
    fused_layer_norm,
    fused_moe)

from paddle.nn.quant import weight_quantize

# import triton

from paddlenlp_ops import trt_llm_fused_moe
paddle.seed(2)


def GetQuantizedWeights(quant_method, w1, w2, arch=80):
    """
    Quantizes the weights for the experts' layers and returns the quantized weights and scales.
    :param quant_method: The quantization method to use.
    :return: Quantized bmm_w0, bmm_w1, scale0, scale1
    """
    num_expert = 256
    bmm_w0 = w1
    bmm_w1 = w2
    d_model = 7168
    d_feedforward = 128
    
    if quant_method != "None":
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = weight_quantize(bmm_w0[i], algo=quant_method)
            # print(fc0_expert_weights_for_ref_i.shape)
            # exit(0) [256, 7168]
            fc0_expert_weights_for_ref_list.append(
                fc0_expert_weights_for_ref_i.reshape(
                    [d_model, d_feedforward * 2]
                    if quant_method == "weight_only_int8"
                    else [d_model, d_feedforward]
                )
            )
            scale0.append(fc0_expert_weights_scale_for_ref_i)

        fc1_expert_weights_for_ref_list = []
        scale1 = []
        for i in range(num_expert):
            fc1_expert_weights_for_ref_i, fc1_expert_weights_scale_for_ref_i = weight_quantize(bmm_w1[i], algo=quant_method)
            fc1_expert_weights_for_ref_list.append(
                fc1_expert_weights_for_ref_i.reshape(
                    [d_feedforward, d_model]
                    if quant_method == "weight_only_int8"
                    else [d_feedforward, d_model // 2]
                )
            )
            scale1.append(fc1_expert_weights_scale_for_ref_i)
        
        bmm_w0_quantized = paddle.to_tensor(fc0_expert_weights_for_ref_list)
        bmm_w1_quantized = paddle.to_tensor(fc1_expert_weights_for_ref_list)
        scale0 = paddle.to_tensor(scale0)
        scale1 = paddle.to_tensor(scale1)
        
        return bmm_w0_quantized, bmm_w1_quantized, scale0, scale1
    else:
        return bmm_w0, bmm_w1, None, None

# Constants
DTYPES = paddle.bfloat16
M = 1024 # Batch size, token_num

TP = 16
N = 2048 // TP  # Intermediate size
K = 7168  # Hidden size
E = 256  # Number of experts
TOP_KS = 8
BLOCK_SIZE = [128, 128]  # Block-wise
SEEDS = 0

# Define parameters
topk = TOP_KS
block_size = BLOCK_SIZE

def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):
        # return paddle.randn(shape, dtype=dtype) / 10
        
    return paddle.empty(shape, dtype=dtype).normal_(mean, std)

# Gate logits (score after gate)
score = create_random_cuda_tensor([M, E], paddle.float32).cast("float32")

# Bfloat16 input
a = paddle.randn((M, K), dtype=paddle.bfloat16) / 100

# a = create_random_cuda_tensor([M, K], paddle.bfloat16, mean=0, std=0.01)

w1 = paddle.rand((E, K, 2 * N), dtype=paddle.bfloat16) / 100
w2 = paddle.rand((E , N, K), dtype=paddle.bfloat16)/ 100
# w1 = create_random_cuda_tensor([E, K, 2 * N], paddle.bfloat16, mean=0, std=0.01)
# w2 = create_random_cuda_tensor([E ,N, K], paddle.bfloat16, mean=0, std=0.01)

# print(w1)
print("((((((((((((((((((((((((((()))))))))))))))))))))))))))")

print(a.shape)
print(score.shape)
print(w1.shape)
print(w2.shape)

# [110, 2048]
# [110, 64]
# [64, 2048, 2816]
# [64, 1408, 2048]






def trt_bf16():
    paddle.device.synchronize()
    start = time.time()
    out = trt_llm_fused_moe(
            a,
            score,
            w1,
            w2,
            None,
            None,
            None,
            topk,
            0,
            "none",
            "Swiglu",
            8192,
        )
    paddle.device.synchronize()
    end = time.time()
    print(f"trt bf16 : {((end - start) * 1000)} ms")




def trt_win8(quant_method):
    bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method, w1, w2)
    # print(bmm_w0_quantized.shape)
    # print(bmm_w1_quantized.shape)
    # [256, 7168, 256]
    # [256, 128, 7168]
    
    paddle.device.synchronize()
    start = time.time()
    out = trt_llm_fused_moe(
            a,
            score,
            bmm_w0_quantized,
            bmm_w1_quantized,
            scale0,
            scale1,
            None,
            topk,
            0,
            quant_method,
            "Swiglu",
            1024,
        )
    paddle.device.synchronize()
    end = time.time()
    print(f"trt wint8 : {((end - start) * 1000)} ms")

# def paddle_bf16():
#     paddle.device.synchronize()
#     start = time.time()
#     fused_moe_out = fused_moe(
#                 a,
#                 score,
#                 w1,
#                 w2,
#                 None,
#                 None,
#                 None,
#                 None,
#                 # quant_method,
#                 quant_method,
#                 topk,
#                 False,
#             )
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"paddle bf16 : {((end - start) * 1000)} ms")


def paddle_win8(quant_method):
    # a, b = paddle.chunk(w1, 2, axis=-1)
    # trt_weight_1 = paddle.concat([b,a], axis=-1)
    bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method, w1, w2)
    paddle.device.synchronize()
    start = time.time()
    fused_moe_out = fused_moe(
                a,
                score,
                bmm_w0_quantized,
                bmm_w1_quantized,
                None,
                scale0,
                None,
                scale1,
                quant_method,
                topk,
                False,
            )
    paddle.device.synchronize()
    end = time.time()
    print(f"paddle bf16 : {((end - start) * 1000)} ms")


# for i in range(20):
#     paddle_bf16()


# for i in range(20):
#     trt_bf16()


for i in range(1):
    quant_method = "weight_only_int8"
    trt_win8(quant_method)

# for i in range(10):
#     quant_method = "weight_only_int8"
#     paddle_win8(quant_method)



#  def moe_fp8_no_block(i):
#     """Function to test FP8 per-tensor fused MoE."""
#     paddle.device.synchronize()
#     start = time.time()

#     out = fused_moe(
#         a, # bf16
#         w1_fp8,
#         w2_fp8,
#         score,
#         topk,
#         renormalize=True,
#         use_fp8_w8a8=True,
#         w1_scale=w1_s,
#         w2_scale=w2_s,
#     )
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"fp8 no block {i} : {((end - start) * 1000)} ms")


# Run tests
# for i in range(10):
#     moe(i)

# for i in range(1):
#     moe_fp8(i)

# for i in range(10):
#     moe_fp8_no_block(i)

