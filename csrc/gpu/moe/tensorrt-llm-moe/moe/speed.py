

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

# import triton

from paddlenlp_ops import trt_llm_fused_moe
paddle.seed(2)

# Constants
DTYPES = paddle.bfloat16
M = 1 # Batch size, token_num

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
            "Swiglu"
        )
    paddle.device.synchronize()
    end = time.time()
    print(f"trt bf16 : {((end - start) * 1000)} ms")

def paddle_bf16():
    paddle.device.synchronize()
    start = time.time()
    fused_moe_out = fused_moe(
                a,
                score,
                w1,
                w2,
                None,
                None,
                None,
                None,
                # quant_method,
                "none",
                topk,
                False,
            )
    paddle.device.synchronize()
    end = time.time()
    print(f"paddle bf16 : {((end - start) * 1000)} ms")


for i in range(10):
    paddle_bf16()


# for i in range(10):
#     trt_bf16()






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

