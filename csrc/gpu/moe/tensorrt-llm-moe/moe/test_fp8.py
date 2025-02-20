# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import paddle



from paddlenlp_ops import trt_llm_fused_moe


import sys
sys.path.append('/home/gaoziyuan/PaddleNLP')
from csrc.gpu.moe.fused_moe_triton.fused_moe import fused_moe



# Constants
DTYPES = paddle.bfloat16
# DTYPES = paddle.float32
# DTYPES = paddle.float8_e4m3fn
M = 8  # Batch size, token_num
TP = 16
N = 2048 // TP  # Intermediate size
K = 7168  # Hidden size
E = 256  # Number of experts
TOP_KS = 8
BLOCK_SIZE = [128, 128]  # Block-wise
SEEDS = 0

# Set seed for reproducibility
paddle.seed(SEEDS)

# Define parameters
topk = TOP_KS
block_size = BLOCK_SIZE

# Gate logits (score after gate)
score = paddle.randn((M, E), dtype=paddle.float32)

# Bfloat16 input
a = paddle.randn((M, K), dtype=DTYPES) / 10
w1 = paddle.rand((E, 2 * N, K), dtype=DTYPES)
w2 = paddle.rand((E, K, N), dtype=DTYPES)


# # FP8 scaling
factor_for_scale = 1e-2
fp8_max, fp8_min = 448.0, -448.0
w1_ = (w1 - 0.5) * 2 * fp8_max
w2_ = (w2 - 0.5) * 2 * fp8_max
w1_fp8 = w1_.clip(min=fp8_min, max=fp8_max).to(paddle.float8_e4m3fn)
w2_fp8 = w2_.clip(min=fp8_min, max=fp8_max).to(paddle.float8_e4m3fn)


a_ = (a - 0.5) * 2 * fp8_max
a_fp8 = a_.clip(min=fp8_min, max=fp8_max).to(paddle.float8_e4m3fn)




# # Calculate block tiles for w1 and w2
# block_n, block_k = block_size
# n_tiles_w1 = (2 * N + block_n - 1) // block_n
# n_tiles_w2 = (K + block_n - 1) // block_n
# k_tiles_w1 = (K + block_k - 1) // block_k
# k_tiles_w2 = (N + block_k - 1) // block_k

# # Scale for w1 and w2
# w1_s = paddle.rand((E, n_tiles_w1, k_tiles_w1), dtype=paddle.float32) * factor_for_scale
# w2_s = paddle.rand((E, n_tiles_w2, k_tiles_w2), dtype=paddle.float32) * factor_for_scale


factor_for_scale = 1e-2
scale1 = paddle.rand((E,), dtype=paddle.float32) * factor_for_scale
scale2 = paddle.to_tensor(2, dtype=paddle.float32) * factor_for_scale
scale3 = paddle.rand((E,), dtype=paddle.float32) * factor_for_scale


def moe(i):
    """Function to test bfloat16 fused MoE."""
    paddle.device.synchronize()

    activation_str = "Swiglu"
    quant_method = "fp8_pre_tensor"
    active_rows = M

    start = time.time()
    # w1_reshape = w1.reshape([E, K, -1])
    # w2_reshape = w2.reshape([E, -1, K])
    # out = fused_moe(a, w1, w2, score, topk, renormalize=False)

    out = trt_llm_fused_moe(
            a_fp8,
            score,
            w1_fp8.reshape([E, K, -1]),
            w2_fp8.reshape([E, -1, K]),
            scale1,
            scale2,
            scale3,
            activation_str,
            active_rows,
            topk,
            quant_method,
        )
    # out_2 = fused_moe(a, w1, w2, score, topk, renormalize=True)
    print(paddle.max(out))
    # print(out_2)
    # print(out - out_2)
    paddle.device.synchronize()
    end = time.time()
    print(f"fp32 {i} : {((end - start) * 1000)} ms")


# def moe_triton(i):
#     """Function to test bfloat16 fused MoE."""
#     paddle.device.synchronize()

   
#     start = time.time()
   
#     out = fused_moe(a, w1, w2, score, topk, renormalize=False)
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"triton fp32 {i} : {((end - start) * 1000)} ms")


# def moe_fp8(i):
#     """Function to test FP8 block-wise fused MoE."""
#     paddle.device.synchronize()
#     start = time.time()
#     out = fused_moe(
#         a,
#         w1_fp8,
#         w2_fp8,
#         score,
#         topk,
#         renormalize=False,
#         use_fp8_w8a8=True,
#         w1_scale=w1_s,
#         w2_scale=w2_s,
#         block_shape=block_size,
#     )
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"fp8 {i} : {((end - start) * 1000)} ms")


# def moe_fp8_no_block(i):
#     """Function to test FP8 per-tensor fused MoE."""
#     paddle.device.synchronize()
#     start = time.time()
#     out = fused_moe(
#         a,
#         w1_fp8,
#         w2_fp8,
#         score,
#         topk,
#         renormalize=False,
#         use_fp8_w8a8=True,
#         w1_scale=w1_s.reshape([E, -1]),
#         w2_scale=w2_s.reshape([E, -1]),
#     )
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"fp8 no block {i} : {((end - start) * 1000)} ms")


# Run tests
for i in range(1):
    moe(i)

# for i in range(10):
#     moe_triton(i)

# for i in range(10):
#     moe_fp8_no_block(i)