

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

# export PYTHONPATH=$PYTHONPATH:/home/gaoziyuan/PaddleNLP

import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import paddle
# import triton

from paddlenlp_ops import trt_llm_fused_moe


# from csrc.gpu.moe.fused_moe_triton import fused_moe

# Constants
DTYPES = paddle.bfloat16
M = 8 # Batch size, token_num

TP = 16
N = 2048 // TP  # Intermediate size
K = 4096  # Hidden size
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
a = paddle.randn((M, K), dtype=paddle.bfloat16) / 10
w1 = paddle.rand((E, K, 2 * N), dtype=paddle.bfloat16)  / 10
w2 = paddle.rand((E , N, K), dtype=paddle.bfloat16)  / 10

# FP8 scaling
factor_for_scale = 1e-2
fp8_max, fp8_min = 448.0, -448.0
w1_ = (w1 - 0.5) * 2 * fp8_max
w2_ = (w2 - 0.5) * 2 * fp8_max
w1_fp8 = w1_.clip(min=fp8_min, max=fp8_max).to(paddle.float8_e4m3fn)
w2_fp8 = w2_.clip(min=fp8_min, max=fp8_max).to(paddle.float8_e4m3fn)

# Calculate block tiles for w1 and w2
block_n, block_k = block_size
n_tiles_w1 = (2 * N + block_n - 1) // block_n
n_tiles_w2 = (K + block_n - 1) // block_n
k_tiles_w1 = (K + block_k - 1) // block_k
k_tiles_w2 = (N + block_k - 1) // block_k

# Scale for w1 and w2
w1_s = paddle.rand((E, n_tiles_w1, k_tiles_w1), dtype=paddle.float32) * factor_for_scale
w2_s = paddle.rand((E, n_tiles_w2, k_tiles_w2), dtype=paddle.float32) * factor_for_scale
# print(w1_s.shape)

# w1_s = paddle.rand((E, n_tiles_w1 * k_tiles_w1), dtype=paddle.float32) * factor_for_scale
# w2_s = paddle.rand((E, n_tiles_w2 * k_tiles_w2), dtype=paddle.float32) * factor_for_scale


# def moe(i):
#     """Function to test bfloat16 fused MoE."""
#     paddle.device.synchronize()
#     start = time.time()
#     out = fused_moe(a, w1, w2, score, topk, renormalize=False)
#     paddle.device.synchronize()
#     end = time.time()
#     print(f"bf16 {i} : {((end - start) * 1000)} ms")


def moe_fp8(i):
    """Function to test FP8 block-wise fused MoE."""
    paddle.device.synchronize()
    start = time.time()

    activation_str = "Swiglu"
    # activation_str = "silu"
    quant_method = "fp8_block_wise"
    active_rows = M
    
    out = trt_llm_fused_moe(
            a,
            score,
            w1_fp8,
            w2_fp8,
            # w1_s.reshape([E, k_tiles_w1, -1]),
            # w2_s.reshape([E, k_tiles_w2, -1]),
            w1_s,
            w2_s,
            None,
            topk,
            0,
            quant_method,
            # "Swiglu"
            # "Swiglu"
            "silu"
        )

    paddle.device.synchronize()
    end = time.time()
    print(out)
    if paddle.isnan(out).sum().item() > 0:
        print("fuck !")
    print(f"fp8 {i} : {((end - start) * 1000)} ms")


# def moe_fp8_no_block(i):
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

for i in range(1):
    moe_fp8(i)

# for i in range(10):
#     moe_fp8_no_block(i)

