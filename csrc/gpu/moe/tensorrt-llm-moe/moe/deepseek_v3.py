# export PYTHONPATH=$PYTHONPATH:/home/gaoziyuan/PaddleNLP
import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import paddle
from paddlenlp_ops import trt_llm_fused_moe


# Constants
DTYPES = paddle.bfloat16
M = 32 # Batch size, token_num

TP = 8
N = 2048 // TP # Intermediate size 256
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
input_a = paddle.randn((M, K), dtype=paddle.bfloat16) / 10
w1 = paddle.rand((E, 2 * N,K), dtype=paddle.bfloat16) 
w2 = paddle.rand((E ,K, N), dtype=paddle.bfloat16)

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

# w1_fp8 = paddle.rand([E, 2 * N, K]).to(paddle.float8_e4m3fn) / 10
# w2_fp8 = paddle.rand([E, K, N]).to(paddle.float8_e4m3fn) / 10
w1_s = paddle.rand((E, n_tiles_w1, k_tiles_w1), dtype=paddle.float32)* factor_for_scale
w2_s = paddle.rand((E, k_tiles_w2, n_tiles_w2), dtype=paddle.float32)* factor_for_scale


paddle.device.synchronize()
start = time.time()

quant_method = "fp8_block_wise"

out = trt_llm_fused_moe(
        input_a, # input
        score, # gate logits
        w1_fp8,
        w2_fp8,
        w1_s, # w1 scale
        w2_s, # w2 scale
        None,
        topk,
        0,
        quant_method, # fp8_block_wise
        "Swiglu"
    )

paddle.device.synchronize()
end = time.time()
print(out)
if paddle.isnan(out).sum().item() > 0:
    print("nan !!!!!!")
print(f"fp8 : {((end - start) * 1000)} ms")










# /root/paddlejob/workspace/env_run/output/gaoziyuan/2023.1.1/bin/nsys profile -t cuda,osrt,nvtx -o paddle.bs1_0829 -w true --force-overwrite true python /root/paddlejob/workspace/env_run/output/gaoziyuan/PaddleNLP/csrc/gpu/moe/tensorrt-llm-moe/moe/deepseek_v3.py 2>&1 |tee run_deep.log
# /root/paddlejob/workspace/env_run/output/gaoziyuan/2023.1.1/bin/nsys profile -t cuda,osrt,nvtx -o paddle.bs1_0829 -w true --force-overwrite true python /root/paddlejob/workspace/env_run/output/gaoziyuan/PaddleNLP/csrc/gpu/moe/tensorrt-llm-moe/moe/speed.py

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

# for i in range(1):
#     moe_fp8(i)

# for i in range(10):
#     moe_fp8_no_block(i)

