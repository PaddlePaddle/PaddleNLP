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

import time

import paddle  # 初始化分布式环境

from paddlenlp.ops.triton_ops.fused_moe import fused_moe

# Constants
DTYPES = paddle.bfloat16
M = 64  # Batch size, token_num 32k 32 64
TP = 16
N = 2048 // TP  # Intermediate size
# N = 2048 # 128 256
K = 7168  # Hidden size 7168 512 1024 2048
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
# chosen_num = 8
# score[:, chosen_num:] -= 5

# Bfloat16 input
a = paddle.randn((M, K), dtype=paddle.bfloat16) / 10
w1 = paddle.rand((E, 2 * N, K), dtype=paddle.bfloat16)
w2 = paddle.rand((E, K, N), dtype=paddle.bfloat16)

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


def moe_fp8():
    """Function to test FP8 block-wise fused MoE."""
    paddle.device.synchronize()
    start = time.time()
    out = fused_moe(
        a,
        w1_fp8,
        w2_fp8,
        score,
        topk,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
    )
    paddle.device.synchronize()
    end = time.time()
    print(f"fp8 triton moe : {((end - start) * 1000)} ms")
    start_dg = time.time()
    dg_out = fused_moe(
        a,
        w1_fp8,
        w2_fp8,
        score,
        topk,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
        use_dg=True,
    )
    paddle.device.synchronize()
    end_dg = time.time()
    print(f"fp8 moe dg : {((end_dg - start_dg) * 1000)} ms")

    for i in range(M):
        row_diff = paddle.mean(
            paddle.abs(dg_out.to(paddle.float32)[i] - out.to(paddle.float32)[i])
            / paddle.mean(paddle.abs(out.to(paddle.float32)[i]))
        )
        if row_diff > 0.03:
            print(f"Row {i} difference: {row_diff}")

    rel_diff = paddle.mean(paddle.abs(dg_out.to(paddle.float32) - out.to(paddle.float32))) / paddle.mean(
        paddle.abs(out.to(paddle.float32))
    )
    print(f"Relative difference: {rel_diff}")
    assert rel_diff < 0.03


def moe_fp8_tl():
    """Function to test FP8 block-wise fused MoE."""
    fused_moe(
        a,
        w1_fp8,
        w2_fp8,
        score,
        topk,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
    )


def moe_fp8_dg():
    """Function to test FP8 block-wise fused MoE."""
    fused_moe(
        a,
        w1_fp8,
        w2_fp8,
        score,
        topk,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
        use_dg=True,
    )


moe_fp8()

# for _ in range(100):
#     moe_fp8_tl()

# for _ in range(100):
#     moe_fp8_dg()

# moe_fp8_dg()
