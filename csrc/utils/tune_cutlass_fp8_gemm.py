# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import paddle
from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_fused


def setup_args():
    """Setup export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_min", type=int, help="range of gemm shape: m_min")
    parser.add_argument("--m_max", type=int, help="range of gemm shape: m_max")
    parser.add_argument("--n", nargs="+", type=int, help="List of gemm shape: n")
    parser.add_argument("--k", nargs="+", type=int, help="List of gemm shape: k")
    args = parser.parse_args()
    return args


def gemm(m, n, k):
    A = paddle.ones([m, k], dtype="float8_e4m3fn")
    B = paddle.ones([n, k], dtype="float8_e4m3fn")
    res = cutlass_fp8_fp8_half_gemm_fused(
        A, B, bias=None, transpose_x=False, transpose_y=True, output_dtype="bfloat16", scale=0.5, act="identity"
    )
    return res


if __name__ == "__main__":
    args = setup_args()

    m_min = args.m_min
    m_max = args.m_max
    ns = args.n
    ks = args.k

    for m in range(m_min, m_max + 32, 32):
        if m > m_max:
            break
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            gemm(m, n, k)
            paddle.device.cuda.empty_cache()
