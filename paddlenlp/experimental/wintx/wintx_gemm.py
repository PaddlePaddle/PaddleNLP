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

import paddle
import triton
import triton.language as tl

from .utils import dequant, swizzle_tile, unpack_bs

__all__ = [
    "gemm_int4_paddle",
]


def get_default_config():
    # 4090: default
    config = triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "SPLIT_K": 1,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=1,
    )

    return [config]


def get_wintx_kernel_config():
    configs = []
    for num_stages in [1]:
        for block_m in [32, 64, 128, 256]:
            for block_n in [32, 64, 128]:  # , 256, 512]:
                for block_k in [32, 64, 128]:  # , 256, 512]:
                    for split_k in [1]:  # , 2, 4, 8]:
                        for warps in [8]:
                            configs.append(
                                triton.Config(
                                    {
                                        "SPLIT_K": split_k,
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": 8,
                                        "num_stages": num_stages,
                                        "num_warps": warps,
                                        # "pre_hook": init_to_zero("c_ptr")
                                    },
                                )
                            )
    return configs


@triton.autotune(
    configs=get_default_config(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bsk,
    stride_bsn,
    group_size: tl.constexpr,
    n_bit: tl.constexpr,
    w_mask: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)

    pack_num: tl.constexpr = 32 // n_bit
    bzp = 1 << (n_bit - 1)

    # swizzle_tile, maybe work...
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # set A/B offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_ak = offs_k

    offs_bk = pid_sp_k * (BLOCK_SIZE_K // pack_num) + tl.arange(0, BLOCK_SIZE_K // pack_num)

    group_nums: tl.constexpr = (BLOCK_SIZE_K - 1) // group_size + 1
    offs_bzk = pid_sp_k * BLOCK_SIZE_K // group_size + tl.arange(0, group_nums)

    # set A/B ptrs
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_mask = offs_am[:, None] < M
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    b_shift_bits = ((offs_k[:, None] % pack_num) * n_bit).to(tl.int32)
    # bzp_shift_bits = ((offs_bn[None, :] % pack_num) * 2).to(tl.int32)

    for k in tl.range(tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):  # range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):

        bs_ptrs = (
            bs_ptr
            + ((offs_bzk[:, None] + k * BLOCK_SIZE_K * SPLIT_K // group_size)) * stride_bsk
            + offs_bn[None, :] * stride_bsn
        )

        bs = tl.load(bs_ptrs)
        # bzp = tl.load(bzp_ptrs)
        bs = unpack_bs(bs, BLOCK_SIZE_N, BLOCK_SIZE_K, group_nums)

        # bzp = (bzp >> bzp_shift_bits) & 0x3

        b = tl.load(b_ptrs, eviction_policy="evict_first")
        # dequant
        b = dequant(b, bs, bzp, b_shift_bits, BLOCK_SIZE_N, BLOCK_SIZE_K, pack_num, w_mask)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        accumulator += tl.dot(a, b.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk // pack_num  # assert BLOCK_SIZE_K % 4 == 0

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def gemm_int2_paddle(x, qw, scales, group_size=None, output=None):
    assert x.is_contiguous(), "A must be contiguous"
    assert qw.is_contiguous(), "B must be contiguous"

    M, K = x.shape
    N = qw.shape[1]

    if group_size is None:
        group_size = K // scales.shape[0]

    if output is None:
        output = paddle.zeros([M, N], dtype=x.dtype)
        # output = paddle.empty([M,N], dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )

    x_stride0, x_stride1 = x.shape[1], 1
    qw_stride0, qw_stride1 = qw.shape[1], 1
    scales_stride0, scales_stride1 = scales.shape[1], 1
    output_stride0, output_stride1 = output.shape[1], 1

    n_bit = 2
    w_mask = 0x3

    gemm_kernel[grid](
        x,
        qw,
        output,
        scales,
        M,
        N,
        K,
        x_stride0,
        x_stride1,
        qw_stride0,
        qw_stride1,
        output_stride0,
        output_stride1,
        scales_stride0,
        scales_stride1,
        group_size,
        n_bit,
        w_mask,
    )
    return output


def gemm_int4_paddle(x, qw, scales, group_size=None, output=None):
    assert x.is_contiguous(), "A must be contiguous"
    assert qw.is_contiguous(), "B must be contiguous"

    M, K = x.shape
    N = qw.shape[1]

    if group_size is None:
        group_size = K // scales.shape[0]

    if output is None:
        output = paddle.empty([M, N], dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )

    x_stride0, x_stride1 = x.shape[1], 1
    qw_stride0, qw_stride1 = qw.shape[1], 1
    scales_stride0, scales_stride1 = scales.shape[1], 1
    output_stride0, output_stride1 = output.shape[1], 1

    n_bit = 4
    w_mask = 0xF
    gemm_kernel[grid](
        x,
        qw,
        output,
        scales,
        M,
        N,
        K,
        x_stride0,
        x_stride1,
        qw_stride0,
        qw_stride1,
        output_stride0,
        output_stride1,
        scales_stride0,
        scales_stride1,
        group_size,
        n_bit,
        w_mask,
    )
    return output
