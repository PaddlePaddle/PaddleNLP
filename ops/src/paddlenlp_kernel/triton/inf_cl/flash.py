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

"""
this code is modified from https://github.com/DAMO-NLP-SG/Inf-CLIP/blob/main/inf_cl/flash.py
"""
import math

import numpy as np
import paddle
import paddle.autograd
import paddle.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _prob_fwd_kernel(
    Q,
    K,
    LSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # start index of sequence length
    start_m = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # loop over k, v and update accumulator
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        # Trying to combine the two masks seem to make the result wrong
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        # Fix out of bound access
        p = tl.where((start_n + offs_n)[None, :] < seqlen_k, p, 0.0)
        # -- update statistics
        lse_i = tl.exp(m_i - m_ij) * lse_i + tl.sum(p, 1)
        m_i = m_ij

    lse_i = m_i + tl.log(lse_i)
    # mask out the padded values
    lse_i = tl.where(offs_m < seqlen_q, lse_i, 0.0)

    tl.store(LSE + offs_m, lse_i)


@triton.jit
def _dq_prob_bwd_kernel(
    Q,
    K,
    dQ,
    LSE,
    dLSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    # start index of sequence length
    start_m = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + ndims * offs_m[:, None]
    dq_ptrs = dQ + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    # setting lse
    lse = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    dlse = tl.load(dLSE + offs_m, mask=offs_m < seqlen_q, other=0.0)

    # loop over k, v and update accumulator
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk_grad, 0.0)
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, "=r, r", [qk_grad], dtype=tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute q grad ----
            # NOTE: tl.float32 adopt tf32, which causes precision inconsistency with torch
            # A solution for this problem
            # Refer to issue: https://github.com/triton-lang/triton/issues/4574
            # if allow_tf32:
            k = tl.inline_asm_elementwise(ASM, "=r, r", [k], dtype=tl.float32, is_pure=True, pack=1)
            q_grad = tl.dot(qk_grad, k)
            # Another solution for this problem
            # Refer to https://github.com/triton-lang/triton/issues/376
            # q_grad = tl.dot(qk_grad, k.to(tl.float32), allow_tf32=False)
            # -- store dq ----
            dq_h = tl.load(dq_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q, other=0.0)
            tl.store(dq_ptrs + offs_hd, dq_h + q_grad, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _dk_prob_bwd_kernel(
    Q,
    K,
    dK,
    LSE,
    dLSE,
    nheads,
    seqlen_q,
    seqlen_k,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    # start index of sequence length
    start_n = tl.program_id(0)

    # initialize offsets
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    dk_ptrs = dK + ndims * offs_n[:, None]

    # loop over q and update accumulator
    end_m = seqlen_q
    for start_m in range(0, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        # setting lse
        lse = tl.load(LSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0)
        dlse = tl.load(dLSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(offs_m + start_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute qk ----
            qk += tl.dot(q, tl.trans(k))

        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_m + offs_m)[:, None] < seqlen_q, qk_grad, 0.0)
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, "=r, r", [qk_grad], dtype=tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            # -- fetch q and k of a single head ----
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(start_m + offs_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            # -- compute k grad ----
            q = tl.inline_asm_elementwise(ASM, "=r, r", [q], dtype=tl.float32, is_pure=True, pack=1)
            k_grad = tl.dot(tl.trans(qk_grad), q)
            # k_grad = tl.dot(tl.trans(qk_grad), q.to(tl.float32))
            # -- store dk ----
            dk_h = tl.load(dk_ptrs + offs_hd, mask=(offs_n)[:, None] < seqlen_k, other=0.0)
            tl.store(dk_ptrs + offs_hd, dk_h + k_grad, mask=(offs_n)[:, None] < seqlen_k)


def _flash_prob_forward(q, k):
    # shape constraints
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == [seqlen_k, nheads, d]
    # assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype, "All tensors must have the same type"
    # assert q.dtype in [paddle.float16, paddle.bfloat16], "Only support fp16 and bf16"

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = paddle.empty((seqlen_q_rounded,), dtype=paddle.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), 1)
    _prob_fwd_kernel[grid](
        q,
        k,
        lse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    lse = lse[:seqlen_q]

    return lse


def _flash_prob_backward(q, k, lse, dlse):
    # shape constraints
    seqlen_q, nheads, d = q.shape
    seqlen_k, _, _ = k.shape
    assert k.shape == [seqlen_k, nheads, d]
    # assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype, "All tensors must have the same type"
    # assert q.dtype in [paddle.float16, paddle.bfloat16], "Only support fp16 and bf16"

    dq = paddle.zeros_like(q, dtype=paddle.float32)
    dk = paddle.zeros_like(k, dtype=paddle.float32)

    q = q.contiguous()
    k = k.contiguous()
    dlse = dlse.contiguous()

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 8
    num_stages = 1
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), 1)
    _dq_prob_bwd_kernel[grid](
        q,
        k,
        dq,
        lse,
        dlse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    BLOCK_N = BLOCK_M
    BLOCK_M = BLOCK_N
    grid = lambda META: (triton.cdiv(seqlen_k, META["BLOCK_N"]), 1)
    _dk_prob_bwd_kernel[grid](
        q,
        k,
        dk,
        lse,
        dlse,
        nheads,
        seqlen_q,
        seqlen_k,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dq = dq[:seqlen_q]
    dk = dk[:seqlen_k]

    return dq, dk


class FlashProb(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q, k):
        lse = _flash_prob_forward(q, k)
        ctx.save_for_backward(q, k, lse)

        return lse

    @staticmethod
    def backward(ctx, dlse):
        q, k, lse = ctx.saved_tensor()
        dq, dk = _flash_prob_backward(q, k, lse, dlse)

        return dq, dk


def _cal_flash_loss(q, k, labels, head_dim=256):
    bq = q.shape[0]
    bk = k.shape[0]
    # NOTE: logits forward or backward should keep fp32 for better precision
    q = q.reshape([bq, -1, head_dim]).cast("float32")
    k = k.reshape([bk, -1, head_dim]).cast("float32")

    lse = FlashProb.apply(q, k)
    numerator = paddle.einsum("mhd,mhd->m", q, k[labels, ...])
    loss = -numerator + lse

    return loss


def cal_flash_loss(q, k, labels=None, scale=None, head_dim=256):
    if labels is None:
        labels = paddle.arange(q.shape[0])
    if scale is not None and scale != 1.0:
        q = q * scale
    return _cal_flash_loss(q, k, labels, head_dim)


if __name__ == "__main__":
    import time

    # Parameters
    num_heads = 3  # Number of attention heads
    seq_length_q = 32768  # Sequence length
    seq_length_k = 32768
    d_model = 256  # Dimension of each head (must be 16, 32, 64, or 128)

    # Randomly initialize inputs
    q = paddle.rand((seq_length_q, num_heads * d_model), dtype=paddle.float32)  # Query
    k = paddle.rand((seq_length_k, num_heads * d_model), dtype=paddle.float32)  # Key
    l = paddle.ones([]) * np.log(1 / 0.02)
    l.stop_gradient = False

    q = F.normalize(q, p=2, axis=-1)
    q.stop_gradient = False
    k = F.normalize(k, p=2, axis=-1)
    k.stop_gradient = False

    q1 = q.clone().detach()
    q1.stop_gradient = False
    k1 = k.clone().detach()
    k1.stop_gradient = False
    l1 = l.clone().detach()
    l1.stop_gradient = False

    labels = paddle.arange(seq_length_q)

    for i in range(1000):

        # A. paddle gradient
        start = time.time()
        qk = paddle.einsum("md,nd->mn", l.exp() * q, k)
        loss = F.cross_entropy(qk, labels, reduction="mean")
        loss.backward()
        end = time.time()

        # B. triton gradient
        start1 = time.time()
        loss1 = cal_flash_loss(q1, k1, labels, l1.exp())
        loss1 = loss1.mean()
        loss1.backward()
        end1 = time.time()

        print("========= Difference =========")
        print(end - start, end1 - start1, l.grad, l1.grad)
        print(paddle.max(paddle.abs(q.grad - q1.grad)), paddle.max(paddle.abs(k.grad - k1.grad)))

        set_to_zero = False
        q.clear_gradient(set_to_zero)
        k.clear_gradient(set_to_zero)
        l.clear_gradient(set_to_zero)
        q1.clear_gradient(set_to_zero)
        k1.clear_gradient(set_to_zero)
        l1.clear_gradient(set_to_zero)
