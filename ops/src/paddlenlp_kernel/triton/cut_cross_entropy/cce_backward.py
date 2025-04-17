# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import paddle
import triton
import triton.language as tl

from .tl_autotune import cce_backward_autotune
from .tl_utils import (
    b_bin_fn,
    tl_and_reduce_fn,
    tl_lock_add,
    tl_softcapping,
    tl_softcapping_grad,
)


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :]

    da_ptrs = da_ptrs + d_inds * stride_ad
    b_ptrs = b_ptrs + d_inds * stride_bd

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)

        da_i = tl.dot(do, b).to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


@triton.jit
def _block_is_filtered(check_val: tl.tensor, filter_eps: tl.tensor) -> tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)


# @cce_backward_autotune()
# @triton.heuristics(
#     {
#         "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
#         "MM_BACK_BLOCK_D": lambda args: args["BLOCK_D"] * 2,
#         "MM_BACK_EVEN_D": lambda args: (args["D"] % (args["BLOCK_D"] * 2)) == 0,
#         "HAS_VALIDS": lambda args: args["Valids"] is not None,
#         "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
#         "FILTER_GRAD": lambda args: args["filter_eps"] is not None,
#         "HAS_TARGETS": lambda args: args["Targets"] is not None,
#         "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
#         "ITEM_DO": lambda args: (args["dOut"].numel() == 1).item(),
#         "GROUP_B": lambda args: 8,
#     }
# )
# @triton.jit
def _cce_backward_kernel(
    E,
    C,
    LSE,
    dOut,
    grad_scale,
    Valids,
    VocabOrdering,
    softcap,
    Targets,
    dE,
    dELocks,
    dC,
    dCLocks,
    B,
    D,
    V,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    filter_eps,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
    FILTER_GRAD: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_v_in_group) % group_size_b)
    pid_v = (pid % num_v_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v)

    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            e = tl.load(e_ptrs)
            c = tl.load(c_ptrs)
        else:
            e = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0)

        accum = tl.dot(e, c, accum)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    if HAS_VALIDS:
        lse = tl.load(LSE + (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B)
    else:
        lse = tl.load(LSE + offs_b)

    d_accum = tl.exp(accum - lse[:, None])

    if HAS_TARGETS:
        targets = tl.load(Targets + ((offs_b + 1) if SHIFT else offs_b))
        is_target = targets[:, None] == offs_v[None, :]
        d_accum += tl.where(is_target, -1.0, 0.0)
    else:
        is_target = None

    accum_valid_mask = ((pid_b * BLOCK_B + tl.arange(0, BLOCK_B))[:, None] < B) & (
        (pid_v * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] < V
    )
    d_accum = tl.where(accum_valid_mask, d_accum, 0.0)

    if FILTER_GRAD:
        if _block_is_filtered(tl.abs(d_accum), filter_eps):
            return

    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)

    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        d_out = tl.load(dOut + ((offs_b + 1) if SHIFT else offs_b))[:, None]

    d_out = grad_scale * d_out

    d_accum = (d_accum * d_out).to(e_ptrs.dtype.element_ty)

    b_mask = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)[:, None]) < B
    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)[:, None]) < V

    lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
    dELocks += lock_offset

    _mm_backward(
        d_accum,
        dE + (offs_b[:, None] * stride_eb),
        b_mask,
        dELocks,
        n_de_locks_1,
        C + offs_v[:, None] * stride_cv,
        v_mask,
        stride_ed,
        stride_cd,
        D,
        MM_BACK_BLOCK_D,
        MM_BACK_EVEN_D,
    )

    lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
    dCLocks += lock_offset

    _mm_backward(
        tl.trans(d_accum),
        dC + (offs_v[:, None] * stride_cv),
        v_mask,
        dCLocks,
        n_dc_locks_1,
        E + (offs_b[:, None] * stride_eb),
        b_mask,
        stride_cd,
        stride_ed,
        D,
        MM_BACK_BLOCK_D,
        MM_BACK_EVEN_D,
    )


# Fix issues https://github.com/apple/ml-cross-entropy/issues/6
_cce_backward_kernel = triton.jit(_cce_backward_kernel)
_cce_backward_kernel = triton.heuristics(
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: args["BLOCK_D"] * 2,
        "MM_BACK_EVEN_D": lambda args: (args["D"] % (args["BLOCK_D"] * 2)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
        "FILTER_GRAD": lambda args: args["filter_eps"] is not None,
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "ITEM_DO": lambda args: (args["dOut"].numel() == 1).item(),
        "GROUP_B": lambda args: 8,
    }
)(_cce_backward_kernel)
_cce_backward_kernel = cce_backward_autotune()(_cce_backward_kernel)


def cce_backward_kernel(
    do: paddle.Tensor,
    e: paddle.Tensor,
    c: paddle.Tensor,
    lse: paddle.Tensor,
    valids: Union[paddle.Tensor, None],
    softcap: Union[float, None],
    filter_eps: Union[float, None],
    targets: Union[paddle.Tensor, None] = None,
    shift: bool = False,
    vocab_ordering: Union[paddle.Tensor, None] = None,
    grad_scale: float = 1.0,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    assert do.numel().item() in (e.shape[0], 1)
    assert c.shape[1] == e.shape[1]
    assert lse.shape[0] == e.shape[0] or (valids is not None and lse.shape[0] == valids.shape[0])
    assert e.dtype in (
        paddle.float16,
        paddle.bfloat16,
    ), "Backwards requires embeddings to be bf16 or fp16"
    assert c.dtype in (
        paddle.float16,
        paddle.bfloat16,
    ), "Backwards requires classifier to be bf16 or fp16"

    do = do.contiguous()
    lse = lse.contiguous()

    de = paddle.zeros_like(e)
    dc = paddle.zeros_like(c)

    assert de.strides == e.strides
    assert dc.strides == c.strides

    if valids is not None:
        assert valids.ndim == 1
        B = valids.shape[0]
    else:
        B = e.shape[0]

    if do.numel().item() > 1:
        do = do.contiguous()
        lse = lse.contiguous()
        assert do.strides[0] == lse.strides[0], f"{do.strides=}, {lse.strides=}"

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(c.shape[0], META["BLOCK_V"]),)

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel().item() == dc.shape[0]
        assert vocab_ordering.strides[0] == 1

    nd_locks = triton.cdiv(c.shape[1], 64)
    de_locks = paddle.zeros((triton.cdiv(B, nd_locks), nd_locks), dtype=paddle.int32)
    dc_locks = paddle.zeros((triton.cdiv(c.shape[0], nd_locks), nd_locks), dtype=paddle.int32)

    _cce_backward_kernel[grid](
        e,
        c,
        lse,
        do,
        grad_scale,
        valids,
        vocab_ordering,
        softcap,
        targets,
        de,
        de_locks,
        dc,
        dc_locks,
        B,
        e.shape[1],
        c.shape[0],
        de_locks.shape[0],
        de_locks.shape[1],
        dc_locks.shape[0],
        dc_locks.shape[1],
        e.strides[0],
        e.strides[1],
        c.strides[0],
        c.strides[1],
        1 if valids is None else valids.strides[0],
        filter_eps,
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
    )

    return de, dc
