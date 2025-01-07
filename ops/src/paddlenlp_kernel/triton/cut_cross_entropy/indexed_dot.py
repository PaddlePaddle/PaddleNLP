# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import paddle
import triton
import triton.language as tl

from .tl_autotune import indexed_dot_autotune
from .tl_utils import b_bin_fn
from .utils import softcapping


@indexed_dot_autotune()
@triton.heuristics(
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "GROUP_B": lambda args: 8,
    }
)
@triton.jit
def _indexed_neg_dot_forward_kernel(
    E,
    C,
    Inds,
    Valids,
    Out,
    B,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_ib,
    stride_vb,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    EVEN_D: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_d_chunks = tl.cdiv(D, BLOCK_D)
    num_d_in_group = GROUP_B * num_d_chunks
    group_id = pid // num_d_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_d_in_group) % group_size_b)
    pid_d = (pid % num_d_in_group) // group_size_b

    offs_b = (tl.arange(0, BLOCK_B) + pid_b * BLOCK_B) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_d = tl.arange(0, BLOCK_D) + pid_d * BLOCK_D
    e_ptrs = E + (stride_eb * offs_b[:, None] + stride_ed * offs_d[None, :])
    if EVEN_D:
        e = tl.load(e_ptrs)
    else:
        e = tl.load(e_ptrs, mask=offs_d[None, :] < D, other=0.0)

    inds = tl.load(Inds + stride_ib * ((offs_b + 1) if SHIFT else offs_b))

    c_ptrs = C + (inds[:, None] * stride_cv + offs_d[None, :] * stride_cd)
    if EVEN_D:
        c = tl.load(c_ptrs)
    else:
        c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0)

    offs_b = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    out_ptrs = Out + offs_b
    dot = (e * c).to(tl.float32)
    neg_dot = -tl.sum(dot, 1).to(out_ptrs.dtype.element_ty)
    tl.atomic_add(out_ptrs, neg_dot, mask=offs_b < B)


def indexed_neg_dot_forward_kernel(
    e: paddle.Tensor,
    c: paddle.Tensor,
    inds: paddle.Tensor,
    shift: bool = False,
    valids: Union[paddle.Tensor, None] = None,
    softcap: Union[float, None] = None,
    out_dtype: Union[paddle.dtype, None] = None,
) -> paddle.Tensor:
    assert inds.ndim == 1
    assert e.ndim == 2
    assert c.ndim == 2
    assert inds.shape[0] == e.shape[0]
    assert c.shape[1] == e.shape[1]

    if valids is not None:
        assert valids.ndim == 1
        B = valids.shape[0]
    else:
        B = e.shape[0]

    out = paddle.zeros((B,), dtype=paddle.float32)

    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(e.shape[1], META["BLOCK_D"]),)

    _indexed_neg_dot_forward_kernel[grid](
        e,
        c,
        inds,
        valids,
        out,
        B,
        e.shape[1],
        e.strides[0],
        e.strides[1],
        c.strides[0],
        c.strides[1],
        inds.strides[0],
        1 if valids is None else valids.strides[0],
        B_BIN=b_bin_fn(B),
        SHIFT=shift,
    )

    if softcap is not None:
        out = softcapping(out, softcap)

    if out_dtype is None:
        out_dtype = e.dtype

    out = out.cast(out_dtype)

    return out
