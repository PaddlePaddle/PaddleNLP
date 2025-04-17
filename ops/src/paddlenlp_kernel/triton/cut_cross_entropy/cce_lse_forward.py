# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Literal, Union, overload

import paddle
import triton
import triton.language as tl

from .tl_autotune import cce_forward_autotune
from .tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping


def get_float32_matmul_precision():
    return "high"


@cce_forward_autotune()
@triton.heuristics(
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        "DOT_PRECISION": lambda args: "tf32" if get_float32_matmul_precision() == "high" else "ieee",
    }
)
@triton.jit
def _cce_lse_forward_kernel(
    E,
    C,
    LSE,
    LA,
    Locks,
    Valids,
    softcap,
    B,
    V,
    D,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_lse_b,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)) % B
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) % V
    offs_d = tl.arange(0, BLOCK_D)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_D:
            e = tl.load(e_ptrs)
            c = tl.load(c_ptrs)
        else:
            e = tl.load(e_ptrs, mask=offs_d[None, :] < D - d * BLOCK_D, other=0.0)
            c = tl.load(c_ptrs, mask=offs_d[:, None] < D - d * BLOCK_D, other=0.0)
        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    v_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    logits = tl.where(v_mask[None, :], accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    o_mask = off_b < B
    if HAS_LA:
        logits = tl.where(o_mask[:, None], logits, 0.0)
        this_avg_logit = tl.sum(logits, 0) / B
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=v_mask)

    this_mx = tl.max(logits, axis=1)
    e = tl.exp(logits - this_mx[:, None])
    this_lse = this_mx + tl.log(tl.sum(e, axis=1))

    lse_ptrs = LSE + (stride_lse_b * off_b)

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.atomic_xchg(this_locks, 0)


@overload
def cce_lse_forward_kernel(
    e,
    c,
    valids: Union[paddle.Tensor, None] = None,
    softcap: Union[float, None] = None,
    return_logit_avg: Literal[False] = False,
) -> paddle.Tensor:
    ...


@overload
def cce_lse_forward_kernel(
    e,
    c,
    valids: Union[paddle.Tensor, None] = None,
    softcap: Union[float, None] = None,
    return_logit_avg: Literal[True] = True,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    ...


@overload
def cce_lse_forward_kernel(
    e,
    c,
    valids: Union[paddle.Tensor, None] = None,
    softcap: Union[float, None] = None,
    return_logit_avg: bool = False,
) -> Union[tuple[paddle.Tensor, paddle.Tensor], paddle.Tensor]:
    ...


def cce_lse_forward_kernel(
    e: paddle.Tensor,
    c: paddle.Tensor,
    valids: Union[paddle.Tensor, None] = None,
    softcap: Union[float, None] = None,
    return_logit_avg: bool = False,
) -> Union[tuple[paddle.Tensor, paddle.Tensor], paddle.Tensor]:
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel().item()
    else:
        B, _ = e.shape

    V, D = c.shape
    # Allocates output.
    lse = paddle.full((B,), -float("inf"), dtype=paddle.float32)

    locks = paddle.full(
        (triton.cdiv(B, 128),),
        0,
        dtype="int32",  # paddle donot support uint32, so we use int32
    )
    if return_logit_avg:
        logit_avg = paddle.full((V,), 0.0, dtype=paddle.float32)
    else:
        logit_avg = None

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        lse,  #
        logit_avg,
        locks,
        valids,
        softcap,
        B,
        V,
        D,  #
        e.strides[0],
        e.strides[1],  #
        c.strides[0],
        c.strides[1],  #
        lse.strides[0],
        1 if valids is None else valids.strides[0],
        num_locks=locks.shape[0],
        B_BIN=b_bin_fn(B),
    )

    if return_logit_avg:
        assert logit_avg is not None
        return lse, logit_avg
    else:
        return lse
