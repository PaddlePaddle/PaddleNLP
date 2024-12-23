# Copyright (c) 2024, Tri Dao.
"""
this code is modified from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_varlen.py
"""
import paddle
import triton
import triton.language as tl
from paddle import Tensor


@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    STATES,
    state_len,
    dim,
    stride_x_seqlen,
    stride_x_dim,
    stride_states_batch,
    stride_states_seqlen,
    stride_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx), end_idx - state_len)
    rows = end_idx - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(
        X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
        mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
        other=0,
    )
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        STATES + rows_states[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim,
        x,
        mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim),
    )


def causal_conv1d_varlen_states(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.

    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = paddle.empty([batch, state_len, dim], dtype=x.dtype).transpose([0, 2, 1])
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    _causal_conv1d_varlen_states[grid](
        x,
        cu_seqlens,
        states,
        state_len,
        dim,
        x.strides[0],
        x.strides[1],
        states.strides[0],
        states.strides[2],
        states.strides[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return states


def causal_conv1d_varlen_states_ref(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.

    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = paddle.zeros([batch, state_len, dim], dtype=x.dtype).transpose([0, 2, 1])
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = paddle.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx) :] = x[start_idx:end_idx].T
    return states
