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

import paddle
import paddle.nn.functional as F
import pytest
from einops import rearrange
from paddlenlp_kernel.triton.mamba.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
    chunk_state,
    chunk_state_varlen,
)
from paddlenlp_kernel.triton.mamba.ssd_state_passing import _state_passing_fwd

#######################################################################################################################################
# patch paddle.allclose
old_allclose = paddle.allclose


def allclose(a, b, **kwargs):
    return old_allclose(a.cast("float32"), b.cast("float32"), **kwargs)


paddle.allclose = allclose

old_equal_all = paddle.equal_all


def equal_all(a, b):
    return old_equal_all(a.cast("float32"), b.cast("float32"))


paddle.equal_all = equal_all


def requires_grad_(self, value=True):
    self.stop_gradient = not value
    return self


paddle.Tensor.requires_grad_ = requires_grad_
#######################################################################################################################################


def detach_clone(*args):
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])


@pytest.mark.parametrize("dtype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('dtype', [paddle.bfloat16])
@pytest.mark.parametrize("ngroups", [1, 2, 8, "max"])
# @pytest.mark.parametrize('ngroups', [1])
@pytest.mark.parametrize("chunk_size", [64, 128])
# @pytest.mark.parametrize('chunk_size', [128])
def test_chunk_state_varlen(chunk_size, ngroups, dtype):
    rtol, atol = (1e-2, 3e-3)
    # set seed
    paddle.seed(chunk_size + (ngroups if ngroups != "max" else 64))
    batch = 300
    seqlens = paddle.randint(1, 200, (batch,))
    # batch = 3
    # seqlens = paddle.tensor([201, 56, 5])
    cu_seqlens = F.pad(seqlens.cumsum(0).unsqueeze([0, 1]), (1, 0), data_format="NCL").squeeze([0, 1])
    total_seqlen = seqlens.sum().item()
    seq_idx = paddle.concat(
        [paddle.full((s,), i, dtype=paddle.int32) for i, s in enumerate(seqlens)], axis=0
    ).unsqueeze(0)
    dim = 4096
    # dim = 64
    headdim = 64
    # dim = 32
    dstate = 32
    assert dim % headdim == 0
    nheads = dim // headdim
    if ngroups == "max":
        ngroups = nheads
    assert nheads % ngroups == 0
    B = paddle.randn([total_seqlen, ngroups, dstate], dtype=dtype) / 5
    x = paddle.randn([total_seqlen, nheads, headdim], dtype=dtype)
    A = -0.1 * (
        paddle.rand(
            [
                nheads,
            ]
        )
    )
    dt = F.softplus(paddle.randn([total_seqlen, nheads], dtype=paddle.float32) - 4)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum, seq_idx=seq_idx)
    chunk_states, _ = _state_passing_fwd(
        rearrange(chunk_states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1], seq_idx=seq_idx, chunk_size=chunk_size
    )
    chunk_states = rearrange(chunk_states, "... (p n) -> ... p n", n=dstate)
    chunk_states = chunk_states.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)
    dt_rounded = dt_rounded.squeeze(0)
    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_ref = []
    for b in range(batch):
        x_s = x[cu_seqlens[b] : cu_seqlens[b + 1]].unsqueeze(0)
        B_s = B[cu_seqlens[b] : cu_seqlens[b + 1]].unsqueeze(0)
        dt_s = dt[cu_seqlens[b] : cu_seqlens[b + 1]].unsqueeze(0)
        dA_cumsum_s, dt_rounded_s = _chunk_cumsum_fwd(dt_s, A, chunk_size)
        states = chunk_state(B_s, x_s, dt_rounded_s, dA_cumsum_s)
        _, final_states = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"), dA_cumsum_s[:, :, :, -1], chunk_size=chunk_size
        )
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
        out_ref.append(final_states)
    out_ref = paddle.concat(out_ref, axis=0)
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)
