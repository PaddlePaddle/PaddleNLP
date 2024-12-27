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
from paddlenlp_kernel.triton.mamba.layernorm_gated import layernorm_fn, rms_norm_ref

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


@pytest.mark.parametrize("norm_before_gate", [True, False])
# @pytest.mark.parametrize("norm_before_gate", [False])
@pytest.mark.parametrize("has_group", [False, True])
# @pytest.mark.parametrize("has_group", [False])
@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [True])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize("has_bias", [False])
# @pytest.mark.parametrize('dtype', [paddle.float32, paddle.float16, paddle.bfloat16])
@pytest.mark.parametrize("dtype", [paddle.float16])
# @pytest.mark.parametrize("wtype", [paddle.float32, paddle.float16, paddle.bfloat16])
@pytest.mark.parametrize("wtype", [paddle.float32])
@pytest.mark.parametrize("d", [2048, 4096])
# @pytest.mark.parametrize('d', [4096])
def test_layer_norm_gated(d, dtype, wtype, has_bias, has_z, is_rms_norm, has_group, norm_before_gate):
    if not has_z and not norm_before_gate:
        pytest.skip()
    if not norm_before_gate and not is_rms_norm:  # Reference LN isn't implemented for this case yet
        pytest.skip()

    rtol, atol = (1e-5, 1e-5) if dtype == paddle.float32 else (1e-2, 8e-3)
    group_size = None if not has_group else 64
    # set seed
    paddle.seed(0)
    batch = 16
    seqlen = 1024
    x = paddle.randn([batch, seqlen, d], dtype=dtype).requires_grad_()
    if has_z:
        z = paddle.randn([batch, seqlen, d], dtype=dtype).requires_grad_()
    else:
        z = None
    weight = paddle.randn(
        [
            d,
        ],
        dtype=wtype,
    ).requires_grad_()
    if has_bias:
        bias = paddle.randn(
            [
                d,
            ],
            dtype=wtype,
        ).requires_grad_()
    else:
        bias = None
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    z_pt = z.detach().clone().requires_grad_() if z is not None else None
    weight_ref = weight.detach().clone().requires_grad_()
    weight_pt = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    bias_pd = bias.detach().clone().requires_grad_() if bias is not None else None
    out = layernorm_fn(
        x,
        weight,
        bias,
        z=z,
        eps=1e-5,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    if not is_rms_norm:
        if not has_group:
            out_ref = F.layer_norm(
                x_ref.cast("float32"),
                (d,),
                weight=weight_ref.cast("float32"),
                bias=bias_ref.cast("float32") if bias_ref is not None else None,
                epsilon=1e-5,
            )
            out_pd = F.layer_norm(x_pt.cast(wtype), (d,), weight=weight_pt, bias=bias_pd, epsilon=1e-5)
        else:
            out_ref = rearrange(
                F.layer_norm(
                    rearrange(x_ref, "... (g d) -> ... g d", d=group_size).cast("float32"), (group_size,), epsilon=1e-5
                ),
                "... g d -> ... (g d)",
            ) * weight_ref.cast("float32")
            if has_bias:
                out_ref = out_ref + bias_ref.cast("float32")
            out_pd = (
                rearrange(
                    F.layer_norm(rearrange(x_pt, "... (g d) -> ... g d", d=group_size), (group_size,), epsilon=1e-5),
                    "... g d -> ... (g d)",
                )
                * weight_pt
            )
            if has_bias:
                out_pd = out_pd + bias_pd
        if has_z and norm_before_gate:
            out_ref = out_ref * F.silu(z_ref.cast("float32"))
            out_pd = out_pd * F.silu(z_pt)
    else:
        out_ref = rms_norm_ref(
            x_ref, weight_ref, bias_ref, z=z_ref, eps=1e-5, group_size=group_size, norm_before_gate=norm_before_gate
        )
        out_pd = rms_norm_ref(
            x_pt,
            weight_pt,
            bias_pd,
            z=z_pt,
            eps=1e-5,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            upcast=False,
        )
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    print(f"Max diff Paddle = {(out_pd - out_ref).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pd - out_ref).abs().max().item() + atol

    g = paddle.randn(out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g.cast(out_ref.dtype))
    out_pd.backward(g.cast(out_pd.dtype))
    print(f"Max dx diff = {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"Max dx diff Paddle = {(x_pt.grad - x_ref.grad).abs().max().item()}")
    if has_z:
        print(f"Max dz diff = {(z.grad - z_ref.grad).abs().max().item()}")
        print(f"Max dz diff Paddle = {(z_pt.grad - z_ref.grad).abs().max().item()}")
    print(f"Max dw diff = {(weight.grad - weight_ref.grad).abs().max().item()}")
    print(f"Max dw diff Paddle = {(weight_pt.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"Max db diff = {(bias.grad - bias_ref.grad).abs().max().item()}")
        print(f"Max db diff Paddle = {(bias_pd.grad - bias_ref.grad).abs().max().item()}")
    assert (x.grad - x_ref.grad).abs().max().item() <= 2 * (x_pt.grad - x_ref.grad).abs().max().item() + atol
    if has_z:
        assert (z.grad - z_ref.grad).abs().max().item() <= 2 * (z_pt.grad - z_ref.grad).abs().max().item() + atol
    assert (weight.grad - weight_ref.grad).abs().max().item() <= 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max().item() + atol
    if has_bias:
        assert (bias.grad - bias_ref.grad).abs().max().item() <= 2 * (
            bias_pd.grad - bias_ref.grad
        ).abs().max().item() + atol
