# Copyright (C) 2023, Tri Dao.


import paddle
import pytest
from einops import repeat
from paddlenlp_kernel.triton.mamba.selective_state_update import (
    selective_state_update,
    selective_state_update_ref,
)

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


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update(dim, dstate, has_z, itype):
    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (5e-3, 1e-2)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 5e-2
        # if torch.version.hip:
        #     atol *= 2
    # set seed
    paddle.seed(0)
    batch_size = 2
    state = paddle.randn([batch_size, dim, dstate], dtype=itype)
    x = paddle.randn([batch_size, dim], dtype=itype)
    dt = paddle.randn([batch_size, dim], dtype=itype)
    dt_bias = (
        paddle.rand(
            [
                dim,
            ]
        )
        - 4.0
    )
    A = -paddle.rand([dim, dstate]) - 1.0
    B = paddle.randn([batch_size, dstate])
    C = paddle.randn([batch_size, dstate])
    D = paddle.randn(
        [
            dim,
        ]
    )
    if has_z:
        z = paddle.randn(x.shape, dtype=x.dtype)
    else:
        z = None
    state_ref = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("tie_hdim", [False, True])
# @pytest.mark.parametrize('tie_hdim', [True])
@pytest.mark.parametrize("ngroups", [1, 2, 4])
# @pytest.mark.parametrize("ngroups", [2])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update_with_heads(dim, dstate, ngroups, has_z, tie_hdim, itype):
    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (5e-3, 3e-2)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 1e-1
    # set seed
    paddle.seed(0)
    batch_size = 2
    headdim = 64
    nheads = dim // headdim
    state = paddle.randn([batch_size, nheads, headdim, dstate], dtype=itype)
    x = paddle.randn([batch_size, nheads, headdim], dtype=itype)
    if not tie_hdim:
        dt = paddle.randn([batch_size, nheads, headdim], dtype=itype)
        dt_bias = paddle.rand([nheads, headdim]) - 4.0
        A = -paddle.rand([nheads, headdim, dstate]) - 1.0
        D = paddle.randn([nheads, headdim])
    else:
        dt = repeat(paddle.randn([batch_size, nheads], dtype=itype), "b h -> b h p", p=headdim)
        dt_bias = repeat(
            paddle.rand(
                [
                    nheads,
                ]
            )
            - 4.0,
            "h -> h p",
            p=headdim,
        )
        A = repeat(
            -paddle.rand(
                [
                    nheads,
                ]
            )
            - 1.0,
            "h -> h p n",
            p=headdim,
            n=dstate,
        )
        D = repeat(
            paddle.randn(
                [
                    nheads,
                ]
            ),
            "h -> h p",
            p=headdim,
        )
    B = paddle.randn([batch_size, ngroups, dstate])
    C = paddle.randn([batch_size, ngroups, dstate])
    if has_z:
        z = paddle.randn(x.shape, dtype=x.dtype)
    else:
        z = None
    state_ref = state.detach().clone()
    # state_og = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)
