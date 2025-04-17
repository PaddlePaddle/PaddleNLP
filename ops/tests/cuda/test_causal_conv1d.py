# Copyright (C) 2024, Tri Dao.

import paddle
import paddle.nn.functional as F
import pytest
from einops import rearrange
from paddlenlp_kernel.cuda.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_ref,
    causal_conv1d_update,
    causal_conv1d_update_ref,
)
from paddlenlp_kernel.triton.causal_conv1d_varlen import (
    causal_conv1d_varlen_states,
    causal_conv1d_varlen_states_ref,
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


@pytest.mark.parametrize("return_final_states", [False, True])
# @pytest.mark.parametrize("return_final_states", [True])
@pytest.mark.parametrize("has_initial_states", [False, True])
# @pytest.mark.parametrize("has_initial_states", [False])
@pytest.mark.parametrize("channel_last", [False, True])
# @pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [3])
@pytest.mark.parametrize("seqlen", [2, 8, 16, 32, 64, 128, 129, 130, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize("dim", [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d(
    dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states
):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")

    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (3e-3, 5e-3)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    paddle.seed(0)
    batch = 2
    # batch = 1
    if not channel_last:
        x = paddle.randn([batch, 4096 + dim + 64, seqlen], dtype=itype)[:, 4096 : 4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            paddle.randn([batch, seqlen, 4096 + dim + 64], dtype=itype)[:, :, 4096 : 4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = paddle.randn([dim, width], dtype=paddle.float32).requires_grad_()
    if has_bias:
        bias = paddle.randn(
            [
                dim,
            ],
            dtype=paddle.float32,
        ).requires_grad_()
    else:
        bias = None
    if has_initial_states:
        initial_states = paddle.randn([batch, width - 1, dim], dtype=itype).transpose([0, 2, 1]).requires_grad_()
    else:
        initial_states = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_() if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(
        x, weight, bias, initial_states=initial_states, return_final_states=return_final_states, activation=activation
    )
    out_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        return_final_states=return_final_states,
        activation=activation,
    )
    if return_final_states:
        out, final_states = out
        out_ref, final_states_ref = out_ref
        print(f"Final states max diff: {(final_states - final_states_ref).abs().max().item()}")
        print(f"Final states mean diff: {(final_states - final_states_ref).abs().mean().item()}")
        assert paddle.allclose(final_states, final_states_ref, rtol=rtol, atol=atol)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(axis=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(axis=-1, keepdim=True)

    g = paddle.randn(out.shape, dtype=out.dtype)
    out.backward(g)
    out_ref.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")
    if has_initial_states:
        print(f"dinitial_states max diff: {(initial_states.grad - initial_states_ref.grad).abs().max().item()}")

    assert paddle.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert paddle.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert paddle.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_initial_states:
        assert paddle.allclose(initial_states.grad, initial_states_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (3e-3, 5e-3)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # rtolw, atolw = (1e-3, 1e-3)
    # set seed
    paddle.seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = paddle.randn([batch, seqlen, dim], dtype=itype).transpose([0, 2, 1])
    state_len = paddle.randint(width - 1, width + 10, (1,)).item()
    conv_state = paddle.randn([batch, state_len, dim], dtype=itype).transpose([0, 2, 1])
    weight = paddle.randn([dim, width], dtype=paddle.float32).requires_grad_()
    if has_bias:
        bias = paddle.randn(
            [
                dim,
            ],
            dtype=paddle.float32,
        ).requires_grad_()
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = paddle.randint(0, 1024, (batch,), dtype=paddle.int32) if has_cache_seqlens else None
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens)
    out_ref = causal_conv1d_update_ref(
        x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.equal_all(conv_state, conv_state_ref)
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_get_states(dim, itype):
    # set seed
    paddle.seed(0)
    seqlens = paddle.randint(1, 32, (100,))
    total_seqlen = seqlens.sum().item()
    x = paddle.randn([total_seqlen, dim], dtype=itype)
    cu_seqlens = F.pad(seqlens.cumsum(0).unsqueeze([0, 1]), (1, 0), data_format="NCL").squeeze([0, 1])
    state_len = 20
    out = causal_conv1d_varlen_states(x, cu_seqlens, state_len)
    out_ref = causal_conv1d_varlen_states_ref(x, cu_seqlens, state_len)
    assert paddle.equal_all(out, out_ref)


# @pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("channel_last", [True])
# @pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
@pytest.mark.parametrize("itype", [paddle.bfloat16])
# @pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("silu_activation", [True])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_bias", [True])
# @pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize(
    # "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
    "seqlen",
    [2048],
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
def test_causal_conv1d_race_condition(seqlen, width, has_bias, silu_activation, itype, channel_last):
    # set seed
    paddle.seed(0)
    batch = 2
    # batch = 1
    dim = 4096 + 32  # Try dim not divisible by 64
    # dim = 64
    if not channel_last:
        x = paddle.randn([batch, 4096 + dim + 64, seqlen], dtype=itype)[:, 4096 : 4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            paddle.randn([batch, seqlen, 4096 + dim + 64], dtype=itype)[:, :, 4096 : 4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = paddle.randn([dim, width], dtype=paddle.float32).requires_grad_()
    if has_bias:
        bias = paddle.randn(
            [
                dim,
            ],
            dtype=paddle.float32,
        ).requires_grad_()
    else:
        bias = None
    activation = None if not silu_activation else "silu"
    out0 = causal_conv1d_fn(x, weight, bias, activation=activation)
    g = paddle.randn(out0.shape, dtype=out0.dtype)
    dx0, dw0, db0 = paddle.autograd.grad(out0, (x, weight, bias), g)
    dw_atol = 1e-4
    # db_atol = 1e-4

    for i in range(10000):
        out = causal_conv1d_fn(x, weight, bias, activation=activation)
        dx, dw, db = paddle.autograd.grad(out, (x, weight, bias), g)
        dw_equal = paddle.allclose(dw, dw0, atol=dw_atol)
        # if not dw_equal:
        #     breakpoint()
        if has_bias:
            pass
            # db_equal = paddle.allclose(db, db0, atol=db_atol)
            # if not db_equal:
            #     breakpoint()
        assert paddle.equal_all(out, out0)
        assert paddle.equal_all(dx, dx0)
        assert dw_equal
        if has_bias:
            assert dw_equal


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [False])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [False])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [2])
@pytest.mark.parametrize("seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [2048])
@pytest.mark.parametrize("dim", [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d_varlen(dim, seqlen, width, has_bias, silu_activation, itype):
    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (3e-3, 5e-3)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    paddle.seed(seqlen + dim + width)
    batch = 3
    seqlens = []
    for b in range(batch):
        nsplits = paddle.randint(1, 5, (1,)).item()
        eos_pos = paddle.randperm(seqlen - 1)[:nsplits].sort()
        seqlens.append(
            paddle.diff(paddle.concat([paddle.to_tensor([-1]), eos_pos, paddle.to_tensor([seqlen - 1])])).tolist()
        )
        assert sum(seqlens[-1]) == seqlen
        assert all(s > 0 for s in seqlens[-1])
    # Only support channel_last
    x = rearrange(
        paddle.randn([batch, seqlen, 4096 + dim + 64], dtype=itype)[:, :, 4096 : 4096 + dim], "b s d -> b d s"
    ).requires_grad_()
    weight = paddle.randn([dim, width], dtype=paddle.float32).requires_grad_()
    if has_bias:
        bias = paddle.randn(
            [
                dim,
            ],
            dtype=paddle.float32,
        ).requires_grad_()
    else:
        bias = None
    seq_idx = paddle.stack(
        [
            paddle.concat([paddle.full((s,), i, dtype=paddle.int32) for i, s in enumerate(sl)], axis=0)
            for sl in seqlens
        ],
        axis=0,
    )
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, seq_idx=seq_idx, activation=activation)
    out_ref = []
    for b in range(batch):
        out_ref_b = []
        for x_s in paddle.split(x_ref[[b]], seqlens[b], axis=2):
            out_ref_b.append(causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation))
        out_ref.append(paddle.concat(out_ref_b, axis=2))
    out_ref = paddle.concat(out_ref, axis=0)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = paddle.randn(out.shape, dtype=out.dtype)
    out_ref.backward(g)
    out.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")

    assert paddle.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert paddle.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert paddle.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
