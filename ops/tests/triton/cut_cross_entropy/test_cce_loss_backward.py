# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import paddle
import paddle.nn.functional as F
import pytest
from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy
from paddlenlp_kernel.triton.cut_cross_entropy.constants import IGNORE_INDEX
from paddlenlp_kernel.triton.cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not paddle.device.is_compiled_with_cuda(), reason="Test requires CUDA")


def cross_entropy(
    input,
    label,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    soft_label=False,
    axis=-1,
    use_softmax=True,
    label_smoothing=0.0,
    name=None,
):
    """
    NOTE: torch cross_entropy is not the same as paddle cross_entropy.
    """
    if ignore_index < 0 and reduction == "mean":
        loss = F.cross_entropy(input, label, reduction="none")
        binary_sequence = paddle.where(loss > 0, paddle.ones_like(loss), paddle.zeros_like(loss))
        count = paddle.sum(binary_sequence)
        if count == 0:
            loss = paddle.sum(loss * binary_sequence)
        else:
            loss = paddle.sum(loss * binary_sequence) / count
        return loss
    return F.cross_entropy(
        input, label, weight, ignore_index, reduction, soft_label, axis, use_softmax, label_smoothing, name
    )


def _grads(
    e: paddle.Tensor,
    c: paddle.Tensor,
    targets: paddle.Tensor,
    softcap: Union[float, None],
    shift: bool,
    reduction: str,
    fp32: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    orig_e, orig_c = e, c
    set_to_zero = False
    e.clear_gradient(set_to_zero)
    c.clear_gradient(set_to_zero)

    N, T = targets.shape
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if fp32:
        e = e.cast("float32")
        c = c.cast("float32")

    logits = e @ c.T
    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = cross_entropy(logits.cast("float32"), targets, ignore_index=IGNORE_INDEX, reduction=reduction)

    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()

    loss.mean().backward()

    assert orig_e.grad is not None
    assert orig_c.grad is not None

    return orig_e.grad.detach().clone(), orig_c.grad.detach().clone()


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce"])
@pytest.mark.parametrize("dtype,error_tol", [(paddle.float16, 1e-3), (paddle.bfloat16, 1e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shape", [(256, 512, 128), (252, 507, 128), (252, 507, 123)])
def test_loss_backward(
    impl: str,
    dtype: paddle.dtype,
    error_tol: float,
    softcap: Union[float, None],
    shift: bool,
    invalids: bool,
    reduction: str,
    shape: tuple[int, int, int],
):
    # paddle.set_float32_matmul_precision("highest")
    # paddle._dynamo.config.cache_size_limit = 256
    paddle.seed(0)

    if dtype == paddle.bfloat16 and not paddle.device.is_compiled_with_cuda():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = paddle.randn((N, D), dtype=dtype) / (D**0.5)
    c = paddle.randn((V, D), dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    targets = paddle.randint(0, V, shape=(N,))

    if invalids:
        inds = paddle.randperm(len(targets))[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    e = e.reshape([4, -1, D])
    targets = targets.reshape(e.shape[0:-1])

    e.stop_gradient = False
    c.stop_gradient = False

    gt = _grads(e, c, targets, softcap, shift, reduction, fp32=True)

    ref = _grads(e, c, targets, softcap, shift, reduction)

    set_to_zero = False
    e.clear_gradient(set_to_zero)
    c.clear_gradient(set_to_zero)
    loss = linear_cross_entropy(e, c, targets, softcap=softcap, shift=shift, reduction=reduction, impl=impl)
    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()
    loss.mean().backward()
    assert e.grad is not None
    assert c.grad is not None

    expected_error = tuple((vgt - vref).abs() for vgt, vref in zip(gt, ref))
    cce_error = tuple((vgt - vcce).abs() for vgt, vcce in zip(gt, (e.grad, c.grad)))

    for i in range(len(expected_error)):
        assert (
            cce_error[i] <= (expected_error[i] + error_tol)
        ).all(), f"{(paddle.nn.functional.relu(cce_error[i] - expected_error[i])).max()=}"
