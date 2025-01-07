# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import paddle
import pytest
from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy
from paddlenlp_kernel.triton.cut_cross_entropy.constants import IGNORE_INDEX
from paddlenlp_kernel.triton.cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not paddle.device.is_compiled_with_cuda(), reason="Test requires CUDA")


def _loss(
    e: paddle.Tensor,
    c: paddle.Tensor,
    targets: paddle.Tensor,
    softcap: Union[float, None],
    shift: bool,
) -> paddle.Tensor:
    N, T = targets.shape
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    logits = e @ c.T
    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = paddle.nn.functional.cross_entropy(
        logits.cast("float32"), targets, ignore_index=IGNORE_INDEX, reduction="none"
    )

    return loss.reshape([N, T])


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce"])
@pytest.mark.parametrize("dtype,error_tol", [(paddle.float32, 1e-5), (paddle.float16, 1e-3), (paddle.bfloat16, 1e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("shape", [(256, 512, 128), (252, 507, 128), (252, 507, 123)])
def test_loss_forward(
    impl: str,
    dtype: paddle.dtype,
    error_tol: float,
    softcap: Union[float, None],
    shift: bool,
    invalids: bool,
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

    e = e.reshape([4, -1, D])

    targets = paddle.randint(0, V, shape=(N,))

    if invalids:
        inds = paddle.randperm(len(targets))[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    targets = targets.reshape(e.shape[0:-1])

    gt = _loss(e.cast("float32"), c.cast("float32"), targets, softcap, shift)

    # paddle.set_float32_matmul_precision("highest" if dtype == paddle.float32 else "high")
    ref = _loss(e, c, targets, softcap, shift)

    cce_loss = linear_cross_entropy(e, c, targets, softcap=softcap, shift=shift, reduction="none", impl=impl)

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_loss).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{paddle.nn.functional.relu((cce_error - expected_error)).max()=}"
