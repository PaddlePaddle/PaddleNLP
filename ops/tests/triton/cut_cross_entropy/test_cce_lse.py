# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import paddle
import paddle.nn.functional as F
import pytest
from paddlenlp_kernel.triton.cut_cross_entropy.cce_lse_forward import (
    cce_lse_forward_kernel,
)
from paddlenlp_kernel.triton.cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not paddle.device.is_compiled_with_cuda(), reason="Test requires CUDA")


def _lse(e: paddle.Tensor, c: paddle.Tensor, softcap: Union[float, None]) -> paddle.Tensor:
    logits = e @ c.T
    if softcap is not None:
        logits = softcapping(logits, softcap)
    return paddle.logsumexp(logits.cast("float32"), axis=-1)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [paddle.float32, paddle.float16, paddle.bfloat16])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shape", [(256, 512, 128), (255, 507, 128), (255, 507, 123)])
def test_lse(dtype: paddle.dtype, softcap: Union[float, None], shape: tuple[int, int, int]):
    # paddle.set_float32_matmul_precision("highest")
    paddle.seed(0)

    if dtype == paddle.bfloat16 and not paddle.device.is_compiled_with_cuda():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = paddle.randn((N, D), dtype=dtype) / (D**0.5)
    c = paddle.randn((V, D), dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    gt = _lse(e.cast("float32"), c.cast("float32"), softcap)

    # paddle.set_float32_matmul_precision("highest" if dtype == paddle.float32 else "high")
    ref = _lse(e, c, softcap)

    cce_lse = cce_lse_forward_kernel(e, c, softcap=softcap)

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_lse).abs()

    assert (cce_error <= (expected_error + 1e-5)).all(), f"{F.relu(cce_error - expected_error).max()=}"
