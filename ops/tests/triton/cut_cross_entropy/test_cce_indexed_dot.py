# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import paddle
import pytest
from paddlenlp_kernel.triton.cut_cross_entropy.indexed_dot import (
    indexed_neg_dot_forward_kernel,
)
from paddlenlp_kernel.triton.cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not paddle.device.is_compiled_with_cuda(), reason="Test requires CUDA")


@skip_no_cuda
@pytest.mark.parametrize("dtype,error_tol", [(paddle.float32, 5e-7), (paddle.float16, 1e-3), (paddle.bfloat16, 1e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shape", [(256, 512, 128), (255, 507, 128), (255, 507, 123)])
def test_indexed_dot(dtype: paddle.dtype, error_tol: float, softcap: float, shape: tuple[int, int, int]):
    paddle.seed(0)

    if dtype == paddle.bfloat16 and not paddle.device.is_compiled_with_cuda():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = paddle.randn((N, D), dtype=dtype) / (D**0.5)
    c = paddle.randn((V, D), dtype=dtype)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    inds = paddle.randint(0, V, shape=(N,))

    gt = -(e.cast("float32") * c[inds].cast("float32")).sum(-1)
    if softcap is not None:
        gt = softcapping(gt, softcap)

    ref = -(e * c[inds]).sum(-1).cast("float32")
    if softcap is not None:
        ref = softcapping(ref, softcap)

    cce_neg_dot = indexed_neg_dot_forward_kernel(e, c, inds, softcap=softcap)

    expected_error = (gt - ref).abs()
    cce_error = (gt - cce_neg_dot).abs()

    assert (
        cce_error <= (expected_error + error_tol)
    ).all(), f"{paddle.nn.functional.relu(cce_error - expected_error).max()=}"
