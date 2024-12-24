# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Union

import numpy as np
import paddle


def softcapping(logits: paddle.Tensor, softcap: float) -> paddle.Tensor:
    return paddle.tanh(logits / softcap) * softcap


def _handle_eps(filter_eps: Union[float, str, None], dtype: paddle.dtype) -> Union[float, None]:
    if filter_eps is None:
        return None
    elif isinstance(filter_eps, float):
        return filter_eps
    elif isinstance(filter_eps, str) and filter_eps == "auto":
        return paddle.finfo(dtype).eps / 32
    else:
        raise RuntimeError(f"Unknown eps {filter_eps=}")


def _build_flat_valids(
    targets: paddle.Tensor,
    ignore_index: int,
    shift: bool,
) -> Union[paddle.Tensor, None]:
    if shift:
        targets = targets[..., 1:]
    else:
        targets = targets.flatten()

    valids = (targets != ignore_index).nonzero().cast(paddle.int32)

    if not shift:
        assert valids.shape[1] == 1
        return valids.squeeze(1) if valids.numel() != targets.numel() else None

    for i in range(targets.ndim - 1):
        valids[:, i] *= targets.strides[i]

    assert targets.strides[-1] == 1

    return valids.sum(1)


def handle_reduction_none(
    batch_shape: list, valids: Union[paddle.Tensor, None], shift: bool, loss: paddle.Tensor
) -> paddle.Tensor:
    if valids is None:
        return loss.reshape(batch_shape)

    full_loss = paddle.zeros(np.prod(batch_shape), dtype=loss.dtype)
    full_loss[(valids + 1) if shift else valids] = loss

    return full_loss.reshape(batch_shape)
