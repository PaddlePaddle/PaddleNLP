# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import enum
from enum import auto
from typing import Union

import paddle
import paddle.nn as nn

from .cce import cce_linear_cross_entropy
from .constants import IGNORE_INDEX
from .doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def linear_cross_entropy(
    e: paddle.Tensor,
    c: paddle.Tensor,
    targets: paddle.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: Union[float, None] = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: Union[float, str, None] = "auto",
    impl: Union[str, LinearCrossEntropyImpl] = LinearCrossEntropyImpl.CCE,
) -> paddle.Tensor:
    """
    :param filter_eps: The threshold value used to determine which locations can be safely ignored
        in gradient computation. The default value of "auto" will automatically choose a value
        based on the input dtype. Only valid for the CCE implementation.
    :param impl: The linear cross entropy implementation to use. Currently supports cce and torch_compile.
    """

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if impl == "cce":
        return cce_linear_cross_entropy(e, c, targets, ignore_index, softcap, reduction, shift, filter_eps)
    else:
        raise NotImplementedError(f"{impl} is not implemented.")


class LinearCrossEntropy(nn.Layer):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: Union[float, None] = None,
        reduction: str = "mean",
        filter_eps: Union[float, str, None] = "auto",
        shift: bool = False,
        impl: Union[str, LinearCrossEntropyImpl] = LinearCrossEntropyImpl.CCE,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift

        self.impl = impl

    def forward(self, e: paddle.Tensor, c: paddle.Tensor, targets: paddle.Tensor) -> paddle.Tensor:
        return linear_cross_entropy(
            e,
            c,
            targets,
            self.ignore_index,
            self.softcap,
            reduction=self.reduction,
            filter_eps=self.filter_eps,
            shift=self.shift,
            impl=self.impl,
        )
