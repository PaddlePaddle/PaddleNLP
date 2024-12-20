# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import Union, cast

import paddle

from .cce_backward import cce_backward_kernel
from .cce_lse_forward import cce_lse_forward_kernel
from .constants import IGNORE_INDEX
from .doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from .indexed_dot import indexed_neg_dot_forward_kernel
from .utils import _build_flat_valids, _handle_eps, handle_reduction_none


@dataclass
class CCEParams:
    targets: paddle.Tensor
    valids: Union[paddle.Tensor, None]
    softcap: Union[float, None]
    reduction: str
    filter_eps: Union[float, None]
    shift: bool
    batch_shape: list


def sort_logit_avg(logit_avg: paddle.Tensor) -> paddle.Tensor:
    return paddle.argsort(logit_avg)


class LinearCrossEntropyFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        e: paddle.Tensor,
        c: paddle.Tensor,
        params: CCEParams,
    ) -> paddle.Tensor:
        needs_grad = not e.stop_gradient or not c.stop_gradient
        return_logit_avg = needs_grad and params.filter_eps is not None

        ret = cce_lse_forward_kernel(
            e,
            c,
            params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
        )
        if return_logit_avg:
            assert isinstance(ret, tuple)
            lse, logit_avg = ret
        else:
            assert isinstance(ret, paddle.Tensor)
            lse = ret
            logit_avg = None

        neg_dot = indexed_neg_dot_forward_kernel(
            e, c, params.targets, params.shift, params.valids, params.softcap, lse.dtype
        )

        nll = neg_dot.add_(lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        ctx.save_for_backward(e, c, lse, params.targets, params.valids, logit_avg)
        ctx.params = params

        return loss

    @staticmethod
    def backward(ctx, grad_out: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, None]:
        h, w, lse, targets, valids, logit_avg = ctx.saved_tensor()

        if logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        params = cast(CCEParams, ctx.params)
        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / lse.numel().item()  # need cast paddle.Tensor to float
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.flatten()
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        de, dc = cce_backward_kernel(
            grad_out,
            h,
            w,
            lse,
            valids,
            params.softcap,
            params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            grad_scale=grad_scale,
        )

        return de, dc


def linear_cross_entropy_apply(
    e: paddle.Tensor,
    c: paddle.Tensor,
    params: CCEParams,
) -> paddle.Tensor:
    loss = LinearCrossEntropyFunction.apply(e, c, params)
    assert isinstance(loss, paddle.Tensor)

    if params.shift and params.reduction == "none":
        loss = loss[..., 1:]

    return loss


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def cce_linear_cross_entropy(
    e: paddle.Tensor,
    c: paddle.Tensor,
    targets: paddle.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: Union[float, None] = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: Union[float, str, None] = "auto",
) -> paddle.Tensor:
    """
    :param filter_eps: The threshold value used to determine which locations can be safely ignored
        in gradient computation. The default value of "auto" will automatically choose a value
        based on the input dtype.
    """
    assert e.shape[0:-1] == targets.shape
    assert e.shape[-1] == c.shape[1]
    # if not paddle.device.cuda.is_bf16_supported():
    #     raise RuntimeError(
    #         "Cut Cross Entropy requires an ampere GPU or newer. "
    #         "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
    #     )

    batch_shape = targets.shape

    e = e.contiguous()
    targets = targets.contiguous()

    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    return linear_cross_entropy_apply(
        e,
        c,
        CCEParams(
            targets,
            valids,
            softcap,
            reduction,
            _handle_eps(filter_eps, e.dtype),
            shift,
            batch_shape,
        ),
    )
