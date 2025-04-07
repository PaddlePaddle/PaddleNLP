# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import paddle

original_linear = paddle.nn.functional.linear

from typing import Literal, Optional

import paddle.nn.functional as F

# from ..linear_utils import RowParallelLinear as PD_RowParallelLinear
from ..linear_utils import ColumnParallelLinear as PD_ColumnParallelLinear
from ..linear_utils import (
    ColumnSequenceParallelLinear as PD_ColumnSequenceParallelLinear,
)
from ..linear_utils import Linear as PD_Linear
from ..linear_utils import RowParallelLinear as PD_RowParallelLinear
from ..linear_utils import RowSequenceParallelLinear as PD_RowSequenceParallelLinear
from .configuration import DeepseekV2Config

try:
    import kitchen

    from .kernel import act_quant, fp8_gemm, weight_dequant
except:
    pass

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    import deep_gemm
    import kitchen.quantization_subchannel_block_hybrid
    from kitchen.quantization import QParams, ScalingType
except:
    pass


__all__ = [
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowSequenceParallelLinear",
    "FP8Linear",
    "FP8DeepseekV2MLP",
]

gemm_impl: Literal["bf16", "fp8"] = "bf16"
block_size = 128


def fp8_linear(
    x: paddle.Tensor, weight: paddle.Tensor, bias: Optional[paddle.Tensor] = None, name=None
) -> paddle.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (paddle.Tensor): The input tensor.
        weight (paddle.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[paddle.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        paddle.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """

    if paddle.in_dynamic_mode():
        if weight.element_size() > 1:
            return original_linear(x, weight, bias)
        elif gemm_impl == "bf16":
            weight = weight_dequant(weight, weight._scale)
            return original_linear(x, weight, bias)
        else:
            x, scale = act_quant(x, block_size)
            y = fp8_gemm(x, scale, weight, weight._scale)
            if bias is not None:
                y += bias
            return y
    else:
        return original_linear(x, weight, bias)


paddle.nn.functional.linear = fp8_linear


def register_scale(self):
    if self.weight.element_size() == 1:
        in_features, out_features = self.weight.shape
        scale_out_features = (out_features + self.block_size - 1) // self.block_size
        scale_in_features = (in_features + self.block_size - 1) // self.block_size
        self.weight_scale_inv = self.create_parameter(
            shape=[scale_in_features, scale_out_features],
            attr=self._weight_attr,
            dtype="float32",
            is_bias=False,
        )
        self.weight._scale = self.weight_scale_inv


class Linear(PD_Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class ColumnParallelLinear(PD_ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class RowParallelLinear(PD_RowParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class ColumnSequenceParallelLinear(PD_ColumnSequenceParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class RowSequenceParallelLinear(PD_RowSequenceParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


def kitchen_quant(x, backend=None, is_1d_scaled=True, return_transpose=False):
    if backend is None:
        backend = kitchen.ops.Backend.CUBLAS
    quant_tile_shape = (1, 128) if is_1d_scaled else (128, 128)
    x_qparams = QParams(
        quant_dtype=paddle.float8_e4m3fn,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        eps=0,
        pow_2_scales=False,
        quant_tile_shape=quant_tile_shape,
    )
    quantize_op = kitchen.quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(backend)
    qresult_ref = quantize_op.quantize(x, x_qparams, return_transpose)
    if return_transpose:
        return (
            qresult_ref.data,
            qresult_ref.scale,
            qresult_ref.data_t,
            qresult_ref.scale_t,
        )
    else:
        return (qresult_ref.data, qresult_ref.scale)


def kitchen_fp8_gemm(x_fp8, x_scale, w_fp8, w_scale, is_a_1d_scaled, is_b_1d_scaled):
    if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
        y = kitchen.ops.fp8_gemm_blockwise(
            a=x_fp8,
            a_decode_scale=x_scale,
            b=w_fp8,
            b_decode_scale=w_scale,
            out_dtype=paddle.bfloat16,
            out=None,
            accumulate=False,
            use_split_accumulator=True,
            is_a_1d_scaled=is_a_1d_scaled,
            is_b_1d_scaled=is_b_1d_scaled,
        )
    else:
        y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], paddle.bfloat16)
    return y


class LinearFP8Func(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        x_orig_shape = x.shape
        # deep_gemm only support 2D
        x = x.reshape([-1, x_orig_shape[-1]])
        # quant
        if not x.is_contiguous():
            x = x.contiguous()
        x_quant, x_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        weight_t = weight.T.contiguous()
        w_quant, w_scale = kitchen_quant(
            weight_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )

        # compute out = mm(x, w_t)
        out = paddle.empty([x.shape[0], weight.shape[-1]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_quant, x_scale), (w_quant, w_scale), out)
        out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])

        # save for bwd
        x_t = x.T
        # padding
        x_t_shape = x_t.shape
        if x_t.shape[-1] % 128 != 0 or x_t.shape[-1] % 512 != 0:
            if (x_t.shape[-1] + 128 - (x_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            x_t = paddle.concat(
                [x_t, paddle.zeros([x_t.shape[0], padding_size - (x_t.shape[-1] % padding_size)], dtype=x_t.dtype)],
                axis=1,
            )
        x_t_quant, x_t_scale = kitchen_quant(
            x_t.contiguous(), backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        ctx.save_for_backward(
            x_t_quant, x_t_scale, weight, paddle.to_tensor(x_t_shape, dtype="int64", place=paddle.CPUPlace())
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        x_t_quant, x_t_scale, weight, x_t_shape = ctx.saved_tensor()
        x_t_shape = x_t_shape.numpy()
        # compute dx = mm(dout, w)
        dx = paddle.empty([x_t_shape[1], x_t_shape[0]], dout.dtype)
        dx_orig_shape = dout.shape[:-1]
        dx_orig_shape.append(x_t_shape[0])
        dout_quant, dout_scale = kitchen_quant(
            dout.reshape([-1, dout.shape[-1]]),
            backend=kitchen.ops.Backend.CUTLASS,
            is_1d_scaled=True,
            return_transpose=False,
        )
        w_quant, w_scale = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        deep_gemm.gemm_fp8_fp8_bf16_nt((dout_quant, dout_scale), (w_quant, w_scale), dx)
        dx = dx.reshape(dx_orig_shape)

        # compute dw = mm(x_t, dout_t)
        dout_t = dout.reshape([-1, dout.shape[-1]]).T.contiguous()
        # padding
        if dout_t.shape[-1] % 128 != 0 or dout_t.shape[-1] % 512 != 0:
            if (dout_t.shape[-1] + 128 - (dout_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            dout_t = paddle.concat(
                [
                    dout_t,
                    paddle.zeros(
                        [dout_t.shape[0], padding_size - (dout_t.shape[-1] % padding_size)], dtype=dout_t.dtype
                    ),
                ],
                axis=1,
            )

        dout_t_quant, dout_t_scale = kitchen_quant(
            dout_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        dweight = kitchen_fp8_gemm(x_t_quant, x_t_scale, dout_t_quant, dout_t_scale, True, True)
        return dx, dweight


class LinearFP8KeepXFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        x_orig_shape = x.shape
        # deep_gemm only support 2D
        x = x.reshape([-1, x_orig_shape[-1]])
        # quant
        x_quant, x_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        _, _, w_quant, w_scale = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )

        # compute out = mm(x, w_t)
        out = paddle.empty([x.shape[0], weight.shape[-1]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_quant, x_scale), (w_quant, w_scale), out)
        out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])

        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensor()
        dx_orig_shape = x.shape
        # padding
        x = x.reshape([-1, x.shape[-1]])
        if x.shape[0] % 128 != 0 or x.shape[0] % 512 != 0:
            if (x.shape[0] + 128 - (x.shape[0] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            x = paddle.concat(
                [x, paddle.zeros([padding_size - (x.shape[0] % padding_size), x.shape[-1]], dtype=x.dtype)],
                axis=0,
            )

        _, _, x_t_quant, x_t_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
        )

        # compute dx = mm(dout, w)
        dx = paddle.empty(x.shape, dout.dtype)

        dout_quant, dout_scale = kitchen_quant(
            dout.reshape([-1, dout.shape[-1]]),
            backend=kitchen.ops.Backend.CUTLASS,
            is_1d_scaled=True,
            return_transpose=False,
        )
        w_quant, w_scale = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        deep_gemm.gemm_fp8_fp8_bf16_nt((dout_quant, dout_scale), (w_quant, w_scale), dx)
        dx = dx.reshape(dx_orig_shape)

        # compute dw = mm(x_t, dout_t)
        dout_t = dout.reshape([-1, dout.shape[-1]])
        if dout_t.shape[0] % 128 != 0 or dout_t.shape[0] % 512 != 0:
            if (dout_t.shape[0] + 128 - (dout_t.shape[0] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            dout_t = paddle.concat(
                [
                    dout_t,
                    paddle.zeros(
                        [padding_size - (dout_t.shape[0] % padding_size), dout_t.shape[-1]], dtype=dout_t.dtype
                    ),
                ],
                axis=0,
            )

        _, _, dout_t_quant, dout_t_scale = kitchen_quant(
            dout_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
        )
        dweight = kitchen_fp8_gemm(x_t_quant, x_t_scale, dout_t_quant, dout_t_scale, True, True)
        return dx, dweight


class FP8Linear(paddle.nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias_attr: bool = False) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        return LinearFP8Func.apply(x, self.weight)


class FP8KeepXLinear(paddle.nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias_attr: bool = False) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        return LinearFP8KeepXFunc.apply(x, self.weight)


class Fuse_FFN_FP8_Func(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w1, w2):
        # deep_gemm only support 2D
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        # ===== o1 = deep_gemm(x_fp8, w1_t_fp8) =====
        x_fp8, x_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        w_t = w1.T.contiguous()

        w1_fp8, w1_sacle = kitchen_quant(
            w_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (w1_fp8, w1_sacle), o1)

        # ===== o2 = swiglu(o1) =====
        # TODO: [Fusion] swiglu + quant
        o2 = swiglu(o1)
        o2_fp8, o2_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
        _, _, w2_t_fp8, w2_t_scale = kitchen_quant(
            w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o2.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale), (w2_t_fp8, w2_t_scale), o3)
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        # ===== save for backward =====
        # TODO: [Fusion] transpose + padding + quant
        x_t = x.T.contiguous()
        if x_t.shape[-1] % 128 != 0 or x_t.shape[-1] % 512 != 0:
            if (x_t.shape[-1] + 128 - (x_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            x_t = paddle.concat(
                [x_t, paddle.zeros([x_t.shape[0], padding_size - (x_t.shape[-1] % padding_size)], dtype=x_t.dtype)],
                axis=1,
            )
        x_t_fp8, x_t_scale = kitchen_quant(
            x_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        ctx.save_for_backward(
            x_t_fp8,
            x_t_scale,
            w1,
            # w1_fp8,
            # w1_sacle,
            o1,
            w2,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        # deep_gemm only support 2D
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        x_t_fp8, x_t_scale, w1, o1, w2, x_orig_shape = ctx.saved_tensor()
        x_orig_shape = x_orig_shape.numpy()

        w2_fp8, w2_scale = kitchen_quant(
            w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )

        # ===== [recompute] o2 = swiglu(o1) =====
        # TODO: [Fusion] swiglu + transpose + quant
        o2 = swiglu(o1)
        o2_t = o2.T.contiguous()
        o2_t_fp8, o2_t_scale = kitchen_quant(
            o2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
        do3_fp8, do3_scale = kitchen_quant(
            do3, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale), (w2_fp8, w2_scale), do2)

        # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
        if o2_t.shape[-1] % 128 != 0 or o2_t.shape[-1] % 512 != 0:
            if (o2_t.shape[-1] + 128 - (o2_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            o2_t = paddle.concat(
                [
                    o2_t,
                    paddle.zeros([o2_t.shape[0], padding_size - (o2_t.shape[-1] % padding_size)], dtype=o2_t.dtype),
                ],
                axis=-1,
            )
            o2_t_fp8, o2_t_scale = kitchen_quant(
                o2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
        do3_t = do3.T.contiguous()
        if do3_t.shape[-1] % 128 != 0 or do3_t.shape[-1] % 512 != 0:
            if (do3_t.shape[-1] + 128 - (do3_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            do3_t = paddle.concat(
                [
                    do3_t,
                    paddle.zeros([do3_t.shape[0], padding_size - (do3_t.shape[-1] % padding_size)], dtype=do3_t.dtype),
                ],
                axis=-1,
            )

        do3_t_fp8, do3_t_scale = kitchen_quant(
            do3_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        dw2 = kitchen_fp8_gemm(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True)

        # ===== do1 = swiglu_grad(o1, None, do2) =====
        # TODO: [Fusion] swiglu_grad + quant
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        do1_fp8, do1_scale = kitchen_quant(
            do1, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        w1_fp8, w1_sacle = kitchen_quant(
            w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )

        # ===== dx = deep_gemm(do1_fp8, w1_fp8)
        dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale), (w1_fp8, w1_sacle), dx)
        if len(x_orig_shape) > 2:
            dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

        # ===== dw1 = deep_gemm(x_t_fp8, do1_t_fp8)
        # TODO: [Fusion] swiglu_grad + transpose + padding + quant
        do1_t = do1.T.contiguous()
        if do1_t.shape[-1] % 128 != 0 or do1_t.shape[-1] % 512 != 0:
            if (do1_t.shape[-1] + 128 - (do1_t.shape[-1] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128

            pad_size = padding_size - (do1_t.shape[1] % padding_size)
            do1_t = paddle.concat([do1_t, paddle.zeros([do1_t.shape[0], pad_size], dtype=do1_t.dtype)], axis=-1)
        do1_t_fp8, do1_t_scale = kitchen_quant(
            do1_t, is_1d_scaled=True, backend=kitchen.ops.Backend.CUBLAS, return_transpose=False
        )
        if numpy.prod(o2_t_fp8.shape) != 0 and numpy.prod(do3_t_fp8.shape) != 0:
            dw1 = kitchen_fp8_gemm(x_t_fp8, x_t_scale, do1_t_fp8, do1_t_scale, True, True)
        else:
            dw1 = paddle.zeros(w1_fp8.shape, do1.dtype)
        return dx, dw1, dw2


class FP8DeepseekV2MLP(paddle.nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.w1 = self.create_parameter(
            shape=[self.hidden_size, self.intermediate_size * 2],
            dtype="bfloat16",
            is_bias=False,
        )
        self.w2 = self.create_parameter(
            shape=[self.intermediate_size, self.hidden_size],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        return Fuse_FFN_FP8_Func.apply(x, self.w1, self.w2)


class FusedFP8DeepseekV2MLP(paddle.nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.w1 = self.create_parameter(
            shape=[4 * self.hidden_size, self.intermediate_size * 2],
            dtype="bfloat16",
            is_bias=False,
        )
        self.w2 = self.create_parameter(
            shape=[4 * self.intermediate_size, self.hidden_size],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        return Fuse_FFN_FP8_Func.apply(x, self.w1, self.w2)
