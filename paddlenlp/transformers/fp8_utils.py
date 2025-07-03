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
import paddle.nn.functional as F

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


try:
    from paddle.incubate.fp8 import deep_gemm
except:
    pass


__all__ = [
    "kitchen_fp8_gemm",
    "dequantize_fp8_to_fp32",
    "FP8Linear",
    "FP8GroupGemmMlpFunctionNode",
]


def kitchen_fp8_gemm(
    x_fp8, x_scale, w_fp8, w_scale, is_a_1d_scaled, is_b_1d_scaled, out=None, rtn_dtype=paddle.bfloat16
):
    if out is not None:
        accumulate = True
        out_dtype = out.dtype
    else:
        accumulate = False
        out_dtype = rtn_dtype
    if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
        y = paddle.incubate.nn.functional.fp8_gemm_blockwise(
            a=x_fp8,
            a_decode_scale=x_scale,
            b=w_fp8,
            b_decode_scale=w_scale,
            out_dtype=out_dtype,
            out=out,
            accumulate=accumulate,
            use_split_accumulator=True,
            is_a_1d_scaled=is_a_1d_scaled,
            is_b_1d_scaled=is_b_1d_scaled,
        )
    else:
        y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], out_dtype)
        if out is not None:
            out = out + y
            return out

    return y


def dequantize_fp8_to_fp32(fp8_tensor, scale):
    # expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
    res = fp8_tensor.reshape([-1, 128]).astype("bfloat16") * (scale.reshape([-1, 1]))
    res = res.reshape(fp8_tensor.shape)

    return res


def padding(x, axis):
    if x.shape[axis] % 512 != 0:
        if (x.shape[axis] + 128 - (x.shape[axis] % 128)) % 512 != 0:
            padding_size = 512
        else:
            padding_size = 128
        pad_size = padding_size - (x.shape[axis] % padding_size)
        if axis == 0:
            x = paddle.concat([x, paddle.zeros([pad_size, x.shape[-1]], dtype=x.dtype)], axis=0)
        else:
            x = paddle.concat([x, paddle.zeros([x.shape[0], pad_size], dtype=x.dtype)], axis=-1)
    return x


class FP8LinearFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        x_orig_shape = x.shape
        x_t = x.T

        # deep_gemm only support 2D
        x = x.reshape([-1, x_orig_shape[-1]]).contiguous()

        # quant
        if x.shape[0] % 512 != 0:
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            x = padding(x, 0)
            _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        else:
            x_fp8, x_scale, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )

        _, _, w_fp8, w_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight, output_scale_transpose=False, quant_method="128x128", input_transpose=True
        )

        out = paddle.empty([x_fp8.shape[0], w_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w_fp8, w_sacle), out)
        out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])

        # save for bwd
        ctx.save_for_backward(x_t_fp8, x_t_scale, weight)
        ctx.x_t_shape = x_t.shape
        return out

    @staticmethod
    def backward(ctx, dout):
        x_t_fp8, x_t_scale, weight = ctx.saved_tensor()

        # ===== dx = deep_gemm(dout_fp8, w_fp8)
        dout_2d = dout.reshape([-1, dout.shape[-1]])
        if dout_2d.shape[0] % 512 != 0:
            dout_fp8, dout_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            dout_2d = padding(dout_2d, 0)
            _, _, dout_t_fp8, dout_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        else:
            dout_fp8, dout_scale, dout_t_fp8, dout_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        w_fp8, w_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight, output_scale_transpose=False, quant_method="128x128", input_transpose=False
        )
        dx = paddle.empty([ctx.x_t_shape[1], ctx.x_t_shape[0]], dout.dtype)
        dx_orig_shape = dout.shape[:-1]
        dx_orig_shape.append(ctx.x_t_shape[0])
        deep_gemm.gemm_fp8_fp8_bf16_nt((dout_fp8, dout_scale.T), (w_fp8, w_sacle), dx)
        dx = dx.reshape(dx_orig_shape)

        # ===== dw1 = deep_gemm(x_t_fp8, dout_t_fp8)
        dweight = kitchen_fp8_gemm(x_t_fp8, x_t_scale, dout_t_fp8, dout_t_scale, True, True, rtn_dtype=paddle.float32)

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
        return FP8LinearFunction.apply(x, self.weight)


class FP8LinearKeepXFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        x_orig_shape = x.shape

        # deep_gemm only support 2D
        x = x.reshape([-1, x_orig_shape[-1]]).contiguous()

        # quant
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )
        _, _, w_fp8, w_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight, output_scale_transpose=False, quant_method="128x128", input_transpose=True
        )

        # compute out = mm(x, w_t)
        out = paddle.empty([x_fp8.shape[0], w_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w_fp8, w_sacle), out)
        out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])

        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensor()
        dx_orig_shape = x.shape

        # padding
        x = padding(x, 0)
        _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )

        w_fp8, w_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight, output_scale_transpose=False, quant_method="128x128", input_transpose=False
        )

        dout_2d = dout.reshape([-1, dout.shape[-1]])
        if dout_2d.shape[0] % 512 != 0:
            dout_fp8, dout_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            dout_2d = padding(dout_2d, 0)
            _, _, dout_t_fp8, dout_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        else:
            dout_fp8, dout_scale, dout_t_fp8, dout_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                dout_2d, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )

        dx = paddle.empty([dout_fp8.shape[0], w_fp8.shape[0]], dout.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((dout_fp8, dout_scale.T), (w_fp8, w_sacle), dx)
        dx = dx.reshape(dx_orig_shape)

        # ===== dw1 = deep_gemm(x_t_fp8, dout_t_fp8)
        dweight = kitchen_fp8_gemm(x_t_fp8, x_t_scale, dout_t_fp8, dout_t_scale, True, True, rtn_dtype=paddle.float32)

        return dx, dweight


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
        return FP8LinearKeepXFunction.apply(x, self.weight)


def fp8_mlp_fwd(x, w1, w2):
    x_orig_shape = x.shape
    x = x.reshape([-1, x_orig_shape[-1]])

    # ===== o1 = deep_gemm(x_fp8, w1_t_fp8) =====
    x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x, output_scale_transpose=True, quant_method="1x128", input_transpose=False
    )

    _, _, w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
        w1, output_scale_transpose=False, quant_method="128x128", input_transpose=True
    )
    o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

    # ===== o2 = swiglu(o1) =====
    o2 = swiglu(o1)
    o2_fp8, o2_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        o2, output_scale_transpose=True, quant_method="1x128", input_transpose=False
    )

    # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
    _, _, w2_t_fp8, w2_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        w2, output_scale_transpose=False, quant_method="128x128", input_transpose=True
    )
    o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o1.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale.T), (w2_t_fp8, w2_t_scale), o3)
    if len(x_orig_shape) > 2:
        o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

    return x_fp8, x_scale, o3


def fp8_mlp_bwd(do3, x_fp8, x_scale, w1, w2):
    do3_orig_shape = do3.shape
    do3 = do3.reshape([-1, do3_orig_shape[-1]])

    x_orig_shape = x_fp8.shape

    _, _, w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
        w1, output_scale_transpose=False, quant_method="128x128", input_transpose=True
    )
    o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=do3.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

    x_dequant_fp16 = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale.T.contiguous())
    x_dequant_fp16 = padding(x_dequant_fp16, 0)

    _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x_dequant_fp16, output_scale_transpose=True, quant_method="1x128", input_transpose=True
    )

    # ===== [recompute] o2 = swiglu(o1) =====
    o2 = swiglu(o1)

    # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
    if do3.shape[0] % 512 != 0:
        do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do3, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )
        do3 = padding(do3, 0)
        _, _, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do3, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )
    else:
        do3_fp8, do3_scale, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do3, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )
    w2_fp8, w2_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        w2, output_scale_transpose=False, quant_method="128x128", input_transpose=False
    )
    do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale.T), (w2_fp8, w2_scale), do2)

    # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
    o2 = padding(o2, 0)
    _, _, o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        o2, output_scale_transpose=True, quant_method="1x128", input_transpose=True
    )

    if hasattr(w2, "main_grad"):
        if w2.main_grad is None:
            w2.main_grad = paddle.zeros(shape=w2.shape, dtype=paddle.float32)
        kitchen_fp8_gemm(
            o2_t_fp8,
            o2_t_scale,
            do3_t_fp8,
            do3_t_scale,
            True,
            True,
            w2.main_grad,
            paddle.float32,
        )
    else:
        if w2.grad is None:
            w2.grad = paddle.zeros(shape=w2.shape, dtype=paddle.float32)
        kitchen_fp8_gemm(
            o2_t_fp8,
            o2_t_scale,
            do3_t_fp8,
            do3_t_scale,
            True,
            True,
            w2.grad,
            paddle.float32,
        )
    if hasattr(w2, "_apply_backward_hook"):
        w2._apply_backward_hook()

    # ===== do1 = swiglu_grad(o1, None, do2) =====
    do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

    # ===== dx = deep_gemm(do1_fp8, w1_fp8)
    if do1.shape[0] % 512 != 0:
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )
        do1 = padding(do1, 0)
        _, _, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )
    else:
        do1_fp8, do1_scale, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )
    w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
        w1, output_scale_transpose=False, quant_method="128x128", input_transpose=False
    )
    dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
    deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale.T), (w1_fp8, w1_sacle), dx)
    if len(x_orig_shape) > 2:
        dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

    # ===== dw1 = deep_gemm(x_t_fp8, do1_t_fp8)
    if hasattr(w1, "main_grad"):
        if w1.main_grad is None:
            w1.main_grad = paddle.zeros(shape=w1.shape, dtype=paddle.float32)
        kitchen_fp8_gemm(
            x_t_fp8,
            x_t_scale,
            do1_t_fp8,
            do1_t_scale,
            True,
            True,
            w1.main_grad,
            paddle.float32,
        )
    else:
        if w1.grad is None:
            w1.grad = paddle.zeros(shape=w1.shape, dtype=paddle.float32)
        kitchen_fp8_gemm(
            x_t_fp8,
            x_t_scale,
            do1_t_fp8,
            do1_t_scale,
            True,
            True,
            w1.grad,
            paddle.float32,
        )
    if hasattr(w1, "_apply_backward_hook"):
        w1._apply_backward_hook()

    return dx


class FP8MlpFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w1, w2):
        # deep_gemm only support 2D
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        # ===== o1 = deep_gemm(x_fp8, w1_t_fp8) =====
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )

        _, _, w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w1, output_scale_transpose=False, quant_method="128x128", input_transpose=True
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

        # ===== o2 = swiglu(o1) =====
        o2 = swiglu(o1)
        o2_fp8, o2_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            o2, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )

        # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
        _, _, w2_t_fp8, w2_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w2, output_scale_transpose=False, quant_method="128x128", input_transpose=True
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale.T), (w2_t_fp8, w2_t_scale), o3)
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        # ===== save for backward =====

        ctx.save_for_backward(
            x_fp8,
            x_scale,
            w1,
            w2,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        # deep_gemm only support 2D
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        x_fp8, x_scale, w1, w2, x_orig_shape = ctx.saved_tensor()
        x_orig_shape = x_orig_shape.numpy()

        _, _, w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w1, output_scale_transpose=False, quant_method="128x128", input_transpose=True
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

        x_dequant_fp16 = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale.T.contiguous())
        x_dequant_fp16 = padding(x_dequant_fp16, 0)

        _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x_dequant_fp16, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )

        # ===== [recompute] o2 = swiglu(o1) =====
        o2 = swiglu(o1)

        # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
        if do3.shape[0] % 512 != 0:
            do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do3, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            do3 = padding(do3, 0)
            _, _, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do3, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        else:
            do3_fp8, do3_scale, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do3, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        w2_fp8, w2_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w2, output_scale_transpose=False, quant_method="128x128", input_transpose=False
        )
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale.T), (w2_fp8, w2_scale), do2)

        # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
        o2 = padding(o2, 0)
        _, _, o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            o2, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )

        dw2 = kitchen_fp8_gemm(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True, rtn_dtype=paddle.float32)

        # ===== do1 = swiglu_grad(o1, None, do2) =====
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        # ===== dx = deep_gemm(do1_fp8, w1_fp8)
        if do1.shape[0] % 512 != 0:
            do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do1, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            do1 = padding(do1, 0)
            _, _, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do1, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        else:
            do1_fp8, do1_scale, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                do1, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
        w1_fp8, w1_sacle = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w1, output_scale_transpose=False, quant_method="128x128", input_transpose=False
        )
        dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale.T), (w1_fp8, w1_sacle), dx)
        if len(x_orig_shape) > 2:
            dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

        # ===== dw1 = deep_gemm(x_t_fp8, do1_t_fp8)
        dw1 = kitchen_fp8_gemm(x_t_fp8, x_t_scale, do1_t_fp8, do1_t_scale, True, True, rtn_dtype=paddle.float32)
        return dx, dw1, dw2


class FP8Mlp(paddle.nn.Layer):
    def __init__(self, config, hidden_size=None, intermediate_size=None, is_moe=False):
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
        return FP8MlpFunction.apply(x, self.w1, self.w2)


def split_group_gemm(x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out):
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x_fp8[start_idx:end_idx], x_scale_tma_align), (w_fp8[i], w_scale[i]), gemm_out[start_idx:end_idx]
        )

        start_idx = end_idx

    return gemm_out


class FP8GroupGemmMlpFunctionNode:
    def __init__(
        self,
        custom_map,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        is_split_group_gemm=False,
        name="experts_group_gemm_contiguous_node",
    ):
        self.experts = custom_map.experts
        self.recompute_fwd_gate_up = recompute_fwd_gate_up
        self.dequant_input = dequant_input
        self.is_split_group_gemm = is_split_group_gemm
        self.tokens_per_expert = None
        self.m_indices = None
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def cached_tensors(self):
        return [
            self.tokens_per_expert,
            self.m_indices,
            self.unzipped_probs,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ]

    def set_cached_tensors(self, tensors):
        (
            self.tokens_per_expert,
            self.m_indices,
            self.unzipped_probs,
            self.input,
            self.input_fp8,
            self.input_scale,
            self.o1,
        ) = tensors

    def clear_cached_tensors(self):
        self.set_cached_tensors([None] * len(self.cached_tensors))

    def reset_statue(self):
        self.tokens_per_expert = None
        self.m_indices = None
        self.clear_activation_tensors()

    def clear_activation_tensors(self):
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def gen_m_indices(self, tokens_per_expert):
        tokens = []
        for i in range(len(tokens_per_expert)):
            tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
        out = paddle.concat(tokens, axis=0)
        return out

    def fwd_gate_up(self, x_bf16, expert_w1, num_expert, tokens_per_expert):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """
        self.tokens_per_expert = tokens_per_expert
        if not self.is_split_group_gemm:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        w1_t_quant, w1_t_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(expert_w1, transpose=True)
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if x_bf16 is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
        else:
            # quant x_bf16
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x_bf16, output_scale_transpose=False, quant_method="1x128", input_transpose=False
            )

        x_scale = paddle.transpose(paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0])

        # compute gemm
        o1 = paddle.empty([x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype)
        if numpy.prod(x_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(x_fp8, x_scale, w1_t_quant, w1_t_scale, tokens_per_expert, o1)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (x_fp8, x_scale), (w1_t_quant, w1_t_scale), o1, m_indices=self.m_indices
                )

        if self.dequant_input:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x_bf16
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o1, unzipped_probs, expert_w2, num_expert, o3=None, clear_o1=False):
        """
        o3 = o2 * w2
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # concat and transpose w2
        w2_quant, w2_sacle = paddle.incubate.nn.functional.fused_stack_transpose_quant(expert_w2, transpose=True)
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_sacle = w2_sacle.reshape([num_expert, -1, w2_sacle.shape[-1]])

        # quant o2
        with paddle.amp.auto_cast(False):
            o2_fp8, o2_scale = paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                o1, unzipped_probs, using_pow2_scaling=True
            )
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0])
        unzipped_probs = unzipped_probs.unsqueeze(-1)

        if clear_o1:
            o1._clear_to_zero_allocation()

        # compute gemm
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, "{} vs {}".format(o3.shape, o3_shape)
            o3.zero_()
        else:
            o3 = paddle.zeros(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(o2_fp8, o2_scale, w2_quant, w2_sacle, self.tokens_per_expert, o3)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (o2_fp8, o2_scale), (w2_quant, w2_sacle), o3, m_indices=self.m_indices
                )
        return o3, unzipped_probs

    def bwd_dowm_input(self, expert_w2, unzipped_grad, o1, inplace_swiglu_prob=False):
        """
        do2 = do3 * w2_t
        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        """
        # recompute concated_w2_2d
        bw_w2_quant, bw_w2_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w2, transpose=False
        )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        # compute gemm
        unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            unzipped_grad, output_scale_transpose=False, quant_method="1x128", input_transpose=False
        )
        do2_s = paddle.empty([unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype)
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    unzipped_grad_fp8, unzipped_grad_scale, bw_w2_quant, bw_w2_scale, self.tokens_per_expert, do2_s
                )
            else:
                unzipped_grad_scale = paddle.transpose(
                    paddle.transpose(unzipped_grad_scale, [1, 0]).contiguous(), [1, 0]
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=self.m_indices,
                )

        with paddle.amp.auto_cast(False):
            do1, probs_grad, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                o1, do2_s, self.unzipped_probs
            )

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input(self, do1, expert_w1, dx=None):
        """
        dx = do1 * w1_t
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # recompute concated_w1_t
        bw_w1_quant, bw_w1_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(
            expert_w1, transpose=False
        )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        # quant do1
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1, output_scale_transpose=False, quant_method="1x128", input_transpose=False
        )

        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
            dx.zero_()
        if numpy.prod(do1_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(do1_fp8, do1_scale, bw_w1_quant, bw_w1_scale, self.tokens_per_expert, dx)
            else:
                do1_scale = paddle.transpose(paddle.transpose(do1_scale, [1, 0]).contiguous(), [1, 0])
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (do1_fp8, do1_scale), (bw_w1_quant, bw_w1_scale), dx, m_indices=self.m_indices
                )

        return dx

    def fused_transpose_split_quant(self, x, tokens_per_expert, pow_2_scales):
        out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales)
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2):
        """
        dw2 = do2_t * do3
        [n, k] = [n, m_sum] * [m_sum, k] (m_sum = sum(tokens_per_expert))
        """
        o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(o2, self.tokens_per_expert, True)
        do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(do3, self.tokens_per_expert, True)

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w2[i], "_apply_backward_hook"):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1, clear_input=False):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """
        if input_x is None:
            if self.dequant_input:
                input_x = paddle.incubate.nn.functional.fused_act_dequant(self.input_fp8, self.input_scale)
            else:
                input_x = self.input
        if clear_input:
            self.input = None
            self.input_fp8 = None
            self.input_scale = None

        input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(input_x, self.tokens_per_expert, True)
        del input_x
        do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(do1, self.tokens_per_expert, True)

        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                kitchen_fp8_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w1[i], "_apply_backward_hook"):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert, origin_token_per_experts, output=None):
        self.origin_token_per_experts = origin_token_per_experts
        if hs_out is None:
            assert self.input_fp8 is not None
            assert self.input_scale is not None
            shape = self.input_fp8.shape
            dtype = paddle.bfloat16
        else:
            shape = hs_out.shape
            dtype = hs_out.dtype

        if shape[0] == 0:
            o3 = paddle.zeros(shape, dtype=dtype)
            self.unzipped_probs = unzipped_probs.unsqueeze(-1)
            return o3

        # get w1/w2
        expert_w1 = [x.w1 for x in self.experts if x is not None]
        expert_w2 = [x.w2 for x in self.experts if x is not None]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, num_expert, tokens_per_expert)
        if not self.recompute_fwd_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        o3, unzipped_probs = self.fwd_down(o1, unzipped_probs, expert_w2, num_expert, clear_o1=clear_o1)

        # save for bwd
        self.unzipped_probs = unzipped_probs
        return o3

    @paddle.no_grad()
    def backward(self, out_grad):
        # recompute expert_w2 and expert_w1
        expert_w1 = [x.w1 for x in self.experts if x is not None]
        expert_w2 = [x.w2 for x in self.experts if x is not None]

        if self.recompute_fwd_gate_up:
            o1 = self.fwd_gate_up(None, expert_w1, len(expert_w1), self.tokens_per_expert)
        else:
            o1 = self.o1

        # do2
        do1, o2_s, probs_grad = self.bwd_dowm_input(expert_w2, out_grad, o1, inplace_swiglu_prob=True)
        del o1
        self.o1 = None

        # dw1
        self.bwd_gate_up_weight(do1, None, expert_w1, clear_input=True)
        self.input_fp8 = None
        self.input_scale = None
        self.input = None

        # dw2
        self.bwd_down_weight(out_grad, o2_s, expert_w2)

        # dx
        dx = self.bwd_gate_up_input(do1, expert_w1, dx=out_grad)
        del do1

        self.reset_statue()
        return dx, probs_grad
