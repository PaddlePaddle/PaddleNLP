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
    import deep_gemm
    import kitchen
    import kitchen.quantization_subchannel_block_hybrid
    from kitchen.quantization import QParams, ScalingType
except:
    pass


__all__ = [
    "kitchen_quant",
    "kitchen_fp8_gemm",
    "dequantize_fp8_to_fp32",
    "FP8Linear",
    "MoeMlpNode",
    "FP8GroupGemmMlpFunctionNode",
]


def kitchen_quant(x, backend=None, is_1d_scaled=True, return_transpose=False, pow_2_scales=True):
    if backend is None:
        backend = kitchen.ops.Backend.CUBLAS
    quant_tile_shape = (1, 128) if is_1d_scaled else (128, 128)
    x_qparams = QParams(
        quant_dtype=paddle.float8_e4m3fn,
        scaling_type=ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W,
        eps=0,
        pow_2_scales=pow_2_scales,
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
        y = kitchen.ops.fp8_gemm_blockwise(
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
            x_fp8, x_scale = kitchen_quant(
                x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            x = padding(x, 0)
            _, _, x_t_fp8, x_t_scale = kitchen_quant(
                x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        else:
            x_fp8, x_scale, x_t_fp8, x_t_scale = kitchen_quant(
                x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )

        _, _, w_fp8, w_sacle = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
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
            dout_fp8, dout_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            dout_2d = padding(dout_2d, 0)
            _, _, dout_t_fp8, dout_t_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        else:
            dout_fp8, dout_scale, dout_t_fp8, dout_t_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        w_fp8, w_sacle = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
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
        x_fp8, x_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        _, _, w_fp8, w_sacle = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
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
        _, _, x_t_fp8, x_t_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
        )

        w_fp8, w_sacle = kitchen_quant(
            weight, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )

        dout_2d = dout.reshape([-1, dout.shape[-1]])
        if dout_2d.shape[0] % 512 != 0:
            dout_fp8, dout_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            dout_2d = padding(dout_2d, 0)
            _, _, dout_t_fp8, dout_t_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        else:
            dout_fp8, dout_scale, dout_t_fp8, dout_t_scale = kitchen_quant(
                dout_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
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


class FP8MlpFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w1, w2):
        # deep_gemm only support 2D
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        # ===== o1 = deep_gemm(x_fp8, w1_t_fp8) =====
        x_fp8, x_scale = kitchen_quant(
            x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        _, _, w1_fp8, w1_sacle = kitchen_quant(
            w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

        # ===== o2 = swiglu(o1) =====
        o2 = swiglu(o1)
        o2_fp8, o2_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
        _, _, w2_t_fp8, w2_t_scale = kitchen_quant(
            w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
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

        _, _, w1_fp8, w1_sacle = kitchen_quant(
            w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_sacle), o1)

        x_dequant_fp16 = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale.T.contiguous())
        x_dequant_fp16 = padding(x_dequant_fp16, 0)

        _, _, x_t_fp8, x_t_scale = kitchen_quant(
            x_dequant_fp16, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
        )

        # ===== [recompute] o2 = swiglu(o1) =====
        o2 = swiglu(o1)

        # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
        if do3.shape[0] % 512 != 0:
            do3_fp8, do3_scale = kitchen_quant(
                do3, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            do3 = padding(do3, 0)
            _, _, do3_t_fp8, do3_t_scale = kitchen_quant(
                do3, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        else:
            do3_fp8, do3_scale, do3_t_fp8, do3_t_scale = kitchen_quant(
                do3, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        w2_fp8, w2_scale = kitchen_quant(
            w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale.T), (w2_fp8, w2_scale), do2)

        # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
        o2 = padding(o2, 0)
        _, _, o2_t_fp8, o2_t_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
        )

        dw2 = kitchen_fp8_gemm(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True, rtn_dtype=paddle.float32)

        # ===== do1 = swiglu_grad(o1, None, do2) =====
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        # ===== dx = deep_gemm(do1_fp8, w1_fp8)
        if do1.shape[0] % 512 != 0:
            do1_fp8, do1_scale = kitchen_quant(
                do1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            do1 = padding(do1, 0)
            _, _, do1_t_fp8, do1_t_scale = kitchen_quant(
                do1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        else:
            do1_fp8, do1_scale, do1_t_fp8, do1_t_scale = kitchen_quant(
                do1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=True
            )
        w1_fp8, w1_sacle = kitchen_quant(
            w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
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


def gen_m_indices(tokens_per_expert):
    tokens = []
    for i in range(len(tokens_per_expert)):
        tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
    out = paddle.concat(tokens, axis=0)
    return out


class FP8GroupGemmMlpFunctionNode:
    def __init__(self, custom_map, mem_efficient=False, name="experts_group_gemm_contiguous_node"):
        self.custom_map = custom_map
        self.mem_efficient = mem_efficient
        self.tokens_per_expert = None
        self.m_indices = None
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def reset_statue(self):
        self.tokens_per_expert = None
        self.m_indices = None
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def fwd_gate_up(self, x_bf16, expert_w1, num_expert, tokens_per_expert):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """
        self.tokens_per_expert = tokens_per_expert
        self.m_indices = gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        w1_t_quant, w1_t_scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(expert_w1, transpose=True)
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        # quant x_bf16
        x_fp8, x_scale = kitchen_quant(
            x_bf16, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        x_scale = paddle.transpose(paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0])

        # compute gemm
        o1 = paddle.zeros([x_bf16.shape[0], w1_t_quant.shape[1]], dtype=x_bf16.dtype)
        if numpy.prod(x_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (x_fp8, x_scale), (w1_t_quant, w1_t_scale), o1, m_indices=self.m_indices
            )

        if self.mem_efficient:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x_bf16

        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o1, unzipped_probs, expert_w2, num_expert):
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
        self.unzipped_probs = unzipped_probs.unsqueeze(-1)

        # compute gemm
        o3 = paddle.zeros([o2_fp8.shape[0], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (o2_fp8, o2_scale), (w2_quant, w2_sacle), o3, m_indices=self.m_indices
            )
        return o3

    def bwd_dowm_input(self, expert_w2, unzipped_grad, o1):
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
        unzipped_grad_fp8, unzipped_grad_scale = kitchen_quant(
            unzipped_grad, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        do2_s = paddle.zeros([unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype)
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (unzipped_grad_fp8, unzipped_grad_scale), (bw_w2_quant, bw_w2_scale), do2_s, m_indices=self.m_indices
            )

        with paddle.amp.auto_cast(False):
            do1, probs_grad, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                o1, do2_s, self.unzipped_probs
            )

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input(self, do1, expert_w1):
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
        do1_fp8, do1_scale = kitchen_quant(
            do1, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        # compute gemm
        dx = paddle.zeros(shape=[do1_fp8.shape[0], bw_w1_quant.shape[1]], dtype=paddle.bfloat16)
        if numpy.prod(do1_fp8.shape) != 0:
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

    def bwd_gate_up_weight(self, do1, input_x, expert_w1):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """
        input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(input_x, self.tokens_per_expert, True)
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

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert, origin_token_per_experts):
        self.origin_token_per_experts = origin_token_per_experts
        if hs_out.shape[0] == 0:
            o3 = paddle.zeros_like(hs_out)
            self.unzipped_probs = unzipped_probs.unsqueeze(-1)
            return o3
        # get w1/w2
        expert_w1 = [x.w1 for x in self.custom_map.experts if x is not None]
        expert_w2 = [x.w2 for x in self.custom_map.experts if x is not None]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, num_expert, tokens_per_expert)
        self.o1 = o1

        # o3
        o3 = self.fwd_down(o1, unzipped_probs, expert_w2, num_expert)

        return o3

    @paddle.no_grad()
    def backward(self, out_grad):
        # recompute expert_w2 and expert_w1
        expert_w1 = [x.w1 for x in self.custom_map.experts if x is not None]
        expert_w2 = [x.w2 for x in self.custom_map.experts if x is not None]

        if self.mem_efficient:
            input = paddle.incubate.nn.functional.fused_act_dequant(self.input_fp8, self.input_scale)
        else:
            input = self.input

        # do2
        do1, o2_s, probs_grad = self.bwd_dowm_input(expert_w2, out_grad, self.o1)

        # release o1 and reset o1
        del self.o1
        self.o1 = None

        # dx
        dx = self.bwd_gate_up_input(do1, expert_w1)

        # dw1
        self.bwd_gate_up_weight(do1, input, expert_w1)

        # release do1 and input
        del do1
        del input
        if self.mem_efficient:
            self.input_fp8 = None
            self.input_scale = None
        else:
            self.input = None

        # dw2
        self.bwd_down_weight(out_grad, o2_s, expert_w2)

        self.reset_statue()
        return dx, probs_grad


class MoeMlpNode:
    def __init__(self, experts, custom_map, name="moe_experts_node"):
        self.experts = experts
        self.x_t_fp8s = []
        self.x_t_scales = []
        self.o1s = []
        self.custom_map = custom_map

    def reset_statue(self):
        self.x_t_fp8s = []
        self.x_t_scales = []
        self.o1s = []
        self.tokens_per_expert = None

    @paddle.no_grad()
    def forward(self, hs_out, hs_scale_out, tokens_per_expert):
        self.tokens_per_expert = tokens_per_expert
        x_fp8_list = paddle.split(hs_out, num_or_sections=self.tokens_per_expert, axis=0)  # FP8 chunk
        x_scale_list = paddle.split(hs_scale_out, num_or_sections=self.tokens_per_expert, axis=0)  # FP8 chunk

        outputs = []
        for i, (chunk, chunk_scale) in enumerate(zip(x_fp8_list, x_scale_list)):
            expert = self.experts[i + self.custom_map.moe_rank * self.custom_map.moe_num_experts_per_device]
            x_fp8 = chunk.contiguous()
            o1 = self.fwd_gate_up(x_fp8, chunk_scale, expert.w1)
            o2 = self.fwd_swiglu(o1)
            o3 = self.fwd_down(o2, expert.w2)

            outputs.append(o3)

            # save for bwd
            x_t = dequantize_fp8_to_fp32(x_fp8, chunk_scale).T.contiguous()
            if x_t.shape[-1] % 128 != 0 or x_t.shape[-1] % 512 != 0:
                if (x_t.shape[-1] + 128 - (x_t.shape[-1] % 128)) % 512 != 0:
                    padding_size = 512
                else:
                    padding_size = 128
                x_t = paddle.concat(
                    [
                        x_t,
                        paddle.zeros([x_t.shape[0], padding_size - (x_t.shape[-1] % padding_size)], dtype=x_t.dtype),
                    ],
                    axis=1,
                )
            x_t_fp8, x_t_scale = kitchen_quant(
                x_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
            )
            self.x_t_fp8s.append(x_t_fp8)
            self.x_t_scales.append(x_t_scale)
            self.o1s.append(o1)
        expert_output = paddle.concat(outputs, axis=0)
        return expert_output

    @paddle.no_grad()
    def backward(self, out_grad, out_grad_scale):
        out_grad_list = paddle.split(out_grad, num_or_sections=self.tokens_per_expert, axis=0)

        out_grad_scale_list = paddle.split(out_grad_scale, num_or_sections=self.tokens_per_expert, axis=0)

        dxs = []
        do2_list = []
        for i, (do3, do3_scale, x_t_fp8, x_t_scale, o1) in enumerate(
            zip(
                out_grad_list,
                out_grad_scale_list,
                self.x_t_fp8s,
                self.x_t_scales,
                self.o1s,
            )
        ):
            expert = self.experts[i + self.custom_map.moe_rank * self.custom_map.moe_num_experts_per_device]
            w1_fp8, w1_scale = kitchen_quant(
                expert.w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
            w2_fp8, w2_scale = kitchen_quant(
                expert.w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
            )
            do2 = self.bwd_dowm_input(do3, do3_scale, w2_fp8, w2_scale)
            do2_list.append(do2)
            do1 = self.bwd_swiglu(o1, do2)
            dx = self.bwd_gate_up_input(do1, w1_fp8, w1_scale)

            if hasattr(expert.w2, "main_grad"):
                expert.w2.main_grad = self.bwd_down_weight(do3, do3_scale, o1, expert.w2.main_grad)
            else:
                expert.w2.grad = self.bwd_down_weight(do3, do3_scale, o1, expert.w2.grad)

            if hasattr(expert.w1, "main_grad"):
                expert.w1.main_grad = self.bwd_gate_up_weight(do1, x_t_fp8, x_t_scale, expert.w1.main_grad)
            else:
                expert.w1.grad = self.bwd_gate_up_weight(do1, x_t_fp8, x_t_scale, expert.w1.grad)

            dxs.append(dx)
        dx = paddle.concat(dxs, axis=0)
        self.reset_statue()
        return dx

    def fwd_gate_up(self, x_fp8, x_scale, w1):
        w1_t_fp8, w1_t_scale = kitchen_quant(
            w1.T.contiguous(), backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_t_fp8.shape[0]], dtype=paddle.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (w1_t_fp8, w1_t_scale), o1)

        return o1

    # ===== o2 = swiglu(o1) =====
    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
    def fwd_down(self, o2, w2):
        o2_fp8, o2_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        w2_t_fp8, w2_t_scale = kitchen_quant(
            w2.T.contiguous(), backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o2.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale), (w2_t_fp8, w2_t_scale), o3)
        return o3

    # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
    def bwd_dowm_input(self, do3_fp8, do3_scale, w2_fp8, w2_sacle):
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], paddle.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale), (w2_fp8, w2_sacle), do2)
        return do2

    # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
    def bwd_down_weight(self, do3_fp8, do3_scale, o1, dw2=None):
        # recompute o2
        o2 = swiglu(o1)
        o2_t = o2.T.contiguous()
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

        do3_t = dequantize_fp8_to_fp32(do3_fp8, do3_scale).T.contiguous()
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
        dw2 = kitchen_fp8_gemm(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True, dw2)
        return dw2

    # ===== do1 = swiglu_grad(o1, None, do2) =====
    def bwd_swiglu(self, o1, do2):
        # TODO: [Fusion] swiglu_grad + quant
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    # ===== dx = deep_gemm(do1_fp8, w1_fp8)
    def bwd_gate_up_input(self, do1, w1_fp8, w1_sacle):
        do1_fp8, do1_scale = kitchen_quant(
            do1, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale), (w1_fp8, w1_sacle), dx)
        return dx

    # ===== dw1 = deep_gemm(x_t_fp8, do1_t_fp8)
    def bwd_gate_up_weight(self, do1, x_t_fp8, x_t_scale, dw1=None):
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
        dw1 = kitchen_fp8_gemm(x_t_fp8, x_t_scale, do1_t_fp8, do1_t_scale, True, True, dw1)
        return dw1
