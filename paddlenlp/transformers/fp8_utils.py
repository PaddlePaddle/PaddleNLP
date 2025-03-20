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
    "ExpertsNode",
]


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


def kitchen_fp8_gemm(x_fp8, x_scale, w_fp8, w_scale, is_a_1d_scaled, is_b_1d_scaled, out=None):
    if out is not None:
        accumulate = True
        out_dtype = out.dtype
    else:
        accumulate = False
        out_dtype = paddle.bfloat16
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
    return y


def dequantize_fp8_to_fp32(fp8_tensor, scale):
    expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
    # 非规整情况，需要截断
    expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
    return fp8_tensor.astype("float32") * expanded_scale


class ExpertsNode:
    def __init__(self, experts, custom_map, name="moe_experts_node"):
        self.experts = experts
        self.outputs = []
        self.x_t_fp8s = []
        self.x_t_scales = []
        self.w1_fp8s = []
        self.w1_sacles = []
        self.w2_fp8s = []
        self.w2_sacles = []
        self.o1s = []
        self.dxs = []
        self.custom_map = custom_map

    def reset_statue(self):
        self.outputs = []
        self.x_t_fp8s = []
        self.x_t_scales = []
        self.w1_fp8s = []
        self.w1_sacles = []
        self.w2_fp8s = []
        self.w2_sacles = []
        self.o1s = []
        self.dxs = []

    def forward(self, hs_out, hs_scale_out, tokens_per_expert):
        self.tokens_per_expert = tokens_per_expert
        x_fp8_list = paddle.split(hs_out, num_or_sections=self.tokens_per_expert, axis=0)  # FP8 chunk
        x_scale_list = paddle.split(hs_scale_out, num_or_sections=self.tokens_per_expert, axis=0)  # FP8 chunk

        for i, (chunk, chunk_scale) in enumerate(zip(x_fp8_list, x_scale_list)):
            expert = self.experts[i + self.custom_map.moe_rank * self.custom_map.moe_num_experts_per_device]
            x_fp8 = chunk.contiguous()
            o1, w1_fp8, w1_sacle = self.fwd_gate_up(x_fp8, chunk_scale, expert.w1)
            o2 = self.fwd_swiglu(o1)
            o3, w2_fp8, w2_sacle = self.fwd_down(o2, expert.w2)

            self.outputs += [o3]

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
            self.x_t_fp8s += [x_t_fp8]
            self.x_t_scales += [x_t_scale]
            self.w1_fp8s += [w1_fp8]
            self.w1_sacles += [w1_sacle]
            self.o1s += [o1]
            self.w2_fp8s += [w2_fp8]
            self.w2_sacles += [w2_sacle]

        expert_output = paddle.concat(self.outputs, axis=0)
        return expert_output

    def backward(self, out_grad, out_grad_scale):
        out_grad_list = paddle.split(out_grad, num_or_sections=self.tokens_per_expert, axis=0)

        out_grad_scale_list = paddle.split(out_grad_scale, num_or_sections=self.tokens_per_expert, axis=0)

        for i, (do3, do3_scale, x_t_fp8, x_t_scale, w1_fp8, w1_sacle, o1, w2_fp8, w2_sacle) in enumerate(
            zip(
                out_grad_list,
                out_grad_scale_list,
                self.x_t_fp8s,
                self.x_t_scales,
                self.w1_fp8s,
                self.w1_sacles,
                self.o1s,
                self.w2_fp8s,
                self.w2_sacles,
            )
        ):
            expert = self.experts[i + self.custom_map.moe_rank * self.custom_map.moe_num_experts_per_device]
            do2 = self.bwd_dowm_input(do3, do3_scale, w2_fp8, w2_sacle)
            do1 = self.bwd_swiglu(o1, do2)
            dx = self.bwd_gate_up_input(do1, w1_fp8, w1_sacle)

            if hasattr(expert.w2, "main_grad"):
                expert.w2.main_grad = self.bwd_down_weight(do3, do3_scale, o1, expert.w2.main_grad)
            else:
                expert.w2.grad = self.bwd_down_weight(do3, do3_scale, o1, expert.w2.grad)

            if hasattr(expert.w1, "main_grad"):
                expert.w1.main_grad = self.bwd_gate_up_weight(do1, x_t_fp8, x_t_scale, expert.w1.main_grad)
            else:
                expert.w1.grad = self.bwd_gate_up_weight(do1, x_t_fp8, x_t_scale, expert.w1.grad)

            self.dxs += [dx]

        dx = paddle.concat(self.dxs, axis=0)

        self.reset_statue()
        return dx

    def fwd_gate_up(self, x_fp8, x_scale, w1):
        w1_fp8, w1_sacle, w1_t_fp8, w1_t_scale = kitchen_quant(
            w1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_t_fp8.shape[0]], dtype=paddle.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (w1_t_fp8, w1_t_scale), o1)
        return o1, w1_fp8, w1_sacle

    # ===== o2 = swiglu(o1) =====
    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    # ===== o3 = deep_gemm(o2_fp8, w2_t_fp8) =====
    def fwd_down(self, o2, w2):
        o2_fp8, o2_scale = kitchen_quant(
            o2, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        w2_fp8, w2_sacle, w2_t_fp8, w2_t_scale = kitchen_quant(
            w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=True
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o2.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale), (w2_t_fp8, w2_t_scale), o3)
        return o3, w2_fp8, w2_sacle

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
