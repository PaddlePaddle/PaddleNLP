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
    "ExpertsNode",
    "ExpertsGroupGemmNode",
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
        out_dtype = paddle.float32
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


class ExpertsGroupGemmNode:
    def __init__(self, experts, custom_map, name="moe_experts_node"):
        self.o1 = None
        self.unzipped_scale = None
        self.unzipped_tokens = None
        self.custom_map = custom_map
        self.unzipped_probs = None
        self.tokens_per_expert = None

    def reset_statue(self):
        self.o1 = None
        self.unzipped_scale = None
        self.unzipped_tokens = None
        self.unzipped_probs = None
        self.tokens_per_expert = None

    def fwd_gate_up(self, x_bf16, expert_w1, expert_w_count, tokens_per_expert):
        # concat w1
        stacked_w1 = paddle.stack(expert_w1, axis=0)
        stacked_w1_t = paddle.transpose(stacked_w1, [0, 2, 1]).contiguous()
        concated_w1_t = stacked_w1_t.reshape([-1, stacked_w1_t.shape[-1]])

        # quant w1
        w1_t_quant, w1_t_scale = kitchen_quant(
            concated_w1_t,
            backend=kitchen.ops.Backend.CUBLAS,
            is_1d_scaled=False,
            return_transpose=False,
        )

        w1_t_quant = w1_t_quant.reshape([expert_w_count, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([expert_w_count, -1, w1_t_scale.shape[-1]])

        # quant x_bf16
        x_fp8, x_scale = kitchen_quant(
            x_bf16, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        # mask group gemm需要输入x是[group,m,n]
        x_fp8 = x_fp8.reshape([expert_w_count, -1, x_fp8.shape[-1]])
        x_scale = x_scale.reshape([expert_w_count, -1, x_scale.shape[-1]])

        o1 = paddle.zeros([expert_w_count, x_fp8.shape[1], w1_t_quant.shape[1]], dtype=x_bf16.dtype)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (x_fp8, x_scale), (w1_t_quant, w1_t_scale), o1, tokens_per_expert, x_fp8.shape[1]
        )
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o2, expert_w2, expert_w_count, tokens_per_expert):
        # concat and transpose w2
        expert_w2 = [x.w2 for x in self.custom_map.experts if x is not None]

        stacked_w2 = paddle.stack(expert_w2, axis=0)
        stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
        concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])

        # quant w2
        w2_quant, w2_sacle = kitchen_quant(
            concated_w2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        w2_quant = w2_quant.reshape([expert_w_count, -1, w2_quant.shape[-1]])
        w2_sacle = w2_sacle.reshape([expert_w_count, -1, w2_sacle.shape[-1]])

        # quant o2
        o2_reshape = o2.reshape([-1, o2.shape[-1]])
        o2_quant, o2_scale = kitchen_quant(
            o2_reshape, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        o2_quant = o2_quant.reshape([expert_w_count, -1, o2_quant.shape[-1]])
        o2_scale = o2_scale.reshape([expert_w_count, -1, o2_scale.shape[-1]])

        # group gemm masked
        o3 = paddle.zeros([expert_w_count, o2_quant.shape[1], w2_quant.shape[1]], dtype=o2.dtype)

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (o2_quant, o2_scale), (w2_quant, w2_sacle), o3, tokens_per_expert, o2_quant.shape[1]
        )
        return o3

    # ===== do2 = deep_gemm(do3_fp8, w2_fp8)
    def bwd_dowm_input(self, expert_w2, unzipped_grad, tokens_per_expert, expected_m):
        # recomput o2
        o2 = self.fwd_swiglu(self.o1)
        o2_s = (o2 * self.unzipped_probs).cast(paddle.bfloat16)

        # recompute concated_w2_2d
        stacked_w2 = paddle.stack(expert_w2, axis=0)
        concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])

        # quant w2
        bw_w2_quant, bw_w2_scale = kitchen_quant(
            concated_w2, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = kitchen_quant(
            unzipped_grad, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        unzipped_grad_fp8 = unzipped_grad_fp8.reshape([len(expert_w2), -1, unzipped_grad_fp8.shape[-1]])
        unzipped_grad_scale = unzipped_grad_scale.reshape([len(expert_w2), -1, unzipped_grad_scale.shape[-1]])
        do2_s = paddle.zeros(
            [len(expert_w2), unzipped_grad_fp8.shape[1], bw_w2_quant.shape[1]], dtype=unzipped_grad.dtype
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (unzipped_grad_fp8, unzipped_grad_scale),
            (bw_w2_quant, bw_w2_scale),
            do2_s,
            tokens_per_expert,
            expected_m,
        )
        # do2: 前向从bfloat16-->float32，反向从float32-->bfloat16,do2 需要保持 bfloat16（因为 o2 是 bfloat16)
        do2 = (do2_s.cast(paddle.float32) * self.unzipped_probs).cast(paddle.bfloat16)

        # probs_grad: probs_grad 需要保持 float32（因为 unzipped_probs 是 float32）
        probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)

        return do2, o2_s, probs_grad

    # ===== do1 = swiglu_grad(o1, None, do2) =====
    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    # ===== dx = deep_gemm(do1_fp8, w1_fp8)

    def bwd_gate_up_input(self, do1, expert_w1, tokens_per_expert, expected_m):
        # recompute concated_w1_t
        stacked_w1 = paddle.stack(expert_w1, axis=0)
        concated_w1_t_2d = stacked_w1.reshape([-1, stacked_w1.shape[-1]])

        # quant w1
        bw_w1_quant, bw_w1_scale = kitchen_quant(
            concated_w1_t_2d, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=False, return_transpose=False
        )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        # quant do1
        do1_fp8_reshape = do1.reshape([-1, do1.shape[-1]])
        do1_fp8, do1_scale = kitchen_quant(
            do1_fp8_reshape, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        do1_fp8 = (do1_fp8.reshape([len(expert_w1), -1, do1_fp8.shape[-1]])).contiguous()
        do1_scale = do1_scale.reshape([len(expert_w1), -1, do1_scale.shape[-1]]).contiguous()

        # group gemm
        dx = paddle.zeros(shape=[len(expert_w1), do1_fp8.shape[1], bw_w1_quant.shape[1]], dtype=paddle.bfloat16)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (do1_fp8, do1_scale), (bw_w1_quant, bw_w1_scale), dx, tokens_per_expert, expected_m
        )
        return dx

    # ===== dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
    def bwd_down_weight(self, out_grad, o2, expert_w2):
        # transpose o2
        group_num = len(expert_w2)
        H2 = o2.shape[-1]

        o2_t = o2.reshape([group_num, -1, H2]).transpose([0, 2, 1]).contiguous().reshape([group_num * H2, -1])

        o2_t_fp8, o2_t_scale = kitchen_quant(
            o2_t, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        o2_t_fp8 = o2_t_fp8.reshape([group_num, -1, o2_t_fp8.shape[-1]])
        o2_t_scale = paddle.split(o2_t_scale, num_or_sections=group_num, axis=-1)

        # quant out_grad
        H1 = out_grad.shape[-1]
        out_grad = (
            out_grad.reshape([group_num, -1, H1]).transpose([0, 2, 1]).contiguous().reshape([group_num * H1, -1])
        )

        out_grad_fp8, out_grad_scale = kitchen_quant(
            out_grad, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )

        out_grad_fp8 = out_grad_fp8.reshape([group_num, H1, -1])  # [4, 8448, 7196]
        out_grad_scale = paddle.split(out_grad_scale, num_or_sections=group_num, axis=-1)

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                expert_w2[i].main_grad = kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                )
            else:
                expert_w2[i].grad = kitchen_fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                )

    # ===== dw1 = deep_gemm(input_x_t_fp8, do1_t_fp8)
    def bwd_gate_up_weight(self, do1, input_x, expert_w1):
        # transpose input_x and quant input_x

        group_num = len(expert_w1)
        H1 = input_x.shape[-1]
        input_x = input_x.reshape([group_num, -1, H1]).transpose([0, 2, 1]).contiguous().reshape([group_num * H1, -1])

        input_x_fp8, input_x_scale = kitchen_quant(
            input_x, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        input_x_fp8 = input_x_fp8.reshape([group_num, -1, input_x_fp8.shape[-1]])
        input_x_scale = paddle.split(input_x_scale, num_or_sections=group_num, axis=-1)

        # transpose do1 and quant do1
        H2 = do1.shape[-1]
        do1 = do1.reshape([group_num, -1, H2]).transpose([0, 2, 1]).contiguous().reshape([group_num * H2, -1])
        do1_fp8, do1_scale = kitchen_quant(
            do1, backend=kitchen.ops.Backend.CUBLAS, is_1d_scaled=True, return_transpose=False
        )
        do1_fp8 = do1_fp8.reshape([group_num, -1, do1_fp8.shape[-1]])
        do1_scale = paddle.split(do1_scale, num_or_sections=group_num, axis=-1)

        # dw1
        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                expert_w1[i].main_grad = kitchen_fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                )
            else:
                expert_w1[i].grad = kitchen_fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                )

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert):
        # get w1
        expert_w1 = [x.w1 for x in self.custom_map.experts if x is not None]

        expert_w_count = len(expert_w1)

        # get w2
        expert_w2 = [x.w2 for x in self.custom_map.experts if x is not None]

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, expert_w_count, tokens_per_expert)
        self.o1 = o1

        # o2
        o2 = self.fwd_swiglu(o1)

        unzipped_probs = unzipped_probs.unsqueeze(-1).reshape([expert_w_count, -1, 1])

        o2_s = (o2 * unzipped_probs).cast(paddle.bfloat16)

        # o3
        o3 = self.fwd_down(o2_s, expert_w2, expert_w_count, tokens_per_expert)

        # save for bwd
        self.unzipped_probs = unzipped_probs
        self.unzipped_tokens = hs_out
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, tokens_per_expert, dispatched_indices, expected_m):
        # recompute expert_w2 and expert_w1
        expert_w2 = [x.w2 for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.w1 for x in self.custom_map.experts if x is not None]

        # do2
        do2, o2_s, probs_grad = self.bwd_dowm_input(expert_w2, out_grad, tokens_per_expert, expected_m)

        # do1
        do1 = self.bwd_swiglu(self.o1, do2)

        # dx
        dx = self.bwd_gate_up_input(do1, expert_w1, tokens_per_expert, expected_m)
        dx = dx.reshape([-1, dx.shape[-1]])

        # dw2
        self.bwd_down_weight(out_grad, o2_s, expert_w2)

        # dw1
        self.bwd_gate_up_weight(do1, self.unzipped_tokens, expert_w1)

        self.reset_statue()
        return dx, probs_grad


class ExpertsNode:
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
