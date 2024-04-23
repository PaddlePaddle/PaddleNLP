# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import paddle
import paddle.distributed as dist
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.incubate.nn.functional import (
    fused_layer_norm,
    fused_rms_norm,
    masked_multihead_attention,
    variable_length_memory_efficient_attention,
)
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.quant import weight_only_linear

from paddlenlp.utils.import_utils import is_paddlenlp_ops_available
from paddlenlp.utils.log import logger

if is_paddlenlp_ops_available():
    from paddlenlp_ops import (
        dequant_int8,
        encode_rotary_qk,
        qkv_transpose_split,
        quant_int8,
        rebuild_padding,
        rebuild_padding_v2,
        transpose_remove_padding,
        write_cache_kv,
    )
else:
    logger.warning(
        "The paddlenlp_ops package is not installed. you can read the docs and install it by hand, "
        "you can refer to: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
    )


__all__ = [
    "FusedMultiTransformerConfig",
    "FusedMultiTransformerBase",
    "FusedMultiTransformerPostLayernorm",
    "FusedMultiTransformerWeightOnly",
    "FusedMultiTransformerWeightOnlyPostLayernorm",
    "FusedBlockMultiTransformer",
    "FusedBlockMultiTransformerWeightOnly",
    "FusedBlockMultiTransformerA8W8",
]


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not in_dynamic_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["dequant_scales"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


class FusedMultiTransformerConfig:
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        weight_only_quant_bits=-1,  # -1 means use Half precision.
        dropout_rate=0.0,
        activation="gelu",
        norm_type="layernorm",
        use_neox_rotary_style=False,
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_weight_scale_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_weight_scale_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_weight_scale_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_weight_scale_attrs=None,
        ffn2_bias_attrs=None,
        qkv_out_scale_attrs=None,
        linear_out_scale_attrs=None,
        ffn1_out_scale_attrs=None,
        ffn2_out_scale_attrs=None,
        linear_shift_attrs=None,
        linear_smooth_attrs=None,
        ffn2_shift_attrs=None,
        ffn2_smooth_attrs=None,
        cache_k_scale_attrs=None,
        cache_v_scale_attrs=None,
        cache_k_out_scale_attrs=None,
        cache_v_out_scale_attrs=None,
        quant_round_type=0,
        quant_max_bound=127.0,
        quant_min_bound=-127.0,
        epsilon=1e-5,
        residual_alpha=1.0,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        kv_num_heads=-1,
        use_dynamic_cachekv_quant=True,
        rank_id=-1,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if kv_num_heads > 0:
            self.kv_num_heads = kv_num_heads
            assert nranks == 1, "nranks should be 1 for kv_num_heads > 0"
        else:
            self.kv_num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.weight_only_quant_bits = weight_only_quant_bits
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_type = norm_type

        self.use_neox_rotary_style = use_neox_rotary_style
        self.normalize_before = normalize_before
        self.ln_scale_attrs = ln_scale_attrs
        self.ln_bias_attrs = ln_bias_attrs
        self.qkv_weight_attrs = qkv_weight_attrs
        self.qkv_weight_scale_attrs = qkv_weight_scale_attrs
        self.qkv_bias_attrs = qkv_bias_attrs
        self.linear_weight_attrs = linear_weight_attrs
        self.linear_weight_scale_attrs = linear_weight_scale_attrs
        self.linear_bias_attrs = linear_bias_attrs
        self.ffn_ln_scale_attrs = ffn_ln_scale_attrs
        self.ffn_ln_bias_attrs = ffn_ln_bias_attrs
        self.ffn1_weight_attrs = ffn1_weight_attrs
        self.ffn1_weight_scale_attrs = ffn1_weight_scale_attrs
        self.ffn1_bias_attrs = ffn1_bias_attrs
        self.ffn2_weight_attrs = ffn2_weight_attrs
        self.ffn2_weight_scale_attrs = ffn2_weight_scale_attrs
        self.ffn2_bias_attrs = ffn2_bias_attrs

        self.qkv_out_scale_attrs = qkv_out_scale_attrs
        self.linear_out_scale_attrs = linear_out_scale_attrs
        self.ffn1_out_scale_attrs = ffn1_out_scale_attrs
        self.ffn2_out_scale_attrs = ffn2_out_scale_attrs
        self.linear_shift_attrs = linear_shift_attrs
        self.linear_smooth_attrs = linear_smooth_attrs
        self.ffn2_shift_attrs = ffn2_shift_attrs
        self.ffn2_smooth_attrs = ffn2_smooth_attrs
        self.cache_k_scale_attrs = cache_k_scale_attrs
        self.cache_v_scale_attrs = cache_v_scale_attrs
        self.cache_k_out_scale_attrs = cache_k_out_scale_attrs
        self.cache_v_out_scale_attrs = cache_v_out_scale_attrs

        self.quant_round_type = quant_round_type
        self.quant_max_bound = quant_max_bound
        self.quant_min_bound = quant_min_bound
        self.use_dynamic_cachekv_quant = use_dynamic_cachekv_quant

        self.epsilon = epsilon
        self.residual_alpha = residual_alpha
        self.num_layers = num_layers
        self.nranks = nranks
        self.rank_id = rank_id
        self.trans_qkvw = trans_qkvw
        self.ring_id = ring_id


class FusedMultiTransformerBase(Layer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__()

        self.config = config

        assert config.embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(
            config.embed_dim
        )
        assert config.num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(config.num_heads)
        assert config.dim_feedforward > 0, "Expected dim_feedforward to be greater than 0, but received {}".format(
            config.dim_feedforward
        )

        # self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = config.epsilon
        self._residual_alpha = config.residual_alpha
        self._trans_qkvw = config.trans_qkvw
        self._ring_id = config.ring_id
        self.nranks = config.nranks
        self.norm_type = config.norm_type
        if self.norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = fused_rms_norm
        else:
            raise NotImplementedError("Only support norm type of [layernorm, rmsnorm]")
        self.use_neox_rotary_style = config.use_neox_rotary_style
        self._norm_weight_dtype = "float32" if self.norm_type == "layernorm" else self._dtype

        self.activation = config.activation

        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        assert self.head_dim * config.num_heads == config.embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if config.nranks > 1:
            assert config.ring_id != -1
        assert config.num_heads % config.nranks == 0
        assert config.dim_feedforward % config.nranks == 0
        self.num_heads = config.num_heads // config.nranks
        self.kv_num_heads = config.kv_num_heads // config.nranks
        dim_feedforward = config.dim_feedforward // config.nranks
        self.dim_feedforward = dim_feedforward

        self.num_layers = config.num_layers
        assert self.num_layers > 0
        if isinstance(config.qkv_weight_attrs, (list, tuple)):
            assert self.num_layers == len(config.qkv_weight_attrs)

        self.weight_dtype = self._dtype
        self.create_params_type = self.get_weight_create_dype()

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []
        self.cache_k_scales, self.cache_v_scales = [], []
        self.cache_k_out_scales, self.cache_v_out_scales = [], []

        for i in range(self.num_layers):
            ln_scale_attr = self.get_attr(config.ln_scale_attrs, i)
            ln_bias_attr = self.get_attr(config.ln_bias_attrs, i)
            qkv_weight_attr = self.get_attr(config.qkv_weight_attrs, i)

            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            linear_weight_attr = self.get_attr(config.linear_weight_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)

            ffn_ln_scale_attr = self.get_attr(config.ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = self.get_attr(config.ffn_ln_bias_attrs, i)
            ffn1_weight_attr = self.get_attr(config.ffn1_weight_attrs, i)
            ffn1_bias_attr = self.get_attr(config.ffn1_bias_attrs, i)
            ffn2_weight_attr = self.get_attr(config.ffn2_weight_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)

            cache_k_scale_attr = self.get_attr(config.cache_k_scale_attrs, i)
            cache_v_scale_attr = self.get_attr(config.cache_v_scale_attrs, i)
            cache_k_out_scale_attr = self.get_attr(config.cache_k_out_scale_attrs, i)
            cache_v_out_scale_attr = self.get_attr(config.cache_v_out_scale_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            self.init_weight_shape(config)

            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn_ln_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype=self._norm_weight_dtype,
            )

            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            ffn1_weight = self.create_parameter(
                shape=self.ffn1_weight_shape,
                attr=ffn1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn1_bias = None
            if ffn1_bias_attr:
                ffn1_bias = self.create_parameter(
                    shape=[dim_feedforward * 2] if config.activation.endswith("glu") else [dim_feedforward],
                    attr=ffn1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            cache_k_scale = None
            if cache_k_scale_attr:
                cache_k_scale = self.create_parameter(
                    shape=[self.num_heads],
                    attr=cache_k_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            cache_v_scale = None
            if cache_v_scale_attr:
                cache_v_scale = self.create_parameter(
                    shape=[self.num_heads],
                    attr=cache_v_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            cache_k_out_scale = None
            if cache_k_out_scale_attr:
                cache_k_out_scale = self.create_parameter(
                    shape=[self.num_heads],
                    attr=cache_k_out_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            cache_v_out_scale = None
            if cache_v_out_scale_attr:
                cache_v_out_scale = self.create_parameter(
                    shape=[self.num_heads],
                    attr=cache_v_out_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            # tensor model parallel
            if config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            self.cache_k_scales.append(cache_k_scale)
            self.cache_v_scales.append(cache_v_scale)
            self.cache_k_out_scales.append(cache_k_out_scale)
            self.cache_v_out_scales.append(cache_v_out_scale)

            self._add_parameter(ln_scale)
            self._add_parameter(ln_bias)
            self._add_parameter(qkv_weight)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_weight)
            self._add_parameter(linear_bias)

            self._add_parameter(ffn_ln_scale)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn1_bias)
            self._add_parameter(ffn2_weight)
            self._add_parameter(ffn2_bias)

            self._add_parameter(cache_k_scale)
            self._add_parameter(cache_v_scale)
            self._add_parameter(cache_k_out_scale)
            self._add_parameter(cache_v_out_scale)

        self.dropout_rate = config.dropout_rate

        from paddle.incubate.nn.functional import fused_linear

        self.linear = fused_linear

    def get_attr(self, attrs, idx):
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param

    def init_weight_shape(self, config):
        self.qkv_weight_shape = (
            [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
            if config.trans_qkvw
            else [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
        )
        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]
        self.ffn1_weight_shape = (
            [self.embed_dim, self.dim_feedforward * 2]
            if self.activation.endswith("glu")
            else [self.embed_dim, self.dim_feedforward]
        )
        self.ffn2_weight_shape = [self.dim_feedforward, self.embed_dim]

    def get_weight_create_dype(self):
        return self._dtype

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0:
            ln_out = self.norm_func(src, self.ln_scales[i], self.ln_biases[i], self._epsilon, begin_norm_axis=1)
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if float(paddle.version.cuda()) < 11.6:
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            if self.qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            return qkv_out
        else:
            # This method requires CUDA version >= 11.6.
            return self.linear(ln_out, self.qkv_weights[i], self.qkv_biases[i], transpose_weight=True)

    def compute_qkv(self, src, residual_input, i):
        ln_out = self.compute_layernorm_before_qkv(src, i)
        qkv_out = self.compute_qkv_linear(ln_out, i)
        return qkv_out, residual_input

    def compute_fmha(
        self,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
    ):
        bsz = input_ids.shape[0]
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        q_out, k_out, v_out = qkv_transpose_split(
            qkv_out, padding_offset, seq_lens, input_ids, self.num_heads, self.head_dim
        )

        # rotary emb (inplace)
        if rotary_embs is not None:
            encode_rotary_qk(
                q_out,
                k_out,
                rotary_embs,
                seq_lens,
                rotary_emb_dims=rotary_emb_dims,
                use_neox=self.use_neox_rotary_style,
            )

        if pre_caches is not None:
            k_out = paddle.concat([pre_caches[i][0, :bsz], k_out], axis=2)
            v_out = paddle.concat([pre_caches[i][1, :bsz], v_out], axis=2)

        # write cache kv (inplace)
        write_cache_kv(k_out, v_out, caches[i], seq_lens + pre_caches_length)

        # cutlass fmha
        qktv_out = variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens + pre_caches_length,
            mask=attn_mask,
            scale=float(self.head_dim**-0.5),
        )

        return transpose_remove_padding(qktv_out, seq_lens, padding_offset)

    def compute_mmha(self, qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i):
        return masked_multihead_attention(
            x=qkv_out,
            cache_kv=caches[i],
            src_mask=attn_mask,
            sequence_lengths=seq_lens,
            rotary_tensor=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            use_neox_rotary_style=self.use_neox_rotary_style,
        )[0]

    def compute_out_linear(self, fmha_out, i):
        return paddle.matmul(fmha_out, self.linear_weights[i])

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        # fmha compute
        if time_step is None:  # context
            fmha_out = self.compute_fmha(
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
            )

        else:
            fmha_out = self.compute_mmha(qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i)

        out_linear_out = self.compute_out_linear(fmha_out, i)

        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        norm_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_activation(self, ffn1_out, i):
        return fused_act_bias_wrapper(ffn1_out, self.ffn1_biases[i], act_method=self.activation)

    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i])

    def compute_ffn2(self, ffn1_out, i):
        return paddle.matmul(ffn1_out, self.ffn2_weights[i])

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):

        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                norm_weight=self.ln_scales[i + 1],
                norm_bias=self.ln_biases[i + 1],
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input

    def pre_process(self, **kwargs):
        pass

    def post_process(self, **kwargs):
        time_step = kwargs.get("time_step", None)
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens = kwargs.get("seq_lens", None)
        input_ids = kwargs.get("input_ids", None)

        if time_step is None:
            out = rebuild_padding(multi_block_output, cum_offsets, seq_lens, input_ids)
        else:
            out = multi_block_output

        return out

    def forward(
        self,
        input_ids,
        src,
        cum_offsets=None,
        padding_offset=None,
        attn_mask=None,
        caches=None,
        pre_caches=None,
        pre_caches_length=0,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
        **kwargs,
    ):
        r"""
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`. Default None.
            rotary_embs (Tensor optional): The RoPE embs for the rotary computation. The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
            rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation, and it is 0 when rotary_embs is None,
                1 when rotary_embs is not None and pos_extra_ids is None, 2 when rotary_embs and pos_extra_ids are both not None. Default 0.
            seq_lens (Tensor optional): The sequence lengths of this batch. The shape is `[bsz]`. Default None.
            time_step (Tensor, optional): The time step tensor for the generation
                model. Which used in decode stage, to represent the time step,
                that is, the real seq_len of CacheKV. The shape is `[1]`, must be
                in CPUPlace. Default None.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """
        self.pre_process(**kwargs)
        kwargs["cum_offsets"] = cum_offsets

        if caches is not None:
            assert len(caches) == len(self.qkv_weights) or len(caches) == 2 * len(self.qkv_weights)

        assert self.num_layers == len(self.qkv_weights)

        residual_input = src
        for i in range(self.num_layers):
            qkv_out, residual_input = self.compute_qkv(src, residual_input, i)
            out_linear_out = self.compute_attn(
                time_step,
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
                **kwargs,
            )
            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)

            # ffn layernorm
            tmp_out, residual_input = self.compute_ffn_layernorm(out_linear_out, residual_input, i)

            # ffn1 matmul
            ffn1_out = self.compute_ffn1(tmp_out, i)
            ffn1_out = self.compute_activation(ffn1_out, i)

            # ffn2 matmul
            ffn2_out = self.compute_ffn2(ffn1_out, i)

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias
            tmp_out, residual_input = self.compute_bias_residual_layernorm(
                ffn2_out, residual_input, i, self.num_layers
            )
            src = tmp_out

        kwargs["time_step"] = time_step
        kwargs["multi_block_output"] = tmp_out
        kwargs["seq_lens"] = seq_lens
        kwargs["input_ids"] = input_ids

        out = self.post_process(**kwargs)
        return out, caches


class FusedMultiTransformerPostLayernorm(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_qkv(self, src, residual_input, i):
        qkv_out = self.compute_qkv_linear(src, i)
        return qkv_out, src

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        tmp_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ln_scales[i],
            norm_bias=self.ln_biases[i],
            epsilon=self._epsilon,
            residual_alpha=self._residual_alpha,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
        )[0]

        return tmp_out, tmp_out

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        tmp_out = self.norm_func(
            ffn2_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            residual_alpha=self._residual_alpha,
            begin_norm_axis=1,
            bias=self.ffn2_biases[i],
            residual=residual_input,
        )[0]
        return tmp_out, tmp_out


class FusedMultiTransformerWeightOnly(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.weight_only_quant_bits = config.weight_only_quant_bits

        assert self.weight_only_quant_bits != -1
        self.weight_dtype = "int" + str(self.weight_only_quant_bits)
        self.weight_scale_dtype = self._dtype
        self.qkv_weights_scale = []
        self.linear_weights_scale = []
        self.ffn1_weights_scale = []
        self.ffn2_weights_scale = []

        for i in range(self.num_layers):

            qkv_weight_scale_attr = self.get_attr(config.qkv_weight_scale_attrs, i)
            linear_weight_scale_attr = self.get_attr(config.linear_weight_scale_attrs, i)
            ffn1_weight_scale_attr = self.get_attr(config.ffn1_weight_scale_attrs, i)
            ffn2_weight_scale_attr = self.get_attr(config.ffn2_weight_scale_attrs, i)

            qkv_weight_scale = self.create_parameter(
                shape=[(config.num_heads + 2 * config.kv_num_heads) * self.head_dim],
                attr=qkv_weight_scale_attr,
                dtype=self.weight_scale_dtype,
                is_bias=False,
            )

            linear_weight_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=linear_weight_scale_attr,
                dtype=self.weight_scale_dtype,
                is_bias=False,
            )

            ffn1_weight_scale = self.create_parameter(
                shape=[config.dim_feedforward * 2] if config.activation.endswith("glu") else [config.dim_feedforward],
                attr=ffn1_weight_scale_attr,
                dtype=self.weight_scale_dtype,
                is_bias=False,
            )

            ffn2_weight_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn2_weight_scale_attr,
                dtype=self.weight_scale_dtype,
                is_bias=False,
            )

            self.qkv_weights_scale.append(qkv_weight_scale)
            self.linear_weights_scale.append(linear_weight_scale)
            self.ffn1_weights_scale.append(ffn1_weight_scale)
            self.ffn2_weights_scale.append(ffn2_weight_scale)

            self._add_parameter(qkv_weight_scale)
            self._add_parameter(linear_weight_scale)
            self._add_parameter(ffn1_weight_scale)
            self._add_parameter(ffn2_weight_scale)

    def get_weight_create_dype(self):
        return "int8"  # If use weightonly int4, params dtype is int8, and one of the dimension will be half.

    def init_weight_shape(self, config):
        super().init_weight_shape(config)

        self.linear_weight_shape = [self.embed_dim, self.num_heads * self.head_dim]
        self.ffn1_weight_shape = (
            [self.dim_feedforward * 2, self.embed_dim]
            if self.activation.endswith("glu")
            else [self.dim_feedforward, self.embed_dim]
        )
        self.ffn2_weight_shape = [self.embed_dim, self.dim_feedforward]

        if config.weight_only_quant_bits == 4:
            self.qkv_weight_shape[0] //= 2
            self.linear_weight_shape[0] //= 2
            self.ffn1_weight_shape[0] //= 2
            self.ffn2_weight_shape[0] //= 2

    def compute_qkv_linear(self, ln_out, i):
        return weight_only_linear(
            ln_out,
            weight=self.qkv_weights[i],
            bias=self.qkv_biases[i],
            weight_scale=self.qkv_weights_scale[i],
            weight_dtype=self.weight_dtype,
        )

    def compute_out_linear(self, fmha_out, i):
        return weight_only_linear(
            fmha_out,
            weight=self.linear_weights[i],
            weight_scale=self.linear_weights_scale[i],
            weight_dtype=self.weight_dtype,
        )

    def compute_ffn1(self, tmp_out, i):
        return weight_only_linear(
            tmp_out,
            weight=self.ffn1_weights[i],
            weight_scale=self.ffn1_weights_scale[i],
            weight_dtype=self.weight_dtype,
        )

    def compute_ffn2(self, ffn1_out, i):
        return weight_only_linear(
            ffn1_out,
            weight=self.ffn2_weights[i],
            weight_scale=self.ffn2_weights_scale[i],
            weight_dtype=self.weight_dtype,
        )


class FusedMultiTransformerWeightOnlyPostLayernorm(
    FusedMultiTransformerWeightOnly, FusedMultiTransformerPostLayernorm
):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)


class FusedMultiTransformerA8W8(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.quant_round_type = config.quant_round_type
        self.quant_max_bound = config.quant_max_bound
        self.quant_min_bound = config.quant_min_bound

        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(
                "FusedMultiTransformer just support float32, float16 and bfloat16 as default dtype, but received {}".format(
                    self._dtype
                )
            )

        self.qkv_out_scales = []
        self.linear_out_scales = []
        self.ffn1_out_scales = []
        self.ffn2_out_scales = []

        self.linear_shifts, self.linear_smooths, self.ffn2_shifts, self.ffn2_smooths = [], [], [], []

        for i in range(self.num_layers):
            qkv_out_scale_attr = self.get_attr(config.qkv_out_scale_attrs, i)
            linear_out_scale_attr = self.get_attr(config.linear_out_scale_attrs, i)
            ffn1_out_scale_attr = self.get_attr(config.ffn1_out_scale_attrs, i)
            ffn2_out_scale_attr = self.get_attr(config.ffn2_out_scale_attrs, i)

            linear_shift_attr = self.get_attr(config.linear_shift_attrs, i)
            linear_smooth_attr = self.get_attr(config.linear_smooth_attrs, i)
            ffn2_shift_attr = self.get_attr(config.ffn2_shift_attrs, i)
            ffn2_smooth_attr = self.get_attr(config.ffn2_smooth_attrs, i)

            qkv_out_scale = self.create_parameter(
                shape=[self.head_dim * 3 * self.num_heads],
                attr=qkv_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            linear_out_scale = self.create_parameter(
                shape=[self.embed_dim],
                attr=linear_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            ffn1_out_scale = self.create_parameter(
                shape=[self.dim_feedforward * 2] if self.activation.endswith("glu") else [self.dim_feedforward],
                attr=ffn1_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            ffn2_out_scale = self.create_parameter(
                shape=[self.embed_dim],
                attr=ffn2_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            linear_shift = None
            if linear_shift_attr:
                linear_shift = self.create_parameter(
                    shape=[self.num_heads * self.head_dim], attr=linear_shift_attr, dtype=self._dtype, is_bias=False
                )

            linear_smooth = None
            if linear_smooth_attr:
                linear_smooth = self.create_parameter(
                    shape=[self.num_heads * self.head_dim], attr=linear_smooth_attr, dtype=self._dtype, is_bias=False
                )

            ffn2_shift = None
            if ffn2_shift_attr:
                ffn2_shift = self.create_parameter(
                    shape=[self.dim_feedforward], attr=ffn2_shift_attr, dtype=self._dtype, is_bias=False
                )

            ffn2_smooth = None
            if ffn2_smooth_attr:
                ffn2_smooth = self.create_parameter(
                    shape=[self.dim_feedforward], attr=ffn2_smooth_attr, dtype=self._dtype, is_bias=False
                )

            self.qkv_out_scales.append(qkv_out_scale)
            self.linear_out_scales.append(linear_out_scale)
            self.ffn1_out_scales.append(ffn1_out_scale)
            self.ffn2_out_scales.append(ffn2_out_scale)

            if linear_shift is not None:
                self.linear_shifts.append(linear_shift)
                self.linear_smooths.append(linear_smooth)
                self.ffn2_shifts.append(ffn2_shift)
                self.ffn2_smooths.append(ffn2_smooth)

            self._add_parameter(qkv_out_scale)
            self._add_parameter(linear_out_scale)
            self._add_parameter(ffn1_out_scale)
            self._add_parameter(ffn2_out_scale)

            self._add_parameter(linear_shift)
            self._add_parameter(linear_smooth)
            self._add_parameter(ffn2_shift)
            self._add_parameter(ffn2_smooth)

    def get_weight_create_dype(self):
        return "int8"

    def init_weight_shape(self, config):
        super().init_weight_shape(config)

        self.linear_weight_shape = [self.embed_dim, self.num_heads * self.head_dim]
        self.ffn1_weight_shape = (
            [self.dim_feedforward * 2, self.embed_dim]
            if self.activation.endswith("glu")
            else [self.dim_feedforward, self.embed_dim]
        )
        self.ffn2_weight_shape = [self.embed_dim, self.dim_feedforward]

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0:
            ln_out = self.norm_func(
                src,
                self.ln_scales[i],
                self.ln_biases[i],
                self._epsilon,
                begin_norm_axis=1,
                quant_scale=self.act_scales["qkv_in_scale"][i],  # quant_in_scale
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
        return qkv_out

    def compute_fmha(
        self,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
    ):
        qkv_out = dequant_int8(qkv_out, self.qkv_out_scales[i], self._dtype)
        if self.qkv_biases[i] is not None:
            qkv_out = paddle.add(qkv_out, self.qkv_biases[i])

        bsz = input_ids.shape[0]
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        q_out, k_out, v_out = qkv_transpose_split(
            qkv_out, padding_offset, seq_lens, input_ids, self.num_heads, self.head_dim
        )

        # rotary emb (inplace)
        if rotary_embs is not None:
            encode_rotary_qk(
                q_out,
                k_out,
                rotary_embs,
                seq_lens,
                rotary_emb_dims=rotary_emb_dims,
                use_neox=self.use_neox_rotary_style,
            )

        if pre_caches is not None:
            k_out = paddle.concat([pre_caches[i][0, :bsz], k_out], axis=2)
            v_out = paddle.concat([pre_caches[i][1, :bsz], v_out], axis=2)

        # write cache kv (inplace)
        write_cache_kv(k_out, v_out, caches[i], seq_lens + pre_caches_length)

        # cutlass fmha
        qktv_out = variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens + pre_caches_length,
            mask=attn_mask,
            scale=float(self.head_dim**-0.5),
        )

        fmha_out = transpose_remove_padding(qktv_out, seq_lens, padding_offset)
        fmha_out = quant_int8(
            fmha_out,
            self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
            self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
            self.act_scales["out_linear_in_scale"][i],
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        return fmha_out

    def compute_mmha(self, qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i):
        return masked_multihead_attention(
            x=qkv_out,
            bias=self.qkv_biases[i],
            cache_kv=caches[i],
            src_mask=attn_mask,
            sequence_lengths=seq_lens,
            rotary_tensor=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            use_neox_rotary_style=self.use_neox_rotary_style,
            qkv_out_scale=self.qkv_out_scales[i],
            out_shift=self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
            out_smooth=self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
            out_scale=self.act_scales["out_linear_in_scale"][i],
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
            compute_dtype=self._fuse_kernel_compute_dtype,
        )[0]

    def compute_out_linear(self, fmha_out, i):
        out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i], False, True)
        return dequant_int8(out_linear_out, self.linear_out_scales[i], self._dtype)

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        norm_out = self.norm_func(
            out_linear_out,
            self.ffn_ln_scales[i],
            self.ffn_ln_biases[i],
            self._epsilon,
            bias=self.linear_biases[i],
            residual=residual_input,
            begin_norm_axis=1,
            quant_scale=self.act_scales["ffn1_in_scale"][i],  # quant_in_scale
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_activation(self, ffn1_out, i):
        return fused_act_bias_wrapper(
            ffn1_out,
            self.ffn1_biases[i],
            act_method=self.activation,
            compute_dtype=self._fuse_kernel_compute_dtype,
            dequant_scales=self.ffn1_out_scales[i],
            shift=self.ffn2_shifts[i] if len(self.ffn2_shifts) > 0 else None,
            smooth=self.ffn2_smooths[i] if len(self.ffn2_smooths) > 0 else None,
            quant_scale=self.act_scales["ffn2_in_scale"][i],
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i], False, True)

    def compute_ffn2(self, ffn1_out, i):
        ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i], False, True)
        ffn2_out = dequant_int8(ffn2_out, self.ffn2_out_scales[i], self._dtype)
        return ffn2_out

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                self.ln_scales[i + 1],
                self.ln_biases[i + 1],
                self._epsilon,
                residual=residual_input,
                begin_norm_axis=1,
                quant_scale=self.act_scales["qkv_in_scale"][i + 1],
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input


class FusedBlockMultiTransformer(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)

        if not self.config.use_dynamic_cachekv_quant:
            k_quant_scales = self.cache_k_scales
            v_quant_scales = self.cache_v_scales
            k_dequant_scales = self.cache_k_out_scales
            v_dequant_scales = self.cache_v_out_scales

        fmha_out = paddle.incubate.nn.functional.block_multihead_attention(
            qkv_out,
            caches[2 * i],
            caches[2 * i + 1],
            kwargs.get("seq_lens_encoder", None),
            kwargs.get("seq_lens_decoder", None),
            kwargs.get("seq_lens_this_time", None),
            kwargs.get("padding_offsets", None),
            kwargs.get("cum_offsets", None),
            kwargs.get("cu_seqlens_q", None),
            kwargs.get("cu_seqlens_k", None),
            kwargs.get("block_tables", None),
            pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
            pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
            k_quant_scales[i] if k_quant_scales is not None else None,
            v_quant_scales[i] if v_quant_scales is not None else None,
            k_dequant_scales[i] if k_dequant_scales is not None else None,
            v_dequant_scales[i] if v_dequant_scales is not None else None,
            None,  # qkv_out_scales
            None,  # qkv_bias
            None,  # out_shifts
            None,  # out_smooths
            rotary_embs,
            attn_mask,
            kwargs.get("tgt_mask", None),
            kwargs.get("max_input_length", -1),
            kwargs.get("block_size", 64),
            self.use_neox_rotary_style,
            self.config.use_dynamic_cachekv_quant,
            quant_round_type=self.config.quant_round_type,
            quant_max_bound=self.config.quant_max_bound,
            quant_min_bound=self.config.quant_min_bound,
        )[0]

        out_linear_out = self.compute_out_linear(fmha_out, i)

        return out_linear_out

    def post_process(self, **kwargs):
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)
        seq_lens_decoder = kwargs.get("seq_lens_decoder", None)
        max_input_length = kwargs.get("max_input_length", -1)

        out = rebuild_padding_v2(multi_block_output, cum_offsets, seq_lens_decoder, seq_lens_encoder, max_input_length)

        return out


class FusedBlockMultiTransformerWeightOnly(FusedBlockMultiTransformer, FusedMultiTransformerWeightOnly):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)


class FusedBlockMultiTransformerA8W8(FusedBlockMultiTransformer, FusedMultiTransformerA8W8):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)

        if not self.config.use_dynamic_cachekv_quant:
            k_quant_scales = self.cache_k_scales
            v_quant_scales = self.cache_v_scales
            k_dequant_scales = self.cache_k_out_scales
            v_dequant_scales = self.cache_v_out_scales

        fmha_out = paddle.incubate.nn.functional.block_multihead_attention(
            qkv_out,
            caches[2 * i],
            caches[2 * i + 1],
            kwargs.get("seq_lens_encoder", None),
            kwargs.get("seq_lens_decoder", None),
            kwargs.get("seq_lens_this_time", None),
            kwargs.get("padding_offsets", None),
            kwargs.get("cum_offsets", None),
            kwargs.get("cu_seqlens_q", None),
            kwargs.get("cu_seqlens_k", None),
            kwargs.get("block_tables", None),
            pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
            pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
            k_quant_scales[i] if k_quant_scales is not None else None,
            v_quant_scales[i] if v_quant_scales is not None else None,
            k_dequant_scales[i] if k_dequant_scales is not None else None,
            v_dequant_scales[i] if v_dequant_scales is not None else None,
            self.qkv_out_scales[i],
            self.qkv_biases[i] if len(self.qkv_biases) > 0 else None,
            self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
            self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
            rotary_embs,
            attn_mask,
            kwargs.get("tgt_mask", None),
            kwargs.get("max_input_length", -1),
            kwargs.get("block_size", 64),
            self.use_neox_rotary_style,
            self.config.use_dynamic_cachekv_quant,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
            out_scale=self.act_scales["out_linear_in_scale"][i],
            compute_dtype=self._fuse_kernel_compute_dtype,
        )[0]

        out_linear_out = self.compute_out_linear(fmha_out, i)

        return out_linear_out
