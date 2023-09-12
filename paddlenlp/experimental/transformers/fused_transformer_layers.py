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
        encode_rotary_qk,
        qkv_transpose_split,
        rebuild_padding,
        transpose_remove_padding,
        write_cache_kv,
    )
else:
    logger.warning(
        "The paddlenlp_ops package is not installed. you can read the docs and install it by hand, "
        "you can refer to: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
    )


__all__ = ["FusedMultiTransformer"]


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
        inputs["bias"] = dequant_scales

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


class FusedMultiTransformer(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        quant_bits=-1,  # -1 means use Half precision.
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
        epsilon=1e-5,
        residual_alpha=1.0,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        name=None,
    ):
        super().__init__()

        assert embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(embed_dim)
        assert num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(num_heads)
        assert dim_feedforward > 0, "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._residual_alpha = residual_alpha
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id
        self.nranks = nranks
        self.norm_type = norm_type
        if norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        else:
            self.norm_func = fused_rms_norm
        self.use_neox_rotary_style = use_neox_rotary_style
        self._norm_weight_dtype = "float32" if self.norm_type == "layernorm" else self._dtype

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.quant_bits = quant_bits
        self.use_weight_only = False
        self.weight_dtype = self._dtype
        self.create_params_type = self._dtype

        if self.quant_bits != -1:
            self.use_weight_only = True
            self.create_params_type = (
                "int8"  # If use weightonly int4, params dtype is int8, and one of the dimension will be half.
            )
            self.weight_dtype = "int" + str(self.quant_bits)

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_weights_scale, self.qkv_biases = [], [], []
        self.linear_weights, self.linear_weights_scale, self.linear_biases = [], [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_weights_scale, self.ffn1_biases = [], [], []
        self.ffn2_weights, self.ffn2_weights_scale, self.ffn2_biases = [], [], []

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        def _add_parameter(param):
            if param is None:
                return
            assert param.name not in self._parameters
            self._parameters[param.name] = param

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_weight_scale_attr = get_attr(qkv_weight_scale_attrs, i)

            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_weight_scale_attr = get_attr(linear_weight_scale_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_weight_scale_attr = get_attr(ffn1_weight_scale_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_weight_scale_attr = get_attr(ffn2_weight_scale_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            # Note(Zhengzekang): Weightonly need weight is ColMajor layout.
            qkv_weight_shape = (
                [3 * num_heads * self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim * 3 * num_heads, self.head_dim]
            )
            qkv_weight_scale = None
            if self.use_weight_only:
                if self.quant_bits == 4:
                    qkv_weight_shape[0] //= 2

                qkv_weight_scale = self.create_parameter(
                    shape=[3 * num_heads * self.head_dim],
                    attr=qkv_weight_scale_attr,
                    dtype=paddle.float32,
                    is_bias=False,
                )

            qkv_weight = self.create_parameter(
                shape=qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[3 * num_heads * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_weight_shape = [num_heads * self.head_dim, embed_dim]
            linear_weight_scale = None
            if self.use_weight_only:
                linear_weight_shape = [embed_dim, num_heads * self.head_dim]
                if self.quant_bits == 4:
                    linear_weight_shape[0] //= 2

                linear_weight_scale = self.create_parameter(
                    shape=[embed_dim],
                    attr=linear_weight_scale_attr,
                    dtype=paddle.float32,
                    is_bias=False,
                )
            linear_weight = self.create_parameter(
                shape=linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype=self._norm_weight_dtype,
            )

            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            ffn1_weight_shape = (
                [embed_dim, dim_feedforward * 2] if activation.endswith("glu") else [embed_dim, dim_feedforward]
            )
            ffn1_weight_scale = None
            if self.use_weight_only:
                ffn1_weight_shape = (
                    [dim_feedforward * 2, embed_dim] if activation.endswith("glu") else [dim_feedforward, embed_dim]
                )
                if self.quant_bits == 4:
                    ffn1_weight_shape[0] //= 2

                ffn1_weight_scale = self.create_parameter(
                    shape=[dim_feedforward * 2],
                    attr=ffn1_weight_scale_attr,
                    dtype=paddle.float32,
                    is_bias=False,
                )
            ffn1_weight = self.create_parameter(
                shape=ffn1_weight_shape,
                attr=ffn1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn1_bias = None
            if ffn1_bias_attr:
                ffn1_bias = self.create_parameter(
                    shape=[dim_feedforward * 2] if activation.endswith("glu") else [dim_feedforward],
                    attr=ffn1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_weight_shape = [dim_feedforward, embed_dim]
            ffn2_weight_scale = None
            if self.use_weight_only:
                ffn2_weight_shape = [embed_dim, dim_feedforward]
                if self.quant_bits == 4:
                    ffn2_weight_shape[0] //= 2

                ffn2_weight_scale = self.create_parameter(
                    shape=[embed_dim],
                    attr=ffn2_weight_scale_attr,
                    dtype=paddle.float32,
                    is_bias=False,
                )

            ffn2_weight = self.create_parameter(
                shape=ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            # tensor model parallel
            if nranks > 1:
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

            if self.use_weight_only:
                self.qkv_weights_scale.append(qkv_weight_scale)
                self.linear_weights_scale.append(linear_weight_scale)
                self.ffn1_weights_scale.append(ffn1_weight_scale)
                self.ffn2_weights_scale.append(ffn2_weight_scale)

            _add_parameter(ln_scale)
            _add_parameter(ln_bias)
            _add_parameter(qkv_weight)
            _add_parameter(qkv_bias)
            _add_parameter(linear_weight)
            _add_parameter(linear_bias)

            _add_parameter(ffn_ln_scale)
            _add_parameter(ffn_ln_bias)
            _add_parameter(ffn1_weight)
            _add_parameter(ffn1_bias)
            _add_parameter(ffn2_weight)
            _add_parameter(ffn2_bias)

            if self.use_weight_only:
                _add_parameter(qkv_weight_scale)
                _add_parameter(linear_weight_scale)
                _add_parameter(ffn1_weight_scale)
                _add_parameter(ffn2_weight_scale)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name

        from paddle.incubate.nn.functional import fused_linear

        self.linear = fused_linear

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
        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        bias_residual_input = src
        ln_out = src
        for i in range(len(caches)):
            if self.normalize_before is True:
                # layernorm
                if i == 0:
                    ln_out = self.norm_func(
                        src, self.ln_scales[i], self.ln_biases[i], self._epsilon, begin_norm_axis=1
                    )

            # qkv compute
            if self.use_weight_only:
                qkv_out = weight_only_linear(
                    ln_out,
                    weight=self.qkv_weights[i],
                    bias=self.qkv_biases[i],
                    weight_scale=self.qkv_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                )
            else:
                qkv_out = self.linear(ln_out, self.qkv_weights[i], self.qkv_biases[i], transpose_weight=True)

            # fmha compute
            if time_step is None:  # context
                """
                qkv: bsz, seq_len, 3, numhead, headsize ->
                q_out: bsz, numhead, seq_len, headsize
                kv_out: 2, bsz, numhead, seq_len, headsize
                """
                q_out, k_out, v_out = qkv_transpose_split(
                    qkv_out, padding_offset, seq_lens, input_ids, self.num_heads // self.nranks, self.head_dim
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
                    k_out = paddle.concat([pre_caches[i][0], k_out], axis=2)
                    v_out = paddle.concat([pre_caches[i][1], v_out], axis=2)

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

            else:
                fmha_out = masked_multihead_attention(
                    x=qkv_out,
                    cache_kv=caches[i],
                    src_mask=attn_mask,
                    sequence_lengths=seq_lens,
                    rotary_tensor=rotary_embs,
                    rotary_emb_dims=rotary_emb_dims,
                    use_neox_rotary_style=self.use_neox_rotary_style,
                )[0]

            # out_linear
            if self.use_weight_only:
                out_linear_out = weight_only_linear(
                    fmha_out,
                    weight=self.linear_weights[i],
                    weight_scale=self.linear_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                )
            else:
                out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i])

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)

            # norm + residual_add_bias
            if self.normalize_before is True:
                norm_out = self.norm_func(
                    out_linear_out,
                    norm_weight=self.ffn_ln_scales[i],
                    norm_bias=self.ffn_ln_biases[i],
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                    bias=self.linear_biases[i],
                    residual=bias_residual_input,
                )
                tmp_out, bias_residual_input = norm_out[0], norm_out[1]
            else:
                tmp_out = self.norm_func(
                    out_linear_out,
                    norm_weight=self.ln_scales[i],
                    norm_bias=self.ln_biases[i],
                    epsilon=self._epsilon,
                    residual_alpha=self._residual_alpha,
                    begin_norm_axis=1,
                    bias=self.linear_biases[i],
                    residual=ln_out,
                )[0]

            # ffn1 matmul
            if self.use_weight_only:
                ffn1_out = weight_only_linear(
                    tmp_out,
                    weight=self.ffn1_weights[i],
                    weight_scale=self.ffn1_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                )
            else:
                ffn1_out = paddle.matmul(tmp_out, self.ffn1_weights[i])
            ffn1_out = fused_act_bias_wrapper(ffn1_out, self.ffn1_biases[i], act_method=self.activation)

            # ffn2 matmul
            if self.use_weight_only:
                ffn2_out = weight_only_linear(
                    ffn1_out,
                    weight=self.ffn2_weights[i],
                    weight_scale=self.ffn2_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                )
            else:
                ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i])

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias
            if self.normalize_before is True:
                if i != len(caches) - 1:
                    norm_out = self.norm_func(
                        ffn2_out,
                        norm_weight=self.ln_scales[i + 1],
                        norm_bias=self.ln_biases[i + 1],
                        epsilon=self._epsilon,
                        begin_norm_axis=1,
                        bias=self.ffn2_biases[i],
                        residual=bias_residual_input,
                    )
                    tmp_out, bias_residual_input = norm_out[0], norm_out[1]
                else:
                    tmp_out = fused_layer_norm(
                        ffn2_out,
                        norm_weight=None,
                        norm_bias=None,
                        epsilon=self._epsilon,
                        begin_norm_axis=1,
                        bias=self.ffn2_biases[i],
                        residual=bias_residual_input,
                    )[0]
            else:
                tmp_out = self.norm_func(
                    ffn2_out,
                    norm_weight=self.ffn_ln_scales[i],
                    norm_bias=self.ffn_ln_biases[i],
                    epsilon=self._epsilon,
                    residual_alpha=self._residual_alpha,
                    begin_norm_axis=1,
                    bias=self.ffn2_biases[i],
                    residual=tmp_out,
                )[0]

            ln_out = tmp_out

        if time_step is None:
            out = rebuild_padding(tmp_out, cum_offsets, seq_lens, input_ids)
        else:
            out = tmp_out
        return out, caches
