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

import paddle
import paddle.distributed as dist
from custom_setup_ops import (
    encode_rotary_qk,
    qkv_transpose_split,
    rebuild_padding,
    transpose_remove_padding,
    write_cache_kv,
)
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.nn import Layer
from paddle.nn.initializer import Constant


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
    else:
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


def norm_helper(
    x,
    residual,
    bias,
    norm_weight,
    norm_bias,
    epsilon,
    residual_alpha,
    norm_type,
    begin_norm_axis,
):
    r"""
    Apply LayerNorm / RMSNorm kernel.
    Args:
        x (Tensor): the input Tensor..
        residual (Tensor, optional): the residual Tensor.
        bias (Tensor, optional): the bias Tensor.
        norm_weight (Tensor): the weight Tensor to affine output.
        bias (Tensor): the bias Tensor to affine output.
        epsilon (float): a small float number to avoid divide 0.
        residual_alpha (float): a factor to scale the residual input.
        norm_type (str): the normalize type, currently only accept `layernorm`, `rmsnorm`.
        begin_norm_axis (int): the begin axis to normalize.
    Returns:
        Tensor: the output Tensor.
    Examples:
        .. code-block:: python
            # required: gpu
    """

    if in_dynamic_mode():
        if residual is not None:
            return paddle._C_ops.norm_helper(
                x,
                residual,
                bias,
                norm_weight,
                norm_bias,
                epsilon,
                residual_alpha,
                norm_type,
                begin_norm_axis,
            )[0:2]
        else:
            return paddle._C_ops.norm_helper(
                x,
                residual,
                bias,
                norm_weight,
                norm_bias,
                epsilon,
                residual_alpha,
                norm_type,
                begin_norm_axis,
            )[0]

    helper = LayerHelper("norm_helper", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mean = helper.create_variable_for_type_inference(dtype=paddle.float32)
    variance = helper.create_variable_for_type_inference(dtype=paddle.float32)
    outputs_dict = {}
    outputs_dict["out"] = out
    outputs_dict["mean"] = mean
    outputs_dict["variance"] = variance

    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    outputs_dict["residual_out"] = residual_out

    inputs = {}
    inputs["x"] = x
    if residual is not None:
        inputs["residual"] = residual
    if bias is not None:
        inputs["bias"] = bias

    if norm_weight is not None:
        inputs["norm_weight"] = norm_weight
    if norm_bias is not None:
        inputs["norm_bias"] = norm_bias

    helper.append_op(
        type="norm_helper",
        inputs=inputs,
        attrs={
            "epsilon": epsilon,
            "residual_alpha": residual_alpha,
            "norm_type": norm_type,
            "begin_norm_axis": begin_norm_axis,
        },
        outputs=outputs_dict,
    )
    return (out, residual_out) if residual is not None else out


def mmha_wrapper(
    x,
    cache_kv=None,
    src_mask=None,
    cum_offsets=None,
    sequence_lengths=None,
    rotary_tensor=None,
    beam_cache_offset=None,
    qkv_out_scale=None,
    out_linear_shift=None,
    out_linear_smooth=None,
    seq_len=1,
    rotary_emb_dims=0,
    use_neox_rotary_style=False,
    out_linear_in_scale=-1,
    quant_round_type=1,
    quant_max_bound=127.0,
    quant_min_bound=-127.0,
):
    if in_dynamic_mode():
        return paddle._C_ops.masked_multihead_attention_(
            x,
            cache_kv,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_linear_shift,
            out_linear_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_linear_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )

    helper = LayerHelper("masked_multihead_attention", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    inputs["cache_kv"] = cache_kv
    if src_mask is not None:
        inputs["src_mask"] = src_mask
    if cum_offsets is not None:
        inputs["cum_offsets"] = cum_offsets
    if sequence_lengths is not None:
        inputs["sequence_lengths"] = sequence_lengths
    if rotary_tensor is not None:
        inputs["rotary_tensor"] = rotary_tensor
    beam_cache_offset_flag = False
    if beam_cache_offset is not None:
        inputs["beam_cache_offset"] = beam_cache_offset
        beam_cache_offset_flag = True
    else:
        beam_cache_offset = helper.create_variable_for_type_inference(dtype="int")
    if qkv_out_scale is not None:
        inputs["qkv_out_scale"] = qkv_out_scale
    if out_linear_shift is not None:
        inputs["out_linear_shift"] = out_linear_shift
    if out_linear_smooth is not None:
        inputs["out_linear_smooth"] = out_linear_smooth

    outputs = {
        "out": out,
        "cache_kv_out": cache_kv,
        "beam_cache_offset_out": beam_cache_offset,
    }
    helper.append_op(
        type="masked_multihead_attention",
        inputs=inputs,
        outputs=outputs,
        attrs={
            "seq_len": seq_len,
            "rotary_emb_dims": rotary_emb_dims,
            "use_neox_rotary_style": use_neox_rotary_style,
            "out_linear_in_scale": out_linear_in_scale,
            "quant_round_type": quant_round_type,
            "quant_max_bound": quant_max_bound,
            "quant_min_bound": quant_min_bound,
        },
    )
    return (out, cache_kv, beam_cache_offset) if beam_cache_offset_flag is not None else (out, cache_kv)


class FusedMultiTransformer(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        norm_type="layernorm",
        use_neox_rotary_style=False,
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_bias_attrs=None,
        epsilon=1e-5,
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
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id
        self.nranks = nranks
        self.norm_type = norm_type
        self.use_neox_rotary_style = use_neox_rotary_style

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

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        def _add_parameter(param):
            assert param.name not in self._parameters
            self._parameters[param.name] = param

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
                # dtype='float32',
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr,
                shape=[embed_dim],
                is_bias=True,
            )
            qkv_weight = self.create_parameter(
                shape=[3 * num_heads * self.head_dim, embed_dim]
                if trans_qkvw
                else [3 * num_heads * self.head_dim, embed_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_bias = self.create_parameter(
                shape=[3 * num_heads * self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
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
                # dtype='float32',
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_bias_attr,
                is_bias=True,
            )
            ffn1_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward * 2],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn1_bias = self.create_parameter(
                shape=[dim_feedforward * 2],
                attr=ffn1_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            ffn2_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
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
        # pre_caches=None,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
    ):
        # print("Time step is: ", time_step)
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
        for i in range(len(caches)):
            # layernorm
            if i == 0:
                ln_out = norm_helper(
                    src, None, None, self.ln_scales[i], None, self._epsilon, 1.0, self.norm_type, begin_norm_axis=1
                )
            # qkv compute
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], transpose_y=True)

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
                encode_rotary_qk(
                    q_out,
                    k_out,
                    rotary_embs,
                    seq_lens,
                    rotary_emb_dims=rotary_emb_dims,
                    use_neox=self.use_neox_rotary_style,
                )
                # write cache kv (inplace)
                write_cache_kv(k_out, v_out, caches[i], seq_lens)

                # cutlass fmha
                qktv_out = paddle.incubate.nn.functional.variable_length_memory_efficient_attention(
                    q_out, k_out, v_out, seq_lens, seq_lens, mask=attn_mask, scale=float(self.head_dim**-0.5)
                )

                fmha_out = transpose_remove_padding(qktv_out, seq_lens, padding_offset)
            else:
                fmha_out = mmha_wrapper(
                    x=qkv_out,
                    cache_kv=caches[i],
                    src_mask=attn_mask,
                    sequence_lengths=seq_lens,
                    rotary_tensor=rotary_embs,
                    rotary_emb_dims=rotary_emb_dims,
                )[0]
            # out_linear
            out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i])

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)

            # norm + residual_add_bias
            tmp_out, bias_residual_input = norm_helper(
                out_linear_out,
                bias_residual_input,
                None,
                self.ffn_ln_scales[i],
                None,
                self._epsilon,
                1.0,
                self.norm_type,
                1,
            )

            # ffn1 matmul
            ffn1_out = paddle.matmul(tmp_out, self.ffn1_weights[i])
            ffn1_out = fused_act_bias_wrapper(ffn1_out, None, act_method=self.activation)
            # ffn2 matmul
            ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i])
            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias
            if i != len(caches) - 1:
                tmp_out, bias_residual_input = norm_helper(
                    ffn2_out,
                    bias_residual_input,
                    None,
                    self.ln_scales[i + 1],
                    None,
                    self._epsilon,
                    1.0,
                    self.norm_type,
                    1,
                )
            else:
                tmp_out, bias_residual_input = norm_helper(
                    ffn2_out, bias_residual_input, None, None, None, self._epsilon, 1.0, self.norm_type, 1
                )

            ln_out = tmp_out

        if time_step is None:
            out = rebuild_padding(tmp_out, cum_offsets, seq_lens, input_ids)
        else:
            out = tmp_out
        return out, caches
