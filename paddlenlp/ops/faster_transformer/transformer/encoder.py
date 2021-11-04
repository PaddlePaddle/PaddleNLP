# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os

import paddle
from paddle.fluid.layer_helper import LayerHelper
import paddle.nn as nn
from paddle.nn import TransformerEncoder, TransformerEncoderLayer

from paddlenlp.utils.log import logger
from paddlenlp.ops.ext_utils import load


def infer_transformer_encoder(
        input,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        attn_out_weight,
        attn_out_bias,
        attn_mask,
        attn_ln_weight,
        attn_ln_bias,
        out_ln_weight,
        out_ln_bias,
        ffn_inter_weight,
        ffn_inter_bias,
        ffn_out_weight,
        ffn_out_bias,
        #   sequence_id_offset,
        #   trt_seqlen_offset,
        #   amax_list,
        n_head,
        size_per_head,
        n_layer=12,
        is_gelu=True,
        remove_padding=False,
        int8_mode=0,
        layer_idx=0,
        allow_gemm_test=False,
        use_trt_kernel=False,
        normalize_before=False):
    """
    Fusion Encoder API intergrating Encoder inference in FasterTransformer. It
    accepts the weight and bias of TransformerEncoder and some other parameters
    for inference.
    """
    helper = LayerHelper('fusion_encoder', **locals())
    inputs = {
        'Input': input,
        'SelfQueryWeight': q_weight,
        'SelfQueryBias': q_bias,
        'SelfKeyWeight': k_weight,
        'SelfKeyBias': k_bias,
        'SelfValueWeight': v_weight,
        'SelfValueBias': v_bias,
        'SelfAttnOutputWeight': attn_out_weight,
        'SelfAttnOutputBias': attn_out_bias,
        "SelfAttnMask": attn_mask,
        'SelfAttnOutputLayernormWeight': attn_ln_weight,
        'SelfAttnOutputLayernormBias': attn_ln_bias,
        'OutputLayernormWeight': out_ln_weight,
        'OutputLayernormBias': out_ln_bias,
        'FFNInterWeight': ffn_inter_weight,
        'FFNInterBias': ffn_inter_bias,
        'FFNOutputWeight': ffn_out_weight,
        'FFNOutputBias': ffn_out_bias,
        # "SequenceIdOffset": sequence_id_offset,
        # "TRTSeqLenOffset": trt_seqlen_offset,
        # 'AmaxList': amax_list
    }
    attrs = {
        'head_num': n_head,
        'size_per_head': size_per_head,
        'is_gelu': is_gelu,
        "remove_padding": remove_padding,
        'int8_mode': int8_mode,
        'num_layer': n_layer,
        'layer_idx': layer_idx,
        'allow_gemm_test': allow_gemm_test,
        'use_trt_kernel': use_trt_kernel,
        'normalize_before': normalize_before
    }
    encoder_out = helper.create_variable(dtype=input.dtype)
    outputs = {"EncoderOut": encoder_out}

    helper.append_op(
        type='fusion_encoder', inputs=inputs, outputs=outputs, attrs=attrs)
    return encoder_out


def encoder_layer_forward(self,
                          src,
                          src_mask,
                          cache=None,
                          sequence_id_offset=None,
                          trt_seq_len=None):
    """
    Redefines `forward` function of `paddle.nn.TransformerEncoderLayer` for
    integrating FasterTransformer for inference.

    The original `forward` function would not be replaced unless
    `enable_faster_encoder` is called by objects of its base class. After
    replacing, objects of `paddle.nn.TransformerEncoderLayer` also have the
    same member variables as before.

    After inference, `disable_faster_encoder` could be called to restore the
    `forward` function of `paddle.nn.TransformerEncoder` and
    `paddle.nn.TransformerEncoderLayer`.

    Args:
        src (Tensor):
            The input of Transformer encoder layer. It is a tensor with shape
            `[batch_size, sequence_length, d_model]`. The data type should be
            float32 or float64.
        src_mask (Tensor, optional):
            A tensor used in multi-head attention to prevents attention to some
            unwanted positions, usually the paddings or the subsequent
            positions. It is a tensor with shape `[batch_size, 1, 1, sequence_length]`.
            When the data type is bool, the unwanted positions have `False`
            values and the others have `True` values. When the data type is int,
            the unwanted positions have 0 values and the others have 1 values.
            When the data type is float, the unwanted positions have `-INF`
            values and the others have 0 values. It can be None when nothing
            wanted or needed to be prevented attention to. Defaults to None.

    Returns:
        src(Tensor|tuple):
            It is a tensor that has the same shape and data type as `enc_input`,
            representing the output of Transformer encoder layer. Or a tuple if
            `cache` is not None, except for encoder layer output, the tuple
            includes the new cache which is same as input `cache` argument but
            `incremental_cache` has an incremental length. See
            `paddle.nn.MultiHeadAttention.gen_cache` and
            `paddle.nn.MultiHeadAttention.forward` for more details.
    """
    if cache is not None:
        raise NotImplementedError("cache in encoder is not supported now")
    src = infer_transformer_encoder(
        input=src,
        q_weight=self.self_attn.q_proj.weight,
        q_bias=self.self_attn.q_proj.bias,
        k_weight=self.self_attn.k_proj.weight,
        k_bias=self.self_attn.k_proj.bias,
        v_weight=self.self_attn.v_proj.weight,
        v_bias=self.self_attn.v_proj.bias,
        attn_out_weight=self.self_attn.out_proj.weight,
        attn_out_bias=self.self_attn.out_proj.bias,
        attn_mask=src_mask,
        attn_ln_weight=self.norm1.weight,
        attn_ln_bias=self.norm1.bias,
        out_ln_weight=self.norm2.weight,
        out_ln_bias=self.norm2.bias,
        ffn_inter_weight=self.linear1.weight,
        ffn_inter_bias=self.linear1.bias,
        ffn_out_weight=self.linear2.weight,
        ffn_out_bias=self.linear2.bias,
        # sequence_id_offset=paddle.to_tensor([]),
        # trt_seqlen_offset=paddle.to_tensor([]),
        # amax_list=paddle.to_tensor([]),  # int8 mode is not supported.
        n_head=self._config['nhead'],
        size_per_head=self._config['d_model'] // self._config['nhead'],
        is_gelu=self._config['activation'] == 'gelu',
        normalize_before=self._config['normalize_before'] == True)
    return src


def encoder_forward(self, src, src_mask=None, cache=None):
    """
    Redefines `forward` function of `paddle.nn.TransformerEncoder` for
    integrating FasterTransformer for inference.

    The original `forward` function would not be replaced unless
    `enable_faster_encoder` is called by objects of its base class. After
    replacing, objects of `paddle.nn.TransformerEncoder` also have the same
    member variables as before.

    After inference, `disable_faster_encoder` could be called to restore the
    `forward` function of `paddle.nn.TransformerEncoder` and
    `paddle.nn.TransformerEncoderLayer`.

    Args:
        src (Tensor):
            The input of Transformer encoder. It is a tensor
            with shape `[batch_size, sequence_length, d_model]`. The data
            type should be float32 or float64.
        src_mask (Tensor, optional):
            A tensor used in multi-head attention to prevents attention to
            some unwanted positions, usually the paddings or the subsequent
            positions. It is a tensor with shape `[batch_size, 1, 1, sequence_length]`.
            When the data type is bool, the unwanted positions have `False`
            values and the others have `True` values. When the data type is
            int, the unwanted positions have 0 values and the others have 1
            values. When the data type is float, the unwanted positions have
            `-INF` values and the others have 0 values. It can be None when
            nothing wanted or needed to be prevented attention to. Defaults
            to None.

    Returns:
        output (Tensor|tuple):
            It is a tensor that has the same shape and data type as `src`,
            representing the output of Transformer encoder. Or a tuple if
            `cache` is not None, except for encoder output, the tuple includes
            the new cache which is same as input `cache` argument but
            `incremental_cache` in it has an incremental length. See
            `paddle.nn.MultiHeadAttention.gen_cache` and
            `paddle.nn.MultiHeadAttention.forward` for more details.
    """

    max_seq_len = src.shape[1]
    # broadcast
    src_mask = paddle.concat(x=[src_mask] * max_seq_len, axis=2)
    output = src
    for i, layer in enumerate(self.layers):
        output = layer(output, src_mask)
    if self.norm is not None:
        output = self.norm(output)
    return output


def enable_faster_encoder(self):
    """
    Compiles fusion encoder operator intergrated FasterTransformer using the
    method of JIT(Just-In-Time) and replaces the `forward` function of
    `paddle.nn.TransformerEncoder` and `paddle.nn.TransformerEncoderLayer`
    objects inherited from `self` to support inference using FasterTransformer.

    Examples:

        .. code-block:: python

            from paddlenlp.ops import enable_faster_encoder, disable_faster_encoder

            model.eval()
            model = enable_faster_encoder(model)
            enc_out = model(src, src_mask)
            model = disable_faster_encoder(model)
    """

    def init_func(layer):
        if isinstance(layer, TransformerEncoderLayer):
            is_usable = True
            if layer._config['bias_attr'] == False:
                logger.warning("`False` for paddle.nn.TransformerEncoder's" \
                               " parameter `bias_attr` is not supported in " \
                               "FasterTransformer by now. The original forward" \
                               " will be involved.")
                is_usable = False
            if layer._config['activation'] not in ('relu', 'gelu'):
                logger.warning("Only 'relu' or 'gelu' is supported by now. " \
                                "The original forward will be involved.")
                is_usable = False
            if is_usable:
                layer.forward = layer._ft_forward
        elif isinstance(layer, TransformerEncoder):
            layer.forward = layer._ft_forward

    if not self.training:
        try:
            load("FasterTransformer", verbose=True)
        except Exception:
            logger.warning(
                "Exception occurs when using FasterTransformer. " \
                "The original forward will be involved. ")
            return self
        for layer in self.children():
            layer.apply(init_func)
    return self


def disable_faster_encoder(self):
    """
    Restores the original `forward` function of `paddle.nn.TransformerEncoder`
    and `paddle.nn.TransformerEncoderLayer` objects inherited from `self`.

    Examples:

        .. code-block:: python

            from paddlenlp.ops import enable_faster_encoder, disable_faster_encoder

            model.eval()
            model = enable_faster_encoder(model)
            enc_out = model(src, src_mask)
            model = disable_faster_encoder(model)
    """

    def init_func(layer):
        if isinstance(layer, (TransformerEncoderLayer, TransformerEncoder)):
            layer.forward = layer._ori_forward

    for layer in self.children():
        layer.apply(init_func)
    return self
