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
from paddle.nn.layer.transformer import _convert_attention_mask, _convert_param_attr_to_list
from paddlenlp.ops import transfer_param


def infer_transformer_encoder(input,
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
                              sequence_id_offset,
                              trt_seqlen_offset,
                              amax_list,
                              n_head,
                              size_per_head,
                              n_layer=12,
                              is_gelu=True,
                              remove_padding=False,
                              int8_mode=0,
                              layer_idx=0,
                              allow_gemm_test=False,
                              use_trt_kernel=False):
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
        "SequenceIdOffset": sequence_id_offset,
        "TRTSeqLenOffset": trt_seqlen_offset,
        'AmaxList': amax_list
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
    }
    encoder_out = helper.create_variable(dtype=input.dtype)
    outputs = {"EncoderOut": encoder_out}

    helper.append_op(
        type='fusion_encoder', inputs=inputs, outputs=outputs, attrs=attrs)
    return encoder_out


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):  #,
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()

        assert d_model > 0, ("Expected d_model to be greater than 0, "
                             "but recieved {}".format(d_model))
        assert nhead > 0, ("Expected nhead to be greater than 0, "
                           "but recieved {}".format(nhead))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but recieved {}".format(dim_feedforward))

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = paddle.nn.MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = paddle.nn.Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = paddle.nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = paddle.nn.Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.norm2 = paddle.nn.LayerNorm(d_model)
        self.dropout1 = paddle.nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = paddle.nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(paddle.nn.functional, activation)

    def forward(self,
                src,
                src_mask,
                cache=None,
                sequence_id_offset=None,
                mem_seq_lens=None,
                trt_seq_len=None):
        if cache is not None:
            raise NotImplementedError(
                "remove padding/rebuild padding is not supported now")
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
            sequence_id_offset=sequence_id_offset,
            trt_seqlen_offset=trt_seq_len,
            amax_list=paddle.to_tensor([]),  # int8 mode is not supported.
            n_head=self._config['nhead'],
            size_per_head=self._config['d_model'] // self._config['nhead'],
            is_gelu=self._config['activation'] == 'gelu')
        return src


class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None, encoder_lib=None):
        if encoder_lib is not None and os.path.isfile(encoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                encoder_lib)
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = paddle.nn.LayerList([(
            encoder_layer
            if i == 0 else type(encoder_layer)(**encoder_layer._config))
                                           for i in range(num_layers)])
        self.norm = norm

    def forward(self, src, src_mask=None, cache=None):
        def call_once_init():
            if src.dtype == 'float16':
                for mod in self.layers:
                    mod.self_attn.q_proj.weight = transfer_param(
                        mod.self_attn.q_proj.weight)
                    mod.self_attn.q_proj.bias = transfer_param(
                        mod.self_attn.q_proj.bias, is_bias=True)
                    mod.self_attn.k_proj.weight = transfer_param(
                        mod.self_attn.k_proj.weight)
                    mod.self_attn.k_proj.bias = transfer_param(
                        mod.self_attn.k_proj.bias, is_bias=True)
                    mod.self_attn.v_proj.weight = transfer_param(
                        mod.self_attn.v_proj.weight)
                    mod.self_attn.v_proj.bias = transfer_param(
                        mod.self_attn.v_proj.bias, is_bias=True)
                    mod.self_attn.out_proj.weight = transfer_param(
                        mod.self_attn.out_proj.weight)
                    mod.self_attn.out_proj.bias = transfer_param(
                        mod.self_attn.out_proj.bias, is_bias=True)
                    mod.norm1.weight = transfer_param(mod.norm1.weight)
                    mod.norm1.bias = transfer_param(
                        mod.norm1.bias, is_bias=True)
                    mod.norm2.weight = transfer_param(mod.norm2.weight)
                    mod.norm2.bias = transfer_param(
                        mod.norm2.bias, is_bias=True)
                    mod.linear1.weight = transfer_param(mod.linear1.weight)
                    mod.linear1.bias = transfer_param(
                        mod.linear1.bias, is_bias=True)
                    mod.linear2.weight = transfer_param(mod.linear2.weight)
                    mod.linear2.bias = transfer_param(
                        mod.linear2.bias, is_bias=True)
            self.is_prepared = 0

        if not hasattr(self, 'is_prepared'):
            call_once_init()
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        mem_seq_lens = paddle.cast(
            paddle.squeeze(
                paddle.sum(src_mask, axis=[3], dtype="int32"), axis=[1, 2]),
            dtype="int32")

        sequence_id_offset = paddle.to_tensor([], dtype="int32")
        batch_size, max_seq_len = src.shape[:2]
        padding_offset = paddle.arange(
            0, batch_size * max_seq_len, max_seq_len, dtype="int32")
        squence_offset_with_padding = mem_seq_lens + padding_offset
        c = paddle.concat([padding_offset, squence_offset_with_padding], axis=0)
        c_r = paddle.reshape(c, [2, -1])
        t = paddle.transpose(c_r, [1, 0])
        trt_seq_len = paddle.reshape(t, [-1])
        trt_seq_len = paddle.concat(
            [
                trt_seq_len, paddle.to_tensor(
                    [batch_size * max_seq_len], dtype=trt_seq_len.dtype)
            ],
            axis=0)
        # broadcast
        src_mask = paddle.concat(x=[src_mask] * max_seq_len, axis=2)
        output = src

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_mask,
                sequence_id_offset=sequence_id_offset,
                mem_seq_lens=mem_seq_lens,
                trt_seq_len=trt_seq_len)

        if self.norm is not None:
            output = self.norm(output)
        return output
