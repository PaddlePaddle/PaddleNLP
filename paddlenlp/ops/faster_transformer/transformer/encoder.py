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


def encoder_layer_forward(self,
                          src,
                          src_mask,
                          cache=None,
                          sequence_id_offset=None,
                          mem_seq_lens=None,
                          trt_seq_len=None):
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
        sequence_id_offset=sequence_id_offset,
        trt_seqlen_offset=trt_seq_len,
        amax_list=paddle.to_tensor([]),  # int8 mode is not supported.
        n_head=self._config['nhead'],
        size_per_head=self._config['d_model'] // self._config['nhead'],
        is_gelu=self._config['activation'] == 'gelu')
    return src


def encoder_forward(self, src, src_mask=None, cache=None):
    if src_mask.dtype == paddle.float16:
        src_mask = paddle.cast(src_mask, "float32")
    mem_seq_lens = paddle.cast(
        paddle.squeeze(
            paddle.sum(src_mask, axis=[3]), axis=[1, 2]),
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
    # TODO(Liujiaqi): add this to be generic
    # src_mask = _convert_attention_mask(src_mask, src.dtype)
    # broadcast
    src_mask = paddle.concat(x=[src_mask] * max_seq_len, axis=2)
    output = src
    if src.dtype == paddle.float16:
        src_mask = paddle.cast(src_mask, dtype="float16")
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


def enable_faster_encoder(self):
    def init_func(layer):
        if isinstance(layer, (TransformerEncoderLayer, TransformerEncoder)):
            layer.forward = layer._ft_forward

    if not self.training:
        for layer in self.children():
            layer.apply(init_func)
    return self


def disable_faster_encoder(self):
    def init_func(layer):
        if isinstance(layer, (TransformerEncoderLayer, TransformerEncoder)):
            layer.forward = layer._ori_forward

    for layer in self.children():
        layer.apply(init_func)
    return self
