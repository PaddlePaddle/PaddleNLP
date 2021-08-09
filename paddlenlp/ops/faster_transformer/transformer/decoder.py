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
import paddle.nn as nn
from paddle.fluid.layer_helper import LayerHelper

from paddlenlp.ops import transfer_param


def infer_transformer_decoder(
        from_tensor, memory_tensor, mem_seq_len, self_ln_weight, self_ln_bias,
        self_q_weight, self_q_bias, self_k_weight, self_k_bias, self_v_weight,
        self_v_bias, self_out_weight, self_out_bias, cross_ln_weight,
        cross_ln_bias, cross_q_weight, cross_q_bias, cross_k_weight,
        cross_k_bias, cross_v_weight, cross_v_bias, cross_out_weight,
        cross_out_bias, ffn_ln_weight, ffn_ln_bias, ffn_inter_weight,
        ffn_inter_bias, ffn_out_weight, ffn_out_bias, old_self_cache,
        old_mem_cache, n_head, size_per_head):
    helper = LayerHelper('fusion_decoder', **locals())

    inputs = {
        "FromTensor": from_tensor,
        "MemoryTensor": memory_tensor,
        "MemSeqLen": mem_seq_len,
        "SelfLayernormWeight": self_ln_weight,
        "SelfLayernormBias": self_ln_bias,
        "SelfQueryWeight": self_q_weight,
        "SelfQueryBias": self_q_bias,
        "SelfKeyWeight": self_k_weight,
        "SelfKeyBias": self_k_bias,
        "SelfValueWeight": self_v_weight,
        "SelfValueBias": self_v_bias,
        "SelfOutWeight": self_out_weight,
        "SelfOutBias": self_out_bias,
        "CrossLayernormWeight": cross_ln_weight,
        "CrossLayernormBias": cross_ln_bias,
        "CrossQueryWeight": cross_q_weight,
        "CrossQueryBias": cross_q_bias,
        "CrossKeyWeight": cross_k_weight,
        "CrossKeyBias": cross_k_bias,
        "CrossValueWeight": cross_v_weight,
        "CrossValueBias": cross_v_bias,
        "CrossOutWeight": cross_out_weight,
        "CrossOutBias": cross_out_bias,
        "FFNLayernormWeight": ffn_ln_weight,
        "FFNLayernormBias": ffn_ln_bias,
        "FFNInterWeight": ffn_inter_weight,
        "FFNInterBias": ffn_inter_bias,
        "FFNOutWeight": ffn_out_weight,
        "FFNOutBias": ffn_out_bias,
        "OldSelfCache": old_self_cache,
        "OldMemCache": old_mem_cache
    }

    attrs = {'n_head': n_head, 'size_per_head': size_per_head}

    decoder_output = helper.create_variable(dtype=memory_tensor.dtype)
    new_self_cache = helper.create_variable(dtype=memory_tensor.dtype)
    new_mem_cache = helper.create_variable(dtype=memory_tensor.dtype)

    outputs = {
        'DecoderOutput': decoder_output,
        'NewSelfCache': new_self_cache,
        'NewMemCache': new_mem_cache
    }

    helper.append_op(
        type='fusion_decoder', inputs=inputs, outputs=outputs, attrs=attrs)

    return decoder_output, new_self_cache, new_mem_cache


class InferTransformerDecoder(nn.Layer):
    def __init__(self,
                 decoder,
                 n_head,
                 size_per_head,
                 decoder_lib=None,
                 use_fp16_decoder=False):

        if decoder_lib is not None and os.path.isfile(decoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                decoder_lib)

        super(InferTransformerDecoder, self).__init__()
        self.n_head = n_head
        self.size_per_head = size_per_head

        if use_fp16_decoder:
            for idx, mod in enumerate(decoder.layers):
                mod.norm1.weight = transfer_param(mod.norm1.weight)
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True)
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

                mod.norm2.weight = transfer_param(mod.norm2.weight)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True)
                mod.cross_attn.q_proj.weight = transfer_param(
                    mod.cross_attn.q_proj.weight)
                mod.cross_attn.q_proj.bias = transfer_param(
                    mod.cross_attn.q_proj.bias, is_bias=True)
                mod.cross_attn.k_proj.weight = transfer_param(
                    mod.cross_attn.k_proj.weight)
                mod.cross_attn.k_proj.bias = transfer_param(
                    mod.cross_attn.k_proj.bias, is_bias=True)
                mod.cross_attn.v_proj.weight = transfer_param(
                    mod.cross_attn.v_proj.weight)
                mod.cross_attn.v_proj.bias = transfer_param(
                    mod.cross_attn.v_proj.bias, is_bias=True)
                mod.cross_attn.out_proj.weight = transfer_param(
                    mod.cross_attn.out_proj.weight)
                mod.cross_attn.out_proj.bias = transfer_param(
                    mod.cross_attn.out_proj.bias, is_bias=True)

                mod.norm3.weight = transfer_param(mod.norm3.weight)
                mod.norm3.bias = transfer_param(mod.norm3.bias, is_bias=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True)

        self.weights = []
        for idx, mod in enumerate(decoder.layers):
            layer_weight = []
            layer_weight.append(mod.norm1.weight)
            layer_weight.append(mod.norm1.bias)
            layer_weight.append(mod.self_attn.q_proj.weight)
            layer_weight.append(mod.self_attn.q_proj.bias)
            layer_weight.append(mod.self_attn.k_proj.weight)
            layer_weight.append(mod.self_attn.k_proj.bias)
            layer_weight.append(mod.self_attn.v_proj.weight)
            layer_weight.append(mod.self_attn.v_proj.bias)
            layer_weight.append(mod.self_attn.out_proj.weight)
            layer_weight.append(mod.self_attn.out_proj.bias)
            layer_weight.append(mod.norm2.weight)
            layer_weight.append(mod.norm2.bias)
            layer_weight.append(mod.cross_attn.q_proj.weight)
            layer_weight.append(mod.cross_attn.q_proj.bias)
            layer_weight.append(mod.cross_attn.k_proj.weight)
            layer_weight.append(mod.cross_attn.k_proj.bias)
            layer_weight.append(mod.cross_attn.v_proj.weight)
            layer_weight.append(mod.cross_attn.v_proj.bias)
            layer_weight.append(mod.cross_attn.out_proj.weight)
            layer_weight.append(mod.cross_attn.out_proj.bias)
            layer_weight.append(mod.norm3.weight)
            layer_weight.append(mod.norm3.bias)
            layer_weight.append(mod.linear1.weight)
            layer_weight.append(mod.linear1.bias)
            layer_weight.append(mod.linear2.weight)
            layer_weight.append(mod.linear2.bias)
            self.weights.append(layer_weight)

    def forward(self, from_tensor, memory_tensor, mem_seq_len, self_cache,
                mem_cache):
        decoder_output = from_tensor
        self_caches = []
        mem_caches = []
        self_cache = paddle.concat(
            [
                self_cache, paddle.zeros(
                    shape=[
                        len(self.weights), 2, 1, paddle.shape(memory_tensor)[0],
                        self.n_head * self.size_per_head
                    ],
                    dtype=self_cache.dtype)
            ],
            axis=2)
        for idx in range(len(self.weights)):
            weight = self.weights[idx]
            decoder_output, new_self_cache, new_mem_cache = infer_transformer_decoder(
                from_tensor=decoder_output,
                memory_tensor=memory_tensor,
                mem_seq_len=mem_seq_len,
                self_ln_weight=weight[0],
                self_ln_bias=weight[1],
                self_q_weight=weight[2],
                self_q_bias=weight[3],
                self_k_weight=weight[4],
                self_k_bias=weight[5],
                self_v_weight=weight[6],
                self_v_bias=weight[7],
                self_out_weight=weight[8],
                self_out_bias=weight[9],
                cross_ln_weight=weight[10],
                cross_ln_bias=weight[11],
                cross_q_weight=weight[12],
                cross_q_bias=weight[13],
                cross_k_weight=weight[14],
                cross_k_bias=weight[15],
                cross_v_weight=weight[16],
                cross_v_bias=weight[17],
                cross_out_weight=weight[18],
                cross_out_bias=weight[19],
                ffn_ln_weight=weight[20],
                ffn_ln_bias=weight[21],
                ffn_inter_weight=weight[22],
                ffn_inter_bias=weight[23],
                ffn_out_weight=weight[24],
                ffn_out_bias=weight[25],
                old_self_cache=self_cache[idx],
                old_mem_cache=mem_cache[idx],
                n_head=self.n_head,
                size_per_head=self.size_per_head)
            self_caches.append(new_self_cache)
            mem_caches.append(new_mem_cache)

        self_cache = paddle.stack(self_caches, axis=0)
        mem_cache = paddle.stack(mem_caches, axis=0)
        return decoder_output, self_cache, mem_cache
