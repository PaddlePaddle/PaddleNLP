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
import time

import paddle
from paddle.fluid.layer_helper import LayerHelper
import paddle.nn as nn

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
                              n_layer,
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


class InferTransformerEncoder(nn.Layer):
    def __init__(self,
                 encoder,
                 n_layer,
                 n_head,
                 size_per_head,
                 is_gelu=True,
                 int8_mode=0,
                 allow_gemm_test=False,
                 use_trt_kernel=False,
                 remove_padding=False,
                 encoder_lib=None,
                 use_fp16_encoder=False,
                 place=None):
        if encoder_lib is not None and os.path.isfile(encoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                encoder_lib)
        super(InferTransformerEncoder, self).__init__()
        self.n_head = n_head
        self.int8_mode = int8_mode
        self.allow_gemm_test = allow_gemm_test
        self.use_trt_kernel = use_trt_kernel
        self.n_layer = n_layer
        self.size_per_head = size_per_head
        self.is_gelu = is_gelu
        self.remove_padding = False
        if remove_padding:
            raise NotImplementedError(
                "remove padding/rebuild padding is not supported now")
        if int8_mode:
            raise NotImplementedError("int8 mode is not supported now")
        if use_fp16_encoder:
            for mod in encoder.layers:
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
                mod.norm1.bias = transfer_param(mod.norm1.bias, is_bias=True)
                mod.norm2.weight = transfer_param(mod.norm2.weight)
                mod.norm2.bias = transfer_param(mod.norm2.bias, is_bias=True)
                mod.linear1.weight = transfer_param(mod.linear1.weight)
                mod.linear1.bias = transfer_param(
                    mod.linear1.bias, is_bias=True)
                mod.linear2.weight = transfer_param(mod.linear2.weight)
                mod.linear2.bias = transfer_param(
                    mod.linear2.bias, is_bias=True)
        self.place = place
        self.times = 0.0
        self.batches = 0
        self.weights = []
        for mod in encoder.layers:
            layer_weight = []
            layer_weight.append(mod.self_attn.q_proj.weight)
            layer_weight.append(mod.self_attn.q_proj.bias)
            layer_weight.append(mod.self_attn.k_proj.weight)
            layer_weight.append(mod.self_attn.k_proj.bias)
            layer_weight.append(mod.self_attn.v_proj.weight)
            layer_weight.append(mod.self_attn.v_proj.bias)
            layer_weight.append(mod.self_attn.out_proj.weight)
            layer_weight.append(mod.self_attn.out_proj.bias)
            layer_weight.append(mod.norm1.weight)
            layer_weight.append(mod.norm1.bias)
            layer_weight.append(mod.norm2.weight)
            layer_weight.append(mod.norm2.bias)
            layer_weight.append(mod.linear1.weight)
            layer_weight.append(mod.linear1.bias)
            layer_weight.append(mod.linear2.weight)
            layer_weight.append(mod.linear2.bias)
            # if not self.int8_mode:
            layer_weight.append(paddle.to_tensor([]))

            self.weights.append(layer_weight)

    def forward(self, enc_input, attn_mask, seq_len):
        if self.remove_padding:
            pass
            # sequence_id_offset = paddle.zeros(
            #     (enc_input.shape[0] * seq_len[0]), dtype="int32")
            # trt_seq_len = paddle.cumsum(
            #     paddle.concat(
            #         [paddle.zeros(
            #             (1, ), dtype=seq_len.dtype), seq_len]),
            #     axis=0,
            #     dtype="int32")
        else:
            sequence_id_offset = paddle.to_tensor([], dtype="int32")
            batch_size = enc_input.shape[0]
            max_seq_len = enc_input.shape[1]
            padding_offset = paddle.arange(
                0, batch_size * max_seq_len, max_seq_len, dtype="int32")
            squence_offset_with_padding = seq_len + padding_offset
            c = paddle.concat(
                [padding_offset, squence_offset_with_padding], axis=0)
            c_r = paddle.reshape(c, [2, -1])
            t = paddle.transpose(c_r, [1, 0])
            trt_seq_len = paddle.reshape(t, [-1])
            trt_seq_len = paddle.concat(
                [
                    trt_seq_len, paddle.to_tensor(
                        [batch_size * max_seq_len], dtype=trt_seq_len.dtype)
                ],
                axis=0)
        paddle.fluid.core._cuda_synchronize(self.place)
        time1 = time.time()
        for idx in range(self.n_layer):
            weight = self.weights[idx]
            enc_out = infer_transformer_encoder(
                input=enc_input,
                q_weight=weight[0],
                q_bias=weight[1],
                k_weight=weight[2],
                k_bias=weight[3],
                v_weight=weight[4],
                v_bias=weight[5],
                attn_out_weight=weight[6],
                attn_out_bias=weight[7],
                attn_mask=attn_mask,
                attn_ln_weight=weight[8],
                attn_ln_bias=weight[9],
                out_ln_weight=weight[10],
                out_ln_bias=weight[11],
                ffn_inter_weight=weight[12],
                ffn_inter_bias=weight[13],
                ffn_out_weight=weight[14],
                ffn_out_bias=weight[15],
                sequence_id_offset=sequence_id_offset,
                trt_seqlen_offset=trt_seq_len,
                amax_list=weight[16],
                n_head=self.n_head,
                size_per_head=self.size_per_head,
                n_layer=self.n_layer,
                is_gelu=self.is_gelu,
                remove_padding=self.remove_padding,
                int8_mode=self.int8_mode,
                layer_idx=idx,
                allow_gemm_test=self.allow_gemm_test,
                use_trt_kernel=self.use_trt_kernel)
            enc_input = enc_out
        if self.batches > 10:
            paddle.fluid.core._cuda_synchronize(self.place)
            self.times += time.time() - time1
        self.batches += 1

        return enc_out
