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
import numpy as np

import paddle
import paddle.nn as nn
from paddle.fluid.layer_helper import LayerHelper

from paddlenlp.ops import transfer_param
from paddlenlp.transformers import WordEmbedding, PositionalEmbedding, position_encoding_init


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
                              remove_padding,
                              n_layer,
                              max_seq_length,
                              int8_mode=0,
                              layer_idx=0,
                              allow_gemm_test=False,
                              use_trt_kernel=False,
                              normalize_before=False):
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
        "remove_padding": remove_padding,
        'int8_mode': int8_mode,
        'num_layer': n_layer,
        'layer_idx': layer_idx,
        'allow_gemm_test': allow_gemm_test,
        'use_trt_kernel': use_trt_kernel,
        'max_seq_len': max_seq_length,
        'normalize_before': normalize_before,
    }
    encoder_out = helper.create_variable(dtype=input.dtype)
    outputs = {"encoder_out": encoder_out}

    helper.append_op(
        type='fusion_encoder', inputs=inputs, outputs=outputs, attrs=attrs)
    return encoder_out


class InferTransformerEncoder(nn.Layer):
    def __init__(self,
                 encoder,
                 n_layer,
                 n_head,
                 size_per_head,
                 max_seq_length=512,
                 int8_mode=0,
                 allow_gemm_test=False,
                 use_trt_kernel=False,
                 remove_padding=False,
                 encoder_lib=None,
                 use_fp16_encoder=False,
                 normalize_before=False):
        if encoder_lib is not None and os.path.isfile(encoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                encoder_lib)
        super(InferTransformerEncoder, self).__init__()
        self.n_head = n_head
        self.max_seq_length = max_seq_length
        self.int8_mode = int8_mode
        self.allow_gemm_test = allow_gemm_test
        self.use_trt_kernel = use_trt_kernel
        self.normalize_before = normalize_before
        self.n_layer = n_layer
        self.size_per_head = size_per_head
        self.remove_padding = remove_padding
        if use_fp16_encoder:
            for idx, mod in enumerate(encoder.layers):
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

        self.weights = []
        for idx, mod in enumerate(encoder.layers):
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
            sequence_id_offset = paddle.zeros(
                (enc_input.shape[0] * seq_len[0]), dtype="int32")
            trt_seq_len = paddle.cumsum(
                paddle.concat(
                    [paddle.zeros(
                        (1, ), dtype=seq_len.dtype), seq_len]),
                axis=0,
                dtype="int32")
        else:
            sequence_id_offset = paddle.to_tensor([], dtype="int32")  # 不需要
            batch_size = enc_input.shape[0]
            max_seq_len = enc_input.shape[1]
            padding_offset = paddle.arange(
                0, batch_size * max_seq_len, max_seq_len, dtype="int32")
            squence_offset_with_padding = seq_len + padding_offset
            c = paddle.concat(
                [padding_offset, squence_offset_with_padding], axis=0)
            c_r = paddle.reshape(c, [2, -1])
            t = paddle.transpose(c_r, [1, 0])  # 一个batch内有效（非padding）区间
            trt_seq_len = paddle.reshape(t, [-1])
            trt_seq_len = paddle.concat(
                [
                    trt_seq_len, paddle.to_tensor(
                        [batch_size * max_seq_len], dtype=trt_seq_len.dtype)
                ],
                axis=0)  # 加上最大的那个index

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
                remove_padding=self.remove_padding,
                n_layer=self.n_layer,
                max_seq_length=self.max_seq_length,
                int8_mode=self.int8_mode,
                layer_idx=idx,
                allow_gemm_test=self.allow_gemm_test,
                use_trt_kernel=self.use_trt_kernel,
                normalize_before=self.normalize_before)
            enc_input = enc_out

        return enc_out


class FasterEncoder(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 max_seq_length,
                 n_layer,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 bos_id=0,
                 int8_mode=0,
                 allow_gemm_test=False,
                 use_trt_kernel=False,
                 remove_padding=False,
                 encoder_lib=None,
                 use_fp16_encoder=False,
                 normalize_before=False):
        super().__init__()
        self.bos_id = bos_id
        self.dropout = dropout
        self.remove_padding = remove_padding
        self.use_fp16_encoder = use_fp16_encoder
        self.max_seq_length = max_seq_length
        self.d_model = d_model

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(
            emb_dim=d_model, max_length=max_seq_length)
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model,
            n_head,
            d_inner_hid,
            dropout,
            activation="gelu",
            attn_dropout=dropout,
            act_dropout=dropout,
            normalize_before=normalize_before)
        custom_encoder = paddle.nn.TransformerEncoder(encoder_layer, n_layer)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            custom_encoder=custom_encoder,
            activation="gelu",
            normalize_before=normalize_before)

        self.encoder = InferTransformerEncoder(
            encoder=self.transformer.encoder,
            n_layer=n_layer,
            n_head=n_head,
            size_per_head=d_model // n_head,
            max_seq_length=max_seq_length,
            int8_mode=int8_mode,
            allow_gemm_test=allow_gemm_test,
            use_trt_kernel=use_trt_kernel,
            remove_padding=remove_padding,
            encoder_lib=encoder_lib,
            use_fp16_encoder=use_fp16_encoder,
            normalize_before=normalize_before)

    def forward(self, enc_input, src_mask, mem_seq_lens):
        self.encoder.eval()
        dtype = "float32"
        if self.use_fp16_encoder:
            dtype = "float16"
            enc_input = paddle.cast(enc_input, dtype)
            src_mask = paddle.cast(src_mask, dtype)
        enc_out = self.encoder(enc_input, src_mask, mem_seq_lens)

        if self.use_fp16_encoder:
            enc_out = paddle.cast(enc_out, "float32")
        return enc_out

    def load(self, init_from_params):
        # Load the trained model
        assert init_from_params, (
            "Please set init_from_params to load the infer model.")
        # To avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict = paddle.load(init_from_params, return_numpy=True)
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            self.max_seq_length, self.d_model)
        if self.use_fp16_encoder:
            for item in self.state_dict():
                if "encoder.layers" in item:
                    model_dict[item] = np.float16(model_dict[item])

        self.load_dict(model_dict)
