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

import paddle
import paddle.nn as nn
from paddle.fluid.layer_helper import LayerHelper

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
                              attn_ln_weight,
                              attn_ln_bias,
                              out_ln_weight,
                              out_ln_bias,
                              ffn_inter_weight,
                              ffn_inter_bias,
                              ffn_out_weight,
                              ffn_out_bias,
                              amax_list,
                              n_head,
                              size_per_head,
                              n_layer,
                              max_seq_len,
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
        'SelfAttnOutputLayernormWeight': attn_ln_weight,
        'SelfAttnOutputLayernormBias': attn_ln_bias,
        'OutputLayernormWeight': out_ln_weight,
        'OutputLayernormBias': out_ln_bias,
        'FFNInterWeight': ffn_inter_weight,
        'FFNInterBias': ffn_inter_bias,
        'FFNOutputWeight': ffn_out_weight,
        'FFNOutputBias': ffn_out_bias,
        'AmaxList': amax_list
    }
    attrs = {
        'head_num': n_head,
        'size_per_head': size_per_head,
        'int8_mode': int8_mode,
        'num_layer': n_layer,
        'layer_idx': layer_idx,
        'allow_gemm_test': allow_gemm_test,
        'use_trt_kernel': use_trt_kernel,
        'max_seq_len': max_seq_len
    }
    encoder_out = helper.create_variable(dtype="float32")
    outputs = {"transformer_output": encoder_out}

    helper.append_op(
        type='fusion_encoder', inputs=inputs, outputs=outputs, attrs=attrs)

    return encoder_out


class InferTransformerEncoder(nn.Layer):
    def __init__(self,
                 encoder,
                 n_layer,
                 n_head,
                 d_model,
                 max_seq_len=512,
                 int8_mode=0,
                 allow_gemm_test=False,
                 use_trt_kernel=False,
                 encoder_lib=None,
                 use_fp16_encoder=False):
        if decoding_lib is not None and os.path.isfile(encoder_lib):
            # Maybe it has been loadad by `ext_utils.load`
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                encoder_lib)
        super(InferTransformerEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.int8_mode = int8_mode
        self.allow_gemm_test = allow_gemm_test
        self.use_trt_kernel = use_trt_kernel
        self.n_layer = n_layer

        if use_fp16_encoder:
            for idx, mod in enuemerate(encoder.layers):
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
            layer_weight.append(self_attn.out_proj.weight)
            layer_weight.append(self_attn.out_proj.bias)
            layer_weight.append(norm1.weight)
            layer_weight.append(norm1.bias)
            layer_weight.append(norm2.weight)
            layer_weight.append(norm2.bias)
            layer_weight.append(linear1.weight)
            layer_weight.append(linear1.bias)
            layer_weight.append(linear2.weight)
            layer_weight.append(linear2.bias)
            # if not self.int8_mode:
            layer_weight.append(paddle.to_tensor([]))

            sef.weights.append(layer_weight)

    def forward(self):
        for idx in range(self.n_layer):
            weight = self.weights[idx]
            transformer_output = infer_transformer_encoder(
                q_weight=weight[0],
                q_bias=weight[1],
                k_weight=weight[2],
                k_bias=weight[3],
                v_weight=weight[4],
                v_bias=weight[5],
                attn_out_weight=weight[6],
                attn_out_bias=weight[7],
                attn_ln_weight=weight[8],
                attn_ln_bias=weight[9],
                out_ln_weight=weight[10],
                out_ln_bias=weight[11],
                ffn_inter_weight=weight[12],
                ffn_inter_bias=weight[13],
                ffn_out_weight=weight[14],
                ffn_out_bias=weight[15],
                amax_list=weight[16],
                n_head=self.n_head,
                size_per_head=self.size_per_head,
                n_layer=self.n_layer,
                max_seq_len=self.max_seq_len,
                int8_mode=self.int8_mode,
                layer_idx=idx,
                allow_gemm_test=self.allow_gemm_test,
                use_trt_kernel=self.use_trt_kernel)
        return encoder_out
