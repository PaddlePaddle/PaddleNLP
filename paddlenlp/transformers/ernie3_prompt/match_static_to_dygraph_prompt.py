# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
import pickle
import paddle
import numpy as np


def match_embedding_param(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "embeddings.word_embeddings.weight"] = "word_embedding_expanded"
    convert_parameter_name_dict[
        "embeddings.position_embeddings.weight"] = "pos_embedding"
    convert_parameter_name_dict[
        "embeddings.position_extra_embeddings.weight"] = "pos_embedding_2d_extra"
    return convert_parameter_name_dict


def match_encoder_param(convert_parameter_name_dict, layer_num=6):
    dygraph_encoder_prefix_names = ["encoder", ]
    static_encoder_prefix_name = "encoder_layer"
    dygraph_proj_names = ["q", "k", "v", "out"]
    static_proj_names = ["query", "key", "value", "output"]
    dygraph_param_names = ["weight", "bias"]
    static_param_names = ["w", "b"]
    dygraph_layer_norm_param_names = ["weight", "bias"]
    static_layer_norm_param_names = ["scale", "bias"]

    # Firstly, converts the multihead_attention to the parameter.
    dygraph_format_name = "{}.layers.{}.self_attn.{}_proj.{}"
    static_format_name = "{}_{}_multi_head_att_{}_fc.{}_0"
    for dygraph_encoder_prefix_name in dygraph_encoder_prefix_names:
        for i in range(0, layer_num):
            for dygraph_proj_name, static_proj_name in zip(dygraph_proj_names,
                                                           static_proj_names):
                for dygraph_param_name, static_param_name in zip(
                        dygraph_param_names, static_param_names):
                    convert_parameter_name_dict[dygraph_format_name.format(dygraph_encoder_prefix_name, i, dygraph_proj_name, dygraph_param_name)] = \
                        static_format_name.format(static_encoder_prefix_name, i, static_proj_name, static_param_name)

    # Secondly, converts the encoder ffn parameter.     
    dygraph_ffn_linear_format_name = "{}.layers.{}.linear{}.{}"
    static_ffn_linear_format_name = "{}_{}_ffn_fc_{}.{}_0"
    for dygraph_encoder_prefix_name in dygraph_encoder_prefix_names:
        for i in range(0, layer_num):
            for j in range(0, 2):
                for dygraph_param_name, static_param_name in zip(
                        dygraph_param_names, static_param_names):
                    convert_parameter_name_dict[dygraph_ffn_linear_format_name.format(dygraph_encoder_prefix_name, i, j+1,  dygraph_param_name)] = \
                    static_ffn_linear_format_name.format(static_encoder_prefix_name, i, j, static_param_name)

    # Thirdly, converts the multi_head layer_norm parameter.
    dygraph_encoder_attention_layer_norm_format_name = "{}.layers.{}.norm1.{}"
    static_encoder_attention_layer_norm_format_name = "{}_{}_post_att_layer_norm_{}"
    for dygraph_encoder_prefix_name in dygraph_encoder_prefix_names:
        for i in range(0, layer_num):
            for dygraph_param_name, static_pararm_name in zip(
                    dygraph_layer_norm_param_names,
                    static_layer_norm_param_names):
                convert_parameter_name_dict[dygraph_encoder_attention_layer_norm_format_name.format(dygraph_encoder_prefix_name, i, dygraph_param_name)] = \
                    static_encoder_attention_layer_norm_format_name.format(static_encoder_prefix_name, i if static_encoder_prefix_name=='encoder_layer' else i+48, static_pararm_name)

    dygraph_encoder_ffn_layer_norm_format_name = "{}.layers.{}.norm2.{}"
    static_encoder_ffn_layer_norm_format_name = "{}_{}_post_ffn_layer_norm_{}"
    for dygraph_encoder_prefix_name in dygraph_encoder_prefix_names:
        for i in range(0, layer_num):
            for dygraph_param_name, static_pararm_name in zip(
                    dygraph_layer_norm_param_names,
                    static_layer_norm_param_names):
                convert_parameter_name_dict[dygraph_encoder_ffn_layer_norm_format_name.format(dygraph_encoder_prefix_name, i, dygraph_param_name)] = \
                    static_encoder_ffn_layer_norm_format_name.format(static_encoder_prefix_name, i, static_pararm_name)

    convert_parameter_name_dict[
        "layer_norm.weight"] = "pre_encoder_layer_norm_scale"
    convert_parameter_name_dict[
        "layer_norm.bias"] = "pre_encoder_layer_norm_bias"
    return convert_parameter_name_dict


def match_mlm_parameter(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "lm_head.lm_transform.weight"] = "server_nlg_mask_lm_trans_fc.w_0"
    convert_parameter_name_dict[
        "lm_head.lm_transform.bias"] = "server_nlg_mask_lm_trans_fc.b_0"
    convert_parameter_name_dict[
        "lm_head.layer_norm.weight"] = "server_nlg_mask_lm_trans_layer_norm_scale"
    convert_parameter_name_dict[
        "lm_head.layer_norm.bias"] = "server_nlg_mask_lm_trans_layer_norm_bias"
    convert_parameter_name_dict[
        "lm_head.lm_out.weight"] = "word_embedding_expanded"
    convert_parameter_name_dict[
        "lm_head.lm_out.bias"] = "server_nlg_mask_lm_out_fc.b_0"
    return convert_parameter_name_dict


def convert_static_to_dygraph_params(dygraph_params_save_path,
                                     static_params_dir,
                                     static_to_dygraph_param_name,
                                     vocab_size_output=46256,
                                     model_name='ernie3'):
    files = os.listdir(static_params_dir)
    state_dict = {}
    for dygraph_para_name, static_para_name in static_to_dygraph_param_name.items(
    ):
        if static_para_name not in files:
            print(static_para_name, "not in files")
            continue
        path = os.path.join(static_params_dir, static_para_name)
        value = paddle.load(path).numpy()
        print('vshape', value.shape)
        if "lm_head" in dygraph_para_name:
            # Note: lm_head parameters do not need add `model_name.` prefix
            if "lm_head.lm_out.weight" in dygraph_para_name:
                state_dict[dygraph_para_name] = np.transpose(
                    value[:vocab_size_output])
            else:
                state_dict[dygraph_para_name] = value
        else:
            state_dict[model_name + '.' + dygraph_para_name] = value

    with open(dygraph_params_save_path, 'wb') as f:
        pickle.dump(state_dict, f)
    params = paddle.load(dygraph_params_save_path)

    cnt = 0
    f = open('transfer.txt', 'w', encoding='utf-8')
    for name in sorted(state_dict.keys()):
        if name in params:
            assert ((state_dict[name] == params[name].numpy()).all() == True)
            cnt += 1
            f.write(f'{name}--{params[name].shape}\n')
        else:
            print(name, 'not in params')
    f.close()
    print(f'Totally convert {cnt} params.')


if __name__ == "__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, layer_num=6)
    convert_parameter_name_dict = match_mlm_parameter(
        convert_parameter_name_dict)
    for dygraph_name, static_name in convert_parameter_name_dict.items():
        print("{}:-------:{}".format(static_name, dygraph_name))

    convert_static_to_dygraph_params(
        dygraph_params_save_path='ernie3_prompt.pdparams',
        static_params_dir='/home/home/ernie3_prompt/paddlenlp/transformers/ernie3/step_500000',
        static_to_dygraph_param_name=convert_parameter_name_dict,
        model_name='ernie3_prompt')
