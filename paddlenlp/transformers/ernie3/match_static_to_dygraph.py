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
        "embeddings.word_embeddings.weight"] = "word_embedding"
    convert_parameter_name_dict[
        "embeddings.position_embeddings.weight"] = "pos_embedding"
    return convert_parameter_name_dict


def match_encoder_param(convert_parameter_name_dict,
                        sharing_layer_num=48,
                        nlg_layer_num=12):
    dygraph_encoder_prefix_names = [
        "sharing_encoder",
        "nlg_encoder",
    ]
    static_encoder_prefix_names = ["encoder_layer", "nlg_encoder_layer"]
    dygraph_proj_names = ["q", "k", "v", "out"]
    static_proj_names = ["query", "key", "value", "output"]
    dygraph_param_names = ["weight", "bias"]
    static_param_names = ["w", "b"]
    dygraph_layer_norm_param_names = ["weight", "bias"]
    static_layer_norm_param_names = ["scale", "bias"]

    # Firstly, converts the multihead_attention to the parameter.
    dygraph_format_name = "{}.layers.{}.self_attn.{}_proj.{}"
    static_format_name = "{}_{}_multi_head_att_{}_fc.{}_0"
    for dygraph_encoder_prefix_name, static_encoder_prefix_name in zip(
            dygraph_encoder_prefix_names, static_encoder_prefix_names):
        for i in range(0, sharing_layer_num
                       if static_encoder_prefix_name == 'encoder_layer' else
                       nlg_layer_num):
            for dygraph_proj_name, static_proj_name in zip(dygraph_proj_names,
                                                           static_proj_names):
                for dygraph_param_name, static_param_name in zip(
                        dygraph_param_names, static_param_names):
                    convert_parameter_name_dict[dygraph_format_name.format(dygraph_encoder_prefix_name, i, dygraph_proj_name, dygraph_param_name)] = \
                        static_format_name.format(static_encoder_prefix_name, i if static_encoder_prefix_name=='encoder_layer' else i+48, static_proj_name, static_param_name)

    # Secondly, converts the encoder ffn parameter.     
    dygraph_ffn_linear_format_name = "{}.layers.{}.linear{}.{}"
    static_ffn_linear_format_name = "{}_{}_ffn_fc_{}.{}_0"
    for dygraph_encoder_prefix_name, static_encoder_prefix_name in zip(
            dygraph_encoder_prefix_names, static_encoder_prefix_names):
        for i in range(0, sharing_layer_num
                       if static_encoder_prefix_name == 'encoder_layer' else
                       nlg_layer_num):
            for j in range(0, 2):
                for dygraph_param_name, static_param_name in zip(
                        dygraph_param_names, static_param_names):
                    convert_parameter_name_dict[dygraph_ffn_linear_format_name.format(dygraph_encoder_prefix_name, i, j+1,  dygraph_param_name)] = \
                    static_ffn_linear_format_name.format(static_encoder_prefix_name, i if static_encoder_prefix_name=='encoder_layer' else i+48, j, static_param_name)

    # Thirdly, converts the multi_head layer_norm parameter.
    dygraph_encoder_attention_layer_norm_format_name = "{}.layers.{}.norm1.{}"
    static_encoder_attention_layer_norm_format_name = "{}_{}_pre_att_layer_norm_{}"
    for dygraph_encoder_prefix_name, static_encoder_prefix_name in zip(
            dygraph_encoder_prefix_names, static_encoder_prefix_names):
        for i in range(0, sharing_layer_num
                       if static_encoder_prefix_name == 'encoder_layer' else
                       nlg_layer_num):
            for dygraph_param_name, static_pararm_name in zip(
                    dygraph_layer_norm_param_names,
                    static_layer_norm_param_names):
                convert_parameter_name_dict[dygraph_encoder_attention_layer_norm_format_name.format(dygraph_encoder_prefix_name, i, dygraph_param_name)] = \
                    static_encoder_attention_layer_norm_format_name.format(static_encoder_prefix_name, i if static_encoder_prefix_name=='encoder_layer' else i+48, static_pararm_name)

    dygraph_encoder_ffn_layer_norm_format_name = "{}.layers.{}.norm2.{}"
    static_encoder_ffn_layer_norm_format_name = "{}_{}_pre_ffn_layer_norm_{}"
    for dygraph_encoder_prefix_name, static_encoder_prefix_name in zip(
            dygraph_encoder_prefix_names, static_encoder_prefix_names):
        for i in range(0, sharing_layer_num
                       if static_encoder_prefix_name == 'encoder_layer' else
                       nlg_layer_num):
            for dygraph_param_name, static_pararm_name in zip(
                    dygraph_layer_norm_param_names,
                    static_layer_norm_param_names):
                convert_parameter_name_dict[dygraph_encoder_ffn_layer_norm_format_name.format(dygraph_encoder_prefix_name, i, dygraph_param_name)] = \
                    static_encoder_ffn_layer_norm_format_name.format(static_encoder_prefix_name, i if static_encoder_prefix_name=='encoder_layer' else i+48, static_pararm_name)

    convert_parameter_name_dict[
        "sharing_layer_norm.weight"] = "server_post_encoder_layer_norm_scale"
    convert_parameter_name_dict[
        "sharing_layer_norm.bias"] = "server_post_encoder_layer_norm_bias"
    convert_parameter_name_dict[
        "nlg_layer_norm.weight"] = "nlg_post_encoder_layer_norm_scale"
    convert_parameter_name_dict[
        "nlg_layer_norm.bias"] = "nlg_post_encoder_layer_norm_bias"
    return convert_parameter_name_dict


def match_sharing_to_nlg_parameter(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "sharing_to_nlg.weight"] = "sharing_to_task_fc.w_0"
    return convert_parameter_name_dict


def match_mlm_parameter(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "lm_head.lm_transform.weight"] = "nlg_mask_lm_trans_fc.w_0"
    convert_parameter_name_dict[
        "lm_head.lm_transform.bias"] = "nlg_mask_lm_trans_fc.b_0"
    convert_parameter_name_dict[
        "lm_head.layer_norm.weight"] = "nlg_mask_lm_trans_layer_norm_scale"
    convert_parameter_name_dict[
        "lm_head.layer_norm.bias"] = "nlg_mask_lm_trans_layer_norm_bias"
    convert_parameter_name_dict[
        "lm_head.lm_out.weight"] = "nlg_mask_lm_out_fc.w_0"
    convert_parameter_name_dict[
        "lm_head.lm_out.bias"] = "nlg_mask_lm_out_fc.b_0"
    return convert_parameter_name_dict


def convert_static_to_dygraph_params(dygraph_params_save_path,
                                     static_params_dir,
                                     static_to_dygraph_param_name,
                                     model_name='ernie3'):
    files = os.listdir(static_params_dir)
    state_dict = {}
    for static_para_name in files:
        path = os.path.join(static_params_dir, static_para_name)

        if static_para_name not in static_to_dygraph_param_name:
            print(static_para_name, "not in static_to_dygraph_param_name")
            continue
        dygraph_para_name = static_to_dygraph_param_name[static_para_name]
        value = paddle.load(path).numpy()
        if "lm_head" in dygraph_para_name:
            # Note: lm_head parameters do not need add `model_name.` prefix
            state_dict[dygraph_para_name] = value
        else:
            state_dict[model_name + '.' + dygraph_para_name] = value

    with open(dygraph_params_save_path, 'wb') as f:
        pickle.dump(state_dict, f)
    params = paddle.load(dygraph_params_save_path)

    cnt = 0
    for name in state_dict.keys():
        if name in params:
            assert ((state_dict[name] == params[name].numpy()).all() == True)
            cnt += 1
        else:
            print(name, 'not in params')
    print(f'Totally convert {cnt} params.')


if __name__ == "__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, sharing_layer_num=48, nlg_layer_num=12)
    convert_parameter_name_dict = match_sharing_to_nlg_parameter(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_mlm_parameter(
        convert_parameter_name_dict)
    static_to_dygraph_param_name = {
        value: key
        for key, value in convert_parameter_name_dict.items()
    }
    for static_name, dygraph_name in static_to_dygraph_param_name.items():
        print("{}:-------:{}".format(static_name, dygraph_name))

    convert_static_to_dygraph_params(
        dygraph_params_save_path='ernie3_10b.pdparams',
        static_params_dir='/guosheng/gongenlei/step_100000',
        static_to_dygraph_param_name=static_to_dygraph_param_name,
        model_name='ernie3')
