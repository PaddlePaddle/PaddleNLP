# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


# 给定静态图参数名称列表st_param_list，找到第一个包含keyword关键字的参数，返回并从列表中删除
def find_and_remove_first_string_with_keyword(st_param_list, keyword):
    for index, string in enumerate(st_param_list):
        if keyword in string:
            found_string = st_param_list.pop(index)
            return found_string
    return None


# 返回不同优化器的动态图静态图参数名称映射规则
def opt_master_param_name_map(optimizer=None):
    if optimizer is None or optimizer == "Adamw":
        return {
            "": "_fp32_master_1",
            "_fp32_master_0_moment1_0": "_fp32_master_1_moment1_0",
            "_fp32_master_0_moment2_0": "_fp32_master_1_moment2_0",
            "_fp32_master_0_beta1_pow_acc_0": "_fp32_master_1_beta1_pow_acc_0",
            "_fp32_master_0_beta2_pow_acc_0": "_fp32_master_1_beta2_pow_acc_0",
        }
    else:
        print("!!!! error, not support this optimizer !!!!")
        exit()


# 加载动手权重、静半权重列表，生成动手权重名称到静半权重名称的映射map
def trans_param_name(dy_model_file, st_model_opt_param_list):
    print("loading origin data ....", flush=1)
    dy_state_dict = paddle.load(dy_model_file)
    print("==> dy param num: ", len(dy_state_dict), flush=1)
    print(dy_state_dict.values(), flush=1)

    st_param_list = [s for s in st_model_opt_param_list if "master" not in s]
    print("==> st model param num: ", len(st_param_list), flush=1)
    print(st_param_list, flush=1)

    print("-------------------------", flush=1)
    dy_to_st_map = {}
    for dy_key, dy_name in dy_state_dict.items():
        if "embedding" in dy_name:
            st_param = find_and_remove_first_string_with_keyword(st_param_list, "embedding")
        elif "linear" in dy_name:
            st_param = find_and_remove_first_string_with_keyword(st_param_list, "linear")
        elif "create_parameter" in dy_name:
            st_param = find_and_remove_first_string_with_keyword(st_param_list, "create_parameter")
        elif "lm_head" in dy_name:
            st_param = find_and_remove_first_string_with_keyword(st_param_list, "lm_head")
        else:
            print("!!!! error, without key: embedding or linear or create_parameter !!!!")
            exit()

        dy_to_st_map[dy_key] = st_param
        for key, value in opt_master_param_name_map().items():
            dy_to_st_map[dy_name + key] = st_param + value

    if len(st_param_list) != 0:
        print("!!!! error, st param list is not empty !!!!")
        exit()

    print("==> dy_to_st_map num: ", len(dy_to_st_map), flush=1)
    print(dy_to_st_map)
    return dy_to_st_map


class MapGenerator:
    def __init__(self, dy_hand_checkpoint_path, dy_hand_model_state_name, st_checkpoint_path, st_model_state_name):
        self.dy_hand_checkpoint_path = dy_hand_checkpoint_path
        self.dy_hand_model_state_name = dy_hand_model_state_name
        self.st_checkpoint_path = st_checkpoint_path
        self.st_model_state_name = st_model_state_name

    def gen(self):
        rank_name_maps = {}
        for idx in range(len(self.dy_hand_model_state_name)):
            dy_model_file = self.dy_hand_checkpoint_path + self.dy_hand_model_state_name[idx]
            st_param_file = self.st_checkpoint_path + self.st_model_state_name[idx]
            st_model_opt_param_list = paddle.load(st_param_file)
            rank_name_maps[idx] = trans_param_name(dy_model_file, st_model_opt_param_list)
        return rank_name_maps


# dy_model_file = "/root/paddlejob/workspace/env_run/output/baidu/dialogue/PaddleNLP/llm/auto_parallel/llama/checkpoints_dy/llama2_pretrain_ckpts/checkpoint-1/model_state.tp00_pp00.keymap"
# st_param_file = "/root/paddlejob/workspace/env_run/output/baidu/dialogue/PaddleNLP/llm/auto_parallel/llama/checkpoints_st/llama2_pretrain_ckpts/checkpoint-1/dist_ckpt/0_0.model.param.name"
# st_model_opt_param_list = paddle.load(st_param_file)
# print("==> st model opt param num: ", len(st_model_opt_param_list), flush=1)
# trans_param_name(dy_model_file, st_model_opt_param_list)
