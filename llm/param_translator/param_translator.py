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

import os

import paddle


class ParamTranslator:
    def __init__(
        self,
        ranks_name_maps,
        need_repartition_tensor_names,
        need_repartition_files,
        model_state_dict_file_names,
        optim_state_dict_file_names,
        dy_hand_checkpoint_path,
        saved_checkpoint_path,
        save_file_names,
    ):

        assert len(ranks_name_maps) == len(model_state_dict_file_names)
        assert len(model_state_dict_file_names) == len(optim_state_dict_file_names)
        assert len(optim_state_dict_file_names) == len(save_file_names)

        self.ranks_name_maps = ranks_name_maps
        self.need_repartition_tensor_names = need_repartition_tensor_names
        self.need_repartition_files = need_repartition_files
        self.model_state_dict_file_names = model_state_dict_file_names
        self.optim_state_dict_file_names = optim_state_dict_file_names
        self.dy_hand_checkpoint_path = dy_hand_checkpoint_path
        self.saved_checkpoint_path = saved_checkpoint_path
        self.save_file_names = save_file_names

    def repartition(self, state_dicts):
        partition_len = len(state_dicts)
        for tensor_name in self.need_repartition_tensor_names:
            concat_input = []
            for state_dict in state_dicts:
                print("========> origin state dict: ", state_dict, flush=1)
                print("get tensor :", tensor_name, flush=1)
                concat_input.append(state_dict[tensor_name])
                print("concat_input :", concat_input, flush=1)
            t = paddle.concat(concat_input)
            print("t: ", t, flush=1)
            split_output = paddle.split(t, partition_len, 1)
            print("split_output: ", split_output, flush=1)
            for state_dict in state_dicts:
                state_dict[tensor_name] = split_output.pop(0)

    def load_dy_state_dict(self, model_state_dict_name, optim_state_dict_name):
        state_rank = {}
        model_state_dict = paddle.load(os.path.join(self.dy_hand_checkpoint_path, model_state_dict_name))
        for key in model_state_dict.keys():
            state_rank[key] = model_state_dict[key]
        optim_state_dict = paddle.load(os.path.join(self.dy_hand_checkpoint_path, optim_state_dict_name))
        optim_state_dict.pop("LR_Scheduler")
        master_weights = optim_state_dict.pop("master_weights")
        for key in optim_state_dict.keys():
            state_rank[key] = optim_state_dict[key]
        for key in master_weights.keys():
            state_rank[key] = master_weights[key]
        return state_rank

    def translate(self, state_dict, name_map):
        translated_state_dict = {}
        for key, val in state_dict.items():
            translated_state_dict[name_map[key]] = val
        return translated_state_dict

    def save_state_dict(self, state_dict, path, state_dict_name):
        paddle.save(state_dict, os.path.join(path, state_dict_name))

    def do_translate(self):
        file_name_to_state_dict = {}
        if len(self.need_repartition_files) > 0:
            state_dicts = []
            for file in self.need_repartition_files:
                state_dict = self.load_dy_state_dict(file["model"], file["optim"])
                need_remove_keys = []
                for k, v in state_dict.items():
                    if k not in self.need_repartition_tensor_names:
                        need_remove_keys.append(k)
                for k in need_remove_keys:
                    state_dict.pop(k)
                state_dicts.append(state_dict)
                file_name_to_state_dict[file["model"] + file["optim"]] = state_dict
            self.repartition(state_dicts)
            for state_dict in state_dicts:
                print(state_dict)

        for rank in range(len(self.ranks_name_maps)):
            need_update_tensor = {}
            print("start to translate rank ", rank)
            cache_key = os.path.join(
                self.dy_hand_checkpoint_path, self.model_state_dict_file_names[rank]
            ) + os.path.join(self.dy_hand_checkpoint_path, self.optim_state_dict_file_names[rank])
            print(cache_key)
            if cache_key in file_name_to_state_dict.keys():
                print("find_cache_key!")
                need_update_tensor = file_name_to_state_dict[cache_key]
            state_dict = self.load_dy_state_dict(
                self.model_state_dict_file_names[rank], self.optim_state_dict_file_names[rank]
            )
            print("==========> need_update_tensor shape and dtype:")
            for k, v in need_update_tensor.items():
                print(k, v.shape, v.dtype)
            for k in need_update_tensor.keys():
                if k in self.need_repartition_tensor_names:
                    state_dict[k] = need_update_tensor[k]
            state_dict = self.translate(state_dict, self.ranks_name_maps[rank])
            self.save_state_dict(state_dict, self.saved_checkpoint_path, self.save_file_names[rank])
