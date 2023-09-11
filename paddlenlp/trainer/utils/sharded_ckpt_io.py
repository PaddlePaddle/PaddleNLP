# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlenlp.transformers.model_utils import _add_variant, get_parameter_dtype
from paddlenlp.transformers.utils import dtype_byte_size  # , paddlenlp_load
from paddlenlp.utils.env import PADDLE_WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from paddlenlp.utils.log import logger

local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

MODEL_PDPARAMS_INDEX_NAME = "model.pdparams.index.json"


class ShardedCkptIO:
    def __init__(self, args, model, optimizer=None, hcg=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.tp_group = None
        self.hcg = hcg
        if self.hcg is None and paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.tp_group = self.hcg.get_model_parallel_group()

    def manipulate_state_dict_and_config(self, model_to_save, weight_name_suffix, safe_serialization=False):
        state_dict = model_to_save.state_dict()
        if self.args.use_hybrid_parallel:
            all_filter_keys = self.filter_params(model_to_save, state_dict, self.optimizer, self.tp_group.rank)
            dtype = get_parameter_dtype(model_to_save)
            assert hasattr(model_to_save, "config")
            model_to_save.config.dtype = str(dtype).split(".")[1]
            config_to_save = copy.deepcopy(model_to_save.config)
            if config_to_save.tensor_parallel_degree > 1:
                state_dict = model_to_save.merge_tensor_parallel_with_shard(
                    state_dict, config_to_save, all_filter_keys
                )
                config_to_save.tensor_parallel_degree = 1

            # build index json file
            self.index_file_list = []
            index_weight_file = {}
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
            weights_name = _add_variant(weights_name, weight_name_suffix)
            for key in state_dict.keys():
                index_weight_file[key] = weights_name
            data_group = self.hcg.get_data_parallel_group()
            if data_group.rank == -1:
                dist.all_gather_object(self.index_file_list, index_weight_file)
            else:
                dist.all_gather_object(self.index_file_list, index_weight_file, group=data_group)

        return state_dict, config_to_save

    def save_sharded_index(self, output_dir):
        # save index json file
        if local_rank == 0:
            sharded_index_json = {}
            final_dict = self.index_file_list[0]
            for i, index_file in enumerate(self.index_file_list):
                if i == 0:
                    continue
                final_dict.update(self.index_file_list[i])
            sharded_index_json["weight_map"] = final_dict

            path = os.path.join(output_dir, MODEL_PDPARAMS_INDEX_NAME)
            with open(path, "w") as f:
                json.dump(sharded_index_json, f, indent=4)

    def filter_params(self, model_to_save, state_dict, optimizer=None, tp_rank=0):
        logger.info("filter params for different workers to save.")

        tp_size = self.tp_group.nranks
        filter_tensor_list = [[] for i in range(tp_size)]
        if tp_rank == 0:
            name_action_mappings = model_to_save._get_tensor_parallel_mappings(model_to_save.config, is_split=False)
            state_keys_map = model_to_save._resolve_prefix_keys(
                name_action_mappings.keys(), state_dict.keys(), ignore_error=True
            )
            for k, v in state_keys_map.items():
                name_action_mappings[v] = name_action_mappings.pop(k)

            tensor_bytes_dict = {}
            for (k, v) in state_dict.items():
                if k in name_action_mappings:
                    tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
                else:
                    tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)

            # Sort by tensor storage.
            tensor_bytes_dict = sorted(tensor_bytes_dict.items(), key=lambda x: x[1])
            keys_list = [key for key, byte in tensor_bytes_dict]
            # [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
            tp_range = np.arange(0, tp_size)
            tensor_cnt, tp_cnt = 0, 0
            while tensor_cnt < len(state_dict):
                filter_tensor_list[tp_range[tp_cnt]].append(keys_list[tensor_cnt])
                tensor_cnt += 1
                tp_cnt += 1
                if tp_cnt == tp_size:
                    tp_cnt = 0
        dist.broadcast_object_list(
            filter_tensor_list, src=self.hcg.get_model_parallel_group_src_rank(), group=self.tp_group
        )
        return filter_tensor_list
