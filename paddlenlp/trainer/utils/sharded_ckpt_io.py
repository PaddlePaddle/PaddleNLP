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

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlenlp.transformers.model_utils import get_parameter_dtype  # _add_variant
from paddlenlp.transformers.utils import dtype_byte_size  # , paddlenlp_load
from paddlenlp.utils.log import logger


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

    def manipulate_state_dict_and_config(self, model_to_save):
        state_dict = model_to_save.state_dict()
        if self.args.use_hybrid_parallel:
            # state_dict, all_filter_keys = self.filter_params(state_dict, self.optimizer, self.tp_group.rank)
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
        return state_dict, config_to_save

    def filter_params(self, model_to_save, state_dict, optimizer=None, tp_rank=0):
        logger.info("filter params for different workers to save.")

        tp_size = self.tp_group.nranks
        filter_tensor_list = [[] for i in range(tp_size)]
        if tp_rank == 0:
            name_action_mappings = model_to_save._get_tensor_parallel_mappings(model_to_save.config, is_split=False)
            state_keys_map = model_to_save._resolve_prefix_keys(name_action_mappings.keys(), state_dict.keys())
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
