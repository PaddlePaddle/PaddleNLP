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

# import json
# import os
from collections import OrderedDict

import numpy as np
import paddle

# import paddle.distributed as dist
from paddle.distributed import fleet

# from paddlenlp.transformers.model_utils import _add_variant, get_parameter_dtype
from paddlenlp.transformers.utils import dtype_byte_size  # paddlenlp_load
from paddlenlp.utils.log import logger

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"
MODEL_META_NAME = "model_meta.json"
SHARDING_META_NAME = "shard_meta.json"


class ShardedCkptIO:
    def __init__(self, args, model, optimizer=None, hcg=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.tp_group = None
        if self.hcg is None and paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.tp_group = self.hcg.get_model_parallel_group()

    def manipulate_state_dict_and_config(self, model_to_save):
        weight_name_suffix = self.args.weight_name_suffix()

        state_dict = model_to_save.state_dict()
        state_dict = filter_params(state_dict, self.optimizer, self.tp_group.rank)

        config_to_save = None
        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel
        if merge_tensor_parallel:
            dtype = get_parameter_dtype(model_to_save)
            assert hasattr(model_to_save, "config")
            model_to_save.config.dtype = str(dtype).split(".")[1]
            config_to_save = copy.deepcopy(model_to_save.config)
            if config_to_save.tensor_parallel_degree > 1:
                state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                config_to_save.tensor_parallel_degree = 1
                if config_to_save.tensor_parallel_rank != 0:
                    logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                    return
                # if variant is not None and "tp" in variant:
                if "tp" in weight_name_suffix:
                    weight_name_suffix = "_".join([x for x in weight_name_suffix.split("_") if "tp" not in x])

        if self.args.bf16 and self.args.should_save_sharding_stage1_model:
            param_names_in_master_weights = []
            optimzier_state_dict = self.optimizer.state_dict()
            assert "master_weights" in optimzier_state_dict
            param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys())
            state_dict = exclude_paramters_in_state_dict(
                state_dict, param_names_in_master_weights, self.sharding_group
            )
            logger.info(
                "param_names_in_master_weights len:{}, bf16 state_dict len:{}, :{}".format(
                    len(param_names_in_master_weights), len(state_dict), state_dict
                )
            )
        return state_dict, config_to_save, weight_name_suffix

    def filter_params(self, state_dict, optimizer=None, tp_rank=0):

        logger.info(f"filter params for different workers to save.")

        filtered_state_dict = OrderedDict()

        if tp_rank == 0:
            tensor_bytes_dict = {}
            for (k, v) in state_dict.items():
                tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)
            tensor_bytes_dict = sorted(state_dict.items(), key=lambda x: x[1])
            tp_size = self.tp_group.nranks
            # [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
            # tp_range = np.arange(0, tp_size)
            # tp_range = None

        for (k, v) in state_dict.items():
            filtered_state_dict[k] = v
        return filtered_state_dict
