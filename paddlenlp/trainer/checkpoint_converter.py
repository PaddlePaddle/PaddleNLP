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

import json
import os
import re
from functools import reduce

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.checkpoint.metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
)

MODEL_WEIGHT_SUFFIX = ".pdparams"
OPTIMIZER_WEIGHT_SUFFIX = ".pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
MODEL_META_FILE_NAME = "model_meta.json"


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class DynamicToStaticShardingV2CheckpointConverter:
    def __init__(self, dynamic_ckpt_path, model_state_global_shape):
        self.path = dynamic_ckpt_path
        self.model_state_global_shape = model_state_global_shape
        self.model_meta = json.load(open(os.path.join(dynamic_ckpt_path, MODEL_META_FILE_NAME)))
        (
            self.cur_rank_model_state_file_names,
            self.cur_rank_optimizer_state_file_names,
        ) = self.get_local_checkpoint_file_names()
        self.cur_rank_loaded_state_dict = {}
        (
            self.global_model_state_file_names,
            self.global_optimizer_state_file_names,
        ) = self.get_all_checkpoint_file_names()

    def get_local_checkpoint_file_names(self):
        cur_rank_files = os.listdir(self.path)
        cur_rank_model_state_file_names = []
        cur_rank_optimizer_state_file_names = []
        global_model_state_file_names = []
        global_optimizer_state_file_names = []
        for file in cur_rank_files:
            if file.endswith(MODEL_WEIGHT_SUFFIX):
                cur_rank_model_state_file_names.append(file)
            elif file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                cur_rank_optimizer_state_file_names.append(file)
        if SCHEDULER_NAME in cur_rank_model_state_file_names:
            cur_rank_model_state_file_names.remove(SCHEDULER_NAME)
        return cur_rank_model_state_file_names, cur_rank_optimizer_state_file_names

    def get_all_checkpoint_file_names(self):
        cur_rank_model_state_file_names, cur_rank_optimizer_state_file_names = self.get_local_checkpoint_file_names()
        use_dist = True if paddle.distributed.get_world_size() > 1 else False
        global_model_state_file_names = []
        global_optimizer_state_file_names = []
        if use_dist:
            paddle.distributed.init_parallel_env()
            paddle.distributed.all_gather_object(global_model_state_file_names, cur_rank_model_state_file_names)
            paddle.distributed.all_gather_object(
                global_optimizer_state_file_names, cur_rank_optimizer_state_file_names
            )
        else:
            global_model_state_file_names = [cur_rank_model_state_file_names]
            global_optimizer_state_file_names = [cur_rank_optimizer_state_file_names]

        return global_model_state_file_names, global_optimizer_state_file_names

    def get_local_load_files(self, rank_to_files):
        import copy

        file_to_ranks = {}
        for rank, files in rank_to_files.items():
            for file in files:
                if file not in file_to_ranks:
                    file_to_ranks[file] = []
                file_to_ranks[file].append(rank)
        rank_to_not_read_files = copy.copy(rank_to_files)
        rank_to_read_files = {rank: [] for rank in rank_to_not_read_files.keys()}
        for file, ranks in file_to_ranks.items():
            if len(ranks) == 1:
                rank = ranks[0]
                rank_to_read_files[rank].append(file)
                rank_to_not_read_files[rank].remove(file)
                if len(rank_to_not_read_files[rank]) == 0:
                    rank_to_not_read_files.pop(rank)

        def get_least_read_files_ranks(rank_to_read_files):
            nums = [(rank, len(files)) for rank, files in rank_to_read_files.items()]
            nums = sorted(nums, key=lambda x: x[1])
            ranks = [rank for rank, num in nums if num == nums[0][1]]
            return ranks

        def get_read_rank_file(rank_to_not_read_files, ranks):
            if len(rank_to_not_read_files) == 0:
                return (None, None)
            nums = [(rank, len(files)) for rank, files in rank_to_not_read_files.items() if rank in ranks]
            nums = sorted(nums, key=lambda x: x[1])
            rank = nums[0][0]
            return (rank, rank_to_not_read_files[rank][0])

        def update(rank_to_read_files, rank_to_not_read_files, rank_file):
            rank, file = rank_file
            if rank is None and file is None:
                return
            if rank not in rank_to_read_files:
                rank_to_read_files[rank] = []
            rank_to_read_files[rank].append(file)
            # update rank_to_not_read_files
            file_to_ranks = {}
            for r, files in rank_to_not_read_files.items():
                for f in files:
                    if f not in file_to_ranks:
                        file_to_ranks[f] = []
                    file_to_ranks[f].append(r)

            if file in file_to_ranks:
                for r in file_to_ranks[file]:
                    rank_to_not_read_files[r].remove(file)
                    if len(rank_to_not_read_files[r]) == 0:
                        rank_to_not_read_files.pop(r)

        while len(rank_to_not_read_files) > 0:
            ranks = get_least_read_files_ranks(rank_to_read_files)
            rank_file = get_read_rank_file(rank_to_not_read_files, ranks)
            update(rank_to_read_files, rank_to_not_read_files, rank_file)

        cur_rank = paddle.distributed.get_rank()
        if cur_rank in rank_to_read_files:
            return rank_to_read_files[cur_rank]
        else:
            return []

    def extract_distribution_strategy_from_file_name(self, file_name):
        pp_degree = 0
        tp_degree = 0
        sharding_degree = 0
        pattern_pp = r"pp(\d+)"
        pattern_tp = r"tp(\d+)"
        pattern_shard = r"shard(\d+)"
        match_pp = re.search(pattern_pp, file_name)
        if match_pp:
            pp_degree = int(match_pp.group(1))
        match_tp = re.search(pattern_tp, file_name)
        if match_tp:
            tp_degree = int(match_tp.group(1))
        match_shard = re.search(pattern_shard, file_name)
        if match_shard:
            sharding_degree = int(match_shard.group(1))
        return (tp_degree, pp_degree, sharding_degree)

    def gen_matadata_for_optimizer(self):
        rank_access_files = {}

        for rank in range(paddle.distributed.get_world_size()):
            rank_access_files[rank] = (
                self.global_model_state_file_names[rank] + self.global_optimizer_state_file_names[rank]
            )

        # Determine which files need to be read for each rank.
        # When each node has a checkpoint path, it will fail
        need_read_files = self.get_local_load_files(rank_access_files)

        sharded_tensor_infos = {}
        file_to_model_state_names = {}
        file_to_optimizer_state_names = {}
        model_state_dict_info = {}
        cur_rank_sharded_tensor_infos = {}

        for file in need_read_files:
            if OPTIMIZER_WEIGHT_SUFFIX in file:
                state_dict = paddle.load(os.path.join(self.path, file), return_numpy=True)
                state_dict.pop("LR_Scheduler")
                master_weights = state_dict.pop("master_weights")
                # Extract master weights
                for k, v in master_weights.items():
                    state_dict[k] = v
                # Based on the checkpoint file name, determine the pp_degree, tp_degree, and sharding_degree of the tensor in the current file.
                distributed_rank = self.extract_distribution_strategy_from_file_name(file)
                dist_strategy_key = (
                    "tp" + "{:02d}".format(distributed_rank[0]) + "_" + "pp" + "{:02d}".format(distributed_rank[1])
                )

                # Map model weight names to their corresponding names of master_weights in the optimizer state.
                structure_name_mapping = self.model_meta["sharding_metas"][dist_strategy_key]["structure_name_mapping"]

                # The rule for renaming is to change the master_weights name in the optimizer state to the model weight name, and then append the tp_degree.
                renamed_state_dict = {}
                for k, v in state_dict.items():
                    for prame_name, opt_name in structure_name_mapping.items():
                        if opt_name in k:
                            new_key = k.replace(opt_name, prame_name) + "_tp" + "{:02d}".format(distributed_rank[0])
                        else:
                            new_key = k.replace(opt_name, prame_name)
                            renamed_state_dict[new_key] = v
                            # Calculate the local_shape.
                            cur_rank_sharded_tensor_infos[(new_key, file)] = [v.shape, str(v.dtype)]

                # Cache the renamed state dict
                self.cur_rank_loaded_state_dict[file] = renamed_state_dict

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        # Obtain the local_shape information of the tensor on all ranks.
        all_rank_sharded_tensor_infos = []
        if use_dist:
            paddle.distributed.all_gather_object(all_rank_sharded_tensor_infos, cur_rank_sharded_tensor_infos)
        else:
            all_rank_sharded_tensor_infos = [cur_rank_sharded_tensor_infos]

        global_sharded_tensor_infos = {}
        for rank_sharded_tensor_infos in all_rank_sharded_tensor_infos:
            for k, v in rank_sharded_tensor_infos.items():
                if k not in global_sharded_tensor_infos:
                    global_sharded_tensor_infos[k] = v

        # Collect sharding information.
        key_to_sharded_info = {}
        for k, v in global_sharded_tensor_infos.items():
            distributed_rank = self.extract_distribution_strategy_from_file_name(k[1])
            if k[0] not in key_to_sharded_info:
                key_to_sharded_info[k[0]] = [[distributed_rank[2], v[0], v[1], k[1]]]
            else:
                key_to_sharded_info[k[0]].append([distributed_rank[2], v[0], v[1], k[1]])

        # x[0] records the sharding rank.
        for k, v in key_to_sharded_info.items():
            v.sort(key=lambda x: x[0])

        state_dict_metadata = {}
        storage_metadata = {}

        # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
        for k, v in key_to_sharded_info.items():
            global_offset = 0
            for item in v:
                local_tensor_meta_data = LocalTensorMetadata((global_offset,), item[1], item[2])
                local_tensor_index = LocalTensorIndex(k, (global_offset,))
                global_offset += item[1][0]
                if k not in state_dict_metadata:
                    state_dict_metadata[k] = [local_tensor_meta_data]
                else:
                    state_dict_metadata[k].append(local_tensor_meta_data)
                storage_metadata[local_tensor_index] = item[3] + ".distcp"

        # Save the metadata and the renamed tensor read by this rank.
        metadata = Metadata(state_dict_metadata, storage_metadata, None)
        write_path = os.path.join(self.path, "tmp")
        for file in self.cur_rank_loaded_state_dict:
            paddle.save(self.cur_rank_loaded_state_dict[file], os.path.join(write_path, file + ".distcp"))
        if 0 == paddle.distributed.get_rank():
            paddle.save(metadata, os.path.join(write_path, "0.metadata"))

    def concat_optimier_state_dict(self):
        # Obtain the global_shape passed in semi-automatic parallel mode on each card in the static graph.
        all_rank_model_state_global_shapes = []
        use_dist = True if paddle.distributed.get_world_size() > 1 else False
        if use_dist:
            paddle.distributed.all_gather_object(all_rank_model_state_global_shapes, self.model_state_global_shape)
        else:
            all_rank_model_state_global_shapes = [self.model_state_global_shape]

        self.model_state_global_shape = {}
        for rank_model_state_global_shape in all_rank_model_state_global_shapes:
            for k, v in rank_model_state_global_shape.items():
                self.model_state_global_shape[k] = v

        # Obtain the names and shapes of all model parameters.
        global_model_state_shapes = {}
        sharding_metas_keys = []
        pp_degree = self.model_meta["parallel_config"]["pp_degree"]
        mp_degree = self.model_meta["parallel_config"]["mp_degree"]
        for i in range(pp_degree):
            for j in range(mp_degree):
                sharding_metas_keys.append("tp{:02d}_pp{:02d}".format(i, j))
        for key in sharding_metas_keys:
            param_meta = self.model_meta["sharding_metas"][key]["param_meta"]
            for k, v in param_meta.items():
                global_model_state_shapes[k] = v[0]

        use_dist = True if paddle.distributed.get_world_size() > 1 else False
        world_size = paddle.distributed.get_world_size()

        # Distribute all model parameters evenly across each card for loading
        global_model_state_flattened_shapes = {}
        global_model_state_size = 0
        for k, v in global_model_state_shapes.items():
            flattened_size = reduce(lambda x, y: x * y, v)
            global_model_state_size += flattened_size
            global_model_state_flattened_shapes[k] = flattened_size

        partition_model_state_keys = []
        avg_size = global_model_state_size // world_size

        cur_rank_model_state_keys = []

        cur_rank_size = 0
        for k, v in global_model_state_flattened_shapes.items():
            cur_rank_size += v
            cur_rank_model_state_keys.append(k)
            if cur_rank_size > avg_size:
                partition_model_state_keys.append(cur_rank_model_state_keys)
                cur_rank_model_state_keys = []
                cur_rank_size = 0

        # Since an absolutely even distribution is not achievable, some tanks may not need to load, but the load_state_dict interface might throw an error. Therefore, it is necessary to forcefully assign a parameter.
        nend_append = world_size - len(partition_model_state_keys)
        for i in range(nend_append):
            partition_model_state_keys.append([partition_model_state_keys[0][0]])

        cur_rank = paddle.distributed.get_rank()

        cur_rank_need_load_model_state_keys = partition_model_state_keys[cur_rank]

        # Generate the optimizer states corresponding to the model weights.
        optimizer_state_dict = {}
        for key in cur_rank_need_load_model_state_keys:
            for tp_rank in range(self.model_meta["parallel_config"]["mp_degree"]):
                tp_rank_suffix = "_tp{:02d}".format(tp_rank)
                optimizer_state_dict[key + "_fp32_master_0_moment1_0" + tp_rank_suffix] = paddle.zeros(
                    (global_model_state_flattened_shapes[key],), "float32"
                )
                optimizer_state_dict[key + "_fp32_master_0_moment2_0" + tp_rank_suffix] = paddle.zeros(
                    (global_model_state_flattened_shapes[key],), "float32"
                )
                optimizer_state_dict[key + tp_rank_suffix] = paddle.zeros(
                    (global_model_state_flattened_shapes[key],), "float32"
                )
            optimizer_state_dict[key + "_fp32_master_0_beta1_pow_acc_0"] = paddle.zeros((1,), "float32")
            optimizer_state_dict[key + "_fp32_master_0_beta2_pow_acc_0"] = paddle.zeros((1,), "float32")

        dist.load_state_dict(optimizer_state_dict, os.path.join(self.path, "tmp"))

        # Reshape
        for k, v in optimizer_state_dict.items():
            if v.shape[0] > 1 and "_tp" in k:
                for master_weight_key, shape in global_model_state_shapes.items():
                    if master_weight_key in k:
                        reshaped_v = v.reshape(shape)
                        optimizer_state_dict[k] = reshaped_v

        concat_optimier_state_dict = {}

        optimizer_state_key_to_tp_keys = {}
        for key in optimizer_state_dict.keys():
            # Count how each key is split into keys ending with ‘_tpXX’.
            # optimizer_state_key_to_tp_keys ： {key:[key_tp00,key_tp01]}
            if "_pow_acc_0" not in key:
                if key[:-5] not in optimizer_state_key_to_tp_keys:
                    optimizer_state_key_to_tp_keys[key[:-5]] = [key]
                else:
                    optimizer_state_key_to_tp_keys[key[:-5]].append(key)
            else:
                optimizer_state_key_to_tp_keys[key] = [key]
        for key, value in optimizer_state_key_to_tp_keys.items():
            if len(value) == 1:
                continue
            value.sort(key=lambda x: int(x[-2:]))

        for key, tp_keys in optimizer_state_key_to_tp_keys.items():
            # Optimizer states with a shape of 1 could be replicated; here, perform a check.
            is_replicated = True
            tp_tensor = optimizer_state_dict[tp_keys[0]]
            for tp_key in tp_keys:
                if not np.array_equal(tp_tensor.numpy(), optimizer_state_dict[tp_key].numpy()):
                    is_replicated = False
                    break
            if is_replicated:
                concat_optimier_state_dict[key] = tp_tensor
                continue
            else:
                tp_tensors = []
                for tp_key in tp_keys:
                    tp_tensors.append(optimizer_state_dict[tp_key])
                # Derive the partition strategy based on the global_shape, then concatenate.
                axis = 0
                global_shape = []
                # Find the global_shape.
                for k, shape in self.model_state_global_shape.items():
                    if k in tp_key:
                        global_shape = shape
                        break
                assert len(global_shape) != 0
                tp_shape = tp_tensors[0].shape
                assert (tp_shape[0] == global_shape[0] and len(tp_tensors) * tp_shape[1] == global_shape[1]) or (
                    tp_shape[1] == global_shape[1] and len(tp_tensors) * tp_shape[0] == global_shape[0]
                )
                if tp_shape[0] == global_shape[0]:
                    axis = 1
                concat_optimier_state_dict[key] = paddle.concat(tp_tensors, axis=axis)

        file_name = "{:02d}".format(cur_rank) + ".distcp"
        local_tensor_meta_data = {}
        local_tensor_index = {}
        for k, v in concat_optimier_state_dict.items():
            # Generate metadata.
            local_shape = v.shape
            global_offset = tuple([0] * len(local_shape))
            dtype = str(v.dtype)
            local_tensor_meta_data[k] = LocalTensorMetadata(global_offset, local_shape, dtype)
            local_tensor_index[k] = [LocalTensorIndex(k, global_offset), file_name]

        global_local_tensor_meta_data = []
        global_local_tensor_index = []

        if use_dist:
            paddle.distributed.all_gather_object(global_local_tensor_meta_data, local_tensor_meta_data)
            paddle.distributed.all_gather_object(global_local_tensor_index, local_tensor_index)
        else:
            global_local_tensor_meta_data = [local_tensor_meta_data]
            global_local_tensor_index = [local_tensor_index]

        state_dict_metadata = {}
        for tensor_meta_data in global_local_tensor_meta_data:
            for k, v in tensor_meta_data.items():
                if k not in state_dict_metadata:
                    state_dict_metadata[k] = [v]
                else:
                    state_dict_metadata[k].append(v)

        storage_metadata = {}
        for tensor_index in global_local_tensor_index:
            for k, v in tensor_index.items():
                storage_metadata[v[0]] = v[1]

        meta_data = Metadata(state_dict_metadata, storage_metadata, None)

        save_path = os.path.join(self.path, "tmp2")

        if cur_rank == 0:
            paddle.save(meta_data, os.path.join(save_path, "0.metadata"))

        paddle.save(concat_optimier_state_dict, os.path.join(save_path, file_name))
