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

import paddle
import paddle.distributed as dist
from paddle.distributed.checkpoint.load_state_dict import (
    _load_state_dict,
    get_local_load_files,
)
from paddle.distributed.checkpoint.metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
)

MODEL_WEIGHT_SUFFIX = ".pdparams"
OPTIMIZER_WEIGHT_SUFFIX = ".pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
MODEL_META_FILE_NAME = "model_meta.json"


class CheckpointConverter:
    def __init__(self, dynamic_ckpt_path, model_state, parameter_to_structured_name):
        self.use_dist = True if paddle.distributed.get_world_size() > 1 else False
        self.path = dynamic_ckpt_path
        self.semi_auto_model_state = model_state
        self.parameter_to_structured_name = self.gather_global_object(parameter_to_structured_name)
        model_state_global_shape = {}
        for k, v in model_state.items():
            model_state_global_shape[k] = v.shape
        self.model_state_global_shape = self.gather_global_object(model_state_global_shape)
        self.cur_rank = paddle.distributed.get_rank()

        self.save_sharded_model = self.get_save_sharded_model_flag()

        (
            self.cur_rank_model_state_file_names,
            self.cur_rank_optimizer_state_file_names,
        ) = self.get_local_checkpoint_file_names()

        self.global_model_state_file_names = self.gather_global_object(self.cur_rank_model_state_file_names)

        self.global_optimizer_state_file_names = self.gather_global_object(self.cur_rank_optimizer_state_file_names)

        self.initial_distributed_configuration()

    def get_save_sharded_model_flag(self):
        if self.cur_rank == 1:
            save_sharded_model_flag = [os.path.exists(os.path.join(self.path, MODEL_META_FILE_NAME))]
        else:
            save_sharded_model_flag = []
        save_sharded_model_flag = self.gather_global_object(save_sharded_model_flag)
        return save_sharded_model_flag[0]

    def gather_global_object(self, cur_rank_object):
        all_rank_objects = []
        if self.use_dist:
            paddle.distributed.all_gather_object(all_rank_objects, cur_rank_object)
        else:
            all_rank_objects = [all_rank_objects]

        if isinstance(cur_rank_object, list):
            return [item for sublist in all_rank_objects for item in sublist]
        elif isinstance(cur_rank_object, dict):
            global_map = {}
            for rank_map in all_rank_objects:
                global_map.update(rank_map)
            return global_map
        else:
            raise ValueError("cur_rank_object should be either a list or a dict")

    def get_local_checkpoint_file_names(self):
        cur_rank_files = os.listdir(self.path)
        cur_rank_model_state_file_names = []
        cur_rank_optimizer_state_file_names = []
        for file in cur_rank_files:
            if file.endswith(MODEL_WEIGHT_SUFFIX):
                cur_rank_model_state_file_names.append(file)
            elif file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                cur_rank_optimizer_state_file_names.append(file)
        if SCHEDULER_NAME in cur_rank_model_state_file_names:
            cur_rank_model_state_file_names.remove(SCHEDULER_NAME)
        return cur_rank_model_state_file_names, cur_rank_optimizer_state_file_names

    def get_distribution_rank_from_file_name(self, file_name):
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

    def initial_distributed_configuration(self):
        self.pp_degree = 0
        self.tp_degree = 0
        self.sharding_degree = 0

        all_files = self.global_model_state_file_names + self.global_optimizer_state_file_names

        for file in all_files:
            (tp_degree, pp_degree, sharding_degree) = self.get_distribution_rank_from_file_name(file)
            self.pp_degree = max(self.pp_degree, pp_degree)
            self.tp_degree = max(self.tp_degree, tp_degree)
            self.sharding_degree = max(self.sharding_degree, sharding_degree)

        self.pp_degree = self.pp_degree + 1
        self.tp_degree = self.tp_degree + 1
        self.sharding_degree = self.sharding_degree + 1

    def infer_sharding_stage1_v(self):
        sharding_stage1_v = [2]
        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX) and sharding_stage1_v[0] == 2:
                for k, v in state_dict.items():
                    # Under shardingv2, the optimizer state is first flattened and then split.
                    if "_moment" in k and len(v.shape) != 1:
                        sharding_stage1_v = [1]
                        break

        sharding_stage1_v = self.gather_global_object(sharding_stage1_v)
        if 1 in sharding_stage1_v:
            return 1
        return 2

    def infer_is_sharding_stage3(self):
        if self.sharding_degree == 1:
            return False
        if self.pp_degree > 1 or self.tp_degree > 1:
            # Currently, sharding stage 3 does not support concurrent use with tensor parallelism (TP) and pipeline parallelism (PP).
            return False

        is_sharding_stage3 = True

        file_to_state_shape_mapping = {}
        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                state_shape_mapping = {}
                for k, v in state_dict.items():
                    state_shape_mapping[k] = v.shape
                file_to_state_shape_mapping[file] = state_shape_mapping
        global_file_to_state_shape_mapping = self.gather_global_object(file_to_state_shape_mapping)

        state_dict_std = global_file_to_state_shape_mapping[list(global_file_to_state_shape_mapping.keys())[0]]

        for file, state_dict in global_file_to_state_shape_mapping.items():
            if state_dict != state_dict_std:
                is_sharding_stage3 = False
                break
        return is_sharding_stage3

    def optimizer_state_name_to_master_weight_name(self, optimizer_state_name):
        return optimizer_state_name.split(".")[0]

    def optimizer_state_file_name_to_model_state_file_name(self, optimizer_state_file_name):
        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(optimizer_state_file_name)
        for model_state_file in self.global_model_state_file_names:
            distributed_rank = self.get_distribution_rank_from_file_name(model_state_file)
            if tp_rank == distributed_rank[0] and pp_rank == distributed_rank[1]:
                return model_state_file
        return None

    def optimizer_key_to_model_state_key(self, optimizer_key):
        adamw_optimizer_key_suffix = [
            ".w_0_beta1_pow_acc_0",
            ".w_0_beta2_pow_acc_0",
            ".w_0_moment1_0",
            ".w_0_moment2_0",
            ".w_0",
        ]
        model_state_key = optimizer_key
        for suffix in adamw_optimizer_key_suffix:
            if model_state_key.endswith(suffix):
                # Remove the suffix from model_state_key
                model_state_key = model_state_key[: -len(suffix)]
                break
        return model_state_key

    def partition_parameters(self, model_state_shapes, is_sort, shard_num):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List]
        """
        # Copy from python/paddle/distributed/fleet/meta_optimizers/dygraph_optimizer/dygraph_sharding_optimizer.py
        mapping = {}
        for rank_ in range(shard_num):
            mapping[rank_] = []
        sizes = [0] * shard_num

        parameters = model_state_shapes.copy()

        if is_sort:
            parameters.sort(key=lambda p: reduce(lambda x, y: x * y, p[1]), reverse=True)

        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param[1], 1)
            assert numel > 0, f"param [{param[0]}] should larger than 0, but it is [{numel}]"
            sizes[rank] += numel

        return mapping

    def rename_using_model_meta(self, file_name):
        if not hasattr(self, "model_meta"):
            try:
                self.model_meta = json.load(open(os.path.join(self.path, MODEL_META_FILE_NAME)))
            except Exception as e:
                print(e)
        distributed_rank = self.get_distribution_rank_from_file_name(file_name)
        dist_strategy_key = (
            "tp" + "{:02d}".format(distributed_rank[0]) + "_" + "pp" + "{:02d}".format(distributed_rank[1])
        )
        # Map model weight names to their corresponding names of master_weights in the optimizer state.
        if file_name.endswith(OPTIMIZER_WEIGHT_SUFFIX):
            structure_name_mapping = self.model_meta["sharding_metas"][dist_strategy_key]["structure_name_mapping"]
            master_weight_name_to_model_weight_name_mapping = {}
            for k, v in structure_name_mapping.items():
                master_weight_name_to_model_weight_name_mapping[v.split(".")[0]] = k

            renamed_state_dict = {}
            (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
            state_dict = self.cur_rank_loaded_state_dict[file_name]
            for k, v in state_dict.items():
                master_weight_name = self.optimizer_state_name_to_master_weight_name(k)
                model_weight_name = master_weight_name_to_model_weight_name_mapping[master_weight_name]
                new_k = k.replace(master_weight_name, model_weight_name)
                renamed_state_dict[new_k] = v
            return renamed_state_dict
        else:
            return self.cur_rank_loaded_state_dict[file_name]

    def rename_using_optimizer_state_order(self, file_name):
        if not hasattr(self, "global_file_to_state_dict_keys_mapping"):
            file_to_state_dict_keys_mapping = {}
            for file, state_dict in self.cur_rank_loaded_state_dict.items():
                file_to_state_dict_keys_mapping[file] = list(state_dict.keys())

            self.global_file_to_state_dict_keys_mapping = self.gather_global_object(file_to_state_dict_keys_mapping)

        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
        if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
            model_state_file_name = self.optimizer_state_file_name_to_model_state_file_name(file)
            assert model_state_file_name is not None
            model_state_keys = self.global_file_to_state_dict_keys_mapping[model_state_file_name]
            optimizer_state_keys = self.global_file_to_state_dict_keys_mapping[file]

            master_weight_name_to_model_weight_name_mapping = {}
            for i in range(len(model_state_keys)):
                master_weight_name = self.optimizer_state_name_to_master_weight_name(optimizer_state_keys[i])
                master_weight_name_to_model_weight_name_mapping[master_weight_name] = model_state_keys[i]

            state_dict = self.cur_rank_loaded_state_dict[file_name]
            renamed_state_dict = {}
            (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
            for k, v in state_dict.items():
                master_weight_name = self.optimizer_state_name_to_master_weight_name(k)
                model_weight_name = master_weight_name_to_model_weight_name_mapping[master_weight_name]
                new_k = k.replace(master_weight_name, model_weight_name)
                renamed_state_dict[new_k] = v

            return renamed_state_dict
        else:
            return self.cur_rank_loaded_state_dict[file_name]

    def load_state_dict_and_rename(self):
        rank_access_files = {}
        if self.save_sharded_model:
            rank_access_files[self.cur_rank] = self.cur_rank_optimizer_state_file_names
        else:
            rank_access_files[self.cur_rank] = (
                self.cur_rank_model_state_file_names + self.cur_rank_optimizer_state_file_names
            )

        need_read_files = get_local_load_files(self.gather_global_object(rank_access_files))

        self.cur_rank_loaded_state_dict = {}

        for file in need_read_files:
            self.cur_rank_loaded_state_dict[file] = paddle.load(os.path.join(self.path, file))

        file_to_master_weights_keys = {}

        self.optimizer_state_with_master_weights = False

        for file, state_dict in self.cur_rank_loaded_state_dict.items():
            if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                state_dict.pop("LR_Scheduler")
                if "master_weights" in state_dict:
                    self.optimizer_state_with_master_weights = True
                    master_weights = state_dict.pop("master_weights")
                    file_to_master_weights_keys[file] = list(master_weights.keys())
                    for k, v in master_weights.items():
                        # In sharding stage3, ‘@slice’ will be added in front of the key for master_weight, which is removed here.
                        k = k.replace("slice@", "")
                        state_dict[k] = v

                # Standardize the state names of the AdamW optimizer.
                adamw_optimizer_param_suffix_name_mapping = {
                    ".w_0_fp32_master_0_moment1_0": ".w_0_moment1_0",
                    ".w_0_fp32_master_0_moment2_0": ".w_0_moment2_0",
                    ".w_0_fp32_master_0_beta1_pow_acc_0": ".w_0_beta1_pow_acc_0",
                    ".w_0_fp32_master_0_beta2_pow_acc_0": ".w_0_beta2_pow_acc_0",
                }

                unified_name_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k
                    for suffix in adamw_optimizer_param_suffix_name_mapping:
                        if k.endswith(suffix):
                            new_k = k.replace(suffix, adamw_optimizer_param_suffix_name_mapping[suffix])
                            break
                    unified_name_state_dict[new_k] = v

                self.cur_rank_loaded_state_dict[file] = unified_name_state_dict

        # After the rank has finished loading the files it needs, it can infer sharding_stage1_v and is_sharding_stage3.
        self.sharding_stage1_v = self.infer_sharding_stage1_v()
        self.is_sharding_stage3 = self.infer_is_sharding_stage3()

        # In sharding stage3, the parameters need to be reordered based on whether they are sliced.
        # The threshold for determining whether to slice is segment_size, with a default value of 2**20.
        # However, sharding stage3 allows users to specify their own unsliced layers, which seems to be incompatible here.
        if self.is_sharding_stage3:
            segment_size = 2**20
            for file, state_dict in self.cur_rank_loaded_state_dict.items():
                if file.endswith(MODEL_WEIGHT_SUFFIX):
                    sliced_pramaeters = []
                    unseliced_pramaeters = []
                    sorted_state_dict = {}
                    for k, v in state_dict.items():
                        if v.numel() > segment_size:
                            sliced_pramaeters.append(k)
                        else:
                            unseliced_pramaeters.append(k)
                    for k in sliced_pramaeters + unseliced_pramaeters:
                        sorted_state_dict[k] = state_dict.pop(k)
                    self.cur_rank_loaded_state_dict[file] = sorted_state_dict

        self.global_file_to_master_weights_keys = self.gather_global_object(file_to_master_weights_keys)

        # rename and record sharded_tensor_info
        cur_rank_sharded_tensor_infos = {}

        # 1. Handling the sharding stage1 v2 scenario, where the save_sharded_model flag must be enabled, independent of master_weights.
        if self.sharding_degree > 1 and self.sharding_stage1_v == 2 and not self.is_sharding_stage3:
            assert self.save_sharded_model
            for file, state_dict in self.cur_rank_loaded_state_dict.items():
                # The rule for renaming is to change the master_weights name in the optimizer state to the model weight name,
                # and then append the tp_degree.
                renamed_state_dict = self.rename_using_model_meta(file)
                (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                for new_k, v in renamed_state_dict.items():
                    if new_k not in cur_rank_sharded_tensor_infos:
                        cur_rank_sharded_tensor_infos[new_k] = [
                            [
                                {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                v.shape,
                                str(v.dtype).split(".")[1],
                                file,
                            ]
                        ]
                    else:
                        cur_rank_sharded_tensor_infos[new_k].append(
                            [
                                {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                v.shape,
                                str(v.dtype).split(".")[1],
                                file,
                            ]
                        )

                self.cur_rank_loaded_state_dict[file] = renamed_state_dict
        # 2. In handling the sharding stage1 v1 scenario, the optimizer states are distributed across different ranks.
        # We need to obtain the name mapping by simulating the partitioning method, without concern for the presence of master_weights.
        elif self.sharding_degree > 1 and self.sharding_stage1_v == 1 and not self.is_sharding_stage3:
            if not self.save_sharded_model:
                file_to_state_dict_shapes_mapping = {}
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    shapes = []
                    for k, v in state_dict.items():
                        shapes.append([k, v.shape])
                    file_to_state_dict_shapes_mapping[file] = shapes

                global_file_to_state_dict_shapes_mapping = self.gather_global_object(file_to_state_dict_shapes_mapping)

                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                    sharding_optimizer_state_shards = []
                    if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                        for k, v in global_file_to_state_dict_shapes_mapping.items():
                            (tp_rank_, pp_rank_, sharding_rank_) = self.get_distribution_rank_from_file_name(k)
                            if tp_rank == tp_rank_ and pp_rank == pp_rank_ and k.endswith(OPTIMIZER_WEIGHT_SUFFIX):
                                sharding_optimizer_state_shards.append([v, sharding_rank_])
                        model_state_file_name = self.optimizer_state_file_name_to_model_state_file_name(file)
                        model_state_shapes = global_file_to_state_dict_shapes_mapping[model_state_file_name]
                        sharding_optimizer_state_shards.sort(key=lambda x: x[1])

                        partition_result_0 = self.partition_parameters(model_state_shapes, False, self.sharding_degree)
                        partition_result_1 = self.partition_parameters(model_state_shapes, True, self.sharding_degree)

                        for k, v in partition_result_0.items():
                            v = sorted(v, key=model_state_shapes.index)
                            partition_result_0[k] = v

                        for k, v in partition_result_1.items():
                            v = sorted(v, key=model_state_shapes.index)
                            partition_result_1[k] = v

                        sharding_sort_parameters = False

                        for i in range(len(sharding_optimizer_state_shards)):
                            if not sharding_sort_parameters:
                                state_shard = sharding_optimizer_state_shards[i][0]
                                partitioned_shard = partition_result_0[i]
                                for j in range(len(partitioned_shard)):
                                    if partitioned_shard[j][1] != state_shard[j][1]:
                                        sharding_sort_parameters = True
                                        break

                        if sharding_sort_parameters:
                            for i in range(len(sharding_optimizer_state_shards)):
                                state_shard = sharding_optimizer_state_shards[i][0]
                                partitioned_shard = partition_result_1[i]
                                for j in range(len(partitioned_shard)):
                                    assert partitioned_shard[j][1] == state_shard[j][1]

                        if sharding_sort_parameters:
                            partition_result = partition_result_1
                        else:
                            partition_result = partition_result_0

                        master_weight_name_to_model_weight_name_mapping = {}
                        for i in range(len(sharding_optimizer_state_shards)):
                            state_shard = sharding_optimizer_state_shards[i][0]
                            partitioned_shard = partition_result[i]
                            for j in range(len(partitioned_shard)):
                                master_weight_name = self.optimizer_state_name_to_master_weight_name(state_shard[j][0])
                                master_weight_name_to_model_weight_name_mapping[
                                    master_weight_name
                                ] = partitioned_shard[j][0]

                        renamed_state_dict = {}
                        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)

                        # In this branch, sharding does not split the optimizer states; it merely relocates them to different cards.
                        # Therefore, the sharding information can now be directly removed.
                        for k, v in state_dict.items():
                            master_weight_name = self.optimizer_state_name_to_master_weight_name(k)
                            model_weight_name = master_weight_name_to_model_weight_name_mapping[master_weight_name]
                            new_k = k.replace(master_weight_name, model_weight_name)
                            renamed_state_dict[new_k] = v
                            if new_k not in cur_rank_sharded_tensor_infos:
                                cur_rank_sharded_tensor_infos[new_k] = [
                                    [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                                ]
                            else:
                                cur_rank_sharded_tensor_infos[new_k].append(
                                    [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                                )

                        self.cur_rank_loaded_state_dict[file] = renamed_state_dict
                    else:
                        for k, v in state_dict.items():
                            if k not in cur_rank_sharded_tensor_infos:
                                cur_rank_sharded_tensor_infos[k] = [
                                    [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                                ]
                            else:
                                cur_rank_sharded_tensor_infos[k].append(
                                    [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                                )
            else:
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    renamed_state_dict = self.rename_using_model_meta(file)
                    (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                    for new_k, v in renamed_state_dict.items():
                        if new_k not in cur_rank_sharded_tensor_infos:
                            cur_rank_sharded_tensor_infos[new_k] = [
                                [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                            ]
                        else:
                            cur_rank_sharded_tensor_infos[new_k].append(
                                [{"tp_rank": tp_rank}, v.shape, str(v.dtype).split(".")[1], file]
                            )

                    self.cur_rank_loaded_state_dict[file] = renamed_state_dict
        else:
            # 3. Handling the case of disabling sharding, independent of master_weights, but without considering the save_sharded_model flag.
            if not self.save_sharded_model:
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                    if file.endswith(OPTIMIZER_WEIGHT_SUFFIX):

                        renamed_state_dict = self.rename_using_optimizer_state_order(file)

                        (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                        for new_k, v in renamed_state_dict.items():
                            if new_k not in cur_rank_sharded_tensor_infos:
                                cur_rank_sharded_tensor_infos[new_k] = [
                                    [
                                        {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                        v.shape,
                                        str(v.dtype).split(".")[1],
                                        file,
                                    ]
                                ]
                            else:
                                cur_rank_sharded_tensor_infos[new_k].append(
                                    [
                                        {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                        v.shape,
                                        str(v.dtype).split(".")[1],
                                        file,
                                    ]
                                )

                        self.cur_rank_loaded_state_dict[file] = renamed_state_dict
                    else:
                        for k, v in state_dict.items():
                            if k not in cur_rank_sharded_tensor_infos:
                                cur_rank_sharded_tensor_infos[k] = [
                                    [
                                        {"tp_rank": tp_rank, "sharding_rank": -1},
                                        v.shape,
                                        str(v.dtype).split(".")[1],
                                        file,
                                    ]
                                ]
                            else:
                                cur_rank_sharded_tensor_infos[k].append(
                                    [
                                        {"tp_rank": tp_rank, "sharding_rank": -1},
                                        v.shape,
                                        str(v.dtype).split(".")[1],
                                        file,
                                    ]
                                )

            else:
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    # The rule for renaming is to change the master_weights name in the optimizer state to the model weight name,
                    # and then append the tp_degree.
                    renamed_state_dict = self.rename_using_model_meta(file)
                    (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file)
                    for new_k, v in renamed_state_dict.items():
                        if new_k not in cur_rank_sharded_tensor_infos:
                            cur_rank_sharded_tensor_infos[new_k] = [
                                [
                                    {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                    v.shape,
                                    str(v.dtype).split(".")[1],
                                    file,
                                ]
                            ]
                        else:
                            cur_rank_sharded_tensor_infos[new_k].append(
                                [
                                    {"tp_rank": tp_rank, "sharding_rank": sharding_rank},
                                    v.shape,
                                    str(v.dtype).split(".")[1],
                                    file,
                                ]
                            )

                    self.cur_rank_loaded_state_dict[file] = renamed_state_dict
        # gather global sharded tensor infos
        sharded_tensor_infos = self.gather_global_object({self.cur_rank: cur_rank_sharded_tensor_infos})

        self.global_sharded_tensor_infos = {}
        for rank, sharded_tensor_info in sharded_tensor_infos.items():
            for k, v in sharded_tensor_info.items():
                if k not in self.global_sharded_tensor_infos:
                    self.global_sharded_tensor_infos[k] = v
                else:
                    self.global_sharded_tensor_infos[k] += v

    def gen_metadata_for_tp_sharded_tensor(self):
        for k, v in self.global_sharded_tensor_infos.items():
            v.sort(key=lambda x: x[0]["tp_rank"])

        state_dict_metadata = {}
        storage_metadata = {}
        # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
        for k, v in self.global_sharded_tensor_infos.items():
            global_offset = 0
            local_shape = v[0][1]
            model_state_name = self.optimizer_key_to_model_state_key(k)
            if "_pow_acc_0" not in k:
                global_shape = self.model_state_global_shape[model_state_name]
            else:
                global_shape = (1,)

            assert len(local_shape) == len(global_shape)
            axis = -1
            for i in range(len(local_shape)):
                if local_shape[i] != global_shape[i]:
                    axis = i
                    break

            is_replicated = axis == -1
            global_offset = [0] * len(local_shape)

            if is_replicated:
                v = [v[0]]

            for item in v:
                local_tensor_meta_data = LocalTensorMetadata(tuple(global_offset), item[1], item[2])
                local_tensor_index = LocalTensorIndex(k, tuple(global_offset))
                global_offset[axis] += item[1][axis]
                if k not in state_dict_metadata:
                    state_dict_metadata[k] = [local_tensor_meta_data]
                else:
                    state_dict_metadata[k].append(local_tensor_meta_data)
                storage_metadata[local_tensor_index] = item[3]

            metadata = Metadata(state_dict_metadata, storage_metadata, None)
            source_state_dict = self.cur_rank_loaded_state_dict

        return metadata, source_state_dict

    def gen_metadata_and_prepare_source_state_dict(self):
        self.load_state_dict_and_rename()
        if self.sharding_degree > 1 and self.sharding_stage1_v == 2 and not self.is_sharding_stage3:
            for k, v in self.global_sharded_tensor_infos.items():
                v.sort(key=lambda x: x[0]["sharding_rank"])

            state_dict_metadata = {}
            storage_metadata = {}
            # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
            for k, v in self.global_sharded_tensor_infos.items():
                global_offset = [0] * self.tp_degree
                for item in v:
                    tp_rank = item[0]["tp_rank"]
                    k_with_tp_rank = k + "_tp" + "{:02d}".format(tp_rank)
                    local_tensor_meta_data = LocalTensorMetadata((global_offset[tp_rank],), item[1], item[2])
                    local_tensor_index = LocalTensorIndex(k_with_tp_rank, (global_offset[tp_rank],))
                    global_offset[tp_rank] += item[1][0]
                    if k_with_tp_rank not in state_dict_metadata:
                        state_dict_metadata[k_with_tp_rank] = [local_tensor_meta_data]
                    else:
                        state_dict_metadata[k_with_tp_rank].append(local_tensor_meta_data)
                    storage_metadata[local_tensor_index] = item[3]

            metadata_for_merge_sharding = Metadata(state_dict_metadata, storage_metadata, None)

            source_state_dict_for_merge_sharding = {}
            for file_name, state_dict in self.cur_rank_loaded_state_dict.items():
                renamed_state_dict = {}
                (tp_rank, pp_rank, sharding_rank) = self.get_distribution_rank_from_file_name(file_name)
                for k, v in state_dict.items():
                    if self.global_sharded_tensor_infos[k][0][0]["tp_rank"] != -1:
                        k_with_tp_rank = k + "_tp" + "{:02d}".format(tp_rank)
                        renamed_state_dict[k_with_tp_rank] = v
                    else:
                        renamed_state_dict[k] = v

                source_state_dict_for_merge_sharding[file_name] = renamed_state_dict

            assert self.model_meta is not None
            global_model_state_shapes = []
            sharding_metas_keys = []
            for i in range(self.pp_degree):
                for j in range(self.tp_degree):
                    sharding_metas_keys.append("tp{:02d}_pp{:02d}".format(i, j))
            for key in sharding_metas_keys:
                param_meta = self.model_meta["sharding_metas"][key]["param_meta"]
                for k, v in param_meta.items():
                    global_model_state_shapes.append([k, v[0]])

            # Distribute all model parameters evenly across each card for loading

            world_size = paddle.distributed.get_world_size()

            partition_mapping = self.partition_parameters(global_model_state_shapes, True, world_size)

            partition_model_state_keys = []
            for cur_rank, partition_model_state in partition_mapping.items():
                partition_model_state_keys.append([item[0] for item in partition_model_state])

            param_meta = {}
            for i in range(self.tp_degree):
                for j in range(self.pp_degree):
                    key = "tp{:02d}_pp{:02d}".format(i, j)
                    pm = self.model_meta["sharding_metas"][key]["param_meta"]
                    for k, v in pm.items():
                        param_meta[k] = v

            param_flattened_shapes = {}
            for k, v in param_meta.items():
                param_flattened_shapes[k] = reduce(lambda x, y: x * y, v[0])

            cur_rank_need_load_model_state_keys = partition_model_state_keys[self.cur_rank]

            # Generate the optimizer states corresponding to the model weights.
            optimizer_state_dict = {}
            for key in cur_rank_need_load_model_state_keys:
                for tp_rank in range(self.tp_degree):
                    tp_rank_suffix = "_tp{:02d}".format(tp_rank)
                    optimizer_state_dict[key + ".w_0_moment1_0" + tp_rank_suffix] = paddle.zeros(
                        (param_flattened_shapes[key],), "float32"
                    )
                    optimizer_state_dict[key + ".w_0_moment2_0" + tp_rank_suffix] = paddle.zeros(
                        (param_flattened_shapes[key],), "float32"
                    )
                    if self.optimizer_state_with_master_weights:
                        optimizer_state_dict[key + ".w_0" + tp_rank_suffix] = paddle.zeros(
                            (param_flattened_shapes[key],), "float32"
                        )
                    # When handling tensor parallelism (TP), if some tensors are replicated, we initially assume that they are partitioned.
                    # Later, when these are compared with the global shape, we realize that they are replicated.

                    optimizer_state_dict[key + ".w_0_beta1_pow_acc_0" + tp_rank_suffix] = paddle.zeros((1,), "float32")
                    optimizer_state_dict[key + ".w_0_beta2_pow_acc_0" + tp_rank_suffix] = paddle.zeros((1,), "float32")

            # merge sharding
            _load_state_dict(optimizer_state_dict, source_state_dict_for_merge_sharding, [metadata_for_merge_sharding])

            # Reshape
            for k, v in optimizer_state_dict.items():
                if v.shape[0] > 1 and "_tp" in k:
                    param_name = self.optimizer_key_to_model_state_key(k[:-5])
                    param_shape = param_meta[param_name][0]
                    assert v.numel() == reduce(lambda x, y: x * y, param_shape)
                    reshaped_v = v.reshape(param_shape)
                    optimizer_state_dict[k] = reshaped_v
            concat_optimier_state_dict = {}

            optimizer_state_key_to_tp_keys = {}
            for key in optimizer_state_dict.keys():
                # Count how each key is split into keys ending with ‘_tpXX’.
                # optimizer_state_key_to_tp_keys ： {key:[key_tp00,key_tp01]}
                key_removed_tp_rank = key[:-5]
                if key_removed_tp_rank not in optimizer_state_key_to_tp_keys:
                    optimizer_state_key_to_tp_keys[key_removed_tp_rank] = [key]
                else:
                    optimizer_state_key_to_tp_keys[key_removed_tp_rank].append(key)

            for key, value in optimizer_state_key_to_tp_keys.items():
                value.sort(key=lambda x: int(x[-2:]))

            for key, tp_keys in optimizer_state_key_to_tp_keys.items():
                model_state_name = self.optimizer_key_to_model_state_key(key)
                local_shape = optimizer_state_dict[tp_keys[0]].shape
                if "_pow_acc_0" not in key:
                    global_shape = self.model_state_global_shape[model_state_name]
                else:
                    global_shape = (1,)

                assert len(local_shape) == len(global_shape)

                axis = -1
                for i in range(len(local_shape)):
                    if local_shape[i] != global_shape[i]:
                        axis = i
                        break

                is_replicated = axis == -1
                tp_tensors = []
                for tp_key in tp_keys:
                    tp_tensors.append(optimizer_state_dict[tp_key])

                if not is_replicated:
                    # Derive the partition strategy based on the global_shape, then concatenate.
                    concat_optimier_state_dict[key] = paddle.concat(tp_tensors, axis=axis)
                else:
                    concat_optimier_state_dict[key] = tp_tensors[0]

            fake_file_name = "{:02d}".format(self.cur_rank) + ".distcp"
            local_tensor_meta_data = {}
            local_tensor_index = {}
            for k, v in concat_optimier_state_dict.items():
                # Generate metadata.
                local_shape = v.shape
                global_offset = tuple([0] * len(local_shape))
                dtype = str(v.dtype).split(".")[1]
                local_tensor_meta_data[k] = LocalTensorMetadata(global_offset, local_shape, dtype)
                local_tensor_index[k] = [LocalTensorIndex(k, global_offset), fake_file_name]

            global_local_tensor_meta_data = []
            global_local_tensor_index = []

            use_dist = True if paddle.distributed.get_world_size() > 1 else False

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
            source_state_dict = {fake_file_name: concat_optimier_state_dict}

            return meta_data, source_state_dict

        elif self.sharding_degree > 1 and self.sharding_stage1_v == 1 and not self.is_sharding_stage3:
            return self.gen_metadata_for_tp_sharded_tensor()
        else:
            if self.is_sharding_stage3:
                for k, v in self.global_sharded_tensor_infos.items():
                    v.sort(key=lambda x: x[0]["sharding_rank"])

                state_dict_metadata = {}
                storage_metadata = {}
                # After obtaining the local_shape and sharding rank of each tensor, the global offset of each tensor can be calculated.
                for k, v in self.global_sharded_tensor_infos.items():
                    global_offset = 0
                    for item in v:
                        local_tensor_meta_data = LocalTensorMetadata((global_offset,), item[1], item[2])
                        local_tensor_index = LocalTensorIndex(k, (global_offset,))
                        global_offset += item[1][0]
                        if k not in state_dict_metadata:
                            state_dict_metadata[k] = [local_tensor_meta_data]
                        else:
                            state_dict_metadata[k].append(local_tensor_meta_data)
                        storage_metadata[local_tensor_index] = item[3]

                metadata_for_merge_sharding = Metadata(state_dict_metadata, storage_metadata, None)

                model_state_shapes = []
                dtype = ""
                for file, state_dict in self.cur_rank_loaded_state_dict.items():
                    if file.endswith(MODEL_WEIGHT_SUFFIX):
                        for k, v in state_dict.items():
                            model_state_shapes.append([k, v.shape])
                            dtype = str(v.dtype).split(".")[1]

                dtypes = self.gather_global_object([dtype])
                for dtype_s in dtypes:
                    if len(dtype_s) > 0:
                        dtype = dtype_s

                assert len(dtype) > 0

                global_model_state_shapes = self.gather_global_object(model_state_shapes)

                partition_result = self.partition_parameters(
                    global_model_state_shapes, True, paddle.distributed.get_world_size()
                )

                cur_rank_merger_model_params = partition_result[self.cur_rank]
                target_state_dict = {}
                for item in cur_rank_merger_model_params:
                    key = item[0]
                    shape = item[1]
                    flatten_shape = reduce(lambda a, b: a * b, item[1])
                    target_state_dict[key] = paddle.zeros(shape, dtype)
                    target_state_dict[key + ".w_0_moment1_0"] = paddle.zeros((flatten_shape,), "float32")
                    target_state_dict[key + ".w_0_moment2_0"] = paddle.zeros((flatten_shape,), "float32")
                    if self.optimizer_state_with_master_weights:
                        target_state_dict[key + ".w_0"] = paddle.zeros((flatten_shape,), "float32")
                    # When handling tensor parallelism (TP), if some tensors are replicated, we initially assume that they are partitioned.
                    # Later, when these are compared with the global shape, we realize that they are replicated.

                    target_state_dict[key + ".w_0_beta1_pow_acc_0"] = paddle.zeros((1,), "float32")
                    target_state_dict[key + ".w_0_beta2_pow_acc_0"] = paddle.zeros((1,), "float32")

                # TODO(zhuxinming) To resolve hanging during the loading of weights in sharding stage 3.
                _load_state_dict(target_state_dict, self.cur_rank_loaded_state_dict, [metadata_for_merge_sharding])

            else:
                return self.gen_metadata_for_tp_sharded_tensor()

    def rename_semi_auto_state_dict(self):
        need_remove_key_pattern = ["eager_tmp", "learning_rate", "@GRAD@MERG", "gradient_merge_"]

        need_remove_key = set()
        for key in self.semi_auto_model_state.keys():
            for pattern in need_remove_key_pattern:
                if pattern in key:
                    need_remove_key.add(key)
                    break

        for key in need_remove_key:
            self.semi_auto_model_state.pop(key)

        adamw_optimizer_status_name_suffix_mappings = {
            "_fp32_master_1_moment1_0": ".w_0_moment1_0",
            "_fp32_master_1_moment2_0": ".w_0_moment2_0",
            "_fp32_master_1_beta1_pow_acc_0": ".w_0_beta1_pow_acc_0",
            "_fp32_master_1_beta2_pow_acc_0": ".w_0_beta2_pow_acc_0",
            "_fp32_master_1": ".w_0",
            "_moment1_0": ".w_0_moment1_0",
            "_moment2_0": ".w_0_moment2_0",
            "_beta1_pow_acc_0": ".w_0_beta1_pow_acc_0",
            "_beta2_pow_acc_0": ".w_0_beta2_pow_acc_0",
        }

        def rename(old_name, map1, map2):
            for i in range(1, len(old_name)):
                str1 = old_name[:i]
                str2 = old_name[i:]
                if (str1 in map1) and (str2 in map2):
                    transformed_str1 = map1[str1]
                    transformed_str2 = map2[str2]
                    return transformed_str1 + transformed_str2
            return None

        renamed_state_dict = {}
        for key, value in self.semi_auto_model_state.items():
            if key in self.parameter_to_structured_name.values():
                new_name = key
            else:
                new_name = rename(key, self.parameter_to_structured_name, adamw_optimizer_status_name_suffix_mappings)
            assert new_name is not None
            renamed_state_dict[new_name] = value

        self.semi_auto_model_state = renamed_state_dict

    def load_from_dynamic_checkpoint(self):
        self.rename_semi_auto_state_dict()
        metadata, source_state_dict = self.gen_metadata_and_prepare_source_state_dict()
        if self.save_sharded_model:
            model_params = {}
            for k, v in self.semi_auto_model_state.items():
                if k in self.parameter_to_structured_name.values():
                    model_params[k] = v
            for k in model_params.keys():
                self.semi_auto_model_state.pop(k)

            appended_master_weight_names = []

            for k, v in model_params.items():
                master_weight = k + ".w_0"
                if master_weight not in self.semi_auto_model_state:
                    appended_master_weight_names.append(master_weight)
                    tmp_tensor = paddle.zeros(v.shape, "float32")
                    dist_tmp_tensor = dist.shard_tensor(tmp_tensor, v.process_mesh, v.placements)
                    self.semi_auto_model_state[master_weight] = dist_tmp_tensor

            _load_state_dict(self.semi_auto_model_state, source_state_dict, [metadata])
            for k, v in model_params.items():
                master_weight = self.semi_auto_model_state[k + ".w_0"]
                cast_master_weight = paddle.cast(master_weight._local_value(), "bfloat16")
                paddle.assign(cast_master_weight, v._local_value())
            for k in appended_master_weight_names:
                self.semi_auto_model_state.pop(k)

        else:
            _load_state_dict(self.semi_auto_model_state, source_state_dict, [metadata])
