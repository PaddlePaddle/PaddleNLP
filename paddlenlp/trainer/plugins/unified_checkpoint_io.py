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
import gc
import json
import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from tqdm.auto import tqdm

from paddlenlp.trainer.trainer_utils import get_last_checkpoint
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    _load_state_dict_into_model,
    get_parameter_dtype,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import (
    device_guard,
    dtype_byte_size,
    get_checkpoint_shard_files,
)
from paddlenlp.utils.distributed import distributed_gather
from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger

local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))


class UnifiedCkptIO:
    def __init__(self, args, hcg=None):
        self.args = args
        self.tp_group = None
        self.hcg = hcg
        if self.hcg is None and paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.tp_group = self.hcg.get_model_parallel_group()

    def manipulate_state_dict_and_config(self, model_to_save, weight_name_suffix, safe_serialization=False):
        state_dict = model_to_save.state_dict()
        if self.args.use_hybrid_parallel:
            all_filter_keys = self.filter_params(model_to_save, state_dict, self.tp_group.rank)
            dtype = get_parameter_dtype(model_to_save)
            assert hasattr(model_to_save, "config")
            model_to_save.config.dtype = str(dtype).split(".")[1]
            config_to_save = copy.deepcopy(model_to_save.config)
            if config_to_save.tensor_parallel_degree > 1:
                state_dict = self.merge_tensor_parallel_with_shard(
                    model_to_save, state_dict, config_to_save, all_filter_keys
                )
                # do we need to change?
                config_to_save.tensor_parallel_degree = 1

            # build index json file
            self.index_file_list = []
            self.total_size_list = []
            index_weight_file = {}
            total_size = 0
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
            weights_name = _add_variant(weights_name, weight_name_suffix)
            for key, weight in state_dict.items():
                index_weight_file[key] = weights_name
                total_size += weight.numel() * dtype_byte_size(weight.dtype)
            total_size = paddle.to_tensor(total_size)
            data_group = self.hcg.get_data_parallel_group()
            if data_group.rank == -1:
                dist.all_gather_object(self.index_file_list, index_weight_file)
                dist.all_gather(self.total_size_list, total_size)
            else:
                dist.all_gather_object(self.index_file_list, index_weight_file, group=data_group)
                dist.all_gather(self.total_size_list, total_size, group=data_group)

        return state_dict, config_to_save

    def save_sharded_index(self, output_dir, safe_serialization=False):
        # save index json file
        if local_rank == 0:
            sharded_index_json = {}
            final_dict = self.index_file_list[0]
            for i, index_file in enumerate(self.index_file_list):
                if i == 0:
                    continue
                final_dict.update(self.index_file_list[i])
            sharded_index_json["weight_map"] = final_dict

            self.total_size_list = [i.item() for i in self.total_size_list]
            sharded_index_json["metadata"] = {"total_size": sum(self.total_size_list)}

            if not safe_serialization:
                path = os.path.join(output_dir, PADDLE_WEIGHTS_INDEX_NAME)
            else:
                path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
            with open(path, "w") as f:
                json.dump(sharded_index_json, f, indent=4)

    def filter_params(self, model_to_save, state_dict, tp_rank=0):
        logger.info("filter params for different workers to save.")

        tp_size = self.tp_group.nranks
        filter_tensor_list = [[] for i in range(tp_size)]
        if tp_rank == 0:
            tp_actions = model_to_save._get_tensor_parallel_mappings(model_to_save.config, is_split=False)
            state_keys_map = model_to_save._resolve_prefix_keys(
                tp_actions.keys(), state_dict.keys(), ignore_error=True
            )
            for k, v in state_keys_map.items():
                tp_actions[v] = tp_actions.pop(k)

            tensor_bytes_dict = {}
            for (k, v) in state_dict.items():
                if k in tp_actions:
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

    def merge_tensor_parallel_with_shard(self, model_to_save, state_dict, config, all_filter_keys):
        tp_actions = model_to_save._get_tensor_parallel_mappings(config, is_split=False)
        state_keys_map = model_to_save._resolve_prefix_keys(tp_actions.keys(), state_dict.keys(), ignore_error=True)

        for k, v in state_keys_map.items():
            tp_actions[v] = tp_actions.pop(k)

        state_dict_to_save = {}

        for i, filter_keys in enumerate(all_filter_keys):
            is_dst = paddle.distributed.get_rank(self.tp_group) == i
            for key in filter_keys:
                tensor = state_dict[key]
                if key in tp_actions:
                    ret = distributed_gather(tensor, dst=i, group=self.tp_group, offload=True)
                    action = tp_actions.pop(key)
                    tensor = action(ret) if is_dst else None
                else:
                    tensor = tensor.cpu().numpy() if is_dst else None

                # keep state dict use paddle.tensor
                if isinstance(tensor, np.ndarray):
                    with device_guard("cpu"):
                        tensor = paddle.Tensor(tensor, zero_copy=True)

                if is_dst:
                    state_dict_to_save[key] = tensor

        if len(tp_actions) > 0:
            for x in tp_actions.keys():
                logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

        return state_dict_to_save

    def save_unified_checkpoint(self, model, output_dir, state_dict=None):
        if not isinstance(model, PretrainedModel):
            if isinstance(unwrap_model(model), PretrainedModel):
                config_to_save = None
                state_dict, config_to_save = self.manipulate_state_dict_and_config(
                    unwrap_model(model),
                    self.args.weight_name_suffix,
                    safe_serialization=False,
                )
                unwrap_model(model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    config_to_save=config_to_save,
                    variant=self.args.weight_name_suffix,
                    is_main_process=self.args.should_save,
                    safe_serialization=False,
                )
            else:
                logger.info("Trainer.model is not a `PretrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                paddle.save(
                    state_dict,
                    os.path.join(output_dir, _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix)),
                )
        else:
            if isinstance(model, PretrainedModel):
                config_to_save = None
                state_dict, config_to_save = self.manipulate_state_dict_and_config(
                    model,
                    self.args.weight_name_suffix,
                    safe_serialization=False,
                )
                model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    config_to_save=config_to_save,
                    variant=self.args.weight_name_suffix,
                    is_main_process=self.args.should_save,
                    safe_serialization=False,
                )
        self.save_sharded_index(output_dir, safe_serialization=False)

    def load_unified_checkpoint(self, model, resume_from_checkpoint=None, safe_serialization=False):
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        index_filename = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename),
        )

        loaded_keys = sharded_metadata["all_checkpoint_keys"]
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())

        def _remove_unused_keys(
            state_dict,
            model_state_dict,
        ):
            unused_keys = set(state_dict.keys()) - set(model_state_dict.keys())
            for unused_key in unused_keys:
                del state_dict[unused_key]
            return unused_keys

        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

        for shard_file in resolved_archive_file:
            pre_tensor_parallel_split = False
            if shard_file.endswith(".safetensors") and model.config.tensor_parallel_degree > 1:
                pre_tensor_parallel_split = True
                assert loaded_keys is not None, "loaded_keys is not None."
                tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys)
            # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
            state_dict = load_state_dict(
                shard_file, tp_actions if pre_tensor_parallel_split else None, set(expected_keys)
            )

            _ = _remove_unused_keys(state_dict, model_state_dict)
            if model.config.tensor_parallel_degree > 1 and not pre_tensor_parallel_split:
                logger.info("Converting state_dict to Tensor Parallel Format")
                # ignore error for multi shard, since only parts of data
                state_dict = model.convert_tensor_parallel(
                    None, model.config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
                )
            error_msgs += _load_state_dict_into_model(model, state_dict, "")

            # force memory release
            del state_dict
            gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
