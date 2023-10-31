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
from paddle.distributed import fleet
from tqdm.auto import tqdm

from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _load_state_dict_into_model,
    get_parameter_dtype,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import (
    device_guard,
    dtype_byte_size,
    get_checkpoint_shard_files,
    is_safetensors_available,
)
from paddlenlp.utils.distributed import distributed_gather
from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger

if is_safetensors_available():

    # from safetensors.numpy import load_file as safe_load_file
    from safetensors.numpy import save_file as safe_save_file


__all__ = [
    "load_unified_checkpoint",
    "save_unified_checkpoint",
]


def save_unified_checkpoint(args, model, output_dir, safe_serialization=False):
    """save unified checkpoint

    Args:
        args (TrainingArguments): Training Arguments
        model (PretrainedModel): model to save
        output_dir (str): save dir
        safe_serialization (bool, optional): use safetensors. Defaults to False.

    Raises:
        ValueError: if model is not an instance of `PretrainedModel` and the model cannot be saved
    """
    if isinstance(model, PretrainedModel):
        model_to_save = model
    elif isinstance(unwrap_model(model), PretrainedModel):
        model_to_save = unwrap_model(model)
    else:
        raise ValueError("Unified checkpoint only supports PretrainedModel")

    config_to_save = None
    state_dict, config_to_save, shard_file, sharded_index = unified_checkpoint_into_shards(
        args, model_to_save, safe_serialization=safe_serialization
    )

    save_directory = output_dir
    os.makedirs(save_directory, exist_ok=True)
    if safe_serialization:
        for k in list(state_dict.keys()):
            if isinstance(state_dict[k], paddle.Tensor):
                state_dict[k] = state_dict.pop(k).cpu().numpy()
        safe_save_file(state_dict, os.path.join(save_directory, shard_file), metadata={"format": "np"})
    else:
        paddle.save(state_dict, os.path.join(save_directory, shard_file))

    # Attach architecture to the config
    config_to_save.architectures = [model_to_save.__class__.__name__]
    # Save the config
    if args.should_save:
        config_to_save.save_pretrained(save_directory)

    if sharded_index is not None:
        if not safe_serialization:
            path = os.path.join(output_dir, PADDLE_WEIGHTS_INDEX_NAME)
        else:
            path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)

        with open(path, "w") as f:
            json.dump(sharded_index, f, indent=4)


def load_unified_checkpoint(model, resume_from_checkpoint: str, safe_serialization=False) -> None:
    """Load potential model checkpoint

    Args:
        model (PretrainedModel): Your model to load
        resume_from_checkpoint (str): path of the checkpoint to load

    Returns:
        None
    """

    index_filename = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME

    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )

    loaded_keys = sharded_metadata["all_checkpoint_keys"]

    model_state_dict = model.state_dict()
    expected_keys = set(list(model_state_dict.keys()))
    missing_keys = expected_keys - set(loaded_keys)

    if len(missing_keys) > 0:
        raise ValueError(f"missing_keys: {missing_keys}")

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
        # TODO: check if  no expected_keys in shard_file, then don't load it
        if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
            continue

        pre_tensor_parallel_split = False
        if shard_file.endswith(".safetensors") and model.config.tensor_parallel_degree > 1:
            pre_tensor_parallel_split = True
            assert loaded_keys is not None, "loaded_keys is not None."
            tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys, ignore_error=True)
        # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
        state_dict = load_state_dict(shard_file, tp_actions if pre_tensor_parallel_split else None, expected_keys)

        if not pre_tensor_parallel_split:
            # Since we load all keys but we only need one of pipeline stages
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


def unified_checkpoint_into_shards(
    args,
    model_to_save,
    safe_serialization=False,
):
    """Get state_dict and config to save

    Args:
        model_to_save (nn.Layer): model to, save
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.

    Returns:
        tuple: state_dict, config, shard_file: file name, sharded_index: map for weight to file name.
    """
    assert hasattr(model_to_save, "config")

    state_dict = model_to_save.state_dict()

    all_filter_keys = filter_params(model_to_save, state_dict)

    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.dtype = str(dtype).split(".")[1]
    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        state_dict = merge_tensor_parallel_with_shard(model_to_save, state_dict, config_to_save, all_filter_keys)

    if config_to_save.tensor_parallel_degree > 1:
        # do we need to change?
        config_to_save.tensor_parallel_degree = 1

    # build index json file
    index_file_list = []
    total_size_list = []
    index_weight_file = {}
    total_size = 0
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME

    # TODO: fix process_index
    shard_file = weights_name.replace(
        ".pdparams", f"-{args.process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.pdparams"
    )
    shard_file = shard_file.replace(
        ".safetensors", f"-{args.process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.safetensors"
    )

    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    hcg = fleet.get_hybrid_communicate_group()

    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    logger.info("Unified checkpoint generating sharded_index json files.")
    if tp_group.nranks > 1:
        paddle.distributed.all_gather_object(index_file_list, index_weight_file, tp_group)
        paddle.distributed.all_gather_object(total_size_list, total_size, tp_group)
    if pp_group.nranks > 1:
        pp_index_file_list = []
        pp_total_size_list = []
        paddle.distributed.all_gather_object(
            pp_index_file_list, index_file_list if len(index_file_list) > 0 else index_weight_file, pp_group
        )
        paddle.distributed.all_gather_object(
            pp_total_size_list, total_size_list if len(total_size_list) > 0 else total_size, pp_group
        )
        if isinstance(pp_total_size_list[0], list):
            total_size_list = [y for x in pp_total_size_list for y in x]
            index_file_list = [y for x in pp_index_file_list for y in x]
        else:
            index_file_list = pp_index_file_list
            total_size_list = pp_total_size_list

    # for pure sharding
    if len(index_file_list) == 0 and len(total_size_list) == 0:
        index_file_list = [index_weight_file]
        total_size_list = [total_size]

    sharded_index = get_sharded_index(
        index_file_list,
        total_size_list,
    )

    return state_dict, config_to_save, shard_file, sharded_index


def get_sharded_index(
    index_file_list,
    total_size_list,
):
    # save index json file
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    if local_rank == 0:
        sharded_index_json = {}

        sharded_index_json["metadata"] = {"total_size": sum(total_size_list)}

        weight_map = {}
        for i, index_file in enumerate(index_file_list):
            weight_map.update(index_file_list[i])

        sharded_index_json["weight_map"] = weight_map
        return sharded_index_json

    return None


def filter_params(model_to_save, state_dict):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # for pure sharding or pure pp
    if tp_size <= 1:
        return [list(state_dict.keys())]

    filter_tensor_list = [[] for i in range(tp_size)]

    if tp_rank == 0:
        tensor_bytes_dict = {}

        for (k, v) in state_dict.items():
            if hasattr(v, "is_distributed") and v.is_distributed:
                tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
            else:
                tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)

        filter_tensor_list = []
        current_block = []
        current_block_size = 0
        total_size = 0

        max_shard_size = (sum(tensor_bytes_dict.values()) + tp_size - 1) // tp_size

        for index, (key, weight_size) in enumerate(tensor_bytes_dict.items()):
            # If this weight is going to tip up over the maximal size, we split.
            # if current_block_size + weight_size > max_shard_size:
            if total_size + weight_size > max_shard_size * (len(filter_tensor_list) + 1) or (
                len(tensor_bytes_dict) - index < (tp_size - len(filter_tensor_list))
            ):
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    filter_tensor_list.append(current_block)
                current_block = []
                current_block_size = 0

            current_block.append(key)
            current_block_size += weight_size
            total_size += weight_size

        filter_tensor_list.append(current_block)
        assert len(filter_tensor_list) == tp_size, "Error, partition failed!"

    paddle.distributed.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    return filter_tensor_list


def merge_tensor_parallel_with_shard(model_to_save, state_dict, config, all_filter_keys):
    logger.info("Unified checkpoint merge tensor parallel in shards")
    tp_actions = model_to_save.get_tensor_parallel_convert_actions(
        model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
    )

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    # tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
    if hcg.get_pipe_parallel_group().nranks > 1:
        filter_keys = set([y for x in all_filter_keys for y in x])
        for key in list(tp_actions.keys()):
            if key not in filter_keys:
                tp_actions.pop(key)

    state_dict_to_save = {}
    for i, filter_keys in enumerate(all_filter_keys):
        is_dst = tp_rank == i
        for key in filter_keys:
            tensor = state_dict[key]
            if key in tp_actions:
                ret = distributed_gather(tensor, dst=i, group=tp_group, offload=True)
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
