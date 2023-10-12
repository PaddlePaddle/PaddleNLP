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
    """_summary_

    Args:
        args (_type_): _description_
        model (_type_): _description_
        output_dir (_type_): _description_
        safe_serialization (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
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


def load_unified_checkpoint(model: paddle.nn.Layer, resume_from_checkpoint: str, safe_serialization=False) -> None:
    """Load potential model checkpoint

    Args:
        model (nn.Layer): _description_
        resume_from_checkpoint (str): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

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
            tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys, ignore_error=True)
        # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
        state_dict = load_state_dict(shard_file, tp_actions if pre_tensor_parallel_split else None, set(expected_keys))

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
        tuple: state_dict, and  config
    """
    assert hasattr(model_to_save, "config")

    state_dict = model_to_save.state_dict()

    all_filter_keys = filter_params(model_to_save, state_dict)

    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.dtype = str(dtype).split(".")[1]
    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        state_dict = merge_tensor_parallel_with_shard(model_to_save, state_dict, config_to_save, all_filter_keys)
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
        total_size += weight.numel() * dtype_byte_size(weight.dtype)

    total_size = paddle.to_tensor(total_size)

    hcg = fleet.get_hybrid_communicate_group()
    data_group = hcg.get_data_parallel_group()

    if data_group.rank == -1:
        paddle.distributed.all_gather_object(index_file_list, index_weight_file)
        paddle.distributed.all_gather(total_size_list, total_size)
    else:
        paddle.distributed.all_gather_object(index_file_list, index_weight_file, group=data_group)
        paddle.distributed.all_gather(total_size_list, total_size, group=data_group)

    print("index_file_list:\n", index_file_list)
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
        final_dict = index_file_list[0]

        total_size_list = [i.item() for i in total_size_list]
        sharded_index_json["metadata"] = {"total_size": sum(total_size_list)}

        for i, index_file in enumerate(index_file_list):
            if i == 0:
                continue
            final_dict.update(index_file_list[i])

        sharded_index_json["weight_map"] = dict(sorted(final_dict.items()))
        return sharded_index_json

    return None


def filter_params(model_to_save, state_dict):
    logger.info("filter params for different workers to save.")
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    filter_tensor_list = [[] for i in range(tp_size)]

    if tp_rank == 0:
        tp_actions = model_to_save.get_tensor_parallel_convert_actions(
            model_to_save.config,
            state_dict.keys(),
            is_split=False,
            ignore_error=True,
        )
        tensor_bytes_dict = {}

        for (k, v) in state_dict.items():
            if k in tp_actions:
                tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
                assert v.is_distributed, f"Tensor {k} shape: {v.shape} should be distrbuted tensor"
            else:
                tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)
                assert (
                    not hasattr(v, "is_distributed") or v.is_distributed is False
                ), f"Tensor {k} shape: {v.shape}, has no action for tensor merge"

        # TODO(ZHUI): Need better partion ways while keep the tensor order
        # Sort by tensor storage.
        tensor_bytes_dict = sorted(tensor_bytes_dict.items(), key=lambda x: x[1])
        keys_list = [key for key, byte in tensor_bytes_dict]

        # [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
        tensor_cnt = 0
        while tensor_cnt < len(state_dict):
            filter_tensor_list[tensor_cnt % tp_size].append(keys_list[tensor_cnt])
            tensor_cnt += 1

    paddle.distributed.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    return filter_tensor_list


def merge_tensor_parallel_with_shard(model_to_save, state_dict, config, all_filter_keys):
    tp_actions = model_to_save.get_tensor_parallel_convert_actions(
        model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
    )

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    # tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
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
