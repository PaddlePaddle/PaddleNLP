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

import paddle
import paddle.distributed as dist
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
    dtype_byte_size,
    get_checkpoint_shard_files,
    get_optimizer_shard_files,
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
    from safetensors.numpy import save_file as safe_save_file


PADDLE_OPTIMIZER_NAME = "optimizer.pdopt"
PADDLE_OPTIMIZER_INDEX_NAME = "optimizer.pdopt.index.json"
SAFE_OPTIMIZER_NAME = "optimizer.safetensors"
SAFE_OPTIMIZER_INDEX_NAME = "optimizer.safetensors.index.json"
PADDLE_MASTER_WEIGHTS_NAME = "master_weights.pdparams"
PADDLE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.pdparams.index.json"
SAFE_MASTER_WEIGHTS_NAME = "master_weights.safetensors"
SAFE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.safetensors.index.json"


__all__ = [
    "load_unified_checkpoint",
    "save_unified_checkpoint",
    "load_unified_optimizer",
    "save_unified_optimizer",
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
        tp_actions = model_to_save.get_tensor_parallel_convert_actions(
            model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
        )
        state_dict = merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys)

    if config_to_save.tensor_parallel_degree > 1:
        # do we need to change?
        config_to_save.tensor_parallel_degree = 1

    # build index json file
    index_weight_file = {}
    total_size = 0
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME

    shard_file = get_sharded_file_name(args, weights_name)
    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_file_list, total_size_list = gather_sharded_object(index_weight_file, total_size)
    sharded_index = get_sharded_index(
        index_file_list,
        total_size_list,
    )

    return state_dict, config_to_save, shard_file, sharded_index


def save_unified_optimizer(args, model, optimizer, output_dir, safe_serialization=False):
    """save unified optimizer

    Args:
        args (TrainingArguments): Training Arguments
        optimizer (Optimizer): optimizer to save
        output_dir (str): Save directory.
        safe_serialization (bool, optional): Whether to use safetensors. Defaults to False.

    """
    # Split into naive optimizer params and master weights.
    results = unified_optimizer_into_shards(args, model, optimizer, safe_serialization=safe_serialization)
    master_weight_state_dict = None
    if len(results) == 1:
        optim_state_dict, shard_optim_file, sharded_optim_index = results
    else:
        optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
        master_weight_state_dict, shard_master_weight_file, sharded_master_weight_index = results[1]

    save_directory = output_dir
    os.makedirs(save_directory, exist_ok=True)

    if safe_serialization:
        for k in list(optim_state_dict.keys()):
            if isinstance(optim_state_dict[k], paddle.Tensor):
                optim_state_dict[k] = optim_state_dict.pop(k).cpu().numpy()
        safe_save_file(optim_state_dict, os.path.join(save_directory, shard_optim_file), metadata={"format": "np"})
        if master_weight_state_dict is not None:
            for k in list(master_weight_state_dict.keys()):
                if isinstance(master_weight_state_dict[k], paddle.Tensor):
                    master_weight_state_dict[k] = master_weight_state_dict.pop(k).cpu().numpy()
            safe_save_file(
                master_weight_state_dict,
                os.path.join(save_directory, shard_master_weight_file),
                metadata={"format": "np"},
            )
    else:
        paddle.save(optim_state_dict, os.path.join(save_directory, shard_optim_file))
        if master_weight_state_dict is not None:
            paddle.save(master_weight_state_dict, os.path.join(save_directory, shard_master_weight_file))

    if sharded_optim_index is not None:
        if not safe_serialization:
            path = os.path.join(output_dir, PADDLE_OPTIMIZER_INDEX_NAME)
            master_path = os.path.join(output_dir, PADDLE_MASTER_WEIGHTS_INDEX_NAME)
        else:
            path = os.path.join(output_dir, SAFE_OPTIMIZER_INDEX_NAME)
            master_path = os.path.join(output_dir, SAFE_MASTER_WEIGHTS_INDEX_NAME)

        with open(path, "w") as f:
            json.dump(sharded_optim_index, f, indent=4)

        if master_weight_state_dict is not None:
            with open(master_path, "w") as f:
                json.dump(sharded_master_weight_index, f, indent=4)


def load_unified_optimizer(model, optimizer, resume_from_checkpoint, safe_serialization=False):
    """Load potential model checkpoint

    Args:
        model (PretrainedModel): Your model to load
        resume_from_checkpoint (str): path of the checkpoint to load

    Returns:
        None
    """
    # init and get optimizer LR_Scheduler
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    if not safe_serialization:
        index_filename, index_filename_master_weights = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    model_state_dict = model.state_dict()
    model_keys = list(model_state_dict.keys())
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings
    expected_keys = []
    for key in list(sharded_metadata["all_optimizer_keys"]):
        key_name = key.split("/")
        if key_name[0] not in struct2static_name_mappings:
            # this process don't need to load this key
            continue
        expected_keys.append(key)
    expected_keys = set(expected_keys)

    loaded_keys = sharded_metadata["all_optimizer_keys"]
    missing_keys = expected_keys - set(loaded_keys)
    if len(missing_keys) > 0:
        raise ValueError(f"optimizer missing_keys: {missing_keys}")

    # This should always be a list but, just to be sure.
    if not isinstance(resolved_archive_file, list):
        resolved_archive_file = [resolved_archive_file]
    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading optimizer shards")

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}

        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename_master_weights),
        )
        expected_keys_mw = set(struct2static_name_mappings.keys())
        loaded_keys_mw = sharded_metadata_mw["all_optimizer_keys"]
        missing_keys_mw = expected_keys_mw - set(loaded_keys_mw)
        if len(missing_keys_mw) > 0:
            raise ValueError(f"optimizer missing_master_weights_keys: {missing_keys_mw}")

        if not isinstance(resolved_archive_file_mw, list):
            resolved_archive_file_mw = [resolved_archive_file_mw]
        if len(resolved_archive_file_mw) > 1:
            resolved_archive_file_mw = tqdm(resolved_archive_file_mw, desc="Loading master weights shards")

    def load_resolved_archive_file(
        resolved_archive_file, sharded_metadata, loaded_keys, expected_keys, is_master_weight=False
    ):
        returned_state_dict = {}
        # load optimizer
        for shard_file in resolved_archive_file:
            # TODO: check if no expected_keys in shard_file, then don't load it
            if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                continue

            if shard_file.endswith(".safetensors"):
                # assert model_keys is not None, "model_keys is None." TODO: correct the assert
                if model.config.tensor_parallel_degree > 1:
                    tp_actions = model.get_tensor_parallel_convert_actions(model.config, model_keys, ignore_error=True)
                    if not is_master_weight:
                        tp_actions = mapping_optimizer_tp_actions(tp_actions, loaded_keys)

                    # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                    state_dict = load_state_dict(shard_file, tp_actions, expected_keys)
                else:
                    # for pipeline model, we don't need to use tp_actions
                    state_dict = load_state_dict(shard_file, None, expected_keys)

            returned_state_dict.update(state_dict)
            # force memory release
            del state_dict
            gc.collect()
        return returned_state_dict

    state_dict_optim = load_resolved_archive_file(resolved_archive_file, sharded_metadata, loaded_keys, expected_keys)
    if has_master_weights:
        state_dict_master_weight = load_resolved_archive_file(
            resolved_archive_file_mw, sharded_metadata_mw, loaded_keys_mw, expected_keys_mw, is_master_weight=True
        )

    # rename optimizer param
    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            key_name = "_".join([static_name, "fp32_master_0", key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        returned_optim_state_dict[key_name] = state_dict_optim[key]
        returned_optim_state_dict[key_name].name = key_name

    if has_master_weights:
        for key in list(state_dict_master_weight.keys()):
            static_name = struct2static_name_mappings[key]
            returned_optim_state_dict["master_weights"][static_name] = state_dict_master_weight[key]
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, "fp32_master_0"])

    returned_optim_state_dict = nested_copy_place(
        returned_optim_state_dict, place=paddle.framework._current_expected_place()
    )

    return returned_optim_state_dict


def unified_optimizer_into_shards(
    args,
    model,
    optimizer,
    safe_serialization=False,
):
    """Get optimizer state dict and master weight state dict.

    Args:
        optimizer (Optimizer): optimizer to save.
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.
    """
    optim_state_dict = nested_copy(optimizer.state_dict())
    master_weights = None
    if "master_weights" in optim_state_dict.keys():
        master_weights = optim_state_dict["master_weights"]
        optim_state_dict.pop("master_weights")
    if "LR_Scheduler" in optim_state_dict.keys():
        optim_state_dict.pop("LR_Scheduler")

    # get optimizer param mappings
    static2struct_name_mappings = {}
    for k, v in model.state_dict().items():
        static2struct_name_mappings[v.name] = k

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

    # filter optimizer param
    if master_weights is not None:
        filter_master_keys = filter_params(model, master_weights, is_optimizer=True)
    filter_optim_keys = filter_params(model, optim_state_dict, is_optimizer=True)

    tp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    tp_size = tp_group.nranks

    if tp_size > 1:
        # get tp_actions
        model_keys = []
        for key in optim_state_dict.keys():
            base_model_key = key.split("/")[0]
            if base_model_key not in model_keys:
                model_keys.append(base_model_key)
        tp_actions = model.get_tensor_parallel_convert_actions(
            model.config, model_keys, is_split=False, ignore_error=True
        )
        optim_state_dict = merge_tensor_parallel_for_optimizer(
            optim_state_dict,
            tp_actions,
            filter_optim_keys,
        )
        if master_weights is not None:
            master_weights = merge_tensor_parallel_for_optimizer(
                master_weights,
                tp_actions,
                filter_master_keys,
            )

    # build index json file
    index_optimizer_file, index_master_weight_file = {}, {}
    total_optim_size, total_master_weight_size = 0, 0
    optimizer_name = SAFE_OPTIMIZER_NAME if safe_serialization else PADDLE_OPTIMIZER_NAME
    master_weights_name = SAFE_MASTER_WEIGHTS_NAME if safe_serialization else PADDLE_MASTER_WEIGHTS_NAME
    shard_optimizer_file = get_sharded_file_name(args, optimizer_name, is_optimizer=True)
    shard_master_weight_file = get_sharded_file_name(args, master_weights_name, is_optimizer=True)

    for key, weight in optim_state_dict.items():
        index_optimizer_file[key] = shard_optimizer_file
        total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    if master_weights is not None:
        for key, weight in master_weights.items():
            index_master_weight_file[key] = shard_master_weight_file
            total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_optimizer_filelist, total_optim_size_list = gather_sharded_object(
        index_optimizer_file, total_optim_size, is_optimizer=True
    )
    sharded_optim_index = get_sharded_index(index_optimizer_filelist, total_optim_size_list)
    if master_weights is not None:
        index_master_weight_filelist, total_master_weight_size_list = gather_sharded_object(
            index_master_weight_file, total_master_weight_size, is_optimizer=True
        )
        sharded_master_weight_index = get_sharded_index(index_master_weight_filelist, total_master_weight_size_list)

    if sharded_optim_index is not None and master_weights is not None:
        sharded_optim_index["master_weights"] = True

    if master_weights is None:
        return (optim_state_dict, shard_optimizer_file, sharded_optim_index)
    else:
        return (optim_state_dict, shard_optimizer_file, sharded_optim_index), (
            master_weights,
            shard_master_weight_file,
            sharded_master_weight_index,
        )


def get_sharded_file_name(args, file_name, is_optimizer=False):
    if not is_optimizer:
        shard_file = file_name.replace(
            ".pdparams",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.pdparams",
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.safetensors",
        )
    else:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        shard_file = file_name.replace(
            ".pdparams", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdparams"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.safetensors",
        )
        shard_file = shard_file.replace(
            ".pdopt", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdopt"
        )
    return shard_file


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


def gather_sharded_object(index_file, total_size, is_optimizer=False):

    index_file_list, total_size_list = [], []

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    logger.info("Unified checkpoint generating sharded_index json files.")

    if tp_group.nranks > 1:
        dist.all_gather_object(index_file_list, index_file, tp_group)
        dist.all_gather_object(total_size_list, total_size, tp_group)
    if pp_group.nranks > 1:
        pp_index_file_list = []
        pp_total_size_list = []
        dist.all_gather_object(
            pp_index_file_list, index_file_list if len(index_file_list) > 0 else index_file, pp_group
        )
        dist.all_gather_object(
            pp_total_size_list, total_size_list if len(total_size_list) > 0 else total_size, pp_group
        )
        index_file_list = pp_index_file_list
        total_size_list = pp_total_size_list

    index_file_list = flatten_list(index_file_list)
    total_size_list = flatten_list(total_size_list)

    # for pure sharding
    if len(index_file_list) == 0 and len(total_size_list) == 0:
        index_file_list = [index_file]
        total_size_list = [total_size]
    if is_optimizer:
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_group.nranks > 1:
            sharding_index_file_list = []
            sharding_total_size_list = []
            dist.all_gather_object(sharding_index_file_list, index_file_list, sharding_group)
            dist.all_gather_object(sharding_total_size_list, total_size_list, sharding_group)
            index_file_list = flatten_list(sharding_index_file_list)
            total_size_list = flatten_list(sharding_total_size_list)

    return index_file_list, total_size_list


def generate_base_static_name(vname):
    # return base static name and specific type name, like [embedding_0.w_0, moment1_0]
    if "fp32_master_0" in vname:
        vname = vname.split("_fp32_master_0_")
        return vname[0], vname[1]
    else:
        vname = vname.split(".")
        a = vname[0] + "." + vname[1][:3]
        b = vname[1][4:]
        return a, b


def filter_params(model_to_save, state_dict, is_optimizer=False):
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
        model_state_dict = model_to_save.state_dict()
        for (k, v) in state_dict.items():
            model_v = model_state_dict[k.split("/")[0]] if is_optimizer else v
            if hasattr(model_v, "is_distributed") and model_v.is_distributed:
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

    dist.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    return filter_tensor_list


def merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys):
    logger.info("Unified checkpoint merge tensor parallel in shards")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
    if hcg.get_pipe_parallel_group().nranks > 1:
        filter_keys = set([y for x in all_filter_keys for y in x])
        for key in list(tp_actions.keys()):
            if key not in filter_keys:
                tp_actions.pop(key)

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            key = filter_keys[i]
            tensor = state_dict[key]
            if key in tp_actions:
                ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                action = tp_actions.pop(key)
                tensor = action(ret) if is_dst else None
            else:
                tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

            if is_dst:
                state_dict_to_save[key] = tensor

    if len(tp_actions) > 0:
        for x in tp_actions.keys():
            logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

    return state_dict_to_save


def merge_tensor_parallel_for_optimizer(state_dict, tp_actions, all_filter_keys):
    logger.info("Unified optimizer tensor parallel in shards")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            # get base model key
            model_key = filter_keys[i].split("/")[0]
            tensor = state_dict[filter_keys[i]]
            if model_key in tp_actions:
                # for example: beta1, beta2
                if tensor.numel().item() == 1:
                    tensor = (
                        tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None
                    )  # Need broadcast when loaded
                else:
                    ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                    action = tp_actions[model_key]
                    tensor = action(ret) if is_dst else None
            else:
                tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

            if is_dst:
                state_dict_to_save[filter_keys[i]] = tensor

    return state_dict_to_save


def mapping_optimizer_tp_actions(tp_actions, optimizer_loaded_keys):
    """# conert param.name to
    param.key/moment1_0
    or param.key/beta1_XXX
    or param.key/beta2_XXX
    Args:
        tp_actions (dict): dictionay of tensor parallel actions {key: action}
        optimizer_loaded_keys (list or set): [param.key1/moment1_0, param.key2/beta1_XXX, param.key3/beta2_XXX]
    Returns:
        dict: new dictionay of tensor parallel actions {key: action}
    """
    new_actions = {}
    for key in optimizer_loaded_keys:
        key_base = key.split("/")[0]
        if ("moment" in key.split("/")[1] or "velocity" in key.split("/")[1]) and key_base in tp_actions:
            new_actions[key] = tp_actions[key_base]
    return new_actions


def nested_copy(inputs):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy(inputs[key])
        return outputs
    return inputs


def nested_copy_place(inputs, place=None):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy_place(inputs[key], place)
        return outputs
    if isinstance(inputs, paddle.Tensor):
        inputs = inputs._copy_to(place, False)

    return inputs


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list
