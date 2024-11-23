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
"""Support Sharding Stage1 V2(split param) for Unified Checkpoint"""

import gc
import os
from itertools import chain

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from tqdm.auto import tqdm

from paddlenlp.peft import LoRAModel, PrefixModelForCausalLM
from paddlenlp.transformers.model_utils import load_state_dict, unwrap_model
from paddlenlp.utils.env import (
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
)
from paddlenlp.utils.nested import nested_copy

from .utils import (
    FP32_MASTER,
    generate_base_static_name,
    get_expected_state_dict,
    get_optimizer_shard_files,
    mapping_optimizer_tp_actions,
)

__all__ = ["gather_splited_param_for_optimizer", "load_unified_optimizer_split_param"]


def merge_splited_param(
    state_dict, partial_tensor_list, param_shape_info, send_table, recv_table, is_master_weights=False
):
    """Merge the splited param in sharding group."""
    global_rank = dist.get_rank()
    for key in list(state_dict.keys()):
        if state_dict[key].numel().item() == 1:  # for example: beta1, beta2
            continue

        static_name = key if is_master_weights else generate_base_static_name(key)[0]
        shape, numel, index, padded_size = param_shape_info[static_name]
        if static_name not in partial_tensor_list:
            state_dict[key] = state_dict[key].reshape(shape)
            continue

        recv_rank = recv_table[static_name]
        send_info = send_table[static_name]

        base_padding_start = index + numel
        base_padding_end = index + padded_size

        if global_rank == recv_rank:
            tmp_tensor_list = []
            for send_rank, begin, end in send_info:
                padding_start = max(begin, base_padding_start)
                padding_end = min(end, base_padding_end)

                if send_rank == recv_rank:
                    tensor = (
                        state_dict[key] if padding_start >= padding_end else state_dict[key][: padding_start - begin]
                    )
                    tmp_tensor_list.append(tensor)
                else:
                    length = end - begin if padding_start >= padding_end else padding_start - begin
                    tmp_tensor = paddle.empty(shape=[length], dtype=state_dict[key].dtype)
                    dist.stream.recv(tmp_tensor, src=send_rank)
                    tmp_tensor_list.append(tmp_tensor)
            state_dict[key] = paddle.concat(tmp_tensor_list, axis=0).reshape(shape)
        else:
            for send_rank, begin, end in send_info:
                padding_start = max(begin, base_padding_start)
                padding_end = min(end, base_padding_end)
                if global_rank == send_rank:
                    tensor = (
                        state_dict[key] if padding_start >= padding_end else state_dict[key][: padding_start - begin]
                    )
                    dist.stream.send(tensor, dst=recv_rank)
                    state_dict.pop(key)
    return state_dict


def gather_splited_param_for_optimizer(optimizer):
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    global_rank = dist.get_rank()
    param_slice_info = {}
    param_shape_info = {}

    for buffer in optimizer._inner_opt._comm_buffer_list:
        for key in buffer._sharding_param_grad_view.keys():
            param_slice_info[key] = (
                buffer._sharding_param_grad_view[key]._param_begin,
                buffer._sharding_param_grad_view[key]._param_end,
            )
            param_shape_info[key] = (
                buffer._sharding_param_grad_view[key]._param.shape,
                buffer._sharding_param_grad_view[key]._param.numel().item(),
                buffer._sharding_param_grad_view[key]._index,
                buffer._sharding_param_grad_view[key]._padded_size,
            )
    param_slice_info["global_rank"] = global_rank
    param_slice_info_list = []
    dist.all_gather_object(param_slice_info_list, param_slice_info, group=sharding_group)

    optim_state_dict = nested_copy(optimizer.state_dict())
    master_weights = None
    if "master_weights" in optim_state_dict.keys():
        master_weights = optim_state_dict.pop("master_weights")
    if "LR_Scheduler" in optim_state_dict.keys():
        optim_state_dict.pop("LR_Scheduler")

    # deal with optimizer param
    partial_tensor_list = []
    for key in list(optim_state_dict.keys()):
        static_name, _ = generate_base_static_name(key)
        if static_name in param_slice_info.keys():
            if optim_state_dict[key].numel().item() == 1:  # for example: beta1, beta2
                continue
            begin, end = param_slice_info[static_name]
            shape, numel, _, _ = param_shape_info[static_name]
            if end - begin == numel:  # full tensor
                optim_state_dict[key] = optim_state_dict[key].reshape(shape)
            elif end <= begin:  # empty tensor
                continue
            else:  # partial tensor, end > begin but end - begin < numel
                partial_tensor_list.append(static_name)

    send_table = {}
    recv_table = {}
    for key in partial_tensor_list:
        sharding_ranklist = []
        for slice_info in param_slice_info_list:
            begin, end = slice_info[key]
            if end > begin:
                sharding_ranklist.append((slice_info["global_rank"], begin, end))
        recv_table[key] = sharding_ranklist[0][0]  # which sharding_rank to recv the splited tensor
        send_table[key] = [(rank, begin, end) for rank, begin, end in sharding_ranklist]

    merge_splited_param(optim_state_dict, partial_tensor_list, param_shape_info, send_table, recv_table, False)
    if master_weights is not None:
        merge_splited_param(master_weights, partial_tensor_list, param_shape_info, send_table, recv_table, True)
    return optim_state_dict, master_weights


def load_unified_optimizer_split_param(args, model, optimizer, resume_from_checkpoint):
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    typename_set = set()
    for key in sharded_metadata["weight_map"].keys():
        _, typename = key.split("/")
        typename_set.add(typename)

    model_state_dict = get_expected_state_dict(model)
    model_keys = list(model_state_dict.keys())
    static2struct_name_mappings = {v.name: k for k, v in model_state_dict.items()}  # get optimizer param mappings
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}

    expected_keys = []
    param_slice_info = {}
    param_shape_info = {}

    comm_buffer_list = optimizer._inner_opt._comm_buffer_list
    if hasattr(args, "enable_sharding_comm_overlap") and args.enable_sharding_comm_overlap:
        comm_buffer_list = list(chain(*model._chunk_2_comm_buffers.values()))
        model = unwrap_model(model)

    for buffer in comm_buffer_list:
        for key in buffer._sharding_param_grad_view.keys():
            begin = buffer._sharding_param_grad_view[key]._param_begin
            end = buffer._sharding_param_grad_view[key]._param_end
            if end > begin:
                expected_keys.append(key)
                shape = buffer._sharding_param_grad_view[key]._param.shape
                numel = buffer._sharding_param_grad_view[key]._param.numel().item()
                index = buffer._sharding_param_grad_view[key]._index
                padded_size = buffer._sharding_param_grad_view[key]._padded_size
                param_slice_info[key] = (begin, end)
                param_shape_info[key] = (shape, numel, index, padded_size)

    expected_keys = set([static2struct_name_mappings.get(name, None) for name in expected_keys])
    expected_keys_optim = []
    for key in expected_keys:
        for typename in typename_set:
            expected_keys_optim.append(f"{key}/{typename}")
    expected_keys_optim = set(expected_keys_optim)

    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading optimizer shards")

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}
        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename_master_weights),
        )
        if len(resolved_archive_file_mw) > 1:
            resolved_archive_file_mw = tqdm(resolved_archive_file_mw, desc="Loading master weights shards")

    def load_resolved_archive_file(resolved_archive_file, sharded_metadata, expected_keys, is_master_weights=False):
        returned_state_dict = {}

        if model.config.tensor_parallel_degree > 1:
            if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                tp_actions = model._get_tensor_parallel_convert_actions(model_keys, is_split=True, ignore_error=True)
            else:
                tp_actions = model.get_tensor_parallel_convert_actions(model.config, model_keys, ignore_error=True)
            if not is_master_weights:
                tp_actions = mapping_optimizer_tp_actions(tp_actions, expected_keys)

        for shard_file in resolved_archive_file:
            if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                continue
            if model.config.tensor_parallel_degree > 1:
                state_dict = load_state_dict(shard_file, tp_actions, expected_keys, device="cpu")
            else:
                state_dict = load_state_dict(shard_file, None, expected_keys, device="cpu")
            returned_state_dict.update(state_dict)
            del state_dict
            gc.collect()

        return returned_state_dict

    # get tp params
    state_dict_optim = load_resolved_archive_file(resolved_archive_file, sharded_metadata, expected_keys_optim)

    # need to split param for different sharding rank, maybe need to deal with oom issue.
    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        static_name = struct2static_name_mappings.get(key_name[0], None)

        if state_dict_optim[key].numel().item() > 1:
            begin, end = param_slice_info[static_name]
            shape, numel, index, padded_size = param_shape_info[static_name]
            state_dict_optim[key] = state_dict_optim[key].reshape([-1])
            state_dict_optim[key] = state_dict_optim[key][begin - index : end - index]

            padding_start = max(begin, index + numel)
            padding_end = min(end, index + padded_size)
            if padding_start < padding_end:
                state_dict_optim[key] = paddle.concat(
                    (
                        state_dict_optim[key],
                        paddle.zeros([padding_end - padding_start], dtype=state_dict_optim[key].dtype),
                    )
                )
        if has_master_weights:
            key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])

        state_dict_optim[key] = state_dict_optim[key]._copy_to(paddle.framework._current_expected_place(), False)

        returned_optim_state_dict[key_name] = state_dict_optim.pop(key)
        returned_optim_state_dict[key_name].name = key_name

    if has_master_weights:
        state_dict_master_weight = load_resolved_archive_file(
            resolved_archive_file_mw,
            sharded_metadata_mw,
            expected_keys,
            is_master_weights=True,
        )

        for key in list(state_dict_master_weight.keys()):
            static_name = struct2static_name_mappings.get(key, None)
            if state_dict_master_weight[key].numel().item() > 1:
                begin, end = param_slice_info[static_name]
                shape, numel, index, padded_size = param_shape_info[static_name]
                state_dict_master_weight[key] = state_dict_master_weight[key].reshape([-1])
                state_dict_master_weight[key] = state_dict_master_weight[key][begin - index : end - index]

                padding_start = max(begin, index + numel)
                padding_end = min(end, index + padded_size)
                if padding_start < padding_end:
                    state_dict_master_weight[key] = paddle.concat(
                        (
                            state_dict_master_weight[key],
                            paddle.zeros([padding_end - padding_start], dtype=state_dict_master_weight[key].dtype),
                        )
                    )
            state_dict_master_weight[key] = state_dict_master_weight[key]._copy_to(
                paddle.framework._current_expected_place(), False
            )
            returned_optim_state_dict["master_weights"][static_name] = state_dict_master_weight.pop(key)
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    return returned_optim_state_dict
