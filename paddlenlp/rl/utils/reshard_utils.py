# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from contextlib import contextmanager

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology
from paddle.distributed.fleet.base.topology import (
    CommunicateTopology,
    HybridCommunicateGroup,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker

from paddlenlp.transformers.model_utils import unwrap_model
from paddlenlp.utils.log import logger


@contextmanager
def init_rollout_env(tensor_parallel_degree, seed=100):
    """
    Initialize the rollout environment for parallel training.

    Args:
        tensor_parallel_degree (int): Tensor parallel degree, indicating how many GPUs the model is distributed across.
        seed (int, optional): Random seed, defaults to 100.

    Returns:
        ContextManager: A context manager for controlling the initialization and cleanup of the rollout environment.
    """
    hcg = fleet.get_hybrid_communicate_group()
    hcg_mp_group_func = hcg.get_model_parallel_group
    hcg_mp_size_func = hcg.get_model_parallel_world_size
    hcg_mp_rank_func = hcg.get_model_parallel_rank
    hcg_sdp_group_func = hcg.get_sharding_parallel_group
    hcg_dp_group_func = hcg.get_data_parallel_group

    tp_mp_group_func = topology._HYBRID_PARALLEL_GROUP.get_model_parallel_group
    tp_mp_size_func = topology._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size
    tp_mp_rank_func = topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank

    world_size = dist.get_world_size()
    infer_topo = CommunicateTopology(
        hybrid_group_names=["data", "pipe", "sharding", "sep", "model"],
        dims=[world_size // tensor_parallel_degree, 1, 1, 1, tensor_parallel_degree],
    )
    infer_hcg = HybridCommunicateGroup(infer_topo)

    tp_group = infer_hcg.get_model_parallel_group()
    sdp_group = infer_hcg.get_sharding_parallel_group()
    dp_group = infer_hcg.get_data_parallel_group()
    hcg.get_model_parallel_group = lambda: tp_group
    hcg.get_model_parallel_world_size = lambda: tp_group.nranks
    hcg.get_model_parallel_rank = lambda: tp_group.rank
    hcg.get_sharding_parallel_group = lambda: sdp_group
    hcg.get_data_parallel_group = lambda: dp_group

    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_group = lambda: tp_group
    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size = lambda: tp_group.nranks
    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank = lambda: tp_group.rank

    def _get_rng_state(seed=0):
        """get_rng_state"""
        origin_rng_state = paddle.get_cuda_rng_state()
        paddle.seed(seed)
        rng_state = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(origin_rng_state)
        return rng_state

    if "model_parallel_rng" not in get_rng_state_tracker().states_:
        local_seed = 2023 + 1 + tp_group.rank
        get_rng_state_tracker().add("model_parallel_rng", local_seed)

    orig_rng_state = paddle.get_rng_state()
    rng_state = _get_rng_state(seed)
    paddle.set_rng_state(rng_state)
    yield
    hcg.get_model_parallel_group = hcg_mp_group_func
    hcg.get_model_parallel_world_size = hcg_mp_size_func
    hcg.get_model_parallel_rank = hcg_mp_rank_func
    hcg.get_model_parallel_group = hcg_mp_group_func
    hcg.get_sharding_parallel_group = hcg_sdp_group_func
    hcg.get_data_parallel_group = hcg_dp_group_func

    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_group = tp_mp_group_func
    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size = tp_mp_size_func
    topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank = tp_mp_rank_func
    paddle.set_rng_state(orig_rng_state)


@paddle.no_grad()
def pp_reshard(tgt_tensor, src_model_state_dict, src_tensor_meta_info, pp_rank, pp_group):
    """
    Redistribute tensors from the source model state dictionary to match the target batch size and return the
        redistributed tensor.

    Args:
        tgt_tensor (paddle.Tensor): The target tensor, used to determine the data type and shape of the redistributed tensor.
        src_model_state_dict (dict): The source model state dictionary, containing tensors that need to be redistributed.
        src_tensor_meta_info (dict): Metadata of the source tensor, containing the following key-value pairs:
            - "pipeline_key" (str): The key name of the source tensor in the source model state dictionary.
            - "pipeline_src_rank" (int): The source batch size rank to which the source tensor belongs.
            - "shape" (tuple): The shape of the source tensor.
        pp_rank (int): The rank of the current process in the pipeline group.
        pp_group (paddle.distributed.ProcessGroup): The pipeline group used for broadcast operations.

    Returns:
        paddle.Tensor: The redistributed tensor, with the same data type and shape as the target tensor.
    """
    src_tensor_key = src_tensor_meta_info["pipeline_key"]
    src_tensor_pp_rank = src_tensor_meta_info["pipeline_src_rank"]
    src_tensor_shape = src_tensor_meta_info["shape"]

    if src_tensor_pp_rank == pp_rank:
        src_tensor = src_model_state_dict.pop(src_tensor_key)
        resharded_tensor = src_tensor.clone()
        cpu_src_tensor = src_tensor.pin_memory()
        cpu_src_tensor._share_buffer_to(src_tensor)
    else:
        resharded_tensor = paddle.empty(src_tensor_shape)

    resharded_tensor = resharded_tensor.astype(tgt_tensor.dtype)
    dist.broadcast(
        resharded_tensor,
        src=pp_group.ranks[src_tensor_pp_rank],
        group=pp_group,
        sync_op=True,
    )
    return resharded_tensor


@paddle.no_grad()
def mp_reshard(
    src_tensor,
    tgt_tensor,
    meta_dict,
    train_tp_group,
    rollout_tp_group,
):
    """
    Convert the model parameters from the training TP distribution to the rollout TP distribution and return the new tensor.
    If the distributions of the two TP groups are the same, the original tensor is returned directly.

    Args:
        src_tensor (paddle.Tensor, optional): The tensor to be converted, defaults to None.
        tgt_tensor (paddle.Tensor, optional): The target tensor to store the converted tensor, defaults to None.
        meta_dict (dict, optional): A dictionary containing meta-information such as whether it is distributed and the split_axis,
            defaults to None.
        train_tp_group (paddle.distributed.DistributedGroup, optional): The distribution of the training TP group, defaults to None.
        rollout_tp_group (paddle.distributed.DistributedGroup, optional): The distribution of the rollout TP group, defaults to None.

    Returns:
        paddle.Tensor, optional: The converted tensor. If the distributions of the two TP groups are the same, the original tensor is
            returned directly.

    Raises:
        None
    """
    if rollout_tp_group.nranks == train_tp_group.nranks:
        return src_tensor

    if meta_dict["is_distributed"]:
        res = []
        if train_tp_group.nranks > 1:
            paddle.distributed.all_gather(res, src_tensor, group=train_tp_group, sync_op=True)
        else:
            res = [src_tensor]
        if hasattr(tgt_tensor, "is_distributed") and tgt_tensor.is_distributed:
            assert hasattr(tgt_tensor, "split_axis"), f"{tgt_tensor.name} has no split_axis!"
            concat_tensor = paddle.concat(res, meta_dict["split_axis"])
            del res
            all_parts = paddle.split(concat_tensor, rollout_tp_group.nranks, tgt_tensor.split_axis)
            del concat_tensor
            return all_parts[rollout_tp_group.rank]
        else:
            return paddle.concat(res, meta_dict["split_axis"])
    return src_tensor


def init_reshard_mappings(model, training_args, pp_rank, pp_group):
    """
    Initialize reshard mappings and return a global metadata dictionary.
    If the model is trained with multiple pipeline parallelism degrees,
    it will set pipeline name mappings. If the training model is single pipeline parallelism,
    it will replace all parameter names with names without the '_layers.' prefix.
    Then, for each parameter, create a tuple containing the parameter name, pipeline key,
    source rank, shape, and whether it is distributed.
    Finally, if the training model has multiple pipeline parallelism degrees,
    use the `dist.all_gather_object` function to merge the local metadata dictionary
    with the metadata dictionaries of other processes.

    Args:
        model (paddle.nn.Layer): Model instance.
        training_args (obj:`TrainingArguments`): Training configuration instance, including pipeline parallelism.
        pp_rank (int, optional): Horizontal parallelism rank of the current process (default: 0).
        pp_group (obj:`dist.ProcessGroup`, optional): Horizontal parallelism process group of the current process
            (default: None).

    Returns:
        dict: Global metadata dictionary, including pipeline key, source rank, shape, and distribution status for each
            parameter.
    """
    global_meta_dict = {}
    if training_args.pipeline_parallel_degree > 1:
        model._layers._set_pipeline_name_mapping()
        local_name_mapping_dict = model._layers._single_to_pp_mapping
    else:
        local_name_mapping_dict = {}
        for k in model.state_dict():
            local_name_mapping_dict[k] = k.replace("_layers.", "")
    local_model_state_dict = unwrap_model(model).state_dict()
    local_meta_dict = {}
    for k, v in local_name_mapping_dict.items():
        if training_args.pipeline_parallel_degree == 1:
            k = k.replace("_layers.", "")
        pipeline_key = v
        pipeline_tensor = local_model_state_dict[pipeline_key]
        local_meta_dict[k] = {
            "pipeline_key": pipeline_key,
            "pipeline_src_rank": pp_rank,
            "shape": pipeline_tensor.shape,
        }
        local_meta_dict[k]["is_distributed"] = False
        if hasattr(pipeline_tensor, "is_distributed"):
            local_meta_dict[k]["is_distributed"] = pipeline_tensor.is_distributed
        local_meta_dict[k]["split_axis"] = None
        if hasattr(pipeline_tensor, "split_axis"):
            local_meta_dict[k]["split_axis"] = pipeline_tensor.split_axis
    if training_args.pipeline_parallel_degree > 1:
        gathered_local_meta_dict = []
        dist.all_gather_object(gathered_local_meta_dict, local_meta_dict, group=pp_group)
    else:
        gathered_local_meta_dict = [local_meta_dict]
    for meta_dict in gathered_local_meta_dict:
        global_meta_dict.update(meta_dict)
    return global_meta_dict


@paddle.no_grad()
def reshard_to_rollout(
    train_model,
    rollout_model,
    global_meta_dict,
    pp_rank,
    pp_group,
    rollout_tp_group,
    train_tp_group,
):
    """
    Convert the model from training mode to inference mode and redistribute its parameters to meet the requirements of distributed and
        parallel computing.

    Args:
        train_model (paddle.nn.Layer): The original model, which should be in training mode.
        rollout_model (paddle.nn.Layer): The target model, which should be in inference mode.
        global_meta_dict (dict): A dictionary containing meta-information about each parameter, including names, sizes, etc.
        pp_rank (int): The global process ID of the current process (prediction process).
        pp_group (paddle.distributed.ProcessGroup): The prediction process group.
        rollout_tp_group (paddle.distributed.ProcessGroup): The inference process group.
        train_tp_group (paddle.distributed.ProcessGroup): The training process group.

    Returns:
        None: This function does not return any value.

    Raises:
        None: This function does not raise any exceptions.
    """
    train_model_state_dict = train_model.state_dict()
    rollout_model_state_dict = rollout_model.state_dict()
    param_numel = [(k, np.prod(v.shape)) for k, v in rollout_model_state_dict.items()]
    param_numel.sort(key=lambda x: x[1], reverse=True)

    for k, _ in param_numel:
        v = rollout_model_state_dict[k]
        resharded_tensor = pp_reshard(v, train_model_state_dict, global_meta_dict[k], pp_rank, pp_group)
        resharded_tensor = mp_reshard(
            resharded_tensor,
            v,
            global_meta_dict[k],
            train_tp_group,
            rollout_tp_group,
        )
        assert resharded_tensor.dtype == v.dtype, f"dtype wrong {k} {resharded_tensor.dtype} {v.dtype}"
        assert resharded_tensor.shape == v.shape, f"shape wrong {k} {resharded_tensor.shape} {v.shape}"
        resharded_tensor._share_buffer_to(v)
        resharded_tensor._clear()

    missing_keys = train_model_state_dict.keys()
    num_missing_keys = len(missing_keys)
    assert num_missing_keys == 0, f"missing {num_missing_keys} keys after reshard policy: {missing_keys}"
    logger.info("[Reshard] Done")
