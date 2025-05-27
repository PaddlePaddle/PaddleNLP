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
    dist.broadcast(resharded_tensor, src=pp_group.ranks[src_tensor_pp_rank], group=pp_group, sync_op=True)
    return resharded_tensor


@paddle.no_grad()
def mp_reshard(
    src_tensor,
    tgt_tensor,
    meta_dict,
    train_tp_group,
    rollout_tp_group,
):
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
    train_model, rollout_model, global_meta_dict, pp_rank, pp_group, rollout_tp_group, train_tp_group
):
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
