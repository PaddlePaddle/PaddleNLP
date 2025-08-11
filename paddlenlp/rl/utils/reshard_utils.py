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

from copy import copy
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


class ReshardController:
    def __init__(
        self,
        train_tensor_parallel_degree,
        infer_tensor_parallel_degree, # 这时候输入就已经是rollout_tensor_tp了
        pipeline_parallel_degree=1,
        sharding_parallel_degree=1,
        sep_parallel_degree=1,
        seed=100,
        **kwargs
    ):
        self.train_tensor_parallel_degree = train_tensor_parallel_degree
        self.infer_tensor_parallel_degree = infer_tensor_parallel_degree
        self.pipeline_parallel_degree = pipeline_parallel_degree
        self.sharding_parallel_degree = sharding_parallel_degree
        self.sep_parallel_degree = sep_parallel_degree
        self.seed = seed
        self.orig_rng_state = paddle.get_rng_state()
        self.orig_cuda_rng_state = self._get_rng_state()
        self.train_hcg = fleet.get_hybrid_communicate_group()
        self.train_tp_group, self.train_dp_group, self.train_sdp_group = (
            self.train_hcg.get_model_parallel_group(),
            self.train_hcg.get_data_parallel_group(),
            self.train_hcg.get_sharding_parallel_group(),
        )
        self.gather_in_micro_dp = kwargs.get("gather_in_micro_dp", False)
        self.infer_tp_group, self.infer_dp_group, self.infer_sdp_group = self.init_rollout_env()
        self.set_train_env()

        self.is_train = True


    def init_rollout_env(self):
        gather_in_micro_dp = self.gather_in_micro_dp
        if not gather_in_micro_dp:
            world_size = dist.get_world_size()
            infer_topo = CommunicateTopology(
                hybrid_group_names=["data", "pipe", "sharding", "sep", "model"],
                dims=[
                    world_size
                    // self.infer_tensor_parallel_degree
                    // self.pipeline_parallel_degree
                    // self.sharding_parallel_degree
                    // self.sep_parallel_degree,
                    self.pipeline_parallel_degree,
                    self.sharding_parallel_degree,
                    self.sep_parallel_degree,
                    self.infer_tensor_parallel_degree,
                ],
            )

            infer_hcg = HybridCommunicateGroup(infer_topo)
            infer_tp_group = infer_hcg.get_model_parallel_group()
            infer_dp_group = infer_hcg.get_data_parallel_group()
            infer_sdp_group = infer_hcg.get_sharding_parallel_group()
        else:
            world_size = dist.get_world_size()
            micro_dp = self.train_tensor_parallel_degree // self.infer_tensor_parallel_degree
            # Fu 不行，需要自己手动创建并行组
            # order: DP PP SDP SEP TP micro_DP
            # DP: concat([train DP group, micro_DP group])
            # TP: interval= micro DP
            # PP/SDP/SEP == train PP/SDP/SEP


            topo_order = ["data", "pipe", "sharding", "sep", "model"]
            # infer_topo = CommunicateTopology(
            #     hybrid_group_names=["model", "pipe", "data", "sharding", "sep"],
            #     dims=[
            #         self.tensor_parallel_degree,
            #         self.pipeline_parallel_degree,
            #         world_size
            #         // self.tensor_parallel_degree
            #         // self.pipeline_parallel_degree
            #         // self.sharding_parallel_degree
            #         // self.sep_parallel_degree,
            #         self.sharding_parallel_degree,
            #         self.sep_parallel_degree,
            #     ],
            # )

            all_tp_dp_ranks = []
            # 获取所有卡的tp group，按照dp，sdp，pp group进行聚合；三次通信不如直接all gather后去重
            paddle.distributed.all_gather_object(all_tp_dp_ranks, (self.train_tp_group.ranks,self.train_dp_group.ranks))
            all_tp_ranks = [t[0] for t in all_tp_dp_ranks]
            all_train_dp_ranks = [t[1] for t in all_tp_dp_ranks]
            all_tp_ranks = [list(t) for t in set(tuple(sub) for sub in all_tp_ranks)] # 去重
            all_train_dp_ranks = [list(t) for t in set(tuple(sub) for sub in all_train_dp_ranks)]


            rank = paddle.distributed.get_rank()
            # 在train TP group中，划分micro dp group和rollout TP group
            # for micro_dp_idx in range(micro_dp): # 这个是micro dp, rollout t
            #     for rollout_tp_idx in range(self.tensor_parallel_degree):
                    # idx = micro_dp_idx * micro_dp + rollout_tp_idxp
            train_ranks = self.train_tp_group.ranks
            micro_group_lst, infer_tp_group_lst = [], []
            for tp_ranks in all_tp_ranks:
                # micro_dp_group  
                for rollout_tp_idx in range(self.infer_tensor_parallel_degree):
                    micro_rank_lst = []
                    for micro_dp_idx in range(micro_dp):
                        idx = rollout_tp_idx * micro_dp + micro_dp_idx
                        micro_rank_lst.append(tp_ranks[idx])
                    micro_group_lst.append(micro_rank_lst)
                # infer_tp_group
                for micro_dp_idx in range(micro_dp):
                    infer_tp_rank_lst = []
                    for rollout_tp_idx in range(self.infer_tensor_parallel_degree):
                        idx = rollout_tp_idx * micro_dp + micro_dp_idx
                        infer_tp_rank_lst.append(tp_ranks[idx])
                    infer_tp_group_lst.append(infer_tp_rank_lst)

            # 融合DP和micro DP
            infer_dp_group_lst = []
            for micro_ranks in micro_group_lst:
                infer_dp_rank_lst = []
                for micro_rank in micro_ranks:
                    for train_dp_ranks in all_train_dp_ranks:
                        if micro_rank in train_dp_ranks:
                            infer_dp_rank_lst.extend(train_dp_ranks)
                            break
                infer_dp_rank_lst = list(set(infer_dp_rank_lst))
                infer_dp_group_lst.append(infer_dp_rank_lst)
            infer_dp_group_lst = [list(t) for t in set(tuple(sub) for sub in infer_dp_group_lst)]
            print(f"Fu infer dp group lst:{infer_dp_group_lst} micro_group_lst:{micro_group_lst}, infer tp group list:{infer_tp_group_lst}")
            

            # micro_group_dict={0:[0,1],1:[2,3],2:[4,5],3:[6,7]}
            for ranks in micro_group_lst:
                gp = paddle.distributed.new_group(ranks=ranks)
                print(f"Fu [micro dp group create] create a group {gp.id}:{gp.ranks}")
                if rank in ranks:
                    self.micro_dp_group = gp
            for ranks in infer_dp_group_lst:
                print(f"Fu [before infer dp group create] begin to create a dp group:{ranks}")
                gp = paddle.distributed.new_group(ranks=ranks)
                print(f"Fu [infer dp group create] create a group {gp.id}:{gp.ranks}")
                if rank in ranks:
                    infer_dp_group = gp
            for ranks in infer_tp_group_lst:
                gp = paddle.distributed.new_group(ranks=ranks)
                print(f"Fu [infer tp group create] create a group {gp.id}:{gp.ranks}")
                if rank in ranks:
                    infer_tp_group = gp
            infer_sdp_group = self.train_sdp_group
            print(f"Fu rank:{rank}, micro_dp_group:{self.micro_dp_group}, infer_dp_group:{infer_dp_group}, infer_tp_group:{infer_tp_group}")

        print(f"Fu infer tp group:{infer_tp_group}, dp group:{infer_dp_group}, sdp group:{infer_sdp_group}")
        return (infer_tp_group, infer_dp_group, infer_sdp_group)

    def _get_rng_state(self):
        """get_rng_state"""
        origin_rng_state = paddle.get_cuda_rng_state()
        paddle.seed(self.seed)
        rng_state = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(origin_rng_state)
        return rng_state

    def set_rollout_env(self, msg=""):
        if "model_parallel_rng" not in get_rng_state_tracker().states_:
            local_seed = 2025 + 1 + self.train_hcg.get_model_parallel_rank()
            get_rng_state_tracker().add("model_parallel_rng", local_seed)

        paddle.set_rng_state(self.orig_cuda_rng_state)

        hcg = fleet.get_hybrid_communicate_group()
        hcg.get_model_parallel_group = lambda: self.infer_tp_group
        hcg.get_model_parallel_world_size = lambda: self.infer_tp_group.nranks
        hcg.get_model_parallel_rank = lambda: self.infer_tp_group.rank
        hcg.get_sharding_parallel_group = lambda: self.infer_sdp_group
        hcg.get_data_parallel_group = lambda: self.infer_dp_group
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_group = lambda: self.infer_tp_group
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size = lambda: self.infer_tp_group.nranks
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank = lambda: self.infer_tp_group.rank
        self.log(msg, False)
        self.is_train = False

    def set_train_env(self, msg=""):
        hcg = fleet.get_hybrid_communicate_group()
        hcg.get_model_parallel_group = lambda: self.train_tp_group
        hcg.get_model_parallel_world_size = lambda: self.train_tp_group.nranks
        hcg.get_model_parallel_rank = lambda: self.train_tp_group.rank
        hcg.get_sharding_parallel_group = lambda: self.train_sdp_group
        hcg.get_data_parallel_group = lambda: self.train_dp_group
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_group = lambda: self.train_tp_group
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size = lambda: self.train_tp_group.nranks
        topology._HYBRID_PARALLEL_GROUP.get_model_parallel_rank = lambda: self.train_tp_group.rank

        paddle.set_rng_state(self.orig_rng_state)
        self.log(msg, True)
        self.is_train = True

    def log(self, msg, is_train=False):
        msg = f"for {msg}" if len(msg) > 0 else ""
        if is_train:
            logger.warning(
                f"Recover train env done {msg}. [Global TP]: {fleet.get_hybrid_communicate_group().get_model_parallel_world_size()}, [Train TP]: {self.train_tp_group.nranks}, [Infer TP]: {self.infer_tp_group.nranks}"
            )
        else:
            logger.warning(
                f"Set rollout env done {msg}. [Global TP]: {fleet.get_hybrid_communicate_group().get_model_parallel_world_size()}, [Train TP]: {self.train_tp_group.nranks}"
            )


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
    **kwargs,
):
    gather_in_micro_dp = kwargs.get("gather_in_micro_dp", False)
    
    if not gather_in_micro_dp:
        rollout_tp_group = kwargs.get("rollout_tp_group", None)
        train_tp_group = kwargs.get("train_tp_group", None)

        if rollout_tp_group.nranks == train_tp_group.nranks:
            return src_tensor

        if meta_dict["is_distributed"]:
            res = []
            if train_tp_group.nranks > 1:
                paddle.distributed.all_gather(res, src_tensor, group=train_tp_group, sync_op=True)
            else:
                res = [src_tensor]
            if hasattr(tgt_tensor, "is_distributed") and tgt_tensor.is_distributed:
                merge_fn = meta_dict["merge_tensor_fn"]
                concat_tensor = merge_fn(
                    res, transpose=False, is_old_qkv=False, is_naive_2fuse=False, is_navice_3fuse=False, keep_on_gpu=True
                )
                del res
                split_fn = meta_dict["split_tensor_fn"]
                split_part = split_fn(
                    concat_tensor, transpose=False, is_old_qkv=False, is_naive_2fuse=False, is_naive_3fuse=False
                )
                del concat_tensor
                return split_part
            else:
                merge_fn = meta_dict["merge_tensor_fn"]
                concat_tensor = merge_fn(
                    res, transpose=False, is_old_qkv=False, is_naive_2fuse=False, is_navice_3fuse=False, keep_on_gpu=True
                )
                return concat_tensor
        return src_tensor
    else:
        micro_dp_group = kwargs.get("micro_dp_group", None)
        rollout_tp = kwargs.get("rollout_tp", None)
        train_tp = kwargs.get("train_tp", None)

        if rollout_tp==train_tp:
            return src_tensor
        
        if meta_dict['is_distributed']: # 像input layernorm的is_distributed就是False，即为不分片（但是SP不就是给它分片）
            # 训练权重按照micro dp并行组聚合。这里暂未考虑sdp情况，回头需要测试一下
            res = []
            assert train_tp>rollout_tp, "train_tp must greater than rollout_tp for larger throughtout in rollout"
            
            paddle.distributed.all_gather(res, src_tensor, group=micro_dp_group, sync_op=True)
            merge_fn = meta_dict["merge_tensor_fn"]
            concat_tensor = merge_fn(
                res, transpose=False, is_old_qkv=False, is_naive_2fuse=False, is_naive_3fuse=False, keep_on_gpu=True
            )
            return concat_tensor
        return src_tensor
        # if meta_dict.get("split_axis",None) is not None:
        #     # print(f"meta_dict:{meta_dict}")
        #     paddle.distributed.all_gather(res, src_tensor, group=micro_dp_group, sync_op=True)
        #     # if src_tensor.ndim==1: # ？？？在layernorm。weigth时报错没有split_axis。为什么上面的就不会报错呢？
        #     #     meta_dict["split_axis"] = 0
        #     return paddle.concat(res, meta_dict["split_axis"])
        # # print(f"Fu meta_dict has none split_axis, {meta_dict}")
        # return src_tensor # 针对layernorm的TP情况


def init_reshard_mappings(model, training_args, pp_rank, pp_group, rollout_tp_group):
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
    if training_args.pipeline_parallel_degree > 1:
        gathered_local_meta_dict = []
        dist.all_gather_object(gathered_local_meta_dict, local_meta_dict, group=pp_group)
    else:
        gathered_local_meta_dict = [local_meta_dict]
    for meta_dict in gathered_local_meta_dict:
        global_meta_dict.update(meta_dict)
    
    if (
        training_args.tensor_parallel_degree != training_args.rollout_tensor_parallel_degree or training_args.pipeline_parallel_degree > 1
    ):
        model_class = type(model)
        tensor_parallel_config = copy(model.config)
        tensor_parallel_config.tensor_parallel_degree = training_args.rollout_tensor_parallel_degree
        tensor_parallel_config.tensor_parallel_rank = rollout_tp_group.rank

        merge_tensor_fn_dict = model_class._get_tensor_parallel_mappings(config=tensor_parallel_config, is_split=False)
        for k,v in merge_tensor_fn_dict.items():
            key = k if k in global_meta_dict else f"{model.config.model_type}.{k}"
            global_meta_dict[key]["merge_tensor_fn"] = v
        
        if training_args.rollout_tensor_parallel_degree > 1:
            split_tensor_fn_dict = model_class._get_tensor_parallel_mappings(
                config=tensor_parallel_config, is_split=True
            )
            for k,v in split_tensor_fn_dict.items():
                key = k if k in global_meta_dict else f"{model.config.model_type}.{k}"
                global_meta_dict[key]["split_tensor_fn"] = v
    return global_meta_dict


@paddle.no_grad()
# def reshard_to_rollout(
#     train_model, rollout_model, global_meta_dict, pp_rank, pp_group, rollout_tp_group, train_tp_group, gather_in_micro_dp=False
# ):
def reshard_to_rollout(
    train_model, rollout_model, global_meta_dict, pp_rank, pp_group, **kwargs
):
    train_model_state_dict = train_model.state_dict()
    rollout_model_state_dict = rollout_model.state_dict()
    param_numel = [(k, np.prod(v.shape)) for k, v in rollout_model_state_dict.items()]
    param_numel.sort(key=lambda x: x[1], reverse=True)

    for k, _ in param_numel:
        v = rollout_model_state_dict[k]
        resharded_tensor = pp_reshard(v, train_model_state_dict, global_meta_dict[k], pp_rank, pp_group)
        # resharded_tensor = mp_reshard(
        #     resharded_tensor,
        #     v,
        #     global_meta_dict[k],
        #     train_tp_group,
        #     rollout_tp_group, # 注意当gather_in_micro_dp为True时，rollout_tp_group其实是rollout_dp_group，这里懒得改了
        #     gather_in_micro_dp,
        # )
        resharded_tensor = mp_reshard(
            resharded_tensor,
            v,
            global_meta_dict[k],
            **kwargs,
        )
        assert resharded_tensor.dtype == v.dtype, f"dtype wrong {k} {resharded_tensor.dtype} {v.dtype}"
        assert resharded_tensor.shape == v.shape, f"shape wrong {k} {resharded_tensor.shape} {v.shape}"
        resharded_tensor._share_buffer_to(v)
        resharded_tensor._clear()

    missing_keys = train_model_state_dict.keys()
    num_missing_keys = len(missing_keys)
    assert num_missing_keys == 0, f"missing {num_missing_keys} keys after reshard policy: {missing_keys}"
    logger.warning("[Reshard] Done")
