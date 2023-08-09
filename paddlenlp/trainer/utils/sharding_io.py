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

import json
import os
from collections import OrderedDict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)

from paddlenlp.transformers.model_utils import _add_variant, unwrap_optimizer
from paddlenlp.utils.log import logger

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"
MODEL_META_NAME = "model_meta.json"
SHARDING_META_NAME = "shard_meta.json"


class ShardingIO:
    def __init__(self, args, model, optimizer=None, hcg=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.hcg = hcg
        self.sharding_group = None
        if self.hcg is None and paddle.distributed.get_world_size() > 1 and self.args.use_hybrid_parallel:
            self.hcg = fleet.get_hybrid_communicate_group()
            self.sharding_group = self.hcg.get_sharding_parallel_group()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def load_state_dict_from_checkpoint_with_reshard(self, resume_from_checkpoint, load_one_state_dict_func):
        """load state_dict from_checkpoint with reshard, Only load model state dict."""
        parallel_config = self._load_distributed_strategy(resume_from_checkpoint)
        pp_degree = parallel_config["pp_degree"]
        pp_degree = parallel_config["pp_degree"]
        mp_degree = parallel_config["mp_degree"]
        sharding_degree = parallel_config["sharding_degree"]
        self.args.pipeline_parallel_degree == pp_degree
        self.args.tensor_parallel_degree == mp_degree
        cur_sharding_degree = self.args.sharding_parallel_degree

        state_dict = OrderedDict()

        for i in range(self.args.sharding_parallel_rank, sharding_degree, cur_sharding_degree):
            tmp = load_one_state_dict_func(resume_from_checkpoint, self.args.sharded_weight_name_suffix(i))
            for (k, v) in tmp.items():
                state_dict[k] = v
            del tmp

        def filter_func(name):
            return True

        state_dict = self._all_gather_state_dict(state_dict, filter_func)

        if self.args.bf16:
            state_dict = self._recover_params_from_master_weights(state_dict)

        return state_dict

    def load_optimizer_state_with_reshard(self, checkpoint, load_optimizer_state_func):
        """load state_dict of multiple shard from_checkpoint, Only load model state dict."""
        parallel_config = self._load_distributed_strategy(checkpoint)
        pp_degree = parallel_config["pp_degree"]
        mp_degree = parallel_config["mp_degree"]
        sharding_degree = parallel_config["sharding_degree"]
        assert self.args.pipeline_parallel_degree == pp_degree
        assert self.args.tensor_parallel_degree == mp_degree
        cur_sharding_degree = self.args.sharding_parallel_degree

        def need_reshard():
            if sharding_degree != cur_sharding_degree:
                return True
            sharding_meta = self._load_sharding_meta(checkpoint)
            param2rank = sharding_meta["param2rank"]
            optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
            assert optimizer
            assert len(param2rank) == len(optimizer._param2rank)
            for (k, v) in param2rank.items():
                assert k in optimizer._param2rank
                if optimizer._param2rank[k] != int(v):
                    return True
            return False

        if not need_reshard():
            logger.info("do not need reshard")
            return load_optimizer_state_func(checkpoint, self.args.optimizer_name_suffix)
        logger.info("reshard optimizer state")
        state_dict = OrderedDict()
        master_weights = OrderedDict()
        lr_scheduler = {}

        for i in range(self.args.sharding_parallel_rank, sharding_degree, cur_sharding_degree):
            tmp = load_optimizer_state_func(checkpoint, self.args.sharded_weight_name_suffix(i))

            if not tmp:
                continue

            for (k, v) in tmp.items():
                if k == "master_weights":
                    for (kk, vv) in v.items():
                        master_weights[kk] = vv
                    continue
                if k == "LR_Scheduler":
                    lr_scheduler[i] = v
                    continue
                state_dict[k] = v

            del tmp

            # gather all opt names
        # list of list
        opt_names_list = self._all_gather_simple_object(list(state_dict.keys()))
        opt_names = []
        for e in opt_names_list:
            opt_names.extend(e)

        # opt name to param name
        opt_to_p = self._map_optimizer_state_to_param(opt_names)

        optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
        param2rank = optimizer._param2rank

        def all_gather_state_dict(state_dict, filter_func):
            remote_state_dict_keys = [k for k in state_dict.keys() if not filter_func(k)]
            tmp_state_dict = OrderedDict()
            for k in remote_state_dict_keys:
                tmp_state_dict[k] = state_dict[k]
                state_dict.pop(k)
            tmp_state_dict = self._all_gather_state_dict(tmp_state_dict, filter_func)
            for (k, v) in tmp_state_dict.items():
                state_dict[k] = v
            return state_dict

        def opt_filter_func(name):
            assert name in opt_to_p, f"name {name} not in opt_to_p"
            param_name = opt_to_p[name]
            assert param_name in param2rank, f"param_name {param_name} not in param2rank param2"
            return param2rank[param_name] == self.args.sharding_parallel_rank

        state_dict = all_gather_state_dict(state_dict, opt_filter_func)

        def master_weights_filter_func(name):
            assert (name in param2rank) or (name in opt_to_p), f"name {name} not in param2rank or opt_to_p"
            if name in opt_to_p:
                name = opt_to_p[name]
            return param2rank[name] == self.args.sharding_parallel_rank

        # master weights
        master_weights = all_gather_state_dict(master_weights, master_weights_filter_func)
        state_dict["master_weights"] = master_weights

        # lr scheduler
        print(lr_scheduler)
        lr_schedulers = self._all_gather_simple_object(lr_scheduler)
        lr_scheduler = {}
        for e in lr_schedulers:
            for (k, v) in e.items():
                lr_scheduler[k] = v
        if lr_scheduler:
            state_dict["LR_Scheduler"] = lr_scheduler[0]

        return state_dict

    def save_distributed_model_meta(self, dir):
        if not self.args.use_hybrid_parallel:
            return

        if not self.args.should_save_sharding_stage1_model:
            return

        nranks = dist.get_world_size()
        if nranks <= 1:
            return

        model_meta = {}
        parallel_config = self._get_distributed_strategy()
        if parallel_config:
            model_meta["parallel_config"] = parallel_config
        sharding_metas = self._gather_sharding_metas()
        if sharding_metas:
            model_meta["sharding_metas"] = sharding_metas

        if dist.get_rank():
            return

        path = os.path.join(dir, MODEL_META_NAME)
        with open(path, "w") as f:
            json.dump(model_meta, f, indent=4)

    def _get_distributed_strategy(self):
        pp_degree = 1
        mp_degree = 1
        sharding_degree = 1
        vpp_degree = 1
        nranks = dist.get_world_size()
        if self.args.use_hybrid_parallel and nranks > 1:
            if dist.get_rank():
                return
            hcg = fleet.get_hybrid_communicate_group()
            mp_degree = hcg.get_model_parallel_world_size()
            pp_degree = hcg.get_pipe_parallel_world_size()
            sharding_degree = hcg.get_sharding_parallel_world_size()
            """
            if pp_degree > 1:
                assert isinstance(model, fleet.meta_parallel.PipelineParallel), "must be pipeline model"
                vpp_degree = model._layers.get_num_virtual_stages()
            """
        parallel_config = {
            "pp_degree": pp_degree,
            "mp_degree": mp_degree,
            "sharding_degree": sharding_degree,
            "vpp_degree": vpp_degree,
        }
        return parallel_config

    def _recover_params_from_master_weights(self, state_dict):
        assert isinstance(self.optimizer._inner_opt, DygraphShardingOptimizer)
        param2rank = self.optimizer._inner_opt._param2rank
        opt_state_dict = self.optimizer.state_dict()
        assert "master_weights" in opt_state_dict
        master_weigths = opt_state_dict["master_weights"]
        param_names_in_master_weights = list(master_weigths.keys())
        tmp = []
        logger.debug("param_names_in_master_weights:{}".format(param_names_in_master_weights))
        paddle.distributed.all_gather_object(tmp, param_names_in_master_weights, group=self.sharding_group)
        sharding_group_param_names = [v for item in tmp for v in item]
        logger.debug("sharding_group_param_names:{}".format(sharding_group_param_names))
        model_state_dict = self.model.state_dict()
        logger.info("before recover, model_state_dict number: {}".format(len(model_state_dict)))
        for key, param in model_state_dict.items():
            if param.name in master_weigths:
                assert param.shape == master_weigths[param.name].shape
                paddle.assign(paddle.cast(master_weigths[param.name].cuda(), paddle.bfloat16), model_state_dict[key])
            if param.name in sharding_group_param_names:
                paddle.distributed.broadcast(
                    model_state_dict[key],
                    src=self.sharding_group.ranks[param2rank[param.name]],
                    group=self.sharding_group,
                    sync_op=True,
                )
        logger.info("after recover, casted model_state_dict number: {}".format(len(model_state_dict)))
        state_dict.update(model_state_dict)
        return state_dict

    def _all_gather_simple_object(self, obj, group=None):
        if group is None:
            group = self.hcg.get_sharding_parallel_group()
        res = []
        paddle.distributed.all_gather_object(res, obj, group)
        return res

    def _load_model_meta(self, dir):
        meta_path = os.path.join(dir, MODEL_META_NAME)
        assert os.path.exists(meta_path), f"{meta_path} not exist"
        with open(meta_path, "r") as handle:
            model_dist_meta = json.load(handle)
        assert "parallel_config" in model_dist_meta
        return model_dist_meta

    def _load_distributed_strategy(self, dir):
        model_dist_meta = self._load_model_meta(dir)
        parallel_config = model_dist_meta["parallel_config"]
        assert "pp_degree" in parallel_config
        assert "mp_degree" in parallel_config
        assert "sharding_degree" in parallel_config
        return parallel_config

    def _load_sharding_meta(self, dir):
        suffix = f"tp{self.args.tensor_parallel_rank:0>2d}_pp{self.args.pipeline_parallel_rank:0>2d}"
        distributed_model_meta = self._load_model_meta(dir)
        if "sharding_metas" in distributed_model_meta:
            sharding_metas = distributed_model_meta["sharding_metas"]
            assert suffix in sharding_metas
            sharding_meta = sharding_metas[suffix]
            assert "param2rank" in sharding_meta
            return sharding_meta

        # for backward compatibility
        meta_path = os.path.join(dir, _add_variant(SHARDING_META_NAME, suffix))
        assert os.path.exists(meta_path), f"{meta_path} not exist"
        with open(meta_path, "r") as f:
            sharding_meta = json.load(f)
        assert "param2rank" in sharding_meta
        return sharding_meta

    def _map_optimizer_state_to_param(self, optimizer_state_names):
        optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)
        all_names = list(optimizer._param2rank.keys())
        all_names.extend(list(optimizer_state_names))
        all_names.sort()
        pre_p_name = ""
        opt_to_p = {}
        for n in all_names:
            if n in optimizer._param2rank:
                # we get a param
                pre_p_name = n
            else:
                assert pre_p_name, n
                opt_to_p[n] = pre_p_name
        return opt_to_p

    def _all_gather_state_dict(self, state_dict, filter_func, group=None):
        if group is None:
            group = self.hcg.get_sharding_parallel_group()
        res = OrderedDict()

        def map_func(weight):
            if isinstance(weight, paddle.Tensor):
                weight = weight.numpy()
            return weight

        state_dict = {k: map_func(v) for (k, v) in state_dict.items()}

        meta_dict = {}
        for (k, v) in state_dict.items():
            # src rank
            meta_dict[k] = (v.dtype, v.shape, group.rank)

        meta_dict_list = self._all_gather_simple_object(meta_dict, group)

        total_meta_dict = {}
        for meta_dict in meta_dict_list:
            for (k, v) in meta_dict.items():
                assert k not in total_meta_dict
                total_meta_dict[k] = v

        meta_list = list(total_meta_dict.items())
        meta_list = sorted(meta_list, key=lambda x: x[0])
        for (k, meta) in meta_list:
            dtype, shape, rank = meta
            if rank == group.rank:
                assert k in state_dict
                tensor = paddle.to_tensor(state_dict[k])
            else:
                tensor = paddle.to_tensor(np.empty(shape, dtype))
            logger.info(f"broadcast {k} from {rank}")
            # broadcast the tensor
            paddle.distributed.broadcast(
                tensor,
                src=group.ranks[rank],
                group=group,
                sync_op=True,
            )
            if filter_func(k):
                res[k] = tensor.cpu()
            del tensor
        return res

    def _gather_sharding_metas(self):
        nranks = dist.get_world_size()
        if not self.args.use_hybrid_parallel or nranks <= 1:
            return None
        if self.args.sharding_parallel_rank != 0:
            return None
        if self.args.data_parallel_rank != 0:
            return None
        optimizer = unwrap_optimizer(self.optimizer, DygraphShardingOptimizer)

        if not optimizer:
            return None

        param2rank = {k: v for (k, v) in optimizer._param2rank.items()}

        model = self.model
        structure_name_mapping = {k: v.name for (k, v) in model.state_dict().items()}

        sharding_metas = {}
        sharding_meta = {}

        sharding_meta["param2rank"] = param2rank
        sharding_meta["structure_name_mapping"] = structure_name_mapping
        suffix = f"tp{self.args.tensor_parallel_rank:0>2d}_pp{self.args.pipeline_parallel_rank:0>2d}"
        sharding_metas[suffix] = sharding_meta
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_model_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        if self.args.tensor_parallel_rank != 0:
            return None
        sharding_metas_list = self._all_gather_simple_object(sharding_metas, self.hcg.get_pipe_parallel_group())
        sharding_metas = {k: v for e in sharding_metas_list for (k, v) in e.items()}
        return sharding_metas
