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

import os
import random

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from ppfleetx.distributed.apis import comm_groups
from ppfleetx.utils.log import logger

__all__ = ["init_dist_env"]

_seed = None
_dp_seed = None
_hcg = None


def set_seed(seed):
    if dist.get_world_size() > 1:
        # obtain rank message of hybrid parallel
        hcg = get_hcg()
        mp_rank = hcg.get_model_parallel_rank()
        pp_rank = hcg.get_stage_id()
        data_world_rank = get_data_world_rank()
        data_world_size = get_data_world_size()
    else:
        mp_rank, pp_rank, data_world_rank, data_world_size = 0, 0, 0, 1

    # NOTE: the commented seeds are set only for precision validation
    # seed += 100 * pp_rank
    # random.seed(seed)
    # np.random.seed(seed)
    # paddle.seed(seed)

    random.seed(seed + data_world_rank)
    np.random.seed(seed + data_world_rank)
    paddle.seed(seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = seed + 123 + mp_rank * 10 + pp_rank * 1000 + data_world_size
    global_seed = seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)

    logger.info("The global seed is set to {} and local seed is set to {}.".format(global_seed, local_seed))

    global _seed
    global _dp_seed
    _seed = seed
    _dp_seed = global_seed


def set_hcg(hcg):
    global _hcg
    _hcg = hcg


def get_hcg():
    global _hcg
    return _hcg


def get_seed():
    global _seed
    return _seed


def get_dp_seed():
    global _dp_seed
    return _dp_seed


def init_dist_env(config):
    paddle.set_device(config.Global.device)

    strategy = fleet.DistributedStrategy()
    if config.Distributed.mp_degree == 1 and config.Distributed.sharding.sharding_degree == 1:
        order = ["pp", "dp", "sharding", "mp"]
    else:
        order = ["dp", "pp", "sharding", "mp"]

    strategy.hybrid_configs = {
        "dp_degree": config.Distributed.dp_degree,
        "mp_degree": config.Distributed.mp_degree,
        "pp_degree": config.Distributed.pp_degree,
        "sharding_degree": config.Distributed.sharding.sharding_degree,
        "order": order,
    }

    if config.Distributed.pp_degree > 1:
        if "sequence_parallel" in config.Model:
            if config.Model.sequence_parallel:
                assert config.Global.enable_partial_send_recv is False, (
                    "if config.Distributed.pp_degree > 1 and config.Model.sequence_parallel is True, "
                    "config.Global.enable_partial_send_recv should be set False."
                )

    strategy.pipeline_configs = {
        "accumulate_steps": config.Global.local_batch_size // config.Global.micro_batch_size,
        "micro_batch_size": config.Global.micro_batch_size,
        "enable_partial_send_recv": config.Global.enable_partial_send_recv,
    }

    # set control in tensor parallel
    seed = config.Global.seed
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    hcg = comm_groups.create_hcg(strategy, hcg_name=config.Distributed.hcg)
    set_hcg(hcg)


def get_local_rank():
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def get_data_world_size():
    if paddle.distributed.get_world_size() == 1:
        return 1

    hcg = get_hcg()
    dp_size = hcg.get_data_parallel_world_size()
    sharding_size = hcg.get_sharding_parallel_world_size()

    return dp_size * sharding_size


def get_data_world_rank():
    if paddle.distributed.get_world_size() == 1:
        return 0

    hcg = get_hcg()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    sharding_size = hcg.get_sharding_parallel_world_size()

    return dp_rank * sharding_size + sharding_rank


def work_at_local_rank0(func):
    def wrapper(*args, **kwargs):
        local_rank = 0
        if paddle.fluid.core.is_compiled_with_dist() and paddle.distributed.get_world_size() > 1:
            local_rank = paddle.distributed.ParallelEnv().dev_id
        if local_rank == 0:
            func(*args, **kwargs)

    return wrapper
