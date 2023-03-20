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
import inspect
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import TensorParallel, get_rng_state_tracker
from paddle.distributed.sharding import group_sharded_parallel

from paddlenlp.utils.log import logger

try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers


def _rotate_checkpoints(save_total_limit, use_mtime=False, output_dir=None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def _sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

    return checkpoints_sorted


def all_gather(v, group=None):
    if paddle.distributed.get_world_size() <= 1:
        return v.item()
    ret = []
    paddle.distributed.all_gather(ret, v, group=group)
    concat = paddle.concat(ret, axis=0)
    return concat.mean().item()


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


def wrap_sharding_2_3(model, optimizer, scaler, dist_config):
    """_summary_
    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        scaler (_type_): _description_
        dist_config (_type_): _description_
    Returns:
        _type_: _description_
    """
    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    if dist_config.dp_degree > 1 and dist_config.sharding_stage == 3:
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.mp_degree > 1:
        assert dist_config.sharding_stage == 2, "only support mp + sharding stage2 hybrid parallel now."
        model = TensorParallel(model, hcg, strategy=None)

    level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # origin_model = model
    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=dist_config.sharding_offload,
        dp_group=dp_group if dp_group.nranks > 1 else None,
    )
    return model, optimizer, scaler


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())
