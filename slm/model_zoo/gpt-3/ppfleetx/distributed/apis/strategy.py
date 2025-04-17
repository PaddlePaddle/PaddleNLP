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

import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import TensorParallel
from paddle.distributed.parallel import sync_params_buffers
from paddle.distributed.sharding import group_sharded_parallel
from ppfleetx.distributed.apis import env


def wrap_with_fleet(dist_config, model, optimizer=None, scaler=None):
    if dist_config.sharding.sharding_stage in [2, 3]:
        assert dist_config.pp_degree == 1, "sharding stage2/3 will support pipeline parallel later"
        return wrap_sharding_2_3(dist_config, model, optimizer, scaler)
    else:
        return wrap_3D_parallel(dist_config, model, optimizer, scaler)


def wrap_sharding_2_3(dist_config, model, optimizer=None, scaler=None):
    hcg = env.get_hcg()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    if dist_config.dp_degree > 1 and dist_config.sharding.sharding_stage == 3:
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.mp_degree > 1:
        assert dist_config.sharding.sharding_stage == 2, "only support mp + sharding stage2 hybrid parallel now."
        model = TensorParallel(model, hcg, strategy=None)

    level = "p_g_os" if dist_config.sharding.sharding_stage == 3 else "os_g"
    origin_model = model
    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=dist_config.sharding.sharding_offload,
        dp_group=dp_group if dp_group.nranks > 1 else None,
    )

    if dist_config.sharding.reduce_overlap:
        model._set_reduce_overlap(dist_config.sharding.reduce_overlap)

    if dist_config.sharding.broadcast_overlap:
        optimizer._set_broadcast_overlap(dist_config.sharding.broadcast_overlap, layers=origin_model, num_groups=2)

    return model, optimizer, scaler


def wrap_3D_parallel(dist_config, model, optimizer=None, scaler=None):
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer) if optimizer is not None else optimizer
    scaler = fleet.distributed_scaler(scaler) if scaler is not None else scaler

    return model, optimizer, scaler
