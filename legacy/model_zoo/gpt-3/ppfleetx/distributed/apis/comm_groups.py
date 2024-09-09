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

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.base.orthogonal_strategy import OrthogonalStrategy
from paddle.distributed.fleet.base.strategy_group import (
    DPGroup,
    MPGroup,
    PPGroup,
    ShardingGroup,
)


def create_hcg(strategy, hcg_name):
    if hcg_name == "HybridCommunicateGroup":
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
    else:
        dist.init_parallel_env()
        hcg = eval("{}".format(hcg_name))(strategy)

    return hcg


class Hybrid4DCommGroup(OrthogonalStrategy):
    def __init__(self, list_of_strategy=None, fused_strategy_dict={}):
        list_of_strategy = (
            [
                ("dp", 1, DPGroup),
                ("mp", 1, MPGroup),
                ("pp", 1, PPGroup),
                ("sharding", 1, ShardingGroup),
            ]
            if list_of_strategy is None
            else list_of_strategy
        )

        fused_strategy_dict["check"] = ["mp", "pp"]

        super().__init__(list_of_strategy, fused_strategy_dict)

    # data parallel
    def get_data_parallel_rank(self):
        return self.rank_in_strategy("dp")

    def get_data_parallel_world_size(self):
        return self.strategy_group("dp").world_size

    def get_data_parallel_group(self):
        return self.strategy_group("dp").group

    def get_data_parallel_group_src_rank(self):
        return self.strategy_group("dp").group.ranks[0]

    # tensor parallel
    def get_model_parallel_rank(self):
        return self.rank_in_strategy("mp")

    def get_model_parallel_world_size(self):
        return self.strategy_group("mp").world_size

    def get_model_parallel_group(self):
        return self.strategy_group("mp").group

    def get_model_parallel_group_src_rank(self):
        return self.strategy_group("mp").group.ranks[0]

    # pipeline parallel
    def get_stage_id(self):
        return self.rank_in_strategy("pp")

    def get_pipe_parallel_world_size(self):
        return self.strategy_group("pp").world_size

    def get_pipe_parallel_group(self):
        return self.strategy_group("pp").group

    def get_p2p_groups(self):
        return self.strategy_group("pp").p2p_groups

    # group sharded parallel
    def get_sharding_parallel_rank(self):
        return self.rank_in_strategy("sharding")

    def get_sharding_parallel_world_size(self):
        return self.strategy_group("sharding").world_size

    def get_sharding_parallel_group(self):
        return self.strategy_group("sharding")

    def get_sharding_parallel_group_src_rank(self):
        return self.strategy_group("sharding").ranks[0]

    # check parallel group
    def get_check_parallel_group(self):
        return self.strategy_group("check").group
