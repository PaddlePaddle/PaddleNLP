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

import paddle

from paddlenlp.transformers import GLMModel

tensor_parallel_degree = paddle.distributed.get_world_size()
tensor_parallel_rank = paddle.distributed.get_rank()
strategy = paddle.distributed.fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": tensor_parallel_degree,
    "pp_degree": 1,
    "sharding_degree": 1,
}
paddle.distributed.fleet.init(is_collective=True, strategy=strategy)


# def func(self, *args, **kwargs):
#     return

# # 屏蔽init_weights
# GLMModel.init_weights = func
# paddle.set_default_dtype("float16")

model = GLMModel.from_pretrained(
    "THUDM/glm-large-chinese",
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
    # dtype="float16",
    low_cpu_mem_usage=True,
)

model.eval()
ret = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))

print("paddle mp", ret.logits.abs().mean().item())
