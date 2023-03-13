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
from modeling import GLMModel

# from paddlenlp.transformers import GLMConfig

mp_degree = paddle.distributed.get_world_size()
mp_rank = paddle.distributed.get_rank()
strategy = paddle.distributed.fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": mp_degree,
    "pp_degree": 1,
    "sharding_degree": 1,
}
paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

# model = AutoModel.from_pretrained("THUDM/glm-large-chinese",
# config=GLMConfig.from_pretrained("glm-large-chinese")
# config.mp_degree=mp_degree
# config.mp_rank=mp_rank

# from modeling import GLMModel;

model = GLMModel.from_pretrained(
    "THUDM/glm-large-chinese",
    from_hf_hub=True,
    mp_degree=mp_degree,
    mp_rank=mp_rank,
)

model.eval()
ret = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))

# torch 2.1089835166931152
print("paddle mp", ret.logits.abs().mean().item())
