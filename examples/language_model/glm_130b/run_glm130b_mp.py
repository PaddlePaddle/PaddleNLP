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

import numpy as np
import paddle
from modeling import GLM130BConfig, GLM130BModel

paddle.set_default_dtype("float16")

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

config = GLM130BConfig(
    **{
        "hidden_size": 192,
        "mlp_hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "length_per_sample": 100,
        "max_length": 128,
        "vocab_size_base": 768,
        "activation": "geglu",
        "layernorm_epsilon": 1e-5,
        "paddle_dtype": "float16",
        "attention_dropout_prob": 0,
        "attention_scale": True,
        "embedding_dropout_prob": 0,
        "initializer_range": 0.0052,
        "output_dropout_prob": 0,
        "output_predict": True,
        "position_encoding_2d": False,
        "recompute": False,
        "vocab_size": 150528,
    }
)

model = GLM130BModel.from_pretrained(
    "local_random_glm",
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
    low_cpu_mem_usage=True,
)

model.eval()

input_ids = paddle.to_tensor(np.load("torch_cache/torch_input_ids.npy"))
position_ids = paddle.to_tensor(np.load("torch_cache/torch_position_ids.npy"))
attention_mask = paddle.to_tensor(np.load("torch_cache/torch_attention_mask.npy"))

result = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)

print("paddle mp", result.logits.abs().mean().item())
