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
from safetensors import safe_open

# from safetensors.numpy import save_file

rank = paddle.distributed.get_rank()
world_size = paddle.distributed.get_world_size()


def paddle_load(path):
    path = path + ".pdparams"
    weight = paddle.load(path, return_numpy=True)
    # save_file(weight, path.replace("pdparams", "safetensors" ))

    final_weight = {}
    for k, v in weight.items():
        final_weight[k] = np.split(v, world_size, axis=-1)[rank]
    return final_weight


def safe_load(path):
    path = path + ".safetensors"
    final_weight = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            slice_ = f.get_slice(key)
            size = slice_.get_shape()[0]
            block_size = size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
            tensor = slice_[start:stop]
            final_weight[key] = tensor
            print(key, tensor.shape)
    return final_weight


# name = "llama-13b"
name = "model_state"

if __name__ == "__main__":
    import time

    # s1 =  time.time()
    # a = paddle_load(name)
    # s2 = time.time()
    # print(s2 - s1)

    s3 = time.time()
    b = safe_load(name)
    print(b["lm_head.weight"])
    s4 = time.time()
    print(s4 - s3)
