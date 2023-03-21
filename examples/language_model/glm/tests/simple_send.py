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

import time

import numpy as np
import paddle
import paddle.distributed as dist

data = None
dist.init_parallel_env()
local_rank = dist.get_rank()
gp = paddle.distributed.new_group([0, 1], backend="gloo")
tensor_list = []

paddle.set_device("cpu")

if local_rank == 0:
    data = {"foo": 100, "bar": 5, "cat": 650}
else:
    data = {"new": 100, "zero": 5, "commit": 650}
task = dist.all_gather_object(tensor_list, data, group=gp)
print("\n\nTest all gather object:")
print(tensor_list)

# if dist.get_rank() == 0:
#     data = paddle.to_tensor([7, 8, 9])
#     print(data)
#     dist.send(data, dst=1, group=gp)
# else:
#     data = paddle.to_tensor([1, 2, 3])
#     dist.recv(data, src=0, group=gp)
# print(data)


paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)

# if paddle.distributed.ParallelEnv().local_rank == 0:
#   np_data = np.array([[4, 5, 6], [4, 5, 6]])
# else:
#   np_data = np.array([[1, 2, 3], [1, 2, 3]])
#
# data = paddle.to_tensor(np_data)
# print("\n\nTest all gather object:")
# print("data", data)
# paddle.distributed.broadcast(data, 1)
# out = data.numpy()
# print(out)

dtype = "float32"
shape = [32, 1024, 1024]
repeat = 102
numel = np.prod(shape)
print("numel:", numel)

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)

if dist.get_rank() == 0:
    data = paddle.randn(shape, dtype=dtype)
else:
    data = paddle.empty(shape=shape, dtype=dtype)


s1 = time.time()
calc_stream = False
for i in range(repeat):
    if dist.get_rank() == 0:
        # data = paddle.randn(shape, dtype=dtype)
        dist.send(data, dst=1)  # , use_calc_stream=calc_stream)
    else:
        # data = paddle.empty(shape=shape,dtype=dtype)
        dist.recv(data, src=0)  # , use_calc_stream=calc_stream)
        data.cpu()
        # data.numpy()

print(data.numel())
s2 = time.time()
print("Using time: ", s2 - s1)
print(4 * numel * repeat / 2**30 / (s2 - s1), "GB/s")

# print(data)

exit(0)
