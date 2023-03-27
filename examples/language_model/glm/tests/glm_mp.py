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
import tempfile

import numpy as np
import paddle

from paddlenlp.transformers import GLMModel

GLMModel.init_weights = lambda *_: None


def main():
    world_size = paddle.distributed.get_world_size()
    dp_degree = 2 if world_size >= 4 else 1
    tensor_parallel_degree = world_size // dp_degree

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": dp_degree,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

    hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    tensor_parallel_rank = mp_group.rank

    model = GLMModel.from_pretrained(
        "THUDM/glm-large-chinese",
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
        dtype="float16",
        low_cpu_mem_usage=True,
    )

    model.eval()
    loss = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))
    ret = loss.logits.abs().mean().item()
    np.testing.assert_allclose(ret, 2.109375, rtol=1e-4)

    with tempfile.TemporaryDirectory() as tempdir:
        model.save_pretrained(save_dir=tempdir, merge_tensor_parallel=False)
        paddle.distributed.barrier()
        print(os.listdir(tempdir))
        load_model = GLMModel.from_pretrained(
            tempdir,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            dtype="float16",
            low_cpu_mem_usage=True,
        )
        load_model.eval()
        loss = load_model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))
        ret = loss.logits.abs().mean().item()
        np.testing.assert_allclose(ret, 2.109375, rtol=1e-4)

    with tempfile.TemporaryDirectory() as tempdir:
        object_list = []
        paddle.distributed.all_gather_object(object_list, tempdir, group=mp_group)
        tempdir = object_list[0]
        model.save_pretrained(save_dir=tempdir, merge_tensor_parallel=True)
        paddle.distributed.barrier()
        print(os.listdir(tempdir))
        load_model = GLMModel.from_pretrained(
            tempdir,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            dtype="float16",
            low_cpu_mem_usage=True,
        )
        load_model.eval()
        loss = load_model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))
        ret = loss.logits.abs().mean().item()
        np.testing.assert_allclose(ret, 2.109375, rtol=1e-4)


if __name__ == "__main__":
    main()
