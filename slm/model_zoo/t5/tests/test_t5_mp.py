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

# import sys
import unittest

import numpy as np
import paddle
import torch
from test_parallel_dygraph_dataparallel import TestMultipleGpus

import paddlenlp


def load_torch(path, *args, **kwargs):
    import torch

    state = torch.load(path, map_location="cpu")
    for key in list(state.keys()):
        v = state.pop(key)
        state[key] = v.numpy()
    return state


# hack load torch, it has problem to load torch ckpt.
paddlenlp.utils.serialization.load_torch = load_torch
paddlenlp.transformers.conversion_utils.load_torch = load_torch


class TestT5(unittest.TestCase):
    def testTorchT5(self):
        from transformers import AutoModel

        model = AutoModel.from_pretrained("t5-small", trust_remote_code=True)
        model.eval()
        loss = model(
            input_ids=torch.arange(100, 110, dtype=torch.long).reshape(1, -1),
            decoder_input_ids=torch.arange(100, 105, dtype=torch.long).reshape(1, -1),
        )
        ret = loss.last_hidden_state.abs().mean().item()
        # Torch T5 has bug in GELU activation
        np.testing.assert_allclose(ret, 0.1365441530942917, rtol=1e-7)

    def testConvertedPaddleT5(self):
        from paddlenlp.transformers import AutoModel

        model = AutoModel.from_pretrained("t5-small", from_hf_hub=True)
        model.eval()
        loss = model(
            input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]),
            decoder_input_ids=paddle.arange(100, 105, dtype="int64").reshape([1, -1]),
            return_dict=True,
        )
        ret = loss.last_hidden_state.abs().mean().item()
        np.testing.assert_allclose(ret, 0.1365441381931305, rtol=1e-7)

    @unittest.skip("Skip export!")
    def testPaddleT5(self):
        from paddlenlp.transformers import T5Model

        model = T5Model.from_pretrained("t5-small", dtype="float32")
        model.eval()
        loss = model(
            input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]),
            decoder_input_ids=paddle.arange(100, 105, dtype="int64").reshape([1, -1]),
            return_dict=True,
        )
        ret = loss.last_hidden_state.abs().mean().item()
        np.testing.assert_allclose(ret, 0.1365441381931305, rtol=1e-7)

        # # dy2static
        # input_spec = [
        #     paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
        #     paddle.static.InputSpec(shape=[None, 2, None], dtype="int64"),  # pos_ids
        #     paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),  # attn_ids
        # ]
        # with tempfile.TemporaryDirectory() as tempdir:
        #     paddlenlp.transformers.export_model(
        #         model=model,
        #         input_spec=input_spec,
        #         path=tempdir,
        #     )

    # TODO: support @ decorate for multi-gpus tests
    @unittest.skip("Skip for reuqired multi-gpus!")
    def testPaddleTensorParallelT5(self):
        """_summary_"""
        from modeling import T5Model as AutoModel

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
        model = AutoModel.from_pretrained(
            "t5-small",
            from_hf=True,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
        )
        model.eval()


class TestT5TensorParallel(TestMultipleGpus):
    def testPaddleTensorParallelT5(self):
        self.run_4gpu("t5_mp.py")
