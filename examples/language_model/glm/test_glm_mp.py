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


class TestGLM(TestMultipleGpus):
    # @unittest.skip("")
    def testGlmMP(self):
        self.run_2gpu("glm_mp.py")


class TestCkptShard(unittest.TestCase):
    # def setUp(self):
    #     sys.path.insert(0, ".")
    #     print("xxxxxx")
    def test_import(self):
        import inspect

        import paddlenlp

        print(inspect.getfile(paddlenlp))

    # @unittest.skip("")
    def testTorch(self):
        from transformers import AutoModel

        model = AutoModel.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)
        model.eval()
        loss = model(input_ids=torch.arange(100, 110, dtype=torch.long).reshape(1, -1))
        # print(ret)
        ret = loss.logits.abs().mean().item()
        np.testing.assert_allclose(ret, 2.1089835166931152, rtol=1e-7)
        print("torch", ret)

    @unittest.skip("")
    def testPaddle(self):
        from paddlenlp.transformers import AutoModel

        model = AutoModel.from_pretrained("THUDM/glm-large-chinese")
        model.eval()
        loss = model(input_ids=paddle.arange(100, 110, dtype="int64").reshape([1, -1]))
        ret = loss.logits.abs().mean().item()
        np.testing.assert_allclose(ret, 2.1089835166931152, rtol=1e-7)
        print("paddle", ret)

    def test_qkv_convertor(self):
        """test_qkv_convertor"""
        hidden_size = 8
        mp_degree = 4
        num_attention_heads = 4
        # head_dim = hidden_size // num_attention_heads

        from paddlenlp.transformers.conversion_utils import (
            merge_tensor_parallel_weight,
            naive_merged_qkv_to_tensor_parallel_qkv,
            split_tensor_parallel_weight,
            tensor_parallel_qkv_to_naive_merged_qkv,
        )

        naive_merged_qkv = np.arange(3 * hidden_size * hidden_size).reshape([hidden_size, -1])
        tensor_parallel_qkv = naive_merged_qkv_to_tensor_parallel_qkv(naive_merged_qkv, num_attention_heads)
        new_naive_merged_qkv = tensor_parallel_qkv_to_naive_merged_qkv(tensor_parallel_qkv, num_attention_heads)
        np.testing.assert_equal(new_naive_merged_qkv, naive_merged_qkv)
        # print("tensor_parallel_qkv", tensor_parallel_qkv)
        np.testing.assert_equal(
            tensor_parallel_qkv[0],
            [0, 1, 8, 9, 16, 17, 2, 3, 10, 11, 18, 19, 4, 5, 12, 13, 20, 21, 6, 7, 14, 15, 22, 23],
        )

        mp_qkv_splited = split_tensor_parallel_weight(tensor_parallel_qkv, mp_degree)
        new_tensor_parallel_qkv = merge_tensor_parallel_weight(mp_qkv_splited)
        # print("mp_qkv_splited", mp_qkv_splited[0])
        np.testing.assert_equal(new_tensor_parallel_qkv, tensor_parallel_qkv)
        np.testing.assert_equal(mp_qkv_splited[0][0], [0, 1, 8, 9, 16, 17])

        # raise ValueError()

    @unittest.skip("Skip for reuqired multi-gpus!")
    def testGlmMP(self):
        """_summary_"""
        from modeling import GLMModel as AutoModel

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
        model = AutoModel.from_pretrained(
            "THUDM/glm-large-chinese", from_hf=True, mp_degree=mp_degree, mp_rank=mp_rank
        )
        model.eval()
        model()
