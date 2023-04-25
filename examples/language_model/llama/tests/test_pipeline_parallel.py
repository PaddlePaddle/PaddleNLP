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

import unittest

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from modeling_pp import LlamaForCausalLMPipe
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import PipelineParallel

from paddlenlp.transformers import LlamaForCausalLM


class TestLlama(unittest.TestCase):
    def test_pipeline_model(self):
        world_size = paddle.distributed.get_world_size()
        pp_degree = world_size
        tp_degree = 1
        if world_size > 2:
            pp_degree = 2
            assert world_size % pp_degree == 0
            tp_degree = world_size // pp_degree

        # pp_degree = -1
        if pp_degree == -1:
            tp_degree = world_size
            pp_degree = 1

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tp_degree,
            "pp_degree": pp_degree,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()

        if pp_degree > 1:
            model_class = LlamaForCausalLMPipe
        else:
            model_class = LlamaForCausalLM

        model = model_class.from_pretrained(
            "facebook/tiny-random-llama",
            tensor_parallel_degree=tp_degree,
            tensor_parallel_rank=hcg.get_model_parallel_rank(),
            lm_shift_labels=False,
            # use_flash_attention=True,
        )

        model.eval()

        for k, v in model.state_dict().items():
            print(k, v.shape, v.dtype, v.abs().sum().item())
            if k == "lm_head.weight":
                print(v)

        input_ids = paddle.to_tensor([[x for x in range(100, 110)]], dtype="int64")
        labels = paddle.to_tensor([[x for x in range(101, 111)]], dtype="int64")

        if pp_degree > 1:
            pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            ret = pp_model.eval_batch(data=[input_ids, labels], compute_loss=True)
        else:
            # pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            # pp_model.data = [input_ids, labels]
            # ret = pp_model._forward_step(None)
            ret = model(input_ids=input_ids, labels=labels)
            ret = ret[0]

        np.testing.assert_allclose(ret.item(), 10.49988270, atol=1e-7)
        print("ret", ret.item())

        # single_model = LlamaForCausalLM.from_pretrained("facebook/tiny-random-llama", lm_shift_labels=False)
        # single_model.eval()
        # ret = single_model(input_ids=input_ids, labels=labels)
        # np.testing.assert_allclose(ret[0].item(), 10.49988270, atol=1e-7)

        # print("ret", ret[0].item())


if __name__ == "__main__":
    TestLlama().test_pipeline_model()

# 3 bugs to fix in paddlepaddle
# pp_layers.py
# def _construct_shared_comm(self):
#     shared_comm = {}
#     if self._topo.get_dim("pipe") == 1:
#         return shared_comm

# topology.py
# def _set_p2p_group(self):
#     self.send_next_group = None
#     self.send_prev_group = None
#     self.recv_next_group = None
#     self.recv_prev_group = None
#     if self._pp_degree <= 1:
#         return

# pipeline_parallel.py
# def _load_micro_batch(self, cache_id, stage=None):
#     inputs = self.data
#     if stage == "fisrt":
#         assert self.is_pipeline_first_stage()
#         assert len(inputs) == 2, "length of input should be 2"
#         return self._load_micro_batch_impl(inputs[0], cache_id)
#     elif stage== "last":
#         assert self.is_pipeline_last_stage()
#         assert len(inputs) == 2, "length of input should be 2"
#         return self._load_micro_batch_impl(inputs[1], cache_id)
#     else:
#         inputs = None
#
#
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./ pytest -s -v tests/test_pipeline_parallel.py
# PYTHONPATH=/ssd2/zhonghui03/Datasets/PaddleNLP:$PYTHONPATH  PYTHONPATH=$PYTHONPATH:./ python   -m paddle.distributed.launch --gpus 0,1,2,3  tests/test_pipeline_parallel.py
