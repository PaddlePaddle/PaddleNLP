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
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import PipelineParallel

from paddlenlp.transformers import GPTConfig, GPTForCausalLM, GPTForCausalLMPipe


class TestGPT(unittest.TestCase):
    def test_sequence_model(self):
        model_name_or_path = "gpt2-medium-en"
        seq_len = 1024
        batch_size = 2
        input_ids = paddle.arange(100, 100 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])
        labels = paddle.arange(101, 101 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])

        world_size = paddle.distributed.get_world_size()
        pp_degree = 2
        tp_degree = world_size // pp_degree
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tp_degree,
            "pp_degree": pp_degree,
            "sharding_degree": 1,
        }
        strategy.pipeline_configs = {"enable_partial_send_recv": False if pp_degree > 1 else True}
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        tensor_parallel_rank = mp_group.rank

        if pp_degree > 1:
            model_class = GPTForCausalLMPipe
        else:
            model_class = GPTForCausalLM

        config = GPTConfig.from_pretrained(model_name_or_path)
        config.seq_length = seq_len
        config.use_flash_attention = False
        config.fuse_attention_qkv = False
        config.recompute_granularity = "full"
        config.virtual_pp_degree = 1
        config.use_recompute = False

        config.tensor_parallel_degree = tp_degree
        config.tensor_parallel_rank = tensor_parallel_rank
        config.tensor_parallel_output = False
        # when tp_degree > 1, sequence_parallel can be set to True
        config.sequence_parallel = True
        config.fuse_sequence_parallel_allreduce = False

        model = model_class.from_pretrained(model_name_or_path, config=config, dtype="float32")
        model.eval()

        if pp_degree > 1:
            pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            pp_model.accumulate_steps = batch_size  # for micro_batch_size * acc_steps == batch_size
            ret_mp_pp = pp_model.eval_batch(data=[input_ids, labels], compute_loss=True)
        else:
            ret_mp_pp = model(input_ids=input_ids, labels=labels)[0]

        # run model for single device
        config.tensor_parallel_degree = 1
        config.tensor_parallel_rank = -1
        config.sequence_parallel = False
        single_model = GPTForCausalLM.from_pretrained(model_name_or_path, config=config, dtype="float32")
        single_model.eval()
        ret_single = single_model(input_ids=input_ids, labels=labels)[0]

        # output all results
        print(f"ret mp{tp_degree} pp{pp_degree}", float(ret_mp_pp))
        print("ret single", float(ret_single))

        diff = (ret_single - ret_mp_pp) / ret_single
        print(f"diff: {float(diff)}")
        np.testing.assert_allclose(float(ret_single), ret_mp_pp, rtol=1.5e-7)


if __name__ == "__main__":
    TestGPT().test_sequence_model()
# python -m paddle.distributed.launch --gpus 0,1,2,3  tests/test_pipeline_parallel.py
