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

        model_name_or_path = "gpt2-medium-en"

        # segment with method: layer:GPTDecoderLayer; result: 0, 13, 27
        # stage=0, global_rank=0 ,layer_number=13
        # 0: GPTEmbeddingPipe
        # 1: GPTDecoderLayerPipe
        # ......
        # 12: GPTDecoderLayerPipe
        # stage=1, global_rank=0 ,layer_number=14
        # ......
        # 25: LayerNormPipe
        # 26: GPTLMHead                  Note: GPTLMHead is not in ckpt!
        # loss: GPTPretrainingCriterion

        seq_len = 1024
        batch_size = 2

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
        config.sequence_parallel = True

        config.fuse_sequence_parallel_allreduce = False

        input_ids = paddle.arange(100, 100 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])
        labels = paddle.arange(101, 101 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])

        single_model = GPTForCausalLM.from_pretrained(model_name_or_path, config=config, dtype="float32")
        single_model.eval()
        ret_single = single_model(input_ids=input_ids, labels=labels)
        single_model_state_dict = single_model.state_dict()

        # hidden_size = 4096
        model = model_class.from_pretrained(model_name_or_path, config=config, dtype="float32")
        model.set_state_dict(single_model_state_dict)
        model.eval()

        if pp_degree > 1:
            pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            pp_model.accumulate_steps = batch_size  # for micro_batch_size * acc_steps == batch_size
            ret = pp_model.eval_batch(data=[input_ids, labels], compute_loss=True)
        else:
            # pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            # pp_model.data = [input_ids, labels]
            # ret1 = pp_model._forward_step(None)
            ret = model(input_ids=input_ids, labels=labels)
            ret = ret[0]

        print(f"ret mp{tp_degree} pp{pp_degree}", ret.item())
        ret_mp_pp = ret.item()

        print("ret single", float(ret_single[0]))
        print(
            f"diff: {(float(ret_single[0])- ret_mp_pp)/float(ret_single[0])}",
        )
        np.testing.assert_allclose(float(ret_single[0]), ret_mp_pp, rtol=1.5e-7)


if __name__ == "__main__":
    TestGPT().test_sequence_model()
# python -m paddle.distributed.launch --gpus 0,1,2,3  tests/test_pipeline_parallel.py
