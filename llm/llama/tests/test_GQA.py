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

from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoModelForCausalLMPipe,AutoTokenizer


class TestLlama(unittest.TestCase):
    def test_GQA(self):
        world_size = paddle.distributed.get_world_size()
        pp_degree = world_size
        tp_degree = 1

        if world_size > 2:
            pp_degree = 2
            assert world_size % pp_degree == 0
            tp_degree = world_size // pp_degree

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tp_degree,
            "pp_degree": pp_degree,
            "sharding_degree": 1,
        }
        #strategy.pipeline_configs = {"enable_partial_send_recv": False if pp_degree > 1 else True}
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        tensor_parallel_rank = mp_group.rank

        if pp_degree > 1:
            model_class = AutoModelForCausalLMPipe
        else:
            model_class = AutoModelForCausalLM

        model_name_or_path = "meta-llama/Llama-2-7b"

        seq_len = 2048
        batch_size = 2

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.seq_length = seq_len
        config.num_key_value_heads = 8 # gqa
        config.max_position_embeddings = max(config.max_position_embeddings, seq_len)
        config.vocab_size = max(config.vocab_size, ((tokenizer.vocab_size - 1) // 128 + 1) * 128)
        config.use_flash_attention = True
        config.use_fused_rope = True
        config.use_fused_rms_norm = True
        config.fuse_attention_qkv = True
        config.recompute_granularity = "full"
        config.virtual_pp_degree = 1
        config.use_recompute = False

        config.tensor_parallel_degree = tp_degree
        config.tensor_parallel_rank = tensor_parallel_rank
        config.tensor_parallel_output = False
        config.sequence_parallel = False

        config.fuse_sequence_parallel_allreduce = False

        # hidden_size = 4096
        model = model_class.from_config(
            config,
            dtype="float16",
        )

        model.eval()

        input_ids = paddle.arange(100, 100 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])
        labels = paddle.arange(101, 101 + batch_size * seq_len, dtype="int64").reshape([batch_size, seq_len])
    
        attention_mask = None
        if pp_degree > 1:
            pp_model = PipelineParallel(layers=model, hcg=hcg, strategy=strategy)
            pp_model.accumulate_steps = batch_size  # for micro_batch_size * acc_steps == batch_size
            ret = pp_model.eval_batch(data=[input_ids, labels], compute_loss=True)
        else:
            ret = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            ret = ret[0]

        print(f"ret mp{tp_degree} pp{pp_degree}", ret.item())
        ret_mp_pp = ret.item()
        




if __name__ == "__main__":
    TestLlama().test_GQA()
