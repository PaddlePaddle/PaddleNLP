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
import unittest

import paddle

from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from tests.testing_utils import require_gpu
from tests.transformers.test_modeling_common import ids_tensor


class ShardingStage3Tester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-llama")
        model = AutoModelForCausalLM.from_pretrained("__internal_testing__/tiny-random-llama")
        model.config.eos_token_id = -1
        world_size = paddle.distributed.get_world_size()

        with tempfile.TemporaryDirectory() as tempdir:
            args_dict = {
                "sharding": "stage3",
                "sharding_parallel_degree": world_size,
                "fp16": True,
                "fp16_opt_level": "O2",
                "output_dir": os.path.join(tempdir, "output"),
            }
            parser = PdArgumentParser((TrainingArguments,))
            args = parser.parse_dict(args_dict)[0]
            trainer = Trainer(model, args=args, tokenizer=tokenizer)
            trainer.create_optimizer_and_scheduler(num_training_steps=10)
            trainer._wrap_model(trainer.model_wrapped)
        cls.model = trainer.model
        cls.model.eval()
        input_ids = ids_tensor([1, 5], vocab_size=model.config.vocab_size, dtype="int64")
        attention_mask = paddle.ones_like(input_ids, dtype="bool")
        cls.input_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "synced_gpus": True,
        }
        cls.generation_config = GenerationConfig(max_length=10 + paddle.distributed.get_rank(), trunc_input=False)

    @require_gpu(2)
    def test_synced_gpus_greedy(self):
        with paddle.no_grad():
            self.generation_config.decode_strategy = "greedy_search"
            self.model.generate(**self.input_kwargs, generation_config=self.generation_config)

    @require_gpu(2)
    def test_synced_gpus_sample(self):
        with paddle.no_grad():
            self.generation_config.decode_strategy = "sampling"
            self.generation_config.top_k = 8
            self.model.generate(**self.input_kwargs, generation_config=self.generation_config)

    @require_gpu(2)
    def test_synced_gpus_beam_search(self):
        with paddle.no_grad():
            self.generation_config.decode_strategy = "beam_search"
            self.generation_config.num_beams = 4
            self.model.generate(**self.input_kwargs, generation_config=self.generation_config)

    @require_gpu(2)
    def test_synced_gpus_group_beam_search(self):
        with paddle.no_grad():
            self.generation_config.decode_strategy = "beam_search"
            self.generation_config.num_beams = 4
            self.generation_config.num_beam_groups = 2
            self.model.generate(**self.input_kwargs, generation_config=self.generation_config)


if __name__ == "__main__":
    # ShardingStage3Tester().test_synced_gpus_sample()
    unittest.main()

# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./ pytest -s -v tests/test_synced_gpus.py
# PYTHONPATH=/ssd2/zhonghui03/Datasets/PaddleNLP:$PYTHONPATH  PYTHONPATH=$PYTHONPATH:./ python   -m paddle.distributed.launch --gpus 0,1,2,3  tests/test_pipeline_parallel.py
