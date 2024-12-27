# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import unittest

import paddle
import paddle.distributed.fleet as fleet

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from tests.llm.testing_utils import LLMTest


class GpusInference(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = AutoModelForCausalLM

    def __init__(self, model_name_or_path):
        super().__init__()
        self.setUp()
        self.init_dist_env()
        self.model_name_or_path = model_name_or_path
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def init_dist_env(self, config: dict = {}):
        world_size = paddle.distributed.get_world_size()
        strategy = fleet.DistributedStrategy()
        hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        hybrid_configs.update(config)
        strategy.hybrid_configs = hybrid_configs

        fleet.init(is_collective=True, strategy=strategy)
        fleet.get_hybrid_communicate_group()

    def run_inference(self, out_path):
        config_params = {"inference_model": True, "append_attn": True, "max_length": 48, "output_file": out_path}
        self.run_predictor(config_params)

    def tearDown(self):
        LLMTest.tearDown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="the golden result")
    parser.add_argument("--tensor_parallel_degree", type=int, default="1", help="Path to the output directory")
    parser.add_argument("--pipeline_parallel_degree", type=int, default="1", help="Path to the output directory")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="the golden result")
    args = parser.parse_args()

    inference = GpusInference(args.model_name_or_path)
    inference.run_inference(args.save_path)
