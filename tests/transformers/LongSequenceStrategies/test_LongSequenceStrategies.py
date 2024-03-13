# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import sys
import unittest

from parameterized import parameterized_class

from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest

from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoModel

@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/tiny-random-llama", LlamaForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm", ChatGLMForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm2", ChatGLMv2ForCausalLM],
        ["__internal_testing__/tiny-fused-qwen-inference5.2", QWenForCausalLM],
    ],
)
@parameterized_class.expand(
    ["strategy_type", "strategy_name"],
    [
        ["EmbeddingStrategies", "RotaryEmbedding"],
        ["EmbeddingStrategies", "NTKScalingRotaryEmbedding"],
        ["EmbeddingStrategies", "LinearScalingRotaryEmbedding"],
        ["EmbeddingStrategies", "DynamicNTKScalingRotaryEmbedding"],
        ["AttentionStrategies", "AttentionWithLinearBias"],
    ],
)
class TestLongSequenceStrategiesTest(LLMTest, unittest.TestCase):
    inference_config = ""
    model_dir: str = ""

    def setUp(self) -> None:
        LLMTest.setUp(self)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_long_sequence_strategies(self):
        input = "中国的首都是"
        model_config = AutoConfig.from_pretrained(
           self.model_name_or_path
        )
        if self.strategy_type == "EmbeddingStrategies":
            model_config.alibi = False 
        else:
            model_config.alibi = True 
        model_config.use_long_strategies = True
        model_config.long_sequence_strategy_type = self.strategy_type
        model_config.long_sequence_strategy_name = self.strategy_name
        if self.strategy_name == "DynamicNTKScalingRotaryEmbedding":
            model_config.max_position_embeddings =  1024
        else:
            model_config.max_position_embeddings =  2048
        if model_config.alibi:
            model_config.long_sequence_init_args = {}
        else:
            if self.model_name_or_path in ["__internal_testing__/tiny-fused-chatglm" , "__internal_testing__/tiny-fused-chatglm2"]
                position_encoding_2d = True
            else:
                position_encoding_2d = False
            model_config.long_sequence_init_args ={"dim": int(model_config.hidden_size / model_config.num_attention_heads) 
            ,"max_position_embeddings":model_config.max_position_embeddings,"base":10000,"scaling_factor":1,"position_encoding_2d":position_encoding_2d}
        finetune_config = load_test_config(self.config_path, "finetune", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        with argv_context_guard(finetune_config):
            from finetune_generation import main

            main()

        # TODO(wj-Mcat): disable chatglm2 test temporarily
        if self.model_dir not in ["qwen", "baichuan", "chatglm2"]:
            self.run_predictor({"inference_model": True})

        self.run_predictor({"inference_model": False})
        
        

