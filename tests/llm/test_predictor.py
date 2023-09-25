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

import os
import unittest

from parameterized import parameterized_class

from paddlenlp.transformers import AutoTokenizer, LlamaForCausalLM

from .testing_utils import LLMTest


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [["__internal_testing__/tiny-random-llama", LlamaForCausalLM]],
)
class PredictorTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_predictor(self):
        self.run_predictor({"inference_model": True})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)
        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[min_length // 2] == no_inference_item[min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.25)
        self.assertGreater(count / len(result_0), 0.4)
