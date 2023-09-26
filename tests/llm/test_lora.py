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
import sys
import unittest

import paddle
from parameterized import parameterized_class

from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir", "enable_compare"],
    [
        ["llama", False],
        # TODO(wj-Mcat): to enable chatglm/chatglm2 unit test
        # ["chatglm"],
        # ["chatglm2"],
        ["bloom"],
    ],
)
class LoraTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/lora.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        sys.path.remove(self.model_codes_dir)

    def test_lora(self):
        self.disable_static()
        paddle.set_default_dtype("float32")

        lora_config = load_test_config(self.config_path, "lora", self.model_dir)

        lora_config["dataset_name_or_path"] = self.data_dir
        lora_config["output_dir"] = self.output_dir

        with argv_context_guard(lora_config):
            from finetune_generation import main

            main()

        # merge weights
        merge_lora_weights_config = {
            "model_name_or_path": lora_config["model_name_or_path"],
            "lora_path": lora_config["output_dir"],
            "merge_model_path": lora_config["output_dir"],
        }
        with argv_context_guard(merge_lora_weights_config):
            from merge_lora_params import merge

            merge()

        if self.model_dir not in ["chatglm2"]:
            self.run_predictor({"inference_model": True})

        self.run_predictor({"inference_model": False})
