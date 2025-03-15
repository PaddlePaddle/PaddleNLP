# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
    ["model_dir"],
    [
        ["llama"],
        # ["chatglm"], @skip("Skip and wait to fix.")
        # ["chatglm2"], @skip("Skip and wait to fix.")
        # ["bloom"], @skip("Skip and wait to fix.")
        ["qwen"],
        ["baichuan"],
    ],
)
class LoRAProTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/lorapro.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        sys.path.remove(self.model_codes_dir)

    def test_lorapro(self):
        self.disable_static()
        paddle.set_default_dtype("float32")

        lora_config = load_test_config(self.config_path, "lorapro", self.model_dir)
        lora_config["output_dir"] = self.output_dir
        lora_config["dataset_name_or_path"] = self.data_dir
        # use_quick_lora
        lora_config["use_quick_lora"] = True

        with argv_context_guard(lora_config):
            from run_finetune import main

            main()

        # merge weights
        merge_lora_weights_config = {
            "lora_path": lora_config["output_dir"],
            "model_name_or_path": lora_config["model_name_or_path"],
            "output_path": lora_config["output_dir"],
        }
        with argv_context_guard(merge_lora_weights_config):
            from tools.merge_lora_params import merge

            merge()

        # TODO(wj-Mcat): disable chatglm2 test temporarily
        if self.model_dir not in ["qwen", "baichuan", "chatglm2", "llama"]:
            self.run_predictor({"inference_model": True})

        self.run_predictor({"inference_model": False})
