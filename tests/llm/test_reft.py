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
    ["model_dir"],
    [
        ["llama"],
        ["baichuan"],
    ],
)
class ReftTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/reft.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        sys.path.remove(self.model_codes_dir)

    def test_reft(self):
        self.disable_static()
        paddle.set_default_dtype("float32")

        reft_config = load_test_config(self.config_path, "reft", self.model_dir)
        reft_config["output_dir"] = self.output_dir
        reft_config["dataset_name_or_path"] = self.data_dir

        with argv_context_guard(reft_config):
            from run_finetune import main

            main()

        perdict_params = {
            "model_name_or_path": reft_config["model_name_or_path"],
            "reft_path": self.output_dir,
            "data_file": os.path.join(self.data_dir, "dev.json"),
            "batch_size": 8,
        }
        self.run_reft_predictor(perdict_params)


if __name__ == "__main__":
    unittest.main()
