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

from parameterized import parameterized_class

from tests.testing_utils import (
    argv_context_guard,
    load_test_config,
    skip_for_none_ce_case,
)

from .testing_utils import LLMTest


# TODO(wj-Mcat): disable chatglm2 test temporarily
@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
        ["bloom"],
        ["chatglm"],
        # ["chatglm2"],
        ["qwen"],
        ["baichuan"],
    ],
)
class PrefixTuningTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/prefix_tuning.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        sys.path.remove(self.model_codes_dir)

    @skip_for_none_ce_case
    def test_prefix_tuning(self):
        prefix_tuning_config = load_test_config(self.config_path, "prefix_tuning", self.model_dir)

        prefix_tuning_config["dataset_name_or_path"] = self.data_dir
        prefix_tuning_config["output_dir"] = self.output_dir
        with argv_context_guard(prefix_tuning_config):
            from run_finetune import main

            main()

        if self.model_dir not in ["qwen", "baichuan"]:
            self.run_predictor(
                {
                    "inference_model": True,
                    "prefix_path": self.output_dir,
                    "model_name_or_path": prefix_tuning_config["model_name_or_path"],
                }
            )

        self.run_predictor(
            {
                "inference_model": False,
                "prefix_path": self.output_dir,
                "model_name_or_path": prefix_tuning_config["model_name_or_path"],
            }
        )
