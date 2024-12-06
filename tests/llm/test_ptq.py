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
from unittest import skip

from parameterized import parameterized_class

from tests.llm.testing_utils import LLMTest
from tests.testing_utils import argv_context_guard, load_test_config


@parameterized_class(
    ["model_dir"],
    [["llama"]],
)
class FinetuneTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/ptq.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        sys.path.insert(0, self.model_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_ptq(self):
        finetune_config = load_test_config(self.config_path, "ptq", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        with argv_context_guard(finetune_config):
            from run_quantization import main

            main()

        self.run_predictor({"inference_model": True})

    def test_blha(self):
        finetune_config = load_test_config(self.config_path, "ptq", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        with argv_context_guard(finetune_config):
            from run_quantization import main

            main()

        self.run_predictor({"inference_model": True, "block_attn": True})

    @skip("Skip and wait to fix.")
    def test_ptq_smooth(self):
        finetune_config = load_test_config(self.config_path, "ptq", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir
        finetune_config["smooth"] = True

        with argv_context_guard(finetune_config):
            from run_quantization import main

            main()

        self.run_predictor({"inference_model": True})
        self._read_result(os.path.join(self.output_dir, "predict.json"))

    @skip("Skip and wait to fix.")
    def test_ptq_shift(self):
        finetune_config = load_test_config(self.config_path, "ptq", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir
        finetune_config["shift"] = True

        with argv_context_guard(finetune_config):
            from run_quantization import main

            main()

        self.run_predictor({"inference_model": True})

    # TODO(@wufeisheng): add test_ptq_shift_smooth_all_linear
