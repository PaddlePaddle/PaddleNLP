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
import shutil
import sys
import tempfile
import unittest

from parameterized import parameterized_class

from paddlenlp.utils.downloader import get_path_from_url
from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
    ],
)
class FinetuneTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/finetune.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.data_dir = tempfile.mkdtemp()
        sys.path.insert(0, self.model_dir)

        # Run pretrain
        URL = "https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz"
        get_path_from_url(URL, root_dir=self.data_dir)
        self.data_dir = os.path.join(self.data_dir, "data")
        self.use_small_datasets()

    def use_small_datasets(self):
        # use 20 examples
        def use_few_examples(file):
            with open(os.path.join(self.data_dir, file), "r", encoding="utf8") as f:
                lines = [line.strip() for line in f.readlines()]
            with open(os.path.join(self.data_dir, file), "w+", encoding="utf8") as f:
                f.write("\n".join(lines[:20]))

        shutil.copyfile(
            os.path.join(self.data_dir, "dev.json"),
            os.path.join(self.data_dir, "validation.json"),
        )
        use_few_examples("train.json")
        use_few_examples("dev.json")
        use_few_examples("validation.json")

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        shutil.rmtree(self.data_dir)

    def test_pretrain(self):
        finetune_config = load_test_config(self.config_path, "finetune", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        with argv_context_guard(finetune_config):
            from finetune_generation import main

            main()

        self._test_inference_predictor()
        self._test_predictor()

    def _test_inference_predictor(self):
        self.run_predictor({"inference_model": "true"})

    def _test_predictor(self):
        self.run_predictor({"inference_model": "false"})
