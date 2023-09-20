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
class PretrainTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/pretrain.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.dataset_dir = tempfile.mkdtemp()
        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

        sys.path.remove(self.model_codes_dir)
        shutil.rmtree(self.dataset_dir)

    def test_pretrain(self):
        # Run pretrain
        URL = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy"
        URL2 = "https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz"
        get_path_from_url(URL, root_dir=self.dataset_dir)
        get_path_from_url(URL2, root_dir=self.dataset_dir)

        pretrain_config = load_test_config(self.config_path, "pretrain", self.model_dir)

        pretrain_config["input_dir"] = self.dataset_dir
        pretrain_config["output_dir"] = self.output_dir

        with argv_context_guard(pretrain_config):
            from run_pretrain import main

            main()

        self.run_predictor({"inference_model": True})
        self.run_predictor({"inference_model": False})
