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
from unittest import TestCase

from tests.testing_utils import argv_context_guard, load_test_config


class GPTTest(TestCase):
    def setUp(self) -> None:
        self.path = "./model_zoo/gpt"
        self.config_path = "./tests/fixtures/model_zoo/gpt.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    # TODO(wj-Mcat): disable old gpt `run_pretrain.py` and will be uncomment in: https://github.com/PaddlePaddle/PaddleNLP/pull/4614
    @unittest.skip("disable for old gpt")
    def test_pretrain(self):

        # 1. run pretrain
        pretrain_config = load_test_config(self.config_path, "pretrain")
        with argv_context_guard(pretrain_config):
            from run_pretrain import do_train

            do_train()

        # 2. export model
        export_config = {
            "model_type": pretrain_config["model_type"],
            "model_path": pretrain_config["output_dir"],
            "output_path": os.path.join(pretrain_config["output_dir"], "export_model"),
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # 3. infer model
        infer_config = {
            "model_type": export_config["model_type"],
            "model_path": export_config["output_path"],
            "select_device": pretrain_config["device"],
        }
        with argv_context_guard(infer_config):
            from deploy.python.inference import main

            main()

    def test_msra_ner(self):
        config = load_test_config(self.config_path, "msra_ner")
        with argv_context_guard(config):
            from run_msra_ner import do_train

            do_train()

    def test_run_glue(self):
        config = load_test_config(self.config_path, "glue")
        with argv_context_guard(config):
            from run_glue import do_train

            do_train()

    def test_generation(self):
        config = load_test_config(self.config_path, "generation")
        with argv_context_guard(config):
            from run_generation import run

            run()
