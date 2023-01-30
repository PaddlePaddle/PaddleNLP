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
import os
from unittest import TestCase

from tests.testing_utils import load_argv, construct_argv


class GPTTest(TestCase):
    def setUp(self) -> None:
        self.path = "./model_zoo/gpt"
        self.config_path = "./tests/fixtures/model_zoo/gpt.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_pretrain(self):

        # 1. run pretrain
        argv, config = load_argv(self.config_path, "pretrain", return_dict=True)
        device = config["device"]
        sys.argv = argv
        from run_pretrain import do_train

        do_train()

        # 2. export model
        from export_model import main

        config = {
            "model_type": config["model_type"],
            "model_path": config["output_dir"],
            "output_path": os.path.join(config["output_dir"], "export_model"),
        }
        argv = construct_argv(config)
        sys.argv = argv
        main()

        # 3. infer model
        from deploy.python.inference import main

        config = {"model_type": config["model_type"], "model_path": config["output_path"], "select_device": device}
        argv = construct_argv(config)
        sys.argv = argv
        main()

    def test_msra_ner(self):
        argv = load_argv(self.config_path, "msra_ner")
        sys.argv = argv

        from run_msra_ner import do_train

        do_train()

    def test_run_glue(self):
        argv = load_argv(self.config_path, "glue")
        sys.argv = argv
        from run_glue import do_train

        do_train()

    def test_generation(self):
        argv = load_argv(self.config_path, "generation")
        sys.argv = argv

        from run_generation import run

        run()
