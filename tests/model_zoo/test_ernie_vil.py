# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import TestCase, skip

from paddlenlp.utils import install_package
from paddlenlp.utils.downloader import get_path_from_url
from tests.testing_utils import argv_context_guard, load_test_config


class ErnieViLTest(TestCase):
    def setUp(self) -> None:
        self.path = "./model_zoo/ernie-vil2.0"
        self.config_path = "./tests/fixtures/model_zoo/ernie_vil.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    @skip("Skip and wait to fix.")
    def test_finetune(self):
        install_package("lmdb", "1.3.0")
        if not os.path.exists("./tests/fixtures/Flickr30k-CN"):
            URL = "https://paddlenlp.bj.bcebos.com/tests/Flickr30k-CN-small.zip"
            get_path_from_url(URL, root_dir="./tests/fixtures")
        # 0. create dataseit
        # todo: @w5688414  fix it

        # 1. run finetune
        finetune_config = load_test_config(self.config_path, "finetune")
        with argv_context_guard(finetune_config):
            from run_finetune import do_train

            do_train()

        # 2. export model
        export_config = {
            "model_path": finetune_config["output_dir"],
            "output_path": finetune_config["output_dir"],
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # 3. infer model
        infer_config = {
            "image_path": "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "model_dir": export_config["output_path"],
            "device": finetune_config["device"],
        }
        with argv_context_guard(infer_config):
            from deploy.python.infer import main

            main()
