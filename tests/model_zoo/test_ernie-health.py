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
from unittest import TestCase

from paddlenlp.utils.downloader import get_path_from_url_with_filelock
from paddlenlp.utils.log import logger
from tests.testing_utils import argv_context_guard, load_test_config


class ERNIEHEALTH_Test(TestCase):
    def download_corpus(self, input_dir):
        os.makedirs(input_dir, exist_ok=True)
        files = [
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-health/data/samples_ids.npy",
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-health/data/samples_idx.npz",
        ]

        for file in files:
            file_name = file.split("/")[-1]
            file_path = os.path.join(input_dir, file_name)
            if not os.path.exists(file_path):
                logger.info(f"start to download corpus: <{file_name}> into <{input_dir}>")
                get_path_from_url_with_filelock(file, root_dir=input_dir)

    def setUp(self) -> None:
        self.path = "./model_zoo/ernie-health"
        self.config_path = "./tests/fixtures/model_zoo/ernie-health.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_pretrain(self):
        pretrain_config = load_test_config(self.config_path, "pretrain")

        self.download_corpus(pretrain_config["input_dir"])
        with argv_context_guard(pretrain_config):

            from run_pretrain_trainer import main

            main()
