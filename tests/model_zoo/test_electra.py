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

CUDA_VISIBLE_DEVICES = 1


class ELECTRA_Test(TestCase):
    def download_corpus(self, input_dir):
        os.makedirs(input_dir, exist_ok=True)
        files = [
            "https://paddle-qa.bj.bcebos.com/paddlenlp/BookCorpus.tar.gz",
        ]

        for file in files:
            file_name = file.split("/")[-1]
            file_path = os.path.join(input_dir, file_name)
            if not os.path.exists(file_path):
                logger.info(f"start to download corpus: <{file_name}> into <{input_dir}>")
                get_path_from_url_with_filelock(file, root_dir=input_dir)

    def setUp(self) -> None:
        self.path = "./model_zoo/electra"
        self.config_path = "./tests/fixtures/model_zoo/electra.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_pretrain(self):

        # 1. run pretrain
        pretrain_config = load_test_config(self.config_path, "pretrain")
        self.download_corpus(pretrain_config["input_dir"])
        with argv_context_guard(pretrain_config):
            from run_pretrain import do_train

            do_train()

        # 2. get_ft_model
        ft_config = {
            "model_dir": pretrain_config["output_dir"],
        }
        with argv_context_guard(ft_config):
            from get_ft_model import main

            main()

        # 3. run glue
        glue_config = load_test_config(self.config_path, "glue")
        glue_config["output_dir"] = "pretrained_model/model"
        glue_config["model_name_or_path"] = pretrain_config["output_dir"]
        with argv_context_guard(glue_config):
            from run_glue import do_train

            do_train()

        # 4. export model
        export_config = {
            "model_name": pretrain_config["model_name_or_path"],
            "output_model_dir": "infer_model/model",
            "input_model_dir": glue_config["output_dir"],
        }
        with argv_context_guard(export_config):
            from export_model import main

            main()

        # infer model of samples
        infer_config = {
            "model_file": "infer_model/model/__internal_testing__/chinese-electra-small.pdmodel",
            "params_file": "infer_model/model/__internal_testing__/chinese-electra-small.pdiparams",
            "predict_sentences": "uneasy mishmash of styles and genres ."
            "director rob marshall went out gunning to make a great one .",
            "batch_size": 2,
            "max_seq_length": 128,
            "model_name": pretrain_config["model_name_or_path"],
        }
        with argv_context_guard(infer_config):
            from deploy.python.predict import predict

            predict()

    def test_glue(self):

        glue_config = load_test_config(self.config_path, "glue")
        with argv_context_guard(glue_config):
            from run_glue import do_train

            do_train()
