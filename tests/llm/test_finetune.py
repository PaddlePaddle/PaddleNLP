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

import json
import os
import sys
import unittest

from parameterized import parameterized_class

from paddlenlp.utils.env import SAFE_OPTIMIZER_INDEX_NAME
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
        ["chatglm"],
        # ["bloom"], @skip("Skip and wait to fix.")
        ["chatglm2"],
        ["qwen"],
        ["qwen2"],
        ["baichuan"],
    ],
)
class FinetuneTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/finetune.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        sys.path.insert(0, self.model_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_finetune(self):
        finetune_config = load_test_config(self.config_path, "finetune", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        with argv_context_guard(finetune_config):
            from run_finetune import main

            main()

        # TODO(wj-Mcat): disable chatglm2 test temporarily
        if self.model_dir not in ["qwen", "qwen2", "baichuan", "chatglm2"]:
            self.run_predictor({"inference_model": True})

        self.run_predictor({"inference_model": False})


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
    ],
)
class CkptQuantTest(LLMTest, TestMultipleGpus):
    config_path: str = "./tests/fixtures/llm/finetune.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        sys.path.insert(0, self.model_dir)
        self.run_sft = "llm/run_finetune.py"

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_ckpt_quant(self):
        finetune_config = load_test_config(self.config_path, "ckpt_quant", self.model_dir)

        finetune_config["dataset_name_or_path"] = self.data_dir
        finetune_config["output_dir"] = self.output_dir

        self.runfirst(finetune_config)

        # get `quant_ckpt_resume_times`
        with open(os.path.join(self.output_dir, "checkpoint-1", SAFE_OPTIMIZER_INDEX_NAME), "r") as r:
            index = json.loads(r.read())
        quant_ckpt_resume_times = index["quant_ckpt_resume_times"]

        self.rerun(finetune_config)

        self.assertEqual(quant_ckpt_resume_times, 0)

    def runfirst(self, train_args):
        self.run_n1c2(self.run_sft, **train_args)

    def rerun(self, train_args):
        self.run_n1c2(self.run_sft, **train_args)
