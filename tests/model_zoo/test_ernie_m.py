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
from unittest import TestCase

from parameterized import parameterized_class

from tests.testing_utils import argv_context_guard, load_test_config  # is_slow_test


@parameterized_class(
    ["task_type"],
    [["cross-lingual-transfer"], ["translate-train-all"]],
)
class ErnieMTest(TestCase):
    task_type: str = "cross-lingual-transfer"

    def setUp(self) -> None:
        self.path = "./model_zoo/ernie-m"
        self.config_path = "./tests/fixtures/model_zoo/ernie-m.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_classifier(self):
        finetune_config = load_test_config(self.config_path, "classifier")

        finetune_config["task_type"] = self.task_type

        # 1. finetune and export model
        with argv_context_guard(finetune_config):
            from run_classifier import do_train

            do_train()

        # delete for FD https://github.com/PaddlePaddle/PaddleNLP/pull/4891

        # # 2. infer model
        # infer_config = {
        #     "model_name_or_path": finetune_config["model_name_or_path"],
        #     "model_path": os.path.join(finetune_config["export_model_dir"], "export", "model"),
        #     "device": finetune_config["device"],
        # }
        # with argv_context_guard(infer_config):
        #     from deploy.predictor.inference import main

        #     main()

        # # if using gpu, test infering with precision_mode 'fp16'
        # if is_slow_test():
        #     infer_config.update({"infer_config": "fp16"})
        #     with argv_context_guard(infer_config):
        #         from deploy.predictor.inference import main

        #         main()
