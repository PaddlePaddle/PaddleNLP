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
import tempfile
from unittest import TestCase

import paddle

from tests.testing_utils import argv_context_guard, load_test_config, slow


class PredictorTest(TestCase):
    def setUp(self) -> None:
        self.path = "./llm"
        self.config_path = "./tests/fixtures/llm/predictor.yaml"
        sys.path.insert(0, self.path)
        paddle.disable_static()
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        sys.path.remove(self.path)
        paddle.disable_static()
        self.tempdir.cleanup()

    @slow
    def test_dynamic_predictor(self):
        # to avoid the same parameter
        paddle.utils.unique_name.switch()
        predict_config = load_test_config(self.config_path, "inference-predict")
        predict_config["output_file"] = os.path.join(self.tempdir.name, "predict.json")
        with argv_context_guard(predict_config):
            from predictor import predict

            predict()

        # to static
        paddle.utils.unique_name.switch()
        config = load_test_config(self.config_path, "inference-to-static")
        config["output_path"] = self.tempdir.name
        with argv_context_guard(config):
            from export_model import main

            main()

        # inference
        config = load_test_config(self.config_path, "inference-infer")
        config["model_name_or_path"] = self.tempdir.name
        config["output_file"] = os.path.join(self.tempdir.name, "infer.json")
        with argv_context_guard(config):
            from predictor import predict

            predict()
