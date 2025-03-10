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
import shutil
import sys
import tempfile

import paddle

from tests.testing_utils import argv_context_guard, load_test_config


class LLMTest:
    config_path: str = None
    data_dir = "./tests/fixtures/llm/data/"

    def setUp(self) -> None:
        self.root_path = "./llm"
        self.output_dir = tempfile.mkdtemp()
        self.inference_output_dir = tempfile.mkdtemp()
        sys.path.insert(0, self.root_path)
        self.disable_static()
        paddle.set_default_dtype("float32")

    def tearDown(self) -> None:
        sys.path.remove(self.root_path)
        shutil.rmtree(self.output_dir)
        shutil.rmtree(self.inference_output_dir)
        self.disable_static()
        paddle.device.cuda.empty_cache()

    def disable_static(self):
        paddle.utils.unique_name.switch()
        paddle.disable_static()

    def _read_result(self, file):
        result = []
        # read output field from json file
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                result.append(data["output"])
        return result

    def run_predictor(self, config_params=None):
        if config_params is None:
            config_params = {}

        # to avoid the same parameter
        self.disable_static()
        predict_config = load_test_config(self.config_path, "inference-predict")
        predict_config["output_file"] = os.path.join(self.output_dir, "predict.json")
        predict_config["model_name_or_path"] = self.output_dir
        predict_config.update(config_params)

        with argv_context_guard(predict_config):
            from predict.predictor import predict

            predict()

        # prefix_tuning dynamic graph do not support to_static
        if not predict_config["inference_model"]:
            return

        # to static
        self.disable_static()
        config = load_test_config(self.config_path, "inference-to-static")
        config["output_path"] = self.inference_output_dir
        config["model_name_or_path"] = self.output_dir
        config.update(config_params)
        with argv_context_guard(config):
            from predict.export_model import main

            main()

        # inference
        self.disable_static()
        config = load_test_config(self.config_path, "inference-infer")
        config["model_name_or_path"] = self.inference_output_dir
        config["output_file"] = os.path.join(self.inference_output_dir, "infer.json")

        config_params.pop("model_name_or_path", None)
        config.update(config_params)
        with argv_context_guard(config):
            from predict.predictor import predict

            predict()

        self.disable_static()

        predict_result = self._read_result(predict_config["output_file"])
        infer_result = self._read_result(config["output_file"])
        assert len(predict_result) == len(infer_result)

        for predict_item, infer_item in zip(predict_result, infer_result):
            self.assertEqual(predict_item, infer_item)

    def run_reft_predictor(self, predict_config=None):
        predict_config["output_file"] = os.path.join(self.output_dir, "predict.json")
        with argv_context_guard(predict_config):
            from predict.reft_predictor import main

            main()
