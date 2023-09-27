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

import json
import os
import subprocess
import sys
import tempfile
import unittest

import paddle
import yaml


class InfereneTest(unittest.TestCase):
    config_path: str = "./test_tipc/llm/fixtures/predictor.yaml"

    def setUp(self) -> None:
        paddle.set_default_dtype("float32")
        self.output_path = tempfile.mkdtemp()
        sys.path.insert(0, "../llm")
        self.model_name = os.getenv("MODEL_NAME")
        self.run_predictor_shell_path = os.path.join(os.path.dirname(__file__), "inference/run_predictor.sh")

    def tearDown(self) -> None:
        sys.path.remove("../llm")

    def _load_config(self, key):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config[key]

    def _read_result(self, file):
        result = []
        # read output field from json file
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                result.append(data["output"])
        return result

    def test_predictor(self):
        config = self._load_config(self.model_name)
        config["output_path"] = self.output_path
        command_prefix = " ".join([f"{key}={value}" for key, value in config.items()])

        # 1.run dynamic model
        subprocess.run(
            command_prefix + " bash " + self.run_predictor_shell_path, stdout=sys.stdout, stderr=sys.stderr, shell=True
        )

        dynamic = self._read_result(os.path.join(self.output_path, "dynamic.json"))
        static = self._read_result(os.path.join(self.output_path, "static.json"))
        self.assertListEqual(dynamic, static)

        # 2.run fused-mt model
        subprocess.run(
            command_prefix + " inference_model true bash tests/test_tipc/llm/inference/run_predictor.sh",
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
        )

        fused_dynamic = self._read_result(os.path.join(self.output_path, "dynamic.json"))
        fused_static = self._read_result(os.path.join(self.output_path, "static.json"))
        self.assertListEqual(fused_dynamic, fused_static)

        # 3. compare the generation text of dynamic & inference model
        assert len(fused_static) == len(static)
        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(fused_static, static):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[min_length // 2] == no_inference_item[min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        print("full_match", full_match)
        print(full_match / len(static))
        print("precision:", count)
        print(count / len(static))

        self.assertGreater(full_match / len(static), 0.6)
        self.assertGreater(count / len(static), 0.8)
