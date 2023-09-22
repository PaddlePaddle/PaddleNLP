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
import subprocess
import sys
import tempfile
import unittest

import yaml
from parameterized import parameterized_class


@parameterized_class(
    ["config_key"],
    [
        ["llama"],
    ],
)
class InfereneTest(unittest.TestCase):
    config_path: str = "./tests/test_tipc/llm/fixtures/predictor.yaml"
    config_key: str = None

    def setUp(self) -> None:
        self.output_path = tempfile.mkdtemp()
        sys.path.insert(0, "./llm")

    def tearDown(self) -> None:
        sys.path.remove("./llm")

    def _load_config(self, key):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config[key]

    def test_predictor(self):
        config = self._load_config(self.config_key)
        config["output_path"] = self.output_path
        command_prefix = " ".join([f"{key}={value}" for key, value in config.items()])

        # 1.run fused-mt model
        subprocess.run(
            command_prefix + " bash tests/test_tipc/llm/inference/run_predictor.sh",
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        assert os.path.exists(os.path.join(self.output_path, "dynamic.json"))
        assert os.path.exists(os.path.join(self.output_path, "static.json"))
