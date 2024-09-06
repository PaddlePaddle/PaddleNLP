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

from paddlenlp.utils.downloader import get_path_from_url_with_filelock


class InferenceTest(unittest.TestCase):
    config_path: str = "./test_tipc/llm/fixtures/predictor.yaml"
    predictor_shell_name = "inference/run_predictor.sh"

    ce_testing_base_url = "https://paddlenlp.bj.bcebos.com/tests/ce"
    predict_file_name = "predict.json"

    def setUp(self) -> None:
        paddle.set_default_dtype("float32")
        self.output_path = tempfile.mkdtemp()
        sys.path.insert(0, "../llm")
        self.model_name = os.getenv("MODEL_NAME")
        self.run_predictor_shell_path = os.path.join(os.path.dirname(__file__), self.predictor_shell_name)

        self.log_file = open(os.path.join(self.output_path, "log.log"), "w")

    def tearDown(self) -> None:
        sys.path.remove("../llm")
        self.log_file.close()

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

    def compare_result(self, result_1, result_2):
        """
        compare two result from predictor
        """
        result_1_result = self._read_result(os.path.join(self.output_path, result_1))
        result_2_result = self._read_result(os.path.join(self.output_path, result_2))

        assert len(result_1_result) == len(result_2_result)
        count, full_match = 0, 0
        for item_1, item_2 in zip(result_1_result, result_2_result):
            min_length = min(len(item_1), len(item_2))
            count += int(item_1[: min_length // 2] == item_2[: min_length // 2])
            full_match += int(item_1[:min_length] == item_2[:min_length])

        return full_match / len(result_1_result), count / len(result_1_result)

    def test_predictor(self):
        config = self._load_config(self.model_name)

        # 0. download the ground-truth file for comparing
        get_path_from_url_with_filelock(
            os.path.join(self.ce_testing_base_url, config["model_name"], self.predict_file_name),
            root_dir=self.output_path,
        )

        config["output_path"] = self.output_path
        command_prefix = " ".join([f"{key}={value}" for key, value in config.items()])

        # 1.run dynamic model
        subprocess.run(
            command_prefix + " bash " + self.run_predictor_shell_path, stdout=sys.stdout, stderr=sys.stderr, shell=True
        )

        full_match_acc, _ = self.compare_result("dynamic.json", "static.json")
        self.assertGreater(full_match_acc, 0.8)

        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.75)

        # 2.run fused-mt model
        subprocess.run(
            command_prefix + " inference_model=true bash " + self.run_predictor_shell_path,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
        )

        # 在不同环境下的 A100 下测试 full_match_acc 有可能不是为 1.0；可是这边设置了 `precision` 数值，CE 会针对于此数据做监控，一旦有
        # 异常会发送异常报告，也可以达到监控的效果。
        full_match_acc, half_match_acc = self.compare_result("dynamic.json", "static.json")
        print("precision:", full_match_acc)
        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.75)
        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.75)

        # 3. run sample decoding & benchmark on fused-mt model
        subprocess.run(
            command_prefix
            + " top_p=0.7 decode_strategy=sampling benchmark=1 inference_model=true bash "
            + self.run_predictor_shell_path,
            stdout=self.log_file,
            stderr=self.log_file,
            shell=True,
        )

        # sampling: the full-matach acc must be less than 0.1
        full_match_acc, half_match_acc = self.compare_result("dynamic.json", "static.json")
        self.assertLessEqual(full_match_acc, 0.55)
        self.assertLessEqual(half_match_acc, 0.85)

        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertLessEqual(full_match_acc, 0.55)
        self.assertLessEqual(half_match_acc, 0.85)

        # read ips value from log file
        ips = self._read_ips_from_log_file()
        self.assertGreaterEqual(ips, 80)

    def _read_ips_from_log_file(self):
        with open(os.path.join(self.output_path, "log.log"), "r") as f:
            content = f.read()

        print(content)
        keyword = "IPS:"
        ips_index = content.index(keyword)
        if ips_index == -1:
            return None

        content = content[ips_index + len(keyword) :]
        token_unit_index = content.index("tokens/s")
        ips = content[:token_unit_index]
        return float(ips)


class PTuningInfereneTest(InferenceTest):
    predictor_shell_name = "inference/run_predictor_precaches.sh"
    config_path = "./test_tipc/llm/fixtures/predictor-ptuning.yaml"

    predict_file_name = "predict-ptuning.json"

    def setUp(self) -> None:
        super().setUp()

    def _load_config(self, key):
        config = super()._load_config(key)

        for file in ["pre_caches.npy", "prefix_config.json", "prefix_model_state.pdparams"]:
            get_path_from_url_with_filelock(
                os.path.join(self.ce_testing_base_url, config["model_name"], file), root_dir=self.output_path
            )

        config["prefix_path"] = self.output_path
        config["export_precache"] = 1
        return config

    def test_predictor(self):
        if self.model_name == "chatglm2":
            return
        config = self._load_config(self.model_name)

        # 0. download the ground-truth file for comparing
        get_path_from_url_with_filelock(
            os.path.join(self.ce_testing_base_url, config["model_name"], self.predict_file_name),
            root_dir=self.output_path,
        )

        config["output_path"] = self.output_path
        command_prefix = " ".join([f"{key}={value}" for key, value in config.items()])

        # 1.run dynamic model
        subprocess.run(
            command_prefix + " bash " + self.run_predictor_shell_path, stdout=sys.stdout, stderr=sys.stderr, shell=True
        )

        full_match_acc, _ = self.compare_result("dynamic.json", "static.json")
        self.assertGreater(full_match_acc, 0.8)

        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.8)

        # 2.run fused-mt model
        subprocess.run(
            command_prefix + " inference_model=true bash " + self.run_predictor_shell_path,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
        )

        full_match_acc, half_match_acc = self.compare_result("dynamic.json", "static.json")
        print("precision:", full_match_acc)

        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.8)
        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertGreater(full_match_acc, 0.6)
        self.assertGreater(half_match_acc, 0.8)

        # 3. run sample decoding & benchmark on fused-mt model
        subprocess.run(
            command_prefix
            + " top_p=0.7 decode_strategy=sampling benchmark=1 inference_model=true bash "
            + self.run_predictor_shell_path,
            stdout=self.log_file,
            stderr=self.log_file,
            shell=True,
        )

        # sampling: the full-matach acc must be less than 0.1
        full_match_acc, half_match_acc = self.compare_result("dynamic.json", "static.json")
        self.assertLessEqual(full_match_acc, 0.55)
        self.assertLessEqual(half_match_acc, 0.85)

        full_match_acc, half_match_acc = self.compare_result(self.predict_file_name, "static.json")
        self.assertLessEqual(full_match_acc, 0.55)
        self.assertLessEqual(half_match_acc, 0.85)

        # read ips value from log file
        ips = self._read_ips_from_log_file()
        self.assertGreaterEqual(ips, 50)
