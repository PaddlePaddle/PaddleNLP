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
import unittest

import paddle
from parameterized import parameterized_class

from tests.testing_utils import argv_context_guard, load_test_config

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
        # ["chatglm"], @skip("Skip and wait to fix.")
        # ["chatglm2"], @skip("Skip and wait to fix.")
        # ["bloom"], @skip("Skip and wait to fix.")
        ["qwen"],
        ["baichuan"],
    ],
)
class MoraTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/mora.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)

        self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
        sys.path.insert(0, self.model_codes_dir)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)
        sys.path.remove(self.model_codes_dir)

    def test_mora(self):
        self.disable_static()
        paddle.set_default_dtype("float32")
        mora_config = load_test_config(self.config_path, "mora", self.model_dir)
        mora_config["output_dir"] = self.output_dir
        mora_config["dataset_name_or_path"] = self.data_dir
        # use_quick_lora
        mora_config["use_quick_lora"] = False

        with argv_context_guard(mora_config):
            from run_finetune import main

            main()

        # merge weights
        merge_lora_weights_config = {
            "lora_path": mora_config["output_dir"],
            "model_name_or_path": mora_config["model_name_or_path"],
            "output_path": mora_config["output_dir"],
        }
        with argv_context_guard(merge_lora_weights_config):
            from tools.merge_lora_params import merge

            merge()

        # TODO(wj-Mcat): disable chatglm2 test temporarily
        if self.model_dir not in ["qwen", "baichuan", "chatglm2"]:
            self.run_predictor({"inference_model": True})

        self.run_predictor({"inference_model": False})


# @parameterized_class(
#     ["model_dir"],
#     [
#         ["llama"],
#         ["qwen"],
#     ],
# )
# class LoraChatTemplateTest(LLMTest, unittest.TestCase):
#     config_path: str = "./tests/fixtures/llm/lora.yaml"
#     model_dir: str = None

#     def setUp(self) -> None:
#         LLMTest.setUp(self)

#         self.model_codes_dir = os.path.join(self.root_path, self.model_dir)
#         sys.path.insert(0, self.model_codes_dir)

#         self.rounds_data_dir = tempfile.mkdtemp()
#         shutil.copyfile(
#             os.path.join(self.data_dir, "train.json"),
#             os.path.join(self.rounds_data_dir, "train.json"),
#         )
#         shutil.copyfile(
#             os.path.join(self.data_dir, "dev.json"),
#             os.path.join(self.rounds_data_dir, "dev.json"),
#         )
#         self.create_multi_turns_data(os.path.join(self.rounds_data_dir, "train.json"))
#         self.create_multi_turns_data(os.path.join(self.rounds_data_dir, "dev.json"))

#     def create_multi_turns_data(self, file: str):
#         result = []
#         with open(file, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 data["src"] = [data["src"]] * 3
#                 data["tgt"] = [data["tgt"]] * 3
#                 result.append(data)

#         with open(file, "w", encoding="utf-8") as f:
#             for data in result:
#                 line = json.dumps(line)
#                 f.write(line + "\n")

#     def tearDown(self) -> None:
#         LLMTest.tearDown(self)
#         sys.path.remove(self.model_codes_dir)

#     def test_lora(self):
#         self.disable_static()
#         paddle.set_default_dtype("float32")

#         lora_config = load_test_config(self.config_path, "lora", self.model_dir)

#         lora_config["dataset_name_or_path"] = self.rounds_data_dir
#         lora_config["chat_template"] = "./tests/fixtures/chat_template.json"
#         lora_config["output_dir"] = self.output_dir

#         with argv_context_guard(lora_config):
#             from run_finetune import main

#             main()

#         # merge weights
#         merge_lora_weights_config = {
#             "model_name_or_path": lora_config["model_name_or_path"],
#             "lora_path": lora_config["output_dir"],
#             "merge_model_path": lora_config["output_dir"],
#         }
#         with argv_context_guard(merge_lora_weights_config):
#             from tools.merge_lora_params import merge

#             merge()

#         if self.model_dir not in ["chatglm2", "qwen", "baichuan"]:
#             self.run_predictor({"inference_model": True})

#         self.run_predictor({"inference_model": False})
