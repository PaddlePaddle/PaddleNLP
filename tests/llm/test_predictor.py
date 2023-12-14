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
import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import (  # ChatGLMForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    ChatGLMForCausalLM,
    ChatGLMv2ForCausalLM,
    LlamaForCausalLM,
)
from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    url_file_exists,
)

from .testing_utils import LLMTest


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/tiny-random-llama", LlamaForCausalLM],
        ["__internal_testing__/tiny-fused-bloom", BloomForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm", ChatGLMForCausalLM],
        ["__internal_testing__/tiny-fused-chatglm2", ChatGLMv2ForCausalLM],
    ],
)
class PredictorTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("float32")
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_predictor(self):
        self.run_predictor({"inference_model": True})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)

        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.25)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)

    def test_wint8(self):
        self.run_predictor({"inference_model": True, "quant_type": "weight_only_int8"})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        assert len(result_0) == len(result_1)
        count, full_match = 0, 0

        for inference_item, no_inference_item in zip(result_0, result_1):
            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.1)

        if self.model_name_or_path == "__internal_testing__/tiny-fused-chatglm":
            self.assertGreaterEqual(count / len(result_0), 0.3)
        else:
            self.assertGreaterEqual(count / len(result_0), 0.4)


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [["__internal_testing__/tiny-random-llama", LlamaForCausalLM]],
)
class PredictorPrecacheTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()

        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)
        self.download_precache_files()

    def download_precache_files(self):
        files = [
            "prefix_config.json",
            "config.json",
            "model_state.pdparams",
            "pre_caches.npy",
            "prefix_model_state.pdparams",
        ]
        for file in files:
            file_url = os.path.join(COMMUNITY_MODEL_PREFIX, self.model_name_or_path, file)
            if not url_file_exists(file_url):
                continue
            get_path_from_url_with_filelock(file_url, root_dir=self.output_dir)

    def test_predictor(self):
        self.run_predictor({"inference_model": True, "export_precache": True, "prefix_path": self.output_dir})
        result_0 = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.run_predictor({"inference_model": False, "export_precache": True, "prefix_path": self.output_dir})
        result_1 = self._read_result(os.path.join(self.output_dir, "predict.json"))

        # compare the generation result of inference & dygraph model
        assert len(result_0) == len(result_1)
        count, full_match = 0, 0
        for inference_item, no_inference_item in zip(result_0, result_1):

            min_length = min(len(inference_item), len(no_inference_item))
            count += int(inference_item[: min_length // 2] == no_inference_item[: min_length // 2])
            full_match += int(inference_item[:min_length] == no_inference_item[:min_length])

        self.assertGreaterEqual(full_match / len(result_0), 0.6)
        self.assertGreaterEqual(count / len(result_0), 0.8)
