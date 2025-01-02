# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os
import shutil
import tempfile
import unittest

from parameterized import parameterized, parameterized_class

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from tests.parallel_launch import TestMultipleGpus
from tests.testing_utils import require_gpu

from .testing_utils import LLMTest


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/Qwen2.5-7B-Instruct-tiny-nhl2", AutoModelForCausalLM],
        ["__internal_testing__/Qwen1.5-MoE-A2.7B-Chat-tiny-nhl2", AutoModelForCausalLM],
        ["__internal_testing__/Llama-2-7b-chat-tiny-nhl2", AutoModelForCausalLM],
        ["__internal_testing__/Meta-Llama-3.1-8B-Instruct-tiny-nhl2", AutoModelForCausalLM],
    ],
)
class CommonModelInferenceTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def test_common_model_inference(self):
        self.run_predictor({"inference_model": True, "append_attn": True, "max_length": 48})
        result = self._read_result(os.path.join(self.output_dir, "predict.json"))
        self.assertTrue(len(result) > 0, f"The inference result for {self.model_name_or_path} is empty!")


def levenshtein_similarity(a, b):
    def levenshtein_distance_optimized(a, b):
        m, n = len(a), len(b)

        previous = list(range(n + 1))
        current = [0] * (n + 1)

        for i in range(1, m + 1):
            current[0] = i
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    current[j] = previous[j - 1]
                else:
                    current[j] = 1 + min(previous[j], current[j - 1], previous[j - 1])
            previous, current = current, previous

        return previous[n]

    distance = levenshtein_distance_optimized(a, b)
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


global_result = {}


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["Qwen/Qwen2.5-1.5B-Instruct", AutoModelForCausalLM],
        ["meta-llama/Llama-3.2-3B-Instruct", AutoModelForCausalLM],
    ],
)
class CommonParamInferenceTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self) -> None:
        super().setUp()
        self.model_class.from_pretrained(self.model_name_or_path, dtype="float16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)
        global global_result
        model_tag = os.path.basename(self.model_name_or_path)

        if model_tag not in global_result:
            self.run_predictor({"inference_model": True, "block_attn": True, "max_length": 48})
            self.golden_result = self._read_result(os.path.join(self.output_dir, "predict.json"))
            global_result[model_tag] = self.golden_result
        else:
            self.golden_result = global_result[model_tag]

    @parameterized.expand(
        [
            (
                {
                    "use_fake_parameter": True,
                    "quant_type": "a8w8c8",
                },
            ),
            (
                {
                    "inference_model": False,
                    "block_attn": False,
                },
            ),
            (
                {
                    "append_attn": True,
                },
            ),
        ]
    )
    def test_common_param_inference(self, param_case):

        config_params = {"inference_model": True, "block_attn": True, "max_length": 48}
        config_params.update(param_case)

        self.run_predictor(config_params)

        result = self._read_result(os.path.join(self.output_dir, "predict.json"))
        assert len(self.golden_result) == len(result)

        partial_match, full_match = 0, 0
        for golden_item, result_item in zip(self.golden_result, result):
            score = levenshtein_similarity(golden_item, result_item)
            if score >= 0.95:
                full_match += 1
            if score >= 0.6:
                partial_match += 1

        if not config_params["inference_model"]:
            self.assertGreaterEqual(full_match / len(self.golden_result), 0.3)
            self.assertGreaterEqual(partial_match / len(self.golden_result), 0.4)
        elif config_params.get("use_fake_parameter", False):
            pass
        else:
            self.assertGreaterEqual(full_match / len(self.golden_result), 0.5)
            self.assertGreaterEqual(partial_match / len(self.golden_result), 0.8)


@parameterized_class(
    ["model_name_or_path", "model_class"],
    [
        ["__internal_testing__/Qwen2.5-72B-Instruct-tiny-nhl2", AutoModelForCausalLM],
        ["__internal_testing__/Llama-2-70b-chat-tiny-nhl2", AutoModelForCausalLM],
        ["__internal_testing__/Meta-Llama-3.1-70B-Instruct-tiny-nhl2", AutoModelForCausalLM],
    ],
)
class CommonGpusInferenceTest(TestMultipleGpus, LLMTest):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = None
    model_class = None

    def setUp(self):
        TestMultipleGpus.setUp(self)
        LLMTest.setUp(self)
        self.save_file_path = tempfile.mkdtemp()

    @require_gpu(2)
    def test_muti_gpus_inference(self):
        scripts = "tests/llm/testing_run_gpus_inference.py"
        config = {
            "tensor_parallel_degree": 2,
            "pipeline_parallel_degree": 1,
            "save_path": os.path.join(self.save_file_path, "predict.json"),
            "model_name_or_path": self.model_name_or_path,
        }
        self.run_2gpu(scripts, **config)

        result = self._read_result(os.path.join(self.save_file_path, "predict.json"))
        self.assertTrue(len(result) > 0, f"The inference result for {self.model_name_or_path} is empty!")

    def tearDown(self):
        LLMTest.tearDown(self)
        if os.path.exists(self.save_file_path):
            shutil.rmtree(self.save_file_path)
