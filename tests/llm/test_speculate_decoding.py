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
from __future__ import annotations

import unittest

import paddle

from .testing_utils import LLMTest, argv_context_guard


class SpeculatePredictorTest(LLMTest, unittest.TestCase):
    model_name_or_path: str = "__internal_testing__/tiny-random-llama-hd128"

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("bfloat16")
        self.config_params = {
            "model_name_or_path": self.model_name_or_path,
            "mode": "dynamic",
            "dtype": "bfloat16",
            "max_length": 48,
            "inference_model": 1,
            "speculate_method": None,
        }

    def run_speculate_predictor(self, speculate_params):
        """
        base speculative decoding forward test.
        """
        predict_config = self.config_params
        predict_config.update(speculate_params)

        # dynamic forward
        self.disable_static()
        with argv_context_guard(predict_config):
            from predict.predictor import predict

            predict()

        # to static
        self.disable_static()
        predict_config["output_path"] = self.output_dir
        with argv_context_guard(predict_config):
            from predict.export_model import main

            main()

        # static forward
        self.disable_static()

        predict_config["mode"] = "static"
        predict_config["model_name_or_path"] = self.output_dir

        predict_config.pop("output_path")
        with argv_context_guard(predict_config):
            from predict.predictor import predict

            predict()

    def test_inference_with_reference(self):
        """
        test inference with reference method.
        """
        speculate_params = {
            "speculate_method": "inference_with_reference",
            "speculate_max_draft_token_num": 5,
            "speculate_max_ngram_size": 2,
        }
        self.run_speculate_predictor(speculate_params)


if __name__ == "__main__":
    unittest.main()
