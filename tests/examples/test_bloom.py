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
import sys
import tempfile
from unittest import TestCase

import pytest

from tests.testing_utils import argv_context_guard, load_test_config
from tests.transformers.test_modeling_common import DistributedTest


class BloomCPUTest(TestCase):
    def setUp(self) -> None:
        self.path = "./examples/language_model/bloom"
        self.config_path = "./tests/fixtures/examples/bloom.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_predict_generation(self):
        config = load_test_config(self.config_path, "predict_generation")
        with argv_context_guard(config):
            from predict_generation import predict

            predict()

    def test_export_and_infer_generation(self):
        config = load_test_config(self.config_path, "export_generation")
        # 1. do export generation
        with tempfile.TemporaryDirectory() as tempdir:
            config["output_path"] = os.path.join(tempdir, "bloom")
            with argv_context_guard(config):
                from export_generation import main

                main()
                self.assertTrue(os.path.exists(os.path.join(tempdir, "bloom.pdmodel")))

    def test_export_glue(self):
        config = load_test_config(self.config_path, "export_glue")
        with tempfile.TemporaryDirectory() as tempdir:
            config["output_path"] = os.path.join(tempdir, "bloom")
            with argv_context_guard(config):
                from export_glue import main

                main()
                self.assertTrue(os.path.exists(os.path.join(tempdir, "bloom.pdmodel")))


class BloomGenerationDistributedTest(DistributedTest):
    def setUp(self) -> None:
        super().setUp()

        self.path = "./examples/language_model/bloom"
        self.config_path = "./tests/fixtures/examples/bloom.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    @pytest.mark.skip("skip for test")
    def test_pipeline(self):
        # 1. test for fine-tune scripts
        with tempfile.TemporaryDirectory() as tempdir:
            config = load_test_config(self.config_path, "finetune_generation")
            config["output_dir"] = os.path.join(tempdir, "bloom")
            config["mp_degree"] = self.get_world_size()
            with argv_context_guard(config):
                self.run_on_gpu(
                    training_script=os.path.join(self.path, "finetune_generation.py"), training_script_args=config
                )
