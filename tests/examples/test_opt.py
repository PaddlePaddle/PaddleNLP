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

from tests.testing_utils import argv_context_guard, load_test_config
from tests.transformers.test_modeling_common import slow


class OPTTest(TestCase):
    def setUp(self) -> None:
        self.path = "./examples/language_model/opt"
        self.config_path = "./tests/fixtures/examples/opt.yaml"
        sys.path.insert(0, self.path)

    def tearDown(self) -> None:
        sys.path.remove(self.path)

    def test_predict_generation(self):
        config = load_test_config(self.config_path, "predict_generation")
        with argv_context_guard(config):
            from predict_generation import predict

            predict()

    @slow
    def test_pipelines(self):
        finetune_config = load_test_config(self.config_path, "finetune_generation")
        with tempfile.TemporaryDirectory() as tmp_dir:
            finetune_config["output_dir"] = os.path.join(tmp_dir, "exports")
            # 1. do finetune
            with argv_context_guard(finetune_config):
                from finetune_generation import main

                main()

            # 2. do export
            export_config = {"model_name_or_path": finetune_config["output_dir"]}
            with tempfile.TemporaryDirectory() as finetune_dir:
                export_config["output_path"] = os.path.join(finetune_dir, "opt")
                with argv_context_guard(export_config):
                    from export_generation import main

                    main()

                    self.assertTrue(os.path.exists(export_config["output_path"] + ".pdmodel"))

                # 3. do inference
                infer_config = {"model_dir": finetune_dir, "model_prefix": "opt"}
                with argv_context_guard(infer_config):
                    from infer_generation import main

                    main()
