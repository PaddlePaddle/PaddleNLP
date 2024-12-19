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

import os
import unittest
from tempfile import TemporaryDirectory

from paddlenlp.transformers import AutoModel
from tests.testing_utils import argv_context_guard

from .testing_utils import LLMTest


class MergeTest(LLMTest, unittest.TestCase):
    def setUp(self) -> None:
        LLMTest.setUp(self)

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_merge(self):
        self.disable_static()
        with TemporaryDirectory() as tempdir:
            model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert", dtype="float32")
            pd_path = os.path.join(tempdir, "pd_model")
            model.save_pretrained(pd_path)
            safe_path = os.path.join(tempdir, "safe_model")
            model.save_pretrained(safe_path, safe_serialization="safetensors")

            merge_config = {"model_path_str": ",".join([pd_path, safe_path]), "output_path": tempdir}
            with argv_context_guard(merge_config):
                from tools.merge_weight import merge

                merge()
