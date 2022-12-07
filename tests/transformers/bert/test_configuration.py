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

import os.path
import tempfile
import unittest

from paddlenlp.transformers import BertConfig


class BertConfigTest(unittest.TestCase):
    def test_load_from_hf(self):
        """test load config from hf"""
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-BertModel", from_hf_hub=True)
        assert config.hidden_size == 32

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            assert os.path.exists(os.path.join(tempdir, "config.json"))

    def test_config_mapping(self):
        class FakeBertConfig(BertConfig):
            pass

        with tempfile.TemporaryDirectory() as tempdir:
            config = FakeBertConfig.from_pretrained("bert-base-uncased")

            config.save_pretrained(tempdir)

            FakeBertConfig.standard_config_map = {"hidden_size": "fake_field"}

            loaded_config = FakeBertConfig.from_pretrained(tempdir)
            assert loaded_config["fake_field"] == config.hidden_size
