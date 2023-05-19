# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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
import random
import tempfile
import unittest

from paddlenlp.transformers import AutoConfig
from paddlenlp.utils.env import CONFIG_NAME


class AutoConfigTest(unittest.TestCase):
    def test_built_in_model_class_config(self):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        number = random.randint(0, 10000)
        self.assertEqual(config.hidden_size, 768)

        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # there is no architectures in config.json
            with open(os.path.join(tempdir, AutoConfig.config_file), "r", encoding="utf-8") as f:
                config_data = json.load(f)

            self.assertNotIn("architectures", config_data)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    def test_community_model_class(self):
        # OPT model do not support PretrainedConfig, but can load it as the AutoConfig object
        config = AutoConfig.from_pretrained("facebook/opt-125m")

        self.assertEqual(config.hidden_size, 768)

        number = random.randint(0, 10000)
        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    def test_from_hf_hub(self):
        config = AutoConfig.from_pretrained("facebook/opt-66b", from_hf_hub=True)
        self.assertEqual(config.hidden_size, 9216)

    def test_load_from_legacy_config(self):
        number = random.randint(0, 10000)
        legacy_config = {"init_class": "BertModel", "hidden_size": number}
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, AutoConfig.legacy_config_file), "w", encoding="utf-8") as f:
                json.dump(legacy_config, f, ensure_ascii=False)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            self.assertEqual(auto_config.hidden_size, number)

    def test_from_pretrained_cache_dir(self):
        model_id = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoConfig.from_pretrained(model_id, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_id, CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_id, model_id)))
