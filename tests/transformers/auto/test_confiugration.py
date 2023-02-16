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


class AutoConfigTest(unittest.TestCase):

    def test_built_in_model_class_config(self):
        config = AutoConfig.from_pretrained("bert-base-uncased")
        number = random.randint(0, 10000)
        assert config.hidden_size == 768
        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # there is no architectures in config.json
            with open(os.path.join(tempdir, AutoConfig.config_file), "r", encoding="utf-8") as f:
                config_data = json.load(f)

            assert "architectures" not in config_data

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            assert auto_config.hidden_size == number

    def test_community_model_class(self):
        # OPT model do not support PretrainedConfig, but can load it as the AutoConfig object
        config = AutoConfig.from_pretrained("facebook/opt-125m")

        assert config.hidden_size == 768

        number = random.randint(0, 10000)
        config.hidden_size = number

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)

            # but it can load it as the PretrainedConfig class
            auto_config = AutoConfig.from_pretrained(tempdir)
            assert auto_config.hidden_size == number
