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

import os
import tempfile
import unittest

import paddlenlp
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.env import TOKENIZER_CONFIG_NAME


class AutoTokenizerTest(unittest.TestCase):
    @unittest.skip("skipping due to connection error!")
    def test_from_aistudio(self):
        tokenizer = AutoTokenizer.from_pretrained("PaddleNLP/tiny-random-bert", from_aistudio=True)
        self.assertIsInstance(tokenizer, paddlenlp.transformers.BertTokenizer)

    def test_from_pretrained_cache_dir(self):
        model_name = "__internal_testing__/tiny-random-bert"
        with tempfile.TemporaryDirectory() as tempdir:
            AutoTokenizer.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, TOKENIZER_CONFIG_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))
