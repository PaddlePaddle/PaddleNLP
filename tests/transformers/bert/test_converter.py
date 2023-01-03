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

import tempfile
import unittest

from paddlenlp.transformers.bert.converter import BertConverter, BertLogitComparer
from tests.testing_utils import require_package


class BertConverterTest(unittest.TestCase):
    @require_package("transformers")
    def test_token_classification(self):
        with tempfile.TemporaryDirectory() as tempdir:
            from transformers import BertConfig, BertForTokenClassification

            config: BertConfig = BertConfig.from_pretrained("bert-base-uncased")
            config.hidden_sizee = 32
            config.num_hidden_layers = 4
            config.num_attention_heads = 4

            pytorch_model = BertForTokenClassification(config)
            pytorch_model.save_pretrained(tempdir)

            # convert the weight file
            converter = BertConverter(input_dir=tempdir)
            converter.convert()

            # compare the state_dict
            comparer = BertLogitComparer(input_dir=tempdir)

            result = comparer.compare_logits(tempdir, tempdir)
            self.assertTrue(result)

    @require_package("transformers")
    def test_smoke_comparer(self):
        """only test whether it will work fine"""
        with tempfile.TemporaryDirectory() as tempdir:
            from transformers import BertConfig, BertModel

            config: BertConfig = BertConfig.from_pretrained("bert-base-uncased")
            config.hidden_sizee = 32
            config.num_hidden_layers = 4
            config.num_attention_heads = 4

            pytorch_model = BertModel(config)
            pytorch_model.save_pretrained(tempdir)

            # compare the state_dict
            comparer = BertLogitComparer(input_dir=tempdir)
            comparer.convert()

            comparer.compare_logits(tempdir, tempdir)

    @require_package("transformers")
    def test_smoke(self):
        with tempfile.TemporaryDirectory() as tempdir:
            from transformers import BertConfig, BertModel

            config: BertConfig = BertConfig.from_pretrained("bert-base-uncased")
            config.hidden_sizee = 32
            config.num_hidden_layers = 4
            config.num_attention_heads = 4

            pytorch_model = BertModel(config)
            pytorch_model.save_pretrained(tempdir)

            # convert the weight file
            converter = BertConverter(input_dir=tempdir)
            converter.convert()

            # compare the state_dict
            comparer = BertLogitComparer(input_dir=tempdir)

            result = comparer.compare_logits(tempdir, tempdir)
            self.assertTrue(result)
