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

import unittest

from paddlenlp.transformers import (
    AlbertForTokenClassification,
    AlbertModel,
    BertModel,
    utils,
)
from paddlenlp.transformers.bert.modeling import BertForTokenClassification


class TestUtils(unittest.TestCase):
    """Unittest for paddlenlp.transformers.utils.py module"""

    def test_find_transformer_model_type(self):
        """test for `find_transformer_model_type`"""
        self.assertEqual(utils.find_transformer_model_type(AlbertModel), "albert")
        self.assertEqual(utils.find_transformer_model_type(AlbertForTokenClassification), "albert")
        self.assertEqual(utils.find_transformer_model_type(BertModel), "bert")
        self.assertEqual(utils.find_transformer_model_type(BertForTokenClassification), "bert")


def check_json_file_has_correct_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines) == 1:
            # length can only be 1 if dict is empty
            assert lines[0] == "{}"
        else:
            # otherwise make sure json has correct format (at least 3 lines)
            assert len(lines) >= 3
            # each key one line, ident should be 2, min length is 3
            assert lines[0].strip() == "{"
            for line in lines[1:-1]:
                left_indent = len(lines[1]) - len(lines[1].lstrip())
                assert left_indent == 2
            assert lines[-1].strip() == "}"
