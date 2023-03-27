# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from fast_tokenizer import pretokenizers


class TestByteLevelPreTokenizer(unittest.TestCase):
    def setUp(self):
        self.pretokenized = pretokenizers.PreTokenizedString("Hello my friend, how is your day going?")

    def check_equals(self, add_prefix_space, use_regex, expected_result):
        bytelevel = pretokenizers.ByteLevelPreTokenizer(add_prefix_space=add_prefix_space, use_regex=use_regex)
        bytelevel(self.pretokenized)
        splits = self.pretokenized.get_splits()
        result = [(s, offset) for s, offset, tokens in splits]
        self.assertEqual(result, expected_result)

    def test_pretokenize_with_regex(self):
        expected_result = [
            ("Hello", (0, 5)),
            ("Ġmy", (5, 8)),
            ("Ġfriend", (8, 15)),
            (",", (15, 16)),
            ("Ġhow", (16, 20)),
            ("Ġis", (20, 23)),
            ("Ġyour", (23, 28)),
            ("Ġday", (28, 32)),
            ("Ġgoing", (32, 38)),
            ("?", (38, 39)),
        ]

        self.check_equals(False, True, expected_result)

    def test_pretokenize_without_regex(self):
        expected_result = [("HelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))]
        self.check_equals(False, False, expected_result)

    def test_pretokenize_with_prefix_with_regex(self):
        expected_result = [
            ("ĠHello", (0, 5)),
            ("Ġmy", (5, 8)),
            ("Ġfriend", (8, 15)),
            (",", (15, 16)),
            ("Ġhow", (16, 20)),
            ("Ġis", (20, 23)),
            ("Ġyour", (23, 28)),
            ("Ġday", (28, 32)),
            ("Ġgoing", (32, 38)),
            ("?", (38, 39)),
        ]

        self.check_equals(True, True, expected_result)

    def test_pretokenize_with_prefix_without_regex(self):
        expected_result = [("ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))]
        self.check_equals(True, False, expected_result)
