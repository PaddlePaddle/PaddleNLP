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

import os
import unittest

from pipelines.nodes.file_converter import MarkdownConverter
from pipelines.nodes.file_converter.markdown import MarkdownRawTextConverter


class TestMarkdownConverter(unittest.TestCase):
    def test_conversion(self):
        fixtures_path = "tests/fixtures"
        file_path = os.path.join(fixtures_path, "example_markdown.md")
        converter = MarkdownConverter()
        expected_result = [
            {
                "content": "Heading level 1\nHeading level 2\nI really like using Markdown.\n\nFirst item\nSecond item\nThird item\nFourth item\n",
                "content_type": "text",
                "meta": None,
            }
        ]
        result = converter.convert(file_path)
        self.assertEqual(expected_result, result)


class TestMarkdownRawTextConverter(unittest.TestCase):
    def test_conversion(self):
        fixtures_path = "tests/fixtures"
        file_path = os.path.join(fixtures_path, "example_markdown.md")
        converter = MarkdownRawTextConverter()
        expected_result = [
            {
                "content": "# Heading level 1\n## Heading level 2\nI really like using Markdown.\n\nFirst item\nSecond item\nThird item\nFourth item\n",
                "content_type": "text",
                "meta": None,
            }
        ]
        result = converter.convert(file_path)
        self.assertEqual(expected_result, result)
