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

from pipelines.nodes.file_converter import PDFToTextConverter


class TestPDFToTextConverter(unittest.TestCase):
    def test_conversion(self):
        fixtures_path = "tests/fixtures"
        file_path = os.path.join(fixtures_path, "example_pdf.pdf")
        converter = PDFToTextConverter(language="en")

        expected_result = [
            {
                "content": "A Simple PDF File\nThis is a small demonstration .pdf file -\njust for use in the Virtual Mechanics tutorials. More text. And more\ntext. And more text. And more text. And more text.\nAnd more text. And more text. And more text. And more text. And more\ntext. And more text. Boring, zzzzz. And more text. And more text. And\nmore text. And more text. And more text. And more text. And more text.\nAnd more text. And more text.\nAnd more text. And more text. And more text. And more text. And more\ntext. And more text. And more text. Even more. Continued on page 2 ...\x0cSimple PDF File 2\n...continued from page 1. Yet more text. And more text. And more text.\nAnd more text. And more text. And more text. And more text. And more\ntext. Oh, how boring typing this stuff. But not as boring as watching\npaint dry. And more text. And more text. And more text. And more text.\nBoring. More, a little more text. The end, and just as well.",
                "content_type": "text",
                "meta": None,
            }
        ]
        result = converter.convert(file_path)
        self.assertEqual(expected_result, result)
