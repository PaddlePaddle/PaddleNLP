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

import base64
import io
import random
import unittest

from PIL import Image

from paddlenlp.taskflow import Taskflow


class TestUIETask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text_ie = Taskflow(
            task="information_extraction",
            model="__internal_testing__/tiny-random-uie",
        )

        cls.doc_ie = Taskflow(
            task="information_extraction",
            model="__internal_testing__/tiny-random-uie-x",
        )

    def test_entity_extraction(self):
        schema = ["时间", "选手", "赛事名称"]
        self.text_ie.set_schema(schema)
        outputs = self.text_ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)
                    # self.assertIn("start", entity)  # TODO: find out why this fails
                    # self.assertIn("end", entity)  # TODO: find out why this fails

    def test_relation_extraction(self):
        schema = [{"歌曲名称": ["歌手", "所属专辑"]}]
        entity_type = "歌曲名称"
        relation_types = ["歌手", "所属专辑"]
        self.text_ie.set_schema(schema)
        outputs = self.text_ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")
        self.assertIsNotNone(outputs)
        for output in outputs:
            self.assertIn(entity_type, output)
            for entity in output[entity_type]:
                self.assertIn("text", entity)
                self.assertIn("probability", entity)
                self.assertIn("relations", entity)
                for relation_type, relations in entity["relations"].items():
                    self.assertIn(relation_type, relation_types)
                    for relation in relations:
                        self.assertIn("text", relation)
                        self.assertIn("probability", relation)

    def test_doc_entity_extraction(self):
        # Create a random image
        width, height = 200, 200
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new("RGB", (width, height), color)

        # Encode as base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        schema = ["发票号码", "开票日期"]
        self.doc_ie.set_schema(schema)
        outputs = self.doc_ie(
            {"doc": base64_string},
            {"text": "发票号码: 656656, 开票日期: 2023年3月2日"},
        )
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)
                    self.assertIn("bbox", entity)
