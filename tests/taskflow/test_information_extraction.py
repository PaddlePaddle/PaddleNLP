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

import pytest

from paddlenlp import Taskflow

from ..testing_utils import get_tests_dir


class TestUIETask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uie = Taskflow(
            task="information_extraction",
            model="__internal_testing__/tiny-random-uie",
        )

        cls.uie_m = Taskflow(
            task="information_extraction",
            task_path="PaddleCI/tiny-random-uie-m",
            from_hf_hub=True,
            convert_from_torch=False,
        )

        cls.uie_x = Taskflow(
            task="information_extraction",
            task_path="PaddleCI/tiny-random-uie-x",
            from_hf_hub=True,
            convert_from_torch=False,
        )

    def test_entity_extraction(self):
        schema = ["时间", "选手", "赛事名称"]
        self.uie.set_schema(schema)
        outputs = self.uie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)

        self.uie_m.set_schema(schema)
        outputs = self.uie_m("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)

    def test_relation_extraction(self):
        schema = [{"歌曲名称": ["歌手", "所属专辑"]}]
        entity_type = "歌曲名称"
        relation_types = ["歌手", "所属专辑"]
        self.uie.set_schema(schema)
        outputs = self.uie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")
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

    def test_opinion_extraction(self):
        schema = [{"评价维度": ["观点词", "情感倾向[正向，负向]"]}]
        entity_type = "评价维度"
        relation_types = ["观点词", "情感倾向[正向，负向]"]
        self.uie.set_schema(schema)
        outputs = self.uie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队")
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

    @pytest.mark.skip(reason="todo, fix it")
    def test_doc_entity_extraction(self):
        doc_path = get_tests_dir("fixtures/tests_samples/OCR/custom.jpeg")

        schema = ["进口日期", "申报日期"]
        self.uie_x.set_schema(schema)
        outputs = self.uie_x(
            {"doc": doc_path},
            {"text": "进口日期: 2023年3月2日, 申报日期: 2023年3月2日"},
        )
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)
                    self.assertIn("bbox", entity)

        # Enable layout analysis
        self.uie_x.set_argument({"layout_analysis": True})
        outputs = self.uie_x(
            {"doc": doc_path},
            {"text": "进口日期: 2023年3月2日, 申报日期: 2023年3月2日"},
        )
        self.assertIsNotNone(outputs)
        for output in outputs:
            for field in output:
                self.assertIn(field, schema)
                for entity in output[field]:
                    self.assertIn("text", entity)
                    self.assertIn("probability", entity)
                    # fixme @ZHUI
                    # self.assertIn("bbox", entity)
