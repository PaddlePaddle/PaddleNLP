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

import numpy as np
import paddle
from parameterized import parameterized

from paddlenlp.prompt import AutoTemplate, ManualTemplate, PrefixTemplate, SoftTemplate
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer


class TemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.model = AutoModelForMaskedLM.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.example = {"text_a": "天气晴朗", "text_b": "下雪了", "choices": ["不", "很"], "labels": 0}
        cls.max_length = 20
        cls.tokenizer.add_special_tokens({"additional_special_tokens": ["[O-MASK]"]})
        cls.model.resize_token_embeddings(len(cls.tokenizer))

    def test_initialization(self):
        with self.assertRaises(ValueError):
            prompt = "非AutoTemplate模板中必须定义`text`"
            ManualTemplate(prompt, self.tokenizer, self.max_length)

        with self.assertRaises(ValueError):
            prompt = "{'text': 'text_a'} SoftTemplate必须定义`soft`"
            SoftTemplate(prompt, self.tokenizer, self.max_length, self.model.get_input_embeddings())

        with self.assertRaises(ValueError):
            prompt = "{'text': 'text_a'} PrefixTemplate只定义`prefix`。{'soft'}"
            PrefixTemplate(prompt, self.tokenizer, self.max_length, self.model)

        with self.assertRaises(ValueError):
            prompt = "{'text': 'text_a'}{'prefix': None, 'length':3}PrefixTemplate中`prefix`只能位于句首。"
            PrefixTemplate(prompt, self.tokenizer, self.max_length, self.model)

    @parameterized.expand(
        [
            (
                "{'text': 'text_a'}说明天气{'mask'}好。",
                "[CLS]天气晴朗说明天气[MASK]好。[SEP]",
            ),
            (
                "{'text': 'text_a'}{'hard': '说明天气'}{'mask'}{'hard': '好。'}",
                "[CLS]天气晴朗说明天气[MASK]好。[SEP]",
            ),
            (
                "下边两句话意思一样吗？{'text':'text_a'}{'sep'}{'text':'text_b'}",
                "[CLS]下边两句话意思一样吗？天气晴[SEP]下雪了[SEP]",
            ),
            (
                "{'options':'choices','add_omask':True,'add_prompt':'[OPT]好。'}天气如何？{'text': 'text_a'}",
                "[CLS][O-MASK]不好。[O-MASK]很好。天气如何？天气晴朗[SEP]",
            ),
            (
                "{'text': 'text_a'}{'text': 'text_b', 'truncate': False}这是一个很长的提示，要截断",
                "[CLS]天气下雪了这是一个很长的提示，要截断[SEP]",
            ),
            ("{'text':'text_a'}说明天气{'mask': None, 'length': 3}", "[CLS]天气晴朗说明天气[MASK][MASK][MASK][SEP]"),
        ]
    )
    def test_manual_template(self, prompt, expected_ans):
        template = ManualTemplate(prompt, self.tokenizer, self.max_length)
        encoded_ids = template(self.example)["input_ids"]
        encoded_tokens = self.tokenizer.decode(encoded_ids).replace(" ", "")
        self.assertEqual(encoded_tokens, expected_ans)

    @parameterized.expand(
        [
            (
                "{'text':'text_a', 'position': 4}",
                [0, 4, 5, 6, 7, 0],
            ),
            (
                "{'options':'choices','add_omask':True,'add_prompt':'[OPT]好。'}天气如何？{'text': 'text_a'}",
                [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0],
            ),
        ]
    )
    def test_position(self, prompt, expected_ans):
        template = ManualTemplate(prompt, self.tokenizer, self.max_length)
        encoded_positions = template(self.example)["position_ids"]
        self.assertEqual(encoded_positions, expected_ans)

    def test_token_type(self):
        prompt = "{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}之间的关系是什么？"
        template = ManualTemplate(prompt, self.tokenizer, self.max_length)
        encoded_token_types = template(self.example)["token_type_ids"]
        self.assertEqual(encoded_token_types, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

    def test_attention_mask(self):
        expected_att = np.zeros([15, 15])
        expected_att[1:5, 5:9] = -1e4
        expected_att[5:9, 1:5] = -1e4
        prompt = "{'options':'choices','add_omask':True,'add_prompt':'[OPT]好。'}{'sep'}{'text': 'text_a'}"
        template = ManualTemplate(prompt, self.tokenizer, self.max_length)
        encoded_att = template(self.example)["attention_mask"]
        self.assertListEqual(encoded_att.tolist(), expected_att.tolist())

    @parameterized.expand(
        [
            (
                "{'soft'}{'text': 'text_a'}{'sep'}{'text': 'text_b'}",
                "[CLS][CLS]天气晴朗[SEP]下雪了[SEP]",
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                "{'text': 'text_a'}{'soft': '请判断', 'length': 5}{'mask'}",
                "[CLS]天气晴朗[CLS][SEP][MASK]，的[MASK][SEP]",
                [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0],
            ),
        ]
    )
    def test_soft_template(self, prompt, expected_token_ids, expected_soft_ids):
        template = SoftTemplate(prompt, self.tokenizer, self.max_length, self.model.get_input_embeddings())
        encoded = template(self.example)
        self.assertEqual(self.tokenizer.decode(encoded["input_ids"]).replace(" ", ""), expected_token_ids)
        self.assertEqual(encoded["soft_token_ids"], expected_soft_ids)

    def test_soft_initialization(self):
        prompt = "{'text': 'text_a'}{'soft': '分类'}"
        template = SoftTemplate(prompt, self.tokenizer, self.max_length, self.model.get_input_embeddings())
        expected_tokens = self.tokenizer("分类", add_special_tokens=False)["input_ids"]
        expected_embeds = self.model.get_input_embeddings()(paddle.to_tensor(expected_tokens))
        init_embeds = template.soft_embeddings.weight[1:]
        self.assertTrue(paddle.allclose(init_embeds, expected_embeds, atol=1e-6))

    def test_soft_encoder(self):
        prompt = "{'text': 'text_a'}{'soft': '天气', 'encoder': 'lstm', 'hidden_size': 32}{'soft': None, 'length': 2, 'encoder': 'mlp'}"
        template = SoftTemplate(prompt, self.tokenizer, self.max_length, self.model.get_input_embeddings())
        encoded = template(self.example)
        encoded_tokens = self.tokenizer.decode(encoded["input_ids"]).replace(" ", "")
        encoded_encoder = encoded["encoder_ids"]
        expected_tokens = "[CLS]天气晴朗[CLS][SEP][MASK]，[SEP]"
        expected_encoder = [0, 0, 0, 0, 0, 1, 1, 2, 2, 0]
        self.assertEqual(encoded_tokens, expected_tokens)
        self.assertEqual(encoded_encoder, expected_encoder)

    @parameterized.expand(
        [
            (
                "{'prefix': '新闻类别', 'length': 10, 'encoder': 'lstm'}{'text': 'text_a'}",
                "[CLS][CLS][SEP][MASK]，的、一人有是天气晴朗[SEP]",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0],
            )
        ]
    )
    def test_prefix_template(self, prompt, expected_tokens, expected_soft_tokens):
        template = PrefixTemplate(prompt, self.tokenizer, self.max_length, self.model)
        encoded = template(self.example)
        encoded_tokens = self.tokenizer.decode(encoded["input_ids"]).replace(" ", "")
        self.assertEqual(encoded_tokens, expected_tokens)
        self.assertEqual(encoded["soft_token_ids"], expected_soft_tokens)

    @parameterized.expand(
        [
            (
                "{'text': 'text_a'}说明天气{'mask'}好。",
                "[CLS]天气晴朗说明天气[MASK]好。[SEP]",
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                "{'text': 'text_a'}{'hard': '说明天气'}{'mask'}{'hard': '好。'}",
                "[CLS]天气晴朗说明天气[MASK]好。[SEP]",
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                "{'text': 'text_a'}{'soft': '请判断', 'length': 5}{'mask'}",
                "[CLS]天气晴朗[CLS][SEP][MASK]，的[MASK][SEP]",
                [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0],
            ),
            (
                "{'prefix': '新闻类别', 'length': 10, 'encoder': 'lstm'}{'text': 'text_a'}",
                "[CLS][CLS][SEP][MASK]，的、一人有是天气晴朗[SEP]",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0],
            ),
        ]
    )
    def test_auto_template(self, prompt, expected_tokens, expected_soft_tokens):
        template = AutoTemplate.create_from(prompt, self.tokenizer, self.max_length, self.model)
        encoded = template(self.example)
        encoded_tokens = self.tokenizer.decode(encoded["input_ids"]).replace(" ", "")
        self.assertEqual(encoded_tokens, expected_tokens)
        self.assertEqual(encoded["soft_token_ids"], expected_soft_tokens)


if __name__ == "__main__":
    unittest.main()
