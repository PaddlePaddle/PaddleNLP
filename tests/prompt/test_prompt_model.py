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

from paddlenlp.prompt import (
    AutoTemplate,
    PromptDataCollatorWithPadding,
    PromptModelForGeneration,
    PromptModelForSequenceClassification,
    SoftVerbalizer,
)
from paddlenlp.transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTLMHeadModel,
)


class PromptModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.model = AutoModelForMaskedLM.from_pretrained("__internal_testing__/tiny-random-ernie")
        cls.num_labels = 2
        cls.seq_cls_model = AutoModelForSequenceClassification.from_pretrained(
            "__internal_testing__/tiny-random-ernie", num_labels=cls.num_labels
        )

        cls.template = AutoTemplate.create_from(
            prompt="{'soft'}{'text': 'text'}{'mask'}", tokenizer=cls.tokenizer, max_length=512, model=cls.model
        )
        cls.label_words = {0: "0", 1: "1", 2: "2"}
        cls.verbalizer = SoftVerbalizer(cls.label_words, cls.tokenizer, cls.model)
        cls.data_collator = PromptDataCollatorWithPadding(cls.tokenizer, padding=True, return_tensors="pd")
        cls.prompt_model = PromptModelForSequenceClassification(cls.model, cls.template, cls.verbalizer)

    def test_sequence_classification_no_labels(self):
        examples = [{"text": "百度飞桨深度学习框架"}, {"text": "这是一个测试"}]
        encoded_examples = [self.template(i) for i in examples]
        logits, hidden_states = self.prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], len(self.label_words))
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = self.prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], len(self.label_words))
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_sequence_classification_with_labels(self):
        examples = [{"text": "百度飞桨深度学习框架", "labels": 0}, {"text": "这是一个测试", "labels": 1}]
        encoded_examples = [self.template(i) for i in examples]
        loss, logits, hidden_states = self.prompt_model(
            **self.data_collator(encoded_examples), return_hidden_states=True
        )
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], len(self.label_words))
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = self.prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNotNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], len(self.label_words))
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_efl_no_labels(self):
        prompt_model = PromptModelForSequenceClassification(self.seq_cls_model, self.template, verbalizer=None)
        examples = [{"text": "百度飞桨深度学习框架"}, {"text": "这是一个测试"}]
        encoded_examples = [self.template(i) for i in examples]
        logits, hidden_states = prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], self.num_labels)
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], self.num_labels)
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))

    def test_efl_with_labels(self):
        prompt_model = PromptModelForSequenceClassification(self.seq_cls_model, self.template, verbalizer=None)
        examples = [{"text": "百度飞桨深度学习框架", "labels": 0}, {"text": "这是一个测试", "labels": 1}]
        encoded_examples = [self.template(i) for i in examples]
        loss, logits, hidden_states = prompt_model(**self.data_collator(encoded_examples), return_hidden_states=True)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(logits.shape[1], self.num_labels)
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNotNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.logits.shape[1], self.num_labels)
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))


class PromptModelTestForGeneration(unittest.TestCase):
    def test_generation_with_labels(self):
        self.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-gpt")
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.sep_token = "<sep>"
        self.tokenizer.add_tokens("[Space]", special_tokens=True)
        self.model = GPTLMHeadModel.from_pretrained("__internal_testing__/tiny-random-gpt")

        self.template = AutoTemplate.create_from(
            prompt="{'prefix':'文本摘要', 'encoder': 'mlp'}{'text':'text'}{'sep'}{'text':'labels', 'token_type': 1}",
            tokenizer=self.tokenizer,
            max_length=512,
            model=self.model,
        )

        self.data_collator = PromptDataCollatorWithPadding(self.tokenizer, padding=True, return_tensors="pd")
        self.prompt_model = PromptModelForGeneration(self.model, self.template)
        examples = [
            {
                "text": "日前，方舟子发文直指林志颖旗下爱碧丽推销假保健品，引起哗然。调查发现，爱碧丽没有自己的生产加工厂。其胶原蛋白饮品无核心研发，全部代工生产。号称有“逆生长”功效的爱碧丽“梦幻奇迹限量组”售价高达1080元，实际成本仅为每瓶4元！",
                "labels": "林志颖公司疑涉虚假营销无厂房无研发",
                "id": 0,
            },
            {
                "text": "韩方应对路径可以概括为：企业道歉担责；政府公正不护短；民间祈福关怀。他们深知形象的重要，竭力呵护企业品牌和国家形象。正如有评论，韩国“政府+企业+民众”三位一体式呵护韩国国家形象的“苦心经营”，的确有值得我们借鉴之处。",
                "labels": "从韩亚航空事故看其应对路径",
                "id": 1,
            },
        ]
        encoded_examples = [self.template(i) for i in examples]
        loss, logits, hidden_states = self.prompt_model(
            **self.data_collator(encoded_examples), return_hidden_states=True
        )
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape[0], len(examples))
        self.assertEqual(hidden_states.shape[0], len(examples))

        model_outputs = self.prompt_model(
            **self.data_collator(encoded_examples), return_dict=True, return_hidden_states=True
        )
        self.assertIsNotNone(model_outputs.loss)
        self.assertEqual(model_outputs.logits.shape[0], len(examples))
        self.assertEqual(model_outputs.hidden_states.shape[0], len(examples))


if __name__ == "__main__":
    unittest.main()
